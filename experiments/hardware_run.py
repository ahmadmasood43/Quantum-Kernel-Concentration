"""IBM hardware kernel experiment (T9) and matched noise-model comparison (T10).

Steps (each writes into results/hardware/<tag>/):

  prepare  Select points, build compute-uncompute circuits for every pair and
           diagonal, transpile against the backend's live target, and save the
           ISA circuits (QPY), their depth / 2Q counts, the design, a
           calibration snapshot, and an estimate of QPU time. No QPU time used.
  preflight  Run the exact saved ISA circuits locally (free) and refuse to
           pass unless: the circuit file is complete and consistent; an ideal
           simulation reproduces the exact kernel values within shot noise and
           returns 1 on every diagonal; the calibrated noise model degrades the
           diagonals beyond readout error alone (gates were not compiled away);
           the analysis runs on the simulated counts; and the projected QPU time
           (scaled by the cost measured on the pilot) fits the remaining
           allocation with a reserve.
  submit   Requires a passed preflight on byte-identical circuits, the same
           shot count, and --confirm; re-checks the live allocation; then
           submits one SamplerV2 job with every circuit. USES QPU TIME.
  fetch    Wait for the job, save raw counts and job metrics, snapshot the
           calibration at the job's run time, and sample the same ISA circuits
           with the same shot count under the noise model built from that
           snapshot (scale 1) and under an ideal device (scale 0).
  analyze  Recompute all statistics offline from the saved counts.

--fake runs the whole path against FakeMarrakesh with Aer standing in for the
device. Its outputs go to results/hardware/dryrun_<tag>/ and are labelled
FAKE; they are pipeline checks only and never hardware results.

    .venv/bin/python experiments/hardware_run.py prepare --backend ibm_marrakesh --tag hw1
    .venv/bin/python experiments/hardware_run.py preflight --tag hw1 --shots 1000
    .venv/bin/python experiments/hardware_run.py submit  --tag hw1 --shots 1000 --confirm
    .venv/bin/python experiments/hardware_run.py fetch   --tag hw1
    .venv/bin/python experiments/hardware_run.py analyze --tag hw1
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from qkc.config import RESULTS, ROOT
from qkc.hwexp import bootstrap_excess_var, isa_stats, kernel_circuit, p_zero, pair_list, select_points, spread_stats
from qkc.kernels import quantum_kernel
from qkc.noise import build_noise_model, calibration_summary, save_snapshot

OPT_LEVEL = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag_dir(tag: str, fake: bool) -> Path:
    return RESULTS / "hardware" / (f"dryrun_{tag}" if fake else tag)


def get_backend(name: str | None, fake: bool):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

        return FakeMarrakesh(), None
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(name="qkc", instance="open-instance")
    return service.backend(name), service


def cmd_prepare(args):
    out = tag_dir(args.tag, args.fake)
    if (out / "design.json").exists():
        raise SystemExit(f"{out} already prepared; use a new --tag")
    out.mkdir(parents=True)
    backend, _ = get_backend(args.backend, args.fake)
    pm = generate_preset_pass_manager(optimization_level=OPT_LEVEL, backend=backend, seed_transpiler=args.seed)
    pairs = pair_list(args.m, include_diagonal=True)
    design = {"backend": backend.name, "fake": args.fake, "prepared_utc": now(), "optimization_level": OPT_LEVEL,
              "seed_transpiler": args.seed, "m": args.m, "seed": args.seed, "reps": 2, "pairs": pairs, "blocks": []}
    circuits, stats = [], []
    for n in args.qubits:
        X, y = select_points(n, args.m, args.seed)
        for lam in args.lambdas:
            K = quantum_kernel(X, None, lam)
            isas = pm.run([kernel_circuit(X[i], X[j], lam) for i, j in pairs])
            first = len(circuits)
            for c, ((i, j), isa) in enumerate(zip(pairs, isas)):
                s = isa_stats(isa)
                s.update({"n": n, "lam": lam, "i": i, "j": j, "k_exact": float(K[i, j]),
                          "duration_s": float(isa.estimate_duration(backend.target))})
                stats.append(s)
                circuits.append(isa)
            design["blocks"].append({"n": n, "lam": lam, "first": first, "count": len(pairs),
                                     "X": X.tolist(), "y": y.tolist()})
    with open(out / "circuits.qpy", "wb") as f:
        qpy.dump(circuits, f)
    pd.DataFrame(stats).to_csv(out / "isa_stats.csv", index=False)
    (out / "design.json").write_text(json.dumps(design, indent=1))
    save_snapshot(calibration_summary(backend), out / "calibration_prepare.json")

    rep_delay = getattr(backend, "default_rep_delay", None) or 250e-6
    dur = np.array([s["duration_s"] for s in stats])
    for shots in (1000, 2000, 4000):
        est = shots * float(np.sum(dur + rep_delay))
        print(f"  {len(circuits)} circuits x {shots} shots: rough QPU-time estimate {est:.0f} s "
              f"(shots x (circuit duration + rep delay {rep_delay * 1e6:.0f} us); overheads not included)")
    df = pd.DataFrame(stats)
    off = df[df.i != df.j]
    print(off.groupby(["n", "lam"])[["depth", "n_2q"]].agg(["median", "max"]).to_string())
    print(f"Prepared {out}")


def _load(out: Path):
    design = json.loads((out / "design.json").read_text())
    with open(out / "circuits.qpy", "rb") as f:
        circuits = qpy.load(f)
    return design, circuits


REP_DELAY_DEFAULT = 250e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _naive_estimate(stats: pd.DataFrame, shots: int, rep_delay: float = REP_DELAY_DEFAULT) -> float:
    return shots * float((stats.duration_s + rep_delay).sum())


def _cost_factor() -> tuple[float, str]:
    """Measured QPU charge / naive estimate, from completed real runs (pilot first)."""
    best = None
    for d in sorted((RESULTS / "hardware").iterdir()):
        mfile, sfile, jfile = d / "job_metrics.json", d / "isa_stats.csv", d / "job.json"
        if d.name.startswith("dryrun_") or not (mfile.exists() and sfile.exists() and jfile.exists()):
            continue
        charged = (json.loads(mfile.read_text()).get("usage") or {}).get("qpu_charge_time_seconds")
        shots = json.loads(jfile.read_text()).get("shots")
        if charged and shots:
            f = charged / _naive_estimate(pd.read_csv(sfile), shots)
            best = (f, d.name) if best is None or f > best[0] else best  # most pessimistic
    return best if best else (2.0, "no completed run; default 2.0")


def cmd_preflight(args):
    out = tag_dir(args.tag, args.fake)
    design, circuits = _load(out)
    stats = pd.read_csv(out / "isa_stats.csv")
    cal = json.loads((out / "calibration_prepare.json").read_text())
    pairs = [tuple(p) for p in design["pairs"]]
    checks, ok = {}, True

    def check(name, cond, detail):
        nonlocal ok
        checks[name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    expected = sum(b["count"] for b in design["blocks"])
    check("circuit_count", len(circuits) == expected == len(stats), f"{len(circuits)} circuits, design {expected}, stats {len(stats)}")
    widths_ok = all(circuits[b["first"] + c].num_clbits == b["n"] for b in design["blocks"] for c in range(b["count"]))
    check("measured_bits", widths_ok, "every circuit measures exactly n classical bits")
    diag = stats[stats.i == stats.j]
    off = stats[stats.i != stats.j]
    gates_kept = (diag.groupby("n").n_2q.min() >= off.groupby("n").n_2q.min() - 4).all() and (diag.n_2q > 0).all()
    check("diagonal_gates_kept", gates_kept, "diagonal circuits keep their 2Q gates: " +
          ", ".join(f"n={n}: {int(v)}" for n, v in diag.groupby("n").n_2q.min().items()))

    shots = args.shots
    ideal = AerSimulator(noise_model=build_noise_model(cal, 0.0), seed_simulator=args.seed).run(circuits, shots=shots).result()
    noisy = AerSimulator(noise_model=build_noise_model(cal, 1.0), seed_simulator=args.seed + 1).run(circuits, shots=shots).result()
    zmax, diag_ideal, rows = 0.0, [], []
    for b in design["blocks"]:
        n = b["n"]
        for c, (i, j) in enumerate(pairs):
            idx = b["first"] + c
            k = stats.k_exact.iloc[idx]
            p_i = p_zero(ideal.get_counts(idx), n)
            p_n = p_zero(noisy.get_counts(idx), n)
            zmax = max(zmax, abs(p_i - k) / np.sqrt(max(k * (1 - k), 1e-4) / shots))
            if i == j:
                diag_ideal.append(p_i)
            phys = [int(q) for q in str(stats.physical_qubits.iloc[idx]).strip("[]").split(",")][:n]
            ro_only = float(np.prod([1 - (cal["qubits"][q]["readout_error"] or 0) for q in phys]))
            rows.append({"n": n, "lam": b["lam"], "diag": i == j, "p_noisy": p_n, "ro_only": ro_only})
    check("ideal_matches_exact", zmax < 5.0, f"max |p_ideal - k_exact| = {zmax:.2f} shot-noise SDs (limit 5)")
    check("ideal_diagonal_is_one", all(v == 1.0 for v in diag_ideal), f"{len(diag_ideal)} diagonal circuits, min {min(diag_ideal):.4f}")

    r = pd.DataFrame(rows)
    detail, gate_noise_ok = [], True
    for n, g in r[r.diag].groupby("n"):
        ro_only = float(g.ro_only.mean())  # P(all-zero) if only readout error acted, per-circuit qubits
        dm = float(g.p_noisy.mean())
        detail.append(f"n={n}: diag {dm:.3f} vs readout-only {ro_only:.3f}")
        if n >= 4:
            gate_noise_ok &= dm < ro_only
    check("noise_model_degrades_diagonal_beyond_readout", gate_noise_ok, "; ".join(detail))

    try:
        for b in design["blocks"]:
            Kh = np.eye(design["m"])
            for c, (i, j) in enumerate(pairs):
                Kh[i, j] = Kh[j, i] = p_zero(noisy.get_counts(b["first"] + c), b["n"])
            spread_stats(Kh[np.triu_indices(design["m"], 1)], shots)
            bootstrap_excess_var(Kh, shots, n_boot=200)
        check("analysis_runs", True, "spread statistics and bootstrap computed on simulated counts")
    except Exception as e:  # pragma: no cover
        check("analysis_runs", False, repr(e))

    factor, source = _cost_factor()
    naive = _naive_estimate(stats, shots)
    projected = naive * factor
    remaining = None
    if not args.fake:
        _, service = get_backend(design["backend"], False)
        remaining = service.usage().get("usage_remaining_seconds")
    fits = remaining is None or remaining - projected >= args.reserve
    check("qpu_budget", fits, f"naive {naive:.0f} s x measured cost factor {factor:.2f} ({source}) = {projected:.0f} s projected; "
          f"remaining {remaining} s; reserve {args.reserve:.0f} s")

    report = {"passed": ok, "checked_utc": now(), "shots": shots, "qpy_sha256": _sha256(out / "circuits.qpy"),
              "projected_qpu_s": projected, "cost_factor": factor, "cost_factor_source": source,
              "remaining_at_check_s": remaining, "checks": checks}
    (out / "preflight.json").write_text(json.dumps(report, indent=1))
    print(f"PREFLIGHT {'PASSED' if ok else 'FAILED'} -> {out / 'preflight.json'}")
    if not ok:
        raise SystemExit(1)


def cmd_submit(args):
    out = tag_dir(args.tag, args.fake)
    if (out / "job.json").exists():
        raise SystemExit(f"{out} already has a submitted job; refusing to resubmit")
    design, circuits = _load(out)
    backend, service = get_backend(design["backend"], args.fake)
    if args.fake:
        sim = AerSimulator.from_backend(backend)
        # independent seed: the stand-in device must not share random draws with the matched simulation
        res = sim.run(circuits, shots=args.shots, seed_simulator=args.seed + 1000).result()
        counts = [res.get_counts(i) for i in range(len(circuits))]
        (out / "counts.json").write_text(json.dumps({"FAKE": True, "shots": args.shots, "counts": counts}))
        (out / "job.json").write_text(json.dumps({"FAKE": True, "job_id": None, "shots": args.shots,
                                                   "submitted_utc": now()}, indent=1))
        print(f"FAKE run complete: {out}")
        return
    pf_path = out / "preflight.json"
    if not pf_path.exists():
        raise SystemExit("No preflight.json: run `preflight` first.")
    pf = json.loads(pf_path.read_text())
    if not pf.get("passed"):
        raise SystemExit("Preflight did not pass; not submitting.")
    if pf["qpy_sha256"] != _sha256(out / "circuits.qpy"):
        raise SystemExit("circuits.qpy changed since preflight; re-run preflight.")
    if pf["shots"] != args.shots:
        raise SystemExit(f"Preflight was run for {pf['shots']} shots, not {args.shots}; re-run preflight.")
    usage = service.usage()
    usage.pop("instance_id", None)  # account identifier: keep it out of the public repo
    remaining = usage.get("usage_remaining_seconds")
    projected = pf["projected_qpu_s"]
    print(f"QPU allocation remaining: {remaining} s; projected for this job: {projected:.0f} s; reserve {args.reserve:.0f} s")
    if remaining is not None and remaining - projected < args.reserve:
        raise SystemExit("Projected usage would cut into the reserve; not submitting.")
    if not args.confirm:
        raise SystemExit("Dry check only. Add --confirm to submit (this USES QPU TIME).")
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    job = sampler.run([(c,) for c in circuits], shots=args.shots)
    info = {"job_id": job.job_id(), "backend": backend.name, "shots": args.shots, "submitted_utc": now(),
            "n_circuits": len(circuits), "usage_before": usage}
    (out / "job.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"Submitted job {info['job_id']} to {backend.name}; saved {out / 'job.json'}")


def _matched_sims(out, circuits, shots, cal, seed):
    sims = {}
    for label, scale in (("noise_model", 1.0), ("ideal_shots", 0.0)):
        sim = AerSimulator(noise_model=build_noise_model(cal, scale), seed_simulator=seed)
        res = sim.run(circuits, shots=shots).result()
        sims[label] = [res.get_counts(i) for i in range(len(circuits))]
    (out / "sim_counts.json").write_text(json.dumps({"shots": shots, **sims}))


def cmd_fetch(args):
    out = tag_dir(args.tag, args.fake)
    design, circuits = _load(out)
    job_info = json.loads((out / "job.json").read_text())
    shots = job_info["shots"]
    backend, service = get_backend(design["backend"], args.fake)
    if args.fake:
        cal = calibration_summary(backend)
    else:
        job = service.job(job_info["job_id"])
        result = job.result()  # blocks until done
        counts = [result[i].data.meas.get_counts() for i in range(len(circuits))]
        metrics = job.metrics()
        (out / "counts.json").write_text(json.dumps({"FAKE": False, "shots": shots, "counts": counts}))
        (out / "job_metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
        run_ts = (metrics.get("timestamps") or {}).get("running") or (metrics.get("timestamps") or {}).get("finished")
        run_dt = datetime.fromisoformat(str(run_ts).replace("Z", "+00:00")) if run_ts else None
        target = backend.target_history(datetime=run_dt) if run_dt else backend.target
        cal = calibration_summary(backend, target=target, properties_datetime=run_dt)
        cal["job_run_utc"] = str(run_ts)
        print(f"Job {job_info['job_id']}: status {job.status()}, usage {metrics.get('usage')}")
    save_snapshot(cal, out / "calibration_run.json")
    _matched_sims(out, circuits, shots, cal, design["seed"])
    print(f"Fetched and simulated: {out}")
    cmd_analyze(args)


def cmd_analyze(args):
    out = tag_dir(args.tag, args.fake)
    design, _ = _load(out)
    hw = json.loads((out / "counts.json").read_text())
    sims = json.loads((out / "sim_counts.json").read_text())
    shots, pairs, m = hw["shots"], [tuple(p) for p in design["pairs"]], design["m"]
    sources = {("FAKE_device" if hw["FAKE"] else "hardware"): hw["counts"],
               "noise_model": sims["noise_model"], "ideal_shots": sims["ideal_shots"]}
    rows, entries = [], []
    for blk in design["blocks"]:
        n, lam = blk["n"], blk["lam"]
        X = np.array(blk["X"])
        K = quantum_kernel(X, None, lam)
        iu = np.triu_indices(m, 1)
        for src, counts in sources.items():
            Kh = np.eye(m)
            for c, (i, j) in enumerate(pairs):
                v = p_zero(counts[blk["first"] + c], n)
                Kh[i, j] = Kh[j, i] = v
                entries.append({"source": src, "n": n, "lam": lam, "i": i, "j": j, "k_exact": K[i, j], "k_hat": v})
            sp = spread_stats(Kh[iu], shots)
            lo, hi = bootstrap_excess_var(Kh, shots, n_boot=2000, seed=design["seed"])
            rows.append({"source": src, "n": n, "lam": lam, "m": m, "shots": shots,
                         "exact_mean": float(K[iu].mean()), "exact_var": float(K[iu].var(ddof=1)),
                         **{f"meas_{k}": v for k, v in sp.items()}, "excess_var_lo": lo, "excess_var_hi": hi,
                         "diag_mean": float(np.mean(np.diag(Kh))),
                         "mae_vs_exact": float(np.mean(np.abs(Kh[iu] - K[iu])))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(entries).to_csv(out / "entries.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "analysis_meta.json").write_text(json.dumps({"analyzed_utc": now(), "git_commit": commit}, indent=1))
    cols = ["source", "n", "lam", "exact_var", "meas_raw_var", "meas_excess_var", "excess_var_lo", "excess_var_hi",
            "diag_mean", "mae_vs_exact"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "preflight", "submit", "fetch", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--tag", required=True)
        sp.add_argument("--fake", action="store_true")
        sp.add_argument("--seed", type=int, default=0)
        if name == "prepare":
            sp.add_argument("--backend", default="ibm_marrakesh")
            sp.add_argument("--qubits", nargs="+", type=int, default=[2, 4, 6])
            sp.add_argument("--lambdas", nargs="+", type=float, default=[0.0625, 0.25, 1.0])
            sp.add_argument("--m", type=int, default=10)
        if name in ("preflight", "submit"):
            sp.add_argument("--shots", type=int, default=1000)
            sp.add_argument("--reserve", type=float, default=120.0, help="seconds of QPU allocation to keep unused")
        if name == "submit":
            sp.add_argument("--confirm", action="store_true", help="actually submit (uses QPU time)")
    args = p.parse_args()
    {"prepare": cmd_prepare, "preflight": cmd_preflight, "submit": cmd_submit, "fetch": cmd_fetch,
     "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
