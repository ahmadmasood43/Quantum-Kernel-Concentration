"""Noise-model simulation of kernel estimation (T8) and noise-strength sweep (H6).

For each qubit count n and bandwidth lambda, m Fashion-MNIST points are
selected (outer-fold-0 training rows, seed 0). Every pair (i<j) and every
diagonal (i, i) compute-uncompute circuit is transpiled to the snapshot's
coupling map and native gates, then sampled with `shots` shots under the
snapshot noise model scaled by each factor in --scales (0 = ideal, finite shots).

    .venv/bin/python experiments/noise_sim.py --calibration results/hardware/calibration_ibm_marrakesh_<ts>.json

Writes results/noise/entries.parquet (one row per circuit and scale) and
results/noise/summary.csv (one row per n, lambda, scale) + meta.json.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from qkc.config import RESULTS, ROOT
from qkc.hwexp import bootstrap_excess_var, isa_stats, kernel_circuit, p_zero, pair_list, select_points, spread_stats
from qkc.kernels import quantum_kernel
from qkc.noise import build_noise_model, coupling_map, load_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", required=True)
    p.add_argument("--qubits", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.add_argument("--lambdas", nargs="+", type=float, default=[2.0**-4, 2.0**-2, 2.0**-1, 1.0])
    p.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/noise")
    args = p.parse_args()

    cal = load_snapshot(args.calibration)
    cmap = coupling_map(cal)
    pm = generate_preset_pass_manager(
        optimization_level=3, coupling_map=cmap, basis_gates=["cz", "rz", "sx", "x"], seed_transpiler=args.seed
    )
    models = {s: build_noise_model(cal, s) for s in args.scales}
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    entries, summary = [], []
    t0 = time.time()
    pairs = pair_list(args.m, include_diagonal=True)
    for n in args.qubits:
        X, y = select_points(n, args.m, args.seed)
        for lam in args.lambdas:
            K = quantum_kernel(X, None, lam)
            circs = [kernel_circuit(X[i], X[j], lam) for i, j in pairs]
            isas = pm.run(circs)
            st = [isa_stats(c) for c in isas]
            for s in args.scales:
                sim = AerSimulator(noise_model=models[s], seed_simulator=args.seed)
                res = sim.run(isas, shots=args.shots).result()
                Kh = np.eye(args.m)
                for c, (i, j) in enumerate(pairs):
                    v = p_zero(res.get_counts(c), n)
                    Kh[i, j] = Kh[j, i] = v
                    entries.append({"n": n, "lam": lam, "scale": s, "i": i, "j": j, "k_exact": K[i, j], "k_hat": v,
                                    "depth": st[c]["depth"], "n_2q": st[c]["n_2q"]})
                iu = np.triu_indices(args.m, 1)
                sp = spread_stats(Kh[iu], args.shots)
                lo, hi = bootstrap_excess_var(Kh, args.shots, n_boot=1000, seed=args.seed)
                offd = [c for c, (i, j) in enumerate(pairs) if i != j]
                summary.append({
                    "n": n, "lam": lam, "scale": s, "m": args.m, "shots": args.shots,
                    "exact_mean": float(K[iu].mean()), "exact_var": float(K[iu].var(ddof=1)),
                    **{f"meas_{k}": v for k, v in sp.items()},
                    "excess_var_lo": lo, "excess_var_hi": hi,
                    "diag_mean": float(np.mean(np.diag(Kh))),  # measured k(x, x); 1 on an ideal device
                    "median_depth": float(np.median([st[c]["depth"] for c in offd])),
                    "median_n_2q": float(np.median([st[c]["n_2q"] for c in offd])),
                    "max_n_2q": int(max(st[c]["n_2q"] for c in offd)),
                })
                print(f"n={n:2d} lam={lam:.4f} scale={s:.1f} exact_var={summary[-1]['exact_var']:.2e} "
                      f"excess_var={sp['excess_var']:.2e} [{lo:.1e},{hi:.1e}] diag={summary[-1]['diag_mean']:.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        # write after every n so partial progress is kept
        pd.DataFrame(entries).to_parquet(outdir / "entries.parquet", index=False)
        pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {**vars(args), "calibration_backend": cal["backend"], "calibration_snapshot_utc": cal["snapshot_utc"],
            "calibration_last_update": cal["calibration_last_update"], "optimization_level": 3,
            "basis_gates": ["cz", "rz", "sx", "x"], "git_commit": commit, "runtime_s": round(time.time() - t0, 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
