"""Main simulator sweep: accuracy and kernel statistics for every
(dataset, model, n, seed, outer fold, bandwidth, shot budget).

Examples:
    # main runs (T1-T4, T6)
    .venv/bin/python experiments/run_sweep.py --dataset fmnist --model quantum --jobs 4
    .venv/bin/python experiments/run_sweep.py --dataset fmnist --model rbf --jobs 4
    # ablations (H5)
    .venv/bin/python experiments/run_sweep.py --dataset fmnist --model quantum --reps 1 --tag reps1 --shots inf
    .venv/bin/python experiments/run_sweep.py --dataset fmnist --model quantum --product --tag product --shots inf
    .venv/bin/python experiments/run_sweep.py --dataset fmnist --model quantum --n-per-class 50 --tag m100 --shots inf

Output: results/sweep/<tag>/<dataset>_<model>_n<n>_seed<s>.parquet plus a
.meta.json per shard. Existing shards are skipped, so an interrupted run resumes.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from qkc import config


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_shard(task: dict) -> str:
    from threadpoolctl import threadpool_limits

    import qiskit
    import sklearn

    from qkc.data import load_fmnist_pair, make_synthetic
    from qkc.folds import load_or_create_folds
    from qkc.kernels import quantum_kernel, rbf_kernel
    from qkc.sweep import sweep_fold

    out = Path(task["out"])
    if out.exists():
        return f"skip {out.name}"
    t0 = time.time()
    n, seed, npc = task["n"], task["seed"], task["n_per_class"]
    with threadpool_limits(limits=task["threads"]):
        if task["dataset"] == "fmnist":
            X, y = load_fmnist_pair(seed, n_per_class=npc)
            use_pca = True
        else:
            X, y = make_synthetic(seed, n_features=n, n_per_class=npc, separation=task["synth_sep"])
            use_pca = False
        stem = task["dataset"] if npc == config.N_PER_CLASS else f"{task['dataset']}_m{2 * npc}"
        folds, fold_hash = load_or_create_folds(stem, seed, y)

        if task["model"] == "quantum":
            reps, ent = task["reps"], not task["product"]
            kfn = lambda A, B, lam: quantum_kernel(A, B, lam, reps, ent)
            grid, fid = config.LAMBDA_GRID, True
        else:
            kfn = lambda A, B, g: rbf_kernel(A, B, g / A.shape[1])
            grid, fid = config.RBF_G_GRID, False

        rows = []
        for k, fold in enumerate(folds):
            rng_seed = int(np.random.SeedSequence([seed, n, k, npc]).generate_state(1)[0])
            for r in sweep_fold(X, y, fold, n, seed, kfn, grid, task["shots"], use_pca, rng_seed, fid):
                rows.append({"dataset": task["dataset"], "model": task["model"], "tag": task["tag"],
                             "n": n, "seed": seed, "outer_fold": k, **r})
    df = pd.DataFrame(rows)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    meta = {k: v for k, v in task.items() if k != "out"}
    meta.update({
        "grid": list(map(float, grid)), "C_grid": config.C_GRID.tolist(), "fold_file": stem,
        "fold_file_sha256": fold_hash, "git_commit": task["git_commit"], "python": platform.python_version(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "qiskit": qiskit.__version__,
        "runtime_s": round(time.time() - t0, 1),
    })
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, out)  # shard appears only when complete
    return f"done {out.name} ({time.time() - t0:.0f}s)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fmnist", "synthetic"], required=True)
    p.add_argument("--model", choices=["quantum", "rbf"], required=True)
    p.add_argument("--qubits", nargs="+", type=int, default=list(config.QUBIT_COUNTS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--shots", nargs="+", type=float, default=list(config.SHOTS))
    p.add_argument("--reps", type=int, default=config.FEATURE_MAP_REPS)
    p.add_argument("--product", action="store_true", help="drop ZZ entangling terms")
    p.add_argument("--n-per-class", type=int, default=config.N_PER_CLASS)
    p.add_argument("--tag", default="main")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--synth-sep", type=float, default=2.0)  # = qkc.data.SYNTH_SEPARATION
    args = p.parse_args()

    outdir = config.RESULTS / "sweep" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    commit = git_commit()
    tasks = [
        {"dataset": args.dataset, "model": args.model, "n": n, "seed": s, "shots": args.shots,
         "reps": args.reps, "product": args.product, "n_per_class": args.n_per_class, "tag": args.tag,
         "synth_sep": args.synth_sep, "threads": threads, "git_commit": commit,
         "out": str(outdir / f"{args.dataset}_{args.model}_n{n}_seed{s}.parquet")}
        for n in sorted(args.qubits, reverse=True)  # largest first for load balance
        for s in args.seeds
    ]
    t0 = time.time()
    if args.jobs == 1:
        for t in tasks:
            print(run_shard(t), flush=True)
    else:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = [ex.submit(run_shard, t) for t in tasks]
            for f in as_completed(futs):
                print(f.result(), f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
