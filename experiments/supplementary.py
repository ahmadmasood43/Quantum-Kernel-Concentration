"""Supplementary numbers quoted in the report and docs, computed from saved
results (no experiments are re-run here).

    .venv/bin/python experiments/supplementary.py

1. Null calibration of the binomial dispersion test (simulation, fixed seed).
2. Week 1 grid-edge rates: how often the Week 1 search (results/cv/, the
   narrower V1 grids) selected the lowest bandwidth value.
3. Shot-limit extrapolation at lambda = 1: the qubit count at which the
   expected number of all-zero outcomes N * mean(k) drops below 1, from a
   log-linear fit of the mean kernel over n = 2-16 (an extrapolation beyond
   the measured range).
4. Provenance of the main sweep: the commits recorded in its metadata and
   the code files that differ between them.

Writes results/tables/supplementary.json.
"""

import glob
import json
import subprocess

import numpy as np
import pandas as pd

from qkc import config
from qkc.sweep import dispersion_test, sample_gram

OUT = config.RESULTS / "tables" / "supplementary.json"


def dispersion_null_calibration(reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c, N, m in ((0.6, 500, 30), (0.99, 1000, 60), (0.95, 10_000, 60)):
        K = np.full((m, m), c)
        np.fill_diagonal(K, 1.0)
        p = np.array([dispersion_test(sample_gram(K, N, rng, True), N)[1] for _ in range(reps)])
        rows.append({"c": c, "shots": N, "m": m, "reps": reps, "rejection_rate": float(np.mean(p < 0.05))})
    return rows


def week1_grid_edges():
    rows = []
    for ds in ("fmnist", "synthetic"):
        for model, grid in (("quantum", config.LAMBDA_GRID_V1), ("rbf", config.RBF_G_GRID_V1)):
            d = pd.read_csv(config.RESULTS / "cv" / f"{ds}_{model}.csv")
            lower = np.isclose(d.h_best, grid.min())
            upper = np.isclose(d.h_best, grid.max())
            per_n = d.assign(lower=lower).groupby("n").lower.mean()
            rows.append({"dataset": ds, "model": model, "lower_edge_overall": float(lower.mean()),
                         "upper_edge_overall": float(upper.mean()),
                         "lower_edge_min_over_n": float(per_n.min()), "lower_edge_max_over_n": float(per_n.max())})
    return rows


def shot_limit_extrapolation():
    files = glob.glob(str(config.RESULTS / "sweep" / "main" / "fmnist_quantum_*.parquet"))
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    k = q[~np.isfinite(q.shots) & np.isclose(q.h, 1.0)].groupby("n").offdiag_mean.mean()
    slope, intercept = np.polyfit(k.index, np.log(k.values), 1)
    return {"fit": {"intercept": float(intercept), "slope_per_qubit": float(slope),
                    "halving_qubits": float(np.log(2) / -slope)},
            "n_star": {f"{int(N)}": float((np.log(1.0 / N) - intercept) / slope) for N in (1e3, 1e4, 1e5)},
            "note": "extrapolation beyond the measured n = 2-16"}


def sweep_provenance():
    commits = sorted({json.load(open(f))["git_commit"] for f in glob.glob(str(config.RESULTS / "sweep" / "main" / "*.meta.json"))})
    files = ["src/qkc/kernels.py", "src/qkc/sweep.py", "src/qkc/preprocess.py", "src/qkc/data.py",
             "src/qkc/folds.py", "src/qkc/config.py", "experiments/run_sweep.py"]
    diffs = {}
    for c in commits[1:]:
        out = subprocess.run(["git", "diff", "--stat", commits[0], c, "--", *files], cwd=config.ROOT,
                             capture_output=True, text=True).stdout.strip()
        diffs[c[:7]] = out or "identical"
    return {"commits": [c[:7] for c in commits], "code_diff_vs_first": diffs}


def main():
    res = {"dispersion_null_calibration": dispersion_null_calibration(),
           "week1_grid_edges": week1_grid_edges(),
           "shot_limit_extrapolation": shot_limit_extrapolation(),
           "sweep_provenance": sweep_provenance()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
