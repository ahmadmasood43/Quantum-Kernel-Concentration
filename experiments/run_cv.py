"""Nested-CV test accuracy for quantum and classical models on fixed folds.

Examples:
    .venv/bin/python experiments/run_cv.py --dataset fmnist --models majority linear logreg rbf quantum
    .venv/bin/python experiments/run_cv.py --dataset synthetic --models rbf quantum --qubits 2 4 6

Writes results/cv/<dataset>_<model>.csv (one row per seed x n x outer fold) and a
matching .meta.json with the configuration, fold-file hashes, and package versions.
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import qiskit
import sklearn
from sklearn.linear_model import LogisticRegression

from qkc import config
from qkc.data import SYNTH_SEPARATION, load_fmnist_pair, make_synthetic
from qkc.evaluate import nested_cv
from qkc.folds import load_or_create_folds
from qkc.kernels import quantum_kernel, rbf_kernel
from qkc.preprocess import fit_transform_split



def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load(dataset: str, seed: int, n: int):
    if dataset == "fmnist":
        X, y = load_fmnist_pair(seed)
        return X, y, True
    X, y = make_synthetic(seed, n_features=n, separation=SYNTH_SEPARATION)
    return X, y, False


def run_majority(y, folds):
    out = []
    for k, f in enumerate(folds):
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        maj = np.bincount(y[tr]).argmax()
        out.append({"outer_fold": k, "h_best": np.nan, "C_best": np.nan, "h_at_grid_edge": False,
                    "inner_acc": np.nan, "test_acc": float((y[te] == maj).mean())})
    return out


def run_logreg(X, y, folds, n, seed, use_pca):
    out = []
    for k, f in enumerate(folds):
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        scores = np.zeros(len(config.C_GRID))
        for inner in f["inner"]:
            itr, iva = np.asarray(inner["train"]), np.asarray(inner["val"])
            A, B, _ = fit_transform_split(X, itr, iva, n, seed, use_pca)
            for b, C in enumerate(config.C_GRID):
                clf = LogisticRegression(C=C, max_iter=5000).fit(A, y[itr])
                scores[b] += (clf.predict(B) == y[iva]).mean()
        b = int(np.argmax(scores))
        A, B, _ = fit_transform_split(X, tr, te, n, seed, use_pca)
        clf = LogisticRegression(C=config.C_GRID[b], max_iter=5000).fit(A, y[tr])
        out.append({"outer_fold": k, "h_best": np.nan, "C_best": float(config.C_GRID[b]),
                    "h_at_grid_edge": False, "inner_acc": float(scores[b] / len(f["inner"])),
                    "test_acc": float((clf.predict(B) == y[te]).mean())})
    return out


def run_model(model, X, y, folds, n, seed, use_pca):
    if model == "majority":
        return run_majority(y, folds)
    if model == "logreg":
        return run_logreg(X, y, folds, n, seed, use_pca)
    if model == "linear":
        kfn, grid = (lambda A, B, h: A @ (A if B is None else B).T), np.array([1.0])
    elif model == "rbf":
        kfn, grid = (lambda A, B, g: rbf_kernel(A, B, g / A.shape[1])), config.RBF_G_GRID_V1
    elif model == "quantum":
        kfn = lambda A, B, lam: quantum_kernel(A, B, lam, config.FEATURE_MAP_REPS)
        grid = config.LAMBDA_GRID_V1
    else:
        raise ValueError(model)
    # linear kernel: rows of K_te are test points against train points
    return nested_cv(X, y, folds, n, seed, kfn, grid, use_pca)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fmnist", "synthetic"], required=True)
    p.add_argument("--models", nargs="+", default=["majority", "linear", "logreg", "rbf", "quantum"])
    p.add_argument("--qubits", nargs="+", type=int, default=list(config.QUBIT_COUNTS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--outdir", default=str(config.RESULTS / "cv"))
    args = p.parse_args()

    outdir = config.ROOT / args.outdir  # absolute paths override ROOT when joined
    outdir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        rows, fold_hashes = [], {}
        t0 = time.time()
        for seed in args.seeds:
            for n in args.qubits:
                X, y, use_pca = load(args.dataset, seed, n)
                folds, h = load_or_create_folds(args.dataset, seed, y)
                fold_hashes[f"seed{seed}"] = h
                for r in run_model(model, X, y, folds, n, seed, use_pca):
                    rows.append({"dataset": args.dataset, "model": model, "seed": seed, "n": n, **r})
            print(f"[{model}] seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        stem = outdir / f"{args.dataset}_{model}"
        df.to_csv(stem.with_suffix(".csv"), index=False)
        meta = {
            "dataset": args.dataset, "model": model, "qubits": args.qubits, "seeds": args.seeds,
            "lambda_grid": config.LAMBDA_GRID_V1.tolist(), "rbf_g_grid": config.RBF_G_GRID_V1.tolist(),
            "C_grid": config.C_GRID.tolist(), "feature_map_reps": config.FEATURE_MAP_REPS,
            "n_per_class": config.N_PER_CLASS, "fmnist_classes": config.FMNIST_CLASSES,
            "synthetic_separation": SYNTH_SEPARATION, "fold_file_sha256": fold_hashes,
            "git_commit": git_commit(), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "qiskit": qiskit.__version__,
            "runtime_s": round(time.time() - t0, 1),
        }
        stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        summ = df.groupby("n")["test_acc"].agg(["mean", "std"]).round(3)
        print(f"== {args.dataset} / {model} ==\n{summ.T.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
