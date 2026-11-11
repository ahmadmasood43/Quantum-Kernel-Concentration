r"""Sweep engine: one pass over (outer fold, inner splits, bandwidth grid) that
records, for every bandwidth h and every shot budget N:

* inner-CV accuracy (best C) and outer-test accuracy at that C -> accuracy
  as a function of fixed h, and (by selecting h on inner accuracy afterwards)
  tuned accuracy;
* off-diagonal statistics of the exact outer-training Gram matrix;
* a test of whether the finite-shot Gram matrix still carries any spread
  beyond shot noise.

Finite-shot model. On an ideal device, the compute-uncompute estimate of
k(x, x') with N shots is Binomial(N, k)/N; diagonal entries are exactly 1.
Entries are sampled independently.

Distinguishability test (binomial dispersion test). Under
H0: "every off-diagonal kernel value equals a common constant c" (complete
concentration), the M off-diagonal estimates are i.i.d. Binomial(N, c)/N, so
D = (M-1) s^2 / (c_hat (1 - c_hat) / N) is approximately chi^2_{M-1}.
Rejecting H0 (p < 0.05) means the spread of the measured kernel cannot be
explained by shot noise alone. The test is valid under H0 regardless of the
dependence structure of the true kernel, because under H0 the only randomness
is the independent shot noise.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import chi2
from sklearn.svm import SVC

from qkc.config import C_GRID
from qkc.preprocess import fit_transform_split

KernelFn = Callable[[np.ndarray, np.ndarray | None, float], np.ndarray]


def sample_gram(K: np.ndarray, shots: float, rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    """Finite-shot estimate of a fidelity Gram matrix (shots = inf returns K)."""
    if not np.isfinite(shots):
        return K
    N = int(shots)
    P = np.clip(K, 0.0, 1.0)
    if not symmetric:
        return rng.binomial(N, P) / N
    iu = np.triu_indices_from(P, k=1)
    Kh = np.ones_like(P)
    vals = rng.binomial(N, P[iu]) / N
    Kh[iu] = vals
    Kh[(iu[1], iu[0])] = vals
    return Kh


def dispersion_test(Kh: np.ndarray, shots: float) -> tuple[float, float]:
    """Return (statistic / dof, p-value) of the binomial dispersion test on the
    off-diagonal entries of a finite-shot Gram matrix. NaN for exact kernels."""
    if not np.isfinite(shots):
        return np.nan, np.nan
    v = Kh[np.triu_indices_from(Kh, k=1)]
    M = v.size
    c = v.mean()
    denom = c * (1.0 - c) / shots
    if denom <= 0.0:
        return 0.0, 1.0  # every estimate identical (all 0 or all 1): no detectable spread
    D = (M - 1) * v.var(ddof=1) / denom
    return D / (M - 1), float(chi2.sf(D, M - 1))


def _scores(Ktr, ytr, Kte, yte) -> np.ndarray:
    out = np.empty(len(C_GRID))
    for b, C in enumerate(C_GRID):
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(Ktr, ytr)
        out[b] = (clf.predict(Kte) == yte).mean()
    return out


def sweep_fold(
    X, y, fold, n_components, seed, kernel_fn: KernelFn, h_grid, shots_list, use_pca, rng_seed, is_fidelity
) -> list[dict]:
    """All records for one outer fold. `is_fidelity` enables shot sampling and
    the dispersion test (only meaningful for kernels estimated as probabilities)."""
    shots_list = list(shots_list) if is_fidelity else [np.inf]
    nH, nN, nC = len(h_grid), len(shots_list), len(C_GRID)
    inner = np.zeros((nH, nN, nC))
    outer = np.zeros((nH, nN, nC))
    stats = []
    rng = np.random.default_rng(rng_seed)

    splits = [(np.asarray(f["train"]), np.asarray(f["val"]), False) for f in fold["inner"]]
    splits.append((np.asarray(fold["train"]), np.asarray(fold["test"]), True))
    for tr, te, is_outer in splits:
        Xtr, Xte, _ = fit_transform_split(X, tr, te, n_components, seed, use_pca)
        for a, h in enumerate(h_grid):
            Ktr = kernel_fn(Xtr, None, h)
            Kte = kernel_fn(Xte, Xtr, h)
            if is_outer:
                iu = np.triu_indices_from(Ktr, k=1)
                v = Ktr[iu]
                stats.append(
                    {
                        "offdiag_mean": float(v.mean()),
                        "offdiag_var": float(v.var(ddof=1)),
                        "mean_k1mk": float(np.mean(v * (1.0 - v))),
                        "n_pairs": int(v.size),
                    }
                )
            for c, N in enumerate(shots_list):
                Ktr_h = sample_gram(Ktr, N, rng, symmetric=True)
                Kte_h = sample_gram(Kte, N, rng, symmetric=False)
                s = _scores(Ktr_h, y[tr], Kte_h, y[te])
                if is_outer:
                    outer[a, c] = s
                    D, p = dispersion_test(Ktr_h, N)
                    stats[a].setdefault("disp", {})[c] = (D, p)
                else:
                    inner[a, c] += s / len(fold["inner"])

    records = []
    for a, h in enumerate(h_grid):
        for c, N in enumerate(shots_list):
            b = int(np.argmax(inner[a, c]))  # ties -> smaller C
            st = stats[a]
            D, p = st["disp"][c]
            records.append(
                {
                    "h": float(h),
                    "shots": float(N),
                    "C_best": float(C_GRID[b]),
                    "inner_acc": float(inner[a, c, b]),
                    "test_acc": float(outer[a, c, b]),
                    "offdiag_mean": st["offdiag_mean"],
                    "offdiag_var": st["offdiag_var"],
                    "mean_k1mk": st["mean_k1mk"],
                    "n_pairs": st["n_pairs"],
                    # ratio of true spread to the expected shot-noise variance
                    "signal_to_shot": (st["offdiag_var"] / (st["mean_k1mk"] / N)) if np.isfinite(N) and st["mean_k1mk"] > 0 else np.nan,
                    "disp_stat": D,
                    "disp_p": p,
                }
            )
    return records


def select_tuned(df, group_cols):
    """Tuned-h results: within each group, pick the h with the highest inner
    accuracy (ties -> smallest h). Uses inner-CV accuracy only, never test."""
    df = df.sort_values(group_cols + ["h"])
    idx = df.groupby(group_cols)["inner_acc"].idxmax()
    return df.loc[idx]
