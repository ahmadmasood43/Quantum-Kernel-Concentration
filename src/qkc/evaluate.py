"""Nested cross-validation with precomputed kernels, shared by all kernel models."""

from collections.abc import Callable

import numpy as np
from sklearn.svm import SVC

from qkc.config import C_GRID
from qkc.preprocess import fit_transform_split

# kernel_fn(XA, XB_or_None, h) -> Gram matrix; h is the bandwidth hyperparameter.
KernelFn = Callable[[np.ndarray, np.ndarray | None, float], np.ndarray]


def _svm_score(Ktr, ytr, Kte, yte, C) -> float:
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(Ktr, ytr)
    return float((clf.predict(Kte) == yte).mean())


def nested_cv(X, y, folds, n_components, seed, kernel_fn: KernelFn, h_grid, use_pca=True) -> list[dict]:
    """For each outer fold: select (h, C) by inner CV on the outer-train rows only,
    refit on the full outer-train rows, and score once on the outer-test rows.
    """
    records = []
    for k, fold in enumerate(folds):
        tr = np.asarray(fold["train"])
        te = np.asarray(fold["test"])

        # Inner CV: preprocessing is refit on each inner-train split.
        inner_scores = np.zeros((len(h_grid), len(C_GRID)))
        for inner in fold["inner"]:
            # Fold files store absolute row indices (subsets of the outer-train rows).
            itr, iva = np.asarray(inner["train"]), np.asarray(inner["val"])
            Xitr, Xiva, _ = fit_transform_split(X, itr, iva, n_components, seed, use_pca)
            for a, h in enumerate(h_grid):
                K_tr = kernel_fn(Xitr, None, h)
                K_va = kernel_fn(Xiva, Xitr, h)
                for b, C in enumerate(C_GRID):
                    inner_scores[a, b] += _svm_score(K_tr, y[itr], K_va, y[iva], C)
        inner_scores /= len(fold["inner"])
        a, b = np.unravel_index(np.argmax(inner_scores), inner_scores.shape)
        h_best, C_best = float(h_grid[a]), float(C_GRID[b])

        Xtr, Xte, _ = fit_transform_split(X, tr, te, n_components, seed, use_pca)
        K_tr = kernel_fn(Xtr, None, h_best)
        K_te = kernel_fn(Xte, Xtr, h_best)
        acc = _svm_score(K_tr, y[tr], K_te, y[te], C_best)
        records.append(
            {
                "outer_fold": k,
                "h_best": h_best,
                "C_best": C_best,
                "h_at_grid_edge": bool(a in (0, len(h_grid) - 1)),
                "inner_acc": float(inner_scores[a, b]),
                "test_acc": acc,
            }
        )
    return records
