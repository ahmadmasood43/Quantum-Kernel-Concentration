"""Checks that no preprocessing or model-selection step sees outer-test rows."""

import numpy as np

from qkc.evaluate import nested_cv
from qkc.folds import make_folds
from qkc.kernels import rbf_kernel
from qkc.preprocess import fit_transform_split


def _toy(seed=0, m=60, d=12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, d))
    y = (X[:, 0] + 0.3 * rng.standard_normal(m) > 0).astype(int)
    return X, y


def test_preprocessor_statistics_come_from_train_rows_only():
    X, y = _toy()
    tr, te = np.arange(40), np.arange(40, 60)
    _, _, pre = fit_transform_split(X, tr, te, n_components=3, seed=0)
    np.testing.assert_allclose(pre.named_steps["scale_in"].mean_, X[tr].mean(0))


def test_corrupting_test_rows_does_not_change_preprocessing_fit():
    X, y = _toy()
    tr, te = np.arange(40), np.arange(40, 60)
    X_bad = X.copy()
    X_bad[te] = 1e6 * np.random.default_rng(9).standard_normal(X[te].shape)
    Xtr_a, _, _ = fit_transform_split(X, tr, te, 3, 0)
    Xtr_b, _, _ = fit_transform_split(X_bad, tr, te, 3, 0)
    np.testing.assert_allclose(Xtr_a, Xtr_b)


def test_model_selection_ignores_outer_test_rows():
    X, y = _toy()
    folds = make_folds(y, seed=0)
    te0 = np.asarray(folds[0]["test"])
    X_bad = X.copy()
    X_bad[te0] = 1e6 * np.random.default_rng(3).standard_normal(X[te0].shape)

    kfn = lambda A, B, g: rbf_kernel(A, B, g / A.shape[1])
    grid = np.logspace(-2, 1, 4)
    r_clean = nested_cv(X, y, folds[:1], 3, 0, kfn, grid)
    r_bad = nested_cv(X_bad, y, folds[:1], 3, 0, kfn, grid)
    assert r_clean[0]["h_best"] == r_bad[0]["h_best"]
    assert r_clean[0]["C_best"] == r_bad[0]["C_best"]
    assert r_clean[0]["inner_acc"] == r_bad[0]["inner_acc"]


def test_outer_folds_partition_the_data_and_inner_folds_stay_inside_train():
    _, y = _toy()
    folds = make_folds(y, seed=0)
    all_test = np.concatenate([f["test"] for f in folds])
    assert sorted(all_test.tolist()) == list(range(len(y)))
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        inner_val = []
        for inner in f["inner"]:
            # inner folds hold absolute row indices drawn only from outer-train rows
            assert set(inner["train"]) | set(inner["val"]) == set(f["train"])
            assert not set(inner["train"]) & set(inner["val"])
            inner_val += inner["val"]
        assert sorted(inner_val) == sorted(f["train"])
