import numpy as np
import pandas as pd

from qkc.folds import make_folds
from qkc.kernels import rbf_kernel
from qkc.sweep import dispersion_test, sample_gram, select_tuned, sweep_fold


def test_sample_gram_is_symmetric_unbiased_with_unit_diagonal():
    rng = np.random.default_rng(0)
    K = np.full((40, 40), 0.3)
    np.fill_diagonal(K, 1.0)
    Kh = sample_gram(K, 1000, rng, symmetric=True)
    np.testing.assert_array_equal(Kh, Kh.T)
    np.testing.assert_array_equal(np.diag(Kh), 1.0)
    off = Kh[np.triu_indices(40, 1)]
    assert abs(off.mean() - 0.3) < 0.005
    # binomial variance c(1-c)/N
    assert abs(off.var() / (0.3 * 0.7 / 1000) - 1.0) < 0.15


def test_exact_shots_return_input_unchanged():
    K = np.eye(3)
    assert sample_gram(K, np.inf, np.random.default_rng(0), True) is K


def test_dispersion_test_has_nominal_size_under_complete_concentration():
    """Constant true kernel -> p-values roughly uniform -> about 5% rejections."""
    rng = np.random.default_rng(1)
    K = np.full((30, 30), 0.6)
    np.fill_diagonal(K, 1.0)
    pvals = [dispersion_test(sample_gram(K, 500, rng, True), 500)[1] for _ in range(1000)]
    rate = np.mean(np.array(pvals) < 0.05)
    assert 0.03 < rate < 0.07


def test_dispersion_test_detects_spread_well_above_shot_noise():
    rng = np.random.default_rng(2)
    A = rng.uniform(0.4, 0.8, size=(30, 30))
    K = (A + A.T) / 2
    np.fill_diagonal(K, 1.0)
    _, p = dispersion_test(sample_gram(K, 10_000, rng, True), 10_000)
    assert p < 1e-6


def test_select_tuned_uses_inner_accuracy_only():
    df = pd.DataFrame(
        {"g": [0, 0, 0], "h": [0.1, 0.2, 0.4], "inner_acc": [0.7, 0.9, 0.9], "test_acc": [1.0, 0.5, 0.6]}
    )
    picked = select_tuned(df, ["g"])
    assert picked["h"].item() == 0.2  # highest inner, tie -> smaller h; test_acc ignored


def test_sweep_fold_runs_and_reports_every_h_and_shot():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 6))
    y = (X[:, 0] > 0).astype(int)
    fold = make_folds(y, seed=0)[0]
    # an RBF kernel takes values in (0, 1], so it can stand in for a fidelity kernel here
    kfn = lambda A, B, g: rbf_kernel(A, B, g / A.shape[1])
    recs = sweep_fold(X, y, fold, 3, 0, kfn, [0.1, 1.0], [np.inf, 100], True, 0, is_fidelity=True)
    assert len(recs) == 4
    exact = [r for r in recs if not np.isfinite(r["shots"])]
    assert all(np.isnan(r["disp_p"]) for r in exact)
    assert all(0 <= r["test_acc"] <= 1 for r in recs)


def test_select_tuned_is_correct_with_duplicate_index_labels():
    a = pd.DataFrame({"g": [0, 0], "h": [0.1, 0.2], "inner_acc": [0.9, 0.5], "test_acc": [0.8, 0.1]})
    b = pd.DataFrame({"g": [1, 1], "h": [0.1, 0.2], "inner_acc": [0.5, 0.9], "test_acc": [0.2, 0.7]})
    picked = select_tuned(pd.concat([a, b]), ["g"])  # index labels 0,1,0,1
    assert len(picked) == 2 and picked["test_acc"].tolist() == [0.8, 0.7]
