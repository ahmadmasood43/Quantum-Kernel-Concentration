"""Leakage-free preprocessing: every transform is fit on training rows only."""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor(n_components: int, seed: int, use_pca: bool = True) -> Pipeline:
    """Standardize -> PCA(n) -> standardize each component to unit variance.

    The final per-component standardization puts every qubit's input on the
    same scale, so the bandwidth lambda has the same meaning for all n.
    """
    steps = [("scale_in", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    steps.append(("scale_out", StandardScaler()))
    return Pipeline(steps)


def fit_transform_split(X, train_idx, test_idx, n_components: int, seed: int, use_pca: bool = True):
    pre = make_preprocessor(n_components, seed, use_pca)
    Xtr = pre.fit_transform(X[train_idx])
    Xte = pre.transform(X[test_idx])
    return Xtr, Xte, pre
