"""Dataset loading: Fashion-MNIST binary pair and a controlled synthetic set."""

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from qkc.config import DATA_RAW, FMNIST_CLASSES, N_PER_CLASS

# Official Fashion-MNIST files from the dataset authors' repository.
FMNIST_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FMNIST_FILES = {
    "images": "train-images-idx3-ubyte.gz",
    "labels": "train-labels-idx1-ubyte.gz",
}
# SHA-256 of the files used for every reported result (see data/README.md).
FMNIST_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
}
SYNTH_SEPARATION = 2.0  # Bayes-optimal accuracy Phi(1) ~ 0.841 for every n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 {actual} does not match the recorded {expected}")


def download_fmnist(dest: Path = DATA_RAW) -> dict[str, Path]:
    """Download (if needed) and verify the Fashion-MNIST training files."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fname in FMNIST_FILES.items():
        p = dest / fname
        if not p.exists():
            urllib.request.urlretrieve(FMNIST_BASE + fname, p)
        verify_sha256(p, FMNIST_SHA256[fname])
        paths[key] = p
    return paths


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_fmnist_pair(seed: int, n_per_class: int = N_PER_CLASS, classes=FMNIST_CLASSES):
    """Class-balanced subsample of two Fashion-MNIST training classes.

    Returns X of shape (2*n_per_class, 784) as float64 in [0, 1] and y in {0, 1}.
    """
    paths = download_fmnist()
    images = _read_idx(paths["images"]).reshape(-1, 784)
    labels = _read_idx(paths["labels"])
    rng = np.random.default_rng(seed)
    idx = []
    for c in classes:
        pool = np.flatnonzero(labels == c)
        idx.append(rng.choice(pool, size=n_per_class, replace=False))
    idx = np.concatenate(idx)
    X = images[idx].astype(np.float64) / 255.0
    y = (labels[idx] == classes[1]).astype(int)
    return X, y


def make_synthetic(seed: int, n_features: int, n_per_class: int = N_PER_CLASS, separation: float = SYNTH_SEPARATION):
    """Two isotropic Gaussian classes in R^n with means at +/- (separation/2) * u.

    u is a random unit vector, so the Bayes-optimal accuracy is
    Phi(separation / 2) regardless of n.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_features)
    u /= np.linalg.norm(u)
    X0 = rng.standard_normal((n_per_class, n_features)) - 0.5 * separation * u
    X1 = rng.standard_normal((n_per_class, n_features)) + 0.5 * separation * u
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_per_class, int), np.ones(n_per_class, int)]
    return X, y
