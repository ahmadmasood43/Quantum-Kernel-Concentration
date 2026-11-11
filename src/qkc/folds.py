"""Fixed, on-disk nested cross-validation folds shared by every model."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from qkc.config import FOLDS_DIR, N_INNER_FOLDS, N_OUTER_FOLDS


def make_folds(y: np.ndarray, seed: int) -> list[dict]:
    """Stratified outer folds, each with stratified inner folds on its training part."""
    outer = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        inner = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=seed * 1000 + k)
        inner_folds = [
            {"train": tr[a].tolist(), "val": tr[b].tolist()}
            for a, b in inner.split(np.zeros(len(tr)), y[tr])
        ]
        folds.append({"train": tr.tolist(), "test": te.tolist(), "inner": inner_folds})
    return folds


def folds_path(dataset: str, seed: int) -> Path:
    return FOLDS_DIR / f"{dataset}_seed{seed}.json"


def load_or_create_folds(dataset: str, seed: int, y: np.ndarray) -> tuple[list[dict], str]:
    """Return folds and the SHA-256 of the fold file (stored with every result).

    `dataset` is the fold-file stem, e.g. "fmnist" or "fmnist_m200" for a
    different subsample size (training-set-size ablation).
    """
    p = folds_path(dataset, seed)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": dataset, "seed": seed, "n": int(len(y)), "folds": make_folds(y, seed)}
        p.write_text(json.dumps(payload))
    text = p.read_text()
    payload = json.loads(text)
    if payload["n"] != len(y):
        raise ValueError(f"{p} was built for n={payload['n']}, got {len(y)}")
    return payload["folds"], hashlib.sha256(text.encode()).hexdigest()
