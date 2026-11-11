"""Kernel-estimation circuits for noisy simulation and IBM hardware, and the
statistics used to compare them.

A kernel entry k(x_i, x_j) is estimated as the frequency of the all-zero
outcome of U(lambda x_i) followed by U^dagger(lambda x_j) (compute-uncompute).
Diagonal circuits (i = j) are noiselessly the identity, so their all-zero
frequency directly measures how much device noise pulls k(x, x) below 1.
"""

import numpy as np
from qiskit import QuantumCircuit

from qkc.data import load_fmnist_pair
from qkc.folds import load_or_create_folds
from qkc.kernels import feature_map_circuit
from qkc.preprocess import fit_transform_split


def select_points(n: int, m: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """m class-balanced Fashion-MNIST points in the n-dimensional feature space.

    Taken from the outer-fold-0 training rows of the given seed, with the
    preprocessing fit on those rows (no test rows involved).
    """
    X, y = load_fmnist_pair(seed)
    folds, _ = load_or_create_folds("fmnist", seed, y)
    tr = np.asarray(folds[0]["train"])
    Xtr, _, _ = fit_transform_split(X, tr, tr[:1], n, seed)
    ytr = y[tr]
    rng = np.random.default_rng(10_000 + seed)
    picks = []
    for c in (0, 1):
        idx = np.flatnonzero(ytr == c)
        picks.append(rng.choice(idx, size=m // 2, replace=False))
    sel = np.sort(np.concatenate(picks))
    return Xtr[sel], ytr[sel]


def pair_list(m: int, include_diagonal: bool = True) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if include_diagonal:
        pairs += [(i, i) for i in range(m)]
    return pairs


def kernel_circuit(xi, xj, bandwidth: float, reps: int = 2) -> QuantumCircuit:
    """Compute-uncompute circuit for k(x_i, x_j).

    The barrier stops the transpiler from cancelling U(x) against U^dagger(x):
    without it, every diagonal circuit compiles to a bare measurement and
    probes only readout error. With it, diagonal and off-diagonal circuits
    have the same gate structure, so diagonals measure the full device noise.
    """
    qc = feature_map_circuit(xi, bandwidth, reps)
    qc.barrier()
    qc.compose(feature_map_circuit(xj, bandwidth, reps).inverse(), inplace=True)
    qc.measure_all()
    return qc


def isa_stats(isa: QuantumCircuit) -> dict:
    """Depth and gate counts of a transpiled (ISA) circuit, and the physical qubits it uses."""
    ops = isa.count_ops()
    layout = isa.layout.final_index_layout() if isa.layout is not None else list(range(isa.num_qubits))

    def is_2q_gate(inst):  # barriers and measurements are not gates
        return inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure")

    two_q = sum(1 for inst in isa.data if is_2q_gate(inst))
    return {
        "depth": isa.depth(lambda inst: inst.operation.name != "barrier"),
        "depth_2q": isa.depth(is_2q_gate),
        "n_2q": two_q,
        "n_cz": int(ops.get("cz", 0)),
        "n_sx": int(ops.get("sx", 0)),
        "n_x": int(ops.get("x", 0)),
        "n_rz": int(ops.get("rz", 0)),
        "physical_qubits": [int(q) for q in layout],
    }


def p_zero(counts: dict, n: int) -> float:
    shots = sum(counts.values())
    return counts.get("0" * n, 0) / shots


def spread_stats(k_hat: np.ndarray, shots: int) -> dict:
    """Off-diagonal spread of measured kernel values.

    raw_var:      sample variance of the measured values (includes shot noise)
    shot_var:     estimated mean shot-noise variance, mean(p(1-p))/(N-1), which is
                  unbiased for mean(k(1-k))/N
    excess_var:   raw_var - shot_var, an estimate of the spread of the
                  underlying (noisy-device) kernel values
    """
    v = np.asarray(k_hat, dtype=float)
    raw = float(v.var(ddof=1))
    shot = float(np.mean(v * (1 - v)) / (shots - 1))
    return {"mean": float(v.mean()), "raw_var": raw, "shot_var": shot, "excess_var": raw - shot}


def bootstrap_excess_var(K_hat: np.ndarray, shots: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI for excess_var by resampling *points* (pairs share points, so
    resampling pairs independently would understate the uncertainty)."""
    m = K_hat.shape[0]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(m, size=m, replace=True)
        sub = K_hat[np.ix_(idx, idx)]
        a, b = np.triu_indices(m, 1)
        keep = idx[a] != idx[b]  # drop self-pairs created by duplicate draws
        v = sub[a[keep], b[keep]]
        if v.size > 1:
            out.append(spread_stats(v, shots)["excess_var"])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)
