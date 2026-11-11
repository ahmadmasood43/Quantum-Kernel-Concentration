r"""Bandwidth-scaled IQP-type ZZ fidelity kernel and the classical RBF kernel.

Feature map (one feature per qubit, linear-chain entanglement, `reps` layers),
applied to z = lambda * x:

    U(z) = prod_{r=1}^{reps} [ D(z) H^{\otimes n} ],
    D(z) = exp( i * sum_j z_j Z_j  +  i * sum_{j} z_j z_{j+1} Z_j Z_{j+1} ).

D is diagonal in the computational basis, so the statevector is simulated
exactly with a fast Walsh-Hadamard transform and an elementwise phase. Qubit j
corresponds to bit j of the basis index (Qiskit's little-endian convention).

Kernel: k(x, x') = |<psi(x') | psi(x)>|^2.
"""

import hashlib

import numpy as np
from qiskit import QuantumCircuit


def _hadamard_all(states: np.ndarray, n: int) -> np.ndarray:
    """Apply H to every qubit of a batch of statevectors, shape (m, 2**n).

    In-place fast Walsh-Hadamard butterfly on a copy: one pass per qubit,
    each touching the whole batch once, with no per-axis reallocation.
    """
    m, dim = states.shape
    out = states.copy()
    h = 1
    while h < dim:
        v = out.reshape(m, dim // (2 * h), 2, h)
        a = v[:, :, 0, :].copy()
        v[:, :, 0, :] += v[:, :, 1, :]
        np.subtract(a, v[:, :, 1, :], out=v[:, :, 1, :])
        h *= 2
    out /= np.sqrt(float(dim))
    return out


def _spins(n: int) -> np.ndarray:
    """s[j, b] = +1 if bit j of basis index b is 0 else -1 (Z eigenvalue)."""
    b = np.arange(2**n)
    bits = (b[None, :] >> np.arange(n)[:, None]) & 1
    return 1.0 - 2.0 * bits


def feature_states(X: np.ndarray, bandwidth: float, reps: int = 2) -> np.ndarray:
    """Statevectors psi(lambda * x) for each row of X, shape (m, 2**n)."""
    Z = bandwidth * np.asarray(X, dtype=np.float64)
    m, n = Z.shape
    s = _spins(n)
    # phase(b) = sum_j z_j s_j(b) + sum_j z_j z_{j+1} s_j(b) s_{j+1}(b)
    phase = Z @ s
    if n > 1:
        phase += (Z[:, :-1] * Z[:, 1:]) @ (s[:-1] * s[1:])
    diag = np.exp(1j * phase)
    # First layer: H^{(x)n}|0...0> is the uniform superposition, so skip the transform.
    states = diag / np.sqrt(2.0**n)
    for _ in range(reps - 1):
        states = _hadamard_all(states, n)
        states *= diag
    return states


_STATE_CACHE: dict[tuple, np.ndarray] = {}
_STATE_CACHE_MAX = 4


def _cached_states(X: np.ndarray, bandwidth: float, reps: int) -> np.ndarray:
    """feature_states with a small cache keyed on the exact input bytes.

    Nested CV computes K(train, train) and then K(val, train) with the same
    training rows; the cache avoids preparing the training states twice.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    key = (hashlib.blake2b(X.tobytes(), digest_size=16).hexdigest(), X.shape, float(bandwidth), reps)
    S = _STATE_CACHE.get(key)
    if S is None:
        S = feature_states(X, bandwidth, reps)
        if len(_STATE_CACHE) >= _STATE_CACHE_MAX:
            _STATE_CACHE.pop(next(iter(_STATE_CACHE)))
        _STATE_CACHE[key] = S
    return S


def quantum_kernel(XA: np.ndarray, XB: np.ndarray | None, bandwidth: float, reps: int = 2) -> np.ndarray:
    """Exact (noise-free, infinite-shot) fidelity kernel matrix."""
    SA = _cached_states(XA, bandwidth, reps)
    SB = SA if XB is None else _cached_states(XB, bandwidth, reps)
    return np.abs(SA.conj() @ SB.T) ** 2


def feature_map_circuit(x: np.ndarray, bandwidth: float, reps: int = 2) -> QuantumCircuit:
    """The same feature map as a Qiskit circuit (used for cross-checks and hardware).

    exp(i a Z) = RZ(-2a) and exp(i a Z Z) = RZZ(-2a), each up to a global phase.
    """
    z = bandwidth * np.asarray(x, dtype=np.float64)
    n = len(z)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for j in range(n):
            qc.rz(-2.0 * z[j], j)
        for j in range(n - 1):
            qc.rzz(-2.0 * z[j] * z[j + 1], j, j + 1)
    return qc


def rbf_kernel(XA: np.ndarray, XB: np.ndarray | None, gamma: float) -> np.ndarray:
    XB = XA if XB is None else XB
    d2 = (XA**2).sum(1)[:, None] + (XB**2).sum(1)[None, :] - 2.0 * XA @ XB.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def offdiag_stats(K: np.ndarray) -> dict:
    """Mean and variance of the off-diagonal entries of a square Gram matrix."""
    iu = np.triu_indices_from(K, k=1)
    v = K[iu]
    return {"offdiag_mean": float(v.mean()), "offdiag_var": float(v.var(ddof=1))}
