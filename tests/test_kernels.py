import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qkc.kernels import feature_map_circuit, feature_states, quantum_kernel


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("bandwidth", [0.05, 0.7, 2.0])
def test_numpy_kernel_matches_qiskit_statevector(n, bandwidth):
    rng = np.random.default_rng(n)
    X = rng.standard_normal((4, n))
    K = quantum_kernel(X, None, bandwidth, reps=2)
    svs = [Statevector(feature_map_circuit(x, bandwidth, reps=2)) for x in X]
    K_ref = np.array([[abs(a.inner(b)) ** 2 for b in svs] for a in svs])
    np.testing.assert_allclose(K, K_ref, atol=1e-10)


def test_states_are_normalized_and_kernel_is_valid():
    X = np.random.default_rng(0).standard_normal((20, 6))
    S = feature_states(X, 1.0)
    np.testing.assert_allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-12)
    K = quantum_kernel(X, None, 1.0)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-12


def test_zero_bandwidth_limit_gives_constant_kernel():
    X = np.random.default_rng(1).standard_normal((10, 4))
    np.testing.assert_allclose(quantum_kernel(X, None, 1e-8), 1.0, atol=1e-10)
