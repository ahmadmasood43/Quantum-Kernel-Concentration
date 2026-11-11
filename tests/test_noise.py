import numpy as np
import pytest
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qkc.hwexp import bootstrap_excess_var, kernel_circuit, p_zero, spread_stats
from qkc.kernels import quantum_kernel
from qkc.noise import build_noise_model, calibration_summary


@pytest.fixture(scope="module")
def setup():
    be = FakeMarrakesh()
    cal = calibration_summary(be)
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0)
    rng = np.random.default_rng(0)
    circs, exact = [], []
    for n in (2, 3):
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        circs.append(pm.run(kernel_circuit(a, b, 0.4)))
        exact.append(quantum_kernel(a[None], b[None], 0.4)[0, 0])
    return be, cal, circs, exact


def _pzero(nm, circs, shots=20_000):
    sim = AerSimulator(noise_model=nm, seed_simulator=1)
    res = sim.run(circs, shots=shots).result()
    return np.array([p_zero(res.get_counts(i), n) for i, n in enumerate((2, 3))])


def test_scale_zero_is_ideal(setup):
    _, cal, circs, exact = setup
    np.testing.assert_allclose(_pzero(build_noise_model(cal, 0.0), circs), exact, atol=0.01)


def test_builder_agrees_with_aer_device_model(setup):
    be, cal, circs, _ = setup
    ours = _pzero(build_noise_model(cal, 1.0), circs)
    aer = _pzero(NoiseModel.from_backend(be), circs)
    np.testing.assert_allclose(ours, aer, atol=0.02)


def test_more_noise_lowers_fidelity_of_identity_circuit(setup):
    _, cal, _, _ = setup
    be = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, seed_transpiler=0)
    x = np.random.default_rng(5).standard_normal(4)
    isa = pm.run(kernel_circuit(x, x, 0.7))  # identity in the absence of noise
    vals = []
    for s in (0.0, 1.0, 3.0):
        sim = AerSimulator(noise_model=build_noise_model(cal, s), seed_simulator=2)
        vals.append(p_zero(sim.run(isa, shots=20_000).result().get_counts(), 4))
    assert vals[0] == 1.0 and vals[0] > vals[1] > vals[2]


def test_spread_stats_remove_shot_noise_on_average():
    rng = np.random.default_rng(0)
    k = np.full(5000, 0.4)
    v = rng.binomial(1000, k) / 1000
    st = spread_stats(v, 1000)
    assert abs(st["excess_var"]) < 3e-5  # true spread is zero
    assert abs(st["shot_var"] / (0.4 * 0.6 / 1000) - 1) < 0.02


def test_bootstrap_ci_contains_true_spread():
    rng = np.random.default_rng(1)
    m, N = 12, 4000
    z = rng.uniform(0.2, 0.8, m)
    K = np.clip(np.add.outer(z, z) / 2, 0, 1)
    Kh = rng.binomial(N, K) / N
    Kh = np.triu(Kh, 1) + np.triu(Kh, 1).T
    true = K[np.triu_indices(m, 1)].var(ddof=1)
    lo, hi = bootstrap_excess_var(Kh, N, n_boot=500)
    assert lo < true < hi


def test_diagonal_circuit_is_not_compiled_away_and_barriers_are_not_counted():
    from qkc.hwexp import isa_stats

    be = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0)
    x, x2 = np.random.default_rng(4).standard_normal((2, 4))
    diag = isa_stats(pm.run(kernel_circuit(x, x, 0.5)))
    off = isa_stats(pm.run(kernel_circuit(x, x2, 0.5)))
    assert diag["n_cz"] > 0 and diag["n_2q"] == diag["n_cz"]
    assert abs(diag["n_cz"] - off["n_cz"]) <= 4  # same structure up to routing differences
    two = isa_stats(pm.run(kernel_circuit(x[:2], x[:2], 0.5)))
    assert two["n_2q"] == two["n_cz"]  # the 2-qubit barrier is not counted as a gate
