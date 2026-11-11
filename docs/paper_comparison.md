# Comparison with the reference papers

Checked on 2026-11-11 against the arXiv versions of:

- **[SW]** R. Shaydulin and S. M. Wild, *Importance of kernel bandwidth in quantum machine learning*, Phys. Rev. A 106, 042407 (2022); arXiv:2111.05451v4.
- **[TWCH]** S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes, *Exponential concentration in quantum kernel methods*, Nat. Commun. 15, 5200 (2024); arXiv:2208.11060v2.
- Also relevant, and cited in [TWCH] §"Attempts to mitigate": **[SSCPKW]** L. Slattery, R. Shaydulin, S. Chakrabarti, M. Pistoia, S. Khairy, S. M. Wild, *Numerical evidence against advantage with quantum fidelity kernels on classical data*, Phys. Rev. A 107, 062417 (2023). Metadata verified via Crossref; the paper itself has not been read yet.

Our numbers are Fashion-MNIST, exact simulation unless stated, 5 seeds × 5 outer folds, from `results/sweep/main/`, `results/noise/` and `results/hardware/hw1/` (`make figures` regenerates the tables). The paper statements below are paraphrases with their location in the paper. Each one must be re-checked against the paper before it is cited as a specific claim in the report.

## 1. Setup differences that limit quantitative comparison

| | Ours | [SW] | [TWCH] numerics |
|---|---|---|---|
| Embedding | IQP-type, 2 layers, **nearest-neighbour** ZZ, on λx | IQP-type, 2 layers, **all-to-all** ZZ (Eqs. 4–5), plus a Hamiltonian-evolution map | Mainly a hardware-efficient embedding (HEE) |
| Bandwidth | x → λx (ZZ terms scale as λ²) | Same form, "scaling factor" λ (Eq. 5) | Not studied (bandwidth mentioned as a mitigation) |
| Data | Fashion-MNIST T-shirt/top vs Pullover; 300 points per seed; standardize → PCA → per-component standardization, all fit inside each fold | Fashion-MNIST "dresses (class 0) and shirts (class 3)" after Huang et al.'s pipeline; also KMNIST and PLAsTiCC; per-datapoint standardization → PCA; 800 train / 200 test, one split | MNIST, PCA to n features |
| Model selection | Nested CV over (λ, C), 12 × 4 grid, fixed folds | 5-fold CV over C; λ swept | — |
| Qubits | 2–16 (exact); 2–10 noise model; 2–6 hardware | Up to 26 (simulation) | Up to roughly 10–20 depending on the figure |
| Shots | Exact binomial per entry, N = 10²–10⁵, **model selection under the same shots** | 5000 shots, approximated as i.i.d. Gaussian noise whose SD is estimated from a 5×5 submatrix; PSD projection (Fig. 2d) | N = 1000 in several figures |
| Noise | Calibration-based device model of ibm_marrakesh (2026-11-11), scaled ×0–2; real hardware | None | Local Pauli noise on single-qubit gates only (stated limitation, Fig. 8 text) |

Because of these differences we compare **direction and functional form**, not absolute numbers.

## 2. Claim-by-claim

| # | Claim in the paper (location) | Our result | Verdict |
|---|---|---|---|
| 1 | [TWCH] Def. 1 / Eq. 11: concentration can be diagnosed by an off-diagonal variance that is O(1/bⁿ), b > 1; for kernels in [0,1], an exponentially small mean implies this | λ = 1: Var falls as e^(−0.452 ± 0.033 n), i.e. b ≈ 1.57 per qubit; mean kernel as e^(−0.370 n), b′ ≈ 1.45. Exponential fit R² 0.999 vs power law 0.930 | **Consistent**: exponential concentration toward an exponentially small value, in the sense of Def. 1, over n = 2–16 |
| 2 | [SW] Fig. 3d: the kernel value between two points decays exponentially with qubit count | Same as row 1 (mean kernel halves every 1.9 qubits) | **Consistent** |
| 3 | [SW] §Results: λ = 1 (data with unit standard deviation) is a bad choice for all datasets and feature maps they considered | Fixed λ = 1 accuracy: 0.778 (n=2), peaking at 0.817 (n=8), down to 0.757 (n=16); tuned ≈ 0.95 | **Consistent** |
| 4 | [SW] Fig. 3b / text: large λ drives K toward the identity, small λ drives all entries toward 1; best performance in between | Accuracy vs fixed λ rises then falls at every n; small λ: off-diagonal mean → 1 and variance ~10⁻⁹ (λ = 2⁻¹⁰, n = 2) | **Consistent** |
| 5 | [SW] §Results: with optimized bandwidth, performance holds up or improves with qubit count and matches the best classical methods; for Fashion-MNIST it plateaus around 13 qubits | Tuned accuracy 0.935 (n=2) → 0.947–0.959 for n ≥ 4; RBF 0.934–0.958; paired quantum − RBF ≤ 0.005 at every n | **Consistent in direction**. Our plateau starts earlier (n ≈ 6), plausibly because our class pair is easier (linear SVM reaches 0.96) |
| 6 | [SW] Fig. 3a / text: one fixed λ works well across their qubit range; the optimal λ "decreases slightly" with qubit count; they expect it to decrease further at larger n | Median selected λ\*: 2⁻³ (n=2), 2⁻⁴ (n=4–10, 14), 2⁻⁵ (n=12, 16). Accuracy is flat across λ ∈ [2⁻⁶, 2⁻³] at n=16 (0.949–0.952) | **Consistent**: slight decrease and a wide flat optimum (they also note that "a wide range of bandwidth values work well") |
| 7 | [SW] Fig. 2d: with 5000 shots, good accuracy is recovered once an appropriate bandwidth is chosen | Tuned with model selection under shots, n = 16: 0.941 (N=10²), 0.953 (10³), 0.958 (10⁴), 0.953 (10⁵) vs 0.953 exact | **Consistent, and extended**: exact binomial sampling of every entry, selection under the same budget, N down to 100 |
| 8 | [TWCH] Prop. 1: if values concentrate toward an exponentially small value, a poly(n)-shot Loschmidt-echo estimate of the Gram matrix equals the identity with probability → 1 | At λ = 1 and n ≤ 16 the true spread still exceeds shot noise (ratio 646 at N = 10⁴, n = 16). Extrapolating the fitted mean, N·k̄ < 1 near n ≈ 17 / 24 / 30 for N = 10³ / 10⁴ / 10⁵ | **Consistent mechanism; regime not reached** in our range. The n values are an extrapolation, not a measurement |
| 9 | [TWCH] Thm. 3: noise concentrates kernel values toward 1/2ⁿ, with a bound decaying exponentially in the number of noisy layers; their simulations put noise only on single-qubit gates and they expect real hardware (worse entangling gates) to concentrate faster | Noise model (CZ errors included), λ = 1: spread shrinks ×1.2–1.5 for n ≤ 6 and ×7–20 for n = 8–10. Hardware: k(x,x) = 0.74 at n = 6, λ = 1 vs 0.83 predicted by the calibrated model and 0.94 for readout alone | **Consistent**, and the hardware data support their expectation: real-device decay exceeds the calibrated model |
| 10 | [TWCH] §Attempts to mitigate: rescaling avoids *expressivity*-induced concentration, but [SSCPKW] found numerically (≤ 20 qubits) that it is unlikely to give a quantum advantage | Tuned quantum ≈ RBF ≈ linear SVM at every n (differences within fold noise) | **Consistent**. Our "no advantage" result reproduces prior work; it is not new |

## 3. What we add beyond the two papers (to be framed modestly in the report)

1. **Bandwidth selected under a shot budget.** λ\* moves up as shots decrease (median 2⁻² to 2⁻³ at N = 10², down to 2⁻⁵ to 2⁻⁶ at N = 10⁵). This is the trade-off between concentration and shot noise appearing directly in model selection. [SW] fixed λ from exact kernels before adding sampling noise.
2. **Statistical distinguishability is not learnability.** At n = 16, N = 10⁴, a pooled binomial dispersion test detects spread at every λ, including λ = 2⁻¹⁰, where the per-entry signal-to-shot ratio is 0.6 and accuracy is 0.63. The per-entry ratio predicts accuracy; the pooled test does not.
3. **Bandwidth does not protect against noise-induced concentration on hardware.** At λ = 2⁻⁴ the true spread is about 0.014 (SD), while hardware kernel values deviate from exact by 0.10–0.14 on average (n = 4–6). The small-λ kernels that the bandwidth argument relies on are the ones most exposed to device error. [TWCH] Thm. 3 implies that rescaling does not remove noise-induced concentration; our hardware runs show it directly at n ≤ 6.

## 4. Consequences

- **Report:** cite [SSCPKW] for the "no advantage" point and present our quantum ≈ RBF result as agreement with it, not as a new finding.
- **Novelty assessment (matrix item X4):** the "classically approximable after bandwidth tuning" angle is largely covered by [SSCPKW]. The remaining candidates are items 1 and 3 above, especially the interaction between bandwidth, shots, and hardware noise. A literature search on those specific points is still needed before claiming novelty.
