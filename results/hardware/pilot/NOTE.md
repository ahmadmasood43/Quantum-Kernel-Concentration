# Pilot run (ibm_marrakesh, job dbq5udoou4as73cukgg0, 2026-11-11)

Purpose: measure the real QPU cost per circuit before sizing the main run
(21 circuits x 1000 shots, n = 2, lambda = 1, m = 6). Charged: 8 s.

Known limitation: these circuits were built before the compute/uncompute
barrier was added to `qkc.hwexp.kernel_circuit`. The transpiler therefore
compiled each diagonal circuit U(x)U^dagger(x) to a bare measurement, so the
pilot's diagonal values (`diag_mean`) reflect readout error only, not gate
noise. Off-diagonal kernel estimates are unaffected. The main run (`hw1`)
uses the corrected circuits.
