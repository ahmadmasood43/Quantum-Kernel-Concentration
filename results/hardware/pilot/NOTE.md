# Pilot run (ibm_marrakesh, job dbq5udoou4as73cukgg0, 2026-11-11)

Purpose: measure the real QPU cost per circuit before sizing the main run
(21 circuits x 1000 shots, n = 2, lambda = 1, m = 6). Charged: 8 s.

Known limitation: these circuits were built before the compute/uncompute
barrier was added to `qkc.hwexp.kernel_circuit`. The transpiler therefore
compiled each diagonal circuit U(x)U^dagger(x) to a bare measurement, so the
pilot's diagonal values (`diag_mean`) reflect readout error only, not gate
noise. Off-diagonal kernel estimates are unaffected. The main run (`hw1`)
uses the corrected circuits.

The pilot also predates two later additions: it has no `preflight.json` (the
preflight step was introduced for `hw1`), and its `isa_stats.csv` was written
before barriers were excluded from two-qubit gate counts, so its `n_2q` and
`depth_2q` columns overcount. `experiments/analyze.py` recounts gates from the
saved `circuits.qpy` when building the resource table.
