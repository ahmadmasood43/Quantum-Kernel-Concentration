# Quantum-Kernel-Concentration

AI 631 Quantum Machine Learning (Fall 2026), Project #5: **Kernel Concentration and Bandwidth Scaling**.
Muhammad Ahmad Masood, Muhammad Waqas.

> How fast do quantum kernel values concentrate as qubit count grows, and does a bandwidth hyperparameter restore learnability?

- Project proposal: [report/proposal_revised/](report/proposal_revised/) (IEEEtran). Where the study departs from it, the reasons are given in the report's "Deviations from the proposal" subsection.
- Report draft: [report/main.tex](report/main.tex)
- Requirement tracking: [docs/compliance_matrix.md](docs/compliance_matrix.md)
- Comparison with the reference papers: [docs/paper_comparison.md](docs/paper_comparison.md)

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make setup   # creates .venv, installs pinned requirements and the qkc package
make test    # full test suite (about 1 minute)
make demo    # tests + a one-seed run at n = 2, 4 (writes results/demo/, a few seconds)
```

The Fashion-MNIST training files are downloaded on first use and checked against the SHA-256 checksums in [data/README.md](data/README.md).

## Reproducing results

Everything below reads and writes files under `results/`. Seeds, grids, and dataset choices are defined in one place, `src/qkc/config.py`. Every results file has a matching `.meta.json` recording its grids, fold-file SHA-256 hashes, git commit, and package versions.

| Output | Command | Used in report | Notes |
|---|---|---|---|
| `results/sweep/main/` | `make sweep` | yes | Main study: quantum (exact and finite-shot) and RBF, n = 2–16, both datasets, 5 seeds × 5 outer folds. Several hours on a 16-core machine; resumable |
| `results/sweep/{reps1,reps3,product,m100,m500}/` | `make ablations` | yes | Ablations: feature-map layers, no entanglement, training-set size |
| `results/noise/` | `make noise` | yes | Noise-model simulation, n = 2–10, noise scale ×0–2, from the saved ibm_marrakesh calibration snapshot |
| `results/hardware/hw1/`, `results/hardware/pilot/` | see [IBM hardware](#ibm-hardware) | yes | Raw counts, ISA circuits, calibration at run time, matched simulations |
| `report/figures/*.pdf`, `report/tables/*.tex`, `results/tables/*.csv`, `results/RESULTS.md` | `make figures` | yes | Built only from saved results |
| `results/tables/supplementary.json` | `make figures` | yes | Dispersion-test calibration, Week 1 grid-edge rates, shot-limit extrapolation, sweep provenance |
| `results/cv/` | `make cv-fmnist`, `make cv-synthetic` | no | Week 1 snapshot with the narrower Week 1 grids; superseded by `results/sweep/main/` |

`make all` runs the test suite, the sweep, the ablations, the noise simulation, the figures and the report in order. IBM hardware runs are separate on purpose.

**Sweep provenance.** The shards in `results/sweep/main/` record four commits because they were produced across several invocations. The code they ran is identical, apart from a fix to `select_tuned`, which only affects post-processing (listed in `supplementary.json`).

### Figures

All figures come from `experiments/analyze.py` (`make figures`):

| File | Shows |
|---|---|
| `concentration_variance.pdf` | Off-diagonal kernel variance vs. qubit count for several λ (H1) |
| `accuracy_vs_lambda.pdf` | Test accuracy vs. fixed bandwidth for several n |
| `tuned_accuracy.pdf` | Tuned quantum (exact, 10⁴ and 10³ shots) vs. RBF and linear SVM |
| `selected_lambda.pdf` | Median selected λ* vs. n for each shot budget |
| `shot_noise_map.pdf` | Ratio of kernel spread to shot-noise variance over (n, λ) |
| `noise_model_variance.pdf`, `noise_model_diagonal.pdf` | Noise-model spread and measured k(x, x) vs. n |
| `hardware_vs_noise_hw1.pdf` | Hardware vs. noise model vs. ideal shots (excess variance, 95% CIs) |
| `hardware_vs_noise_pilot.pdf` | Same for the pilot run |

Tables: `report/tables/decay_rates.tex` (H1 fits), `tuned_accuracy.tex`, and `resources.tex` (qubits, depth, two-qubit gates, shots).

## IBM hardware

The account's Open plan provides 600 s of QPU time per 28-day period. To keep it safe, submission only happens after a local preflight of the exact circuits.

```bash
.venv/bin/python scripts/save_ibm_account.py      # once; the key is read with a hidden prompt, stored in ~/.qiskit/
.venv/bin/python scripts/check_ibm_access.py      # read-only: lists backends
.venv/bin/python experiments/snapshot_backend.py --backend ibm_marrakesh   # read-only calibration snapshot

P=".venv/bin/python experiments/hardware_run.py"
$P prepare   --backend ibm_marrakesh --tag hw1     # build + transpile + save circuits (no QPU)
$P preflight --tag hw1 --shots 1000                # local checks + projected QPU time (no QPU)
$P submit    --tag hw1 --shots 1000 --confirm      # USES QPU TIME; refuses without a passed preflight
$P fetch     --tag hw1                             # counts, run-time calibration, matched simulations
```

Add `--fake` to any step to dry-run the whole path on a fake backend. Its outputs go to `results/hardware/dryrun_*` (git-ignored) and are labelled FAKE.

## Report and proposal

The report uses IEEEtran. On Debian/Ubuntu:

```bash
sudo apt install texlive-latex-extra texlive-publishers texlive-science latexmk
make proposal   # report/proposal_revised/proposal.pdf
make report     # report/main.pdf
```

## Tests

| File | Checks |
|---|---|
| `tests/test_kernels.py` | Statevector kernel matches Qiskit `Statevector` to 1e-10; kernel validity; product-map factorization |
| `tests/test_leakage.py` | Preprocessing and model selection never see held-out rows; fold structure |
| `tests/test_sweep.py` | Shot sampling, dispersion-test size and power, tuned-λ selection |
| `tests/test_noise.py` | Noise model vs. Aer's device model; diagonal circuits keep their gates; spread statistics |
| `tests/test_data.py` | Dataset checksum verification; synthetic-data defaults |

## Repository layout

| Path | Contents |
|---|---|
| `src/qkc/` | Data, fixed folds, preprocessing, kernels, nested CV, sweep engine, noise models, hardware circuits |
| `experiments/` | Entry points: `run_sweep.py`, `noise_sim.py`, `hardware_run.py`, `snapshot_backend.py`, `analyze.py`, `supplementary.py`, `run_cv.py` (Week 1) |
| `tests/` | Test suite (above) |
| `data/` | Dataset and preprocessing documentation; fixed fold files |
| `results/` | Saved results with metadata |
| `report/` | Report (`main.tex`), figures, tables, bibliography, project proposal |
| `docs/` | Compliance matrix; comparison with the reference papers |
| `scripts/` | IBM account setup and access check |
