# Quantum-Kernel-Concentration

AI 631 Quantum Machine Learning (Fall 2026), Project #5: **Kernel Concentration and Bandwidth Scaling**.
Muhammad Ahmad Masood, Muhammad Waqas.

> How fast do quantum kernel values concentrate as qubit count grows, and does a bandwidth hyperparameter restore learnability?

Project status and requirement tracking: [docs/compliance_matrix.md](docs/compliance_matrix.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make setup   # creates .venv, installs pinned requirements and the qkc package
make test    # kernel cross-check against Qiskit, leakage tests
make demo    # one-seed end-to-end run at n = 2, 4 (writes results/demo/)
```

### IBM Quantum access (needed for the hardware experiments only)

Save your API key once. It is read with a hidden prompt and stored in `~/.qiskit/`, outside this repository:

```bash
.venv/bin/python scripts/save_ibm_account.py
.venv/bin/python scripts/check_ibm_access.py   # read-only: lists backends, submits no jobs
```

### LaTeX

The report uses IEEEtran. On Debian/Ubuntu:

```bash
sudo apt install texlive-latex-extra texlive-publishers texlive-science latexmk
```

Build the synopsis and the report draft:

```bash
make synopsis
make report
```

## Reproducing results

| Output | Command |
|--------|---------|
| `results/cv/fmnist_<model>.csv` (+ `.meta.json`) | `make cv-fmnist` |
| `results/cv/synthetic_<model>.csv` (+ `.meta.json`) | `make cv-synthetic` |

Seeds, grids, and dataset choices are defined in one place, `src/qkc/config.py`. Each results file comes with a `.meta.json` recording the grids, the fold-file SHA-256 hashes, the git commit, and the package versions.

## Repository layout

| Path | Contents |
|------|----------|
| `docs/` | Requirement-to-deliverable compliance matrix |
| `src/qkc/` | Data loading, fixed folds, preprocessing, quantum/RBF kernels, nested CV |
| `experiments/` | Experiment entry points |
| `tests/` | Kernel validation against Qiskit; leakage tests |
| `data/` | Dataset and preprocessing documentation ([data/README.md](data/README.md)); fixed fold files |
| `results/` | Saved results with metadata |
| `report/` | Report draft (`main.tex`), synopsis, bibliography |
| `scripts/` | IBM account setup and access check |

Figure-reproduction commands will be added to the table above as the figures are produced.
