# Quantum-Kernel-Concentration

AI 631 Quantum Machine Learning (Fall 2026), Project #5: **Kernel Concentration and Bandwidth Scaling**.
Muhammad Ahmad Masood, Muhammad Waqas.

> How fast do quantum kernel values concentrate as qubit count grows, and does a bandwidth hyperparameter restore learnability?

Project status and requirement tracking: [docs/compliance_matrix.md](docs/compliance_matrix.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
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

Build the synopsis:

```bash
cd report/synopsis && latexmk -pdf synopsis.tex
```

## Repository layout

| Path | Contents |
|------|----------|
| `docs/` | Requirement-to-deliverable compliance matrix |
| `report/synopsis/` | Initial synopsis (IEEEtran) and verified bibliography |
| `scripts/` | IBM account setup and access check |

Data pipeline, experiments, results, and figure-reproduction instructions will be added as the corresponding milestones land.
