PY := .venv/bin/python
QUBITS ?= 2 4 6 8 10 12 14 16

.PHONY: setup test cv-fmnist cv-synthetic sweep ablations noise figures demo synopsis report clean-tex

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

test:
	$(PY) -m pytest -q

# Nested-CV test accuracy for all models on the fixed folds (results/cv/).
cv-fmnist:
	$(PY) experiments/run_cv.py --dataset fmnist --models majority linear logreg rbf quantum --qubits $(QUBITS)

cv-synthetic:
	$(PY) experiments/run_cv.py --dataset synthetic --models majority linear logreg rbf quantum --qubits $(QUBITS)

JOBS ?= 4
CAL ?= $(lastword $(sort $(wildcard results/hardware/calibration_ibm_marrakesh_*.json)))

# Main simulator sweep (T1-T4, T6): exact + finite-shot quantum kernels, RBF baseline.
sweep:
	for ds in fmnist synthetic; do for m in quantum rbf; do \
	  $(PY) experiments/run_sweep.py --dataset $$ds --model $$m --jobs $(JOBS); done; done

# Ablations (H5), Fashion-MNIST, exact kernels only.
ablations:
	$(PY) experiments/run_sweep.py --dataset fmnist --model quantum --reps 1 --tag reps1 --shots inf --qubits 2 4 8 12 16 --jobs $(JOBS)
	$(PY) experiments/run_sweep.py --dataset fmnist --model quantum --reps 3 --tag reps3 --shots inf --qubits 2 4 8 12 16 --jobs $(JOBS)
	$(PY) experiments/run_sweep.py --dataset fmnist --model quantum --product --tag product --shots inf --qubits 2 4 8 12 16 --jobs $(JOBS)
	$(PY) experiments/run_sweep.py --dataset fmnist --model quantum --n-per-class 50 --tag m100 --shots inf --qubits 2 4 8 12 16 --jobs $(JOBS)
	$(PY) experiments/run_sweep.py --dataset fmnist --model quantum --n-per-class 250 --tag m500 --shots inf --qubits 2 4 8 12 --jobs $(JOBS)

# Noise-model simulation and noise-strength sweep (T8, H6) from a saved calibration snapshot.
noise:
	$(PY) experiments/noise_sim.py --calibration $(CAL)

# Every report figure and table from saved results.
figures:
	$(PY) experiments/analyze.py

# Quick end-to-end check for the weekly demo (one seed, small n); writes to results/demo/, never results/cv/.
demo:
	$(PY) -m pytest -q
	$(PY) experiments/run_cv.py --dataset fmnist --models rbf quantum --qubits 2 4 --seeds 0 --outdir results/demo

synopsis:
	cd report/synopsis && latexmk -pdf -interaction=nonstopmode -halt-on-error synopsis.tex

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-tex:
	cd report && latexmk -c main.tex
	cd report/synopsis && latexmk -c synopsis.tex
