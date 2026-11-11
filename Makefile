PY := .venv/bin/python
QUBITS ?= 2 4 6 8 10 12 14 16

.PHONY: setup test cv-fmnist cv-synthetic demo synopsis report clean-tex

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
