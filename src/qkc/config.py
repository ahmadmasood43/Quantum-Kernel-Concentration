"""Single source of truth for seeds, grids, and dataset choices."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FOLDS_DIR = ROOT / "data" / "folds"
RESULTS = ROOT / "results"

# Subsampling seeds: each seed draws a different class-balanced subsample and
# its own fixed folds. Results are reported as mean +/- std over these seeds.
SEEDS = (0, 1, 2, 3, 4)

# Fashion-MNIST class pair (labels per the dataset paper): 0 = T-shirt/top, 2 = Pullover.
FMNIST_CLASSES = (0, 2)
N_PER_CLASS = 150  # 300 points per seed in total

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

QUBIT_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)

# Week 1 grids (kept so experiments/run_cv.py reproduces the committed Week 1
# results exactly). 16-36% of Week 1 selections hit the lower edge.
GRID_SIZE_V1 = 10
LAMBDA_GRID_V1 = np.logspace(-2, np.log10(np.pi), GRID_SIZE_V1)
RBF_G_GRID_V1 = np.logspace(-3, 1, GRID_SIZE_V1)

# Main grids (Week 2 onward). lambda = 2^k for k = -10..1, which includes
# lambda = 1 (the unscaled kernel used in H1). RBF gamma = g / n with g = lambda^2:
# for small lambda the fidelity kernel is 1 - O(lambda^2 ||x - x'||^2), so this
# maps the two searches onto each other point by point. Same size, same protocol.
LAMBDA_GRID = 2.0 ** np.arange(-10, 2)
RBF_G_GRID = LAMBDA_GRID**2
GRID_SIZE = len(LAMBDA_GRID)

# Shot budgets for finite-shot kernel estimation (np.inf = exact kernel).
SHOTS = (np.inf, 1e2, 1e3, 1e4, 1e5)

C_GRID = np.array([0.1, 1.0, 10.0, 100.0])

FEATURE_MAP_REPS = 2
