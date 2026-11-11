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

# Bandwidth grids. The quantum and RBF searches use the same grid size and the
# same nested-CV protocol. Selections that land on a grid edge are logged so
# the range can be widened if needed.
GRID_SIZE = 10
LAMBDA_GRID = np.logspace(-2, np.log10(np.pi), GRID_SIZE)
# RBF: gamma = g / n_features, with g on a log grid. After per-feature
# standardization this matches sklearn's gamma="scale" at g = 1.
RBF_G_GRID = np.logspace(-3, 1, GRID_SIZE)
C_GRID = np.array([0.1, 1.0, 10.0, 100.0])

FEATURE_MAP_REPS = 2
