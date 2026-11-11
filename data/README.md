# Data

## Fashion-MNIST (primary dataset)

Source: the dataset authors' repository (Xiao, Rasul, Vollgraf, arXiv:1708.07747),
`https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/`.

The first run downloads the files automatically into `data/raw/` (git-ignored), or you can download them explicitly:

```bash
.venv/bin/python -c "from qkc.data import download_fmnist; download_fmnist()"
```

Files used, with SHA-256 checksums recorded on 2026-11-11:

| File | Bytes | SHA-256 |
|------|-------|---------|
| `train-images-idx3-ubyte.gz` | 26421880 | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | 29515 | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |

Only the official **training** split is used, as a pool to subsample from. The official test split is not used.

### Subsampling (`qkc.data.load_fmnist_pair`)

- Class pair: label 0 (T-shirt/top) vs. label 2 (Pullover), relabelled 0/1.
- For each seed in `qkc.config.SEEDS = (0, 1, 2, 3, 4)`, 150 images per class are drawn without replacement using `numpy.random.default_rng(seed)`, giving 300 class-balanced points per seed.
- Pixels are scaled to [0, 1].

## Synthetic dataset (`qkc.data.make_synthetic`)

Two isotropic Gaussian classes in R^n (n = number of qubits), with means at ±(s/2)·u, where u is a random unit vector and s = 2.0. The Bayes-optimal accuracy is Φ(1) ≈ 0.841 for every n, so any change with n comes from the model rather than the data. There is no PCA step; features go straight to the per-fold standardizer.

## Preprocessing (`qkc.preprocess`)

Each step is fit on the training rows of the current (inner or outer) fold only, then applied to the held-out rows:

1. `StandardScaler` on the input features.
2. `PCA(n_components = n)` (Fashion-MNIST only).
3. `StandardScaler` on each component, so every qubit receives unit-variance input before the bandwidth λ is applied.

`tests/test_leakage.py` checks this: it corrupts the held-out rows and asserts that the fitted transforms and the hyperparameters chosen by model selection do not change.

## Fixed folds (`data/folds/`)

`<dataset>_seed<s>.json` holds 5 stratified outer folds. Each outer fold contains 3 stratified inner folds built on its training rows. All indices are absolute row indices into the seed's subsample. The files are created on first use, committed, and then reused by every model. Each result's `.meta.json` records the SHA-256 of the fold file it used.
