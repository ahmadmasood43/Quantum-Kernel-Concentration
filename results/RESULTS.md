# Generated results summary

Produced by `experiments/analyze.py` from saved results; regenerate rather than edit.

- main sweep: {'fmnist': array([10, 12, 14, 16,  2,  4,  6,  8]), 'synthetic': array([10, 12, 14, 16,  2,  4,  6,  8])} quantum; {'fmnist': 8, 'synthetic': 8} RBF
- figure: concentration_variance.pdf
- H1 [fmnist] lambda=1: d ln(var)/dn = -0.452 +/- 0.033 (95% CI, 5 seeds); R^2 exp 0.999 vs power 0.930
- H1 [synthetic] lambda=1: d ln(var)/dn = -0.927 +/- 0.027 (95% CI, 5 seeds); R^2 exp 0.998 vs power 0.893
- figure: accuracy_vs_lambda.pdf
- figure: tuned_accuracy.pdf
- figure: selected_lambda.pdf
- figure: shot_noise_map.pdf (orange contour: true spread equals shot-noise variance)
- table: tuned_accuracy
- paired quantum(exact)-RBF [fmnist] mean diff by n: 2:+0.001, 4:-0.002, 6:+0.001, 8:+0.005, 10:+0.003, 12:-0.005, 14:-0.001, 16:-0.004
- paired quantum(exact)-RBF [synthetic] mean diff by n: 2:-0.000, 4:+0.000, 6:+0.004, 8:+0.005, 10:+0.004, 12:-0.003, 14:+0.001, 16:+0.005
- Falsifier [fmnist] n=16, N=1e4: 12 distinguishable bandwidths (2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1); tuned acc 0.958 vs lambda=1 0.754, Wilcoxon p=6e-06 -> H2 not rejected
- Falsifier [synthetic] n=16, N=1e4: 12 distinguishable bandwidths (2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1); tuned acc 0.829 vs lambda=1 0.519, Wilcoxon p=6e-06 -> H2 not rejected
- Proposal falsifier [fmnist] n=16, lambda* at N=1000: spread/noise at 5,000 shots median 1009 (ok); acc 0.953 vs RBF 0.957 (Wilcoxon worse p=0.33); vs chance max p=1.9e-62 -> not rejected
- Proposal falsifier [fmnist] n=16, lambda* at N=10000: spread/noise at 5,000 shots median 878 (ok); acc 0.958 vs RBF 0.957 (Wilcoxon worse p=0.6); vs chance max p=7.1e-65 -> not rejected
- Proposal falsifier [synthetic] n=16, lambda* at N=1000: spread/noise at 5,000 shots median 75 (ok); acc 0.817 vs RBF 0.825 (Wilcoxon worse p=0.063); vs chance max p=5.9e-27 -> not rejected
- Proposal falsifier [synthetic] n=16, lambda* at N=10000: spread/noise at 5,000 shots median 66 (ok); acc 0.829 vs RBF 0.825 (Wilcoxon worse p=0.73); vs chance max p=8.4e-29 -> not rejected
- table: ablations (5 variants)
- figures: noise_model_variance.pdf, noise_model_diagonal.pdf
- table: resources (14 rows)
- hardware: pilot -> hardware_vs_noise_pilot.pdf
- hardware: hw1 -> hardware_vs_noise_hw1.pdf
