"""Build every figure and table from saved results (no experiments are run here).

    .venv/bin/python experiments/analyze.py

Inputs (whatever exists is used; missing parts are skipped with a notice):
    results/sweep/main/*.parquet          main simulator sweep
    results/sweep/{reps1,reps3,product,m100,m500}/*.parquet   ablations
    results/cv/*_{linear,logreg,majority}.csv                 bandwidth-free baselines
    results/noise/summary.csv                                 noise-model simulation
    results/hardware/<tag>/summary.csv (not dryrun_*)         IBM hardware
Outputs:
    report/figures/*.pdf, results/tables/*.csv, report/tables/*.tex, results/RESULTS.md
"""

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from qkc.config import RESULTS, ROOT
from qkc.sweep import select_tuned

FIG = ROOT / "report" / "figures"
TEX = ROOT / "report" / "tables"
TAB = RESULTS / "tables"
for d in (FIG, TEX, TAB):
    d.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill): categorical slots 1-3 and a blue sequential ramp.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
COL_W, FULL_W = 3.45, 7.0
DS_LABEL = {"fmnist": "Fashion-MNIST", "synthetic": "Synthetic"}
LOG = []


def note(msg):
    print(msg)
    LOG.append(msg)


def lam_label(h):
    k = int(round(np.log2(h)))
    return r"$\lambda=1$" if k == 0 else rf"$\lambda=2^{{{k}}}$"


def load_sweep(tag):
    files = sorted(glob.glob(str(RESULTS / "sweep" / tag / "*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def seed_ci(df, value, by):
    """Average folds within each seed, then mean and 95% t-interval across seeds."""
    per_seed = df.groupby(by + ["seed"])[value].mean().reset_index()
    g = per_seed.groupby(by)[value]
    out = g.agg(["mean", "std", "count"]).reset_index()
    t = stats.t.ppf(0.975, np.maximum(out["count"] - 1, 1))
    out["ci"] = t * out["std"] / np.sqrt(out["count"])
    return out


def write_tex(df, name, caption, label, fmt):
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt.get(c, str)(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / f"{name}.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main sweep
def fig_concentration(q):
    ex = q[~np.isfinite(q.shots)]
    lams = [2.0**k for k in (-8, -6, -4, -2, 0, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), sharey=True)
    for ax, ds in zip(axes, ["fmnist", "synthetic"]):
        d = ex[ex.dataset == ds]
        if d.empty:
            continue
        # variances span many decades: average and form CIs in log space (geometric mean over seeds)
        d = d.assign(log_var=np.log(d.offdiag_var.clip(lower=1e-300)))
        s = seed_ci(d, "log_var", ["h", "n"])
        ends = []
        for c, lam in zip(SEQ, lams):
            r = s[np.isclose(s.h, lam)].sort_values("n")
            if r.empty:
                continue
            gm = np.exp(r["mean"])
            ax.errorbar(r.n, gm, yerr=[gm - np.exp(r["mean"] - r.ci), np.exp(r["mean"] + r.ci) - gm],
                        color=c, marker="o", capsize=0, elinewidth=0.8)
            ends.append([np.log10(gm.iloc[-1]), r.n.iloc[-1], lam])
        ax.set_yscale("log")
        # end labels, nudged apart so they never overlap (min gap = 6% of the axis span in decades)
        lo, hi = np.log10(ax.get_ylim())
        gap = 0.06 * (hi - lo)
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for y10, xn, lam in ends:
            ax.annotate(lam_label(lam), (xn, 10**y10), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK2, annotation_clip=False)
        ax.set_yscale("log")
        ax.set_xlabel("qubits $n$")
        ax.set_title(DS_LABEL[ds], loc="left", color=INK)
        ax.set_xticks(sorted(d.n.unique()))
        ax.set_xlim(right=19.5)
    axes[0].set_ylabel(r"off-diagonal variance $\sigma^2_\lambda(n)$")
    fig.savefig(FIG / "concentration_variance.pdf")
    plt.close(fig)
    note("figure: concentration_variance.pdf")


def table_decay(q):
    """H1: per-seed slope of log(variance) vs n; exponential vs power-law fit quality."""
    ex = q[~np.isfinite(q.shots)]
    rows = []
    for ds in ["fmnist", "synthetic"]:
        d = ex[ex.dataset == ds]
        if d.empty:
            continue
        per_seed = d.groupby(["h", "seed", "n"])["offdiag_var"].mean().reset_index()
        for h, g in per_seed.groupby("h"):
            slopes = [stats.linregress(gs.n, np.log(gs.offdiag_var)).slope for _, gs in g.groupby("seed")]
            m = g.groupby("n")["offdiag_var"].mean()
            r2_exp = stats.linregress(m.index, np.log(m.values)).rvalue ** 2
            r2_pow = stats.linregress(np.log(m.index), np.log(m.values)).rvalue ** 2
            sl = np.array(slopes)
            ci = stats.t.ppf(0.975, len(sl) - 1) * sl.std(ddof=1) / np.sqrt(len(sl))
            rows.append({"dataset": ds, "lambda": h, "slope_per_qubit": sl.mean(), "slope_ci95": ci, "n_seeds": len(sl),
                         "r2_exponential": r2_exp, "r2_powerlaw": r2_pow,
                         "var_n2": m.get(2, np.nan), "var_n16": m.get(16, np.nan)})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "decay_rates.csv", index=False)
    sub = t[np.isclose(t["lambda"], 1.0)]
    for _, r in sub.iterrows():
        note(f"H1 [{r.dataset}] lambda=1: d ln(var)/dn = {r.slope_per_qubit:.3f} +/- {r.slope_ci95:.3f} (95% CI, "
             f"{int(r.n_seeds)} seeds); R^2 exp {r.r2_exponential:.3f} vs power {r.r2_powerlaw:.3f}")
    keep = t[t["lambda"].isin([2.0**k for k in (-8, -6, -4, -2, 0, 1)])].copy()
    keep["lambda"] = keep["lambda"].map(lambda h: f"$2^{{{int(np.log2(h))}}}$")
    keep["dataset"] = keep["dataset"].map(DS_LABEL)
    keep["slope"] = [f"${a:.3f}\\pm{b:.3f}$" for a, b in zip(keep.slope_per_qubit, keep.slope_ci95)]
    keep = keep[["dataset", "lambda", "slope", "r2_exponential", "r2_powerlaw", "var_n16"]]
    keep.columns = ["Data", r"$\lambda$", r"$d\ln\sigma^2/dn$", r"$R^2_{\exp}$", r"$R^2_{\mathrm{pow}}$", r"$\sigma^2(16)$"]
    write_tex(keep, "decay_rates", r"Decay of the off-diagonal kernel variance with qubit count (exact kernels). "
              r"Slope of $\ln\sigma^2_\lambda$ vs.\ $n$ with 95\% CI over 5 seeds, and fit quality of exponential "
              r"($\ln\sigma^2$ vs.\ $n$) and power-law ($\ln\sigma^2$ vs.\ $\ln n$) models.", "tab:decay",
              {r"$R^2_{\exp}$": lambda v: f"{v:.3f}", r"$R^2_{\mathrm{pow}}$": lambda v: f"{v:.3f}",
               r"$\sigma^2(16)$": lambda v: f"{v:.1e}"})


def fig_accuracy_vs_lambda(q):
    ex = q[~np.isfinite(q.shots)]
    ns = [2, 4, 8, 12, 16]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True)
    for ax, ds in zip(axes, ["fmnist", "synthetic"]):
        d = ex[ex.dataset == ds]
        if d.empty:
            continue
        s = seed_ci(d, "test_acc", ["n", "h"])
        for c, n in zip(SEQ[1:], ns):
            r = s[s.n == n].sort_values("h")
            if r.empty:
                continue
            ax.plot(np.log2(r.h), r["mean"], color=c, marker="o", label=f"$n={n}$")
        ax.set_xlabel(r"$\log_2\lambda$ (fixed bandwidth)")
        ax.set_title(DS_LABEL[ds], loc="left", color=INK)
    axes[0].set_ylabel("test accuracy")
    axes[1].legend(loc="lower left", ncol=1)
    fig.savefig(FIG / "accuracy_vs_lambda.pdf")
    plt.close(fig)
    note("figure: accuracy_vs_lambda.pdf")


def tuned_tables(q, r):
    tq = select_tuned(q, ["dataset", "n", "seed", "outer_fold", "shots"])
    tr = select_tuned(r, ["dataset", "n", "seed", "outer_fold"]) if r is not None else None
    return tq, tr


def fig_tuned_accuracy(tq, tr, base):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=False)
    for ax, ds in zip(axes, ["fmnist", "synthetic"]):
        series = []
        for shots, ls, lab in ((np.inf, "-", "quantum, exact"), (1e4, "--", r"quantum, $N=10^4$"),
                               (1e3, ":", r"quantum, $N=10^3$")):
            d = tq[(tq.dataset == ds) & (tq.shots == shots)]
            if not d.empty:
                series.append((seed_ci(d, "test_acc", ["n"]), BLUE, ls, lab))
        if tr is not None and not tr[tr.dataset == ds].empty:
            series.append((seed_ci(tr[tr.dataset == ds], "test_acc", ["n"]), ORANGE, "-", "RBF"))
        if base is not None and not base[(base.dataset == ds) & (base.model == "linear")].empty:
            series.append((seed_ci(base[(base.dataset == ds) & (base.model == "linear")], "test_acc", ["n"]),
                           AQUA, "-", "linear"))
        for s, c, ls, lab in series:
            s = s.sort_values("n")
            ax.plot(s.n, s["mean"], color=c, ls=ls, marker="o", label=lab)
            ax.fill_between(s.n, (s["mean"] - s.ci).clip(upper=1), (s["mean"] + s.ci).clip(upper=1),
                            color=c, alpha=0.10, lw=0)
        if ds == "synthetic":
            ax.axhline(stats.norm.cdf(1.0), color=INK2, lw=0.8, ls="-.")
            ax.annotate("Bayes optimal", (1.0, stats.norm.cdf(1.0)), xycoords=("axes fraction", "data"),
                        xytext=(0, -9), textcoords="offset points", ha="right", fontsize=6.5, color=INK2)
        ax.set_xlabel("qubits $n$")
        ax.set_title(DS_LABEL[ds], loc="left", color=INK)
        ax.set_ylim(top=min(ax.get_ylim()[1], 1.0))
    axes[0].set_ylabel("test accuracy (tuned)")
    axes[0].legend(loc="lower right")
    fig.savefig(FIG / "tuned_accuracy.pdf")
    plt.close(fig)
    note("figure: tuned_accuracy.pdf")


def fig_selected_lambda(tq):
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    d = tq[tq.dataset == "fmnist"]
    for shots, ls, lab in ((np.inf, "-", "exact"), (1e4, "--", r"$N=10^4$"), (1e3, ":", r"$N=10^3$"),
                           (1e2, "-.", r"$N=10^2$")):
        g = d[d.shots == shots].groupby("n")["h"]
        if g.ngroups == 0:
            continue
        med = np.log2(g.median())
        ax.plot(med.index, med.values, color=BLUE, ls=ls, marker="o", label=lab)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"median selected $\log_2\lambda^*$")
    ax.set_title("Fashion-MNIST", loc="left", color=INK)
    ax.legend()
    fig.savefig(FIG / "selected_lambda.pdf")
    plt.close(fig)
    note("figure: selected_lambda.pdf")


def fig_shot_map(q):
    d = q[(q.dataset == "fmnist") & q.shots.isin([1e3, 1e4])]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fe"] + SEQ)
    for ax, N in zip(axes, [1e3, 1e4]):
        g = d[d.shots == N].groupby(["n", "h"])["signal_to_shot"].median().unstack("n")
        Z = np.clip(g.values, 1e-3, 1e4)
        im = ax.pcolormesh(np.arange(Z.shape[1] + 1), np.arange(Z.shape[0] + 1), Z, cmap=cmap,
                           norm=LogNorm(1e-3, 1e4), shading="flat")
        ax.contour(np.arange(Z.shape[1]) + 0.5, np.arange(Z.shape[0]) + 0.5, np.log10(Z), levels=[0],
                   colors=[ORANGE], linewidths=1.6)
        ax.set_xticks(np.arange(Z.shape[1]) + 0.5, [str(n) for n in g.columns])
        ax.set_yticks(np.arange(Z.shape[0]) + 0.5, [f"{int(np.log2(h))}" for h in g.index])
        ax.set_xlabel("qubits $n$")
        ax.set_title(rf"$N=10^{{{int(np.log10(N))}}}$ shots", loc="left", color=INK)
        ax.grid(False)
    axes[0].set_ylabel(r"$\log_2\lambda$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(r"$\sigma^2_\lambda / \overline{k(1-k)}/N$ (median)")
    fig.savefig(FIG / "shot_noise_map.pdf")
    plt.close(fig)
    note("figure: shot_noise_map.pdf (orange contour: true spread equals shot-noise variance)")


def falsifier(q, tq):
    """Operationalization (fixed before looking at n = 16 results):
    (a) at n = 16, N = 1e4, a bandwidth is 'distinguishable' if the dispersion
        test rejects (p < 0.05) in more than half of the 25 outer folds;
    (b) tuned-lambda vs lambda = 1 test accuracy at n = 16, N = 1e4: one-sided
        Wilcoxon signed-rank test over the 25 paired outer folds, alpha = 0.05.
    H2 is rejected if no bandwidth is distinguishable in (a) OR (b) is not significant."""
    rows = []
    for ds in ["fmnist", "synthetic"]:
        d = q[(q.dataset == ds) & (q.n == 16) & (q.shots == 1e4)]
        if d.empty:
            continue
        per_h = d.groupby("h").agg(reject_frac=("disp_p", lambda p: float(np.mean(p < 0.05))),
                                   s2s_median=("signal_to_shot", "median"), acc=("test_acc", "mean")).reset_index()
        per_h.insert(0, "dataset", ds)
        per_h.to_csv(TAB / f"falsifier_n16_N1e4_{ds}.csv", index=False)
        dist = per_h[per_h.reject_frac > 0.5]
        a_ok = not dist.empty
        tuned = tq[(tq.dataset == ds) & (tq.n == 16) & (tq.shots == 1e4)].set_index(["seed", "outer_fold"]).test_acc
        lam1 = d[np.isclose(d.h, 1.0)].set_index(["seed", "outer_fold"]).test_acc
        j = pd.concat([tuned.rename("tuned"), lam1.rename("lam1")], axis=1).dropna()
        diff = j.tuned - j.lam1
        p = stats.wilcoxon(diff, alternative="greater").pvalue if (diff != 0).any() else 1.0
        b_ok = p < 0.05
        rows.append({"dataset": ds, "n_distinguishable_lambdas": len(dist),
                     "distinguishable_lambdas": ",".join(f"2^{int(np.log2(h))}" for h in dist.h),
                     "acc_tuned": j.tuned.mean(), "acc_lambda1": j.lam1.mean(), "wilcoxon_p": p,
                     "H2_rejected": (not a_ok) or (not b_ok)})
        note(f"Falsifier [{ds}] n=16, N=1e4: {len(dist)} distinguishable bandwidths "
             f"({rows[-1]['distinguishable_lambdas'] or 'none'}); tuned acc {j.tuned.mean():.3f} vs lambda=1 "
             f"{j.lam1.mean():.3f}, Wilcoxon p={p:.2g} -> H2 {'REJECTED' if rows[-1]['H2_rejected'] else 'not rejected'}")
    if rows:
        pd.DataFrame(rows).to_csv(TAB / "falsifier.csv", index=False)


def table_tuned(tq, tr, base):
    rows = []
    for ds in ["fmnist", "synthetic"]:
        for lab, d in (("quantum (exact)", tq[(tq.dataset == ds) & ~np.isfinite(tq.shots)]),
                       ("quantum ($N=10^4$)", tq[(tq.dataset == ds) & (tq.shots == 1e4)]),
                       ("quantum ($N=10^3$)", tq[(tq.dataset == ds) & (tq.shots == 1e3)]),
                       ("RBF", tr[tr.dataset == ds] if tr is not None else pd.DataFrame()),
                       ("linear SVM", base[(base.dataset == ds) & (base.model == "linear")] if base is not None else pd.DataFrame()),
                       ("logistic reg.", base[(base.dataset == ds) & (base.model == "logreg")] if base is not None else pd.DataFrame())):
            if d.empty:
                continue
            s = seed_ci(d, "test_acc", ["n"]).set_index("n")
            row = {"Data": DS_LABEL[ds], "Model": lab}
            for n in (2, 4, 8, 12, 16):
                row[f"$n={n}$"] = f"{s.loc[n, 'mean']:.3f}" if n in s.index else "--"
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "tuned_accuracy.csv", index=False)
    write_tex(t, "tuned_accuracy", "Test accuracy with tuned bandwidth (mean over 5 seeds $\\times$ 5 outer folds; "
              "nested CV on identical fixed folds).", "tab:tuned", {})
    note("table: tuned_accuracy")
    # paired quantum - RBF
    if tr is not None:
        for ds in ["fmnist", "synthetic"]:
            a = tq[(tq.dataset == ds) & ~np.isfinite(tq.shots)].set_index(["n", "seed", "outer_fold"]).test_acc
            b = tr[tr.dataset == ds].set_index(["n", "seed", "outer_fold"]).test_acc
            dd = (a - b).dropna().groupby(level="n").agg(["mean", "std"])
            if not dd.empty:
                note(f"paired quantum(exact)-RBF [{ds}] mean diff by n: " +
                     ", ".join(f"{n}:{v:+.3f}" for n, v in dd["mean"].items()))


# --------------------------------------------------------------------------- ablations
def table_ablations(q_main):
    rows = []
    variants = {"main (reps=2, ZZ, m=300)": q_main}
    for tag, lab in (("reps1", "reps=1"), ("reps3", "reps=3"), ("product", "product map (no ZZ)"),
                     ("m100", "m=100 points"), ("m500", "m=500 points")):
        d = load_sweep(tag)
        if d is not None:
            variants[lab] = d
    if len(variants) == 1:
        note("ablations: none found yet")
        return
    for lab, d in variants.items():
        d = d[(d.dataset == "fmnist") & ~np.isfinite(d.shots)]
        if d.empty:
            continue
        tuned = select_tuned(d, ["n", "seed", "outer_fold"])
        acc = seed_ci(tuned, "test_acc", ["n"]).set_index("n")["mean"]
        var1 = d[np.isclose(d.h, 1.0)].groupby("n").offdiag_var.mean()
        lam = tuned.groupby("n").h.median()
        for n in sorted(acc.index):
            rows.append({"variant": lab, "n": n, "acc_tuned": acc[n], "median_lambda_star": lam[n],
                         "var_lambda1": var1.get(n, np.nan)})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "ablations.csv", index=False)
    note(f"table: ablations ({len(variants) - 1} variants)")


# --------------------------------------------------------------------------- noise model
def fig_noise():
    p = RESULTS / "noise" / "summary.csv"
    if not p.exists():
        note("noise model: no results yet")
        return
    s = pd.read_csv(p)
    lams = sorted(s.lam.unique())
    fig, axes = plt.subplots(1, len(lams), figsize=(FULL_W, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    styles = {0.0: ("-", "ideal (shots only)"), 0.5: ("--", r"noise $\times0.5$"), 1.0: ("-", r"noise $\times1$"),
              2.0: (":", r"noise $\times2$")}
    for ax, lam in zip(axes, lams):
        d = s[s.lam == lam]
        ex = d[d.scale == d.scale.min()].sort_values("n")
        ax.plot(ex.n, ex.exact_var, color=INK2, ls="-", lw=1, marker=None, label="exact")
        for sc, g in d.groupby("scale"):
            g = g.sort_values("n")
            ls, lab = styles.get(sc, ("-", f"x{sc}"))
            color = BLUE if sc > 0 else AQUA
            ax.errorbar(g.n, g.meas_excess_var.clip(lower=1e-7),
                        yerr=[(g.meas_excess_var - g.excess_var_lo).clip(lower=0), (g.excess_var_hi - g.meas_excess_var).clip(lower=0)],
                        color=color, ls=ls, marker="o", elinewidth=0.6, label=lab)
        ax.set_yscale("log")
        ax.set_title(lam_label(lam), loc="left", color=INK)
        ax.set_xlabel("qubits $n$")
    axes[0].set_ylabel("off-diagonal spread (excess var.)")
    axes[-1].legend(loc="lower left", fontsize=6)
    fig.savefig(FIG / "noise_model_variance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for sc, g in s.groupby("scale"):
        g = g.groupby("n").diag_mean.mean()
        ls, lab = styles.get(sc, ("-", f"x{sc}"))
        ax.plot(g.index, g.values, color=BLUE if sc > 0 else AQUA, ls=ls, marker="o", label=lab)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"measured $k(x,x)$ (ideal: 1)")
    ax.legend()
    fig.savefig(FIG / "noise_model_diagonal.pdf")
    plt.close(fig)
    s.to_csv(TAB / "noise_model_summary.csv", index=False)
    note("figures: noise_model_variance.pdf, noise_model_diagonal.pdf")


# --------------------------------------------------------------------------- hardware
def hardware():
    dirs = [Path(d) for d in glob.glob(str(RESULTS / "hardware" / "*")) if Path(d).is_dir()
            and not Path(d).name.startswith("dryrun_") and (Path(d) / "summary.csv").exists()]
    if not dirs:
        note("hardware: no completed hardware runs yet")
        return
    for d in dirs:
        s = pd.read_csv(d / "summary.csv")
        s.to_csv(TAB / f"hardware_{d.name}.csv", index=False)
        ns, lams = sorted(s.n.unique()), sorted(s.lam.unique())
        fig, axes = plt.subplots(1, len(lams), figsize=(FULL_W if len(lams) > 1 else COL_W, 2.3), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, lam in zip(axes, lams):
            g = s[s.lam == lam]
            width = 0.25
            for k, (src, c) in enumerate((("ideal_shots", AQUA), ("noise_model", ORANGE), ("hardware", BLUE))):
                h = g[g.source == src].set_index("n").reindex(ns)
                x = np.arange(len(ns)) + (k - 1) * width
                ax.bar(x, h.meas_excess_var, width=width * 0.9, color=c, label=src.replace("_", " "))
                ax.errorbar(x, h.meas_excess_var, yerr=[h.meas_excess_var - h.excess_var_lo, h.excess_var_hi - h.meas_excess_var],
                            fmt="none", ecolor=INK, elinewidth=0.7)
            ax.set_xticks(np.arange(len(ns)), [f"$n={n}$" for n in ns])
            ax.set_title(lam_label(lam), loc="left", color=INK)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("off-diagonal spread (excess var.)")
        axes[-1].legend(fontsize=6)
        fig.savefig(FIG / f"hardware_vs_noise_{d.name}.pdf")
        plt.close(fig)
        note(f"hardware: {d.name} -> hardware_vs_noise_{d.name}.pdf")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-tag", default="main", help="results/sweep/<tag> to analyze (dev runs use their own tag)")
    args = ap.parse_args()
    q = load_sweep(args.sweep_tag)
    if q is None:
        note("main sweep: no shards yet")
    else:
        qq = q[q.model == "quantum"]
        rr = q[q.model == "rbf"]
        rr = None if rr.empty else rr
        base = None
        bfiles = [f for m in ("linear", "logreg") for f in glob.glob(str(RESULTS / "cv" / f"*_{m}.csv"))]
        if bfiles:
            base = pd.concat([pd.read_csv(f) for f in bfiles], ignore_index=True)
        note(f"main sweep: {qq.groupby(['dataset']).n.unique().to_dict()} quantum; "
             f"{'no' if rr is None else rr.groupby('dataset').n.nunique().to_dict()} RBF")
        if not qq.empty:
            fig_concentration(qq)
            table_decay(qq)
            fig_accuracy_vs_lambda(qq)
            tq, tr = tuned_tables(qq, rr)
            fig_tuned_accuracy(tq, tr, base)
            fig_selected_lambda(tq)
            fig_shot_map(qq)
            table_tuned(tq, tr, base)
            falsifier(qq, tq)
            table_ablations(qq)
    fig_noise()
    hardware()
    (RESULTS / "RESULTS.md").write_text("# Generated results summary\n\nProduced by `experiments/analyze.py` from saved "
                                        "results; regenerate rather than edit.\n\n" + "\n".join(f"- {l}" for l in LOG) + "\n")


if __name__ == "__main__":
    main()
