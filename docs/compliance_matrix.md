# AI 631 Project #5 — Requirement-to-Deliverable Compliance Matrix

**Project:** Kernel Concentration and Bandwidth Scaling
**Research question:** How fast do quantum kernel values concentrate as qubit count grows, and does a bandwidth hyperparameter restore learnability?
**Group:** Muhammad Ahmad Masood, Muhammad Waqas
**Matrix revision:** R3, 2026-11-24 (branch `week2`)
**Next scheduled update:** 2026-11-29 (Week 3 review)

Status legend: `NOT STARTED` · `IN PROGRESS` · `DONE` (evidence linked) · `OVERDUE` · `BLOCKED` · `UNKNOWN`

A row moves to `DONE` only when the evidence column links to a committed file, a results artifact, or (for hardware) a stored IBM job ID plus its raw counts. A plan or a draft doesn't count. Evidence on another branch is marked with the branch name.

---

## A. Compulsory technical requirements

| ID | Requirement | Expected artifact / experiment | Deadline | Status | Acceptance criterion |
|----|-------------|--------------------------------|----------|--------|----------------------|
| T1 | Quantum kernel sweep, 2–16 qubits × bandwidth | `experiments/sweep_sim.py`; fidelity kernel with feature map on scaled inputs `λ·x`; grid n ∈ {2,4,…,16}, λ on a log grid (≈ 8–10 values); results in `results/sweep/*.parquet` | Nov 22 | **DONE** — branch `experimentation`: `results/sweep/main/` (160 shards + metadata): quantum and RBF, n = 2–16, λ = 2^-10..2^1, 5 seeds × 5 folds, both datasets; `make sweep`. This branch: `notebooks/` (exact, n = 2–16; Fashion-MNIST sweep loaded from stored results, AG News and Speech Commands computed in full), λ ∈ {10^-3, …, 1} | Every (n, λ, seed, fold) cell present in the results file; the run reproduces bit-for-bit from the documented seed; config is logged next to the results |
| T2 | Off-diagonal kernel variance analysis | `Var_{i≠j}[k(x_i,x_j)]` and mean off-diagonal value per (n, λ); log-scale plot vs n with fitted decay rate per λ | Nov 22 | **DONE** — branch `experimentation`: `report/figures/concentration_variance.pdf`, `report/tables/decay_rates.tex`; λ = 1 slope −0.452 ± 0.033 per qubit (Fashion-MNIST), −0.927 ± 0.027 (synthetic), exponential fit preferred over power law. This branch: slopes and fits in `report/week2_notebook/` Table II; Speech Commands plateau caused by nine identical training-clip pairs | Figure plus table of fitted exponential/polynomial decay rates with CIs over ≥5 seeds; analysis script regenerates the figure from saved results |
| T3 | Classification test accuracy | Precomputed-kernel SVM per (n, λ); C chosen by inner CV; test accuracy on held-out outer folds | Nov 22 | **DONE** — branch `experimentation`: fixed-λ and tuned-λ test accuracy in `results/sweep/main/`; `accuracy_vs_lambda.pdf`, `tuned_accuracy.pdf`, `report/tables/tuned_accuracy.tex`. This branch: `notebooks/`, `results/cv/` (Week 1) | Mean ± std test accuracy over fixed outer folds; no test data used during model selection (verified by a code audit and a unit test) |
| T4 | Finite-shot kernel estimation vs concentration | Shot-sampled kernel estimates for N ∈ {e.g. 10², 10³, 10⁴, 10⁵}; compare true off-diagonal variance against shot-noise variance ≈ k(1−k)/N | Nov 22 | **DONE** — branch `experimentation`: N ∈ {10²,…,10⁵} with model selection under shots; `shot_noise_map.pdf`, `selected_lambda.pdf`; proposal falsifier evaluated in `results/tables/falsifier_proposal.csv` (not rejected). This branch: shot sweep in `notebooks/`; IQP on Speech Commands triggers the falsifier (McNemar p = 0.008) | Plot/table that identifies, for each λ and N, the qubit count where concentration-induced spread falls below shot-noise spread (statistical test stated, e.g. variance-ratio CI); accuracy under finite shots reported |
| T5 | Dataset with PCA to n features | Fashion-MNIST (binary class pair, documented) **and/or** controlled synthetic set; PCA → n components fit **inside each training fold** | Nov 15 | **DONE** — [data/README.md](../data/README.md) (source, SHA-256, class pair 0 vs 2, 150/class/seed); `src/qkc/data.py`, `src/qkc/preprocess.py`; `tests/test_leakage.py` | `data/README.md` with download/preprocessing steps; test asserting that scaler/PCA are fit only on training indices; class pair and subsample size documented |
| T6 | RBF SVM baseline, comparable bandwidth search | RBF SVM on identical folds/features; γ searched on a log grid of the same size and selection protocol as λ | Nov 15 | **DONE** — `results/cv/*_rbf.csv`; 10-point grids for both γ and λ, same nested CV (`src/qkc/evaluate.py`); stated in `report/main.tex` §III. The lowest grid value was selected in 17–19% of fits overall | Baseline accuracy per n on the same folds; grid sizes and selection procedure identical to the quantum model and stated in Methods |
| T7 | Noise-free quantum simulation | Statevector (exact) kernel computation for all n ≤ 16 | Nov 15 (first model) / Nov 22 (full) | **DONE** — `src/qkc/kernels.py` (batched WHT statevector), all n = 2–16; `tests/test_kernels.py` matches Qiskit `Statevector` to 1e-10 for n ∈ {1,2,3,5}, λ ∈ {0.05,0.7,2} | Exact kernels match an independent implementation (e.g. Qiskit vs a hand-rolled NumPy statevector) to 1e-10 on small n |
| T8 | Realistic noise-model simulation | Aer simulation with a noise model derived from an IBM backend's calibration (T1/T2, gate and readout errors), transpiled to that backend's coupling map/basis | Nov 22 | **DONE** — branch `experimentation`: `results/noise/` (n = 2–10, 4 λ, scale 0–2, m = 16, 4,000 shots) from the ibm_marrakesh snapshot of 2026-11-11; `noise_model_variance.pdf`, `noise_model_diagonal.pdf`. This branch: archived IBM Manila calibration in `notebooks/` | Backend name and calibration timestamp recorded; transpiled depth and 2Q-gate counts logged per circuit; variance vs n plotted alongside noise-free results |
| T9 | ≥1 genuine IBM hardware experiment | Kernel matrix circuits run on a real IBM QPU via Qiskit Runtime | Nov 29 | **DONE** — branch `experimentation`: ibm_marrakesh, 2026-11-11. Pilot: job `dbq5udoou4as73cukgg0` (21 circuits, 8 s). Main: job `dbq68tgqtr0c738fd5qg`, 495 circuits × 1000 shots, n ∈ {2,4,6}, λ ∈ {2^-4,2^-2,1}, m = 10, opt. level 3; 135 s charged; ISA depth up to 126, ≤ 40 CZ; raw counts, QPY, preflight, run-time calibration in `results/hardware/hw1/`. QPU used 143 s of 600 s | Stored job IDs, backend name, run timestamp, shots, raw counts, transpiled circuits (QPY), and the calibration snapshot at run time; results never mixed with simulator data |
| T10 | Hardware vs noise-model variance at 2–3 qubit counts | Same data points, same transpiled circuits, same shot count on hardware and on a noise model built from **the same backend's calibration at run time**; e.g. n ∈ {2, 4, 6} | Nov 29 | **DONE** — branch `experimentation`: `results/hardware/hw1/summary.csv`: excess off-diagonal variance with point-bootstrap 95% CIs for hardware, matched noise model (calibration at run time), and ideal shots. CIs overlap in 9/9 blocks; hardware/model ratio 0.75–1.3 except n=4, λ=2^-4 (3.2). Hardware k(x,x) below the model at n ≥ 4 (0.74 vs 0.83 at n=6, λ=1). Figure from `make figures` | Side-by-side table/figure of off-diagonal variance (with bootstrap CIs) for each selected n; differences discussed honestly |

## B. Starter harness (NB11)

| ID | Requirement | Expected artifact / experiment | Deadline | Status | Acceptance criterion |
|----|-------------|--------------------------------|----------|--------|----------------------|
| H1 | Use NB11 capstone starter notebook | Fixed folds, per-fold preprocessing, matched baselines, inner-split tuning, ablations, noise sweep | Nov 15 | IN PROGRESS — the pipeline was written on 2026-11-11, before NB11 was obtained (2026-11-15), and does not use the notebook. It has similar folds, per-fold preprocessing, baselines and inner-split tuning; it does not use shot-sampled kernels, the label-shuffle test or AUC. The wording in `report/main.tex` ("follows the protocol") still needs correcting | Report states accurately whether NB11 was used and names the differences |
| H2 | Fixed folds | `data/folds/*.json` with stratified outer/inner fold indices generated from a fixed seed | Nov 15 | **DONE** — `data/folds/{fmnist,synthetic}_seed{0..4}.json` (5 outer × 3 inner, stratified); SHA-256 in every `results/cv/*.meta.json` | Fold files committed; every model reads the same files; hash of each fold file recorded in results metadata |
| H3 | Leakage-free preprocessing | `sklearn.Pipeline` (scaler → PCA → feature scaling to angle range) fit per training fold | Nov 15 | **DONE** — `tests/test_leakage.py` (4 tests). These tests caught a real inner-fold indexing bug (inner folds mapped onto outer-test rows), which was fixed before any results were produced | Unit test that fails if any transform sees test indices |
| H4 | Classical baselines | RBF SVM (T6), linear SVM, logistic regression, majority class | Nov 15 | **DONE** — `results/cv/*_{majority,linear,logreg,rbf}.csv`, both datasets, n = 2–16 | Results committed with metadata |
| H5 | Ablation studies | e.g. feature-map depth/reps, entangling vs product map, λ fixed vs tuned, training-set size | Nov 22 | **DONE** — branch `experimentation`: `results/sweep/{reps1,reps3,product,m100,m500}/` (120 shards; `make ablations`), summary `results/tables/ablations.csv`: tuned accuracy 0.946–0.958 for all feature-map variants at n = 16; variance at λ = 1, n = 16: reps 3 1.1e-5 < main 1.1e-4 < product map 3.5e-4. This branch: Trotter, angle-precision and initialization controls in `notebooks/` | Each ablation has a config, saved results, and a figure/table in the report |
| H6 | Noise sweeps | Scale noise-model error rates (e.g. ×0, ×0.5, ×1, ×2) and measure variance/accuracy | Nov 22 | **DONE** — branch `experimentation`: noise scale ∈ {0, 0.5, 1, 2} in `results/noise/summary.csv` | Sweep results saved; figure in Results |

## C. Milestones

| ID | Milestone (weight) | Expected artifact | Deadline | Status | Acceptance criterion |
|----|--------------------|-------------------|----------|--------|----------------------|
| M1 | Initial synopsis (5%) | Project proposal (IEEEtran): RQ, hypotheses, falsifier, data/models, baselines, weekly plan | Nov 8 → **extended** (new date TBD, see D2) | IN PROGRESS — `QML_Project_Proposal.pdf`; branch `experimentation`: `report/proposal_revised/`; submission not recorded here | PDF builds, ≤ 2 pages; .tex committed; submitted through the course channel |
| M2 | Week 1 review (15%) | Data pipeline, fixed folds, classical baselines, first working quantum model, Intro + Methods drafts, code on GitHub | Nov 15 (Sun) | **DONE** pending review — technical items done; Intro + Methods in `report/main.tex`; pushed | T5, T6, H2–H4 DONE; T7 working for ≥ 2 values of n; `report/` has Intro + Methods; pushed to the course repo under ahmadmasood43 |
| M3 | Week 2 review (25%) | Full simulator results, noise-model experiments, ablations, draft Results | Nov 22 (Sun) | IN PROGRESS — results and 2-page progress report on this branch; deadline Nov 22 has passed and submission is not recorded here | T1–T4, T7, T8, H5, H6 DONE; Results draft with figures |
| M4 | Week 3 review (35%) | Real IBM hardware result; final presentation if scheduled that day | Nov 29 (Sun) | IN PROGRESS — branch `experimentation`: T9, T10 done on 2026-11-11; slides not started | T9, T10 DONE with stored job IDs; slides ready if the presentation slot is Nov 29 |
| M5 | Final submission (20%) | Final report (PDF + LaTeX), tagged GitHub release, presentation for remaining groups | Dec 6 (Sun) | NOT STARTED | All rows in this matrix DONE; `git tag` release exists on the course repo |
| M6 | Weekly Sunday demo readiness | Current report draft, code, and completed experiments demonstrable | Every Sunday | IN PROGRESS — `make demo` runs tests and a one-seed run; `make report` builds | `make demo` (or equivalent) runs a short end-to-end experiment in < 5 min; current report PDF builds |

## D. Formal report

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| R1 | IEEEtran two-column (chosen) | `report/main.tex` | Nov 15 (skeleton) | **DONE** — `report/main.tex` (IEEEtran) builds with `make report` | Builds to PDF with a single documented command |
| R2 | 6–8 pages + references | `report/main.pdf` | Dec 6 | NOT STARTED | Page count within range |
| R3 | Sections: Abstract, Intro, Background, Methods, Results, Discussion, Conclusion, References | Section headings in `main.tex` | Dec 6 | IN PROGRESS — all sections present in the 2-page Week 1 report; final report not written | All eight present |
| R4 | Explicit research question + falsifiable hypothesis | Intro/Methods text | Nov 8 (synopsis) / Dec 6 | IN PROGRESS — RQ, H1/H2 and the proposal's falsifier in `report/main.tex` §I and `report/week2_notebook/main.tex` §I | Falsifier stated quantitatively (e.g. "if, at fixed N shots, tuned λ fails to keep off-diagonal variance above the shot-noise floor for n ≥ n₀, the hypothesis is rejected") |
| R5 | Qubit counts, circuit depth, 2Q gate counts, shot counts | Auto-generated table from logged transpilation stats | Nov 29 | **DONE** — branch `experimentation`: `report/tables/resources.tex` (generated by `make figures`): logical circuits for exact simulation, ISA depth and CZ counts (min/median/max) for noise model and hardware, shots. The counts in this branch's reports are typed by hand | Table generated by script from saved metadata, not typed by hand |
| R6 | Author-contributions statement | Section in report | Dec 6 | NOT STARTED | Accurate per-person statement |
| R7 | Honest comparison with classical baselines | Results + Discussion | Dec 6 | IN PROGRESS — `report/week2_notebook/` Table I, including cases where RBF is better | Quantum vs RBF on identical folds, including cases where RBF wins |
| R8 | Verified, non-hallucinated references | `report/refs.bib`, `report/week2_notebook/refs.bib` | Every draft | IN PROGRESS — 7 entries verified via Crossref/DataCite; 2 dataset entries verified via arXiv | Every entry checked against Crossref/DataCite/publisher before inclusion |

## E. Presentation

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| P1 | ~12 min talk + 5 min Q&A | `presentation/` slides (problem, literature, circuit, methods, baselines, simulator, hardware, limitations, conclusions) | Nov 29 or Dec 6 (slot TBD, see D3) | NOT STARTED | Timed rehearsal ≤ 12 min; every listed topic has ≥ 1 slide |
| P2 | Every member presents a meaningful portion | Speaker allocation across both members | Same | NOT STARTED | Each member has a section of the talk assigned and rehearsed |

## F. GitHub deliverables

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| G1 | Working, reproducible source code | `src/qkc/` package + `experiments/` scripts | Nov 15 → Dec 6 | IN PROGRESS — `src/qkc/`, `experiments/run_cv.py`, `notebooks/`, `Makefile`; full code on branch `experimentation`; no `make all` on this branch | Fresh clone + setup + `make all` reproduces results |
| G2 | README with setup instructions | `README.md` | Nov 15 | IN PROGRESS — README covers the Week 1 code; `notebooks/` not yet documented there | A new user can set up from the README alone |
| G3 | Dependency requirements | `requirements.txt` with pinned versions (Python 3.12, uv venv `.venv/`) | Nov 15 | IN PROGRESS — `requirements.txt` covers the Week 1 code; the notebooks list their own versions and also need `pylatexenc` | Pinned versions; fresh venv install verified |
| G4 | Fixed random seeds | Central seed config; seeds stored in every results file | Nov 15 | **DONE** — seeds in `src/qkc/config.py`; seed column in every results CSV; rerun gives byte-identical CSVs (checked via `make demo` ×2) | Re-run produces identical simulator outputs |
| G5 | Dataset/preprocessing instructions | `data/README.md` | Nov 15 | **DONE** — [data/README.md](../data/README.md) | Covers download, class pair, subsampling, PCA |
| G6 | Saved experimental results | `results/` (small files committed; large ones as release assets) | Nov 22 → Dec 6 | IN PROGRESS — `results/cv/`, `notebooks/results/`; remaining results on branch `experimentation` | Every report figure/table traceable to a results file |
| G7 | Reproduce every figure/table | `make figures` + mapping table in README | Dec 6 | IN PROGRESS — on branch `experimentation` (`make figures`); not on this branch | Each figure/table lists its generating command |
| G8 | Report source files | `report/` | Dec 6 | IN PROGRESS — `report/main.tex`, `report/week2_notebook/`; final report not written | .tex, .bib, figures committed |
| G9 | Meaningful commits from every member, under correct identity | Each member commits from their own GitHub account | Ongoing | IN PROGRESS — commits exist from more than one GitHub account | `git log` authors map to the correct GitHub accounts; no AI co-author trailers; no commits made under another member's name |
| G10 | Tagged final release | `v1.0` GitHub release | Dec 6 | NOT STARTED | Tag present on the course repo |

## G. Optional research extensions (Priority 4–5; only after mandatory rows are on track)

| ID | Item | Status | Gate |
|----|------|--------|------|
| X1 | Bandwidth-scaling law: λ*(n) that keeps variance above the shot floor | NOT STARTED | Only once M3 is complete |
| X2 | Total shot cost to reach target accuracy vs n | NOT STARTED | Only once M3 is complete |
| X3 | Classical approximability of small-λ kernels (e.g. comparison with low-order/RBF-like expansions) | NOT STARTED | Only once M4 is complete |
| X4 | Assessment of NeurIPS/ICML/ICLR defensibility | NOT STARTED | After M5; written as an honest assessment, not a claim |

---

## Week 1 results snapshot (exact simulation, mean test accuracy over 5 seeds × 5 outer folds)

| Dataset | Model | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| Fashion-MNIST | quantum (tuned λ) | 0.936 | 0.944 | 0.957 | 0.951 | 0.959 |
| Fashion-MNIST | RBF (tuned γ) | 0.935 | 0.947 | 0.952 | 0.955 | 0.956 |
| Fashion-MNIST | linear SVM | 0.935 | 0.954 | 0.956 | 0.957 | 0.961 |
| Synthetic (Bayes ≈ 0.841) | quantum (tuned λ) | 0.856 | 0.842 | 0.819 | 0.827 | 0.830 |
| Synthetic | RBF (tuned γ) | 0.859 | 0.844 | 0.821 | 0.821 | 0.827 |

Paired quantum − RBF differences are within ±0.007 at every n, with per-fold std 0.012–0.030. The median selected λ falls from 0.13 (n=2) to 0.036 (n ≥ 8) on Fashion-MNIST. Source: `results/cv/`. This is a progress snapshot, not a report claim.

## Blockers (as of R3, 2026-11-24)

| ID | Blocker | Status | Resolution needed from |
|----|---------|--------|------------------------|
| B1 | Wrong git identity / no push access | **RESOLVED** — remote is `git@github-personal:ahmadmasood43/Quantum-Kernel-Concentration.git`; repo-local author is ahmadmasood43; SSH auth as ahmadmasood43 verified | — |
| B2 | NB11 availability | **RESOLVED** — NB11 obtained on 2026-11-15, after the pipeline was written (see H1) | — |
| B3 | IBM Quantum credentials | **RESOLVED** — saved by the student; 143 s of 600 s used on 2026-11-11 (see T9) | — |
| B4 | No LaTeX toolchain | **RESOLVED** — TeX Live installed by the student; reports build | — |

## Open decisions

| ID | Question | Status |
|----|----------|--------|
| D1 | Group membership | **RESOLVED** — Muhammad Ahmad Masood and Muhammad Waqas; both are listed as authors |
| D2 | New synopsis deadline after the extension | OPEN — date needed |
| D3 | Presentation slot: Nov 29 or Dec 6? | OPEN |
| D5 | Dataset | **RESOLVED** — Fashion-MNIST binary pair (primary) + controlled synthetic Gaussian mixture |

## Revision log

| Rev | Date | Change |
|-----|------|--------|
| R0 | 2026-11-11 | Baseline matrix; environment audit; blockers B1–B4 recorded |
| R1 | 2026-11-11 | B1, B2 resolved; env + pinned requirements; IBM setup scripts; synopsis .tex + verified .bib; group membership and dataset decided |
| R2 | 2026-11-11 | B3, B4 resolved; Week 1 pipeline, tests, baselines, and full exact-simulation CV sweep (n = 2–16) done; synopsis PDF built; report skeleton with Intro + Methods |
| R3 | 2026-11-24 | Statuses corrected: the 2026-11-15 edits had marked completed work (T1–T4, T8–T10, H5, H6) as not started and some open rows as done. Week 2 notebooks and report added |
