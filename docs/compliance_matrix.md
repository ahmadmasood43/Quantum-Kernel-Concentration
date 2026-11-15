# AI 631 Project #5 — Requirement-to-Deliverable Compliance Matrix

**Project:** Kernel Concentration and Bandwidth Scaling
**Research question:** How fast do quantum kernel values concentrate as qubit count grows, and does a bandwidth hyperparameter restore learnability?
**Group:** Muhammad Ahmad Masood, Muhammad Waqas
**Matrix revision:** R2, 2026-11-11
**Next scheduled update:** 2026-11-15 (Week 1 review)

Status legend: `NOT STARTED` · `IN PROGRESS` · `DONE` (evidence linked) · `OVERDUE` · `BLOCKED` · `UNKNOWN`

A row moves to `DONE` only when the evidence column links to a committed file, a results artifact, or (for hardware) a stored IBM job ID plus its raw counts. A plan or a draft doesn't count.

---

## A. Compulsory technical requirements

| ID | Requirement | Expected artifact / experiment | Deadline | Status | Acceptance criterion |
|----|-------------|--------------------------------|----------|--------|----------------------|
| T1 | Quantum kernel sweep, 2–16 qubits × bandwidth | `experiments/sweep_sim.py`; fidelity kernel with feature map on scaled inputs `λ·x`; grid n ∈ {2,4,…,16}, λ on a log grid (≈ 8–10 values); results in `results/sweep/*.parquet` | Nov 22 | NOT STARTED | Every (n, λ, seed, fold) cell present in the results file; the run reproduces bit-for-bit from the documented seed; config is logged next to the results |
| T2 | Off-diagonal kernel variance analysis | `Var_{i≠j}[k(x_i,x_j)]` and mean off-diagonal value per (n, λ); log-scale plot vs n with fitted decay rate per λ | Nov 22 | NOT STARTED | Figure plus table of fitted exponential/polynomial decay rates with CIs over ≥5 seeds; analysis script regenerates the figure from saved results |
| T3 | Classification test accuracy | Precomputed-kernel SVM per (n, λ); C chosen by inner CV; test accuracy on held-out outer folds | Nov 22 | NOT STARTED | Mean ± std test accuracy over fixed outer folds; no test data used during model selection (verified by a code audit and a unit test) |
| T4 | Finite-shot kernel estimation vs concentration | Shot-sampled kernel estimates for N ∈ {e.g. 10², 10³, 10⁴, 10⁵}; compare true off-diagonal variance against shot-noise variance ≈ k(1−k)/N | Nov 22 | NOT STARTED | Plot/table that identifies, for each λ and N, the qubit count where concentration-induced spread falls below shot-noise spread (statistical test stated, e.g. variance-ratio CI); accuracy under finite shots reported |
| T5 | Dataset with PCA to n features | Fashion-MNIST (binary class pair, documented) **and/or** controlled synthetic set; PCA → n components fit **inside each training fold** | Nov 15 | **DONE** — [data/README.md](../data/README.md) (source, SHA-256, class pair 0 vs 2, 150/class/seed); `src/qkc/data.py`, `src/qkc/preprocess.py`; `tests/test_leakage.py` | `data/README.md` with download/preprocessing steps; test asserting that scaler/PCA are fit only on training indices; class pair and subsample size documented |
| T6 | RBF SVM baseline, comparable bandwidth search | RBF SVM on identical folds/features; γ searched on a log grid of the same size and selection protocol as λ | Nov 15 | **DONE** — `results/cv/*_rbf.csv`; 10-point grids for both γ and λ, same nested CV (`src/qkc/evaluate.py`); stated in `report/main.tex` §III-D. Follow-up: 16–36% of selections hit the lower grid edge, so both grids need extending downward before Week 2 | Baseline accuracy per n on the same folds; grid sizes and selection procedure identical to the quantum model and stated in Methods |
| T7 | Noise-free quantum simulation | Statevector (exact) kernel computation for all n ≤ 16 | Nov 15 (first model) / Nov 22 (full) | **DONE** — `src/qkc/kernels.py` (batched WHT statevector), all n = 2–16; `tests/test_kernels.py` matches Qiskit `Statevector` to 1e-10 for n ∈ {1,2,3,5}, λ ∈ {0.05,0.7,2} | Exact kernels match an independent implementation (e.g. Qiskit vs a hand-rolled NumPy statevector) to 1e-10 on small n |
| T8 | Realistic noise-model simulation | Aer simulation with a noise model derived from an IBM backend's calibration (T1/T2, gate and readout errors), transpiled to that backend's coupling map/basis | Nov 22 | NOT STARTED | Backend name and calibration timestamp recorded; transpiled depth and 2Q-gate counts logged per circuit; variance vs n plotted alongside noise-free results |
| T9 | ≥1 genuine IBM hardware experiment | Kernel matrix circuits run on a real IBM QPU via Qiskit Runtime | Nov 29 | NOT STARTED — access verified: Open plan, 600 s QPU per 28-day period, backends ibm_marrakesh / ibm_fez / ibm_kingston (156 q, native cz/rz/sx/x); [results/hardware/backends.json](../results/hardware/backends.json) | Stored job IDs, backend name, run timestamp, shots, raw counts, transpiled circuits (QPY), and the calibration snapshot at run time; results never mixed with simulator data |
| T10 | Hardware vs noise-model variance at 2–3 qubit counts | Same data points, same transpiled circuits, same shot count on hardware and on a noise model built from **the same backend's calibration at run time**; e.g. n ∈ {2, 4, 6} | Nov 29 | NOT STARTED (depends on T9) | Side-by-side table/figure of off-diagonal variance (with bootstrap CIs) for each selected n; differences discussed honestly |

## B. Starter harness (NB11)

| ID | Requirement | Expected artifact / experiment | Deadline | Status | Acceptance criterion |
|----|-------------|--------------------------------|----------|--------|----------------------|
| H1 | Use NB11 capstone starter notebook | Pipeline follows the NB11 protocol: one fixed fold object reused by every model, leakage-free preprocessing fit inside each fold, matched baselines (majority, logistic regression, linear SVM, tuned RBF), bandwidth tuned on inner splits only | Nov 15 | **DONE** — `src/qkc/folds.py`, `src/qkc/preprocess.py`, `src/qkc/evaluate.py`, `experiments/run_cv.py`; differences from NB11 stated in `report/main.tex` §III-B | Report states that the pipeline follows the NB11 protocol and names the differences |
| H2 | Fixed folds | `data/folds/*.json` with stratified outer/inner fold indices generated from a fixed seed | Nov 15 | **DONE** — `data/folds/{fmnist,synthetic}_seed{0..4}.json` (5 outer × 3 inner, stratified); SHA-256 in every `results/cv/*.meta.json` | Fold files committed; every model reads the same files; hash of each fold file recorded in results metadata |
| H3 | Leakage-free preprocessing | `sklearn.Pipeline` (scaler → PCA → feature scaling to angle range) fit per training fold | Nov 15 | **DONE** — `tests/test_leakage.py` (4 tests). These tests caught a real inner-fold indexing bug (inner folds mapped onto outer-test rows), which was fixed before any results were produced | Unit test that fails if any transform sees test indices |
| H4 | Classical baselines | RBF SVM (T6), linear SVM, logistic regression, majority class | Nov 15 | **DONE** — `results/cv/*_{majority,linear,logreg,rbf}.csv`, both datasets, n = 2–16 | Results committed with metadata |
| H5 | Ablation studies | e.g. feature-map depth/reps, entangling vs product map, λ fixed vs tuned, training-set size | Nov 22 | NOT STARTED | Each ablation has a config, saved results, and a figure/table in the report |
| H6 | Noise sweeps | Scale noise-model error rates (e.g. ×0, ×0.5, ×1, ×2) and measure variance/accuracy | Nov 22 | NOT STARTED | Sweep results saved; figure in Results |

## C. Milestones

| ID | Milestone (weight) | Expected artifact | Deadline | Status | Acceptance criterion |
|----|--------------------|-------------------|----------|--------|----------------------|
| M1 | Initial synopsis (5%) | `report/synopsis/synopsis.tex` + `refs.bib` — IEEEtran: RQ, H1/H2, falsifier, data/models, baselines, weekly plan | Nov 8 → **extended** (new date TBD, see D2) | NOT STARTED | PDF builds, ≤ 2 pages; .tex committed; submitted through the course channel |
| M2 | Week 1 review (15%) | Data pipeline, fixed folds, classical baselines, first working quantum model, Intro + Methods drafts, code on GitHub | Nov 15 (Sun) | **DONE** — all technical items DONE; Intro + Methods drafted in `report/main.tex`; **not yet pushed** | T5, T6, H2–H4 DONE; T7 working for ≥ 2 values of n; `report/` has Intro + Methods; pushed to the course repo under ahmadmasood43 |
| M3 | Week 2 review (25%) | Full simulator results, noise-model experiments, ablations, draft Results | Nov 22 (Sun) | NOT STARTED | T1–T4, T7, T8, H5, H6 DONE; Results draft with figures |
| M4 | Week 3 review (35%) | Real IBM hardware result; final presentation if scheduled that day | Nov 29 (Sun) | NOT STARTED | T9, T10 DONE with stored job IDs; slides ready if the presentation slot is Nov 29 |
| M5 | Final submission (20%) | Final report (PDF + LaTeX), tagged GitHub release, presentation for remaining groups | Dec 6 (Sun) | NOT STARTED | All rows in this matrix DONE; `git tag` release exists on the course repo |
| M6 | Weekly Sunday demo readiness | Current report draft, code, and completed experiments demonstrable | Every Sunday | NOT STARTED | `make demo` (or equivalent) runs a short end-to-end experiment in < 5 min; current report PDF builds |

## D. Formal report

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| R1 | IEEEtran two-column (chosen) | `report/main.tex` | Nov 15 (skeleton) | **DONE** — `report/main.tex` builds with `make report`; all eight sections present, Intro + Methods drafted | Builds to PDF with a single documented command |
| R2 | 6–8 pages + references | `report/main.pdf` | Dec 6 | NOT STARTED | Page count within range |
| R3 | Sections: Abstract, Intro, Background, Methods, Results, Discussion, Conclusion, References | Section headings in `main.tex` | Dec 6 | NOT STARTED | All eight present |
| R4 | Explicit research question + falsifiable hypothesis | Intro/Methods text | Nov 8 (synopsis) / Dec 6 | NOT STARTED | Falsifier stated quantitatively (e.g. "if, at fixed N shots, tuned λ fails to keep off-diagonal variance above the shot-noise floor for n ≥ n₀, the hypothesis is rejected") |
| R5 | Qubit counts, circuit depth, 2Q gate counts, shot counts | Auto-generated table from logged transpilation stats | Nov 29 | NOT STARTED | Table generated by script from saved metadata, not typed by hand |
| R6 | Author-contributions statement | Section in report | Dec 6 | NOT STARTED | Accurate per-person statement |
| R7 | Honest comparison with classical baselines | Results + Discussion | Dec 6 | NOT STARTED | Quantum vs RBF on identical folds, including cases where RBF wins |
| R8 | Verified, non-hallucinated references | `report/refs.bib` (7 entries, shared with synopsis) | Every draft | NOT STARTED | Every entry checked against Crossref/DataCite/publisher before inclusion |

## E. Presentation

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| P1 | ~12 min talk + 5 min Q&A | `presentation/` slides (problem, literature, circuit, methods, baselines, simulator, hardware, limitations, conclusions) | Nov 29 or Dec 6 (slot TBD, see D3) | NOT STARTED | Timed rehearsal ≤ 12 min; every listed topic has ≥ 1 slide |
| P2 | Every member presents a meaningful portion | Speaker allocation across both members | Same | NOT STARTED | Each member has a section of the talk assigned and rehearsed |

## F. GitHub deliverables

| ID | Requirement | Expected artifact | Deadline | Status | Acceptance criterion |
|----|-------------|-------------------|----------|--------|----------------------|
| G1 | Working, reproducible source code | `src/qkc/` package + `experiments/` scripts | Nov 15 → Dec 6 | **DONE** — `src/qkc/`, `experiments/run_cv.py`, `Makefile` (`setup`, `test`, `cv-*`, `demo`, `synopsis`, `report`) | Fresh clone + setup + `make all` reproduces results |
| G2 | README with setup instructions | `README.md` | Nov 15 | **DONE** — setup, IBM access, LaTeX, results-reproduction table, layout | A new user can set up from the README alone |
| G3 | Dependency requirements | `requirements.txt` with pinned versions (Python 3.12, uv venv `.venv/`) | Nov 15 | **DONE** — pinned file written; install verified in `.venv` | Pinned versions; fresh venv install verified |
| G4 | Fixed random seeds | Central seed config; seeds stored in every results file | Nov 15 | **DONE** — seeds in `src/qkc/config.py`; seed column in every results CSV; rerun gives byte-identical CSVs (checked via `make demo` ×2) | Re-run produces identical simulator outputs |
| G5 | Dataset/preprocessing instructions | `data/README.md` | Nov 15 | **DONE** — [data/README.md](../data/README.md) | Covers download, class pair, subsampling, PCA |
| G6 | Saved experimental results | `results/` (small files committed; large ones as release assets) | Nov 22 → Dec 6 | NOT STARTED | Every report figure/table traceable to a results file |
| G7 | Reproduce every figure/table | `make figures` + mapping table in README | Dec 6 | NOT STARTED | Each figure/table lists its generating command |
| G8 | Report source files | `report/` | Dec 6 | NOT STARTED | .tex, .bib, figures committed |
| G9 | Meaningful commits from every member, under correct identity | Ahmad's commits authored as **ahmadmasood43**; Muhammad Waqas commits from his own account | Ongoing | NOT STARTED | `git log` authors map to the correct GitHub accounts; no AI co-author trailers; no commits made under another member's name |
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

## Blockers (as of R1, 2026-11-11)

| ID | Blocker | Status | Resolution needed from |
|----|---------|--------|------------------------|
| B1 | Wrong git identity / no push access | **RESOLVED** — remote is `git@github-personal:ahmadmasood43/Quantum-Kernel-Concentration.git`; repo-local author is ahmadmasood43; SSH auth as ahmadmasood43 verified | — |
| B2 | NB11 availability | **RESOLVED** — NB11 is available; the pipeline follows its protocol (see H1) | — |
| B3 | IBM Quantum credentials | **RESOLVED** — saved by the student; `scripts/check_ibm_access.py` lists 3 backends; usage API reports 600 s per period, 0 s used | — |
| B4 | No LaTeX toolchain | **RESOLVED** — TeX Live installed by the student; synopsis and report build | — |

## Open decisions

| ID | Question | Status |
|----|----------|--------|
| D1 | Group membership | **RESOLVED** — Muhammad Ahmad Masood and Muhammad Waqas; both are listed as authors; responsibility attributions name Muhammad Ahmad Masood |
| D2 | New synopsis deadline after the extension | OPEN — date needed |
| D3 | Presentation slot: Nov 29 or Dec 6? | OPEN |
| D5 | Dataset | **RESOLVED** — Fashion-MNIST binary pair (primary) + controlled synthetic Gaussian mixture |

## Revision log

| Rev | Date | Change |
|-----|------|--------|
| R0 | 2026-11-11 | Baseline matrix; environment audit; blockers B1–B4 recorded |
| R1 | 2026-11-11 | B1, B2 resolved; env + pinned requirements; IBM setup scripts; synopsis .tex + verified .bib; group membership and dataset decided |
| R2 | 2026-11-11 | B3, B4 resolved; Week 1 pipeline, tests, baselines, and full exact-simulation CV sweep (n = 2–16) done; synopsis PDF built; report skeleton with Intro + Methods |
