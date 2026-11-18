# Reproduction scope — v1.4.1

| Asset or operation | Status |
|---|---|
| Local CPU fitting of LP, SASM-Ens, MeanAnchor on three cached cohorts | Runnable; CPSC2018 uses the nine-class protocol |
| CPSC9 frozen direct/interpolation, fullpool, concatenation, native and scaling controls listed by `evaluate_cpsc9.py` | Runnable; native controls only on ECGFounder |
| Normal -> SR sensitivity for supported CPSC9 frozen methods | Runnable, separate from NORM primary |
| Fixed support episodes, query labels, complete 6,877-record CPSC feature arrays | Included in companion assets; labels/episodes also in code snapshot |
| Source-head refitting and source-only CV on frozen source features | Runnable |
| Ningbo/SPH Route-B evaluation | Runnable with their existing protocol |
| DCA and locked-threshold tables (`results/dca_threshold/`) | Preserved from v1.3.0 (identical summary CSV/lock subset also bundled in `results/supplementary/`); retrospective and not deployment validation |
| Supplementary extension bundles (`results/supplementary/`, `scripts/supplementary/`) | Exported CSV/JSON provenance summaries (including Table S18 per-seed `ste_episodes.csv` and `ste_paired_effects.csv` verified offline by `tests/test_ste_supplementary.py`) with generation scripts (`ste_control.py`, etc.); full upstream re-execution requires non-redistributed caches |
| Full-study reference pipelines & FullFT/LoRA scripts (`scripts/reference_pipelines/gpu_worker.py`) | Archived reference implementations and configs; requires independent upstream dataset downloads and pretrained foundation checkpoints |
| Complete Chapman target hyperparameter search | Not regenerated; target development features not distributed |

NORM is the fixed primary Normal mapping. STE has a zero source column and
lambda_STE=0 in primary SASM. The separate tau=10.1 control must not be substituted
for primary results after inspecting query performance. Direct methods use LP
fallback for STE. The native shared comparison covers seven endpoints and omits
STD/STE; it is not the nine-class primary endpoint.

The core optimizer checks stationarity and retries failed solves; it raises on
nonfinite output or a projected maximum gradient above 5e-5. Supports are fixed
and contain both labels for each endpoint. Query labels enter scoring only.
The exported protocol records prior exposure to external results: this release
does not create a prospectively blinded experiment or establish absence of
pretraining-data overlap.

Do not infer that repository completeness equals methodological completeness.
Use the manuscript's stated clinical and statistical limitations. The release
validation report (`VERIFICATION.md`, `validation/RELEASE_CHECKS.json`) states
exact local checks performed.

Archived nine-class target coefficients for the primary NORM mapping are included.
`examples/replay_cpsc9.py` recomputes query scores without refitting. Fresh fitting
uses the same objective and tolerances, but BLAS/solver differences need not yield
bitwise-identical predictions. The refit reference gate applies to all9/matched8
macro endpoints; STE-only AP is reported separately because it is more sensitive
to close-score ordering.
