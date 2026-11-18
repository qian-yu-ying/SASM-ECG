# Release validation — v1.4.1

All checks were performed locally on CPU using completed cached experiment
assets (`v1.4.1` working tree before commit/tag). No remote experiment,
worker, fitting job or GPU job was started.

- Nine-class CPSC2018 structure verified (`examples/verify_cpsc9_protocol.py`):
  6,877 records, 2,755 support-pool records, 4,122 query records, 3,966 query
  waveform groups and 40 fixed support episodes (`STE lambda_STE=0`).
- Companion asset SHA-256 verification passed across all 276 entries in
  `data/ASSET_MANIFEST.json` (`examples/verify_assets.py`).
- All 12 unit tests passed (`python -m unittest discover -s tests -v`), covering
  algebraic gradients, affine coordinate equivalence, zero-lambda unmatched-class
  behavior, optimizer guarantees, `tests/test_dca_release.py` (`v1.3.0` DCA
  protocol & summaries), and `tests/test_ste_supplementary.py` (exact offline
  recomputation of Table S18 means, 10,000-seed bootstrap intervals, paired
  Wilcoxon $p$-values, and 8-contrast Holm corrections from `ste_episodes.csv`).
- Full archived coefficient replay passed (`examples/replay_cpsc9.py`, 80
  episodes x 9 methods x 3 scopes = 2,160 metric rows; max AP/AUROC error vs.
  reference CSV <= 2.22e-16).
- Fullpool PTB-XL source-head stationarity and CV disjointness verified
  (`examples/verify_source_heads.py`).
- Seed-0 fresh primary refit gate passed for ECGFounder and MERL at 5-shot and
  10-shot (`examples/evaluate_cpsc9.py`), alongside `smoke_check.py`.
- All 30 Python files in `scripts/supplementary/` and
  `scripts/reference_pipelines/` passed syntax verification (`py_compile`).

See `validation/RELEASE_CHECKS.json` for scope and environment details and
`docs/REPRODUCTION_SCOPE.md` for execution boundaries.
