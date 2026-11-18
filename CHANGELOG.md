# Current release

## v1.4.1 — 2026-11-18

- Add complete per-seed Table S18 unmatched CPSC2018 `STE` sensitivity artifacts
  (`results/supplementary/ste_episodes.csv`, `ste_summary.csv`,
  `ste_paired_effects.csv`, `ste_protocol.lock.json`) and generator script
  (`scripts/supplementary/ste_control.py`).
- Add `tests/test_ste_supplementary.py` to independently recompute all Table S18
  means, 10,000-seed bootstrap intervals, paired Wilcoxon $p$-values, and
  8-contrast Holm corrections offline in CI.
- Clarify manuscript author attribution (`Chenxi Yang, Guolan Deng, and Yi Wang`,
  Chongqing University) directly in `README.md` alongside `CITATION.cff` and `LICENSE`.

## v1.4.0 — 2026-11-18

- Consolidate all six supplementary CPU and clinical analysis exports under
  `results/supplementary/` (controlled raw-embedding `h` vs. 150-logit `g`
  ablation, bidirectional SPH-dev role swap, Chapman-locked `DevTemperature`
  calibration, unified pinned single-core Xeon Gold 5418Y BLAS-1 latency,
  48-slice age/sex subgroup metrics and `DEMOGRAPHIC_CONFLICT_AUDIT.json`, and
  low-prevalence DCA + query-label-free locked threshold tables).
- Add the archived supplementary generation scripts under
  `scripts/supplementary/` and the full-study baseline, ablation, timing,
  fine-tuning (`FullFT`/`LoRA`, including `gpu_worker.py`), and bootstrap/Holm
  aggregation reference pipelines under `scripts/reference_pipelines/`.
- Add explicit author copyright attribution (`Chenxi Yang, Guolan Deng, and Yi
  Wang`) in `LICENSE` and `LICENSE_NOTICE.md`.
- Preserve all `v1.3.0` `results/dca_threshold/` exports, DCA provenance, and
  `v1.2.0` nine-class CPSC2018 protocol, STE rule, and archived coefficients
  unchanged.

## v1.3.0 — 2026-11-18

- Add DCA and locked-threshold result exports from the retrospective
  supplementary analysis, with protocol and numerical-check records.
- Add author citation metadata and clarify cached-feature replay, record-level
  grouping and the absence of an end-to-end raw-waveform pipeline.
- Keep the v1.2.0 nine-class CPSC2018 protocol, primary STE rule and archived
  coefficients unchanged.

## v1.2.0 — 2026-11-18

- Align CPSC2018 code, cached features, label order, roles and results with the
  complete nine-class 2026-11-17 protocol: 6,877 records and 4,122 query records.
- Export all 40 fixed support episodes and source-head mapping in the same row
  order as the cached features.
- Implement class-specific lambda: STE has no source anchor in the locked primary
  protocol. Direct scores fall back to the fitted LP score on STE.
- Add a dedicated nine-class evaluator, scope slices and per-seed reference checks.
- Reject incompatible CPSC2018 caches. Separate executable cached evaluation from
  exported results-only experiments and raw-waveform reconstruction.
