# Current release

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
