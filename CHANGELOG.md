# Current release

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
