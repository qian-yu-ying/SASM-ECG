# DCA and locked-threshold scope

Release v1.3.0 contains the DCA and threshold exports from the supplementary
analysis completed on 2026-11-18. The analysis reuses frozen predictions and
does not retrain an encoder or access a GPU.

## What is included

- DCA at `p_t = {0.02, 0.05, 0.10, 0.15, 0.20}` for LP, Direct-LP,
  intercept-refitted Direct-LP, SASM-Ens and MeanAnchor.
- Raw and Chapman-development global-temperature representations.
- A Chapman-development global threshold selected from negative scores.
- A per-episode support-set negative 95th-percentile threshold.
- Condition-level, class-level and paired support-seed summaries.

The external score files are evaluated only after the threshold lock is written.
The release manifest and `provenance/dca_threshold_protocol.json` record the
fixed query cohorts, the class denominator, the DCA formula and the uncertainty
unit.

## What the results do not establish

The query cohorts were fixed and the analysis is retrospective. Intervals are
conditional on the 20 support seeds and do not represent patient-level query
uncertainty. Ningbo and CPSC grouping uses waveform groups; this must not be
described as verified patient independence. The support-set quantile rule is an
in-sample threshold estimate and its external specificity can drift. DCA is
conditional on a chosen risk threshold and prevalence and is not a clinical
deployment or net-benefit claim without a prespecified clinical action.

The primary CPSC2018 `lambda_STE=0` protocol is unchanged. The separate
`tau=10.1` sensitivity protocol is not substituted for the primary results.
