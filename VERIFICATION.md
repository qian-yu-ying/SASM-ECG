# Checks performed for this release

- Existing server assets were downloaded using read-only SFTP. No server
  experiment, feature extraction or retraining was started.
- Both backbones' 20 source heads and all six original feature arrays matched
  the study's existing SHA-256 input seals before export.
- Exported feature rows preserve the original numeric values exactly. They
  comprise queries plus the union of the fixed 5-/10-shot supports for 20 seeds.
- Patient identifiers, waveform paths, source out-of-fold sample predictions
  and source training-record lists are not distributed in the numeric assets.
- The objective and certified optimizer match the experimental source at AST
  level. The original intercept and constant-label conventions are retained.
- Finite-difference gradient maximum error: 4.15e-09.
- MeanAnchor shared-objective identity error: 1.42e-14.
- All 12 backbone/cohort/shot combinations completed a seed-0 MeanAnchor fit.
- CPSC2018 5-shot was evaluated over all 20 seeds for ECGFounder and MERL,
  for SASM-Ens, MeanAnchor and LP. All six pairs of Macro-AP/Macro-AUC means
  match the paper's two-decimal rounded values. See validation/ for summaries.
- Three small algebraic unit tests passed.
- Every exported NPZ opens with allow_pickle=False; no object arrays are used.
- Text files were scanned for private paths and server connection details.

## Limits

The other cohort/shot combinations have not all been rerun over 20 seeds in
this release-validation step. Complete baseline reproduction, raw-waveform
extraction, source training, exact historical software-environment replay and
published CPU-latency reproduction were not performed. Local timings are not
portable benchmark guarantees. See validation/tested_environment.json for
the environment actually used here.

## Commands

`python examples/verify_assets.py`

`python -m unittest discover -s tests -v`

`python examples/evaluate_cached.py --seeds all`
