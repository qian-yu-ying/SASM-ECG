# SASM — Cached-feature research release

Source-Anchored Symbolic Matching: Few-Shot Multi-Label ECG Adaptation Across Heterogeneous Diagnostic Vocabularies

This release provides the core method and a limited evaluation path using
precomputed source classifiers, frozen features and fixed support/query episodes.
It does not contain the complete experimental framework. Raw ECG preprocessing,
backbone inference or training, source-head training, hyperparameter searches,
full baseline suites and experiment orchestration are not included.

## Installation and data

Use Python 3.12 and install `pip install -r requirements.txt`.
Extract the companion asset archive into this directory so it creates
`data/features/` and `data/source_heads/`. Large numeric assets are intended for
a GitHub Release or a data archive, not a normal Git source commit.

The package includes 10 source heads per backbone, their standardization
statistics, the affine ECGFounder projection, aligned raw-coordinate heads for
all three cohorts, and the records required by the 20 fixed 5-/10-shot episodes.
These are cached records used by the released benchmark episodes, not complete
copies of the original clinical databases. Excluded and unused records are not
included. Exported indices are remapped; no new splits or episodes are sampled.

## Minimal cached-data evaluation

`python examples/evaluate_cached.py`

This evaluates ECGFounder, CPSC2018, 5-shot, seed 0, for SASM-Ens, MeanAnchor
and LP. A single seed is not the published 20-seed mean.

For the corresponding 20-seed aggregate:

`python examples/evaluate_cached.py --backbone ecgfounder --cohort cpsc2018 --shot 5 --seeds all --output outputs/cpsc2018_5shot.json`

Other supported choices are `merl`, `ningbo`, `sph`, and `--shot 10`.
The main SASM parameters are alpha=0.1 and lambda=10. LP uses alpha=100
for ECGFounder and alpha=10 for MERL. Configuration records are included.
Timing is measured locally after feature loading; it is not a reproduction
of a hardware-independent latency or a guaranteed tenfold speedup.

## Protocol and mathematical scope

Support features determine the target mean and population standard deviation,
with 1e-6 added to the standard deviation, as in the implementation.
Five-/ten-shot refers to sampling positives per category and taking their union;
multilabel co-occurrence can produce more positives per label in that union.
The shared target objective is binary cross-entropy summed over labels and
divided by support count, plus alpha/n times squared weight norm and lambda/n
times squared distance to the anchor. The intercept is free.

ECGFounder source heads consume standardized native linear scores. The
released dense affine projection converts them to pooled-feature coordinates.
MERL source heads consume standardized pooled features directly. Cohort-specific
concept selection is supplied in the mapping files. No automatic clinical alias
discovery or unreviewed string-matching comparator is implemented here.

MeanAnchor averages aligned anchors before one shared-parameter fit. Its
objective identity is not an identity with the predictions of separately fitted
ensemble members. These cached partitions retain the study's existing
exploratory/development history; this release creates no new blinded validation.

## Verification and release scope

See `VERIFICATION.md` for the checks actually performed, and
`data/ASSET_MANIFEST.json` for integrity hashes. `data/paper_reference.json`
contains rounded published values for comparison, not substitute results.
No promise of complete Table 2 reproduction is made for omitted methods.

See `THIRD_PARTY_NOTICES.md` and `LICENSE_NOTICE.md` for attribution and
the distinction between upstream licenses and the author's code license.

## GitHub publishing layout

Upload the contents of the companion code archive to the repository. Attach
the asset ZIP to a GitHub Release and put its actual download link in this
README after the release exists. Do not claim assets are downloadable until
that attachment is published. Extract the asset ZIP inside the code directory;
it supplies data/features/ and data/source_heads/. Verify with:

`python examples/verify_assets.py`

The versioned validation summaries are release checks, not new paper results.
