# SASM-ECG

Source-Anchored Symbolic Matching for few-shot multi-label ECG adaptation.

**v1.4.0 is the current release.** It builds on the v1.2.0 nine-class CPSC2018
reproduction snapshot and the v1.3.0 DCA/locked-threshold release by adding the
complete `results/supplementary/` CSV/JSON bundles, `scripts/supplementary/`
generation/audit scripts, and `scripts/reference_pipelines/` (including
raw-waveform `FullFT`/`LoRA` reference code in `gpu_worker.py`). The
nine-class snapshot includes
6,877 CPSC2018 records, a fixed 2,755-record support pool and a 4,122-record
query set. The labels, feature row order, support episodes, source mappings,
per-class regularization and reference results belong to this same protocol.

This is a cached-feature reproduction package. It does not claim to regenerate
every manuscript experiment from raw waveforms. SASM-authored code uses MIT;
third-party data and model terms remain separate (see THIRD_PARTY_NOTICES.md).

## Install and obtain the matching assets

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

From the **v1.4.0 Release**, download `SASM-ECG_v1.4.0_code.zip` and check it
against `SHA256SUMS.txt` before extracting. This code archive does not contain
raw waveforms, encoder checkpoints or the ignored bulk feature arrays. The
cached result exports and fixed episode metadata are included in the repository;
the release also attaches the complete DCA/threshold result archive. The v1.2.0
tag remains available as the preceding nine-class snapshot. The repository
itself also contains the matching cached arrays and source-head assets used by
the nine-class checks; no raw waveforms, private credentials or full encoder
checkpoints are distributed.

```bash
git clone --branch v1.4.0 --depth 1 https://github.com/qian-yu-ying/SASM-ECG.git
cd SASM-ECG
```

```bash
python examples/verify_assets.py
python examples/verify_cpsc9_protocol.py
python -m unittest discover -s tests -v
python smoke_check.py
```

## Reproduce the nine-class CPSC2018 results

The primary source mapping is Normal -> NORM. STE has no source anchor:
its source column is zero and its class-specific lambda is zero. Alpha remains
0.1 for SASM. Direct and Direct-LP use the fitted LP score on STE. This fallback
is not zero-shot prediction. LP alpha is 100 for ECGFounder and 10 for MERL.
No parameter is selected using the CPSC2018 query scores.

```bash
# One episode, all supported frozen baselines and scaling controls:
python examples/evaluate_cpsc9.py --backbone all --methods all

# Both backbones, 5/10-shot, all 20 support seeds; three primary methods:
python examples/evaluate_cpsc9.py --backbone all --shot all --seeds all --output outputs/cpsc9_primary.json

# All supported frozen methods, including fullpool source and native controls:
python examples/evaluate_cpsc9.py --backbone all --shot all --seeds all --methods all --output outputs/cpsc9_frozen.json

# Prespecified Normal -> SR mapping sensitivity, reported separately:
python examples/evaluate_cpsc9.py --backbone all --shot all --seeds all --normal SR --output outputs/cpsc9_SR.json
```

The dedicated evaluator checks per-seed macro AP and AUROC (all9 and matched8)
against archived nine-class results with absolute tolerance 0.00005 in fraction
units (0.005 percentage points). STE-only ranking metrics are reported as
numerical diagnostics: tiny solver/BLAS differences can reorder close scores.
For independent scoring from the actual archived coefficients, without refitting:

```bash
python examples/replay_cpsc9.py --backbone all --shot all --seeds all --output outputs/cpsc9_replay.json
```

The companion assets include the fitted coefficients for the primary NORM mapping.
Replay and fresh convex fitting are explicitly different verification paths. It reports all9, matched8, native_shared7 and STE slices separately.
Native controls are defined for ECGFounder only. `Native-source-scaled` in the
nine-class protocol uses one anchor divided by the mean source SD, as recorded
in that experiment; it is not described as ten independently fitted native heads.

| CPSC2018 Macro-AP (%) | ECGFounder 5-shot | ECGFounder 10-shot | MERL 5-shot | MERL 10-shot |
|---|---:|---:|---:|---:|
| LP | 62.73 | 70.82 | 47.11 | 54.57 |
| SASM-Ens | 69.53 | 73.44 | 57.07 | 59.86 |
| MeanAnchor | 69.50 | 73.42 | 57.03 | 59.80 |
| Direct-LP | 68.49 | 72.81 | 57.10 | 60.19 |
| FullPool-SASM | 69.53 | 73.32 | 58.01 | 60.57 |

These are 20-seed means with a fixed query set. They describe support-sampling
variability, not 20 independent patient cohorts. Roles are disjoint by waveform
group; waveform groups are not claimed to be identified patients. The protocol
is retrospective and records prior result exposure.

## Other cohorts and source-head refitting

```bash
# Three cohorts, two backbones, 5/10-shot, 20 seeds, LP/SASM-Ens/MeanAnchor:
python examples/evaluate_cached.py --backbone all --cohort all --shot all --seeds all --output outputs/core_all.json

# Route-B evaluator for the unchanged Ningbo/SPH protocols only:
python examples/evaluate_route_b.py --suite fullpool --backbone all --cohort all --shot all --seeds all
python examples/verify_source_heads.py
python examples/refit_source_head.py --backbone ecgfounder --output outputs/source_refit.npz
```

`evaluate_cached.py` also applies the nine-class STE penalty mask for CPSC2018.
It rejects incompatible CPSC cache sizes/protocols. `evaluate_route_b.py` routes
CPSC users to the dedicated nine-class command instead of silently using another
label panel. The source fullpool comprises 17,418 PTB-XL v1.0.3 fold-1--8 records.
Adding `--run-cv` to source refitting reruns source-only selection; target Chapman
features are not included, so a complete target hyperparameter search is not
claimed here. Target fits are local CPU operations on frozen features; timing
excludes encoder extraction and file I/O.

## Results, provenance and boundaries

### DCA and locked thresholds

Release v1.3.0 includes `results/dca_threshold/` exports from a retrospective
analysis of existing frozen predictions. The DCA grid is
`p_t = {0.02, 0.05, 0.10, 0.15, 0.20}` and reports raw and Chapman-development
temperature versions for LP, Direct-LP, intercept-refitted Direct-LP,
SASM-Ens and MeanAnchor. The locked-threshold export contains a global threshold
selected from Chapman development negatives and a per-episode support-set
negative 95th percentile.

External query labels are read only after the threshold locks are written. The
checks report coefficient replay error 0 and exact decision invariance under the
paired temperature representation. These are retrospective record-level results
with support-seed uncertainty conditional on fixed query sets; they are not
prospective clinical deployment validation. The support-set quantile rule does
not guarantee 95% specificity on an external cohort, and DCA values depend on
the chosen risk threshold and cohort prevalence. See
`docs/DCA_THRESHOLD_SCOPE.md` and the release asset for complete tables.

- `results/`: current manuscript figure/table episode exports, plus CPSC9 primary
  per-seed and per-label metrics. These exports include methods beyond executable
  entry points and must not be mistaken for raw-waveform rerun capability.
- `provenance/cpsc9_protocol.lock.json`: fixed configurations and prior-exposure
  statement; `metrics.lock.json`: metric definitions.
- `data/cohorts/cpsc2018/`: labels, roles, record hashes, waveform groups and 40
  fixed support episodes. `data/mappings/cpsc2018.npz`: nine-column concept map.
- `validation/`: actual release checks, with tolerances and scope.
- `docs/REPRODUCTION_SCOPE.md`: what is runnable and what is results-only.

### Supplementary revision bundles and reference pipelines (v1.4.0)

- `results/supplementary/`: Exported summary CSVs and lock/audit JSONs across
  all six supplementary analyses—controlled `h`-vs-`g` PTB-XL ablation
  (`hg_*.csv`), bidirectional SPH-dev role swap (`swap_*.csv`), Chapman-locked
  `DevTemperature` calibration (`calibration_*.csv`, `temperature.lock.json`),
  pinned Xeon Gold 5418Y BLAS-1 timing (`timing_*.csv`, `timing_environment.json`),
  48-slice age/sex subgroups (`subgroup_*.csv`, `DEMOGRAPHIC_CONFLICT_AUDIT.json`),
  and the DCA/locked-threshold summary files (byte-identical to `results/dca_threshold/`).
- `scripts/supplementary/`: Archived generator and audit scripts for the
  supplementary bundles (require upstream full cohort caches and Chapman
  development features).
- `scripts/reference_pipelines/`: Archived full-study training, control,
  ablation, timing, and bootstrap/Holm scripts, including raw-waveform
  `FullFT`/`LoRA` workers (`gpu_worker.py`), requiring independent upstream public
  datasets and pretrained checkpoints.

No raw ECG waveforms, private credentials, remote workers, Chapman development
feature matrices, or full third-party encoder checkpoints are distributed.
Standalone execution covers cached-feature adaptation (`smoke_check.py` and
`examples/`). Integrity hashes verify package identity; they do not certify
clinical equivalence of every mapped label.
