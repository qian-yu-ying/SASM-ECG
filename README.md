# SASM-ECG

Source-Anchored Symbolic Matching for few-shot multi-label ECG adaptation.

**v1.5.2 is the current code/results archival release.** It preserves the
v1.4.1 cached-feature benchmark and adds the v1.5.1 shared-prior multi-label
adapter experiments and v1.5.2 solver, development-grid, calibration-CI,
pathology-DCA and interface-forensics exports. No experiments or historical
result values were changed by this archival update.

Start with [the v1.5 reproduction guide](docs/V1_5_REPRODUCTION.md). A lightweight,
standard-library-only check recomputes the released aggregate metrics:

```bash
python examples/replay_v15_summaries.py --output outputs/v15_summary_replay.json
python examples/verify_release_manifest.py
```

The archived research scripts and the directly runnable summary replay have
different input requirements; the guide identifies those boundaries.

This is a cached-feature reproduction package. It does not claim to regenerate
every manuscript experiment from raw waveforms. SASM-authored code uses MIT;
third-party data and model terms remain separate (see THIRD_PARTY_NOTICES.md).

## Authors and Manuscript Correspondence

Companion software and cached benchmark package for *"Source-Anchored Symbolic
Matching: Few-Shot Multi-Label ECG Adaptation Across Heterogeneous Diagnostic
Vocabularies"* (BSPC manuscript companion archive):

- **Authors**: Chenxi Yang (`20242365@stu.cqu.edu.cn`), Guolan Deng (`dengguolan0201@163.com`), and Yi Wang (`yiwang@cqu.edu.cn`, corresponding author), College of Computer Science, Chongqing University.
- **Repository Hosting**: Official author-maintained release archive under GitHub account `qian-yu-ying` (`https://github.com/qian-yu-ying/SASM-ECG`); see `CITATION.cff` and `LICENSE`.

## Install and obtain the matching assets

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

From the **v1.5.2 Release**, download `SASM-ECG_v1.5.2_code.zip` and check it
against `SHA256SUMS.txt` before extracting. This code archive does not contain
raw waveforms, encoder checkpoints or the ignored bulk feature arrays. The
cached result exports and fixed episode metadata are included in the repository;
the release also attaches the complete DCA/threshold result archive. The v1.2.0
tag remains available as the preceding nine-class snapshot. The repository
itself also contains the matching cached arrays and source-head assets used by
the nine-class checks; no raw waveforms, private credentials or full encoder
checkpoints are distributed.

```bash
git clone --branch v1.5.2 --depth 1 https://github.com/qian-yu-ying/SASM-ECG.git
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

### Supplementary revision bundles and reference pipelines (v1.4.1)

- `results/supplementary/`: Exported summary CSVs and lock/audit JSONs across
  all six supplementary analyses—controlled `h`-vs-`g` PTB-XL ablation
  (`hg_*.csv`), bidirectional SPH-dev role swap (`swap_*.csv`), Chapman-locked
  `DevTemperature` calibration (`calibration_*.csv`, `temperature.lock.json`),
  pinned Xeon Gold 5418Y BLAS-1 timing (`timing_*.csv`, `timing_environment.json`),
  48-slice age/sex subgroups (`subgroup_*.csv`, `DEMOGRAPHIC_CONFLICT_AUDIT.json`),
  DCA/locked-threshold summary files (byte-identical to `results/dca_threshold/`), and unmatched CPSC2018 `STE` $\tau=10.1$ per-seed & Holm paired statistics (`ste_episodes.csv`, `ste_summary.csv`, `ste_paired_effects.csv`, `ste_protocol.lock.json`, verified offline by `tests/test_ste_supplementary.py`).
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

## Modern adapter and micro-revision archive (v1.5.2)

- `scripts/v1.5.1_adapters/`, `results/v1.5.1_adapters/`: four explicitly
  named ECG/BCE adapter extensions, configuration locks, per-episode metrics,
  paired tests, query-cluster intervals, role swap, clinical summaries and timings.
- `scripts/v1.5.2_micro/`, `results/v1.5.2_micro/`: same-solver attribution,
  expanded development-grid analysis, paired calibration intervals, pathology
  DCA, interface forensics and Figure 5 source data/rendering script.
- `provenance/v1.5.1/`, `provenance/v1.5.2/`: development-grid outputs,
  bootstrap draws, condition-level exports and original execution manifests.
- `scripts/historical_runtime/`: preserved local runtime dependencies whose
  hashes match the historical dependency manifest.

The full new extension payload is also attached as
`SASM_v1.5.2_extensions_archive.zip`. Complete v1.5 per-record prediction and
model caches, Chapman development feature matrices, raw waveforms and foundation
checkpoints are not included. The archive scripts retain original host paths;
summary replay is portable and does not load those scripts or contact a server.
