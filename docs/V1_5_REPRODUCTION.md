# v1.5.2 reproduction and access guide

## What was archived

This release adds the completed v1.5.1/v1.5.2 extension assets to v1.4.1.
Original experiment scripts and numerical exports are copied byte-for-byte;
the release does not retune models, change measurements, or rerun experiments.
`provenance/v1.5.2_release_payload.json` records each copied file and SHA-256.
The original execution manifests contain historical absolute paths and hashes.

| Material | Repository path | Access/reproduction scope |
|---|---|---|
| Four shared-prior ECG/BCE adapter implementations | `scripts/v1.5.1_adapters/` | Archived research code; original runtime layout and complete input caches required |
| Adapter configurations, episode metrics, paired tests, calibration/DCA, rules, demographic and timing summaries | `results/v1.5.1_adapters/` | Public tabular/JSON exports; aggregate summaries replayable without model fitting |
| Same-solver attribution, expanded dev grid, calibration CI, pathology DCA and interface forensics | `results/v1.5.2_micro/` | Public outputs, locks, witnesses and Figure 5 source data |
| Micro-revision and figure-generation source | `scripts/v1.5.2_micro/` | Archived research code; see input boundary below |
| Grid outputs, bootstrap draws and condition exports | `provenance/v1.5.1/`, `provenance/v1.5.2/` | Public numeric provenance; not complete per-record prediction/model caches |
| Historical Python runtime dependencies | `scripts/historical_runtime/` | Preserve the paths relative to the historical ROOT when reconstructing the original layout |

## Portable numerical summary replay

No Python packages beyond the standard library, network access, model weights,
or feature extraction are needed for these two commands:

```bash
python examples/replay_v15_summaries.py --output outputs/v15_summary_replay.json
python examples/verify_release_manifest.py
```

The replay checks all 12-condition AP/AUROC means for the v1.5.1 main grid
and the v1.5.2 scope summaries against the archived per-episode rows. It also
checks condition/seed cardinalities. This verifies aggregation/transcription,
not the original waveform-to-prediction computation, bootstrap validity, or
prospective clinical utility. AP, AUROC, ECE, sensitivity and PPV are stored
as fractions in these CSVs; multiply by 100 for percentage display.

`CPSC_matched8_other_all` means all 12 conditions are still included, with
STE omitted only from the CPSC endpoint panel. It is not a CPSC-only mean.
The main-grid, pure-proximity and development-selected configurations retain
their distinct method/STE-policy identifiers; none replaces the other.

## Full refitting and statistical regeneration

Legacy cached-feature commands under `examples/` remain available as described
in `REPRODUCTION_SCOPE.md`. The new archived research scripts are not portable
entry points: they retain original absolute ROOT/OLD/DCA/V151/RUN paths and
historical Linux CPU-affinity settings. Read them before execution, reconstruct
the documented input layout in a separate output directory, and supply the
necessary source/target feature arrays, episode indices, source heads and
prediction/model caches. Importing some research modules creates their output
directory; the portable replay does not import them.

The original observed extension environment is recorded in
`AUDIT_MANIFEST_v1.5.1.json` and `AUDIT_MANIFEST_v1.5.2.json` under their
respective result directories: Python 3.10.22, NumPy 1.26.4, SciPy 1.13.1,
pandas 2.2.3, scikit-learn 1.6.1 and PyTorch 2.8.0+cu128, with GPU visibility
disabled and one BLAS thread per worker. PyTorch's build suffix does not mean
these archived extension experiments used a GPU. This is the recorded
environment, not a claim that the full experiment suite has been rerun on
every platform supported by the legacy package.

Raw waveforms and foundation checkpoints must be obtained from their original
providers under their respective terms. Chapman development feature matrices
and the complete v1.5 fitted per-record prediction/model caches are not
redistributed. Consequently, the released small archive alone is insufficient
to regenerate every full-cohort fit, bootstrap interval or waveform-level
interface comparison. Original manifests identify these dependencies and
document unresolved historical probability-cache/clipping provenance.
No new guarantee of patient identity isolation or pretraining decontamination
is introduced by publishing this release.

## Download choices and checksums

- `git clone --branch v1.5.2 --depth 1 https://github.com/qian-yu-ying/SASM-ECG.git`
  obtains the complete tracked repository, including the inherited cached assets.
- `SASM-ECG_v1.5.2_code.zip` is a code/results snapshot excluding the legacy
  `data/` numeric arrays. Clone the tag for the legacy executable feature caches.
- `SASM_v1.5.2_extensions_archive.zip` contains the new scripts, results and
  provenance directories only; it does not replace the legacy benchmark assets.
- `SHA256SUMS.txt` checks the downloadable archives; `RELEASE_MANIFEST.json`
  checks tracked files, excluding itself. Hashes identify bytes and are not
  independent certification of scientific claims.
