# Supplementary Analysis Scripts

These scripts archive the exact post-hoc CPU evaluation, audit, calibration,
timing, demographic-subgroup, and Decision Curve Analysis (DCA) pipelines
that generated `results/supplementary/` and `results/dca_threshold/`.

## Execution Boundary
- Original execution used the upstream Linux Xeon Gold 5418Y workstation caches
  under `/home/wy/ECGFounder/...` (`common.py`, `audit_prepare.py`, `run_dca.py`),
  which include the non-redistributed Chapman development feature matrices and
  full upstream cohort caches.
- Standalone reproduction on the distributed CPSC2018/Ningbo/SPH cached
  benchmarks runs directly via `smoke_check.py`, `examples/verify_assets.py`,
  `examples/verify_cpsc9_protocol.py`, `examples/evaluate_cpsc9.py`, and
  `examples/replay_cpsc9.py`.

## Script Inventory
- `common.py`, `audit_inputs.py`, `audit_prepare.py`: Shared locked helpers,
  demographic conflict audit (`DEMOGRAPHIC_CONFLICT_AUDIT.json`), and cache bridges.
- `source_hg.py`, `supplements.py`, `tail_pipeline.py`: Controlled raw (`h`) vs.
  150-logit (`g`) ablation, bidirectional SPH role-swap, and Chapman-locked
  `DevTemperature` calibration (`temperature.lock.json`).
- `timing.py`, `subgroups.py`: Single-core BLAS-1 latency benchmark and 48-slice
  age/sex subgroup evaluations.
- `run_dca.py`, `summarize_dca.py`, `summarize.py`, `verify_artifacts.py`,
  `package_results.py`: Locked rule A/B thresholding (`global_thresholds.lock.json`),
  DCA aggregation, and SHA-256 artifact verification.
