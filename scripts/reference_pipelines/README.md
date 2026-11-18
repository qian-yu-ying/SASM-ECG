# Full-Study Reference Pipeline Scripts

These files archive the exact experiment, control, ablation, timing, and statistical aggregation scripts used to produce the CSV tables under `results/`.

## Requirements & Scope
- These scripts are **reference pipeline code**, not standalone runnable without:
  1. Upstream public datasets (PTB-XL, Chapman--Shaoxing, Ningbo, CPSC2018, SPH) downloaded from PhysioNet / Figshare;
  2. Pre-trained encoder repositories & weights (ECGFounder, MERL) and environment paths configured via `SASM_RUN_DIR`, `SASM_ECGFOUNDER_DIR`, `SASM_PREVIOUS_DIR`, and `SASM_TARGET_FULL_DIR`;
  3. Linux (`fcntl` file locking) for the multi-worker queue scripts (`cpu_worker.py`, `gpu_worker.py`, `controls.py`, `run_ablation.py`).
- Pre-fitted 71-concept PTB-XL source classifiers (`data/source_heads/`) were generated upstream prior to target-side adaptation; standalone target-side cached evaluation runs directly from `smoke_check.py` and `examples/evaluate_cached.py` without these server scripts.

## Inventory
- `core.py`, `prepare_inputs.py`: Locked hyperparameter grids, metric wrappers, and Chapman-dev split indexing.
- `cpu_worker.py`, `merl.py`, `gpu_worker.py`: SASM-Ens, LP, Concat-LP, Native-anchor, FullFT, and LoRA runners.
- `controls.py`, `mean_anchor.py`, `headcount.py`: Prior-integration comparisons, MeanAnchor equivalence, and $m=1..10$ head-count sweeps.
- `ablation_common.py`, `run_ablation.py`, `timing.py`: Concept-permutation & scaling ablations and single-core CPU latency benchmark.
- `bootstrap.py`, `patient_bootstrap.py`, `merl_patient_bootstrap.py`, `finalize.py`, `supplement_finalize.py`: 10,000-seed & 2,000-query-unit bootstrap CIs and Holm-Bonferroni correction.
