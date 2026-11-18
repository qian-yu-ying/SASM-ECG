# Release validation — v1.3.0

All checks were performed locally on CPU using already completed experiment
assets. No remote experiment, worker, fitting job or GPU job was started.

- Nine-class CPSC2018 structure: 6,877 records, 2,755 support-pool records,
  4,122 query records, 3,966 query waveform groups and 40 fixed support episodes.
- Eight numerical unit tests passed, including per-class lambda gradients and
  the alpha-only behavior of a zero-lambda unsupported class.
- Fresh primary fitting: two backbones x two shot budgets x 20 support seeds,
  three methods. Per-seed all9/matched8 macro AP/AUROC passed the 0.005 pp gate.
- Archived coefficient replay: 80 episodes x nine methods x three scopes;
  2,160 metric rows (AP and AUROC each). Maximum error was approximately 1.11e-16
  in fraction units. This is scoring verification, not new model fitting.
- Cross-cohort primary seed-0 checks, Ningbo/SPH Route-B seed-0 checks, CPSC SR
  sensitivity seed-0 checks and source-head stationarity checks passed.
- Fresh fits may reorder extremely close STE scores across numerical runtimes.
  STE-only AP is reported separately; exact archived-model consistency is
  verified with replay rather than an unrealistically strict refit equality.

See validation/RELEASE_CHECKS.json for scope and environment, per-condition
validation JSONs for metrics, and docs/REPRODUCTION_SCOPE.md for exclusions.
Package SHA-256 checks certify file identity, not clinical mapping correctness
or absence of backbone pretraining overlap.

The DCA and locked-threshold supplementary exports were checked separately:
240 external episodes were scored, saved-coefficient replay error was 0, the
independent DCA formula check was below 6e-17, and Raw/DevTemperature threshold
decisions were identical. The threshold analysis is conditional on fixed query
cohorts and does not establish prospective clinical performance.
