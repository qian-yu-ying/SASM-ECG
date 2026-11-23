"""Replay archived CSV aggregates only; no fitting, network or research imports."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require_close(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"Nonfinite aggregate: {label}")
    if abs(actual - expected) > TOL:
        raise ValueError(f"Aggregate mismatch {label}: {actual} != {expected}")
    return abs(actual - expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    a = ROOT / "results/v1.5.1_adapters"
    b = ROOT / "results/v1.5.2_micro"
    episodes = rows(a / "metric_episodes.csv")
    main_groups = defaultdict(list)
    for row in episodes:
        if row["calibration"] == "Raw":
            main_groups[tuple(row[k] for k in ("backbone", "cohort", "shot", "method"))].append(row)
    checks = 0
    max_error = 0.0
    condition_values = defaultdict(list)
    for row in rows(a / "main_12cond_summary.csv"):
        key = tuple(row[k] for k in ("backbone", "cohort", "shot", "method"))
        group = main_groups[key]
        if len(group) != 20 or {int(x["seed"]) for x in group} != set(range(20)):
            raise ValueError(f"Expected 20 unique support seeds: {key}")
        for metric in ("ap", "auc"):
            value = fmean(float(x[metric]) for x in group)
            max_error = max(max_error, require_close(value, float(row[metric + "_mean"]), str(key) + metric))
            checks += 1
            if metric == "ap":
                condition_values[row["method"]].append(value)
    overall = {}
    for method, values in sorted(condition_values.items()):
        if len(values) != 12:
            raise ValueError(f"Expected 12 conditions for {method}: {len(values)}")
        overall[method] = 100 * fmean(values)

    micro_groups = defaultdict(list)
    for row in rows(b / "module1_episode_metrics.csv"):
        key = tuple(row[k] for k in ("method", "ste_policy", "calibration", "evaluation_scope"))
        micro_groups[key].append(row)
    for row in rows(b / "module1_scope_summary.csv"):
        key = tuple(row[k] for k in ("method", "ste_policy", "calibration", "evaluation_scope"))
        selected = [x for x in micro_groups[key] if row["scope"] == "Overall" or x["backbone"] == row["scope"]]
        conditions = defaultdict(list)
        for episode in selected:
            conditions[tuple(episode[k] for k in ("backbone", "cohort", "shot"))].append(episode)
        if len(conditions) != int(row["conditions"]):
            raise ValueError(f"Condition count mismatch: {key}, {row['scope']}")
        for ck, group in conditions.items():
            if len(group) != 20 or {int(x["seed"]) for x in group} != set(range(20)):
                raise ValueError(f"Expected 20 unique micro-revision support seeds: {key}, {ck}")
        for metric in ("ap", "auc", "ece", "brier", "nll", "sensitivity95", "ppv95"):
            value = fmean(fmean(float(x[metric]) for x in group) for group in conditions.values())
            max_error = max(max_error, require_close(value, float(row[metric]), str(key) + row["scope"] + metric))
            checks += 1
    report = {
        "all_passed": True,
        "aggregate_checks": checks,
        "max_abs_fraction_error": max_error,
        "tolerance_fraction": TOL,
        "main_12_condition_ap_percent": overall,
        "scope_note": "CPSC_matched8_other_all includes all 12 conditions; only CPSC excludes STE.",
        "boundary": "CSV aggregation replay only; no fitting, waveform processing or bootstrap regeneration.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
