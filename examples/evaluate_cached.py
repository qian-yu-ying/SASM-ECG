"""Evaluate existing cached episodes; no feature extraction or source training."""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from sasm.cache import load_cohort
from sasm.model import fit_cached_features

ALL_BACKBONES = ["ecgfounder", "merl"]
ALL_COHORTS = ["ningbo", "cpsc2018", "sph"]
ALL_SHOTS = [5, 10]
ALL_METHODS = ["sasm-ens", "meananchor", "lp"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", "--data", dest="data", default=str(Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--backbone", choices=ALL_BACKBONES + ["all"], default="ecgfounder")
    p.add_argument("--cohort", "--dataset", dest="cohort", choices=ALL_COHORTS + ["all"], default="cpsc2018")
    p.add_argument("--shot", default="5", help="5, 10, or all")
    p.add_argument("--seeds", default="0", help="comma-separated seed IDs (0..19), or all for the 20-seed mean")
    p.add_argument("--methods", "--method", dest="methods", default="sasm-ens,meananchor,lp",
                   help="comma-separated subset of {sasm-ens,meananchor,lp}, or all")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    root = Path(args.data)
    backbones = ALL_BACKBONES if args.backbone == "all" else [args.backbone]
    cohorts = ALL_COHORTS if args.cohort == "all" else [args.cohort]
    if args.shot == "all":
        shots = ALL_SHOTS
    elif args.shot in ("5", "10"):
        shots = [int(args.shot)]
    else:
        p.error("--shot must be 5, 10, or all")

    seeds = list(range(20)) if args.seeds == "all" else [int(x) for x in args.seeds.split(",")]
    if not seeds or any(s not in range(20) for s in seeds) or len(set(seeds)) != len(seeds):
        p.error("seed IDs must be distinct integers from 0 to 19")

    methods = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    if any(m not in ALL_METHODS for m in methods):
        p.error(f"invalid method in {methods}; choose from {ALL_METHODS} or 'all'")

    cfg_all = json.loads((root / "configurations.json").read_text(encoding="utf8"))
    refs = json.loads((root / "paper_reference.json").read_text(encoding="utf8"))
    all_rows = []
    all_summaries = []

    with threadpool_limits(limits=1):
        for backbone in backbones:
            cfg = cfg_all[backbone]
            for cohort in cohorts:
                features, rw, rb, query, qy, meta = load_cohort(root, backbone, cohort)
                q_feats = features[query].astype(float)
                for shot in shots:
                    cond_rows = []
                    for seed in seeds:
                        ep_path = root / "cohorts" / cohort / "episodes" / f"shot{shot}_seed{seed:02d}.json"
                        episode = json.loads(ep_path.read_text(encoding="utf8"))
                        support = np.asarray(episode["support"], dtype=int)
                        sy = np.asarray(episode["support_y"], dtype=float)
                        assert not np.intersect1d(support, query).size
                        for method in methods:
                            params = dict(cfg["lp"] if method == "lp" else cfg["sasm"])
                            if method != "lp" and "anchor_lambda_multiplier" in meta:
                                params["lam_vec"] = params["lam"] * np.asarray(meta["anchor_lambda_multiplier"])
                            start = time.perf_counter()
                            w, b, diags = fit_cached_features(features[support], sy, rw, rb, method=method, **params)
                            seconds = time.perf_counter() - start
                            scores = q_feats @ w + b
                            row = dict(
                                backbone=backbone,
                                cohort=cohort,
                                shot=shot,
                                seed=seed,
                                method=method,
                                macro_ap=100 * float(average_precision_score(qy, scores, average="macro")),
                                macro_auc=100 * float(roc_auc_score(qy, scores, average="macro")),
                                fit_seconds=seconds,
                                max_gradient=max(d["max_gradient"] for d in diags),
                            )
                            cond_rows.append(row)
                            all_rows.append(row)
                            print(json.dumps(row), flush=True)
                    for method in methods:
                        group = [r for r in cond_rows if r["method"] == method]
                        s = dict(
                            backbone=backbone,
                            cohort=cohort,
                            shot=shot,
                            method=method,
                            n_seeds=len(group),
                            macro_ap_mean=float(np.mean([x["macro_ap"] for x in group])),
                            macro_auc_mean=float(np.mean([x["macro_auc"] for x in group])),
                        )
                        if sorted(seeds) == list(range(20)):
                            s["paper_rounded_reference"] = refs[backbone][cohort][str(shot)][method]
                            # Reference values are rounded to 0.01 percentage points.
                            errors = {
                                "macro_ap": abs(s["macro_ap_mean"] - s["paper_rounded_reference"]["macro_ap"]),
                                "macro_auc": abs(s["macro_auc_mean"] - s["paper_rounded_reference"]["macro_auc"]),
                            }
                            s["reference_abs_error_percentage_points"] = errors
                            s["reference_pass"] = max(errors.values()) <= 0.0051
                            if not s["reference_pass"]:
                                raise RuntimeError("20-seed rounded reference mismatch: " + json.dumps(s))
                        all_summaries.append(s)

    result = dict(rows=all_rows, summary=all_summaries)
    print(json.dumps({"summary": all_summaries}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
