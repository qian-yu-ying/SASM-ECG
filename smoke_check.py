"""Fast CPU smoke check (~3-5 s): verifies assets, runs unit tests, and checks seed-0 reference parity."""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sasm.cache import load_cohort
from sasm.model import fit_cached_features


def check_assets(data_root: Path):
    manifest = json.loads((data_root / "ASSET_MANIFEST.json").read_text(encoding="utf8"))
    errors = []
    for item in manifest["files"]:
        rel = Path(item["file"])
        rel_sub = rel.relative_to("data") if rel.parts[0] == "data" else rel
        path = data_root / rel_sub
        if not path.is_file():
            errors.append(f"{item['file']}: missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            errors.append(f"{item['file']}: integrity mismatch")
    if errors:
        raise SystemExit("Asset verification failed:\n" + "\n".join(errors[:10]))
    print(f"[1/3] Asset SHA-256 verification passed ({len(manifest['files'])} files).")


def run_unit_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if not res.wasSuccessful():
        raise SystemExit(1)
    print(f"[2/3] Algebraic unit tests passed ({res.testsRun} tests).")


def check_seed0_meananchor(data_root: Path):
    features, rw, rb, query, qy, meta = load_cohort(data_root, "ecgfounder", "cpsc2018")
    ep = json.loads((data_root / "cohorts/cpsc2018/episodes/shot5_seed00.json").read_text(encoding="utf8"))
    cfg = json.loads((data_root / "configurations.json").read_text(encoding="utf8"))["ecgfounder"]["sasm"]
    cfg["lam_vec"] = cfg["lam"] * np.asarray(meta["anchor_lambda_multiplier"])
    support = np.asarray(ep["support"], dtype=int)
    sy = np.asarray(ep["support_y"], dtype=float)
    with threadpool_limits(limits=1):
        w, b, diags = fit_cached_features(features[support], sy, rw, rb, method="meananchor", **cfg)
    scores = features[query].astype(float) @ w + b
    ap = 100 * float(average_precision_score(qy, scores, average="macro"))
    auc = 100 * float(roc_auc_score(qy, scores, average="macro"))
    grad = max(d["max_gradient"] for d in diags)
    import csv
    with (ROOT / "results/cpsc9_primary_episodes.csv").open(newline="",encoding="utf-8-sig") as stream:
        ref = next(x for x in csv.DictReader(stream) if x["backbone"]=="ECGFounder" and x["normal_mapping"]=="NORM" and x["shot"]=="5" and x["seed"]=="0" and x["method"]=="MeanAnchor" and x["scope"]=="all9")
    assert grad <= 5e-5, f"stationarity failed: {grad}"
    np.testing.assert_allclose(ap,100*float(ref["ap"]),atol=.005,rtol=0)
    np.testing.assert_allclose(auc,100*float(ref["auc"]),atol=.005,rtol=0)
    print(f"[3/3] Seed-0 MeanAnchor check passed (ECGFounder CPSC2018 5-shot: Macro-AP={ap:.4f}%, Macro-AUROC={auc:.4f}%, max_grad={grad:.2e}).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--skip-asset-hash", action="store_true", help="Skip SHA-256 check across all manifest-listed asset files")
    args = p.parse_args()
    if not args.skip_asset_hash:
        check_assets(args.data_root)
    else:
        print("[1/3] Skipped SHA-256 asset verification (--skip-asset-hash).")
    run_unit_tests()
    check_seed0_meananchor(args.data_root)
    print("SMOKE CHECK PASSED.")


if __name__ == "__main__":
    main()
