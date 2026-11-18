"""Offline verification of Table S18 STE sensitivity statistics from per-seed episodes."""
import csv
import json
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "results" / "supplementary"


class STESupplementaryTests(unittest.TestCase):
    def test_ste_protocol_and_summary_parity(self):
        lock = json.loads((SUPP / "ste_protocol.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["settings"]["tau"], 10.1)
        self.assertEqual(lock["settings"]["source_anchor"], 0)
        self.assertEqual(lock["settings"]["seeds"], list(range(20)))

        with (SUPP / "ste_episodes.csv").open(newline="", encoding="utf-8") as f:
            ep = list(csv.DictReader(f))
        with (SUPP / "ste_summary.csv").open(newline="", encoding="utf-8") as f:
            sm = list(csv.DictReader(f))
        self.assertEqual(len(ep), 1440)
        self.assertGreater(len(sm), 0)

        grouped = {}
        for r in ep:
            key = (r["backbone"], r["shot"], r["normal_mapping"], r["method"], r["policy"])
            grouped.setdefault(key, []).append((float(r["ste_ap"]), float(r["all9_ap"]), float(r["matched8_ap"])))
        for row in sm:
            vals = np.asarray(
                grouped[(row["backbone"], row["shot"], row["normal_mapping"], row["method"], row["policy"])]
            )
            self.assertEqual(len(vals), int(row["n"]))
            self.assertAlmostEqual(float(vals[:, 0].mean()), float(row["ste_ap"]), places=12)
            self.assertAlmostEqual(float(vals[:, 1].mean()), float(row["all9_ap"]), places=12)
            self.assertAlmostEqual(float(vals[:, 2].mean()), float(row["matched8_ap"]), places=12)

    def test_ste_paired_wilcoxon_holm_and_bootstrap_parity(self):
        with (SUPP / "ste_episodes.csv").open(newline="", encoding="utf-8") as f:
            ep = list(csv.DictReader(f))
        with (SUPP / "ste_paired_effects.csv").open(newline="", encoding="utf-8") as f:
            pe = list(csv.DictReader(f))
        self.assertEqual(len(pe), 8)

        rng = np.random.default_rng(2026111701)
        computed = []
        for bb in ["ECGFounder", "MERL"]:
            for shot in ["5", "10"]:
                by_seed = {}
                for r in ep:
                    if (
                        r["backbone"] == bb
                        and r["shot"] == shot
                        and r["normal_mapping"] == "NORM"
                        and r["method"] == "SASM-Ens"
                    ):
                        by_seed.setdefault(int(r["seed"]), {})[r["policy"]] = float(r["ste_ap"])
                for ctrl in ["current_alpha0.1", "LP_fallback"]:
                    d = np.array([by_seed[s]["zero_tau10.1"] - by_seed[s][ctrl] for s in range(20)])
                    boot = rng.choice(d, (10000, 20), replace=True).mean(1)
                    computed.append(
                        dict(
                            backbone=bb,
                            shot=int(shot),
                            contrast=f"zero_tau10.1 minus {ctrl}",
                            delta_pp=float(d.mean() * 100),
                            lo_pp=float(np.quantile(boot, 0.025) * 100),
                            hi_pp=float(np.quantile(boot, 0.975) * 100),
                            p=float(wilcoxon(d, method="auto").pvalue),
                        )
                    )
        ps = np.array([r["p"] for r in computed])
        order = np.argsort(ps)
        holm = np.minimum(1.0, np.maximum.accumulate(ps[order] * (len(ps) - np.arange(len(ps)))))
        for pos, idx in enumerate(order):
            computed[idx]["p_holm"] = float(holm[pos])

        for calc, ref in zip(computed, pe):
            self.assertEqual(calc["backbone"], ref["backbone"])
            self.assertEqual(calc["shot"], int(ref["shot"]))
            self.assertEqual(calc["contrast"], ref["contrast"])
            for col in ["delta_pp", "lo_pp", "hi_pp", "p", "p_holm"]:
                self.assertAlmostEqual(calc[col], float(ref[col]), delta=1e-10)


if __name__ == "__main__":
    unittest.main()
