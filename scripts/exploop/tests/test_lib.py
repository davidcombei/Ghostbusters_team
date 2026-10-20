"""Unit tests on the real run files (read-only): argv model, one-factor validator, log parsers,
verdict rules, eval-profile expansion.  Run: python -m unittest discover scripts/exploop/tests"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exploop_lib as L  # noqa: E402
import supervisor as S  # noqa: E402

REPO = HERE.parents[2]
EXP = REPO / "SLS_setup/exp"
B2F = EXP / "B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623"
B3A = EXP / "B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846"
B2E = EXP / "B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925"
WAVE = json.loads((REPO / "exploop/B3/wave.json").read_text())
KNOWN = L.trainer_flags([REPO / "SLS_setup/b3wave/train.py"])


@unittest.skipUnless(B2F.is_dir() and B3A.is_dir(), "needs the B2f / B3a run dirs")
class TestArgvAndValidator(unittest.TestCase):
    def setUp(self):
        self.base = L.config_argv(B2F)
        self.b3a = L.config_argv(B3A)

    def test_parse_negative_value_is_not_a_flag(self):
        d = L.argv_dict(self.base)
        self.assertEqual(d["--loudness_norm_lufs"], ["-23"])
        self.assertEqual(d["--loudness_norm"], [])
        self.assertEqual(d["--aug_noise_dirs"], ["../data/augmentation_train/rnnoise", "../data/augmentation_train/esc50"])

    def test_b3a_is_one_coupled_factor_vs_b2f(self):
        self.assertEqual(L.changed_flags(self.base, self.b3a),
                         ["--batch_size", "--consistency", "--consistency_lambda", "--consistency_pairs"])
        design = {"kind": "config", "letter": "a", "track": "B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234"}
        self.assertEqual(L.validate_design(WAVE, self.base, self.b3a, design, set(), KNOWN), [])

    def apply(self, change, track="B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"):
        ne = (change.get("set", {}).get("--num_epochs") or ["8"])[0]
        return L.apply_change(self.base, change, {"--track": [track], "--keep_epochs": [ne], "--keep_topk_by_wf1": ["7"]})

    def test_single_flag_ok(self):
        cand = self.apply({"set": {"--aug_codec_profiles": ["random"]}, "remove": []})
        d = {"kind": "config", "letter": "b", "track": "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"}
        self.assertEqual(L.validate_design(WAVE, self.base, cand, d, set(), KNOWN), [])

    def test_two_groups_rejected(self):
        cand = self.apply({"set": {"--aug_codec_profiles": ["random"], "--specaug": []}, "remove": []})
        d = {"kind": "config", "letter": "b", "track": "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"}
        errs = L.validate_design(WAVE, self.base, cand, d, set(), KNOWN)
        self.assertTrue(any("more than one factor" in e for e in errs), errs)

    def test_sampler_group_with_ce_weights_is_one_factor(self):
        cand = self.apply({"set": {"--sampler": ["balanced"], "--sampler_lang_balance": [], "--sampler_bona_frac": ["0.5"],
                                   "--ce_weights": ["0.32", "0.68"]}, "remove": []})
        d = {"kind": "config", "letter": "b", "track": "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"}
        self.assertEqual(L.validate_design(WAVE, self.base, cand, d, set(), KNOWN), [])
        self.assertEqual(L.argv_dict(cand)["--ce_weights"], ["0.32", "0.68"])

    def test_fixed_forbidden_epochs_typo_seed_track(self):
        d = {"kind": "config", "letter": "b", "track": "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"}
        for change, needle in (({"set": {}, "remove": ["--loudness_norm"]}, "fixed flag --loudness_norm"),
                               ({"set": {"--label_smoothing": ["0.05"]}, "remove": []}, "forbidden"),
                               ({"set": {"--num_epochs": ["12"]}, "remove": []}, "--num_epochs must be"),
                               ({"set": {"--specaugg": []}, "remove": []}, "not a flag of the trainer"),
                               ({"set": {"--seed": ["2345"]}, "remove": []}, "--seed may change only")):
            errs = L.validate_design(WAVE, self.base, self.apply(change), d, set(), KNOWN)
            self.assertTrue(any(needle in e for e in errs), (change, errs))
        bad = dict(d, track="B3b_bad")
        errs = L.validate_design(WAVE, self.base, self.apply({"set": {"--specaug": []}, "remove": []}, track="B3b_bad"), bad, set(), KNOWN)
        self.assertTrue(any("must match" in e for e in errs), errs)
        used = L.validate_design(WAVE, self.base, self.apply({"set": {"--specaug": []}, "remove": []}), d,
                                 {"B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s1234"}, KNOWN)
        self.assertTrue(any("already used" in e for e in used), used)

    def test_seed_repeat(self):
        cand = self.apply({"set": {"--seed": ["2345"]}, "remove": []}, track="B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s2345")
        d = {"kind": "seed_repeat", "letter": "b", "track": "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_x_s2345"}
        self.assertEqual(L.validate_design(WAVE, self.base, cand, d, set(), KNOWN), [])


@unittest.skipUnless(B3A.is_dir(), "needs the B3a run dir")
class TestParsers(unittest.TestCase):
    def test_train_log(self):
        rows = L.parse_train_log(B3A / "logs/train.log")
        self.assertEqual(sorted(rows), list(range(1, 9)))
        self.assertAlmostEqual(rows[8]["wf1"], 92.14)
        self.assertAlmostEqual(rows[6]["wf1"], 92.87)
        self.assertAlmostEqual(rows[8]["dev_loss"], 0.172235)

    def test_tqdm_batch_per_s(self):
        log = sorted(REPO.glob("SLS_setup/B3a_*.log"))[0]
        text = L.head_text(log, 3_000_000)
        p = L.parse_tqdm(text)
        self.assertIsNotNone(p, "the B3a log uses batch/s; the parser must read it (the B3 watcher grepped it/s)")
        self.assertEqual(p["n"], 9474)
        self.assertGreater(p["rate_per_s"], 0)
        p2 = L.parse_tqdm("Training:  10%|#  | 10/100 [00:20<03:00,  2.00s/batch]")
        self.assertAlmostEqual(p2["rate_per_s"], 0.5)

    def test_final_epoch_and_model_flags(self):
        self.assertEqual(L.final_epoch(B3A), 8)
        self.assertEqual(L.model_flags(B3A), ["--ssl_name", "Qwen/Qwen3-ASR-1.7B-hf", "--n_layers", "24",
                                              "--loudness_norm", "--loudness_norm_lufs", "-23"])
        if B2E.is_dir():
            self.assertEqual(L.model_flags(B2E), ["--ssl_name", "Qwen/Qwen3-ASR-1.7B-hf", "--n_layers", "24"])

    def test_completion_marker(self):
        log = sorted(REPO.glob("SLS_setup/B3a_*.log"))[-1]
        self.assertIn("Experiment saved to:", L.tail_bytes(log, 20000))


class TestVerdict(unittest.TestCase):
    rules = WAVE["rules"]

    def v(self, run, base, ci_low, clean_drop=0.0):
        ci = {"delta": run - base, "ci_low": ci_low, "ci_high": ci_low + 1}
        return L.decide({"value": run, "clean": 97.6 - clean_drop}, {"value": base, "clean": 97.6}, ci, self.rules)

    def test_table(self):
        self.assertEqual((self.v(92.6, 91.98, 0.1)["decision"], self.v(92.6, 91.98, 0.1)["verdict"]), ("promote", "positive"))
        r = self.v(92.2, 91.98, -0.3)
        self.assertEqual((r["decision"], r["new_best"]), ("revisit", True))
        self.assertTrue(r["verdict"].startswith("null"))
        self.assertEqual(self.v(92.2, 91.98, 0.05)["decision"], "promote")      # new best + CI excl. 0
        self.assertTrue(self.v(92.2, 91.98, 0.05)["verdict"].startswith("null"))  # but inside the +-0.5 band
        self.assertEqual(self.v(91.0, 91.98, -2)["decision"], "drop")
        self.assertEqual(self.v(91.0, 91.98, -2)["verdict"], "negative")
        self.assertEqual(self.v(92.8, 91.98, 0.2, clean_drop=0.8)["decision"], "revisit")
        self.assertEqual(self.v(92.8, 91.98, -0.1)["verdict"], "above band, CI includes 0")

    def test_keep_best_screen(self):
        rules = dict(self.rules, keep_best=True)

        def v(run, base, ci_low):
            return L.decide({"value": run, "clean": 97.6}, {"value": base, "clean": 97.6},
                            {"delta": run - base, "ci_low": ci_low, "ci_high": ci_low + 1}, rules)
        self.assertEqual((v(93.15, 92.14, 0.43)["decision"], v(93.15, 92.14, 0.43)["new_best"]), ("promote", False))
        self.assertEqual((v(92.55, 91.98, -0.01)["decision"], v(92.55, 91.98, -0.01)["new_best"]), ("revisit", False))
        self.assertEqual(v(91.0, 91.98, -2)["decision"], "drop")
        self.assertEqual(v(93.15, 92.14, 0.43)["verdict"], "positive")      # verdicts do not depend on keep_best


class TestProfileExpansion(unittest.TestCase):
    def test_sls_profile_matches_the_skill(self):
        prof = json.loads((HERE.parent / "profiles/sls_local17.json").read_text())
        ctx = {"py": "PY", "run": "SLS_setup/exp/R", "run_rel": "exp/R", "k": "5", "N": "8", "ckpt_final": "SLS_setup/exp/R/ckpt/epoch_8_x.pth",
               "model_flags": ["--ssl_name", "S", "--n_layers", "24"], "names": WAVE["eval"]["names"], "label": "B3a",
               "merge_label": "B3a-top5", "final_label": "B3a-ep8", "best_merge": "B/merge_top5", "best_final": "B/epoch_8",
               "best_merge_label": "B2f-top5", "best_final_label": "B2f-ep8", "ci": "2000", "all_dirs": ["a", "b"], "all_labels": ["A", "B"]}
        cmds = {s["name"]: S.expand_list(s["cmd"], ctx) for s in prof["steps"]}
        self.assertEqual(cmds["select_topk"], ["PY", "../scripts/select_topk.py", "--run", "exp/R", "--k", "5"])
        me = cmds["merge_eval"]
        self.assertEqual(me[:4], ["PY", "scripts/local_eval.py", "--merge_list", "SLS_setup/exp/R/top5_by_wf1/merge_list.txt"])
        self.assertEqual(me[me.index("--names") + 1:], WAVE["eval"]["names"])     # 17 separate args, never one quoted string
        self.assertEqual(len(WAVE["eval"]["names"]), 17)
        pe = cmds["progress_scores"]
        self.assertIn("--model_merging", pe)
        self.assertEqual(pe[pe.index("--score_path") + 1], "exp/R/model_merging_top5/scores.txt")
        self.assertEqual(cmds["ci_merge"][-2:], ["--ci", "2000"])
        self.assertEqual(cmds["compare_all"], ["PY", "scripts/compare_evals.py", "a", "b", "--labels", "A", "B"])


if __name__ == "__main__":
    unittest.main()
