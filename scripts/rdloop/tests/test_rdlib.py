"""Unit tests of scripts/rdloop/rdlib.py (python 3.8 compatible; CPU, seconds).
    python -m unittest scripts/rdloop/tests/test_rdlib.py -v        (from the repo root)"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rdlib as R  # noqa: E402

L = R.L
CYCLE = {"cycle": "T9", "module": "t9wave", "script_prefix": "t9_", "levers": {"min": 2, "max": 3},
         "limits": {"max_extra_vram_gib": 4.0, "max_epoch_time_frac": 1.3, "max_loc_per_lever": 400},
         "smoke": {"max_minutes": 30, "margin_minutes": 5, "s0_minutes": 4.5, "overfit_minutes": 2, "evalpath_minutes": 2}}


def lever(i, **kw):
    lv = {"id": f"lever_{i}", "title": "t", "category": "augmentation",
          "flags": [{"name": f"--new_flag_{i}", "type": "bool", "default": "False", "choices": [], "help": "h"}],
          "default_equivalence": "off = base", "mechanism": "m", "gap_ids": ["G1"], "idea_ids": ["I1"], "sources": [],
          "hypothesis": "h", "predicted_signature": "p", "falsifier": "f", "expect_regex": [f"^Lever{i}: on"],
          "state_dict_change": False, "params_added": False, "dependencies": [], "files": [], "equivalence_test": "x",
          "cost": {"est_loc": 100, "extra_vram_gib": 0.5, "epoch_time_frac": 1.1}, "smoke": {"minutes_est": 4, "checks": []},
          "wave_lever": {"set": {f"--new_flag_{i}": []}, "coupled": [], "targets": "G1"}, "risks": []}
    lv.update(kw)
    return lv


class LintTest(unittest.TestCase):
    def test_good_proposal(self):
        errs, needs = R.lint_proposal({"levers": [lever(1), lever(2)]}, CYCLE, {"--lr"})
        self.assertEqual(errs, [])
        self.assertEqual(needs, {})

    def test_problems(self):
        bad = [lever(1, flags=[{"name": "--lr", "type": "float", "default": "", "choices": [], "help": ""}],
                     wave_lever={"set": {"--other": ["1"]}, "coupled": [], "targets": ""}, expect_regex=["("]),
               lever(2, state_dict_change=True, cost={"est_loc": 900, "extra_vram_gib": 9, "epoch_time_frac": 2.0}),
               lever(3, state_dict_change=True, gap_ids=[]), lever(4)]
        errs, _ = R.lint_proposal({"levers": bad}, CYCLE, {"--lr"})
        text = "\n".join(errs)
        for frag in ("4 levers", "already exists in the base trainer", "needs a default", "not one of the lever's own flags",
                     "does not compile", "est_loc 900", "epoch_time_frac 2.0", "extra_vram_gib 9", "change the state_dict",
                     "gap_ids is empty"):
            self.assertIn(frag, text)

    def test_smoke_budget_and_responses(self):
        lv = [lever(1, smoke={"minutes_est": 12, "checks": []}), lever(2, smoke={"minutes_est": 12, "checks": []}, params_added=True)]
        errs, _ = R.lint_proposal({"levers": lv, "responses": [{"objection_id": "O1.1", "action": "accept", "detail": ""}]},
                                  CYCLE, set(), prev_open_ids=["O1.1", "O1.2"])
        self.assertTrue(any("smoke estimate" in e for e in errs))
        self.assertTrue(any("O1.2" in e for e in errs))

    def test_dependencies(self):
        lv = [lever(1, dependencies=["json"]), lever(2, dependencies=["definitely_missing_pkg_xyz"])]
        errs, needs = R.lint_proposal({"levers": lv}, CYCLE, set(), check_import=lambda n: n == "json")
        self.assertEqual(errs, [])
        self.assertEqual(needs, {"lever_2": ["definitely_missing_pkg_xyz"]})


class DebateTest(unittest.TestCase):
    def obj(self, oid, sev, lid="lever_1", ev=("file.py:1",)):
        return {"id": oid, "lever_id": lid, "severity": sev, "category": "evidence", "claim": "c", "evidence": list(ev),
                "required_change": "r"}

    def test_convergence(self):
        led = {}
        ids = ["lever_1", "lever_2"]
        R.apply_critique(led, {"objections": [self.obj("O1.1", "blocker"), self.obj("O1.2", "major", "lever_2"),
                                              self.obj("O1.3", "minor")]}, 1, ids)
        self.assertEqual({o["id"] for o in R.open_objections(led)}, {"O1.1", "O1.2"})
        R.apply_critique(led, {"objections": [], "resolved": ["O1.1"], "still_open": [{"id": "O1.2", "note": ""}]}, 2, ids)
        self.assertEqual([o["id"] for o in R.open_objections(led)], ["O1.2"])
        R.apply_critique(led, {"objections": [], "resolved": ["O1.2"]}, 3, ids)
        self.assertEqual(R.open_objections(led), [])
        self.assertEqual(R.debate_counts(led)["minor"], 1)

    def test_blocker_without_evidence_is_downgraded(self):
        led = {}
        notes = R.apply_critique(led, {"objections": [self.obj("O1.1", "blocker", ev=())]}, 1, ["lever_1"])
        self.assertEqual(led["O1.1"]["severity"], "minor")
        self.assertTrue(notes)

    def test_moot_and_exhausted(self):
        led = {}
        R.apply_critique(led, {"objections": [self.obj("O1.1", "blocker"), self.obj("O1.2", "major", "lever_2"),
                                              self.obj("O1.3", "major", "")]}, 1, ["lever_1", "lever_2", "lever_3"])
        dropped, contested, plan_level = R.resolve_exhausted(led, ["lever_1", "lever_2", "lever_3"])
        self.assertEqual((dropped, contested, plan_level), (["lever_1"], ["lever_2"], ["O1.3"]))
        R.apply_critique(led, {"objections": []}, 2, ["lever_2", "lever_3"])          # lever_1 removed by the architect
        self.assertEqual(led["O1.1"]["status"], "moot")


class ScopeManifestTest(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="rdloop_repo_"))
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        for p in ("SLS_setup/b3wave/train.py", "SLS_setup/utils/a.py", "scripts/b3_x.sh", "docs/notes.md", "exploop/B3/wave.json"):
            f = self.repo / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "SLS_setup/b3wave/train.py"], check=True)
        (self.repo / "SLS_setup" / "exp" / "R1" / "ckpt").mkdir(parents=True)
        (self.repo / "SLS_setup" / "exp" / "R1" / "ckpt" / "e.pth").write_text("w")
        (self.repo / ".gitignore").write_text("SLS_setup/exp\n")

    def test_manifest_detects_changes(self):
        man = R.build_manifest(self.repo, CYCLE, "rdloop/T9")
        self.assertIn("SLS_setup/utils/a.py", man["files"])          # untracked files are covered too
        self.assertNotIn("exploop/B3/wave.json", man["files"])      # a running wave writes there
        self.assertEqual(R.check_manifest(self.repo, man, CYCLE, "rdloop/T9"), [])
        (self.repo / "SLS_setup" / "t9wave").mkdir()
        (self.repo / "SLS_setup" / "t9wave" / "train.py").write_text("new")      # in scope
        (self.repo / "scripts" / "t9_smoke.sh").write_text("new")               # in scope
        (self.repo / "docs" / "research").mkdir()
        (self.repo / "docs" / "research" / "x.md").write_text("ok")              # excluded
        self.assertEqual(R.check_manifest(self.repo, man, CYCLE, "rdloop/T9"), [])
        (self.repo / "SLS_setup" / "utils" / "a.py").write_text("changed")
        (self.repo / "SLS_setup" / "utils" / "b.py").write_text("new")
        (self.repo / "SLS_setup" / "exp" / "R1" / "ckpt" / "e.pth").write_text("ww")
        (self.repo / "SLS_setup" / "exp" / "R1" / "new.txt").write_text("added files in old runs are fine")
        v = R.check_manifest(self.repo, man, CYCLE, "rdloop/T9")
        self.assertIn("modified: SLS_setup/utils/a.py", v)
        self.assertIn("new file outside the cycle scope: SLS_setup/utils/b.py", v)
        self.assertIn("run file modified: SLS_setup/exp/R1/ckpt/e.pth", v)
        self.assertEqual(len(v), 3)

    def test_code_sha_and_scope(self):
        (self.repo / "SLS_setup" / "t9wave").mkdir()
        f = self.repo / "SLS_setup" / "t9wave" / "x.py"
        f.write_text("a")
        a = R.code_sha(self.repo, CYCLE)
        f.write_text("b")
        self.assertNotEqual(a, R.code_sha(self.repo, CYCLE))
        exact, pre = R.write_scope(self.repo, CYCLE, self.repo / "rdloop" / "T9", "engineer")
        self.assertTrue(R.in_scope(str(f), exact, pre))
        self.assertTrue(R.in_scope(str(self.repo / "scripts" / "t9_eval.py"), exact, pre))
        self.assertFalse(R.in_scope(str(self.repo / "scripts" / "b3_x.sh"), exact, pre))
        self.assertFalse(R.in_scope(str(self.repo / "SLS_setup" / "t9wave_x" / "a.py"), exact, pre))

    def test_freeze(self):
        man = R.build_manifest(self.repo, CYCLE, "rdloop/T9")
        out = self.repo / "frozen.tar.gz"
        self.assertGreaterEqual(R.freeze_untracked(self.repo, man, out), 2)
        self.assertTrue(out.is_file())


class WavesTest(unittest.TestCase):
    def test_yield_rules(self):
        repo = Path(tempfile.mkdtemp(prefix="rdloop_waves_"))
        wd = repo / "exploop" / "W1"
        wd.mkdir(parents=True)
        st = {"status": "RUNNING", "heartbeat_epoch": time.time(), "current": 2,
              "iterations": [{"k": 1, "done": True, "run_dir": "SLS_setup/exp/a"}, {"k": 2, "label": "W1b", "phase": "train", "run_dir": "SLS_setup/exp/b"}],
              "jobs": {}}
        R.write_json(wd / "loop_state.json", st)
        self.assertIn("W1b TRAIN", R.exploop_gpu_busy(repo))
        self.assertEqual(R.active_run_dirs(repo), ["SLS_setup/exp/b/"])
        st["iterations"][1]["phase"] = "analyze"
        R.write_json(wd / "loop_state.json", st)
        self.assertIsNone(R.exploop_gpu_busy(repo))
        st["iterations"][1]["jobs"] = {"analyze": {"pid": os.getpid(), "start_ticks": L.proc_start_ticks(os.getpid())}}
        R.write_json(wd / "loop_state.json", st)
        self.assertIn("ANALYZE", R.exploop_claude_busy(repo))
        st["heartbeat_epoch"] = time.time() - 3600                       # stale = the wave's supervisor is down
        R.write_json(wd / "loop_state.json", st)
        self.assertIsNone(R.exploop_claude_busy(repo))
        st["status"] = "DONE"
        R.write_json(wd / "loop_state.json", st)
        self.assertEqual(R.wave_status(repo, "W1"), "DONE")


class MiscTest(unittest.TestCase):
    def test_norm_url_and_verify(self):
        self.assertEqual(R.norm_url("https://arxiv.org/pdf/2604.23742v2.pdf"), R.norm_url("http://arxiv.org/abs/2604.23742"))
        self.assertEqual(R.norm_url("https://www.isca-archive.org/x/a.pdf?y=1"), "isca-archive.org/x/a")
        ideas = [{"sources": [{"url": "https://arxiv.org/abs/2604.23742"}, {"url": "https://github.com/x/y"}]}]
        v, n = R.verify_sources(ideas, ["https://arxiv.org/pdf/2604.23742"])
        self.assertEqual((v, n), (1, 2))
        self.assertEqual(ideas[0]["verified_sources"], 1)

    def test_compare_scores(self):
        d = Path(tempfile.mkdtemp(prefix="rdloop_scores_"))
        (d / "a").mkdir()
        (d / "b").mkdir()
        (d / "a" / "s1.logits").write_text("x.flac 1.0\ny.flac -2.0\n")
        (d / "b" / "s1.logits").write_text("x.flac 1.00001\ny.flac -2.0\n")
        self.assertTrue(R.compare_scores(d / "a", d / "b", ["s1"], 1e-4)["ok"])
        (d / "b" / "s1.logits").write_text("x.flac -0.1\ny.flac -2.0\n")
        r = R.compare_scores(d / "a", d / "b", ["s1"], 1e-4)
        self.assertFalse(r["ok"])
        self.assertEqual(r["sets"]["s1"]["flips_at_0.5"], 1)

    def test_smoke_argv(self):
        base = ["--track", "B3a_x_s1234", "--num_epochs", "8", "--keep_epochs", "8", "--loudness_norm_lufs", "-23", "--resume", "x"]
        argv = R.smoke_argv(base, CYCLE, "/tmp/cyc", "T9smk_s0_s1234", extra_sets={"--new_flag_1": []})
        d = L.argv_dict(argv)
        self.assertEqual(d["--num_epochs"], ["1"])
        self.assertEqual(d["--track"], ["T9smk_s0_s1234"])
        self.assertEqual(d["--loudness_norm_lufs"], ["-23"])
        self.assertEqual(d["--new_flag_1"], [])
        self.assertNotIn("--resume", d)

    def test_equivalence_argv_drops_augmenters(self):
        base = ["--use_rtc_aug", "--aug_noise_dirs", "a", "b", "--musan", "--musan_dir", "m", "--lr", "2.5e-6"]
        argv = R.set_flags(R.smoke_argv(base, CYCLE, "/tmp/cyc", "T9smk_eqbase_s1234"), {},
                           removes=["--use_rtc_aug", "--musan", "--use_level_aug"])
        d = L.argv_dict(argv)
        self.assertNotIn("--use_rtc_aug", d)
        self.assertNotIn("--musan", d)
        self.assertEqual(d["--musan_dir"], ["m"])                  # store_true removal keeps the neighbours
        self.assertEqual(d["--aug_noise_dirs"], ["a", "b"])
        self.assertEqual(d["--lr"], ["2.5e-6"])

    def test_seed_validates_with_exploop(self):
        plan = {"levers": [lever(1), lever(2, wave_lever={"set": {"--new_flag_2": [], "--new_flag_2b": ["0.5"]}, "coupled": [],
                                                           "targets": "G2"},
                                       flags=[{"name": "--new_flag_2", "type": "bool", "default": "False", "choices": [], "help": ""},
                                              {"name": "--new_flag_2b", "type": "float", "default": "0.0", "choices": [], "help": ""}])]}
        st = {"approved_levers": ["lever_1", "lever_2"], "code_sha": "abc",
              "base": {"label": "B0", "run_dir": "SLS_setup/exp/B0", "readouts": {}}}
        seed = R.build_seed(CYCLE, st, plan, R.REPO)
        wave = {"wave": "W", "track_prefix": "W", "letters": "abc", "trainer": {"entry": ["-m", "b3wave.train"]},
                "search_space": {"allowed": ["--lr"], "coupled": [], "forbidden": [], "fixed": {}, "max_epochs": 10,
                                 "keep_topk_by_wf1": 7, "seed": 1234},
                "early_check": {"expect": ["^b3wave: "]}, "gpu": {"busy_regex": "main_train"}}
        w = R.merge_seed(wave, seed, "T9dry", "/tmp/r")
        self.assertIn("^t9wave: ", w["early_check"]["expect"])
        self.assertIn("^b3wave: ", w["early_check"]["expect"])
        self.assertEqual(w["trainer"]["entry"], ["-m", "t9wave.train"])
        self.assertIn(["--new_flag_2", "--new_flag_2b"], w["search_space"]["coupled"])
        self.assertIn("b\\d+wave\\.(train|checks)", w["gpu"]["busy_regex"])
        base = ["--track", "x", "--num_epochs", "8", "--keep_topk_by_wf1", "7", "--keep_epochs", "8", "--seed", "1234", "--lr", "1e-6"]
        known = {"--track", "--num_epochs", "--keep_topk_by_wf1", "--keep_epochs", "--seed", "--lr", "--new_flag_1", "--new_flag_2", "--new_flag_2b"}
        for i, lev in enumerate(w["lever_queue"]):
            letter = "abc"[i]
            tr = f"T9drya_{lev['id'].replace('_', '')}_s1234".replace("T9drya", f"T9dry{letter}")
            cand = L.apply_change(base, {"set": lev["set"], "remove": []}, {"--track": [tr]})
            b2 = L.apply_change(base, {"set": {}, "remove": []}, {"--track": [tr]})
            self.assertEqual(L.validate_design(w, b2, cand, {"kind": "config", "letter": letter, "track": tr}, set(), known), [])

    def test_settings_shape(self):
        s = R.step_settings("engineer", R.REPO, CYCLE, R.REPO / "rdloop" / "T9", [], ["/x/"], "/usr/bin/python3")
        deny = s["permissions"]["deny"]
        self.assertIn("Agent", deny)
        self.assertTrue(any(d.startswith("Edit(//") and "SLS_setup/b3wave" in d for d in deny))
        self.assertFalse(any("SLS_setup/t9wave" in d for d in deny))
        self.assertIn("guard_hook.py", s["hooks"]["PreToolUse"][0]["hooks"][0]["command"])
        s2 = R.step_settings("scout", R.REPO, CYCLE, R.REPO / "rdloop" / "T9", [], [], "/usr/bin/python3")
        self.assertNotIn("WebFetch", s2["permissions"]["deny"])
        self.assertIn("WebFetch", R.step_settings("readonly", R.REPO, CYCLE, R.REPO / "rdloop" / "T9", [], [], "py")["permissions"]["deny"])

    def test_settings_allow_own_file_in_protected_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cyc = Path(td).resolve()
            (cyc / "debate").mkdir()
            (cyc / "debate" / "critique_v1.md").write_text("x")
            own = str(cyc / "debate" / "proposal_v2.md")
            perms = R.step_settings("readonly", R.REPO, CYCLE, cyc, [own], [], "py")["permissions"]
            self.assertIn(f"Edit(/{own})", perms["allow"])
            self.assertNotIn(f"Edit(/{cyc}/debate/**)", perms["deny"])
            self.assertNotIn(f"Edit(/{own})", perms["deny"])
            self.assertIn(f"Edit(/{cyc}/debate/critique_v1.md)", perms["deny"])
            self.assertIn(f"Edit(/{cyc}/review/**)", perms["deny"])


if __name__ == "__main__":
    unittest.main()
