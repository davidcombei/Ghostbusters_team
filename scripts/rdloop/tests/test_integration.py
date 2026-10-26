"""End-to-end tests of the real rdloop supervisor on a scratch repo (fake claude / trainer / eval, the
real exploop supervisor for the handoff dry run). CPU only, a few minutes.
    python -m unittest scripts/rdloop/tests/test_integration.py -v      (from the repo root)
Set RDLOOP_TEST_DIR to keep the scratch dirs for inspection."""
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

R = H.R


def scratch(name):
    base = os.environ.get("RDLOOP_TEST_DIR")
    if base:
        p = Path(base) / name
        if p.exists():
            import shutil
            shutil.rmtree(p)
        p.mkdir(parents=True)
        return p
    return Path(tempfile.mkdtemp(prefix=f"rdloop_{name}_"))


class CycleTest(unittest.TestCase):
    def run_to_end(self, cd, env=None, timeout=300):
        r = H.sup(cd, "run", env=env, timeout=timeout)
        return r

    def test_full_cycle_auto_gate_with_kill(self):
        repo, cd = H.make_repo(scratch("full"), gate="auto")
        self.assertEqual(H.sup(cd, "init").returncode, 0, H.sup(cd, "status").stdout)
        self.assertTrue((cd / "isolation" / "manifest.json").is_file())
        self.assertIn("WAIT_WAVES", H.sup(cd, "plan").stdout)
        env = {"RDLOOP_FAKE_CRITIC_ROUNDS": "2", "RDLOOP_FAKE_CLAUDE_S": "1.5"}
        p = H.start(cd, env)
        # kill -9 the supervisor while the engineer works; the step must be re-attached / finished
        H.wait_for(lambda: H.state(cd)["phase"] == "implement" and H.state(cd)["jobs"], 240, what="implement step")
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        self.assertEqual(H.sup(cd, "status").returncode, 1)
        r = self.run_to_end(cd, env)
        st = H.state(cd)
        self.assertEqual(st["status"], "AWAITING_USER", r.stdout[-3000:] + json.dumps(st.get("blocked")))
        self.assertEqual(st["approved_levers"], ["aug_x", "head_y"])
        self.assertEqual(len(st["debate"]["rounds"]), 2)
        self.assertEqual(st["debate"]["outcome"], "agreed")
        mod = repo / "SLS_setup" / "t1wave"
        self.assertIn('print(f"t1wave: {T1WAVE_VERSION}")', (mod / "train.py").read_text())
        self.assertIn("--aug_x", (mod / "train.py").read_text())
        g = st["tests"]["gpu"]
        self.assertTrue(g["ok"], json.dumps([(j["name"], j["state"], j.get("note")) for j in g["jobs"]], indent=1))
        names = [j["name"] for j in g["jobs"]]
        for n in ("R1_old_scores", "R2_b3wave_checks_model", "R3_module_gpu_checks", "R4_base_micro", "R4_levers_off_micro",
                  "S0_levers_off", "R4_equivalence", "S_aug_x", "S_head_y", "E_eval_merge_path"):
            self.assertIn(n, names)
        self.assertTrue(st["tests"]["compat"]["ok"])
        seed = R.read_json(cd / "handoff" / "wave_seed.json")
        self.assertEqual(seed["trainer"]["entry"], ["-m", "t1wave.train"])
        self.assertEqual([x["id"] for x in seed["lever_queue"]], ["aug_x", "head_y"])
        self.assertTrue((cd / "handoff" / "dryrun_wave" / "loop_state.json").is_file())
        for f in ("FINAL_SUMMARY.md", "plan_agreed.md", "01_audit.md", "02_sota.md"):
            self.assertTrue((repo / "docs" / "research" / "T1" / f).is_file(), f)
        self.assertIn("aug_x", (repo / "docs" / "research" / "lever_ledger.md").read_text())
        self.assertTrue((repo / "docs" / "research" / "literature.jsonl").is_file())
        self.assertEqual(H.sup(cd, "verify").returncode, 0)
        self.assertFalse(any((cd / "runs").glob("*/last_state.pt")))       # smoke runs cleaned
        self.assertTrue((cd / "impl" / "commands.log").is_file())
        r = H.sup(cd, "run")                                                 # a finished cycle does nothing
        self.assertIn("AWAITING_USER", r.stdout)

    def test_owner_gate_subset_and_reopen(self):
        repo, cd = H.make_repo(scratch("gate"), gate="owner")
        self.assertEqual(H.sup(cd, "init").returncode, 0)
        env = {"RDLOOP_FAKE_CRITIC_ROUNDS": "1"}
        p = H.start(cd, env)
        H.wait_for(lambda: H.state(cd).get("status") == "AWAITING_APPROVAL", 120, what="the plan gate")
        self.assertTrue((cd / "plan_agreed.md").is_file())
        self.assertEqual(H.sup(cd, "status").returncode, 3)
        r = H.sup(cd, "approve", "--levers", "aug_x")
        self.assertEqual(r.returncode, 0, r.stderr)
        p.wait(timeout=300)
        st = H.state(cd)
        self.assertEqual(st["status"], "AWAITING_USER", json.dumps(st.get("blocked")))
        self.assertEqual(st["approved_levers"], ["aug_x"])
        self.assertIn("head_y", st["dropped_levers"])
        self.assertNotIn("--head_y", (repo / "SLS_setup" / "t1wave" / "train.py").read_text())
        # the owner asks for changes to the code -> FIX -> STATIC -> REVIEW -> GPU -> ... -> AWAITING_USER again
        r = H.sup(cd, "reopen", "--at", "implement", "--note", "rename the header line")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(H.state(cd)["phase"], "fix")
        r = self.run_to_end(cd, env)
        st = H.state(cd)
        self.assertEqual(st["status"], "AWAITING_USER", r.stdout[-2000:])
        self.assertGreaterEqual(st["fix"]["n"], 1)

    def test_escape_blocks(self):
        repo, cd = H.make_repo(scratch("escape"))
        H.sup(cd, "init")
        r = self.run_to_end(cd, {"RDLOOP_FAKE_ESCAPE": "1", "RDLOOP_FAKE_CRITIC_ROUNDS": "1"})
        st = H.state(cd)
        self.assertEqual(st["status"], "BLOCKED", r.stdout[-2000:])
        self.assertIn("outside the cycle scope", st["blocked"]["reason"])
        v = R.read_json(cd / "isolation" / "violations.json")
        self.assertIn("new file outside the cycle scope: SLS_setup/utils/evil.py", v["violations"])

    def test_never_agree_needs_owner(self):
        repo, cd = H.make_repo(scratch("never"))
        H.sup(cd, "init")
        r = self.run_to_end(cd, {"RDLOOP_FAKE_NEVER_AGREE": "1"})
        st = H.state(cd)
        self.assertEqual(st["status"], "NEEDS_USER", r.stdout[-2000:])
        self.assertEqual(len(st["debate"]["rounds"]), 3)
        self.assertIn("head_y", st["dropped_levers"])                    # open blocker -> dropped
        self.assertEqual(st["plan"]["contested"], ["aug_x"])             # open major -> contested (auto gate drops it)
        self.assertFalse((repo / "SLS_setup" / "t1wave").exists())

    def test_broken_default_goes_through_fix(self):
        repo, cd = H.make_repo(scratch("fix"))
        H.sup(cd, "init")
        r = self.run_to_end(cd, {"RDLOOP_FAKE_BREAK_DEFAULT": "1", "RDLOOP_FAKE_CRITIC_ROUNDS": "1",
                                 "RDLOOP_FAKE_REVIEW_FINDING": "1"})
        st = H.state(cd)
        self.assertEqual(st["status"], "AWAITING_USER", r.stdout[-3000:] + json.dumps(st.get("blocked")))
        self.assertGreaterEqual(st["fix"]["n"], 2)                           # review finding + equivalence failure
        ev = [json.loads(x) for x in (cd / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["kind"] == "check" and "R4_equivalence: FAIL" in e["msg"] for e in ev))
        self.assertIn("default=False", (repo / "SLS_setup" / "t1wave" / "train.py").read_text())

    def test_needs_user_from_engineer(self):
        repo, cd = H.make_repo(scratch("needs"))
        H.sup(cd, "init")
        self.run_to_end(cd, {"RDLOOP_FAKE_NEEDS_USER": "1", "RDLOOP_FAKE_CRITIC_ROUNDS": "1"})
        st = H.state(cd)
        self.assertEqual(st["status"], "NEEDS_USER")
        self.assertIn("fancy_pkg", " ".join(st["needs_user"]))


if __name__ == "__main__":
    unittest.main()
