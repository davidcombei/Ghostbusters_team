"""End-to-end tests of the real supervisor on a scratch wave (fake trainer / eval / claude; real
launcher rendering, job handling, verdicts, scripts/log_experiment.py on a scratch registry copy).
CPU only, a few minutes.  Run: python -m unittest scripts.exploop.tests.test_integration -v
(from the repo root) or python scripts/exploop/tests/test_integration.py
Set EXPLOOP_TEST_DIR to keep the scratch dirs for inspection."""
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

BASE_ENV = {"EXPLOOP_FAKE_EPOCH_S": "2", "EXPLOOP_FAKE_CLAUDE_S": "1",
            "EXPLOOP_FAKE_SHIFTS": "T1a:0.007,T1b:-0.002,T1c:0.001"}


def scratch(name):
    base = os.environ.get("EXPLOOP_TEST_DIR")
    if base:
        p = Path(base) / name
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.mkdtemp(prefix=f"exploop_{name}_"))


def it_phase(wd, k):
    st = H.state(wd)
    it = next(i for i in st["iterations"] if i["k"] == k)
    return it["phase"], it


class TestFullWaveWithKills(unittest.TestCase):
    def test_wave_survives_supervisor_and_trainer_kills(self):
        root = scratch("full")
        wd = H.make_wave(root, max_iterations=2, slow_s=6)
        env = dict(BASE_ENV, EXPLOOP_FAKE_DIE_AT="2")     # trainer dies silently in epoch 2 of each run (once)
        self.assertEqual(H.sup(wd, "init").returncode, 0)
        self.assertIn("next is DESIGN", H.sup(wd, "plan").stdout)
        p = H.start(wd, env)
        # 1) kill -9 the supervisor while the trainer runs; the trainer must survive and be re-attached
        H.wait_for(lambda: it_phase(wd, 1)[0] == "train" and it_phase(wd, 1)[1]["jobs"].get("train"), 60, what="iter 1 training")
        trainer_pid = it_phase(wd, 1)[1]["jobs"]["train"]["pid"]
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        time.sleep(1)
        self.assertTrue(H.L.proc_alive(trainer_pid, None), "the trainer must outlive the supervisor")
        st = H.sup(wd, "status")
        self.assertEqual(st.returncode, 1, st.stdout)                     # DOWN: no supervisor holds the lock
        p = H.start(wd, env)
        # 2) the trainer's silent death (EXPLOOP_FAKE_DIE_AT) -> automatic --resume
        H.wait_for(lambda: it_phase(wd, 1)[1].get("attempts", {}).get("train_resume", 0) >= 1, 90, what="resume after trainer death")
        H.wait_for(lambda: it_phase(wd, 1)[0] in ("eval", "verdict", "log", "analyze", "decide") or it_phase(wd, 1)[1].get("done"),
                   120, what="iter 1 training complete")
        self.assertLessEqual(len(H.procs("fake_trainer.py")), 1)
        # 3) kill -9 the supervisor during the slow eval step; it must re-attach / rerun and finish
        H.wait_for(lambda: "eval:slow" in it_phase(wd, 1)[1]["jobs"], 60, what="slow eval step")
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        p = H.start(wd, env)
        H.wait_for(lambda: H.state(wd).get("status") == "DONE", 300, poll=1, what="wave DONE")
        p.wait(timeout=30)
        st = H.state(wd)
        its = st["iterations"]
        self.assertEqual([i["done"] for i in its], [True, True])
        v1 = its[0]["verdict"]["merge"]
        self.assertTrue(v1["new_best"])
        self.assertEqual(v1["decision"], "promote")
        self.assertEqual(st["best"]["label"], "T1a")
        self.assertEqual(its[1]["verdict"]["merge"]["decision"], "drop")
        self.assertEqual(its[1]["base_label"], "T1a")                  # iteration 2 built on the new best
        rows = {r["id"]: r for r in H.L.registry_rows(root / "registry.csv")}
        for rid in ("T1a-codecrandom-screen", "T1a-codecrandom-ep3", "T1b-specaug-screen", "T1b-specaug-ep3"):
            self.assertIn(rid, rows)
            self.assertNotEqual(rows[rid]["ci_low"], "")
        self.assertEqual(rows["T1a-codecrandom-screen"]["decision"], "promote")
        for i in its:
            self.assertTrue(Path(i["report"]).is_file())
            launcher = Path(H.REPO / i["launcher"]) if not Path(i["launcher"]).is_absolute() else Path(i["launcher"])
            self.assertIn("ONE factor changed", launcher.read_text())
        self.assertTrue(Path(st["summary_report"]).is_file())
        self.assertIn("DONE", (wd / "STATUS.md").read_text())
        self.assertEqual(H.sup(wd, "status").returncode, 3)
        self.assertEqual(H.procs("fake_trainer.py"), [])


class TestBlocksAndRedesign(unittest.TestCase):
    def test_two_factor_design_blocks_after_three_attempts(self):
        root = scratch("twofactor")
        wd = H.make_wave(root, max_iterations=1)
        H.sup(wd, "init", check=True)
        p = H.start(wd, dict(BASE_ENV, EXPLOOP_FAKE_TWO_FACTORS="1"))
        p.wait(timeout=60)
        st = H.state(wd)
        self.assertEqual(st["status"], "BLOCKED")
        self.assertEqual(st["blocked"]["phase"], "design")
        self.assertEqual(st["iterations"][0]["attempts"]["design"], 3)
        self.assertEqual(H.sup(wd, "status").returncode, 2)
        # resume with the fault removed: the loop continues from DESIGN
        H.sup(wd, "unblock", "--reset-attempts", check=True)
        p = H.start(wd, BASE_ENV)
        H.wait_for(lambda: it_phase(wd, 1)[0] == "train", 60, what="design accepted after resume")
        H.sup(wd, "plan")
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
        self._cleanup(wd)

    def test_bad_header_stops_own_run_and_redesigns(self):
        root = scratch("badheader")
        wd = H.make_wave(root, max_iterations=1)
        H.sup(wd, "init", check=True)
        env = dict(BASE_ENV, EXPLOOP_FAKE_EPOCH_S="8", EXPLOOP_FAKE_DESIGN_EXPECT="^THIS LINE NEVER APPEARS")
        p = H.start(wd, env)
        H.wait_for(lambda: H.state(wd).get("status") == "BLOCKED", 120, what="early check fails twice -> BLOCKED")
        p.wait(timeout=30)
        st = H.state(wd)
        it = st["iterations"][0]
        self.assertEqual(it["early_failures"], 2)
        self.assertIn("early check failed twice", st["blocked"]["reason"])
        self.assertTrue(list((wd / "iter01").glob("design.result.rejected*.json")))
        time.sleep(1)
        self.assertEqual(H.procs("fake_trainer.py"), [], "the supervisor must have stopped its own trainer")

    def test_container_restart_reinstalls_ffmpeg_then_resumes(self):
        root = scratch("restart")
        wd = H.make_wave(root, max_iterations=1)
        H.sup(wd, "init", check=True)
        env = dict(BASE_ENV, EXPLOOP_FAKE_EPOCH_S="3")
        p = H.start(wd, env)
        H.wait_for(lambda: it_phase(wd, 1)[0] == "train" and it_phase(wd, 1)[1].get("run_dir") and
                   it_phase(wd, 1)[1].get("epochs_seen", 0) >= 1, 90, what="epoch 1 of iter 1")
        # simulate a container restart: everything dies, PID 1 identity changes, ffmpeg is gone
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        for pid in H.procs("fake_trainer.py"):
            os.kill(pid, signal.SIGKILL)
        import json
        stp = wd / "loop_state.json"
        s = json.loads(stp.read_text())
        s["pid1_start"] = "0:0"
        stp.write_text(json.dumps(s))
        self.assertIn("container restarted", H.sup(wd, "status").stdout)
        ff = str(root / "bin" / "ffmpeg-fake")
        p = H.start(wd, dict(env, EXPLOOP_FFMPEG=ff))
        H.wait_for(lambda: it_phase(wd, 1)[1].get("train_done"), 120, what="resumed training finished")
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
        ev = (wd / "events.jsonl").read_text()
        self.assertIn("container_restart", ev)
        self.assertIn("ffmpeg reinstalled", ev)
        self.assertIn("--resume", ev)
        self.assertTrue(Path(ff).exists())

    def _cleanup(self, wd):
        for pid in H.procs("fake_trainer.py"):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
