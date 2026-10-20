"""Tests of the read-only wave view (view.py): snapshot fields per phase / status and the 2-line status bar.
Synthetic waves in a tmpdir, plus a read-only check on the real exploop/B3.
Run: python -m unittest scripts.exploop.tests.test_view -v"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exploop_lib as L  # noqa: E402
import view as V  # noqa: E402

REPO = HERE.parents[2]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

TRAIN_LOG_BEST = """[2026-10-09 22:48:25] Epoch 1/8 TrainLoss=0.404831 TrainAcc=90.16% DevLoss=0.099016 DevAcc=98.35% LR=2.469e-06
[2026-10-09 22:48:51] Epoch 1 LocalWF1=90.17 F1clean=96.91 F1noisy=87.28 WF1oracle=90.55 EERnoisy=13.14
[2026-10-09 23:47:47] Epoch 2/8 TrainLoss=0.193157 TrainAcc=97.87% DevLoss=0.123192 DevAcc=98.41% LR=2.289e-06
[2026-10-09 23:48:13] Epoch 2 LocalWF1=91.64 F1clean=97.26 F1noisy=89.24 WF1oracle=92.27 EERnoisy=11.56
[2026-10-10 00:47:04] Epoch 3/8 TrainLoss=0.122162 TrainAcc=98.63% DevLoss=0.151385 DevAcc=98.53% LR=1.976e-06
[2026-10-10 00:47:31] Epoch 3 LocalWF1=91.39 F1clean=97.26 F1noisy=88.87 WF1oracle=92.14 EERnoisy=11.49
"""
TRAIN_LOG_CUR = """[2026-10-10 15:55:00] Epoch 1/8 TrainLoss=0.390000 TrainAcc=90.50% DevLoss=0.095000 DevAcc=98.40% LR=2.469e-06
[2026-10-10 15:55:30] Epoch 1 LocalWF1=90.60 F1clean=97.00 F1noisy=87.90 WF1oracle=90.90 EERnoisy=12.80
[2026-10-10 16:55:00] Epoch 2/8 TrainLoss=0.180000 TrainAcc=97.90% DevLoss=0.118000 DevAcc=98.45% LR=2.289e-06
[2026-10-10 16:55:30] Epoch 2 LocalWF1=91.20 F1clean=97.10 F1noisy=88.70 WF1oracle=91.80 EERnoisy=11.90
"""
TQDM = "".join(f"\rTraining:  {p}%|###       | {i}/9474 [24:58<33:18,  2.86batch/s]" for p, i in ((39, 3700), (40, 3753)))
VALID = "\rValidating:  55%|#####5    | 110/200 [00:20<00:16,  5.10it/s]"


def sleeper(*extra):
    """A live process whose cmdline contains `extra` (e.g. 'supervisor.py')."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", *extra])
    for _ in range(50):
        if L.proc_start_ticks(p.pid) is not None:
            break
        time.sleep(0.01)
    return p


class FakeWave:
    def __init__(self, root, status="RUNNING", phase="train", sup_alive=True, hb_age=5):
        self.root = Path(root)
        self.wd = self.root / "exploop" / "W"
        self.wd.mkdir(parents=True)
        best_run = self.root / "exp" / "Wa_run"
        cur_run = self.root / "exp" / "Wb_run"
        for d, txt in ((best_run, TRAIN_LOG_BEST), (cur_run, TRAIN_LOG_CUR)):
            (d / "logs").mkdir(parents=True)
            (d / "logs" / "train.log").write_text(txt)
        self.stdout_log = self.root / "Wb.log"
        self.stdout_log.write_text("header\n" + TRAIN_LOG_CUR + TQDM)
        prof = self.root / "profile.json"
        prof.write_text(json.dumps({"steps": [{"name": "select_topk"}, {"name": "merge_eval", "gpu": True},
                                              {"name": "final_eval", "gpu": True}, {"name": "ci_merge"}]}))
        L.write_json(self.wd / "wave.json", {
            "wave": "W", "repo": str(self.root), "max_iterations": 3, "letters": "abc", "track_prefix": "W",
            "smoke": "off", "gpu": {"poll_s": 60}, "claude": {"max_total_usd": 150}, "eval": {"profile": str(prof)}})
        self.procs = []
        sup = None
        if sup_alive:
            sup = sleeper("supervisor.py")
            self.procs.append(sup)
        else:
            p = subprocess.Popen([sys.executable, "-c", "pass"])
            p.wait()
            sup = p
        self.st = {
            "wave": "W", "status": status, "current": 2, "cost_usd": 5.646, "waiting": None, "blocked": None,
            "supervisor_pid": sup.pid, "heartbeat_epoch": time.time() - hb_age, "pid1_start": L.pid1_start(),
            "best": {"label": "Wa", "iter": 1, "run_dir": "exp/Wa_run", "merge_value": 93.056, "final_value": 92.94,
                     "registry_ids": {"merge": "Wa-m", "final": "Wa-f"}},
            "iterations": [
                {"k": 1, "label": "Wa", "track": "Wa_t", "phase": "decide", "done": True, "adopted": True,
                 "change_vs_ref": "+ --x 1", "verdict": {"merge": {"run": 93.056, "delta": 1.08, "ci_low": 0.5,
                                                                   "ci_high": 1.6, "decision": "promote", "new_best": True}}},
                {"k": 2, "label": "Wb", "track": "Wb_t", "phase": phase, "done": False, "argv": ["--num_epochs", "8"],
                 "change_vs_ref": "+ --sampler balanced", "run_dir": "exp/Wb_run", "logs": [str(self.stdout_log)],
                 "jobs": {}, "claude_sessions": [], "epochs_seen": 2}]}
        self.events = [{"ts": "2026-10-10 14:46:03", "iter": 2, "phase": "train", "kind": "phase", "msg": "Wb -> TRAIN"},
                       {"ts": "2026-10-10 15:55:30", "iter": 2, "phase": "train", "kind": "epoch", "msg": "Wb epoch 1/8",
                        "data": {"epoch": 1, "row": {"wf1": 90.6}, "delta_vs_best": 0.43}}]
        self.save()

    def it(self):
        return self.st["iterations"][1]

    def save(self):
        L.write_json(self.wd / "loop_state.json", self.st)
        (self.wd / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events))

    def close(self):
        for p in self.procs:
            p.kill()
            p.wait()


def plain(lines):
    return [ANSI.sub("", x) for x in lines]


class TestView(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.waves = []

    def tearDown(self):
        for w in self.waves:
            w.close()
        self.tmp.cleanup()

    def wave(self, **kw):
        w = FakeWave(Path(self.tmp.name) / f"r{len(self.waves)}", **kw)
        self.waves.append(w)
        return w

    def assertTwoLines(self, snap, width=200):
        lines = V.render_line(snap, width, "Opus ctx 34%")
        self.assertEqual(len(lines), 2)
        for x in plain(lines):
            self.assertLessEqual(len(x), width)
        return plain(lines)

    def test_train_progress_and_compare_vs_best(self):
        w = self.wave()
        s = V.snapshot(w.wd)
        self.assertTrue(s["ok"], s.get("error"))
        self.assertEqual(s["supervisor"]["state"], "alive")
        c = s["current"]
        self.assertEqual((c["k"], c["label"], c["phase"]), (2, "Wb", "train"))
        self.assertEqual(s["phases"], [p for p in L.PHASES if p != "smoke"])
        p = c["progress"]
        self.assertEqual((p["epoch"], p["epochs_done"], p["i"], p["n"], p["pct"], p["substep"]), (3, 2, 3753, 9474, 40, "train"))
        self.assertAlmostEqual(p["eta_h"], 1998 / 3600 + 5 * 3496 / 3600 * 1.03, places=3)
        cmp_ = s["compare"]
        self.assertEqual(cmp_["best_label"], "Wa")
        self.assertEqual(cmp_["target_now"], 91.39)                       # the best's LocalWF1 at the epoch in progress
        self.assertEqual(cmp_["best_peak"], {"epoch": 2, "wf1": 91.64})
        self.assertAlmostEqual(cmp_["last"]["d_wf1"], 91.20 - 91.64, places=6)
        self.assertAlmostEqual(cmp_["rows"][0]["d_wf1"], 90.60 - 90.17, places=6)
        self.assertAlmostEqual(cmp_["rows"][0]["d_eer_noisy"], 12.80 - 13.14, places=6)
        self.assertEqual((cmp_["ahead"], cmp_["judged"]), (1, 2))
        self.assertEqual(len(cmp_["spark"]), 2)
        self.assertEqual(len(cmp_["best_spark"]), 3)
        l1, l2 = self.assertTwoLines(s)
        self.assertIn("W RUNNING", l1)
        self.assertIn("it 2/3 Wb", l1)
        self.assertIn("TRAIN", l1)
        self.assertIn("ep 3/8", l2)
        self.assertIn("40%", l2)
        self.assertIn("ep2 LocalWF1 91.20 (-0.44 vs Wa@2)", l2)
        self.assertIn("to beat: Wa@ep3 91.39", l2)
        self.assertIn("best Wa half_B 93.06", l2)
        self.assertIn("$5.65/150", l2)

    def test_validating_substep(self):
        w = self.wave()
        w.stdout_log.write_text(TQDM + VALID)
        s = V.snapshot(w.wd)
        self.assertEqual(s["current"]["progress"]["substep"], "validate")
        self.assertEqual(s["current"]["progress"]["validate"]["pct"], 55)
        self.assertIn("validating 55%", self.assertTwoLines(s)[1])

    def test_narrow_terminal_compacts_the_phase_strip(self):
        s = V.snapshot(self.wave().wd)
        l1, l2 = self.assertTwoLines(s, width=60)
        self.assertIn("TRAIN (3/8)", l1)
        self.assertNotIn("verdict", l1)

    def test_eval_running_step(self):
        w = self.wave(phase="eval")
        job = sleeper()
        w.procs.append(job)
        w.it()["jobs"] = {"eval:final_eval": {"pid": job.pid, "start_ticks": L.proc_start_ticks(job.pid),
                                              "started": time.time() - 600, "started_ts": "x", "log": "l"}}
        w.save()
        s = V.snapshot(w.wd)
        ev = s["current"]["eval"]
        self.assertEqual((ev["done"], ev["total"], ev["running"]["name"]), (2, 4, "final_eval"))
        self.assertEqual([x["state"] for x in ev["steps"]], ["done", "done", "running", "todo"])
        self.assertIn("eval 2/4 ▶ final_eval 10m", self.assertTwoLines(s)[1])

    def test_eval_progress_from_events_without_a_job(self):
        w = self.wave(phase="eval")
        w.events.append({"ts": "t", "iter": 2, "phase": "eval", "kind": "eval_step", "msg": "merge_eval done"})
        w.save()
        self.assertEqual(V.snapshot(w.wd)["current"]["eval"]["done"], 2)

    def test_design_step_shows_claude_activity(self):
        w = self.wave(phase="design")
        job = sleeper()
        w.procs.append(job)
        sid = "11111111-2222-3333-4444-555555555555"
        w.it()["jobs"] = {"design": {"pid": job.pid, "start_ticks": L.proc_start_ticks(job.pid), "session_id": sid,
                                     "started": time.time() - 180, "cmd": ["claude", "-p", "x", "--agent", "exp-designer"]}}
        w.save()
        proj = Path(self.tmp.name) / "projects" / "-repo"
        proj.mkdir(parents=True)
        lines = [{"type": "assistant", "timestamp": "t1", "message": {"content": [
                     {"type": "tool_use", "name": "Read", "input": {"file_path": "/x/SKILL.md"}}]}},
                 {"type": "user", "message": {"content": "ok"}},
                 {"type": "assistant", "timestamp": "t2", "message": {"content": [
                     {"type": "tool_use", "name": "Bash", "input": {"command": "grep ...", "description": "List flags"}}]}}]
        (proj / f"{sid}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        old = V.CLAUDE_PROJECTS
        V.CLAUDE_PROJECTS = Path(self.tmp.name) / "projects"
        try:
            s = V.snapshot(w.wd)
        finally:
            V.CLAUDE_PROJECTS = old
        a = s["claude"]
        self.assertEqual((a["agent"], a["tool_uses"], a["last_tool"], a["last_target"]), ("exp-designer", 2, "Bash", "List flags"))
        self.assertIn("exp-designer 3m · 2 tools · last: Bash List flags", self.assertTwoLines(s)[1])

    def test_blocked_waiting_paused(self):
        w = self.wave(status="BLOCKED")
        w.st["blocked"] = {"phase": "eval", "reason": "merge_eval failed 2 times", "hint": "read the log"}
        w.st["waiting"] = "GPU busy (pid 1234)"
        w.save()
        (w.wd / "PAUSE").touch()
        l1, l2 = self.assertTwoLines(V.snapshot(w.wd), width=300)
        self.assertIn("BLOCKED", l1)
        self.assertIn("BLOCKED in eval: merge_eval failed 2 times - loopctl.sh resume W", l2)
        self.assertIn("PAUSED", l2)
        self.assertIn("waiting: GPU busy", l2)

    def test_supervisor_down_stale_and_container_restart(self):
        s = V.snapshot(self.wave(sup_alive=False).wd)
        self.assertEqual(s["supervisor"]["state"], "down")
        self.assertIn("supervisor DOWN - loopctl.sh resume W", self.assertTwoLines(s)[1])
        s = V.snapshot(self.wave(hb_age=3600).wd)
        self.assertEqual(s["supervisor"]["state"], "stale")
        w = self.wave()
        w.st["pid1_start"] = "0:0"
        w.save()
        s = V.snapshot(w.wd)
        self.assertTrue(s["supervisor"]["container_restarted"])
        self.assertIn("container restarted", self.assertTwoLines(s)[1])

    def test_done_wave(self):
        w = self.wave(status="DONE", sup_alive=False)
        w.st["current"] = None
        w.st["summary_report"] = "docs/summary.md"
        w.save()
        s = V.snapshot(w.wd)
        self.assertEqual((s["supervisor"]["state"], s["wave_phase"]), ("exited", "done"))
        lines = plain(V.render_line(s, 200))
        self.assertIn("wave done", lines[1])
        self.assertIn("docs/summary.md", lines[1])

    def test_missing_or_corrupt_state_never_raises(self):
        w = self.wave()
        (w.wd / "loop_state.json").write_text("{not json")
        s = V.snapshot(w.wd)
        self.assertFalse(s["ok"])
        self.assertIn("loop_state.json", s["error"])
        self.assertIn("view error", plain(V.render_line(s))[0])

    def test_events_carry_seq_for_toasts(self):
        s = V.snapshot(self.wave().wd)
        self.assertEqual(s["events_total"], 2)
        self.assertEqual([e["seq"] for e in s["events"]], [1, 2])
        self.assertEqual(s["events"][1]["data"], {"epoch": 1, "delta_vs_best": 0.43, "wf1": 90.6})

    def test_pick_wave_prefers_the_live_one(self):
        a, b = self.wave(), self.wave(status="DONE")
        root = Path(self.tmp.name) / "all"
        root.mkdir()
        (root / "A").symlink_to(a.wd)
        (root / "B").symlink_to(b.wd)
        self.assertEqual(V.pick_wave(root=root).name, "A")

    def test_line_cli_reads_claude_json(self):
        w = self.wave()
        r = subprocess.run([sys.executable, str(HERE.parent / "view.py"), "line", "--wave", str(w.wd), "--width", "300"],
                           input=json.dumps({"model": {"display_name": "Opus 5.5"}, "context_window": {"used_percentage": 34.2}}),
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = plain(r.stdout.splitlines())
        self.assertEqual(len(out), 2)
        self.assertIn("Opus 5.5 ctx 34%", out[1])


@unittest.skipUnless((REPO / "exploop/B3/loop_state.json").is_file(), "needs the real exploop/B3 wave")
class TestRealWaveReadOnly(unittest.TestCase):
    def test_snapshot_touches_nothing(self):
        wd = REPO / "exploop/B3"
        files = [wd / "loop_state.json", wd / ".supervisor.lock", wd / "events.jsonl", wd / "STATUS.md"]
        before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()}
        listing = sorted(os.listdir(wd))
        s = V.snapshot(wd)
        self.assertTrue(s["ok"], s.get("error"))
        self.assertTrue(s["best"]["label"])
        lines = V.render_line(s, 200)
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(sorted(os.listdir(wd)), listing)
        for p, (mt, sz) in before.items():
            if p.name in ("loop_state.json", "events.jsonl", "STATUS.md"):
                continue        # the live supervisor may rewrite these between the two reads
            self.assertEqual((p.stat().st_mtime_ns, p.stat().st_size), (mt, sz), p)


if __name__ == "__main__":
    unittest.main()
