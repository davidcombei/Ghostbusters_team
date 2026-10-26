"""Tests of scripts/rdloop/view.py (python 3.8 compatible): snapshots of a scratch cycle at several
states, the activity parsed from a fixture transcript, and a status line that never raises.
    python -m unittest scripts/rdloop/tests/test_view.py -v      (from the repo root)"""
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
import view as V  # noqa: E402


def make_cycle(root, state):
    cd = Path(root) / "rdloop" / "T7"
    (cd / "debate").mkdir(parents=True)
    R.write_json(cd / "cycle.json", {"cycle": "T7", "module": "t7wave", "debate": {"max_rounds": 3},
                                     "claude": {"max_total_usd": 150}, "smoke": {"max_minutes": 30}})
    R.write_json(cd / "state.json", state)
    R.write_json(cd / "debate" / "proposal_v1.json", {"levers": [{"id": "aug_x", "title": "a", "flags": [{"name": "--aug_x", "default": "False"}]},
                                                                {"id": "head_y", "title": "h", "flags": []}],
                                                     "responses": [{"objection_id": "O1.1", "action": "accept", "detail": "ok"}]})
    with open(cd / "events.jsonl", "w") as f:
        for i in range(5):
            f.write(json.dumps({"ts": R.now_ts(), "phase": "audit", "kind": "phase" if i % 2 else "job_start", "msg": f"m{i}"}) + "\n")
    return cd


class ViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rdloop_view_"))
        self.projects = self.tmp / "claude" / "projects" / "-x"
        self.projects.mkdir(parents=True)
        R.CLAUDE_PROJECTS = self.tmp / "claude" / "projects"
        sid = "11111111-2222-3333-4444-555555555555"
        lines = [{"type": "assistant", "timestamp": "t1", "message": {"content": [{"type": "text", "text": "Reading the audit first."},
                                                                                  {"type": "tool_use", "name": "WebSearch", "input": {"query": "codec robust deepfake"}}]}},
                 {"type": "assistant", "timestamp": "t2", "message": {"content": [{"type": "tool_use", "name": "WebFetch", "input": {"url": "https://arxiv.org/abs/2604.23742"}}]}},
                 {"type": "user", "message": {"content": "ignored"}}]
        (self.projects / f"{sid}.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        self.sid = sid

    def test_running_debate(self):
        st = {"status": "RUNNING", "phase": "critique", "gate_mode": "owner", "cost_usd": 12.5, "supervisor_pid": os.getpid(),
              "heartbeat_epoch": time.time(), "base": {"label": "B3a", "run_dir": "SLS_setup/exp/B3a"},
              "debate": {"version": 1, "rounds": [], "ledger": {"O1.1": {"id": "O1.1", "severity": "major", "status": "open",
                                                                         "lever_id": "aug_x", "claim": "weak evidence"}}},
              "team": {"research-scout": {"state": "running", "step": "scout a", "session_id": self.sid, "started": time.time() - 65}}}
        cd = make_cycle(self.tmp, st)
        s = V.snapshot(cd)
        self.assertTrue(s["ok"], s.get("error"))
        self.assertEqual(s["phase_key"], "debate")
        self.assertEqual(s["supervisor"]["state"], "alive")
        scout = next(t for t in s["team"] if t["agent"] == "research-scout")
        self.assertEqual(scout["tool_uses"], 2)
        self.assertEqual(scout["last_tool"], "WebFetch")
        self.assertEqual(scout["queries"], ["codec robust deepfake"])
        self.assertEqual(scout["last_text"], "Reading the audit first.")
        self.assertEqual(s["debate"]["open"][0]["id"], "O1.1")
        self.assertEqual([l["id"] for l in s["levers"]], ["aug_x", "head_y"])
        self.assertEqual(s["events_total"], 5)
        self.assertTrue(all(e["kind"] != "job_start" for e in s["events"]))
        line = V.render_line(s)
        self.assertIn("DEBATE", line)
        self.assertIn("scout", line)

    def test_owner_action_and_line(self):
        st = {"status": "AWAITING_APPROVAL", "phase": "gate", "plan": {"version": 2, "levers": ["aug_x"], "contested": []},
              "supervisor_pid": os.getpid(), "heartbeat_epoch": time.time()}
        s = V.snapshot(make_cycle(self.tmp, st))
        self.assertEqual(s["action"]["kind"], "approve")
        self.assertIn("/rd-cycle approve T7", V.render_line(s))

    def test_final_status_hidden_and_line_never_raises(self):
        cd = make_cycle(self.tmp, {"status": "PARKED", "phase": "gate"})
        self.assertIsNone(V.render_line(V.snapshot(cd)))
        (cd / "state.json").write_text("{broken")
        r = subprocess.run([sys.executable, str(HERE.parent / "view.py"), "line", "--cycle", str(cd)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        r = subprocess.run([sys.executable, str(HERE.parent / "view.py"), "snapshot", "--cycle", str(self.tmp / "nope")],
                           capture_output=True, text=True)
        self.assertEqual(json.loads(r.stdout)["ok"], False)


if __name__ == "__main__":
    unittest.main()
