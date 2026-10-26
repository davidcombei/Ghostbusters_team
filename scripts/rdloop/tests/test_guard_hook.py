"""Table tests of scripts/rdloop/guard_hook.py, run as the real hook process (stdin JSON + env).
Run with both interpreters the hook may get:  /usr/bin/python3 (3.8) and the rtc-sdd python.
    python -m unittest scripts/rdloop/tests/test_guard_hook.py -v      (from the repo root)
RDLOOP_TEST_HOOK_PY=/usr/bin/python3 selects the interpreter that runs the hook."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE.parent / "guard_hook.py"
HOOK_PY = os.environ.get("RDLOOP_TEST_HOOK_PY", "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable)
REPO = HERE.parents[2]


def run_hook(tool, tool_input, policy="engineer", files=(), prefixes=(), cwd=None, step="implement", extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RDLOOP_")}
    if step:
        env.update(RDLOOP_STEP=step, RDLOOP_POLICY=policy, RDLOOP_WRITE_FILES=":".join(files),
                   RDLOOP_WRITE_PREFIXES=":".join(prefixes), RDLOOP_WEB_DOMAINS="arxiv.org,github.com,isca-archive.org",
                   RDLOOP_REPO=str(REPO))
    env.update(extra_env or {})
    ev = {"tool_name": tool, "tool_input": tool_input, "cwd": cwd or str(REPO)}
    r = subprocess.run([HOOK_PY, "-I", str(HOOK)], input=json.dumps(ev), capture_output=True, text=True, env=env, timeout=30)
    if r.returncode != 0:
        raise AssertionError(f"hook crashed: {r.stderr}")
    if not r.stdout.strip():
        return None
    return json.loads(r.stdout)["hookSpecificOutput"]["permissionDecisionReason"]


class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rdloop_guard_"))
        cls.mod = REPO / "SLS_setup" / "zz9wave"
        cls.cyc = cls.tmp / "cycle"
        (cls.cyc / "impl").mkdir(parents=True)
        cls.prefixes = [str(cls.mod) + "/", str(REPO / "scripts" / "zz9_"), str(cls.cyc / "impl") + "/", "/tmp/rdloop_ZZ9/"]
        (cls.tmp / "escape").symlink_to("/etc")

    def eng(self, tool, ti, **kw):
        return run_hook(tool, ti, "engineer", prefixes=self.prefixes, **kw)

    def ro(self, tool, ti, policy="readonly", **kw):
        return run_hook(tool, ti, policy, files=[str(self.cyc / "01_audit.md")], **kw)

    # ---------------------------------------------------------------- inactive outside rdloop steps
    def test_noop_without_step(self):
        self.assertIsNone(run_hook("Bash", {"command": "rm -rf /"}, step=None))

    # ---------------------------------------------------------------- write tools
    def test_engineer_writes_in_scope(self):
        self.assertIsNone(self.eng("Write", {"file_path": str(self.mod / "train.py")}))
        self.assertIsNone(self.eng("Edit", {"file_path": str(REPO / "scripts" / "zz9_local_eval.py")}))
        self.assertIsNone(self.eng("Write", {"file_path": str(self.cyc / "impl" / "progress.md")}))
        self.assertIsNone(self.eng("Write", {"file_path": "/tmp/rdloop_ZZ9/implement_core/x.py"}))

    def test_engineer_writes_out_of_scope(self):
        for p in (REPO / "SLS_setup" / "b3wave" / "train.py", REPO / "SLS_setup" / "utils" / "rtc_augment.py",
                  REPO / "scripts" / "b3_smoke.sh", REPO / ".claude" / "agents" / "x.md", self.cyc / "state.json",
                  Path(str(self.mod) + "_evil") / "x.py"):
            self.assertIsNotNone(self.eng("Write", {"file_path": str(p)}), p)

    def test_engineer_dotdot_and_symlink_escape(self):
        self.assertIsNotNone(self.eng("Write", {"file_path": str(self.mod / ".." / "b3wave" / "train.py")}))
        self.assertIsNotNone(self.eng("Write", {"file_path": str(self.tmp / "escape" / "passwd")}))

    def test_readonly_writes_only_its_file(self):
        self.assertIsNone(self.ro("Write", {"file_path": str(self.cyc / "01_audit.md")}))
        self.assertIsNotNone(self.ro("Write", {"file_path": str(self.cyc / "02_sota.md")}))
        self.assertIsNotNone(self.ro("Edit", {"file_path": str(self.mod / "train.py")}))

    # ---------------------------------------------------------------- reads
    def test_progress_scores_never_read(self):
        for pol in ("engineer", "readonly", "critic", "scout"):
            self.assertIsNotNone(run_hook("Read", {"file_path": "/x/SLS_setup/exp/r/model_merging_top5/scores.txt"}, pol))
            self.assertIsNotNone(run_hook("Bash", {"command": "head /mnt/paml-research/RTCFake/data/progress.txt"}, pol))
        self.assertIsNone(run_hook("Read", {"file_path": "/x/SLS_setup/exp/r/local_eval/epoch_8/metrics.json"}, "readonly"))

    # ---------------------------------------------------------------- web
    def test_web(self):
        self.assertIsNone(run_hook("WebFetch", {"url": "https://arxiv.org/abs/2604.23742"}, "scout"))
        self.assertIsNone(run_hook("WebFetch", {"url": "https://www.isca-archive.org/x.pdf"}, "critic"))
        self.assertIsNotNone(run_hook("WebFetch", {"url": "https://evil.example.com/a"}, "scout"))
        self.assertIsNotNone(run_hook("WebFetch", {"url": "https://arxiv.org/abs/1"}, "engineer"))
        self.assertIsNotNone(run_hook("WebSearch", {"query": "x"}, "readonly"))
        self.assertIsNone(run_hook("WebSearch", {"query": "x"}, "scout"))

    # ---------------------------------------------------------------- bash: every policy
    def test_bash_common_denies(self):
        bad = ["pip install foo", "python -m pip install foo", "sudo apt-get install x", "conda install x",
               "git commit -m x", "git -C /repo push", "git checkout -- a.py", "git stash", "kill 123", "pkill -f train",
               "nohup python x.py &", "claude -p hi", "curl https://x", "wget x", "ln -s a b", "chmod +x a",
               "python -m b3wave.train --lr 1", "/root/.conda/envs/rtc-sdd/bin/python -u -m b4wave.train --track x",
               "cd SLS_setup && python -m zz9wave.train --num_epochs 1",
               "python scripts/local_eval.py --model_path x --out y", "$PY ../scripts/select_topk.py --run x",
               "bash scripts/run_B3e.sh", "bash scripts/b3_smoke_train.sh a", "scripts/exploop/loopctl.sh start B3",
               "python scripts/rdloop/supervisor.py run --cycle x", "python -m zz9wave.checks gpu --ckpt x",
               "python -m b3wave.checks model", "CUDA_VISIBLE_DEVICES=0 python x.py", "export CUDA_VISIBLE_DEVICES=0",
               "env -u CUDA_VISIBLE_DEVICES python x.py", "sed -i s/a/b/ SLS_setup/b3wave/train.py",
               "python -c \"import os; os.environ['CUDA_VISIBLE_DEVICES']='0'\"",
               "python -c \"import os; os.environ.update(CUDA_VISIBLE_DEVICES='0')\""]
        for pol in ("engineer", "readonly", "critic"):
            for cmd in bad:
                self.assertIsNotNone(run_hook("Bash", {"command": cmd}, pol, prefixes=self.prefixes), f"{pol}: {cmd}")

    def test_bash_reads_allowed(self):
        ok = ["grep -n 'pip install' docs/augmentations.md", "grep -rn local_eval.py scripts/", "cat SLS_setup/b3wave/train.py",
              "git status --porcelain", "git diff --no-index a b", "git log -3 --oneline", "ls SLS_setup/exp | head",
              "python -m b3wave.train --help", "python -m zz9wave.train --help | grep -- --new_flag",
              "python scripts/local_eval.py --help", "python -c 'import json; print(json.load(open(\"a.json\"))[\"x\"])'",
              "wc -l docs/experiment_log.csv", "python3 -I -c 'import sys; print(sys.version)'",
              "echo \"CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]\"", "printenv CUDA_VISIBLE_DEVICES",
              "echo look at this; echo done", "python -m b3wave.checks all"]
        for pol in ("engineer", "readonly", "critic"):
            for cmd in ok:
                self.assertIsNone(run_hook("Bash", {"command": cmd}, pol, prefixes=self.prefixes), f"{pol}: {cmd}")

    def test_scout_has_no_shell(self):
        self.assertIsNotNone(run_hook("Bash", {"command": "ls"}, "scout"))

    def test_background_denied(self):
        self.assertIsNotNone(self.eng("Bash", {"command": "sleep 1", "run_in_background": True}))

    # ---------------------------------------------------------------- bash: engineer file ops
    def test_engineer_file_ops_scoped(self):
        ok = [f"mkdir -p {self.mod}/sub", f"touch {self.mod}/a.py", f"cp SLS_setup/b3wave/augment.py {self.mod}/augment_base.py",
              f"rm -f {self.cyc}/impl/old.json", f"python -m zz9wave.checks cpu > {self.cyc}/impl/cpu.log 2>&1",
              f"echo x > /dev/null", f"mv {self.mod}/a.py {self.mod}/b.py", "mkdir -p $TMPDIR/x"]
        for cmd in ok:
            self.assertIsNone(self.eng("Bash", {"command": cmd}, extra_env={"TMPDIR": "/tmp/rdloop_ZZ9/implement_core"}), cmd)
        bad = ["rm -rf SLS_setup/b3wave", f"cp {self.mod}/train.py SLS_setup/b3wave/train.py", "touch SLS_setup/utils/x.py",
               f"mv SLS_setup/b3wave/train.py {self.mod}/", "echo hi > SLS_setup/model/x.py", "tee scripts/b3_x.sh < /dev/null",
               "python -c \"import shutil; shutil.rmtree('SLS_setup/b3wave')\"", f"rm {self.tmp}/escape/hosts",
               "mkdir -p rdloop/OTHER/impl"]
        for cmd in bad:
            self.assertIsNotNone(self.eng("Bash", {"command": cmd}), cmd)

    def test_readonly_file_ops(self):
        self.assertIsNotNone(self.ro("Bash", {"command": "echo x > rdloop/x.md"}))
        self.assertIsNotNone(self.ro("Bash", {"command": "cp a b"}))
        self.assertIsNone(self.ro("Bash", {"command": "python - <<'EOF'\nimport json\nprint(1)\nEOF"}))
        self.assertIsNotNone(self.ro("Bash", {"command": "python - <<'EOF'\nopen('x','w').write('1')\nEOF"}))
        self.assertIsNone(self.ro("Bash", {"command": "sort a.txt > /tmp/rdloop_x/sorted.txt"}))


if __name__ == "__main__":
    unittest.main()
