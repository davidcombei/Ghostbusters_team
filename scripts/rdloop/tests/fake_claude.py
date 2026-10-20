#!/usr/bin/env python3
"""Stand-in for `claude -p ... --agent research-* --output-format json --json-schema ...` in the rdloop
tests. Reads RDLOOP_STEP / RDLOOP_CYCLE_DIR / RDLOOP_PROMPT like the real steps, writes the step's
files, prints a result envelope with `structured_output`.

env switches:
  RDLOOP_FAKE_CRITIC_ROUNDS=k   the critic agrees in round k (default 2: one revision)
  RDLOOP_FAKE_NEVER_AGREE=1     blocker on head_y + major on aug_x every round
  RDLOOP_FAKE_ESCAPE=1          the engineer writes SLS_setup/utils/evil.py (outside its scope) from python
  RDLOOP_FAKE_BREAK_DEFAULT=1   --aug_x defaults to True (levers-off != base) until the first fix
  RDLOOP_FAKE_NEEDS_USER=1      the core implementation needs a package the owner must install
  RDLOOP_FAKE_REVIEW_FINDING=1  review round 1 has a major finding (fixed by the next fix round)
  RDLOOP_FAKE_CLAUDE_S          seconds to sleep (default 0.3)
"""
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path


def env(k):
    return os.environ.get(k, "")


def write(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


LEVERS = {
    "aug_x": {"flag": {"name": "--aug_x", "type": "bool", "default": "False", "choices": [], "help": "fake augmentation"},
              "set": {"--aug_x": []}, "expect": ["^AugX: on"],
              "arg": 'b3.add_argument("--aug_x", action="store_true", default=@DEFAULT@)',
              "print": "print(f\"AugX: {'on' if args.aug_x else 'off'}\")"},
    "head_y": {"flag": {"name": "--head_y", "type": "float", "default": "0.0", "choices": [], "help": "fake head"},
               "set": {"--head_y": ["0.5"]}, "expect": ["^HeadY: 0.5"],
               "arg": 'b3.add_argument("--head_y", type=float, default=0.0)', "print": 'print(f"HeadY: {args.head_y:g}")'},
}


def lever(lid, v):
    L = LEVERS[lid]
    return {"id": lid, "title": f"fake {lid} v{v}", "category": "augmentation", "flags": [L["flag"]],
            "default_equivalence": "the flag defaults off; off = base code path", "mechanism": "fake mechanism",
            "gap_ids": ["G1"], "idea_ids": ["I1"], "sources": ["https://arxiv.org/abs/2604.23742"], "hypothesis": "h",
            "predicted_signature": "p", "falsifier": "f", "expect_regex": L["expect"], "state_dict_change": False,
            "params_added": False, "dependencies": [], "files": ["train.py"], "equivalence_test": "S0 == base",
            "cost": {"est_loc": 50, "extra_vram_gib": 0.1, "epoch_time_frac": 1.05}, "smoke": {"minutes_est": 1, "checks": ["runs"]},
            "wave_lever": {"set": L["set"], "coupled": [], "targets": "G1"}, "risks": []}


def main():
    argv = sys.argv[1:]
    sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else str(uuid.uuid4())
    time.sleep(float(env("RDLOOP_FAKE_CLAUDE_S") or 0.3))
    step = env("RDLOOP_STEP")
    cdir = Path(env("RDLOOP_CYCLE_DIR"))
    cyc = json.loads((cdir / "cycle.json").read_text())
    repo = Path(cyc["repo"])
    mod = cyc["module"]
    tag = Path(env("RDLOOP_PROMPT")).name.split(".")[0].split("_", 1)[1]
    mdir = repo / "SLS_setup" / mod
    if step == "audit":
        write(cdir / "01_audit.md", "# audit\n\n" + "evidence line\n" * 30)
        out = {"report_path": str(cdir / "01_audit.md"), "headline": "fake audit",
               "best": {"label": "W1a", "run_dir": "x", "readout_merge": "93", "readout_final": "92", "source": "metrics.json"},
               "gaps": [{"id": f"G{i}", "title": f"gap {i}", "evidence": [{"file": "f", "what": "w", "value": "1"}], "slices": [],
                         "size_note": "", "causes": [], "priority": i, "already_tried": [], "addressable_by": ["augmentation"]}
                        for i in (1, 2)],
               "exhausted_levers": [], "constraints": ["one 4090"]}
    elif step == "scout":
        write(cdir / "02_sota.md", "# sota\n\n" + "paper line\n" * 30)
        out = {"report_path": str(cdir / "02_sota.md"), "headline": "fake scout", "queries": ["rtc deepfake"],
               "ideas": [{"id": f"I{i}", "title": f"idea {i}", "category": "augmentation", "gap_ids": ["G1"], "mechanism": "m",
                          "sources": [{"title": "RTCFake", "url": "https://arxiv.org/abs/2604.23742", "venue_year": "2026", "quote": "q"}],
                          "reported_effect": {"dataset": "d", "metric": "EER", "delta": "-1", "quote": "q"}, "transfer_risk": "r",
                          "sketch": "s", "est_loc": 50, "state_dict_change": False, "dependencies": [], "extra_vram_gib": 0.1,
                          "extra_time_frac": 1.05, "novelty_vs_tried": "n"} for i in range(1, 5)],
               "rejected": []}
    elif step == "propose":
        v = int(re.match(r"v(\d+)", tag).group(1))
        ledger = json.loads((cdir / "debate" / "ledger.json").read_text()) if (cdir / "debate" / "ledger.json").is_file() else {}
        open_ids = [k for k, o in ledger.items() if o["status"] == "open"]
        write(cdir / "debate" / f"proposal_v{v}.md", f"# proposal v{v}\n\n" + "plan line\n" * 30)
        out = {"version": v, "report_path": str(cdir / "debate" / f"proposal_v{v}.md"), "headline": f"fake proposal v{v}",
               "base_label": "W1a", "levers": [lever("aug_x", v), lever("head_y", v)], "shared_changes": [],
               "eval_impact": {"state_dict_change": False, "new_profile": False, "scripts": []},
               "responses": [{"objection_id": o, "action": "accept", "detail": "done"} for o in open_ids],
               "alternatives_response": [], "dropped": []}
    elif step == "critique":
        v = int(re.match(r"v(\d+)", tag).group(1))
        ledger = json.loads((cdir / "debate" / "ledger.json").read_text()) if (cdir / "debate" / "ledger.json").is_file() else {}
        open_ids = [k for k, o in ledger.items() if o["status"] == "open"]
        k = int(env("RDLOOP_FAKE_CRITIC_ROUNDS") or 2)
        objs, resolved, still = [], [], []
        if env("RDLOOP_FAKE_NEVER_AGREE"):
            if not ledger:
                objs = [{"id": "O1.1", "lever_id": "head_y", "severity": "blocker", "category": "leakage", "claim": "leaks",
                         "evidence": ["utils/sim_ops.py:10"], "required_change": "drop it"},
                        {"id": "O1.2", "lever_id": "aug_x", "severity": "major", "category": "budget", "claim": "slow",
                         "evidence": ["x"], "required_change": "cheaper"}]
            still = [{"id": i, "note": "not addressed"} for i in open_ids]
        elif v < k:
            objs = [{"id": f"O{v}.1", "lever_id": "aug_x", "severity": "major", "category": "evidence", "claim": "weak",
                     "evidence": ["02_sota.md:3"], "required_change": "cite"}]
            resolved = open_ids
        else:
            resolved = open_ids
        write(cdir / "debate" / f"critique_v{v}.md", f"# critique v{v}\n\n" + "objection line\n" * 20)
        out = {"version": v, "report_path": str(cdir / "debate" / f"critique_v{v}.md"), "headline": f"fake critique v{v}",
               "objections": objs, "resolved": resolved, "still_open": still,
               "per_lever": [{"lever_id": "aug_x", "stance": "keep", "note": ""}], "alternatives": [], "citation_checks": [],
               "verdict": "agree" if not objs and not still else "revise"}
    elif step in ("implement", "fix"):
        part = tag if step == "implement" else "fix"
        files = []
        needs = []
        if part == "core":
            write(mdir / "checks.py", "import sys\nprint('PASS', sys.argv[1:])\nsys.exit(0)\n")
            write(mdir / "README.md", f"# {mod}\n\n| flag | default |\n|---|---|\n")
            files = [str(mdir / "checks.py"), str(mdir / "README.md")]
            if env("RDLOOP_FAKE_ESCAPE"):
                (repo / "SLS_setup" / "utils" / "evil.py").write_text("escaped")
            if env("RDLOOP_FAKE_NEEDS_USER"):
                needs = [{"what": "pip install fancy_pkg", "why": "the core needs it"}]
        elif part in LEVERS:
            t = (mdir / "train.py").read_text()
            arg = LEVERS[part]["arg"].replace("@DEFAULT@", "True" if env("RDLOOP_FAKE_BREAK_DEFAULT") else "False")
            t = t.replace("# @LEVER_ARGS@", arg + "\n    # @LEVER_ARGS@")
            t = t.replace("# @LEVER_PRINTS@", LEVERS[part]["print"] + "\n    # @LEVER_PRINTS@")
            (mdir / "train.py").write_text(t)
            files = [str(mdir / "train.py")]
        else:   # fix: the broken default is repaired only when the GPU session reported it
            t = (mdir / "train.py").read_text()
            if "(reason: gpu)" in Path(env("RDLOOP_PROMPT")).read_text():
                t = t.replace('action="store_true", default=True', 'action="store_true", default=False')
            t += "\n# fixed\n"
            (mdir / "train.py").write_text(t)
            files = [str(mdir / "train.py")]
        prog = cdir / "impl" / "progress.md"
        write(prog, (prog.read_text() if prog.is_file() else "") + f"\n- {part} done\n")
        lv = [{"id": part, "flags": [{"name": LEVERS[part]["flag"]["name"], "default": LEVERS[part]["flag"]["default"]}],
               "implemented": True, "header_regex": LEVERS[part]["expect"], "where": ["train.py:main"], "deviations": ""}] \
            if part in LEVERS else []
        out = {"part": part, "headline": f"fake {part}", "files": [{"path": f, "purpose": "p"} for f in files], "levers": lv,
               "tests": [{"name": "cpu", "command": f"python -m {mod}.checks cpu", "device": "cpu", "what": "w"}],
               "cpu_results": [{"command": "x", "rc": 0, "summary": "ok"}], "equivalence_argument": "off = base",
               "addressed": ["F1.1"] if step == "fix" else [], "open_issues": [], "needs_user": needs, "progress_note": "ok"}
    elif step == "review":
        n = int(re.match(r"r(\d+)", tag).group(1))
        findings = []
        if env("RDLOOP_FAKE_REVIEW_FINDING") and n == 1:
            findings = [{"id": "F1.1", "lever_id": "aug_x", "severity": "major", "category": "default_off", "file": "train.py",
                         "line": 1, "issue": "x", "fix": "y"}]
        write(cdir / "review" / f"code_review_r{n}.md", f"# review r{n}\n\n" + "finding line\n" * 10)
        out = {"round": n, "report_path": str(cdir / "review" / f"code_review_r{n}.md"), "headline": "fake review",
               "verdict": "fix" if findings else "approve", "findings": findings, "resolved": ["F1.1"] if n > 1 else [],
               "plan_conformance": []}
    elif step == "summary":
        write(cdir / "FINAL_SUMMARY.md", "# final summary\n\n" + "summary line\n" * 40)
        out = {"report_path": str(cdir / "FINAL_SUMMARY.md"), "headline": "fake summary: ready", "status": "ready",
               "levers": [], "dropped": [], "recommended_wave": {"name": "W2", "order": ["aug_x"], "expected_hours": 8, "notes": ""},
               "owner_questions": [], "risks": []}
    else:
        sys.exit(f"unknown step {step}")
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done", "structured_output": out,
                      "total_cost_usd": 0.01, "session_id": sid, "permission_denials": []}))


if __name__ == "__main__":
    main()
