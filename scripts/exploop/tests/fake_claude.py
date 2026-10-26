#!/usr/bin/env python3
"""Stand-in for `claude -p ... --output-format json --json-schema ...` in the exploop tests.

Reads EXPLOOP_STEP / EXPLOOP_WAVE_DIR / EXPLOOP_ITER / EXPLOOP_REPORT_PATH like the real steps,
writes the analyst's report, and prints a result object with `structured_output`.
env: EXPLOOP_FAKE_TWO_FACTORS=1 (design changes two unrelated flags -> rejected by the validator),
EXPLOOP_FAKE_CLAUDE_S (sleep, default 1), EXPLOOP_FAKE_DESIGN_EXPECT (extra expect regex).
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path


def main():
    argv = sys.argv[1:]
    sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else str(uuid.uuid4())
    step = os.environ["EXPLOOP_STEP"]
    wave = json.loads((Path(os.environ["EXPLOOP_WAVE_DIR"]) / "wave.json").read_text())
    k = int(os.environ.get("EXPLOOP_ITER", "0"))
    time.sleep(float(os.environ.get("EXPLOOP_FAKE_CLAUDE_S", "1")))
    if step == "design":
        letter = wave["letters"][k - 1]
        queue = [x for x in wave["lever_queue"] if "<derive>" not in json.dumps(x)]
        lever = queue[(k - 1) % len(queue)]
        change = {"set": lever["set"], "remove": []}
        if os.environ.get("EXPLOOP_FAKE_TWO_FACTORS"):
            change = {"set": {"--aug_codec_profiles": ["random"], "--specaug": []}, "remove": []}
        flag = next(iter(change["set"]))
        prompt = Path(os.environ.get("EXPLOOP_PROMPT", "/nonexistent"))
        n_rej = prompt.read_text().count("early check of the real run failed") if prompt.is_file() else 0
        suffix = f"v{n_rej + 1}" if n_rej else ""
        out = {"iter": k, "letter": letter, "track": f"{wave['track_prefix']}{letter}_fake_{lever['id']}{suffix}_s1234",
               "lever_slug": lever["id"].replace("_", ""), "kind": "config", "base_label": "x", "change": change,
               "change_vs_ref": f"+ {flag} {' '.join(change['set'][flag])}", "rationale": "test", "hypothesis": "test",
               "predicted_signature": "test", "falsifier": "test", "eval_note": "test", "header_text": "test header",
               "expect": [f"^FLAG {flag}="] + ([os.environ["EXPLOOP_FAKE_DESIGN_EXPECT"]] if os.environ.get("EXPLOOP_FAKE_DESIGN_EXPECT") else []),
               "expected_hours": 0.01}
    elif step in ("analyze", "summary"):
        rp = Path(os.environ["EXPLOOP_REPORT_PATH"])
        body = "# fake report\n\n" + "".join(f"## {i}. Section {i}\n\ntext\n\n" for i in range(1, 8))
        rp.write_text(body)
        if step == "analyze":
            out = {"report_path": str(rp), "headline": "fake", "went_ok": ["a"], "went_wrong": ["b"],
                   "largest_remaining_error": "noisy bona",
                   "proposals": [{"rank": 1, "kind": "seed_repeat", "lever": "seed", "change": {"set": {"--seed": ["2345"]}, "remove": []},
                                  "targets": "noise", "hypothesis": "h", "predicted_signature": "p", "falsifier": "f"},
                                 {"rank": 2, "kind": "config", "lever": "specaug", "change": {"set": {"--specaug": []}, "remove": []},
                                  "targets": "overfit", "hypothesis": "h", "predicted_signature": "p", "falsifier": "f"},
                                 {"rank": 3, "kind": "config", "lever": "snr", "change": {"set": {"--aug_snr_db": ["0", "20"]}, "remove": []},
                                  "targets": "snr0", "hypothesis": "h", "predicted_signature": "p", "falsifier": "f"}]}
        else:
            out = {"report_path": str(rp), "headline": "fake summary", "best_label": "x",
                   "recommended_full_run": {"base_run": "x", "flags": "--x", "seeds": [1234, 2345], "readout": "final"},
                   "follow_ups": ["y"]}
    else:
        sys.exit(f"unknown step {step}")
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done",
                      "structured_output": out, "total_cost_usd": 0.01, "session_id": sid, "permission_denials": []}))


if __name__ == "__main__":
    main()
