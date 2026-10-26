#!/usr/bin/env python3
"""exploop supervisor: runs a training-exploration wave unattended, one process per wave.

    supervisor.py run    --wave exploop/<W>     # the loop (start it with loopctl.sh start <W>)
    supervisor.py init   --wave exploop/<W>     # create loop_state.json from wave.json (start block)
    supervisor.py status --wave exploop/<W>     # health; exit 0 running, 1 down, 2 blocked, 3 done
    supervisor.py plan   --wave exploop/<W>     # what would run next (expanded commands), read-only
    supervisor.py tail   --wave exploop/<W> [-n 30]
    supervisor.py unblock --wave exploop/<W> [--reset-attempts]

Per iteration: DESIGN (headless claude, exp-designer) -> RENDER launcher from the template ->
[SMOKE, opt-in] -> TRAIN (wait for an idle GPU, launch, follow train.log, early check, --resume
after a silent death) -> EVAL (the post-train-eval skill's commands + final-epoch readout + CI, from
an eval profile) -> VERDICT (rules from wave.json) -> LOG (scripts/log_experiment.py, b3_log_ci.py)
-> ANALYZE (headless claude, exp-analyst, writes the report) -> DECIDE (update the best).
After the last iteration: SUMMARY (exp-analyst) -> DONE.

All state is on disk (loop_state.json, iterNN/*); every phase re-checks its outputs on disk, so a
killed supervisor (closed terminal, dropped SSH, container restart) continues where it stopped.
The supervisor never deletes run outputs and never kills a process it did not start.
"""
import argparse
import fcntl
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exploop_lib as L  # noqa: E402

PHASES = L.PHASES


class Blocked(Exception):
    pass


class Supervisor:
    def __init__(self, wave_dir):
        self.wave_dir = Path(wave_dir).resolve()
        self.wave = L.read_json(self.wave_dir / "wave.json")
        if not self.wave:
            sys.exit(f"no readable wave.json in {self.wave_dir}")
        self.repo = Path(self.wave["repo"]).resolve()
        self.state_path = self.wave_dir / "loop_state.json"
        self.events_path = self.wave_dir / "events.jsonl"
        self.st = L.read_json(self.state_path)
        self.py = self.wave.get("python", sys.executable)
        self.stop_flag = False
        self.lock_fh = None
        self._profile = None
        self._best_epoch_cache = {}
        self.readonly = False

    # ------------------------------------------------------------------------------ utilities
    def abs(self, p):
        p = Path(str(p))
        return p if p.is_absolute() else (self.repo / p)

    def rel(self, p):
        p = Path(p).resolve()
        try:
            return str(p.relative_to(self.repo))
        except ValueError:
            return str(p)

    def iter_dir(self, it):
        d = self.wave_dir / f"iter{it['k']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def event(self, kind, msg, it=None, **data):
        if self.readonly:
            return
        rec = {"ts": L.now_ts(), "iter": (it or {}).get("k"), "phase": (it or {}).get("phase"), "kind": kind, "msg": msg}
        if data:
            rec["data"] = data
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{rec['ts']}] {kind}: {msg}", flush=True)

    def save(self):
        L.write_json(self.state_path, self.st)

    def block(self, phase, reason, hint="", it=None):
        if self.readonly:
            raise Blocked(reason)
        self.st["status"] = "BLOCKED"
        self.st["blocked"] = {"phase": phase, "reason": reason, "hint": hint, "ts": L.now_ts(),
                              "iter": (it or {}).get("k")}
        self.event("blocked", f"{phase}: {reason}", it)
        self.save()
        self.render()
        raise Blocked(reason)

    def cfg(self, *keys, default=None):
        cur = self.wave
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def iteration(self, k):
        for it in self.st["iterations"]:
            if it["k"] == k:
                return it
        letters = self.wave.get("letters", "abcdefghij")
        it = {"k": k, "letter": letters[k - 1], "label": f"{self.wave['track_prefix']}{letters[k - 1]}",
              "phase": "design", "attempts": {}, "jobs": {}, "logs": [], "claude_sessions": [], "done": False,
              "adopted": False, "rejected_tracks": []}
        self.st["iterations"].append(it)
        self.event("iteration", f"iteration {k} ({it['label']}) started", it)
        return it

    def used_tracks(self, it):
        out = set()
        for i in self.st["iterations"]:
            out |= set(i.get("rejected_tracks", []))
            if i["k"] != it["k"] and i.get("track"):
                out.add(i["track"])
        return out

    def best_dir(self, key):
        return self.abs(self.st["best"]["readouts"][key])

    # ------------------------------------------------------------------------------ jobs
    def start_job(self, it, name, argv, cwd, stdout=None, env_extra=None, stderr=None):
        """Start argv in its own session. With `stdout`, output goes to stdout.exploop_tmp and is
        moved into place only when the command exits 0 (so an 'exists' check means success)."""
        jobs_dir = self.iter_dir(it) / "jobs" if it else self.wave_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        attempt = (it or self.st).setdefault("attempts", {}).get(name, 0)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        rc_file = jobs_dir / f"{safe}.a{attempt}.rc"
        if rc_file.exists():          # our own bookkeeping file from an earlier (reset) attempt
            rc_file.unlink()
        if stdout:
            final = self.abs(stdout)
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(final) + ".exploop_tmp")
            redirect = f"> {shlex.quote(str(tmp))} 2>&1"
            post = f"[ $rc -eq 0 ] && mv -f {shlex.quote(str(tmp))} {shlex.quote(str(final))}; "
            log = final
        else:
            log = jobs_dir / f"{safe}.a{attempt}.log"
            redirect = f"> {shlex.quote(str(log))} 2>" + (shlex.quote(str(stderr)) if stderr else "&1")
            post = ""
        script = (f"cd {shlex.quote(str(cwd))} && {shlex.join([str(a) for a in argv])} {redirect} </dev/null; "
                  f"rc=$?; {post}echo $rc > {shlex.quote(str(rc_file))}")
        env = dict(os.environ)
        env.update(env_extra or {})
        p = subprocess.Popen(["bash", "-c", script], start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        ticks = None
        for _ in range(50):
            ticks = L.proc_start_ticks(p.pid)
            if ticks is not None:
                break
            time.sleep(0.02)
        job = {"name": name, "pid": p.pid, "start_ticks": ticks, "rc_file": str(rc_file), "log": str(log),
               "started": time.time(), "started_ts": L.now_ts(), "attempt": attempt, "cmd": shlex.join([str(a) for a in argv])}
        (it or self.st).setdefault("jobs", {})[name] = job
        self.event("job_start", f"{name} (pid {p.pid}): {job['cmd'][:300]}", it)
        return job

    def job_alive(self, job):
        return L.proc_alive(job.get("pid"), job.get("start_ticks"))

    def job_rc(self, job):
        try:
            return int(Path(job["rc_file"]).read_text().strip())
        except (OSError, ValueError, KeyError):
            return None

    def kill_job(self, job, sig=signal.SIGTERM):
        """Signal the job's process group (only jobs this supervisor started have their own group)."""
        pid = job.get("pid")
        if not pid or not self.job_alive(job):
            return
        try:
            pgid = os.getpgid(pid)
            if pgid != os.getpgid(0):
                os.killpg(pgid, sig)
            else:
                os.kill(pid, sig)
        except OSError:
            pass

    def timeout_job(self, job, limit_s, it):
        """SIGINT -> 60 s -> SIGTERM -> 60 s -> SIGKILL once a job runs longer than limit_s."""
        el = time.time() - job["started"]
        if el < limit_s:
            return False
        stage = job.get("kill_stage", 0)
        since = time.time() - job.get("kill_ts", 0)
        if stage == 0:
            self.event("timeout", f"{job['name']} ran {el / 60:.0f} min > {limit_s / 60:.0f} min: SIGINT", it)
            self.kill_job(job, signal.SIGINT)
            job.update(kill_stage=1, kill_ts=time.time())
        elif stage == 1 and since > 60:
            self.kill_job(job, signal.SIGTERM)
            job.update(kill_stage=2, kill_ts=time.time())
        elif stage == 2 and since > 60:
            self.kill_job(job, signal.SIGKILL)
            job.update(kill_stage=3, kill_ts=time.time())
        return True

    # ------------------------------------------------------------------------------ GPU / preflight
    def ready_for_gpu(self, kind, it=None):
        if (self.wave_dir / "PAUSE").exists():
            self.st["status"] = "PAUSED"
            self.st["waiting"] = "paused by the user (PAUSE file): no new GPU job starts"
            return False
        g = self.wave.get("gpu", {})
        if g.get("enabled", True):
            mem = L.gpu_memory_used_mib()
            busy = L.busy_processes(g.get("busy_regex", "main_train|b3wave\\.train|local_eval|main_eval|eval_checkpoints"))
            if mem is None:
                self.st.update(status="WAITING", waiting="nvidia-smi failed; cannot verify the GPU is idle")
                self.st["gpu_idle_since"] = None
                return False
            if any(m >= int(g.get("max_mem_mib", 1024)) for m in mem) or busy:
                self.st.update(status="WAITING", waiting=f"GPU busy: memory.used {mem} MiB; processes "
                                                          f"{[p for p, _ in busy][:5]} {[c for _, c in busy][:2]}")
                self.st["gpu_idle_since"] = None
                return False
            need = int(g.get("debounce_s", 600)) if kind == "train" else int(g.get("debounce_eval_s", 30))
            if not self.st.get("gpu_idle_since"):
                self.st["gpu_idle_since"] = time.time()
            idle_for = time.time() - self.st["gpu_idle_since"]
            if idle_for < need:
                self.st.update(status="WAITING", waiting=f"GPU idle for {idle_for:.0f}s; need {need}s before a {kind} job")
                return False
        free = L.disk_free_gb(self.repo)
        need_gb = float(self.cfg("disk_gb", kind, default=0) or 0)
        if free is not None and free < need_gb:
            self.st.update(status="WAITING", waiting=f"disk: {free:.0f} GB free < {need_gb:.0f} GB needed for {kind}")
            return False
        self.st.update(status="RUNNING", waiting=None, gpu_idle_since=None)
        return True

    def preflight(self, it):
        for check in self.cfg("trainer", "preflight", default=[]) or []:
            if check == "ffmpeg":
                if not self.ffmpeg_ok():
                    cmd = self.cfg("trainer", "ffmpeg_install_cmd")
                    if not cmd:
                        self.block("train", "ffmpeg (with libopus) missing", "install ffmpeg, then resume", it)
                    self.event("preflight", f"ffmpeg missing (container restart?): running {cmd}", it)
                    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
                    if r.returncode != 0 or not self.ffmpeg_ok():
                        self.block("train", "ffmpeg reinstall failed: " + (r.stderr or r.stdout)[-500:],
                                   "apt-get install -y ffmpeg by hand, then resume", it)
                    self.event("preflight", "ffmpeg reinstalled", it)
            elif check == "pyloudnorm":
                r = subprocess.run([self.py, "-c", "import pyloudnorm"], capture_output=True, text=True)
                if r.returncode != 0:
                    self.block("train", "pyloudnorm not importable by the run python",
                               f"{self.py} -m pip install pyloudnorm==0.2.0, then resume", it)

    def ffmpeg_ok(self):
        exe = os.environ.get("EXPLOOP_FFMPEG", "ffmpeg")
        if not shutil.which(exe):
            return False
        try:
            r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0 and "libopus" in r.stdout

    # ------------------------------------------------------------------------------ claude steps
    def can_spend(self, usd):
        cap = float(self.cfg("claude", "max_total_usd", default=150))
        return float(self.st.get("cost_usd", 0.0)) + float(usd) <= cap

    def claude_step(self, it, step, prompt_fn, schema_name, allowed_writes, extra_env=None):
        """Run one headless claude step as a job. Returns None while running, else (ok, payload|errors)."""
        sc = self.cfg("claude", "steps", step)
        holder = it if it else self.st
        job = holder.setdefault("jobs", {}).get(step)
        if job:
            if self.job_alive(job):
                self.timeout_job(job, float(sc.get("timeout_min", 30)) * 60, it)
                return None
            holder["jobs"].pop(step)
            return self.finish_claude(it, step, job)
        # not running: start (respecting attempts, budget, backoff)
        att = holder.setdefault("attempts", {})
        if att.get(step, 0) >= int(sc.get("attempts", 3)):
            self.block(step, f"{step} failed {att.get(step)} times: {holder.get(step + '_errors', [])[:3]}",
                       f"look at {self.rel(self.iter_dir(it) if it else self.wave_dir)}/{step}.a*.out.json / .err; "
                       f"fix, then `loopctl.sh resume {self.wave['wave']} --reset-attempts`", it)
        if time.time() < holder.get("backoff_until", 0):
            self.st["waiting"] = f"{step}: backing off after an API error until {time.ctime(holder['backoff_until'])}"
            return None
        if not self.can_spend(sc.get("usd", 10)):
            self.block(step, f"Claude budget: spent ${self.st.get('cost_usd', 0):.2f}, cap "
                             f"${self.cfg('claude', 'max_total_usd')}", "raise claude.max_total_usd in wave.json", it)
        n = att.get(step, 0)
        prompt_path = prompt_fn()          # built only when a new attempt starts
        att[step] = n + 1
        d = self.iter_dir(it) if it else self.wave_dir
        out = d / f"{step}.a{n}.out.json"
        err = d / f"{step}.a{n}.err"
        sid = str(uuid.uuid4())
        schema = (HERE / "schemas" / f"{schema_name}.json").read_text(encoding="utf-8")
        prompt = (f"Read {prompt_path} and do exactly what it says. You are an unattended step of the exploop "
                  f"supervisor: nobody can answer questions, so never ask; finish with the structured output.")
        argv = [self.cfg("claude", "bin", default="claude"), "-p", prompt,
                "--agent", sc["agent"],
                "--settings", str(self.abs(self.cfg("claude", "settings"))),
                "--permission-mode", self.cfg("claude", "permission_mode", default="auto"),
                "--permission-prompts", "none",
                "--output-format", "json", "--json-schema", json.dumps(json.loads(schema)),
                "--max-budget-usd", str(sc.get("usd", 10)),
                "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
                "--session-id", sid, "-n", f"exploop-{self.wave['wave']}-{(it or {}).get('label', 'wave')}-{step}-a{n}"]
        if self.cfg("claude", "model"):
            argv += ["--model", self.cfg("claude", "model")]
        env = {"EXPLOOP_STEP": step, "EXPLOOP_WAVE_DIR": str(self.wave_dir), "EXPLOOP_REPO": str(self.repo),
               "EXPLOOP_ITER": str((it or {}).get("k", 0)), "EXPLOOP_PROMPT": str(prompt_path),
               "EXPLOOP_ALLOWED_WRITES": ":".join(str(Path(p).resolve()) for p in allowed_writes),
               "DISABLE_AUTOUPDATER": "1"}
        env.update(extra_env or {})
        holder["snapshot_" + step] = self.snapshot(it)
        job = self.start_job(it, step, argv, self.repo, env_extra=env, stderr=err)
        job.update(out=str(out), err=str(err), session_id=sid)
        # stdout of the job is the JSON result: point the job log at out.json
        holder.setdefault("claude_sessions", []).append({"step": step, "attempt": n, "session_id": sid,
                                                         "status": "running", "out": self.rel(out)})
        return None

    def finish_claude(self, it, step, job):
        holder = it if it else self.st
        out_path = Path(job.get("out"))
        # the job wrote stdout into its .log; copy it to the out.json path the step record names
        log = Path(job["log"])
        if log.is_file():
            shutil.copyfile(log, out_path)
        res = L.read_json(out_path) or {}
        sess = next((s for s in reversed(holder.get("claude_sessions", [])) if s["session_id"] == job.get("session_id")), {})
        cost = float(res.get("total_cost_usd") or 0.0)
        self.st["cost_usd"] = round(float(self.st.get("cost_usd", 0.0)) + cost, 4)
        sess.update(cost=cost)
        errs = []
        violations = self.integrity_violations(it, holder.pop("snapshot_" + step, None), step)
        in_runs = [v for v in violations if v.startswith("/")]           # run-dir files (absolute keys)
        others = [v for v in violations if not v.startswith("/")]        # repo files under protected prefixes
        if in_runs:
            sess["status"] = "integrity violation"
            self.block(step, f"files in the run dirs changed during the {step} step: {in_runs[:8]}",
                       "inspect them (nothing was reverted automatically); if it was you, resume", it)
        if others:
            # deny rules + the guard hook stop the step from writing these; a change here is most likely a
            # concurrent human edit, so warn instead of blocking the wave
            msg = f"protected repo files changed while the {step} step ran (you, or the step?): {others[:8]}"
            self.event("integrity_warning", msg, it)
            self.st.setdefault("warnings", []).append(f"{L.now_ts()} {msg}")
        if not res:
            errs.append(f"no JSON result (rc {self.job_rc(job)}); stderr: {L.tail_bytes(job.get('err', ''), 800)}")
        elif res.get("is_error"):
            msg = str(res.get("result", ""))[:500]
            errs.append(f"claude reported an error: {msg}")
            if re.search(r"rate.?limit|overloaded|529|500|timeout|ECONN|network|auth", msg, re.I):
                mins = [10, 30, 60][min(holder.get("attempts", {}).get(step, 1) - 1, 2)]
                holder["backoff_until"] = time.time() + 60 * mins
        payload = None
        if not errs:
            payload = res.get("structured_output")
            if payload is None:
                payload = _json_from_text(str(res.get("result", "")))
            if not isinstance(payload, dict):
                errs.append("no structured output in the result")
        if res.get("permission_denials"):
            self.event("permission_denials", f"{step}: {len(res['permission_denials'])} tool calls denied", it,
                       denials=res["permission_denials"][:10])
        sess["status"] = "ok" if not errs else "failed"
        if errs:
            holder.setdefault(step + "_errors", []).extend(errs)
            self.event("step_failed", f"{step} attempt {job.get('attempt')}: {errs[0][:300]}", it)
            return (False, errs)
        self.event("step_done", f"{step} attempt {job.get('attempt')} finished (${cost:.2f})", it)
        return (True, payload)

    def snapshot(self, it):
        """Files a claude step must not change: git-visible files under protected prefixes plus the
        current and best run dirs (gitignored)."""
        prot = self.cfg("integrity", "protected", default=["SLS_setup/", "scripts/", "configs/", ".claude/", "docs/experiment_log.csv"])
        snap = {}
        try:
            out = subprocess.run(["git", "status", "--porcelain=v1", "-uall"], cwd=self.repo, capture_output=True,
                                 text=True, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        for line in out.splitlines():
            path = line[3:].strip().strip('"')
            if " -> " in path:
                path = path.split(" -> ")[-1]
            if any(path.startswith(p) for p in prot):
                fp = self.repo / path
                snap[path] = [fp.stat().st_size, fp.stat().st_mtime_ns] if fp.exists() else None
        for d in {self.st.get("best", {}).get("run_dir"), (it or {}).get("run_dir")} - {None}:
            root = self.abs(d)
            if root.is_dir():
                for fp in root.rglob("*"):
                    if fp.is_file():
                        st = fp.stat()
                        snap[str(fp)] = [st.st_size, st.st_mtime_ns]
        return snap

    def integrity_violations(self, it, before, step):
        if before is None:
            return []
        after = self.snapshot(it)
        return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))

    # ------------------------------------------------------------------------------ phases
    def ph_design(self, it):
        dpath = self.iter_dir(it) / "design.result.json"
        if it.get("argv") and dpath.is_file():
            return True
        r = self.claude_step(it, "design", lambda: self.write_design_prompt(it), "design", allowed_writes=[])
        if r is None:
            return False
        ok, payload = r
        if not ok:
            return False
        design = payload
        errs = []
        missing = [k for k in ("track", "lever_slug", "change", "change_vs_ref", "hypothesis", "falsifier") if not design.get(k)]
        if missing:
            errs.append(f"design is missing {missing}")
        base = self.st["best"]["argv"]
        cand = None
        if not errs:
            design["letter"] = it["letter"]
            design["iter"] = it["k"]
            try:
                cand = self.candidate_argv(base, design)
            except (ValueError, TypeError) as exc:
                errs.append(f"cannot apply the change: {exc}")
        if not errs:
            known = L.trainer_flags([self.abs(s) for s in self.cfg("trainer", "flag_sources", default=[])])
            errs = L.validate_design(self.wave, base, cand, design, self.used_tracks(it), known)
        if errs:
            it.setdefault("design_errors", []).extend(errs)
            self.event("design_rejected", "; ".join(errs)[:500], it)
            return False
        L.write_json(dpath, design)
        it.update(argv=cand, track=design["track"], lever_slug=design["lever_slug"], change_vs_ref=design["change_vs_ref"],
                  kind=design.get("kind", "config"), expect=design.get("expect", []), design=self.rel(dpath),
                  base_label=self.st["best"]["label"], argv_checked=False)
        self.event("design", f"{it['label']}: {design['change_vs_ref']} (track {design['track']})", it)
        return True

    def candidate_argv(self, base, design):
        cand_ne = (design["change"].get("set") or {}).get("--num_epochs")
        ne = cand_ne[0] if cand_ne else (L.argv_dict(base).get("--num_epochs") or ["8"])[0]
        overrides = {"--track": [design["track"]], "--keep_epochs": [str(ne)],
                     "--keep_topk_by_wf1": [str(self.wave["search_space"].get("keep_topk_by_wf1", 7))]}
        for flag in ("--track", "--keep_epochs"):
            (design["change"].get("set") or {}).pop(flag, None)
        return L.apply_change(base, design["change"], overrides)

    def ph_render(self, it):
        if it.get("adopted"):
            return True
        path = self.launcher_path(it)
        text = self.render_launcher(it)
        sha = L.sha256_text(text)
        if path.is_file() and L.sha256_file(path) == sha and it.get("argv_checked"):
            return True
        if path.is_file() and L.sha256_file(path) != sha and L.sha256_file(path) not in it.get("rendered_shas", []):
            self.block("render", f"{self.rel(path)} exists and was not written by exploop for this design",
                       "move it away (or choose other letters in wave.json), then resume", it)
        L.write_text(path, text)
        os.chmod(path, 0o755)
        it.setdefault("rendered_shas", []).append(sha)
        r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            self.block("render", f"bash -n failed: {r.stderr[-400:]}", "bug in launcher_template.sh", it)
        r = subprocess.run(["bash", str(path)], capture_output=True, text=True, env={**os.environ, "EXPLOOP_PRINT_ARGV": "1"})
        printed = [x for x in r.stdout.split("\n") if x != ""]
        if printed != [str(x) for x in it["argv"]]:
            self.block("render", "the rendered launcher's argv differs from the design argv",
                       f"diff: {set(printed) ^ set(it['argv'])}", it)
        it["argv_checked"] = True
        it["launcher"] = self.rel(path)
        self.event("render", f"wrote {self.rel(path)}", it)
        return True

    def launcher_path(self, it):
        pat = self.cfg("trainer", "launcher", default="scripts/run_{prefix}{letter}.sh")
        return self.abs(pat.format(prefix=self.wave["track_prefix"], letter=it["letter"], label=it["label"]))

    def render_launcher(self, it):
        design = L.read_json(self.iter_dir(it) / "design.result.json") or {}
        tpl = (HERE / "launcher_template.sh").read_text(encoding="utf-8")
        best = self.st["best"]
        k = self.cfg("eval", "k", default=5)
        merge_ref = L.readout_numbers(self.best_dir("merge") / "metrics.json", self.objective_key(), self.clean_key())
        rules = self.wave.get("rules", {})
        band = rules.get("band", 0.5)
        header = [f"# {it['label']} = {best['label']} with ONE factor changed: {design.get('change_vs_ref', '')}",
                  f"#   generated by exploop (wave {self.wave['wave']}, iteration {it['k']}) from "
                  f"{self.rel(self.iter_dir(it) / 'design.result.json')}"]
        for key, title in (("header_text", ""), ("rationale", "Why: "), ("hypothesis", "Hypothesis: "),
                           ("predicted_signature", "Expected signature: "), ("derivation", "Derivation: "),
                           ("eval_note", "Eval note: ")):
            if design.get(key):
                header += _wrap_comment(title + str(design[key]))
        ref_txt = f"{merge_ref['value']:.2f}" if merge_ref else "?"
        header += _wrap_comment(
            f"Falsification (pre-registered): delta top-{k}-merge {self.objective_key()} vs {best['label']} merge ({ref_txt}) "
            f"> +{rules.get('positive', {}).get('delta_gt', band)} with the paired-bootstrap CI excluding 0 and clean F1 drop "
            f"<= {rules.get('positive', {}).get('clean_drop_le', 0.5)} -> positive; within +-{band} -> null; "
            f"below -{band} -> negative. {design.get('falsifier', '')}")
        trainer = self.wave["trainer"]
        preflight = []
        for check in trainer.get("preflight", []):
            if check == "ffmpeg":
                preflight.append('if ! command -v ffmpeg >/dev/null 2>&1; then\n'
                                 '    echo "ffmpeg not found on PATH: the RTC codec augmentation would silently no-op. Reinstall it first." >&2\n'
                                 '    exit 1\nfi')
            elif check == "pyloudnorm":
                preflight.append('if ! "$PY" -c "import pyloudnorm" >/dev/null 2>&1; then\n'
                                 '    echo "pyloudnorm not importable by $PY: run \'$PY -m pip install pyloudnorm==0.2.0\' first." >&2\n'
                                 '    exit 1\nfi')
        pairs = L.parse_argv(it["argv"])
        arg_lines = []
        for flag, vals in pairs:
            if flag == "--track":
                arg_lines.append('    --track "$TRACK"')
            else:
                arg_lines.append("    " + " ".join([flag] + [shlex.quote(v) for v in vals]))
        repl = {
            "@@HEADER@@": "\n".join(header),
            "@@TRACK@@": it["track"],
            "@@PY@@": self.py,
            "@@WORKDIR@@": str(self.abs(trainer.get("cwd", "SLS_setup"))),
            "@@ENTRY@@": " ".join(shlex.quote(x) for x in trainer["entry"]),
            "@@PREFLIGHT@@": "\n".join(preflight),
            "@@ARGS@@": "\n".join(arg_lines),
            "@@SELF@@": self.rel(self.launcher_path(it)),
        }
        for k2, v in repl.items():
            tpl = tpl.replace(k2, v)
        return tpl

    # ---- smoke (opt-in)
    def ph_smoke(self, it):
        mode = self.wave.get("smoke", "off")
        if mode == "off" or it.get("adopted") or it.get("smoke_ok"):
            return True
        if mode == "first_use" and not self.uses_new_flag(it):
            it["smoke_ok"] = True
            return True
        job = it["jobs"].get("smoke")
        if job and self.job_alive(job):
            if time.time() - job["started"] > float(self.cfg("timeouts", "smoke_min", default=45)) * 60:
                self.kill_job(job)
            return False
        if job:
            it["jobs"].pop("smoke")
            text = L.head_text(job["log"], 2_000_000) + L.tail_bytes(job["log"], 200_000)
            reasons = self.log_problems(text, it, require_done=True)
            L.write_json(self.iter_dir(it) / f"smoke.a{job['attempt']}.json", {"ok": not reasons, "reasons": reasons, "log": job["log"]})
            if not reasons:
                it["smoke_ok"] = True
                self.event("smoke", "smoke run passed", it)
                return True
            it["smoke_failures"] = it.get("smoke_failures", 0) + 1
            if it["smoke_failures"] >= 2:
                self.block("smoke", f"smoke failed twice: {reasons}", f"see {job['log']}", it)
            self.redesign(it, f"the smoke run failed: {reasons}; log tail: {L.tail_bytes(job['log'], 1500)}")
            return False
        if not self.ready_for_gpu("smoke", it):
            return False
        self.preflight(it)
        smoke_args = list(self.cfg("trainer", "smoke_args", default=[])) + ["--track", it["track"] + "_smoke"]
        self.launch_trainer(it, "smoke", smoke_args)
        return False

    def uses_new_flag(self, it):
        design = L.read_json(self.iter_dir(it) / "design.result.json") or {}
        flags = set((design.get("change") or {}).get("set", {}))
        seen = set()
        for cfgp in glob.glob(str(self.abs(self.cfg("trainer", "exp_root", default="SLS_setup/exp")) / "*" / "config.yaml")):
            seen |= set(L.argv_dict(L.config_argv(Path(cfgp).parent)))
        return bool(flags - seen)

    def log_problems(self, text, it, require_done=False):
        reasons = []
        for rx in self.cfg("early_check", "forbid", default=["Traceback", "out of memory", r"failed \("]):
            m = re.search(rx, text)
            if m:
                line = text[max(0, m.start() - 200): m.end() + 200].strip().splitlines()
                reasons.append(f"forbidden pattern {rx!r}: {line[-1][:200] if line else ''}")
        for rx in list(self.cfg("early_check", "expect", default=[])) + list(it.get("expect") or []):
            if not re.search(rx, text, re.M):
                reasons.append(f"expected header line missing: {rx!r}")
        if require_done and "Experiment saved to:" not in text:
            reasons.append("run did not finish ('Experiment saved to:' missing)")
        return reasons

    def redesign(self, it, reason):
        d = self.iter_dir(it)
        n = len(list(d.glob("design.result.rejected*.json")))
        if (d / "design.result.json").is_file():
            os.replace(d / "design.result.json", d / f"design.result.rejected{n}.json")
        it.setdefault("rejected_tracks", []).append(it.get("track"))
        it.setdefault("design_errors", []).append(reason[:3000])
        for key in ("argv", "argv_checked", "run_dir", "track", "smoke_ok", "early_ok", "train_done"):
            it.pop(key, None)
        it["logs"] = []
        it["phase"] = "design"
        it["attempts"]["design"] = 0     # a new design round (bounded by early_check / smoke failure caps)
        self.event("redesign", reason[:300], it)

    # ---- train
    def launch_trainer(self, it, jobname, extra):
        launcher = self.abs(it["launcher"]) if it.get("launcher") else self.launcher_path(it)
        r = subprocess.run(["bash", str(launcher), *extra], cwd=self.repo, capture_output=True, text=True,
                           timeout=600, start_new_session=True)
        m_pid = re.search(r"\(pid (\d+)\)", r.stdout)
        m_log = re.search(r"^Log:\s+(\S+)", r.stdout, re.M)
        if r.returncode != 0 or not m_pid or not m_log:
            self.block(jobname, f"launcher failed (rc {r.returncode}): {(r.stderr or r.stdout)[-600:]}",
                       f"run `bash {self.rel(launcher)}` by hand to see why", it)
        pid = int(m_pid.group(1))
        time.sleep(1)
        job = {"name": jobname, "pid": pid, "start_ticks": L.proc_start_ticks(pid), "log": m_log.group(1),
               "started": time.time(), "started_ts": L.now_ts(), "attempt": it["attempts"].get(jobname, 0),
               "cmd": f"bash {self.rel(launcher)} {' '.join(extra)}".strip(), "resume": "--resume" in extra}
        it["attempts"][jobname] = it["attempts"].get(jobname, 0) + 1
        it["jobs"][jobname] = job
        if jobname == "train":
            it["logs"].append(job["log"])
        self.event("launch", f"{job['cmd']} -> pid {pid}, log {job['log']}", it)
        return job

    def train_complete(self, it):
        if not it.get("logs"):
            return False
        return "Experiment saved to:" in L.tail_bytes(it["logs"][-1], 20000)

    def ph_train(self, it):
        if it.get("train_done"):
            return True
        job = it["jobs"].get("train")
        if not job and it.get("run_dir") and self.train_complete(it):
            it["train_done"] = True
            self.collect_epochs(it)
            return True
        if job and self.job_alive(job):
            self.monitor_train(it, job)
            return False
        if job:
            it["jobs"].pop("train")
            if not it.get("run_dir"):
                self.discover_run_dir(it, job)
            if self.train_complete(it):
                self.collect_epochs(it)
                it["train_done"] = True
                self.st.pop("progress", None)
                self.event("train_done", f"{it['label']} finished: {it.get('run_dir')}", it)
                return True
            tail = L.tail_bytes(job["log"], 30000)
            if re.search(r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError", tail):
                self.block("train", f"the trainer crashed: {tail.strip().splitlines()[-1][:300]}",
                           f"read {job['log']}; fix (flags only), then resume (the run continues with --resume)", it)
            n = it["attempts"].get("train_resume", 0)
            if n >= int(self.cfg("retries", "train_resume", default=3)):
                self.block("train", f"the trainer died without a traceback {n + 1} times",
                           "check `ps -o lstart= -p 1` (container restarts), dmesg (OOM killer), then resume", it)
            it["attempts"]["train_resume"] = n + 1
            self.event("train_died", f"trainer pid {job['pid']} gone without 'Experiment saved' and without a traceback "
                                     f"(container restart / kill); will resume", it)
        # launch: fresh, or --resume when the run dir has a last_state.pt
        if not self.ready_for_gpu("train", it):
            return False
        self.preflight(it)
        extra = []
        if it.get("run_dir") and (self.abs(it["run_dir"]) / "last_state.pt").is_file():
            extra = ["--resume", os.path.relpath(self.abs(it["run_dir"]), self.abs(self.cfg("trainer", "cwd", default="SLS_setup")))]
        self.launch_trainer(it, "train", extra)
        return False

    def discover_run_dir(self, it, job):
        root = self.abs(self.cfg("trainer", "exp_root", default="SLS_setup/exp"))
        cands = [p for p in root.glob(f"{it['track']}_epoch*_bs*_*") if p.is_dir() and p.stat().st_mtime >= job["started"] - 120]
        if not cands:
            m = re.search(r"Experiment saved to: (\S+)", L.tail_bytes(job["log"], 20000))
            if m:
                p = (self.abs(self.cfg("trainer", "cwd", default="SLS_setup")) / m.group(1)).resolve()
                cands = [p] if p.is_dir() else []
        if not cands:
            return None
        run = max(cands, key=lambda p: p.stat().st_mtime)
        if not (run / "config.yaml").is_file():
            return None
        recorded = L.config_argv(run)
        expected = [str(x) for x in it["argv"]]
        if recorded[:len(expected)] != expected:
            self.block("train", f"config.yaml command of {run.name} differs from the launcher argv",
                       "someone edited the launcher? compare config.yaml with the design", it)
        it["run_dir"] = self.rel(run)
        self.event("run_dir", f"run dir {it['run_dir']}", it)
        return run

    def monitor_train(self, it, job):
        self.st["status"] = "RUNNING"
        self.st["waiting"] = None
        if not it.get("run_dir") and time.time() - job["started"] > 5:
            self.discover_run_dir(it, job)
        self.collect_epochs(it)
        log = Path(job["log"])
        tail = L.tail_bytes(log, 65536)
        prog = L.parse_tqdm(tail)
        num_epochs = int((L.argv_dict(it["argv"]).get("--num_epochs") or ["8"])[0])
        if prog:
            done_eps = it.get("epochs_seen", 0)
            prog["epoch"] = done_eps + 1
            if prog.get("epoch_s"):
                prog["eta_h"] = (prog.get("remaining_s") or 0) / 3600 + (num_epochs - done_eps - 1) * prog["epoch_s"] / 3600 * 1.03
            self.st["progress"] = prog
        # early check (fresh launches only): header lines, no traceback / codec failures, ETA
        if not it.get("early_ok") and not job.get("resume"):
            el = time.time() - job["started"]
            text = L.head_text(log, 400000)
            forbid = [r for r in self.log_problems(text, it) if r.startswith("forbidden")]
            ready = el >= float(self.cfg("early_check", "minutes", default=10)) * 60 or (prog and prog["i"] >= 200)
            reasons = forbid
            if ready and not reasons:
                reasons = self.log_problems(text, it)
                max_h = float(self.cfg("timeouts", "max_run_hours", default=12))
                if prog and prog.get("eta_h") and prog["eta_h"] > max_h:
                    reasons.append(f"projected run time {prog['eta_h']:.1f} h > max_run_hours {max_h}")
            if reasons:
                self.event("early_check_failed", "; ".join(reasons)[:500], it)
                self.kill_job(job, signal.SIGTERM)
                time.sleep(20)
                if self.job_alive(job):
                    self.kill_job(job, signal.SIGKILL)
                it["jobs"].pop("train", None)
                it["early_failures"] = it.get("early_failures", 0) + 1
                if it["early_failures"] >= int(self.cfg("retries", "early_check", default=2)):
                    self.block("train", f"early check failed twice: {reasons}", f"see {job['log']}", it)
                self.redesign(it, f"early check of the real run failed and the run was stopped: {reasons}. "
                                  f"Log head: {text[:1500]}")
                return
            if ready:
                it["early_ok"] = True
                self.event("early_check", "header lines present, no traceback / codec failures"
                           + (f", ETA {prog['eta_h']:.1f} h" if prog and prog.get("eta_h") else ""), it)
        # stall: the stdout log must keep growing
        stall = float(self.cfg("timeouts", "stall_min", default=30)) * 60
        if log.is_file() and time.time() - log.stat().st_mtime > stall:
            self.block("train", f"the trainer (pid {job['pid']}) is alive but its log has not changed for "
                                f"{(time.time() - log.stat().st_mtime) / 60:.0f} min",
                       "inspect it (py-spy dump / nvidia-smi); the supervisor did not kill it. Kill it to make the "
                       "loop resume it, or let it continue and `loopctl resume`", it)

    def collect_epochs(self, it):
        if not it.get("run_dir"):
            return
        rows = L.parse_train_log(self.abs(it["run_dir"]) / "logs" / "train.log")
        seen = it.get("epochs_seen", 0)
        best_rows = self.best_epoch_rows()
        for ep in sorted(rows):
            r = rows[ep]
            if ep <= seen or r.get("wf1") is None or r.get("train_loss") is None:
                continue
            ref = best_rows.get(ep, {}).get("wf1")
            d = None if ref is None else r["wf1"] - ref
            self.event("epoch", f"{it['label']} epoch {ep}/{r.get('num_epochs')}: TrainLoss {r['train_loss']:.4f} "
                                f"DevLoss {r['dev_loss']:.4f} LocalWF1 {r['wf1']:.2f}"
                                + (f" ({d:+.2f} vs {self.st['best']['label']}@{ep})" if d is not None else "")
                                + f" F1clean {r['f1_clean']:.2f} F1noisy {r['f1_noisy']:.2f} EERnoisy {r['eer_noisy']:.2f}",
                       it, epoch=ep, row=r, delta_vs_best=d)
            it["epochs_seen"] = ep

    def best_epoch_rows(self):
        rd = self.st.get("best", {}).get("run_dir")
        if not rd:
            return {}
        if rd not in self._best_epoch_cache:
            self._best_epoch_cache[rd] = L.parse_train_log(self.abs(rd) / "logs" / "train.log")
        return self._best_epoch_cache[rd]

    # ---- eval
    def profile(self):
        if self._profile is None:
            ev = self.wave.get("eval", {})
            if ev.get("pipeline"):
                self._profile = ev
            else:
                name = ev.get("profile", "sls_local17")
                p = Path(name) if name.endswith(".json") else HERE / "profiles" / f"{name}.json"
                self._profile = L.read_json(self.abs(p) if not p.is_absolute() else p)
                if not self._profile:
                    sys.exit(f"eval profile not found: {p}")
        return self._profile

    def objective_key(self):
        return self.cfg("objective", "key", default="half_B.wf1.wf1")

    def clean_key(self):
        return self.cfg("objective", "clean_key", default="half_B.wf1.f1_clean")

    def eval_ctx(self, it):
        run = self.abs(it["run_dir"])
        k = int(self.cfg("eval", "k", default=5))
        N = L.final_epoch(run)
        if N is None:
            self.block("eval", f"no metrics.jsonl rows in {it['run_dir']}", "", it)
        ckpts = sorted(glob.glob(str(run / "ckpt" / f"epoch_{N}_*.pth")))
        if len(ckpts) != 1:
            self.block("eval", f"expected one ckpt/epoch_{N}_*.pth, found {len(ckpts)}",
                       "was the final epoch pruned? (--keep_epochs must equal the final epoch)", it)
        best = self.st["best"]
        bN = re.search(r"(\d+)$", Path(best["readouts"]["final"]).name)
        best_final_label = f"{best['label']}-ep{bN.group(1) if bN else 'N'}"
        best_merge_label = f"{best['label']}-top{k}"
        cwd = self.abs(self.cfg("trainer", "cwd", default="SLS_setup"))
        ctx = {
            "py": self.py, "repo": str(self.repo), "run": self.rel(run), "run_rel": os.path.relpath(run, cwd),
            "k": str(k), "N": str(N), "ckpt_final": self.rel(ckpts[0]),
            "label": it["label"], "merge_label": f"{it['label']}-top{k}", "final_label": f"{it['label']}-ep{N}",
            "best_label": best["label"], "best_merge": self.rel(self.best_dir("merge")),
            "best_final": self.rel(self.best_dir("final")), "best_merge_label": best_merge_label,
            "best_final_label": best_final_label, "ci": str(self.cfg("eval", "ci", default=2000)),
            "model_flags": L.model_flags(run), "names": list(self.cfg("eval", "names", default=[])),
        }
        dirs, labels = [ctx["best_merge"]], [best_merge_label]
        for c in self.cfg("eval", "fixed_comparators", default=[]) or []:
            dirs.append(c["dir"])
            labels.append(c["label"])
        dirs += [ctx["best_final"], f"{ctx['run']}/local_eval/merge_top{k}", f"{ctx['run']}/local_eval/epoch_{N}"]
        labels += [best_final_label, ctx["merge_label"], ctx["final_label"]]
        seen, all_dirs, all_labels = set(), [], []
        for d, lab in zip(dirs, labels):
            key = str(self.abs(d).resolve())
            if key not in seen:
                seen.add(key)
                all_dirs.append(d)
                all_labels.append(lab)
        ctx["all_dirs"], ctx["all_labels"] = all_dirs, all_labels
        prof = self.profile()
        ctx["readout_merge"] = expand_str(prof["readouts"]["merge"], ctx)
        ctx["readout_final"] = expand_str(prof["readouts"]["final"], ctx)
        return ctx

    def step_done(self, step, ctx, it):
        if step["name"] in it.get("eval_skipped", []):
            return True
        done = step.get("done", {})
        for f in done.get("exists", []):
            if not self.abs(expand_str(f, ctx)).exists():
                return False
        if done.get("merge_list_matches"):
            new, base, old = (expand_str(x, ctx) for x in done["merge_list_matches"])
            if not _same_merge_list(self.abs(new), self.abs(base), self.abs(old)):
                return False
        if done.get("same_text"):
            a, b = (self.abs(expand_str(x, ctx)) for x in done["same_text"])
            if not (a.is_file() and b.is_file() and a.read_text().strip() == b.read_text().strip()):
                return False
        if done.get("newer_than"):
            a, b = (self.abs(expand_str(x, ctx)) for x in done["newer_than"])
            if not (a.is_file() and b.is_file() and a.stat().st_mtime >= b.stat().st_mtime):
                return False
        for src, dst in step.get("copy", []):
            if not self.abs(expand_str(dst, ctx)).exists():
                return False
        return True

    def ph_eval(self, it):
        ctx = self.eval_ctx(it)
        for step in self.profile()["steps"]:
            name = "eval:" + step["name"]
            if self.step_done(step, ctx, it):
                continue
            job = it["jobs"].get(name)
            if job and self.job_alive(job):
                self.timeout_job(job, float(self.cfg("timeouts", "eval_step_h", default=3)) * 3600, it)
                self.st.update(status="RUNNING", waiting=None)
                return False
            if job:
                it["jobs"].pop(name)
                rc = self.job_rc(job)
                if rc == 0:
                    for src, dst in step.get("copy", []):
                        s, d = self.abs(expand_str(src, ctx)), self.abs(expand_str(dst, ctx))
                        if s.is_file():
                            shutil.copy2(s, d)
                    if self.step_done(step, ctx, it):
                        self.event("eval_step", f"{step['name']} done", it)
                        continue
                    why = "exit 0 but its outputs are missing"
                else:
                    why = f"exit code {rc}"
                self.event("eval_failed", f"{step['name']}: {why}; log {job['log']}", it)
                if it["attempts"].get(name, 0) >= int(self.cfg("retries", "eval_step", default=2)):
                    if step.get("optional"):
                        it.setdefault("eval_skipped", []).append(step["name"])
                        self.event("eval_skipped", f"optional step {step['name']} skipped", it)
                        continue
                    tail = L.tail_bytes(job["log"] + ".exploop_tmp", 1200) or L.tail_bytes(job["log"], 1200)
                    self.block("eval", f"{step['name']} failed {it['attempts'][name]} times ({why}): {tail[-600:]}",
                               f"read {job['log']}(.exploop_tmp); fix, then `loopctl.sh resume {self.wave['wave']} --reset-attempts`", it)
            if step.get("gpu") and not self.ready_for_gpu("eval", it):
                return False
            argv = expand_list(step["cmd"], ctx)
            cc = step.get("cache_check")
            if cc:
                out = self.abs(expand_str(cc["out"], ctx))
                if (out / "scores").is_dir() and not _same_merge_list(self.abs(expand_str(cc["list"], ctx)),
                                                                     self.abs(expand_str(cc["list_base"], ctx)),
                                                                     out / "merge_list.txt"):
                    argv.append("--force")
                    self.event("eval_cache", f"{step['name']}: cached scores belong to other checkpoints -> --force", it)
            it["attempts"][name] = it["attempts"].get(name, 0) + 1
            stdout = expand_str(step["stdout"], ctx) if step.get("stdout") else None
            self.start_job(it, name, argv, self.abs(step.get("cwd", ".")), stdout=stdout)
            return False
        it["eval_ctx"] = {k: v for k, v in ctx.items() if isinstance(v, str)}
        self.event("eval_done", f"{it['label']}: all eval steps done", it)
        return True

    # ---- verdict / log
    def ph_verdict(self, it):
        vpath = self.iter_dir(it) / "verdict.json"
        if it.get("verdict") and vpath.is_file():
            return True
        ctx = self.eval_ctx(it)
        prof = self.profile()
        hib = bool(self.cfg("objective", "higher_is_better", default=True))
        rules = self.wave.get("rules", {})
        out = {}
        for ro, base_key in (("merge", "merge"), ("final", "final")):
            run_m = L.readout_numbers(self.abs(ctx["readout_" + ro]) / "metrics.json", self.objective_key(), self.clean_key())
            base_m = L.readout_numbers(self.best_dir(base_key) / "metrics.json", self.objective_key(), self.clean_key())
            if not run_m or not base_m:
                self.block("verdict", f"missing metrics for the {ro} readout", "", it)
            ci = L.ci_numbers(self.abs(expand_str(prof["ci_files"][ro], ctx))) if prof.get("ci_files") else None
            v = L.decide(run_m, base_m, ci, rules, hib)
            v.update(readout=ctx["readout_" + ro], base_readout=self.rel(self.best_dir(base_key)),
                     base_label=(ctx["best_merge_label"] if ro == "merge" else ctx["best_final_label"]),
                     label=(ctx["merge_label"] if ro == "merge" else ctx["final_label"]))
            out[ro] = v
        out["objective"] = self.objective_key()
        out["rules"] = rules
        out["written"] = L.now_ts()
        L.write_json(vpath, out)
        it["verdict"] = out
        m = out["merge"]
        self.event("verdict", f"{it['label']}: merge {m['run']:.2f} vs {m['base_label']} {m['base']:.2f} "
                              f"({m['delta']:+.2f}, CI {_fmt_ci(m)}) -> {m['decision']}, {m['verdict']}; final "
                              f"{out['final']['run']:.2f} ({out['final']['delta']:+.2f})", it)
        return True

    def ph_log(self, it):
        if it.get("logged"):
            return True
        reg = str(self.abs(self.wave.get("registry", "docs/experiment_log.csv")))
        ctx = self.eval_ctx(it)
        prof = self.profile()
        if not it.get("report"):
            it["report"] = str(self.abs(self.wave["report_dir"]) / f"{L.stamp()}_{it['track']}.md")
        ids = {}
        for ro in ("merge", "final"):
            v = it["verdict"][ro]
            rid = f"{it['label']}-{it['lever_slug']}-" + ("screen" if ro == "merge" else f"ep{ctx['N']}")
            ref = (self.st["best"].get("registry_ids") or {}).get(ro, "")
            tag = Path(ctx["readout_" + ro]).name
            notes = (f"exploop wave {self.wave['wave']} iteration {it['k']} ({ro} readout); verdict: {v['verdict']}; "
                     f"report: {self.rel(it['report'])}")
            existing = L.registry_row(reg, rid)
            ci_file = self.abs(expand_str(prof["ci_files"][ro], ctx)) if prof.get("ci_files") else None
            need_ci = bool(ci_file and ci_file.is_file())
            if existing and existing.get("decision") == v["decision"] and (not need_ci or existing.get("ci_low")):
                ids[ro] = rid
                continue
            cmd = [self.py, str(self.repo / "scripts/log_experiment.py"), "--run_dir", ctx["run"], "--tag", tag,
                   "--id", rid, "--change_vs_ref", it.get("change_vs_ref", ""), "--decision", v["decision"],
                   "--notes", notes, "--registry", reg]
            if ref:
                cmd += ["--ref", ref]
            if existing:
                cmd.append("--replace")
            dry = subprocess.run(cmd + ["--dry_run"], cwd=self.repo, capture_output=True, text=True, timeout=600)
            L.write_text(self.iter_dir(it) / f"log_{ro}.dry_run.txt", dry.stdout + dry.stderr)
            if dry.returncode != 0 or "WARNING" in dry.stdout:
                self.block("log", f"log_experiment dry run for {rid}: rc {dry.returncode}; "
                                  f"{(dry.stdout + dry.stderr).strip().splitlines()[-3:]}",
                           f"see {self.rel(self.iter_dir(it) / f'log_{ro}.dry_run.txt')}", it)
            real = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, timeout=600)
            L.write_text(self.iter_dir(it) / f"log_{ro}.txt", real.stdout + real.stderr)
            if real.returncode != 0:
                self.block("log", f"log_experiment for {rid} failed: {(real.stdout + real.stderr)[-400:]}", "", it)
            if need_ci:
                ci = subprocess.run([self.py, str(self.repo / "scripts/b3_log_ci.py"), "--id", rid, "--ci_json",
                                     str(ci_file), "--registry", reg], cwd=self.repo, capture_output=True, text=True, timeout=300)
                if ci.returncode != 0:
                    self.block("log", f"b3_log_ci for {rid} failed: {(ci.stdout + ci.stderr)[-400:]}", "", it)
            ids[ro] = rid
            self.event("logged", f"registry row {rid} ({v['decision']})", it)
        it["registry_ids"] = ids
        it["logged"] = True
        return True

    # ---- analyze / decide / summary
    def ph_analyze(self, it):
        if not it.get("report"):
            it["report"] = str(self.abs(self.wave["report_dir"]) / f"{L.stamp()}_{it['track']}.md")
        rpath = Path(it["report"])
        apath = self.iter_dir(it) / "analyze.result.json"
        if apath.is_file() and _report_ok(rpath):
            return True
        rpath.parent.mkdir(parents=True, exist_ok=True)
        r = self.claude_step(it, "analyze", lambda: self.write_analyze_prompt(it), "analyze", allowed_writes=[rpath],
                             extra_env={"EXPLOOP_REPORT_PATH": str(rpath)})
        if r is None:
            return False
        ok, payload = r
        if not ok:
            return False
        if not _report_ok(rpath):
            it.setdefault("analyze_errors", []).append(f"report {rpath} missing or without sections 1-7")
            self.event("step_failed", f"analyze: report {self.rel(rpath)} missing or without sections 1-7", it)
            return False
        props = payload.get("proposals") or []
        payload["proposals"] = sorted(props, key=lambda p: (p.get("kind") == "seed_repeat", p.get("rank", 99)))
        payload["report_path"] = self.rel(rpath)
        L.write_json(apath, payload)
        self.event("analyze", f"report {self.rel(rpath)}; top proposal: "
                              f"{(payload['proposals'] or [{}])[0].get('lever', '?')}", it)
        return True

    def ph_decide(self, it):
        if it.get("done"):
            return True
        v = it["verdict"]
        ctx = it.get("eval_ctx") or {}
        if v["merge"]["new_best"]:
            sub = self.abs(ctx.get("run", it["run_dir"])) / f"model_merging_top{ctx.get('k', 5)}" / "submission.zip"
            self.st["best"] = {"label": it["label"], "iter": it["k"], "run_dir": it["run_dir"], "track": it["track"],
                               "readouts": {"merge": v["merge"]["readout"], "final": v["final"]["readout"]},
                               "registry_ids": it.get("registry_ids", {}), "argv": it["argv"],
                               "merge_value": v["merge"]["run"], "final_value": v["final"]["run"],
                               "submission": self.rel(sub) if sub.is_file() else None}
            self.event("new_best", f"{it['label']} is the new best (merge {v['merge']['run']:.2f}, "
                                   f"{v['merge']['delta']:+.2f})", it)
        it["done"] = True
        it["finished_ts"] = L.now_ts()
        k = it["k"]
        self.st["current"] = k + 1 if k < int(self.wave["max_iterations"]) else None
        self.event("iteration_done", f"{it['label']} done ({v['merge']['decision']})", it)
        return True

    def ph_summary(self):
        if self.st.get("summary_done"):
            self.st["status"] = "DONE"
            return False
        table = self.wave_table()
        L.write_text(self.wave_dir / "wave_table.md", table)
        if not self.st.get("summary_report"):
            self.st["summary_report"] = str(self.abs(self.wave["report_dir"]) / f"{L.stamp()}_{self.wave['track_prefix']}_wave_summary.md")
        rpath = Path(self.st["summary_report"])
        r = self.claude_step(None, "summary", lambda: self.write_summary_prompt(rpath), "summary", allowed_writes=[rpath],
                             extra_env={"EXPLOOP_REPORT_PATH": str(rpath)})
        if r is None:
            return False
        ok, payload = r
        if not ok:
            return False
        if not rpath.is_file():
            self.st.setdefault("summary_errors", []).append("summary report missing")
            return False
        L.write_json(self.wave_dir / "summary.result.json", payload)
        self.st["summary_done"] = True
        self.st["status"] = "DONE"
        self.st["next"] = (f"Wave finished. Run `/exp-loop wrapup {self.wave['wave']}` in an interactive Claude session "
                           f"(memory update, list of changed files), upload the best submission.zip by hand.")
        self.event("done", f"wave {self.wave['wave']} done; summary {self.rel(rpath)}")
        return True

    def wave_table(self):
        b = self.st["best"]
        lines = [f"# Wave {self.wave['wave']} iteration table (generated by the supervisor from verdict.json files)", "",
                 "| # | label | track | change vs ref | ref | merge | delta | CI | final | delta | decision | verdict | report |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for it in self.st["iterations"]:
            v = it.get("verdict") or {}
            m, f = v.get("merge") or {}, v.get("final") or {}
            lines.append(f"| {it['k']} | {it['label']} | `{it.get('track')}` | {it.get('change_vs_ref', '')} | "
                         f"{it.get('base_label', '')} | {L._f(m.get('run'))} | {L._f(m.get('delta'), sign=True)} | {_fmt_ci(m)} | "
                         f"{L._f(f.get('run'))} | {L._f(f.get('delta'), sign=True)} | {m.get('decision', '')} | "
                         f"{m.get('verdict', '')} | {self.rel(it['report']) if it.get('report') else ''} |")
        lines += ["", f"Final best: **{b['label']}** (`{b.get('run_dir')}`), merge {L._f(b.get('merge_value'))}, "
                      f"final epoch {L._f(b.get('final_value'))}; submission {b.get('submission')}"]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------------------ prompts
    def common_context(self, it):
        b = self.st["best"]
        hist = []
        for i in self.st["iterations"]:
            if i is it or not i.get("verdict"):
                continue
            m = i["verdict"]["merge"]
            hist.append(f"| {i['label']} | `{i.get('track')}` | {i.get('change_vs_ref')} | {i.get('base_label')} | "
                        f"{m['run']:.2f} | {m['delta']:+.2f} | {_fmt_ci(m)} | {i['verdict']['final']['run']:.2f} | "
                        f"{m['decision']} | {m['verdict']} | {self.rel(i['report']) if i.get('report') else ''} |")
        notes = (self.wave_dir / "user_notes.md")
        notes_txt = notes.read_text(encoding="utf-8").strip() if notes.is_file() else ""
        reports = sorted(glob.glob(str(self.abs(self.wave["report_dir"]) / "*.md")))
        prev = self.cfg("context", "previous_reports", default=[])
        return {
            "wave_json": self.rel(self.wave_dir / "wave.json"),
            "plan": self.rel(self.wave_dir / "plan_snapshot.md"),
            "notes": notes_txt,
            "best": b,
            "history": hist,
            "reports": [self.rel(p) for p in reports] + list(prev),
            "registry": self.wave.get("registry"),
            "template": self.wave.get("report_template"),
        }

    def write_design_prompt(self, it):
        c = self.common_context(it)
        b = c["best"]
        prev = None
        done = [i for i in self.st["iterations"] if i.get("done")]
        if done:
            prev = L.read_json(self.iter_dir(done[-1]) / "analyze.result.json")
        used = sorted(self.used_tracks(it))
        lines = [f"# exploop step DESIGN - wave {self.wave['wave']}, iteration {it['k']} ({it['label']})", "",
                 "You design the next training run of an unattended exploration loop. Read and follow "
                 "`.claude/skills/exp-design/SKILL.md`. Output ONLY the structured result; do not write or edit any file, do not launch anything.", "",
                 "## Inputs (read them)",
                 f"- Wave spec: `{c['wave_json']}` (search_space: allowed / coupled / forbidden / fixed flags, lever_queue, lever_notes, rules)",
                 f"- Approved plan snapshot: `{c['plan']}`",
                 f"- Trainer flags: `grep -n add_argument {' '.join(self.cfg('trainer', 'flag_sources', default=[]))}`",
                 f"- Registry: `{c['registry']}`; all reports so far: " + ", ".join(f"`{r}`" for r in c["reports"][-8:]), "",
                 "## User notes (highest priority; newest last)", "", c["notes"] or "(none)", "",
                 f"## Base = current best: {b['label']}", "",
                 f"- run dir `{b.get('run_dir')}`; readouts `{b['readouts']['merge']}` and `{b['readouts']['final']}`",
                 "- base argv (change ONE factor of this; the supervisor sets --track and --keep_epochs):", "",
                 "```", " ".join(shlex.quote(x) for x in b["argv"]), "```", ""]
        if c["history"]:
            lines += ["## History of this wave (numbers from verdict.json)", "",
                      "| label | track | change | ref | merge | delta | CI | final | decision | verdict | report |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"] + c["history"] + [""]
        if prev:
            lines += [f"## Latest analysis ({prev.get('report_path')}); proposals re-ranked: configuration changes before seed repeats", "",
                      "```json", json.dumps({k: prev.get(k) for k in ("headline", "largest_remaining_error", "went_wrong", "proposals")}, indent=1)[:12000], "```", ""]
        lines += ["## Required output", "",
                  f"- iter = {it['k']}, letter = `{it['letter']}`; track = `{self.wave['track_prefix']}{it['letter']}_<descriptive tokens>_s<seed>` "
                  f"in the style of the previous tracks (e.g. `{b.get('track') or ''}`); unique, not one of {used}",
                  "- `change.set`: {flag: [values]} (a store_true flag -> []); `change.remove`: [flags]. Exactly ONE factor vs the "
                  "base argv; a coupled group from wave.json counts as one factor.",
                  "- `kind`: `config` unless every configuration lever is exhausted or ruled out (then `seed_repeat`, only --seed changes).",
                  "- `expect`: regexes of stdout header lines that prove the change took effect (e.g. `^SpecAugment: on`). Check "
                  "the trainer source for the exact print.",
                  "- `lever_slug`: short id for registry ids (e.g. `codecrand`), letters/digits/dots only.",
                  "- No metric numbers in your prose unless copied from a cited file."]
        errs = it.get("design_errors") or []
        if errs:
            lines += ["", "## Your previous attempt was rejected - fix this", ""] + [f"- {e}" for e in errs[-6:]]
            if it.get("rejected_tracks"):
                lines += ["", f"Tracks already used (pick a new one, e.g. add a `v2` token, even for the same lever): "
                              f"{it['rejected_tracks']}"]
        p = self.iter_dir(it) / f"design.prompt.a{it['attempts'].get('design', 0)}.md"
        L.write_text(p, "\n".join(lines) + "\n")
        return p

    def write_analyze_prompt(self, it):
        c = self.common_context(it)
        ctx = self.eval_ctx(it)
        run = ctx["run"]
        logs = it.get("logs", [])
        files = [f"`{run}/config.yaml`", f"`{run}/metrics.jsonl`", f"`{run}/logs/train.log`",
                 f"`{run}/top{ctx['k']}_by_wf1/merge_list.txt`", f"`{ctx['readout_merge']}/metrics.json`",
                 f"`{ctx['readout_merge']}/probe_report.json`", f"`{ctx['readout_merge']}.log`",
                 f"`{ctx['readout_final']}/metrics.json`", f"`{ctx['readout_final']}/probe_report.json`",
                 f"`{ctx['readout_final']}.log`"] + \
                [f"`{self.rel(p)}`" for p in sorted(self.abs(ctx["readout_merge"]).glob("compare_*"))] + \
                [f"`{self.rel(p)}`" for p in sorted(self.abs(ctx["readout_final"]).glob("compare_*"))] + \
                [f"`{x}` (stdout log; grep header lines, ignore progress bars)" for x in logs]
        lines = [f"# exploop step ANALYZE - wave {self.wave['wave']}, iteration {it['k']} ({it['label']})", "",
                 "You are the analysis agent of an unattended exploration loop. Read and follow `.claude/skills/exp-report/SKILL.md`. "
                 f"Write exactly ONE file, `{self.rel(it['report'])}`, then return the structured result. Never launch anything, "
                 "never edit code, never read progress-split scores, never type a number that is not in a file you read.", "",
                 f"## Run under analysis: {it['label']} = {it.get('base_label')} with ONE change: {it.get('change_vs_ref')}", "",
                 f"- design (hypothesis, predicted signature, falsifier): `{it.get('design')}`",
                 f"- launcher: `{it.get('launcher')}`; verdict (computed by the supervisor; quote it, do not recompute): "
                 f"`{self.rel(self.iter_dir(it) / 'verdict.json')}`",
                 "- files: " + ", ".join(files), "",
                 f"## References",
                 f"- current best before this run: {it.get('base_label')} -> `{ctx['best_merge']}`, `{ctx['best_final']}`, run dir "
                 f"`{self.st['best'].get('run_dir')}`",
                 f"- comparator table with every readout: `{ctx['readout_merge']}/compare_all_readouts.txt`",
                 f"- wave spec `{c['wave_json']}` (rules, lever_queue, lever_notes), plan `{c['plan']}`, registry `{c['registry']}`",
                 f"- report template (follow its structure): `{c['template']}`",
                 "- previous reports: " + ", ".join(f"`{r}`" for r in c["reports"][-8:]), ""]
        if c["history"]:
            lines += ["## History of this wave", "", "| label | track | change | ref | merge | delta | CI | final | decision | verdict | report |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"] + c["history"] + [""]
        lines += ["## User notes", "", c["notes"] or "(none)", "",
                  "## Produce", "",
                  f"`{self.rel(it['report'])}` with sections `## 1. Setup`, `## 2. Results vs references`, `## 3. Per-epoch behaviour`, "
                  "`## 4. What went OK`, `## 5. What went wrong`, `## 6. Verdict on the pre-registered rule`, "
                  "`## 7. Decision and next-experiment proposals` (exp-report skill). Then the structured result: headline, "
                  "went_ok, went_wrong, largest_remaining_error, and at least 3 ranked proposals, each ONE factor vs the "
                  "new best (the supervisor updates the best from verdict.json: if this run is a new best, propose changes to "
                  "this run's flags, else to the previous best's), as {set, remove} flag changes with hypothesis, predicted "
                  "signature and falsifier. Prefer configuration changes; a seed repeat only as the last proposal."]
        p = self.iter_dir(it) / f"analyze.prompt.a{it['attempts'].get('analyze', 0)}.md"
        L.write_text(p, "\n".join(lines) + "\n")
        return p

    def write_summary_prompt(self, rpath):
        c = self.common_context(None)
        lines = [f"# exploop step SUMMARY - wave {self.wave['wave']}", "",
                 "Read and follow `.claude/skills/exp-report/SKILL.md` (wave-summary section). Write exactly ONE file, "
                 f"`{self.rel(rpath)}`, then return the structured result. Never launch anything; never read progress scores.", "",
                 f"- iteration table (from verdict.json files): `{self.rel(self.wave_dir / 'wave_table.md')}`",
                 "- per-iteration reports: " + ", ".join(f"`{self.rel(i['report'])}`" for i in self.st["iterations"] if i.get("report")),
                 f"- wave spec `{c['wave_json']}`, plan `{c['plan']}`, registry `{c['registry']}`",
                 f"- model for the structure: {', '.join(self.cfg('context', 'summary_template', default=[]) or ['docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md'])}",
                 f"- final best: {self.st['best']['label']} (`{self.st['best'].get('run_dir')}`)", "",
                 "Sections: starting point and loop rules; iteration table; what each lever did (lever-response table); "
                 "lessons; recommended full run (exact flags, seeds, readout, which checkpoint to submit) with pre/post-launch "
                 "checks; follow-ups ranked; artefacts (paths)."]
        p = self.wave_dir / f"summary.prompt.a{self.st.get('attempts', {}).get('summary', 0)}.md"
        L.write_text(p, "\n".join(lines) + "\n")
        return p

    # ------------------------------------------------------------------------------ loop
    def render(self):
        if not self.st:
            return
        it = next((i for i in self.st["iterations"] if i["k"] == self.st.get("current")), None)
        rows = {}
        if it and it.get("run_dir"):
            rows = L.parse_train_log(self.abs(it["run_dir"]) / "logs" / "train.log")
        best_rows = {ep: r.get("wf1") for ep, r in self.best_epoch_rows().items()}
        health = {"container": "same boot" if self.st.get("pid1_start") == L.pid1_start() else "RESTARTED since the last supervisor start",
                  "warnings": list(self.st.get("warnings", []))[-3:]}
        if it and it.get("phase") == "log":
            health["warnings"].append("writing docs/experiment_log.csv right now: avoid editing it by hand for a minute")
        prog = self.st.get("progress") if it and it.get("phase") == "train" else None
        text = L.render_status(self.wave, self.st, health, rows, prog, best_rows, self.wave_dir)
        L.write_text(self.wave_dir / "STATUS.md", text)

    def step(self):
        if self.st.get("current") is None:
            return self.ph_summary()
        it = self.iteration(self.st["current"])
        handlers = {"design": self.ph_design, "render": self.ph_render, "smoke": self.ph_smoke, "train": self.ph_train,
                    "eval": self.ph_eval, "verdict": self.ph_verdict, "log": self.ph_log, "analyze": self.ph_analyze,
                    "decide": self.ph_decide}
        progressed = False
        for _ in range(len(PHASES) + 2):
            phase = it["phase"]
            if not handlers[phase](it):
                return progressed
            progressed = True
            if phase == "decide":
                return True
            if it["phase"] != phase:      # a handler moved the iteration (redesign)
                continue
            it["phase"] = PHASES[PHASES.index(phase) + 1]
            self.event("phase", f"{it['label']} -> {it['phase'].upper()}", it)
            self.save()
        return progressed

    def acquire_lock(self):
        self.lock_fh = open(self.wave_dir / ".supervisor.lock", "a+")
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"another supervisor holds {self.wave_dir / '.supervisor.lock'}")

    def run(self):
        if not self.st:
            sys.exit("not initialised: run `loopctl.sh init <wave>` first")
        self.acquire_lock()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_flag", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_flag", True))
        if self.st.get("status") == "BLOCKED":
            self.render()
            print("wave is BLOCKED; run `loopctl.sh resume <wave>` after fixing:", self.st.get("blocked"))
            return 2
        if self.st.get("status") == "DONE":
            self.render()
            return 0
        pid1 = L.pid1_start()
        if self.st.get("pid1_start") and self.st["pid1_start"] != pid1:
            self.event("container_restart", "PID 1 / boot changed since the last supervisor run: every recorded job is gone; "
                                            "training resumes with --resume after the preflight (ffmpeg)")
        self.st.update(pid1_start=pid1, supervisor_pid=os.getpid(), status="RUNNING", started_ts=L.now_ts())
        self.event("supervisor_start", f"pid {os.getpid()}")
        tick = float(self.cfg("gpu", "poll_s", default=60))
        errors = 0
        rc = 0
        while not self.stop_flag:
            if (self.wave_dir / "STOP").exists():
                self.st["status"] = "STOPPED"
                self.event("stopped", "STOP file found: supervisor exits; running jobs continue")
                break
            if self.st.get("status") == "PAUSED" and not (self.wave_dir / "PAUSE").exists():
                self.st["status"] = "RUNNING"
            try:
                progressed = self.step()
                errors = 0
            except Blocked:
                rc = 2
                break
            except Exception as exc:  # keep running; repeated errors block the wave
                errors += 1
                self.event("error", f"{type(exc).__name__}: {exc}", data=traceback.format_exc()[-3000:])
                if errors >= 5:
                    try:
                        self.block("supervisor", f"5 consecutive errors, last: {exc}", "see events.jsonl / logs/supervisor.out")
                    except Blocked:
                        rc = 2
                        break
                progressed = False
            self.st["heartbeat_ts"] = L.now_ts()
            self.st["heartbeat_epoch"] = time.time()
            self.save()
            self.render()
            if self.st.get("status") == "DONE":
                break
            t_end = time.time() + (1 if progressed else tick)
            while time.time() < t_end and not self.stop_flag:
                time.sleep(0.5)
        self.st["supervisor_pid"] = None
        self.save()
        self.render()
        self.event("supervisor_exit", f"status {self.st.get('status')}")
        return rc

    # ------------------------------------------------------------------------------ init / status / plan
    def init(self, force=False):
        if self.st and not force:
            sys.exit(f"{self.state_path} exists; this wave is already initialised")
        start = self.wave["start"]
        b = start["best"]
        best_run = self.abs(b["run_dir"])
        argv = L.config_argv(best_run) if b.get("argv_from", "config") == "config" else b["argv"]
        if not argv:
            sys.exit(f"no command in {best_run}/config.yaml")
        mv = L.readout_numbers(self.abs(b["readouts"]["merge"]) / "metrics.json", self.objective_key(), self.clean_key())
        fv = L.readout_numbers(self.abs(b["readouts"]["final"]) / "metrics.json", self.objective_key(), self.clean_key())
        if not mv or not fv:
            sys.exit("the best's readouts need metrics.json with the objective key")
        k = int(self.cfg("eval", "k", default=5))
        sub = best_run / f"model_merging_top{k}" / "submission.zip"
        self.st = {"wave": self.wave["wave"], "status": "INITIALISED", "created": L.now_ts(), "cost_usd": 0.0,
                   "iterations": [], "current": 1, "attempts": {}, "jobs": {}, "claude_sessions": [],
                   "best": {"label": b["label"], "iter": 0, "run_dir": self.rel(best_run),
                            "track": (L.read_config(best_run).get("args") or {}).get("track"),
                            "readouts": {"merge": b["readouts"]["merge"], "final": b["readouts"]["final"]},
                            "registry_ids": b.get("registry_ids", {}), "argv": argv,
                            "merge_value": mv["value"], "final_value": fv["value"],
                            "submission": self.rel(sub) if sub.is_file() else None},
                   "pid1_start": L.pid1_start()}
        for a in start.get("adopt", []) or []:
            k_it = len(self.st["iterations"]) + 1
            it = self.iteration(k_it)
            if it["letter"] != a["letter"]:
                sys.exit(f"adopted iteration {k_it} must have letter {it['letter']}")
            run = self.abs(a["run_dir"])
            cfg = L.read_config(run)
            argv_a = L.config_argv(run)
            track = (cfg.get("args") or {}).get("track")
            logs = sorted(glob.glob(str(self.abs(self.cfg("trainer", "cwd", default="SLS_setup")) / f"{track}_*.log")),
                          key=os.path.getmtime)
            design = {"iter": k_it, "letter": a["letter"], "track": track, "lever_slug": a["lever_slug"], "kind": "config",
                      "base_label": b["label"], "change": a.get("change", {}), "change_vs_ref": a["change_vs_ref"],
                      "rationale": a.get("rationale", ""), "hypothesis": a.get("hypothesis", ""),
                      "predicted_signature": a.get("predicted_signature", ""), "falsifier": a.get("falsifier", ""),
                      "eval_note": a.get("eval_note", ""), "header_text": "", "expect": a.get("expect", []),
                      "expected_hours": a.get("expected_hours", 0), "adopted": True}
            L.write_json(self.iter_dir(it) / "design.result.json", design)
            it.update(adopted=True, argv=argv_a, track=track, lever_slug=a["lever_slug"], change_vs_ref=a["change_vs_ref"],
                      run_dir=self.rel(run), launcher=a.get("launcher"), logs=logs, base_label=b["label"],
                      design=self.rel(self.iter_dir(it) / "design.result.json"), argv_checked=True, early_ok=True,
                      phase="train")
            self.event("adopt", f"iteration {k_it} = existing run {run.name}", it)
        notes = self.wave_dir / "user_notes.md"
        if not notes.exists():
            L.write_text(notes, f"# User notes for wave {self.wave['wave']}\n\n"
                                "Append with `scripts/exploop/loopctl.sh note <wave> \"...\"`. The DESIGN step reads them "
                                "first (newest last).\n")
        self.abs(self.wave["report_dir"]).mkdir(parents=True, exist_ok=True)
        self.save()
        self.render()
        print(f"initialised {self.state_path}: best {b['label']}, {len(self.st['iterations'])} adopted iteration(s)")

    def health(self):
        lock_held = False
        lp = self.wave_dir / ".supervisor.lock"
        if lp.exists():
            with open(lp, "a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except OSError:
                    lock_held = True
        hb = self.st.get("heartbeat_epoch") or 0
        age = time.time() - hb if hb else None
        return {"running": lock_held, "heartbeat_age_s": age,
                "pid1_changed": bool(self.st.get("pid1_start") and self.st["pid1_start"] != L.pid1_start())}

    def status(self):
        if not self.st:
            print(f"wave {self.wave['wave']}: not initialised")
            return 1
        h = self.health()
        s = self.st.get("status")
        it = next((i for i in self.st["iterations"] if i["k"] == self.st.get("current")), None)
        print(f"wave {self.wave['wave']}: {s}" + (f" - iteration {it['k']} {it['label']} phase {it['phase'].upper()}" if it else ""))
        print(f"  supervisor: {'RUNNING (lock held)' if h['running'] else 'DOWN (no supervisor holds the lock)'}"
              + (f", heartbeat {h['heartbeat_age_s']:.0f}s ago" if h['heartbeat_age_s'] is not None else ""))
        if h["pid1_changed"]:
            print("  container restarted since the supervisor last ran (PID 1 changed): run `loopctl.sh resume`")
        if it:
            for name, job in it.get("jobs", {}).items():
                print(f"  job {name}: pid {job['pid']} {'alive' if self.job_alive(job) else 'gone'} (since {job['started_ts']})")
            if it.get("run_dir"):
                print(f"  run dir: {it['run_dir']} (epochs logged: {it.get('epochs_seen', 0)})")
        if self.st.get("waiting"):
            print(f"  waiting: {self.st['waiting']}")
        if self.st.get("blocked"):
            b = self.st["blocked"]
            print(f"  BLOCKED in {b['phase']}: {b['reason']}\n  hint: {b.get('hint')}")
        b = self.st["best"]
        print(f"  best: {b['label']} merge {L._f(b.get('merge_value'))} final {L._f(b.get('final_value'))} ({b.get('run_dir')})")
        print(f"  claude spend: ${self.st.get('cost_usd', 0):.2f}; status page: {self.rel(self.wave_dir / 'STATUS.md')}")
        if s == "DONE":
            return 3
        if s == "BLOCKED":
            return 2
        return 0 if h["running"] else 1

    def plan(self):
        """Read-only: what the supervisor would do next, with expanded commands."""
        if not self.st:
            print("not initialised")
            return 1
        self.readonly = True
        if self.st.get("current") is None:
            print("all iterations done -> SUMMARY step (exp-analyst)")
            return 0
        it = next((i for i in self.st["iterations"] if i["k"] == self.st["current"]), None)
        if not it:
            print(f"iteration {self.st['current']} not started: next is DESIGN (claude -p --agent exp-designer)")
            return 0
        print(f"iteration {it['k']} {it['label']} ({it.get('track')}), phase {it['phase'].upper()}")
        if it["phase"] in ("design", "render", "smoke"):
            print(f"  next: {it['phase']}")
        if it["phase"] == "train":
            if it.get("run_dir") and self.train_complete(it):
                print(f"  training already complete ({it['run_dir']}): the next tick moves to EVAL")
            else:
                print(f"  launch: bash {it.get('launcher') or self.rel(self.launcher_path(it))}"
                      + (" --resume ..." if it.get("run_dir") else ""))
        if it["phase"] in ("train", "eval") and it.get("run_dir") and L.final_epoch(self.abs(it["run_dir"])):
            try:
                ctx = self.eval_ctx(it)
            except Blocked as exc:
                print(f"  eval would block: {exc}")
                return 2
            print("  eval steps (cwd: command [> stdout]); [done] = outputs already on disk:")
            for step in self.profile()["steps"]:
                done = self.step_done(step, ctx, it)
                cmd = shlex.join(expand_list(step["cmd"], ctx))
                out = f" > {expand_str(step['stdout'], ctx)}" if step.get("stdout") else ""
                print(f"   {'[done]' if done else '[todo]'} {step['name']}{' (GPU)' if step.get('gpu') else ''}\n"
                      f"      ({step.get('cwd', '.')}) {cmd}{out}")
            print(f"  then VERDICT vs {self.st['best']['label']} ({ctx['best_merge']}), LOG to {self.wave.get('registry')}, "
                  f"ANALYZE -> {self.wave['report_dir']}/<ts>_{it.get('track')}.md")
        return 0

    def tail(self, n):
        if not self.events_path.is_file():
            print("no events yet")
            return
        for line in self.events_path.read_text(encoding="utf-8").splitlines()[-n:]:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            it = f" [{e['iter']}]" if e.get("iter") else ""
            print(f"{e['ts']}{it} {e['kind']}: {e['msg']}")

    def unblock(self, reset_attempts):
        if not self.st:
            sys.exit("not initialised")
        if self.st.get("blocked"):
            self.event("unblock", f"cleared: {self.st['blocked'].get('reason', '')[:200]}")
        self.st["blocked"] = None
        if self.st.get("status") in ("BLOCKED", "STOPPED", "INITIALISED"):
            self.st["status"] = "RESUMING"
        if reset_attempts:
            for it in self.st["iterations"]:
                if not it.get("done"):
                    it["attempts"] = {k: 0 for k in it.get("attempts", {}) if k == "train"}
                    it["early_failures"] = 0
                    it["smoke_failures"] = 0
            self.st["attempts"] = {}
        self.save()


# ---------------------------------------------------------------------------------- helpers

LIST_KEYS = ("model_flags", "names", "all_dirs", "all_labels")


def expand_str(s, ctx):
    def rep(m):
        key = m.group(1)
        if key in ctx and not isinstance(ctx[key], list):
            return str(ctx[key])
        return m.group(0)
    return re.sub(r"\{([A-Za-z_]+)\}", rep, str(s))


def expand_list(tokens, ctx):
    out = []
    for t in tokens:
        m = re.fullmatch(r"\{([A-Za-z_]+)\}", str(t))
        if m and m.group(1) in LIST_KEYS:
            out.extend(str(x) for x in ctx[m.group(1)])
        else:
            out.append(expand_str(t, ctx))
    return out


def _same_merge_list(new_list, base, old_list):
    if not (new_list.is_file() and old_list.is_file()):
        return False
    new = sorted(os.path.realpath(base / ln.strip()) if not os.path.isabs(ln.strip()) else os.path.realpath(ln.strip())
                 for ln in new_list.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#"))
    old = sorted(os.path.realpath(ln.strip()) for ln in old_list.read_text().splitlines() if ln.strip())
    return new == old


def _report_ok(path):
    p = Path(path)
    if not p.is_file():
        return False
    text = p.read_text(encoding="utf-8", errors="replace")
    return all(re.search(rf"^##\s+{n}[.)]?\s", text, re.M) for n in range(1, 8))


def _fmt_ci(m):
    if not m or m.get("ci_low") is None:
        return "-"
    return f"[{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]"


def _wrap_comment(text, width=100):
    words = str(text).split()
    lines, cur = [], "#  "
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip("# "):
            lines.append(cur.rstrip())
            cur = "#    "
        cur += w + " "
    if cur.strip("# "):
        lines.append(cur.rstrip())
    return lines


def _json_from_text(text):
    """Fallback when the result has no structured_output: the last fenced ```json block, else the
    longest parseable {...} span."""
    for block in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)):
        try:
            return json.loads(block)
        except ValueError:
            continue
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for st in starts:
        for en in range(len(text), st, -1):
            if text[en - 1] == "}":
                try:
                    return json.loads(text[st:en])
                except ValueError:
                    continue
        break
    return None



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "init", "status", "plan", "tail", "unblock"])
    ap.add_argument("--wave", required=True, help="wave dir, e.g. exploop/B3")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reset-attempts", action="store_true")
    a = ap.parse_args()
    s = Supervisor(a.wave)
    if a.cmd == "run":
        sys.exit(s.run())
    if a.cmd == "init":
        s.init(force=a.force)
    elif a.cmd == "status":
        sys.exit(s.status())
    elif a.cmd == "plan":
        sys.exit(s.plan())
    elif a.cmd == "tail":
        s.tail(a.n)
    elif a.cmd == "unblock":
        s.unblock(a.reset_attempts)


if __name__ == "__main__":
    main()
