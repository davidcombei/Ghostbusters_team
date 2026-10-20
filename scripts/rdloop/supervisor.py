#!/usr/bin/env python3
"""rdloop supervisor: runs one research & development cycle unattended, one process per cycle.

    supervisor.py new     --cycle rdloop/<C> --brief "..." [--auto-gate] [--from-wave W] ...
    supervisor.py init    --cycle rdloop/<C>     # state.json, isolation manifest, frozen copy of untracked files
    supervisor.py run     --cycle rdloop/<C>     # the loop (start it with rdctl.sh start <C>)
    supervisor.py status  --cycle rdloop/<C>     # exit 0 running, 1 down, 2 blocked / needs owner, 3 awaiting owner review / final
    supervisor.py plan    --cycle rdloop/<C>     # what runs next, read-only
    supervisor.py tail    --cycle rdloop/<C> [-n 30]
    supervisor.py approve --cycle rdloop/<C> [--levers a,b]      # plan gate
    supervisor.py gate    --cycle rdloop/<C> owner|auto
    supervisor.py reopen  --cycle rdloop/<C> --at propose|implement
    supervisor.py park | unblock [--reset-attempts] [--accept-manifest] | verify | handed-off --wave W

Phases: WAIT_WAVES -> AUDIT (research-auditor) -> SCOUT (research-scout) -> PROPOSE (research-architect)
-> LINT -> CRITIQUE (research-critic) -> [revise ... up to debate.max_rounds] -> GATE (owner, or auto)
-> SCAFFOLD -> IMPLEMENT core + one step per lever (research-engineer) -> STATIC -> REVIEW
(research-critic) <-> FIX (research-engineer) -> GPU session (regression on the old runs + smoke,
<= 30 min) -> COMPAT (exploop dry run) -> SUMMARY (research-auditor) -> AWAITING_USER.

All state is on disk (state.json + files); every phase re-checks its outputs, so a killed supervisor
(closed terminal, dropped SSH, container restart) continues where it stopped. The supervisor never
commits, never starts an exploop wave, never deletes anything outside its own cycle dir / smoke runs.
"""
import argparse
import fcntl
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
import rdlib as R  # noqa: E402

L = R.L
EXIT_STATUSES = ("AWAITING_USER", "NEEDS_USER", "BLOCKED", "PARKED", "HANDED_OFF")
TRAIN_FORBID = [r"Traceback \(most recent call last\)", r"out of memory", r"CUDA error", r"failed \("]
# R1 sets: clean + matched + heldout, so local_eval can compute (and print) WF1; without them it exits 1
REGRESSION_NAMES = ["dev_online_clean", "sim_matched_v1", "sim_heldout_v1", "probe_orig", "probe_noise_only"]
# utils/rtc_augment.py and utils/level_augment.py seed the augmenter with the worker pid, so no two runs draw
# the same augmentations. The equivalence micro-runs drop these flags to compare the trainers deterministically;
# the augmenter's own levers-off identity is proven bit for bit by the module's CPU checks.
EQUIV_NOAUG_FLAGS = ["--use_rtc_aug", "--musan", "--use_level_aug"]


class Blocked(Exception):
    pass


class Stop(Exception):
    """The cycle reached a status where the supervisor exits (owner needed / final)."""


class RequestError(Exception):
    """A control request (reopen / gate / park) that cannot be applied now."""


class Supervisor:
    def __init__(self, cycle_dir):
        self.cycle_dir = Path(cycle_dir).resolve()
        self.cycle = R.read_json(self.cycle_dir / "cycle.json")
        if not self.cycle:
            sys.exit(f"no readable cycle.json in {self.cycle_dir}")
        self.repo = Path(self.cycle.get("repo") or R.REPO).resolve()
        self.state_path = self.cycle_dir / "state.json"
        self.events_path = self.cycle_dir / "events.jsonl"
        self.st = R.read_json(self.state_path)
        self.py = self.cycle.get("python", sys.executable)
        self.name = self.cycle["cycle"]
        self.lay = R.module_layout(self.cycle)
        self.stop_flag = False
        self.lock_fh = None
        self.readonly = False

    # ------------------------------------------------------------------------------ utilities
    def abs(self, p):
        p = Path(str(p))
        return p if p.is_absolute() else (self.repo / p)

    def rel(self, p):
        return R.rel(self.repo, p)

    def cfg(self, *keys, default=None):
        return R.get(self.cycle, *keys, default=default)

    def event(self, kind, msg, **data):
        if self.readonly:
            return
        rec = {"ts": R.now_ts(), "phase": (self.st or {}).get("phase"), "kind": kind, "msg": msg}
        if data:
            rec["data"] = data
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[{rec['ts']}] {kind}: {msg}", flush=True)

    def save(self):
        if not self.readonly:
            R.write_json(self.state_path, self.st)

    def render(self):
        if self.st and not self.readonly:
            R.write_text(self.cycle_dir / "STATUS.md", R.render_status(self.cycle_view(), self.st, self.cycle_dir))

    def cycle_view(self):
        c = dict(self.cycle)
        c.setdefault("gates", {})
        c["gates"] = dict(c["gates"], plan=self.st.get("gate_mode", R.get(self.cycle, "gates", "plan", default="owner")))
        return c

    def block(self, phase, reason, hint=""):
        if self.readonly:
            raise Blocked(reason)
        self.st["status"] = "BLOCKED"
        self.st["blocked"] = {"phase": phase, "reason": reason, "hint": hint, "ts": R.now_ts()}
        self.st["next"] = f"fix the cause, then `scripts/rdloop/rdctl.sh resume {self.name}`"
        self.event("blocked", f"{phase}: {reason}")
        self.save()
        self.render()
        raise Blocked(reason)

    def needs_user(self, items, next_hint):
        self.st["status"] = "NEEDS_USER"
        self.st["needs_user"] = list(items)
        self.st["next"] = next_hint
        self.event("needs_user", "; ".join(items)[:500])
        self.save()
        self.render()
        raise Stop("NEEDS_USER")

    def set_phase(self, phase, why=""):
        old = self.st.get("phase")
        self.st["phase"] = phase
        if old != phase:
            self.event("phase", f"{str(old).upper()} -> {phase.upper()}" + (f" ({why})" if why else ""))
        self.save()
        return True

    def path(self, *parts):
        p = self.cycle_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def notes_text(self):
        p = self.cycle_dir / "user_notes.md"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def base(self):
        return self.st.get("base") or {}

    def lever_ids(self, which="approved"):
        dropped = self.st.get("dropped_levers") or {}
        ids = (self.st.get("approved_levers") or []) if which == "approved" else self.plan_levers_ids()
        return [i for i in ids if i not in dropped]

    def plan_levers_ids(self):
        return [lv["id"] for lv in (self.plan() or {}).get("levers", [])]

    def plan(self):
        return R.read_json(self.cycle_dir / "plan_agreed.json")

    def plan_lever(self, lid):
        return next((lv for lv in (self.plan() or {}).get("levers", []) if lv["id"] == lid), None)

    def drop_lever(self, lid, stage, reason):
        d = self.st.setdefault("dropped_levers", {})
        if lid not in d:
            d[lid] = f"{stage}: {reason}"[:400]
            self.event("lever_dropped", f"{lid} dropped at {stage}: {reason[:300]}", lever=lid, stage=stage)

    # ------------------------------------------------------------------------------ jobs
    def start_job(self, name, argv, cwd, stdout=None, env_extra=None, stderr=None):
        """Start argv in its own session; rc goes to jobs/<name>.a<k>.rc. With `stdout`, output goes
        to that file (written directly; the job's rc says whether it succeeded)."""
        jobs_dir = self.cycle_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        att = self.st.setdefault("job_attempts", {})
        k = att.get(name, 0)
        att[name] = k + 1
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        rc_file = jobs_dir / f"{safe}.a{k}.rc"
        if rc_file.exists():
            rc_file.unlink()
        if stdout:
            log = self.abs(stdout)
            log.parent.mkdir(parents=True, exist_ok=True)
            redirect = f"> {shlex.quote(str(log))} 2>" + (shlex.quote(str(stderr)) if stderr else "&1")
        else:
            log = jobs_dir / f"{safe}.a{k}.log"
            redirect = f"> {shlex.quote(str(log))} 2>" + (shlex.quote(str(stderr)) if stderr else "&1")
        script = (f"cd {shlex.quote(str(cwd))} && {shlex.join([str(a) for a in argv])} {redirect} </dev/null; "
                  f"echo $? > {shlex.quote(str(rc_file))}")
        env = dict(os.environ)
        env.update({k2: str(v) for k2, v in (env_extra or {}).items()})
        p = subprocess.Popen(["bash", "-c", script], start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        ticks = None
        for _ in range(50):
            ticks = L.proc_start_ticks(p.pid)
            if ticks is not None:
                break
            time.sleep(0.02)
        job = {"name": name, "pid": p.pid, "start_ticks": ticks, "rc_file": str(rc_file), "log": str(log),
               "started": time.time(), "started_ts": R.now_ts(), "attempt": k, "cmd": shlex.join([str(a) for a in argv])}
        self.st.setdefault("jobs", {})[name] = job
        self.event("job_start", f"{name} (pid {p.pid}): {job['cmd'][:300]}")
        return job

    def job_alive(self, job):
        return L.proc_alive(job.get("pid"), job.get("start_ticks"))

    def job_rc(self, job):
        try:
            return int(Path(job["rc_file"]).read_text().strip())
        except (OSError, ValueError, KeyError):
            return None

    def kill_job(self, job, sig=signal.SIGTERM):
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

    def timeout_job(self, job, limit_s):
        """SIGINT -> 60 s -> SIGTERM -> 60 s -> SIGKILL once a job runs longer than limit_s."""
        el = time.time() - job["started"]
        if el < limit_s:
            return False
        stage = job.get("kill_stage", 0)
        since = time.time() - job.get("kill_ts", 0)
        if stage == 0:
            self.event("timeout", f"{job['name']} ran {el / 60:.1f} min > {limit_s / 60:.1f} min: SIGINT")
            self.kill_job(job, signal.SIGINT)
            job.update(kill_stage=1, kill_ts=time.time(), timed_out=True)
        elif stage == 1 and since > 60:
            self.kill_job(job, signal.SIGTERM)
            job.update(kill_stage=2, kill_ts=time.time())
        elif stage == 2 and since > 60:
            self.kill_job(job, signal.SIGKILL)
            job.update(kill_stage=3, kill_ts=time.time())
        return True

    def own_pids(self):
        out = []
        for job in (self.st.get("jobs") or {}).values():
            if self.job_alive(job):
                out.append(job["pid"])
                try:
                    for d in Path("/proc").iterdir():
                        if d.name.isdigit():
                            try:
                                if os.getpgid(int(d.name)) == job["pid"]:
                                    out.append(int(d.name))
                            except OSError:
                                continue
                except OSError:
                    pass
        return out

    # ------------------------------------------------------------------------------ gates
    def ready_for_gpu(self):
        if (self.cycle_dir / "PAUSE").exists():
            self.st.update(status="PAUSED", waiting="paused by the owner (PAUSE file): no GPU job starts")
            return False
        g = self.cfg("gpu", default={}) or {}
        if g.get("yield_to_exploop", True):
            why = R.exploop_gpu_busy(self.repo)
            if why:
                self.st.update(status="WAITING", waiting=why, gpu_idle_since=None)
                return False
        if g.get("enabled", True):
            mem = L.gpu_memory_used_mib()
            busy = L.busy_processes(g.get("busy_regex", "main_train|b\\d+wave\\.(train|checks)|local_eval|main_eval|eval_checkpoints"),
                                    exclude_pids=self.own_pids())
            if mem is None:
                self.st.update(status="WAITING", waiting="nvidia-smi failed; cannot verify the GPU is idle", gpu_idle_since=None)
                return False
            if any(m >= int(g.get("max_mem_mib", 1024)) for m in mem) or busy:
                self.st.update(status="WAITING", gpu_idle_since=None,
                               waiting=f"GPU busy: memory.used {mem} MiB; processes {[p for p, _ in busy][:5]}")
                return False
            need = int(g.get("debounce_s", 120))
            if not self.st.get("gpu_idle_since"):
                self.st["gpu_idle_since"] = time.time()
            idle = time.time() - self.st["gpu_idle_since"]
            if idle < need:
                self.st.update(status="WAITING", waiting=f"GPU idle for {idle:.0f}s; need {need}s before the GPU session")
                return False
        free = L.disk_free_gb(self.repo)
        need_gb = float(self.cfg("disk_gb", "gpu", default=0) or 0)
        if free is not None and free < need_gb:
            self.st.update(status="WAITING", waiting=f"disk: {free:.0f} GB free < {need_gb:.0f} GB needed for the smoke runs")
            return False
        self.st.update(status="RUNNING", waiting=None, gpu_idle_since=None)
        return True

    def wait_for_exploop_claude(self):
        """Module writes during an exploop Claude step would trip that wave's integrity snapshot."""
        why = R.exploop_claude_busy(self.repo)
        if why:
            self.st.update(status="WAITING", waiting=f"{why}; module writes wait for it")
            return True
        if self.st.get("status") == "WAITING":
            self.st.update(status="RUNNING", waiting=None)
        return False

    # ------------------------------------------------------------------------------ claude steps
    def can_spend(self, usd):
        cap = float(self.cfg("claude", "max_total_usd", default=150))
        return float(self.st.get("cost_usd", 0.0)) + float(usd) <= cap

    def claude_step(self, step, tag, prompt_fn, write_files, usd=None):
        """Run one headless claude step as a job. Returns None while running, else (ok, payload|errors)."""
        agent, policy, schema_name, _skill = R.STEPS[step]
        sc = self.cfg("claude", "steps", step, default={}) or {}
        key = f"{step}:{tag}"
        jname = f"claude:{key}"
        job = self.st.setdefault("jobs", {}).get(jname)
        if job:
            if self.job_alive(job):
                self.timeout_job(job, float(sc.get("timeout_min", 45)) * 60)
                return None
            self.st["jobs"].pop(jname)
            return self.finish_claude(step, tag, job)
        att = self.st.setdefault("attempts", {})
        n = att.get(key, 0)
        if n >= int(sc.get("attempts", 3)):
            errs = self.st.get("step_errors", {}).get(key, [])[-3:]
            self.block(step, f"{key} failed {n} times: {errs}",
                       f"look at {self.rel(self.cycle_dir / 'steps')}/{step}_{tag}.a*.out.json / .err; fix, then "
                       f"`rdctl.sh resume {self.name} --reset-attempts`")
        if time.time() < self.st.get("backoff_until", 0):
            self.st["waiting"] = f"{step}: backing off after an API error until {time.ctime(self.st['backoff_until'])}"
            return None
        usd = float(usd if usd is not None else sc.get("usd", 10))
        if not self.can_spend(usd):
            self.block(step, f"Claude budget: spent ${self.st.get('cost_usd', 0):.2f}, cap ${self.cfg('claude', 'max_total_usd')}",
                       "raise claude.max_total_usd in cycle.json, then resume")
        steps_dir = self.cycle_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = steps_dir / f"{step}_{tag}.a{n}.prompt.md"
        R.write_text(prompt_path, prompt_fn())
        att[key] = n + 1
        out = steps_dir / f"{step}_{tag}.a{n}.out.json"
        err = steps_dir / f"{step}_{tag}.a{n}.err"
        exact, prefixes = R.write_scope(self.repo, self.scope_cycle(), self.cycle_dir, policy, files=write_files)
        settings = steps_dir / f"{step}_{tag}.a{n}.settings.json"
        R.write_json(settings, R.step_settings(policy, self.repo, self.scope_cycle(), self.cycle_dir, exact, prefixes, self.py))
        sid = str(uuid.uuid4())
        schema = (HERE / "schemas" / f"{schema_name}.json").read_text(encoding="utf-8")
        prompt = (f"Read {prompt_path} and do exactly what it says. You are an unattended step of the rdloop "
                  f"supervisor: nobody can answer questions, so never ask; finish with the structured output.")
        argv = [self.cfg("claude", "bin", default="claude"), "-p", prompt, "--agent", agent,
                "--settings", str(settings), "--permission-mode", self.cfg("claude", "permission_mode", default="auto"),
                "--permission-prompts", "none", "--output-format", "json", "--json-schema", json.dumps(json.loads(schema)),
                "--max-budget-usd", str(usd), "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
                "--session-id", sid, "-n", f"rdloop-{self.name}-{step}-{tag}-a{n}"]
        if self.cfg("claude", "model"):
            argv += ["--model", self.cfg("claude", "model")]
        tmp = f"{R.tmpdir(self.cycle)}/{step}_{tag}"
        Path(tmp).mkdir(parents=True, exist_ok=True)
        env = {"RDLOOP_STEP": step, "RDLOOP_POLICY": policy, "RDLOOP_CYCLE_DIR": str(self.cycle_dir), "RDLOOP_REPO": str(self.repo),
               "RDLOOP_PROMPT": str(prompt_path), "RDLOOP_WRITE_FILES": ":".join(exact), "RDLOOP_WRITE_PREFIXES": ":".join(prefixes),
               "RDLOOP_WEB_DOMAINS": ",".join(self.cfg("web", "domains", default=R.DEFAULT_WEB_DOMAINS)),
               "CUDA_VISIBLE_DEVICES": "", "TMPDIR": tmp, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "DISABLE_AUTOUPDATER": "1", "PYTHONDONTWRITEBYTECODE": "1",
               # `python` in a step = the run env (torch, transformers), not whatever python3 is first on PATH
               "PATH": f"{Path(self.py).parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
        if policy == "engineer":
            argv = ["nice", "-n", "10"] + argv
        job = self.start_job(jname, argv, self.repo, stdout=str(out), env_extra=env, stderr=str(err))
        job.update(out=str(out), err=str(err), session_id=sid, step=step, tag=tag, agent=agent, attempt=n,
                   prompt=self.rel(prompt_path))
        self.st.setdefault("claude_sessions", []).append({"step": step, "tag": tag, "agent": agent, "attempt": n,
                                                          "session_id": sid, "status": "running", "out": self.rel(out),
                                                          "started": R.now_ts()})
        t = self.st.setdefault("team", {}).setdefault(agent, {})
        t.update(state="running", step=f"{step} {tag}", session_id=sid, attempt=n, started=time.time(),
                 started_ts=R.now_ts(), finished=None, job=jname)
        return None

    def scope_cycle(self):
        c = dict(self.cycle)
        c["_state_dict_change"] = bool(self.st.get("state_dict_change"))
        return c

    def finish_claude(self, step, tag, job):
        agent = job.get("agent")
        key = f"{step}:{tag}"
        res = R.read_json(job.get("out")) or {}
        sess = next((s for s in reversed(self.st.get("claude_sessions", [])) if s["session_id"] == job.get("session_id")), {})
        cost = float(res.get("total_cost_usd") or 0.0)
        self.st["cost_usd"] = round(float(self.st.get("cost_usd", 0.0)) + cost, 4)
        sess.update(cost=cost, finished=R.now_ts())
        team = self.st.setdefault("team", {}).setdefault(agent, {})
        team.update(cost=round(float(team.get("cost", 0)) + cost, 4), finished=time.time(), job=None)
        errs = []
        if job.get("interrupted"):
            errs.append("interrupted by a container restart")
        if not res:
            errs.append(f"no JSON result (rc {self.job_rc(job)}); stderr: {R.tail_bytes(job.get('err', ''), 800)}")
        elif res.get("is_error"):
            msg = str(res.get("result", ""))[:500]
            errs.append(f"claude reported an error: {msg}")
            if re.search(r"rate.?limit|overloaded|529|500|timeout|ECONN|network|auth", msg, re.I):
                mins = [10, 30, 60][min(self.st.get("attempts", {}).get(key, 1) - 1, 2)]
                self.st["backoff_until"] = time.time() + 60 * mins
        payload = None
        if not errs:
            payload = res.get("structured_output")
            if payload is None:
                payload = json_from_text(str(res.get("result", "")))
            if not isinstance(payload, dict):
                errs.append("no structured output in the result")
        if res.get("permission_denials"):
            self.event("permission_denials", f"{step} {tag}: {len(res['permission_denials'])} tool calls denied",
                       denials=res["permission_denials"][:10])
        if R.STEPS[step][1] == "engineer":
            self.audit_transcript(step, tag, job.get("session_id"))
            self.check_isolation(step)
        if job.get("interrupted") and not errs[1:]:
            self.st["attempts"][key] = max(0, self.st["attempts"].get(key, 1) - 1)    # a restart is not a failure
        sess["status"] = "ok" if not errs else "failed"
        team["state"] = "done" if not errs else "failed"
        if errs:
            self.st.setdefault("step_errors", {}).setdefault(key, []).extend(errs)
            self.event("step_failed", f"{step} {tag} attempt {job.get('attempt')}: {errs[0][:300]}")
            return (False, errs)
        self.event("step_done", f"{step} {tag} attempt {job.get('attempt')} finished (${cost:.2f})")
        return (True, payload)

    def audit_transcript(self, step, tag, sid):
        """Log every Bash command and file write of an engineer step to impl/commands.log; flag CUDA
        re-enabling in written files."""
        evs = R.transcript_events(sid)
        lines = [f"## {R.now_ts()} {step} {tag} (session {sid})"]
        for e in evs:
            if e["kind"] != "tool":
                continue
            ti = e["input"]
            if e["name"] == "Bash":
                lines.append(f"$ {ti.get('command', '')}")
            elif e["name"] in ("Write", "Edit", "MultiEdit"):
                lines.append(f"{e['name']} {ti.get('file_path', '')}")
                text = str(ti.get("content", "")) + str(ti.get("new_string", ""))
                if re.search(r"CUDA_VISIBLE_DEVICES['\"]?\]?\s*=", text):
                    self.event("integrity_warning", f"{step} {tag} wrote code that sets CUDA_VISIBLE_DEVICES: {ti.get('file_path')}")
        with open(self.path("impl", "commands.log"), "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")

    def check_isolation(self, where):
        man = R.read_json(self.cycle_dir / "isolation" / "manifest.json")
        if not man:
            return
        v = R.check_manifest(self.repo, man, self.cycle, self.rel(self.cycle_dir))
        if v:
            R.write_json(self.cycle_dir / "isolation" / "violations.json", {"ts": R.now_ts(), "where": where, "violations": v})
            self.block(where, f"files outside the cycle scope changed ({len(v)}): {v[:6]}",
                       f"inspect rdloop/{self.name}/isolation/violations.json (nothing was reverted; untracked originals are "
                       f"in isolation/frozen_untracked.tar.gz). If the change was yours: rdctl.sh resume {self.name} --accept-manifest")

    # ------------------------------------------------------------------------------ prompts
    def header(self, step, tag, write_files):
        agent, policy, _schema, skill = R.STEPS[step]
        b = self.base()
        argv = " ".join(shlex.quote(x) for x in b.get("argv", []))
        notes = self.notes_text().strip() or "(none)"
        writes = "\n".join(f"- `{self.rel(f)}`" for f in write_files) or "- nothing (you return structured output only)"
        if policy == "engineer":
            writes += (f"\n- anything under `{self.lay['module_dir']}`, files named `{self.lay['scripts_prefix']}*`, "
                       f"`rdloop/{self.name}/impl/`, `rdloop/{self.name}/scratch/` and $TMPDIR")
        return f"""# rdloop cycle {self.name} - {step.upper()} {tag} ({agent})

You are an unattended step of the rdloop supervisor (a research & development cycle that builds new,
flag-gated levers for the RTC-SDD detector, which an exploop wave then screens). Nobody can answer
questions: never ask; finish with the structured output that matches the JSON schema.

**First read `.claude/skills/{skill}/SKILL.md` and follow it** (skills are not preloaded in headless runs).

## Cycle
- spec: `rdloop/{self.name}/cycle.json` (focus {self.cfg('focus', default=[])}, exclude {self.cfg('exclude', default=[])},
  levers {self.cfg('levers', 'min', default=2)}-{self.cfg('levers', 'max', default=4)}, limits {json.dumps(self.cfg('limits', default={}))})
- owner brief: `rdloop/{self.name}/brief.md` (read it)
- module this cycle builds: `{self.lay['module_dir']}` (scaffolded from `SLS_setup/{b.get('entry_module')}/`),
  scripts prefix `{self.lay['scripts_prefix']}*`
- base (current best): **{b.get('label')}** `{b.get('run_dir')}` (best of exploop wave {b.get('from_wave')});
  readouts: merge `{b.get('readouts', {}).get('merge')}`, final `{b.get('readouts', {}).get('final')}`;
  final checkpoint `{b.get('ckpt_final')}`; entry module `{b.get('entry_module')}`
- base trainer argv:
```
{argv}
```

## Owner notes (highest priority; the newest note wins)
{notes}

## Files you may write
{writes}

Anything else is denied by the guard hook, and the supervisor checks the sha256 of every file outside
the cycle scope after engineer steps. Never read progress-split scores. Never type a metric value from
memory: copy numbers from files you read and name the file.
"""

    def ctx_docs(self):
        docs = [d for d in self.cfg("context", "docs", default=[]) if self.abs(d).exists()]
        return "\n".join(f"- `{d}`" for d in docs)

    def prompt_audit(self, report):
        b = self.base()
        return self.header("audit", "a", [report]) + f"""
## Task: AUDIT
Analyse the base and everything tried so far; find the gaps that remain. Evidence only.
- registry: `docs/experiment_log.csv`; exploop waves: `exploop/*/loop_state.json`, `exploop/*/wave.json`,
  `exploop/*/plan_snapshot.md`; reports: `docs/experiments-exploration/**` (newest first; wave summaries)
- base readouts: `{b.get('readouts', {}).get('merge')}/metrics.json`, `{b.get('readouts', {}).get('final')}/metrics.json`,
  `probe_report.json`, `compare_ci*.json`; per-epoch: `{b.get('run_dir')}/logs/train.log`, `metrics.jsonl`
- context docs:
{self.ctx_docs()}
- the lever ledger of earlier cycles: `docs/research/lever_ledger.md` (do not re-open exhausted levers)

Write the report to `{self.rel(report)}` (structure in the skill), then return the structured output.
"""

    def prompt_scout(self, report):
        return self.header("scout", "a", [report]) + f"""
## Task: SCOUT
Search the state of the art for speech-deepfake detection under real-time-communication conditions and
turn it into idea cards that target the audit's gaps.
- audit: `rdloop/{self.name}/01_audit.md` and `rdloop/{self.name}/audit.json` (gap ids G*)
- literature already collected: `docs/research/literature.md`, `docs/experiments_roadmap.md` §14
- the code you are proposing for: `SLS_setup/{self.base().get('entry_module')}/README.md`, `SLS_setup/model/sls_model.py`

Every source you cite must be fetched with WebFetch in this step (the supervisor marks a source
verified only when its URL is in your WebFetch calls). Write `{self.rel(report)}`, then return the
structured output.
"""

    def prompt_propose(self, v, report, lint_errors):
        d = self.st.get("debate") or {}
        parts = [self.header("propose", f"v{v}", [report]), f"""
## Task: {'PROPOSE' if v == 1 else 'REVISE'} - proposal version {v}
Turn the audit and the idea cards into a design plan of {self.cfg('levers', 'min', default=2)}-{self.cfg('levers', 'max', default=4)}
flag-gated levers (each flag defaults to the base behaviour; one component per lever; small deltas).
- audit: `rdloop/{self.name}/01_audit.md` / `audit.json`; ideas: `rdloop/{self.name}/02_sota.md` / `scout.json`
  (sources marked `verified: false` were not fetched; do not build a lever on them alone)
- code to extend: `SLS_setup/{self.base().get('entry_module')}/` (train.py, README.md flag table), `SLS_setup/model/`, `SLS_setup/utils/`
Write the human-readable plan to `{self.rel(report)}` and return the structured output with `version: {v}`.
"""]
        if v > 1:
            prev = d.get("version")
            parts.append(f"""
## Debate so far
- your previous proposal: `rdloop/{self.name}/debate/proposal_v{prev}.md` / `.json`
- the critique of it: `rdloop/{self.name}/debate/critique_v{prev}.md` / `.json`
- the objection ledger: `rdloop/{self.name}/debate/ledger.json`
Answer EVERY open objection in `responses` (accept / modify / rebut with evidence) and every critic
alternative in `alternatives_response`. Open blocker / major objections now:
""")
            for o in R.open_objections(d.get("ledger") or {}, R.SEVERITIES):
                parts.append(f"- **{o['id']}** [{o['severity']}] lever `{o.get('lever_id') or 'plan'}`: {o['claim'][:300]} "
                             f"-> required: {o.get('required_change', '')[:200]}\n")
        if self.st.get("reopen_note"):
            parts.append(f"\n## The owner sent the plan back\n{self.st['reopen_note']}\nRead the owner notes above and revise.\n")
        if lint_errors:
            parts.append("\n## Your last attempt failed the supervisor's LINT; fix these and resubmit:\n"
                         + "\n".join(f"- {e}" for e in lint_errors) + "\n")
        return "".join(parts)

    def prompt_critique(self, v, report):
        d = self.st.get("debate") or {}
        done_rounds = len(d.get("rounds", [])) - int(d.get("rounds_offset", 0))
        rounds_left = int(self.cfg("debate", "max_rounds", default=3)) - done_rounds
        ledger = "\n".join(f"- {o['id']} [{o['severity']}, {o['status']}] `{o.get('lever_id') or 'plan'}`: {o['claim'][:200]}"
                           for o in (d.get("ledger") or {}).values()) or "(empty: first round)"
        return self.header("critique", f"v{v}", [report]) + f"""
## Task: CRITIQUE proposal v{v} (round {done_rounds + 1} of {self.cfg('debate', 'max_rounds', default=3)}; {rounds_left - 1} revision(s) left after this one)
Grill the plan: correctness, leakage of eval-only ops / pools, challenge rules, default-off equivalence,
feasibility in the code, VRAM / time budget, confounds, evaluation validity, weak or unverified evidence,
redundancy with exhausted levers. Propose better alternatives where you see them.
- proposal: `rdloop/{self.name}/debate/proposal_v{v}.md` / `.json`
- audit / ideas: `rdloop/{self.name}/01_audit.md`, `02_sota.md`, `scout.json`; lint: `rdloop/{self.name}/debate/lint_v{v}.json`
- objection ledger so far:
{ledger}

Rules: re-raise an unresolved objection under its old id; list fixed ones in `resolved`; give every
blocker concrete evidence (file:line, URL, command output) - a blocker without evidence is downgraded.
Agreement is reached when no blocker / major objection is open. Write `{self.rel(report)}`, then
return the structured output with `version: {v}`.
"""

    def prompt_implement(self, part):
        plan = self.plan() or {}
        lv = self.plan_lever(part) if part != "core" else None
        files = [self.path("impl", "progress.md")]
        done = [p for p in self.st.get("impl", {}).get("parts", []) if p["status"] == "done"]
        head = self.header("implement", part, files)
        body = f"""
## Task: IMPLEMENT part `{part}`
The plan the owner approved: `rdloop/{self.name}/plan_agreed.md` / `plan_agreed.json`; approved levers:
{', '.join(self.lever_ids())}. The module was scaffolded by the supervisor: `{self.lay['module_dir']}train.py` is a
copy of `SLS_setup/{self.base().get('entry_module')}/train.py` (+ the `{self.cycle['module']}:` version line).
Keep `rdloop/{self.name}/impl/progress.md` current (what is done / left) - a retry continues from it.
Parts done so far: {', '.join(p['id'] for p in done) or 'none'}.
"""
        if part == "core":
            body += f"""
This part builds the module skeleton every lever plugs into (see the skill's module contract):
`__init__.py` (version), `README.md` (flag table, reproduction statement), `checks.py` with the targets
`cpu`, `gpu --ckpt <path>` and `overfit --lever <id>` (each prints PASS/FAIL lines, exits non-zero on
failure), `module_sha256` in config.yaml, the eval entry point if a lever changes the state_dict
({'needed: ' + str([x['id'] for x in plan.get('levers', []) if x.get('state_dict_change')]) if self.st.get('state_dict_change') else 'not needed in this cycle'}).
Do not implement any lever yet.
"""
        else:
            body += f"""
Implement lever `{part}` exactly as approved (flags, defaults, header lines, files):
```json
{json.dumps(lv, indent=2)[:6000]}
```
Add its CPU checks to `checks.py cpu` (lever-off equivalence first), its GPU checks to `checks.py gpu`,
and `overfit --lever {part}` if it adds parameters. Run the CPU checks before you finish.
"""
        body += "\nYou run on the CPU only (CUDA is hidden); GPU checks and smoke runs are run by the supervisor later.\n"
        return head + body

    def prompt_review(self, n, report, diff_path, prev):
        return self.header("review", f"r{n}", [report]) + f"""
## Task: CODE REVIEW round {n}
Review the implementation against the approved plan before any GPU time is spent.
- plan: `rdloop/{self.name}/plan_agreed.md` / `.json` (approved levers {', '.join(self.lever_ids())})
- the module: `{self.lay['module_dir']}`; scripts `{self.lay['scripts_prefix']}*`
- diff to review: `{self.rel(diff_path)}` ({'full diff of the module against the base trainer + new files' if n == 1 else 'changes since review round ' + str(n - 1)})
- engineer reports: `rdloop/{self.name}/impl/*.json`, `impl/progress.md`; static checks: `rdloop/{self.name}/tests/static/`
{('- open findings from earlier rounds: ' + ', '.join(prev)) if prev else ''}
Check: plan conformance, default-off equivalence (flags off == base bit for bit), leakage of reserved
eval ops / pools, isolation (nothing outside the module), correctness / numerics, test gaps, and the
exploop contract (log lines, config.yaml command, ckpt names, `Experiment saved to:`). Mark earlier
findings that are fixed in `resolved`. Write `{self.rel(report)}`, then return the structured output.
"""

    def prompt_fix(self, n, reason, material):
        files = [self.path("impl", "progress.md")]
        return self.header("fix", f"f{n}", files) + f"""
## Task: FIX round {n} (reason: {reason})
{material}

Fix the cause inside the module / scripts prefix only; keep every flag's default reproducing the base.
Re-run the CPU checks (`python -m {self.cycle['module']}.checks cpu` from SLS_setup/) before you finish.
List what you addressed in `addressed`; set `part: "fix"`.
"""

    def prompt_summary(self, report, facts_path):
        return self.header("summary", "s", [report]) + f"""
## Task: SUMMARY for the owner
Write the final summary of this cycle: the ideas implemented (and why: gaps, sources), the debate
outcome, what was dropped and why, the regression and smoke results, the cost, and the recommended
exploop wave (order of the levers, expected hours). Facts computed by the supervisor (copy numbers
from here): `{self.rel(facts_path)}`. Also read `rdloop/{self.name}/plan_agreed.md`, `01_audit.md`,
`02_sota.md`, `debate/`, `review/`, `impl/`, `tests/`, `handoff/wave_seed.json`.
Write `{self.rel(report)}` (structure in the rd-audit skill, "Final summary"), then return the
structured output.
"""

    # ------------------------------------------------------------------------------ phases
    def ph_wait_waves(self):
        pending = []
        for w in self.cfg("wait_for_waves", default=[]) or []:
            s = R.wave_status(self.repo, w)
            if s is None:
                self.event("warning", f"wait_for_waves: exploop/{w} has no loop_state.json; not waiting for it")
                continue
            if s != "DONE":
                pending.append(f"{w} ({s})")
        if pending:
            self.st.update(status="WAITING", waiting=f"waiting for exploop wave(s) {', '.join(pending)} to finish (base must be final)")
            return False
        b = R.resolve_base(self.repo, self.cfg("base", default={}) or {})
        if not b:
            self.block("wait_waves", "cannot resolve the base run (cycle.json base: from_wave / run_dir)",
                       "set base.run_dir (+ readouts) or base.from_wave in cycle.json, then resume")
        if self.cfg("base_module"):
            b["entry_module"] = self.cfg("base_module")
        if not b.get("entry_module") or not (self.repo / "SLS_setup" / b["entry_module"] / "train.py").is_file():
            self.block("wait_waves", f"base run {b.get('run_dir')} was not trained by a SLS_setup/<module>/train.py package",
                       "set base_module in cycle.json (e.g. b3wave), then resume")
        if not b.get("ckpt_final") or not self.abs(b["ckpt_final"]).is_file():
            self.block("wait_waves", f"no final-epoch checkpoint for the base ({b.get('run_dir')})", "check the base run, then resume")
        self.st["base"] = b
        self.st.update(status="RUNNING", waiting=None)
        self.event("base", f"base = {b['label']} ({b['run_dir']}), entry module {b['entry_module']}")
        return self.set_phase("audit")

    def ph_audit(self):
        report = self.cycle_dir / "01_audit.md"
        r = self.claude_step("audit", "a", lambda: self.prompt_audit(report), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        errs = []
        if len(p.get("gaps") or []) < 2:
            errs.append("at least 2 gaps are required")
        if not report.is_file() or report.stat().st_size < 200:
            errs.append(f"{self.rel(report)} is missing or empty")
        if errs:
            return self.step_invalid("audit:a", errs)
        R.write_json(self.cycle_dir / "audit.json", p)
        self.team_headline("research-auditor", R.headline("audit", p))
        return self.set_phase("scout")

    def ph_scout(self):
        report = self.cycle_dir / "02_sota.md"
        r = self.claude_step("scout", "a", lambda: self.prompt_scout(report), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        errs = []
        if len(p.get("ideas") or []) < max(4, int(self.cfg("levers", "min", default=2))):
            errs.append(f"at least {max(4, int(self.cfg('levers', 'min', default=2)))} idea cards are required")
        if not report.is_file() or report.stat().st_size < 200:
            errs.append(f"{self.rel(report)} is missing or empty")
        if errs:
            return self.step_invalid("scout:a", errs)
        sid = next((s["session_id"] for s in reversed(self.st.get("claude_sessions", [])) if s["step"] == "scout" and s["status"] == "ok"), None)
        v, n = R.verify_sources(p.get("ideas") or [], R.fetched_urls(sid))
        p["_verified"] = {"verified": v, "sources": n}
        R.write_json(self.cycle_dir / "scout.json", p)
        self.team_headline("research-scout", R.headline("scout", p, {"verified": v, "sources": n}))
        self.event("scout", f"{len(p['ideas'])} ideas; {v}/{n} sources fetched in-session")
        self.st.setdefault("debate", {"version": 0, "rounds": [], "ledger": {}, "lint_failures": 0})
        return self.set_phase("propose")

    def ph_propose(self):
        d = self.st.setdefault("debate", {"version": 0, "rounds": [], "ledger": {}, "lint_failures": 0})
        v = int(d.get("version", 0)) + 1
        nl = int(d.get("lint_try", 0))
        tag = f"v{v}" + (f"l{nl}" if nl else "")
        report = self.path("debate", f"proposal_v{v}.md")
        r = self.claude_step("propose", tag, lambda: self.prompt_propose(v, report, d.get("lint_errors") or []), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        if not report.is_file() or report.stat().st_size < 200:
            return self.step_invalid(f"propose:{tag}", [f"{self.rel(report)} is missing or empty"])
        p["version"] = v
        R.write_json(self.path("debate", f"proposal_v{v}.json"), p)
        d["pending"] = v
        self.team_headline("research-architect", R.headline("propose", p))
        return self.set_phase("lint")

    def ph_lint(self):
        d = self.st["debate"]
        v = d["pending"]
        prop = R.read_json(self.cycle_dir / "debate" / f"proposal_v{v}.json") or {}
        base_flags = L.trainer_flags([self.repo / "SLS_setup" / self.base()["entry_module"] / "train.py"])
        prev_open = [o["id"] for o in R.open_objections(d.get("ledger") or {}, R.SEVERITIES)] if v > 1 else []
        errs, needs = R.lint_proposal(prop, self.cycle, base_flags, prev_open, check_import=self.importable)
        R.write_json(self.path("debate", f"lint_v{v}.json"), {"ts": R.now_ts(), "errors": errs, "needs_user": needs})
        if errs:
            d["lint_failures"] = int(d.get("lint_failures", 0)) + 1
            d["lint_try"] = int(d.get("lint_try", 0)) + 1
            d["lint_errors"] = errs
            self.event("lint_failed", f"proposal v{v}: {len(errs)} problem(s): {errs[:3]}")
            if d["lint_failures"] > 3 * int(self.cfg("debate", "max_rounds", default=3)):
                self.block("lint", f"proposal LINT failed {d['lint_failures']} times; last: {errs[:4]}",
                           f"see rdloop/{self.name}/debate/lint_v{v}.json; adjust cycle.json limits or add a note, then resume")
            return self.set_phase("propose", "lint failed")
        d.update(version=v, lint_errors=[], lint_try=0)
        d.setdefault("needs_user", {})[str(v)] = needs
        if needs:
            self.event("needs_user_deps", f"proposal v{v}: missing dependencies {needs} (lever excluded unless the owner installs them)")
        self.event("lint_ok", f"proposal v{v}: {len(prop.get('levers', []))} levers pass LINT")
        return self.set_phase("critique")

    def importable(self, name):
        mod = re.split(r"[<>=\[ ]", str(name).strip())[0].replace("-", "_")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", mod):
            return False
        try:
            r = subprocess.run([self.py, "-c", f"import {mod}"], capture_output=True, timeout=120,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    def ph_critique(self):
        d = self.st["debate"]
        v = d["version"]
        report = self.path("debate", f"critique_v{v}.md")
        r = self.claude_step("critique", f"v{v}", lambda: self.prompt_critique(v, report), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        if not report.is_file() or report.stat().st_size < 100:
            return self.step_invalid(f"critique:v{v}", [f"{self.rel(report)} is missing or empty"])
        p["version"] = v
        R.write_json(self.path("debate", f"critique_v{v}.json"), p)
        prop = R.read_json(self.cycle_dir / "debate" / f"proposal_v{v}.json") or {}
        ids = [lv["id"] for lv in prop.get("levers", [])]
        ledger = d.setdefault("ledger", {})
        notes = R.apply_critique(ledger, p, v, ids)
        for n_ in notes:
            self.event("debate_note", n_)
        R.write_json(self.path("debate", "ledger.json"), ledger)
        counts = R.debate_counts(ledger)
        agreed = not R.open_objections(ledger)
        d.setdefault("rounds", []).append({"v": v, "levers": ids, "counts": counts, "agreed": agreed,
                                           "critic_verdict": p.get("verdict"), "ts": R.now_ts(),
                                           "stances": {x.get("lever_id"): x.get("stance") for x in p.get("per_lever") or []}})
        self.team_headline("research-critic", R.headline("critique", p, {"counts": counts, "agreed": agreed}))
        self.event("critique", f"v{v}: {counts['blocker']} blocker / {counts['major']} major / {counts['minor']} minor open; "
                               f"{'AGREED' if agreed else 'revise'}", counts=counts, agreed=agreed)
        self.st.pop("reopen_note", None)
        if agreed:
            d["outcome"] = "agreed"
            return self.set_phase("gate", "architect and critic agree")
        if len(d["rounds"]) - int(d.get("rounds_offset", 0)) >= int(self.cfg("debate", "max_rounds", default=3)):
            dropped, contested, plan_level = R.resolve_exhausted(ledger, ids)
            d.update(outcome="exhausted", dropped=dropped, contested=contested, plan_level=plan_level)
            for lid in dropped:
                self.drop_lever(lid, "debate", "open blocker after the last round")
            if plan_level:
                self.needs_user([f"plan-level objection(s) still open after {len(d['rounds'])} rounds: {plan_level} "
                                 f"(see rdloop/{self.name}/debate/ledger.json)"],
                                f"read debate/critique_v{v}.md; add a note (rdctl.sh note {self.name} \"...\") and "
                                f"`rdctl.sh reopen {self.name} --at propose`, or park the cycle")
            left = [i for i in ids if i not in dropped]
            if not left:
                self.needs_user(["every lever has an open blocker after the last debate round"],
                                f"read debate/; add a note and `rdctl.sh reopen {self.name} --at propose`, or park")
            self.event("debate_exhausted", f"dropped {dropped}, contested {contested}")
            return self.set_phase("gate", "debate rounds exhausted")
        return self.set_phase("propose", "revise")

    def build_plan(self):
        d = self.st["debate"]
        v = d["version"]
        prop = R.read_json(self.cycle_dir / "debate" / f"proposal_v{v}.json") or {}
        needs = (d.get("needs_user") or {}).get(str(v)) or {}
        dropped = self.st.get("dropped_levers") or {}
        contested = (d.get("contested") or []) if d.get("outcome") == "exhausted" else []
        levers = []
        for lv in prop.get("levers", []):
            if lv["id"] in dropped:
                continue
            lv = dict(lv)
            lv["contested"] = lv["id"] in contested
            lv["needs_user"] = needs.get(lv["id"], [])
            levers.append(lv)
        plan = dict(prop)
        plan["levers"] = levers
        plan["_debate"] = {"rounds": len(d.get("rounds", [])), "outcome": d.get("outcome"), "counts": R.debate_counts(d.get("ledger") or {}),
                           "dropped": {k: v_ for k, v_ in dropped.items()}}
        R.write_json(self.cycle_dir / "plan_agreed.json", plan)
        sha = R.sha256_file(self.cycle_dir / "plan_agreed.json")
        md = [f"# rdloop {self.name}: agreed plan (proposal v{v})", "",
              f"> base {self.base().get('label')} `{self.base().get('run_dir')}` · module `{self.lay['module_dir']}` · "
              f"debate: {len(d.get('rounds', []))} round(s), outcome **{d.get('outcome')}** · plan sha256 `{sha[:12]}`", "",
              "| lever | flags (default) | gap | status |", "|---|---|---|---|"]
        for lv in levers:
            st = "contested (open major)" if lv["contested"] else ("needs owner: install " + ", ".join(lv["needs_user"]) if lv["needs_user"] else "agreed")
            flags = ", ".join(f"`{f['name']}` ({f.get('default')})" for f in lv.get("flags", []))
            md.append(f"| `{lv['id']}` {lv.get('title', '')} | {flags} | {', '.join(lv.get('gap_ids', []))} | {st} |")
        if dropped:
            md += ["", "Dropped: " + "; ".join(f"`{k}` ({v_})" for k, v_ in dropped.items())]
        md += ["", f"Debate record: `rdloop/{self.name}/debate/` (proposals, critiques, ledger.json).", "",
               "---", "", (self.cycle_dir / "debate" / f"proposal_v{v}.md").read_text(encoding="utf-8")]
        R.write_text(self.cycle_dir / "plan_agreed.md", "\n".join(md) + "\n")
        self.st["plan"] = {"sha": sha, "version": v, "levers": [lv["id"] for lv in levers],
                           "contested": [lv["id"] for lv in levers if lv["contested"]],
                           "needs_user": {lv["id"]: lv["needs_user"] for lv in levers if lv["needs_user"]}}
        self.event("plan", f"plan_agreed.md written: levers {self.st['plan']['levers']} (sha {sha[:12]})")
        return plan

    def ph_gate(self):
        if not self.st.get("plan") or self.st["plan"].get("version") != self.st["debate"]["version"]:
            self.build_plan()
        pl = self.st["plan"]
        default = [i for i in pl["levers"] if i not in pl["contested"] and i not in pl["needs_user"]]
        mode = self.st.get("gate_mode", "owner")
        approved = None
        if mode == "auto":
            approved = default
            by = "auto"
        else:
            ap = R.read_json(self.cycle_dir / "approval.json")
            if ap and ap.get("plan_sha") == pl["sha"]:
                approved = ap.get("levers") or default
                by = "owner"
            else:
                self.st.update(status="AWAITING_APPROVAL", waiting=None,
                               next=f"read rdloop/{self.name}/plan_agreed.md, then `/rd-cycle approve {self.name}` "
                                    f"(rdctl.sh approve {self.name} [--levers a,b]) or send it back: "
                                    f"rdctl.sh note {self.name} \"...\" && rdctl.sh reopen {self.name} --at propose")
                if not self.st.get("gate_announced") == pl["sha"]:
                    self.st["gate_announced"] = pl["sha"]
                    self.event("awaiting_approval", f"plan ready for the owner: levers {pl['levers']}"
                                                    + (f", contested {pl['contested']}" if pl["contested"] else "")
                                                    + (f", needs install {pl['needs_user']}" if pl["needs_user"] else ""))
                return False
        bad = [i for i in approved if i not in pl["levers"]]
        missing = {i: [d for d in pl["needs_user"].get(i, []) if not self.importable(d)] for i in approved if i in pl["needs_user"]}
        missing = {k: v for k, v in missing.items() if v}
        if bad or missing:
            msg = (f"unknown levers {bad}" if bad else "") + (f" dependencies still missing {missing}" if missing else "")
            if mode == "auto":
                approved = [i for i in approved if i not in bad and i not in missing]
            else:
                self.event("approval_rejected", msg)
                try:
                    (self.cycle_dir / "approval.json").rename(self.cycle_dir / f"approval.rejected.{R.stamp()}.json")
                except OSError:
                    pass
                return False
        if not approved:
            self.needs_user(["no lever can be approved automatically (all contested or missing dependencies)"],
                            f"read plan_agreed.md; `rdctl.sh approve {self.name} --levers ...` or reopen / park")
        self.st["approved_levers"] = approved
        self.st["approval"] = {"by": by, "ts": R.now_ts(), "plan_sha": pl["sha"], "levers": approved}
        plan = self.plan()
        self.st["state_dict_change"] = any(lv.get("state_dict_change") for lv in plan["levers"] if lv["id"] in approved)
        for lid in pl["levers"]:
            if lid not in approved:
                self.drop_lever(lid, "gate", "not approved")
        self.st.update(status="RUNNING", next=None)
        self.event("approved", f"plan approved by {by}: levers {approved}")
        return self.set_phase("scaffold")

    def ph_scaffold(self):
        if self.wait_for_exploop_claude():
            return False
        mod = self.cycle["module"]
        base_mod = self.base()["entry_module"]
        mdir = self.repo / self.lay["module_dir"]
        if self.st.get("scaffold", {}).get("done"):
            if self.st.pop("impl_rework", False):
                specs = self.st.setdefault("impl", {}).get("specs", {})
                core = next((p for p in self.st["impl"].get("parts", []) if p["id"] == "core"), {"id": "core", "status": "pending"})
                parts = [core]
                for lid in self.lever_ids():
                    sh = R.sha256_text(json.dumps(self.plan_lever(lid), sort_keys=True))
                    parts.append({"id": lid, "status": "done" if specs.get(lid) == sh else "pending"})
                self.st["impl"]["parts"] = parts
                self.event("rework", f"re-approved plan: levers to (re)implement {[p['id'] for p in parts if p['status'] == 'pending']}")
            return self.set_phase("implement")
        if mdir.exists() and any(mdir.iterdir()):
            self.block("scaffold", f"{self.lay['module_dir']} already exists and was not created by this cycle",
                       "pick another module name in cycle.json (module / script_prefix) or move the directory away, then resume")
        mdir.mkdir(parents=True, exist_ok=True)
        src = (self.repo / "SLS_setup" / base_mod / "train.py").read_text(encoding="utf-8")
        ver = f"{mod.upper()}_VERSION"
        notes = []
        m = re.search(rf"^from {re.escape(base_mod)} import __version__ as (\w+)\s*$", src, re.M)
        if m:
            src = src[:m.end()] + f"\nfrom {mod} import __version__ as {ver}" + src[m.end():]
        else:
            notes.append("version import not patched")
        m = re.search(rf'^(\s*)print\(f"{re.escape(base_mod)}: \{{\w+\}}"\)\s*$', src, re.M)
        if m:
            src = src[:m.end()] + f'\n{m.group(1)}print(f"{mod}: {{{ver}}}")' + src[m.end():]
        else:
            notes.append("version print not patched")
        R.write_text(mdir / "train.py", src)
        R.write_text(mdir / "__init__.py",
                     f'"""\n{mod}: the {self.name} research cycle package (rdloop/{self.name}).\n\nAdditive: nothing in '
                     f'SLS_setup/{base_mod}/, model/, utils/ or main_*.py is changed. `python -u -m {mod}.train` with the base '
                     f'command line reproduces the base (every new flag defaults to the old behaviour); see README.md.\n"""\n\n'
                     f'__version__ = "0.1.0"\n')
        parts = [{"id": "core", "status": "pending"}] + [{"id": i, "status": "pending"} for i in self.lever_ids()]
        self.st["impl"] = {"parts": parts}
        prog = [f"# {mod} implementation progress (rdloop {self.name})", "",
                f"Scaffolded {R.now_ts()} by the supervisor: `{self.lay['module_dir']}train.py` = copy of "
                f"`SLS_setup/{base_mod}/train.py`" + (f" ({'; '.join(notes)})" if notes else " + version line") + ".", "",
                "## Parts", ""] + [f"- [ ] {p['id']}" for p in parts]
        R.write_text(self.path("impl", "progress.md"), "\n".join(prog) + "\n")
        self.st["scaffold"] = {"done": True, "ts": R.now_ts(), "notes": notes}
        self.event("scaffold", f"{self.lay['module_dir']} scaffolded from {base_mod}" + (f" ({notes})" if notes else ""))
        return self.set_phase("implement")

    def ph_implement(self):
        parts = self.st["impl"]["parts"]
        nxt = next((p for p in parts if p["status"] in ("pending", "running")), None)
        if nxt is None:
            if not self.lever_ids():
                self.needs_user(["no lever could be implemented"], f"read rdloop/{self.name}/impl/; reopen or park")
            self.st["code_sha"] = R.code_sha(self.repo, self.cycle)
            return self.set_phase("static")
        if nxt["id"] != "core" and nxt["id"] in (self.st.get("dropped_levers") or {}):
            nxt["status"] = "dropped"
            return True
        if not self.st["jobs"].get(f"claude:implement:{nxt['id']}") and self.wait_for_exploop_claude():
            return False
        usd = self.cfg("claude", "steps", "implement", "usd" if nxt["id"] == "core" else "usd_per_lever", default=20)
        r = self.claude_step("implement", nxt["id"], lambda: self.prompt_implement(nxt["id"]),
                             [self.path("impl", "progress.md")], usd=usd)
        nxt["status"] = "running"
        if r is None:
            return False
        ok, p = r
        if not ok:
            nxt["status"] = "pending"
            return True
        p["part"] = nxt["id"]
        R.write_json(self.path("impl", f"implement_{nxt['id']}.json"), p)
        self.team_headline("research-engineer", R.headline("implement", p))
        if p.get("needs_user"):
            items = [f"{x.get('what')}: {x.get('why')}" for x in p["needs_user"]]
            if nxt["id"] == "core":
                nxt["status"] = "pending"
                self.needs_user(items, f"provide what the engineer needs, then `rdctl.sh resume {self.name}`")
            self.drop_lever(nxt["id"], "implement", "needs the owner: " + "; ".join(items))
            nxt["status"] = "dropped"
            return True
        if nxt["id"] != "core":
            mine = next((x for x in p.get("levers") or [] if x.get("id") == nxt["id"]), None)
            if not mine or not mine.get("implemented"):
                self.drop_lever(nxt["id"], "implement", (mine or {}).get("deviations") or "the engineer could not implement it")
                nxt["status"] = "dropped"
                return True
        nxt["status"] = "done"
        nxt["headline"] = p.get("headline", "")
        if nxt["id"] != "core":
            self.st["impl"].setdefault("specs", {})[nxt["id"]] = R.sha256_text(json.dumps(self.plan_lever(nxt["id"]), sort_keys=True))
        self.event("implemented", f"part {nxt['id']}: {p.get('headline', '')[:200]}")
        return True

    # -- job sequences (STATIC / GPU / COMPAT) -----------------------------------------------
    def run_sequence(self, key, build, gpu=False):
        """Run a list of job specs one after another (state in st.tests[key]); returns the finished
        sequence or None while it runs. A new code_sha restarts the sequence."""
        tests = self.st.setdefault("tests", {})
        seq = tests.get(key)
        sha = self.st.get("code_sha")
        if not seq or seq.get("code_sha") != sha or seq.get("restart"):
            if gpu and not self.ready_for_gpu():
                return None
            seq = {"code_sha": sha, "jobs": build(), "idx": 0, "done": False, "started": time.time(), "started_ts": R.now_ts()}
            tests[key] = seq
            self.event(f"{key}_start", f"{key.upper()}: {len(seq['jobs'])} checks ({', '.join(j['name'] for j in seq['jobs'])})")
        if seq.get("done"):
            return seq
        while seq["idx"] < len(seq["jobs"]):
            j = seq["jobs"][seq["idx"]]
            if j["state"] == "pending":
                if j.get("budget_group") == "smoke" and self.smoke_spent(seq) >= float(self.cfg("smoke", "max_minutes", default=30)) * 60:
                    j.update(state="fail", note="skipped: the smoke budget is used up")
                    seq["idx"] += 1
                    continue
                if j.get("inproc"):
                    t0 = time.time()
                    try:
                        ok, note, extra = getattr(self, "check_" + j["check"])(seq, j)
                    except Exception as exc:       # a broken check must fail visibly, not crash the loop
                        ok, note, extra = False, f"{type(exc).__name__}: {exc}", {}
                    j.update(state="pass" if ok else "fail", note=note, elapsed_s=round(time.time() - t0, 1), **(extra or {}))
                    self.event("check", f"{key} {j['name']}: {'PASS' if ok else 'FAIL'} {note[:200]}")
                    seq["idx"] += 1
                    continue
                if j.get("before"):
                    getattr(self, "pre_" + j["before"])(seq, j)
                job = self.start_job(f"{key}:{j['name']}", j["argv"], self.abs(j.get("cwd", ".")), stdout=j["log"],
                                     env_extra=j.get("env"))
                j.update(state="running", started=time.time(), jobname=job["name"])
                self.st.setdefault("team", {})["supervisor"] = dict(self.st.get("team", {}).get("supervisor", {}), state="running",
                                                                   step=f"{key} {j['name']}", started=time.time(), finished=None)
                return None
            if j["state"] == "running":
                job = self.st["jobs"].get(j.get("jobname"))
                if job and self.job_alive(job):
                    j["elapsed_s"] = round(time.time() - job["started"], 1)
                    if gpu and self.cfg("gpu", "enabled", default=True):
                        mem = L.gpu_memory_used_mib() or []
                        if mem:
                            j["peak_mib"] = max(j.get("peak_mib", 0), max(mem))
                    limit = float(j.get("timeout_s", 1800))
                    if j.get("budget_group") == "smoke":
                        left = float(self.cfg("smoke", "max_minutes", default=30)) * 60 - self.smoke_spent(seq, exclude=j)
                        limit = min(limit, max(left, 0))
                    self.timeout_job(job, limit)
                    return None
                rc = self.job_rc(job) if job else None
                if job:
                    self.st["jobs"].pop(j["jobname"], None)
                    j["elapsed_s"] = round(time.time() - job["started"], 1)
                    j["timed_out"] = bool(job.get("timed_out"))
                j["rc"] = rc
                try:
                    ok, note, extra = getattr(self, "check_" + j["check"])(seq, j)
                except Exception as exc:
                    ok, note, extra = False, f"{type(exc).__name__}: {exc}", {}
                if j.get("timed_out"):
                    ok, note = False, f"timed out after {j['elapsed_s'] / 60:.1f} min. {note}"
                j.update(state="pass" if ok else "fail", note=note, **(extra or {}))
                self.event("check", f"{key} {j['name']}: {'PASS' if ok else 'FAIL'} {note[:200]}")
                seq["idx"] += 1
                continue
            seq["idx"] += 1
        seq["done"] = True
        seq["finished_ts"] = R.now_ts()
        seq["ok"] = all(j["state"] == "pass" or j.get("advisory") for j in seq["jobs"])
        n_fail = sum(1 for j in seq["jobs"] if j["state"] == "fail")
        t = self.st.setdefault("team", {}).setdefault("supervisor", {})
        t.update(state="done" if seq["ok"] else "failed", step=key, finished=time.time(),
                 headline=f"{key}: {len(seq['jobs']) - n_fail}/{len(seq['jobs'])} checks passed")
        R.write_json(self.path("tests", f"{key}.json"), seq)
        return seq

    def smoke_spent(self, seq, exclude=None):
        return sum(float(j.get("elapsed_s") or 0) for j in seq["jobs"] if j.get("budget_group") == "smoke" and j is not exclude)

    def job_spec(self, name, argv, check, cwd=".", log=None, **kw):
        spec = {"name": name, "argv": [str(a) for a in argv], "cwd": str(cwd), "check": check, "state": "pending",
                "log": str(log or self.rel(self.path("tests", kw.pop("group", "x"), f"{name}.log")))}
        spec.update(kw)
        return spec

    # STATIC ---------------------------------------------------------------------------------
    def build_static(self):
        mod, base_mod = self.cycle["module"], self.base()["entry_module"]
        cpu = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
        files = [self.rel(p) for p in R.code_files(self.repo, self.cycle) if p.suffix == ".py"]
        jobs = [self.job_spec("manifest", [], "manifest", inproc=True, group="static"),
                self.job_spec("py_compile", [self.py, "-m", "py_compile", *files], "rc", group="static", env=cpu, timeout_s=300),
                self.job_spec("help", [self.py, "-m", f"{mod}.train", "--help"], "help", cwd="SLS_setup", group="static",
                              env=cpu, timeout_s=300),
                self.job_spec("checks_cpu", [self.py, "-m", f"{mod}.checks", "cpu"], "rc", cwd="SLS_setup", group="static",
                              env=cpu, timeout_s=float(self.cfg("static", "minutes", default=15)) * 60)]
        if (self.repo / "SLS_setup" / base_mod / "checks.py").is_file():
            for t in self.cfg("static", "base_checks", default=["all"]):
                jobs.append(self.job_spec(f"base_checks_{t}", [self.py, "-m", f"{base_mod}.checks", t], "rc", cwd="SLS_setup",
                                          group="static", env=cpu, timeout_s=float(self.cfg("static", "minutes", default=15)) * 60,
                                          env_check=True))
        tests_dir = self.repo / "scripts" / "exploop" / "tests"
        for t in self.cfg("static", "exploop_tests", default=[]):
            if (tests_dir / t).is_file():
                jobs.append(self.job_spec(f"exploop_{Path(t).stem}", [self.py, "-m", "unittest", "discover", "-s", str(tests_dir), "-p", t],
                                          "rc", group="static", env=cpu, timeout_s=600, env_check=True))
        return jobs

    def check_manifest(self, seq, j):
        man = R.read_json(self.cycle_dir / "isolation" / "manifest.json")
        if not man:
            return False, "no isolation manifest (run init)", {}
        v = R.check_manifest(self.repo, man, self.cycle, self.rel(self.cycle_dir))
        if v:
            R.write_json(self.cycle_dir / "isolation" / "violations.json", {"ts": R.now_ts(), "where": "static", "violations": v})
            return False, f"{len(v)} file(s) outside the scope changed: {v[:4]}", {"violations": v[:50]}
        return True, f"{len(man['files'])} protected files and {len(man['runs'])} run files unchanged", {}

    def check_rc(self, seq, j):
        tail = R.tail_bytes(self.abs(j["log"]), 1500).strip().splitlines()[-3:]
        return j.get("rc") == 0, f"rc {j.get('rc')}" + (f": {' | '.join(tail)[:300]}" if j.get("rc") else ""), {}

    def check_help(self, seq, j):
        text = R.tail_bytes(self.abs(j["log"]), 400000)
        flags = [f["name"] for lid in self.lever_ids() for f in (self.plan_lever(lid) or {}).get("flags", [])]
        missing = [f for f in flags if f not in text]
        if j.get("rc") != 0:
            return False, f"--help exited {j.get('rc')}", {}
        return not missing, ("all lever flags present" if not missing else f"flags missing from --help: {missing}"), {}

    def ph_static(self):
        self.st["code_sha"] = R.code_sha(self.repo, self.cycle)
        if (self.st.get("passes") or {}).get("static") == self.st["code_sha"]:
            return self.set_phase("review")
        seq = self.run_sequence("static", self.build_static)
        if seq is None:
            return False
        if seq["ok"]:
            self.st.setdefault("passes", {})["static"] = seq["code_sha"]
            return self.set_phase("review")
        if any(j["name"] == "manifest" and j["state"] == "fail" for j in seq["jobs"]):
            self.check_isolation("static")
        self.block_on_env(seq, "static")
        return self.goto_fix("static", self.failure_material(seq))

    def block_on_env(self, seq, phase):
        """Failures of the OLD code's own checks are not the engineer's to fix: the manifest proves the old
        files are unchanged, so it is the environment (ffmpeg wiped by a restart, data mount, GPU)."""
        env = [j for j in seq["jobs"] if j["state"] == "fail" and j.get("env_check")]
        if env:
            seq["restart"] = True
            self.block(phase, f"checks of the unchanged base code failed: {[j['name'] + ': ' + j.get('note', '')[:160] for j in env]}",
                       f"the module cannot cause this (isolation manifest ok): check the environment (which ffmpeg; "
                       f"/mnt mounts; nvidia-smi) and the logs under rdloop/{self.name}/tests/, then resume")

    def failure_material(self, seq):
        out = []
        for j in seq["jobs"]:
            if j["state"] == "fail" and not j.get("advisory"):
                tail = R.tail_bytes(self.abs(j["log"]), 3000).strip() if j.get("log") and self.abs(j["log"]).is_file() else ""
                out.append(f"### {j['name']}: {j.get('note', '')}\ncommand: `{' '.join(j.get('argv', []))[:400]}` (cwd {j.get('cwd')})\n"
                           f"log: `{j.get('log')}`\n```\n{tail[-2500:]}\n```")
        return "\n\n".join(out)

    def goto_fix(self, reason, material):
        f = self.st.setdefault("fix", {"n": 0})
        f.update(reason=reason, material=material[:20000])
        self.st["passes"] = {}
        return self.set_phase("fix", f"{reason} failed")

    # REVIEW ---------------------------------------------------------------------------------
    def ph_review(self):
        rv = self.st.setdefault("review", {"n": 0, "findings": {}})
        sha = self.st.get("code_sha")
        if (self.st.get("passes") or {}).get("review") == sha:
            return self.set_phase("gpu")
        n = rv["n"] + 1
        report = self.path("review", f"code_review_r{n}.md")
        diff = self.path("review", f"diff_r{n}.patch")
        if not self.st["jobs"].get(f"claude:review:r{n}") and not diff.is_file():
            self.make_review_diff(n, diff)
        prev = [k for k, f in rv["findings"].items() if f["status"] == "open"]
        r = self.claude_step("review", f"r{n}", lambda: self.prompt_review(n, report, diff, prev), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        p["round"] = n
        R.write_json(self.path("review", f"code_review_r{n}.json"), p)
        rv["n"] = n
        for fid in p.get("resolved") or []:
            if fid in rv["findings"]:
                rv["findings"][fid]["status"] = "resolved"
        for f in p.get("findings") or []:
            fid = f.get("id") or f"F{n}.{len(rv['findings']) + 1}"
            rv["findings"][fid] = dict(f, id=fid, status="open", round=n)
        blocking = [f for f in rv["findings"].values() if f["status"] == "open" and f.get("severity") in R.BLOCKING
                    and (not f.get("lever_id") or f["lever_id"] not in (self.st.get("dropped_levers") or {}))]
        self.team_headline("research-critic", f"review r{n}: {len(blocking)} blocking findings open")
        self.event("review", f"round {n}: {len(blocking)} blocking finding(s) open; critic verdict {p.get('verdict')}")
        if not blocking:
            rv["streak"] = 0
            self.st.setdefault("passes", {})["review"] = sha
            return self.set_phase("gpu", "review passed")
        rv["streak"] = int(rv.get("streak", 0)) + 1
        if rv["streak"] >= int(self.cfg("fix", "max_review_rounds", default=3)):
            core = [f for f in blocking if not f.get("lever_id")]
            if core:
                self.needs_user([f"blocking review findings on shared code after {n} rounds: {[f['id'] for f in core]}"],
                                f"read rdloop/{self.name}/review/; fix by hand or add a note, then "
                                f"`rdctl.sh reopen {self.name} --at implement`, or park")
            for f in blocking:
                self.drop_lever(f["lever_id"], "review", f"{f['id']}: {f.get('issue', '')[:200]}")
            if not self.lever_ids():
                self.needs_user(["every lever has blocking review findings"], f"read review/; reopen or park")
            self.st.setdefault("passes", {})["review"] = sha
            return self.set_phase("gpu", "review rounds exhausted; levers with blocking findings dropped")
        mat = "Open review findings (blocker / major) to fix:\n" + "\n".join(
            f"- **{f['id']}** [{f['severity']}, {f.get('category')}] `{f.get('file')}:{f.get('line')}` lever `{f.get('lever_id') or 'core'}`: "
            f"{f.get('issue', '')[:400]} -> fix: {f.get('fix', '')[:300]}" for f in blocking)
        mat += f"\n\nFull review: `rdloop/{self.name}/review/code_review_r{n}.md`."
        return self.goto_fix("review", mat)

    def make_review_diff(self, n, out):
        base_mod = self.base()["entry_module"]
        snap = self.cycle_dir / "review" / f"snap_r{n}"
        if snap.exists():
            shutil.rmtree(snap)
        for p in R.code_files(self.repo, self.cycle):
            dst = snap / self.rel(p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        if n == 1:
            a = self.repo / "SLS_setup" / base_mod / "train.py"
            b = self.repo / self.lay["module_dir"] / "train.py"
            d1 = subprocess.run(["git", "diff", "--no-index", "--", str(a), str(b)], capture_output=True, text=True).stdout
            others = [p for p in R.code_files(self.repo, self.cycle) if p != b]
            d2 = ""
            for p in others:
                d2 += subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", str(p)], capture_output=True, text=True).stdout
            text = f"# train.py vs SLS_setup/{base_mod}/train.py\n{d1}\n# new files\n{d2}"
        else:
            prev = self.cycle_dir / "review" / f"snap_r{n - 1}"
            text = subprocess.run(["git", "diff", "--no-index", "--", str(prev), str(snap)], capture_output=True, text=True).stdout
            text = f"# changes since review round {n - 1}\n{text}"
        R.write_text(out, text)

    # FIX ------------------------------------------------------------------------------------
    def ph_fix(self):
        f = self.st.setdefault("fix", {"n": 0})
        n = f["n"] + 1
        if n > int(self.cfg("fix", "max_total", default=6)) + int(f.get("max_bonus", 0)):
            self.needs_user([f"{n - 1} fix rounds used; last failure: {f.get('reason')}"],
                            f"read rdloop/{self.name}/tests/ and review/; fix by hand or add a note and "
                            f"`rdctl.sh reopen {self.name} --at implement`, or park")
        if not self.st["jobs"].get(f"claude:fix:f{n}") and self.wait_for_exploop_claude():
            return False
        r = self.claude_step("fix", f"f{n}", lambda: self.prompt_fix(n, f.get("reason"), f.get("material", "")),
                             [self.path("impl", "progress.md")])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        p["part"] = "fix"
        R.write_json(self.path("impl", f"fix_f{n}.json"), p)
        f["n"] = n
        self.team_headline("research-engineer", f"fix f{n} ({f.get('reason')}): {len(p.get('addressed') or [])} addressed")
        if p.get("needs_user"):
            self.needs_user([f"{x.get('what')}: {x.get('why')}" for x in p["needs_user"]],
                            f"provide it, then `rdctl.sh resume {self.name}`")
        self.st["code_sha"] = R.code_sha(self.repo, self.cycle)
        self.event("fixed", f"fix f{n} ({f.get('reason')}): {p.get('headline', '')[:200]}")
        return self.set_phase("static")

    # GPU session ----------------------------------------------------------------------------
    def smoke_paths(self):
        tests = self.path("tests", "smoke", "x").parent
        return tests, self.cycle_dir / "runs"

    def write_dev_small(self):
        b = self.base()
        dev = L.argv_dict(b["argv"]).get("--dev_protocol", [None])[0]
        src = self.abs(Path("SLS_setup") / dev) if dev and not os.path.isabs(dev) else Path(str(dev))
        out = self.path("tests", "smoke", "dev_small.txt")
        if src and src.is_file():
            lines = src.read_text(encoding="utf-8").splitlines()[: int(self.cfg("smoke", "dev_lines", default=400))]
            R.write_text(out, "\n".join(lines) + "\n")
        return out

    def build_gpu(self):
        b = self.base()
        mod, base_mod = self.cycle["module"], b["entry_module"]
        self.write_dev_small()
        ck = str(self.abs(b["ckpt_final"]))
        regdir = self.path("tests", "regression", "x").parent
        tag = (self.st.get("code_sha") or "x")[:8]
        names = self.regression_names()
        jobs = []
        jobs.append(self.job_spec("R1_old_scores", [self.py, "scripts/local_eval.py", "--model_path", ck, *b["model_flags"],
                                                    "--names", *names, "--out", str(regdir / f"old_path_{tag}"), "--force"],
                                  "old_scores", group="regression", timeout_s=float(self.cfg("regression", "minutes", default=15)) * 60,
                                  out=str(regdir / f"old_path_{tag}"), kind="regression", env_check=True))
        if (self.repo / "SLS_setup" / base_mod / "checks.py").is_file():
            for i, t in enumerate(self.cfg("regression", "base_checks", default=[]) or []):
                argv = [x.replace("{base_ckpt}", ck) for x in t]
                jobs.append(self.job_spec(f"R2_{base_mod}_checks_{argv[0]}", [self.py, "-m", f"{base_mod}.checks", *argv], "rc",
                                          cwd="SLS_setup", group="regression", timeout_s=900, kind="regression", env_check=True))
        jobs.append(self.job_spec("R3_module_gpu_checks", [self.py, "-m", f"{mod}.checks", "gpu", "--ckpt", ck], "rc",
                                  cwd="SLS_setup", group="regression", timeout_s=900, kind="regression"))
        s0 = float(self.cfg("smoke", "s0_minutes", default=4.5))
        factor = float(self.cfg("smoke", "job_factor", default=1.5))
        runs = self.cycle_dir / "runs"
        if self.cfg("regression", "equiv_micro", default=True):
            noaug = self.cfg("regression", "equiv_remove_flags", default=EQUIV_NOAUG_FLAGS)
            for name, m, tr in (("R4_base_micro", base_mod, f"{self.name}smk_eqbase_s1234"),
                                ("R4_levers_off_micro", mod, f"{self.name}smk_eqoff_s1234")):
                argv = R.set_flags(R.smoke_argv(b["argv"], self.cycle, self.cycle_dir, tr), {}, removes=noaug)
                jobs.append(self.job_spec(name, [self.py, "-u", "-m", f"{m}.train", *argv], "train", cwd="SLS_setup",
                                          group="regression", timeout_s=s0 * factor * 60, track=tr, expect=[f"^{m}: "],
                                          module=m, kind="regression", train_argv=argv))
        tr = f"{self.name}smk_s0_s1234"
        argv = R.smoke_argv(b["argv"], self.cycle, self.cycle_dir, tr)
        jobs.append(self.job_spec("S0_levers_off", [self.py, "-u", "-m", f"{mod}.train", *argv], "train", cwd="SLS_setup",
                                  group="smoke", timeout_s=s0 * factor * 60, track=tr, expect=[f"^{mod}: "], module=mod,
                                  budget_group="smoke", kind="smoke", train_argv=argv))
        if self.cfg("regression", "equiv_micro", default=True):
            jobs.append(self.job_spec("R4_equivalence", [], "equivalence", inproc=True, group="regression", kind="regression"))
        sd_lever = None
        for lid in self.lever_ids():
            lv = self.plan_lever(lid) or {}
            tr = f"{self.name}smk_{lid.replace('_', '')}_s1234"
            sets = (lv.get("wave_lever") or {}).get("set") or {}
            argv = R.smoke_argv(b["argv"], self.cycle, self.cycle_dir, tr, extra_sets=sets)
            est = float((lv.get("smoke") or {}).get("minutes_est", s0))
            jobs.append(self.job_spec(f"S_{lid}", [self.py, "-u", "-m", f"{mod}.train", *argv], "train", cwd="SLS_setup",
                                      group="smoke", timeout_s=est * factor * 60, track=tr, lever=lid, module=mod,
                                      expect=[f"^{mod}: "] + list(lv.get("expect_regex") or []), budget_group="smoke",
                                      kind="smoke", train_argv=argv))
            if lv.get("params_added"):
                jobs.append(self.job_spec(f"O_{lid}", [self.py, "-m", f"{mod}.checks", "overfit", "--lever", lid], "rc",
                                          cwd="SLS_setup", group="smoke", lever=lid, budget_group="smoke", kind="smoke",
                                          timeout_s=float(self.cfg("smoke", "overfit_minutes", default=2)) * factor * 60))
            if lv.get("state_dict_change"):
                sd_lever = lid
        ev_lever = sd_lever or (self.lever_ids()[0] if self.lever_ids() else None)
        if ev_lever:
            out = self.path("tests", "smoke", "evalpath", "x").parent
            mlist = self.path("tests", "smoke", "evalpath_merge_list.txt")
            if sd_lever:
                script = f"{self.lay['scripts_prefix']}local_eval.py"
                argv = [self.py, script, "--merge_list", str(mlist), "--run_config", "@RUN@/config.yaml", "--names", "probe_orig",
                        "--out", str(out), "--force"]
            else:
                argv = [self.py, "scripts/local_eval.py", "--merge_list", str(mlist), *b["model_flags"], "--names", "probe_orig",
                        "--out", str(out), "--force"]
            jobs.append(self.job_spec("E_eval_merge_path", argv, "evalpath", group="smoke", lever=ev_lever, before="evalpath",
                                      budget_group="smoke", kind="smoke", out=str(out), merge_list=str(mlist),
                                      timeout_s=float(self.cfg("smoke", "evalpath_minutes", default=2)) * factor * 60))
        for jj in jobs:
            jj.setdefault("gpu", True)
        return jobs

    def pre_evalpath(self, seq, j):
        """Merge list = the lever's smoke checkpoint twice (exercises merge + load + score)."""
        src = next((x for x in seq["jobs"] if x.get("lever") == j["lever"] and x["check"] == "train"), None)
        run = Path(src["run_dir"]) if src and src.get("run_dir") else None
        ck = sorted((run / "ckpt").glob("epoch_1_*.pth")) if run else []
        R.write_text(j["merge_list"], (f"{ck[0]}\n{ck[0]}\n" if ck else "# no smoke checkpoint\n"))
        j["argv"] = [a.replace("@RUN@", str(run)) for a in j["argv"]] if run else j["argv"]

    def regression_names(self):
        """R1's sets: `regression.names`, or with `full` every set of the base's final readout. Always passed
        as --names: local_eval's default list includes the half-A-only *_mini sets and crashes on half_B."""
        if self.cfg("regression", "full", default=False):
            return sorted(p.stem for p in (self.abs(self.base()["readouts"]["final"]) / "scores").glob("*.logits"))
        return self.cfg("regression", "names", default=REGRESSION_NAMES)

    def check_old_scores(self, seq, j):
        b = self.base()
        old = self.abs(b["readouts"]["final"])
        names = self.regression_names()
        if j.get("rc") != 0:
            return False, f"local_eval exited {j.get('rc')}", {}
        cs = R.compare_scores(old / "scores", Path(j["out"]) / "scores", names, float(self.cfg("regression", "tol_logit", default=1e-4)))
        cm = R.compare_metrics(old / "metrics.json", Path(j["out"]) / "metrics.json", names,
                               float(self.cfg("regression", "tol_metric", default=1e-6)))
        R.write_json(self.path("tests", "regression", "old_scores_compare.json"), {"scores": cs, "metrics": cm})
        worst = max((s.get("max_abs_logit_diff", 0) for s in cs["sets"].values() if isinstance(s.get("max_abs_logit_diff"), float)), default=None)
        ok = cs["ok"] and cm["ok"]
        note = (f"{len(names)} sets re-scored from {b['label']}'s checkpoint match the stored scores (max |Δlogit| {worst:.2e})"
                if ok else f"MISMATCH vs the stored {b['label']} scores: " +
                "; ".join(f"{n}: {s}" for n, s in cs["sets"].items() if not s.get("ok"))[:400])
        return ok, note, {"max_abs_logit_diff": worst}

    def check_train(self, seq, j):
        log = self.abs(j["log"])
        text = R.tail_bytes(log, 2_000_000)
        notes, ok = [], True
        if j.get("rc") != 0:
            ok = False
            notes.append(f"rc {j.get('rc')}")
        if "Experiment saved to:" not in text:
            ok = False
            notes.append("no 'Experiment saved to:'")
        for rx in TRAIN_FORBID:
            if re.search(rx, text):
                ok = False
                notes.append(f"forbidden /{rx}/ in the log")
        missing = [rx for rx in j.get("expect", []) if not re.search(rx, text, re.M)]
        if missing:
            ok = False
            notes.append(f"expected lines missing: {missing}")
        run = R.find_run(self.cycle_dir / "runs", j["track"])
        extra = {"run_dir": str(run) if run else None}
        if not run:
            return False, "; ".join(notes + ["run dir not found"]), extra
        rows = L.parse_train_log(run / "logs" / "train.log")
        r1 = rows.get(1) or {}
        tl, dl = r1.get("train_loss"), r1.get("dev_loss")
        if tl is None or dl is None or not all(isinstance(x, float) and x == x and abs(x) != float("inf") for x in (tl, dl)):
            ok = False
            notes.append("no finite epoch-1 TrainLoss / DevLoss in logs/train.log")
        prog = L.parse_tqdm(text)
        rate = (prog or {}).get("rate_per_s")
        peak = round(j.get("peak_mib", 0) / 1024, 2)
        extra.update(train_loss=tl, dev_loss=dl, wf1=r1.get("wf1"), rate=rate, peak_gib=peak)
        if peak > float(self.cfg("smoke", "max_peak_vram_gib", default=22)):
            ok = False
            notes.append(f"peak VRAM {peak} GiB > {self.cfg('smoke', 'max_peak_vram_gib', default=22)}")
        if j["name"] == "S0_levers_off":
            seq["s0_rate"] = rate
        if j.get("lever") and rate and seq.get("s0_rate"):
            frac = seq["s0_rate"] / rate
            extra["time_frac"] = round(frac, 3)
            lim = float(self.cfg("limits", "max_epoch_time_frac", default=1.3)) * 1.15
            if frac > lim:
                ok = False
                notes.append(f"{frac:.2f}x slower than S0 (limit {lim:.2f}x)")
            bh = R.run_hours(self.abs(self.base()["run_dir"]))
            if bh:
                eta = bh * frac
                extra["eta_full_h"] = round(eta, 2)
                if eta > float(self.cfg("limits", "max_run_hours", default=10)):
                    ok = False
                    notes.append(f"full-run ETA {eta:.1f} h > {self.cfg('limits', 'max_run_hours', default=10)} h")
        cfg_argv = L.config_argv(run)
        if cfg_argv and j.get("train_argv") and cfg_argv != j["train_argv"]:
            notes.append("config.yaml command differs from the launched argv (exploop run-dir discovery needs them equal)")
            ok = False
        if ok:
            ls = run / "last_state.pt"
            if ls.is_file():
                ls.unlink()                         # 4.8 GB; our own smoke run, never needed again
            notes.append(f"epoch 1 TrainLoss {tl:.4f} DevLoss {dl:.4f}" + (f" LocalWF1 {r1.get('wf1')}" if r1.get("wf1") is not None else "")
                         + (f", {rate:.2f} batch/s" if rate else "") + f", peak {peak} GiB")
        return ok, "; ".join(notes), extra

    def check_equivalence(self, seq, j):
        a = next((x for x in seq["jobs"] if x["name"] == "R4_base_micro"), {})
        b = next((x for x in seq["jobs"] if x["name"] == "R4_levers_off_micro"), {})
        if a.get("state") != "pass" or b.get("state") != "pass":
            return False, "needs both equivalence micro-runs (base, levers-off module) to pass", {}
        tol = float(self.cfg("regression", "equiv_tol_rel_loss", default=0.05))
        diffs = {}
        for k in ("train_loss", "dev_loss"):
            x, y = a.get(k), b.get(k)
            diffs[k] = abs(x - y) / max(abs(x), 1e-9) if x is not None and y is not None else None
        ok = all(v is not None and v <= tol for v in diffs.values())
        note = ", ".join(f"{k} rel diff {v:.4f}" if v is not None else f"{k} missing" for k, v in diffs.items())
        return ok, (f"levers-off {self.cycle['module']} reproduces {self.base()['entry_module']} within {tol:.0%} (augmenter off in both): {note}"
                    if ok else f"levers-off module differs from the base trainer beyond {tol:.0%} (augmenter off in both): {note}"), \
            {"rel_diff": diffs}

    def check_evalpath(self, seq, j):
        if j.get("rc") != 0:
            return False, f"eval exited {j.get('rc')}", {}
        m = R.read_json(Path(j["out"]) / "metrics.json")
        s = Path(j["out"]) / "scores" / "probe_orig.logits"
        if not m or not s.is_file():
            return False, "metrics.json / scores missing", {}
        return True, f"merge + eval of the {j['lever']} smoke checkpoint works ({len(R.read_score_file(s))} clips scored)", {}

    def preflight(self):
        """The smoke runs need ffmpeg with libopus (a container restart wipes it and the codec stage then
        silently passes audio through) and pyloudnorm."""
        exe = os.environ.get("RDLOOP_FFMPEG", "ffmpeg")
        ok = False
        if shutil.which(exe):
            try:
                r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60)
                ok = r.returncode == 0 and "libopus" in r.stdout
            except (OSError, subprocess.TimeoutExpired):
                ok = False
        if not ok and self.cfg("gpu", "enabled", default=True):
            self.block("gpu", "ffmpeg with libopus is missing (container restart?)",
                       "apt-get install -y ffmpeg (4.2 has libopus), then resume")
        r = subprocess.run([self.py, "-c", "import pyloudnorm"], capture_output=True, text=True)
        if r.returncode != 0 and self.cfg("gpu", "enabled", default=True):
            self.block("gpu", "pyloudnorm is not importable by the run python", f"{self.py} -m pip install pyloudnorm, then resume")

    def ph_gpu(self):
        sha = self.st.get("code_sha")
        p = self.st.get("passes") or {}
        if p.get("static") != sha:
            return self.set_phase("static", "code changed")
        if p.get("review") != sha:
            return self.set_phase("review", "code changed")
        if p.get("gpu") == sha:
            return self.set_phase("compat")
        cur = (self.st.get("tests") or {}).get("gpu")
        if not cur or cur.get("code_sha") != sha or cur.get("restart"):
            self.preflight()
        seq = self.run_sequence("gpu", self.build_gpu, gpu=True)
        if seq is None:
            return False
        self.block_on_env(seq, "gpu")
        if seq["ok"]:
            self.st.setdefault("passes", {})["gpu"] = sha
            self.team_headline("supervisor", f"regression + smoke passed ({len(seq['jobs'])} checks)")
            return self.set_phase("compat", "regression and smoke passed")
        fails = [j for j in seq["jobs"] if j["state"] == "fail"]
        f = self.st.setdefault("fix", {"n": 0})
        lever_fails = {j["lever"] for j in fails if j.get("lever")}
        core_fails = [j for j in fails if not j.get("lever")]
        if f.get("n", 0) >= int(self.cfg("fix", "max_total", default=6)) - 1 and lever_fails and not core_fails:
            for lid in lever_fails:
                self.drop_lever(lid, "smoke", next(j["note"] for j in fails if j.get("lever") == lid)[:300])
            seq["restart"] = True
            if self.lever_ids():
                return self.set_phase("gpu", "failing levers dropped; rerun")
        self.team_headline("supervisor", f"GPU session: {len(fails)} failing check(s)")
        return self.goto_fix("gpu", "Regression / smoke failures (the supervisor ran these on the GPU):\n\n" + self.failure_material(seq))

    # COMPAT ---------------------------------------------------------------------------------
    def build_compat(self):
        hd = self.cycle_dir / "handoff" / "dryrun_wave"
        sup = self.repo / "scripts" / "exploop" / "supervisor.py"
        return [self.job_spec("seed", [], "seed", inproc=True, group="compat"),
                self.job_spec("validate_levers", [], "validate", inproc=True, group="compat"),
                self.job_spec("smoke_run_contract", [], "run_contract", inproc=True, group="compat"),
                self.job_spec("exploop_init", [self.py, str(sup), "init", "--wave", str(hd), "--force"], "rc", group="compat", timeout_s=300),
                self.job_spec("exploop_plan", [self.py, str(sup), "plan", "--wave", str(hd)], "rc", group="compat", timeout_s=300)]

    def check_seed(self, seq, j):
        plan = self.plan() or {}
        seed = R.build_seed(self.cycle, dict(self.st, approved_levers=self.lever_ids()), plan, self.repo)
        R.write_json(self.path("handoff", "wave_seed.json"), seed)
        wave_name = (self.base().get("from_wave") or (R.latest_wave(self.repo) or {}).get("wave"))
        spec = R.read_json(self.repo / "exploop" / str(wave_name) / "wave.json") if wave_name else None
        if not spec:
            return False, f"no exploop wave.json to merge the seed into (base wave {wave_name})", {}
        hd = self.cycle_dir / "handoff" / "dryrun_wave"
        if hd.exists():
            shutil.rmtree(hd)
        hd.mkdir(parents=True)
        w = R.merge_seed(spec, seed, f"{self.name}dry", str(hd / "reports"))
        w["registry"] = str(hd / "registry.csv")
        if self.abs(spec.get("registry", "docs/experiment_log.csv")).is_file():
            shutil.copy(self.abs(spec.get("registry", "docs/experiment_log.csv")), hd / "registry.csv")
        w["start"] = {"best": dict(seed["start"]["best"])}
        R.write_json(hd / "wave.json", w)
        R.write_text(hd / "plan_snapshot.md", f"# dry run of the {self.name} handoff (never started)\n")
        return True, f"handoff/wave_seed.json: {len(seed['lever_queue'])} levers; dry-run wave {self.rel(hd)}", {}

    def check_validate(self, seq, j):
        w = R.read_json(self.cycle_dir / "handoff" / "dryrun_wave" / "wave.json")
        if not w:
            return False, "no dry-run wave.json", {}
        base_argv = L.config_argv(self.abs(self.base()["run_dir"])) or self.base()["argv"]
        known = L.trainer_flags([self.repo / self.lay["module_dir"] / "train.py"])
        ss = w["search_space"]
        letters = w.get("letters", "abcdefghij")
        errs = []
        for i, lev in enumerate(w["lever_queue"]):
            letter = letters[i % len(letters)]
            ne = (L.argv_dict(base_argv).get("--num_epochs") or ["8"])[0]
            track = f"{w['track_prefix']}{letter}_{lev['id'].replace('_', '')}_s{ss.get('seed', 1234)}"
            cand = L.apply_change(base_argv, {"set": lev["set"], "remove": []},
                                  {"--track": [track], "--keep_epochs": [ne], "--keep_topk_by_wf1": [str(ss.get("keep_topk_by_wf1", 7))],
                                   "--seed": [str(ss.get("seed", 1234))]})
            base = L.apply_change(base_argv, {"set": {}, "remove": []}, {"--track": [track], "--keep_epochs": [ne],
                                  "--keep_topk_by_wf1": [str(ss.get("keep_topk_by_wf1", 7))], "--seed": [str(ss.get("seed", 1234))]})
            e = L.validate_design(w, base, cand, {"kind": "config", "letter": letter, "track": track}, set(), known)
            errs += [f"{lev['id']}: {x}" for x in e]
        return (not errs), ("every seed lever passes exploop's one-factor validator" if not errs else "; ".join(errs)[:600]), {}

    def check_run_contract(self, seq, j):
        g = (self.st.get("tests") or {}).get("gpu") or {}
        errs, n = [], 0
        for x in g.get("jobs", []):
            if x.get("check") != "train" or x.get("module") != self.cycle["module"] or not x.get("run_dir"):
                continue
            run = Path(x["run_dir"])
            n += 1
            if not L.metrics_epochs(run):
                errs.append(f"{run.name}: no metrics.jsonl rows")
            if not list((run / "ckpt").glob("epoch_1_*.pth")):
                errs.append(f"{run.name}: no ckpt/epoch_1_*.pth")
            if not re.match(rf"^{re.escape(x['track'])}_epoch\d+_bs\d+_\d+$", run.name):
                errs.append(f"{run.name}: run dir name does not follow <track>_epoch<N>_bs<B>_<ts>")
        return (n > 0 and not errs), (f"{n} smoke runs follow the exploop run contract" if n and not errs else "; ".join(errs) or "no smoke runs"), {}

    def ph_compat(self):
        sha = self.st.get("code_sha")
        if (self.st.get("passes") or {}).get("compat") == sha:
            return self.set_phase("summary")
        seq = self.run_sequence("compat", self.build_compat)
        if seq is None:
            return False
        if seq["ok"]:
            self.st.setdefault("passes", {})["compat"] = sha
            return self.set_phase("summary", "exploop dry run passed")
        return self.goto_fix("compat", "The exploop handoff dry run failed (the seed is built from the approved plan and your "
                                       "module):\n\n" + self.failure_material(seq))

    # SUMMARY --------------------------------------------------------------------------------
    def facts(self):
        g = (self.st.get("tests") or {}).get("gpu") or {}
        levers = []
        for lid in self.lever_ids():
            lv = self.plan_lever(lid) or {}
            sm = next((x for x in g.get("jobs", []) if x.get("lever") == lid and x["check"] == "train"), {})
            levers.append({"id": lid, "title": lv.get("title"), "flags": [f["name"] + "=" + str(f.get("default")) for f in lv.get("flags", [])],
                           "wave_set": (lv.get("wave_lever") or {}).get("set"), "gap_ids": lv.get("gap_ids"),
                           "sources": lv.get("sources"), "smoke": {k: sm.get(k) for k in ("state", "note", "peak_gib", "rate", "time_frac",
                                                                                           "eta_full_h", "train_loss", "dev_loss", "wf1", "elapsed_s")}})
        latest = R.resolve_base(self.repo, self.cfg("base", default={}) or {}) or {}
        return {"cycle": self.name, "module": self.cycle["module"], "base": {k: self.base().get(k) for k in ("label", "run_dir", "from_wave")},
                "base_now": latest.get("label"), "base_outdated": bool(latest and latest.get("run_dir") != self.base().get("run_dir")),
                "approved_levers": self.st.get("approved_levers"), "implemented_levers": levers,
                "dropped_levers": self.st.get("dropped_levers"), "debate": {k: (self.st.get("debate") or {}).get(k) for k in ("rounds", "outcome")},
                "review_rounds": (self.st.get("review") or {}).get("n"), "fix_rounds": (self.st.get("fix") or {}).get("n"),
                "tests": {k: [{x: j.get(x) for x in ("name", "state", "note", "elapsed_s", "peak_gib", "rate", "time_frac", "eta_full_h")}
                              for j in ((self.st.get("tests") or {}).get(k) or {}).get("jobs", [])] for k in ("static", "gpu", "compat")},
                "code_sha": self.st.get("code_sha"), "cost_usd": self.st.get("cost_usd"),
                "base_run_hours": R.run_hours(self.abs(self.base()["run_dir"])), "wave_seed": f"rdloop/{self.name}/handoff/wave_seed.json"}

    def ph_summary(self):
        report = self.cycle_dir / "FINAL_SUMMARY.md"
        key = "summary:s"
        if not self.st["jobs"].get(f"claude:{key}") and not self.st.get("summary_isolation_ok") == self.st.get("code_sha"):
            self.check_isolation("summary")
            self.st["summary_isolation_ok"] = self.st.get("code_sha")
            R.write_json(self.path("tests", "facts.json"), self.facts())
        facts = self.cycle_dir / "tests" / "facts.json"
        r = self.claude_step("summary", "s", lambda: self.prompt_summary(report, facts), [report])
        if r is None:
            return False
        ok, p = r
        if not ok:
            return True
        if not report.is_file() or report.stat().st_size < 300:
            return self.step_invalid(key, [f"{self.rel(report)} is missing or too short"])
        R.write_json(self.cycle_dir / "summary.json", p)
        self.team_headline("research-auditor", R.headline("summary", p))
        self.publish_docs(p)
        self.st.update(status="AWAITING_USER", waiting=None, phase="awaiting_user",
                       next=f"`/rd-cycle review {self.name}` in `claude --agent research-lead`: start the wave "
                            f"(/exp-loop new <W> --from-research {self.name}), request changes, or park")
        self.event("awaiting_user", f"cycle {self.name} ready for the owner: {p.get('headline', '')[:200]}")
        self.save()
        raise Stop("AWAITING_USER")

    def publish_docs(self, summary):
        """docs/research/<C>/ copies, the lever ledger rows and the literature list (written by the
        supervisor, never by an agent)."""
        ddir = self.abs(self.cfg("docs_dir", default="docs/research"))
        out = ddir / self.name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("01_audit.md", "02_sota.md", "plan_agreed.md", "FINAL_SUMMARY.md"):
            if (self.cycle_dir / f).is_file():
                shutil.copy(self.cycle_dir / f, out / f)
        if (self.cycle_dir / "debate").is_dir():
            dd = out / "debate"
            dd.mkdir(exist_ok=True)
            for p in (self.cycle_dir / "debate").glob("*.md"):
                shutil.copy(p, dd / p.name)
        ledger = ddir / "lever_ledger.md"
        if not ledger.is_file():
            R.write_text(ledger, "# Lever ledger\n\n| date | cycle | lever | flags | gaps | status | smoke | wave result | sources |\n"
                                 "|---|---|---|---|---|---|---|---|---|\n")
        rows = []
        g = (self.st.get("tests") or {}).get("gpu") or {}
        for lid in self.lever_ids():
            lv = self.plan_lever(lid) or {}
            sm = next((x for x in g.get("jobs", []) if x.get("lever") == lid and x["check"] == "train"), {})
            rows.append(f"| {R.now_ts()[:10]} | {self.name} | `{lid}` | {' '.join(f['name'] for f in lv.get('flags', []))} | "
                        f"{', '.join(lv.get('gap_ids', []))} | implemented ({self.cycle['module']}) | {sm.get('state', '?')} "
                        f"{sm.get('time_frac') or ''} | pending | {' '.join(lv.get('sources', [])[:3])} |")
        for lid, why in (self.st.get("dropped_levers") or {}).items():
            rows.append(f"| {R.now_ts()[:10]} | {self.name} | `{lid}` | | | dropped: {why[:120].replace('|', '/')} | | | |")
        with open(ledger, "a", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        lit = ddir / "literature.jsonl"
        have = {}
        if lit.is_file():
            for line in lit.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    have[R.norm_url(rec["url"])] = rec
                except (ValueError, KeyError):
                    continue
        for idea in (R.read_json(self.cycle_dir / "scout.json") or {}).get("ideas", []):
            for s in idea.get("sources") or []:
                k = R.norm_url(s.get("url"))
                if not k:
                    continue
                rec = have.setdefault(k, {"url": s.get("url"), "title": s.get("title"), "venue_year": s.get("venue_year"),
                                          "verified": False, "cycles": [], "ideas": []})
                rec["verified"] = rec["verified"] or bool(s.get("verified"))
                if self.name not in rec["cycles"]:
                    rec["cycles"].append(self.name)
                rec["ideas"].append(f"{self.name}:{idea.get('id')} {idea.get('title', '')[:80]}")
        R.write_text(lit, "".join(json.dumps(r) + "\n" for r in have.values()))
        md = ["# Literature collected by rdloop cycles", "", "Generated from `literature.jsonl` by the rdloop supervisor; "
              "`verified` = the scout fetched the page in-session.", "", "| title | venue | verified | cycles | url |", "|---|---|---|---|---|"]
        for r in have.values():
            md.append(f"| {str(r.get('title', '')).replace('|', '/')[:120]} | {r.get('venue_year', '')} | {'yes' if r.get('verified') else 'no'} | "
                      f"{', '.join(r.get('cycles', []))} | {r.get('url')} |")
        R.write_text(ddir / "literature.md", "\n".join(md) + "\n")
        self.event("docs", f"published to {self.rel(out)}; ledger +{len(rows)} rows; literature {len(have)} sources")

    def ph_awaiting_user(self):
        self.st["status"] = "AWAITING_USER"
        raise Stop("AWAITING_USER")

    # -- helpers ---------------------------------------------------------------------------------
    def step_invalid(self, key, errs):
        self.st.setdefault("step_errors", {}).setdefault(key, []).extend(errs)
        self.event("step_invalid", f"{key}: {errs[:3]}")
        step = key.split(":")[0]
        if self.st["attempts"].get(key, 0) >= int(self.cfg("claude", "steps", step, "attempts", default=3)):
            self.block(step, f"{key} output invalid after {self.st['attempts'][key]} attempts: {errs[:3]}",
                       "see the step's out.json / prompt in rdloop/<C>/steps/, then resume --reset-attempts")
        return True

    def team_headline(self, agent, text):
        t = self.st.setdefault("team", {}).setdefault(agent, {})
        t["headline"] = text
        t["headline_ts"] = R.now_ts()

    # ------------------------------------------------------------------------------ loop
    HANDLERS = ("wait_waves", "audit", "scout", "propose", "lint", "critique", "gate", "scaffold", "implement", "static",
                "review", "fix", "gpu", "compat", "summary", "awaiting_user")

    def step(self):
        progressed = False
        for _ in range(30):
            ph = self.st["phase"]
            before = (ph, json.dumps(self.st.get("attempts", {}), sort_keys=True))
            if not getattr(self, "ph_" + ph)():
                return progressed
            progressed = True
            if (self.st["phase"], json.dumps(self.st.get("attempts", {}), sort_keys=True)) == before and ph == self.st["phase"]:
                return progressed
        return progressed

    def acquire_lock(self, blocking_exit=True):
        self.lock_fh = open(self.cycle_dir / ".supervisor.lock", "a+")
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if blocking_exit:
                sys.exit(f"another supervisor holds {self.cycle_dir / '.supervisor.lock'}")
            self.lock_fh.close()
            self.lock_fh = None
            return False

    def release_lock(self):
        if self.lock_fh:
            try:
                fcntl.flock(self.lock_fh, fcntl.LOCK_UN)
            except OSError:
                pass
            self.lock_fh.close()
            self.lock_fh = None

    def process_requests(self):
        rq = self.cycle_dir / "requests"
        if not rq.is_dir():
            return
        for p in sorted(rq.glob("*.json")):
            req = R.read_json(p) or {}
            try:
                self.apply_request(req)
            except Stop:
                p.unlink()
                raise
            except RequestError as exc:
                self.event("request_rejected", f"{req}: {exc}")
            done = rq / "done"
            done.mkdir(exist_ok=True)
            p.rename(done / p.name)

    def apply_request(self, req):
        cmd = req.get("cmd")
        if cmd == "gate":
            self.st["gate_mode"] = req["mode"]
            self.event("gate_mode", f"plan gate set to {req['mode']} by the owner")
        elif cmd == "park":
            self.st.update(status="PARKED", next=f"parked by the owner; `rdctl.sh reopen {self.name} --at propose` to continue")
            self.kill_own_jobs()
            self.event("parked", "cycle parked by the owner")
            raise Stop("PARKED")
        elif cmd == "reopen":
            self.reopen(req["at"], req.get("note", ""))
        else:
            self.event("warning", f"unknown request {req}")

    def kill_own_jobs(self):
        for job in list((self.st.get("jobs") or {}).values()):
            self.kill_job(job, signal.SIGTERM)
        self.st["jobs"] = {}

    def reopen(self, at, note=""):
        if at not in ("propose", "implement"):
            raise RequestError("reopen --at propose|implement")
        mod = self.cycle["module"]
        for w in R.exploop_waves(self.repo):
            spec_entry = " ".join(R.get(w["spec"], "trainer", "entry", default=[]) or [])
            if at == "implement" and f"{mod}.train" in spec_entry and w["state"].get("status") not in ("DONE", "STOPPED", None):
                raise RequestError(f"exploop wave {w['wave']} is using {mod} ({w['state'].get('status')}): the module is frozen")
        self.kill_own_jobs()
        self.st.update(status="RUNNING", blocked=None, needs_user=[], waiting=None, next=None)
        self.st["passes"] = {}
        self.st.setdefault("tests", {})
        for k in list(self.st["tests"]):
            self.st["tests"][k]["restart"] = True
        if at == "propose":
            d = self.st.setdefault("debate", {"version": 0, "rounds": [], "ledger": {}, "lint_failures": 0})
            d["rounds_offset"] = len(d.get("rounds", []))
            d["lint_try"] = 0
            for k in ("outcome", "contested", "dropped", "plan_level"):
                d.pop(k, None)
            self.st["reopen_note"] = note or "The owner sent the plan back; see the owner notes."
            self.st.pop("plan", None)
            self.st.pop("approval", None)
            self.st.pop("approved_levers", None)
            self.st["dropped_levers"] = {}
            for f in ("approval.json",):
                p = self.cycle_dir / f
                if p.is_file():
                    p.rename(self.cycle_dir / f"approval.superseded.{R.stamp()}.json")
            if self.st.get("scaffold", {}).get("done"):
                self.event("warning", f"reopened at propose after implementation: {self.lay['module_dir']} keeps its code; "
                                      f"the engineer will adapt it (fix) once the new plan is approved")
                self.st["impl_rework"] = True
            self.set_phase("propose", "reopened by the owner")
        else:
            if not self.st.get("approved_levers"):
                raise RequestError("nothing implemented yet: reopen --at propose instead")
            f = self.st.setdefault("fix", {"n": 0})
            f["max_bonus"] = f.get("max_bonus", 0) + 2
            (self.st.get("review") or {})["streak"] = 0
            f.update(reason="owner", material="The owner asked for changes; the owner notes above say what to change.\n" + note)
            self.set_phase("fix", "reopened by the owner")
        self.event("reopened", f"reopened at {at.upper()}" + (f": {note[:200]}" if note else ""))

    def run(self):
        if not self.st:
            sys.exit("not initialised: run `rdctl.sh init <cycle>` first")
        self.acquire_lock()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_flag", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_flag", True))
        try:
            self.process_requests()
        except Stop:
            self.save()
            self.render()
            return 0
        if self.st.get("status") in EXIT_STATUSES:
            self.render()
            print(f"cycle {self.name} is {self.st['status']}: {self.st.get('next') or ''}")
            return 2 if self.st["status"] in ("BLOCKED", "NEEDS_USER") else 0
        pid1 = L.pid1_start()
        if self.st.get("pid1_start") and self.st["pid1_start"] != pid1:
            self.event("container_restart", "PID 1 / boot changed: every recorded job is gone; interrupted steps rerun "
                                            "(not counted as failed attempts)")
            for name, job in list((self.st.get("jobs") or {}).items()):
                job["interrupted"] = True
                if name.startswith("claude:"):
                    key = name.split(":", 1)[1]
                    self.st.setdefault("attempts", {})[key] = max(0, self.st["attempts"].get(key, 1) - 1)
                    agent = job.get("agent")
                    if agent in self.st.get("team", {}):
                        self.st["team"][agent]["state"] = "interrupted"
                self.st["jobs"].pop(name)
            for k, seq in (self.st.get("tests") or {}).items():
                if not seq.get("done"):
                    seq["restart"] = True
            for p in (self.st.get("impl") or {}).get("parts", []):
                if p["status"] == "running":
                    p["status"] = "pending"
        self.st.update(pid1_start=pid1, supervisor_pid=os.getpid(), status="RUNNING", started_ts=R.now_ts())
        self.event("supervisor_start", f"pid {os.getpid()}")
        rc = 0
        errors = 0
        while not self.stop_flag:
            if (self.cycle_dir / "STOP").exists():
                self.st["status"] = "STOPPED"
                self.event("stopped", "STOP file found: supervisor exits; running jobs continue")
                break
            if self.st.get("status") == "PAUSED" and not (self.cycle_dir / "PAUSE").exists():
                self.st["status"] = "RUNNING"
            try:
                self.process_requests()
                progressed = self.step()
                errors = 0
            except Blocked:
                rc = 2
                break
            except Stop:
                rc = 0 if self.st.get("status") in ("AWAITING_USER", "PARKED", "HANDED_OFF") else 2
                break
            except Exception as exc:  # keep running; repeated errors block the cycle
                errors += 1
                self.event("error", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc()[-3000:])
                if errors >= 5:
                    try:
                        self.block("supervisor", f"5 consecutive errors, last: {exc}", "see events.jsonl / logs/supervisor.out")
                    except Blocked:
                        rc = 2
                        break
                progressed = False
            if self.st.get("status") == "RUNNING" and self.st.get("waiting"):
                self.st["waiting"] = None
            self.st["heartbeat_ts"] = R.now_ts()
            self.st["heartbeat_epoch"] = time.time()
            self.save()
            self.render()
            busy_gpu = any(n.startswith("gpu:") for n in (self.st.get("jobs") or {}))
            tick = 1 if progressed else (5 if busy_gpu else float(self.cfg("gpu", "poll_s", default=30)))
            t_end = time.time() + tick
            while time.time() < t_end and not self.stop_flag:
                time.sleep(0.5)
        self.st["supervisor_pid"] = None
        self.st["heartbeat_epoch"] = time.time()
        self.save()
        self.render()
        self.event("supervisor_exit", f"status {self.st.get('status')}")
        self.release_lock()
        return rc

    # ------------------------------------------------------------------------------ CLI commands
    def init(self, force=False):
        if self.st and not force:
            sys.exit(f"{self.state_path} exists; this cycle is already initialised (use --force to start over)")
        for d in ("steps", "debate", "impl", "review", "tests", "handoff", "isolation", "logs", "runs", "scratch", "jobs", "requests"):
            (self.cycle_dir / d).mkdir(parents=True, exist_ok=True)
        man = R.build_manifest(self.repo, self.cycle, self.rel(self.cycle_dir))
        R.write_json(self.cycle_dir / "isolation" / "manifest.json", man)
        n_frozen = R.freeze_untracked(self.repo, man, self.cycle_dir / "isolation" / "frozen_untracked.tar.gz")
        prov = R.resolve_base(self.repo, self.cfg("base", default={}) or {}) or {}
        self.st = {"cycle": self.name, "status": "INITIALISED", "phase": "wait_waves", "created": R.now_ts(), "cost_usd": 0.0,
                   "gate_mode": self.cfg("gates", "plan", default="owner"), "attempts": {}, "jobs": {}, "claude_sessions": [],
                   "team": {}, "base_provisional": {k: prov.get(k) for k in ("label", "run_dir", "from_wave", "entry_module")},
                   "pid1_start": L.pid1_start(), "exploop_lib_sha": R.sha256_file(HERE.parent / "exploop" / "exploop_lib.py"),
                   "next": f"`scripts/rdloop/rdctl.sh start {self.name}` when you want it to run"}
        notes = self.cycle_dir / "user_notes.md"
        if not notes.exists():
            R.write_text(notes, f"# Owner notes for rdloop cycle {self.name}\n\nAppend with `scripts/rdloop/rdctl.sh note {self.name} "
                                "\"...\"`. Every judgement step reads them first (newest last).\n")
        gi = self.repo / "rdloop" / ".gitignore"
        if not gi.exists():
            R.write_text(gi, "*/runs/\n*/isolation/*.tar.gz\n*/jobs/\n*/.supervisor.lock\n")
        self.event("init", f"initialised: manifest of {len(man['files'])} files + {len(man['runs'])} run files; "
                           f"{n_frozen} untracked files frozen; provisional base {prov.get('label')}")
        self.save()
        self.render()
        print(f"initialised {self.rel(self.state_path)}: module {self.lay['module_dir']}, gate {self.st['gate_mode']}, "
              f"provisional base {prov.get('label')} ({prov.get('run_dir')}), waits for {self.cfg('wait_for_waves', default=[])}")
        print(f"isolation: {len(man['files'])} protected files hashed, {len(man['runs'])} run files recorded, "
              f"{n_frozen} untracked files frozen in isolation/frozen_untracked.tar.gz")

    def health(self):
        lock_held = False
        lp = self.cycle_dir / ".supervisor.lock"
        if lp.exists():
            with open(lp, "a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except OSError:
                    lock_held = True
        hb = self.st.get("heartbeat_epoch") or 0
        return {"running": lock_held, "heartbeat_age_s": (time.time() - hb) if hb else None,
                "pid1_changed": bool(self.st.get("pid1_start") and self.st["pid1_start"] != L.pid1_start())}

    def status(self):
        if not self.st:
            print(f"cycle {self.name}: not initialised")
            return 1
        h = self.health()
        s = self.st.get("status")
        print(f"cycle {self.name}: {s} - phase {str(self.st.get('phase')).upper()}")
        print(f"  supervisor: {'RUNNING (lock held)' if h['running'] else 'not running'}"
              + (f", heartbeat {h['heartbeat_age_s']:.0f}s ago" if h['heartbeat_age_s'] is not None else ""))
        if h["pid1_changed"] and s not in EXIT_STATUSES:
            print(f"  container restarted since the supervisor last ran: `rdctl.sh resume {self.name}`")
        for name, job in (self.st.get("jobs") or {}).items():
            print(f"  job {name}: pid {job['pid']} {'alive' if self.job_alive(job) else 'gone'} (since {job['started_ts']})")
        b = self.st.get("base") or self.st.get("base_provisional") or {}
        print(f"  base: {b.get('label')} ({b.get('run_dir')}); module {self.lay['module_dir']}; gate {self.st.get('gate_mode')}")
        if self.st.get("approved_levers"):
            print(f"  approved levers: {self.lever_ids()}")
        if self.st.get("waiting"):
            print(f"  waiting: {self.st['waiting']}")
        if self.st.get("blocked"):
            bl = self.st["blocked"]
            print(f"  BLOCKED in {bl['phase']}: {bl['reason']}\n  hint: {bl.get('hint')}")
        for x in self.st.get("needs_user") or []:
            print(f"  needs the owner: {x}")
        if self.st.get("next"):
            print(f"  next: {self.st['next']}")
        print(f"  claude spend: ${self.st.get('cost_usd', 0):.2f} / ${self.cfg('claude', 'max_total_usd', default=150)}; "
              f"status page: {self.rel(self.cycle_dir / 'STATUS.md')}")
        if s in ("AWAITING_USER", "PARKED", "HANDED_OFF", "AWAITING_APPROVAL"):
            return 3
        if s in ("BLOCKED", "NEEDS_USER"):
            return 2
        return 0 if h["running"] else 1

    def plan_cmd(self):
        if not self.st:
            print("not initialised")
            return 1
        ph = self.st.get("phase")
        what = {"wait_waves": f"wait for exploop waves {self.cfg('wait_for_waves', default=[])} to be DONE, then resolve the base",
                "audit": "AUDIT: claude -p --agent research-auditor -> 01_audit.md + audit.json",
                "scout": "SCOUT: claude -p --agent research-scout (web) -> 02_sota.md + scout.json",
                "propose": "PROPOSE / REVISE: claude -p --agent research-architect -> debate/proposal_vN.{md,json}",
                "lint": "LINT the latest proposal (no Claude call)",
                "critique": "CRITIQUE: claude -p --agent research-critic -> debate/critique_vN.{md,json}",
                "gate": f"GATE ({self.st.get('gate_mode')}): plan_agreed.md -> owner approval (rdctl approve) or auto",
                "scaffold": f"SCAFFOLD {self.lay['module_dir']} from the base module",
                "implement": "IMPLEMENT: claude -p --agent research-engineer, core then one step per approved lever",
                "static": "STATIC checks (CPU): manifest, py_compile, --help, <module>.checks cpu, base checks, exploop tests",
                "review": "CODE REVIEW: claude -p --agent research-critic",
                "fix": "FIX: claude -p --agent research-engineer",
                "gpu": "GPU session (waits for an idle GPU / yields to exploop): regression on the base run + smoke (<= 30 min)",
                "compat": "COMPAT: handoff/wave_seed.json + exploop init/plan dry run",
                "summary": "SUMMARY: claude -p --agent research-auditor -> FINAL_SUMMARY.md, docs/research/<C>/",
                "awaiting_user": f"owner review: /rd-cycle review {self.name}"}.get(ph, ph)
        print(f"cycle {self.name}: status {self.st.get('status')}, phase {str(ph).upper()}\n  next: {what}")
        if ph == "gpu":
            self.readonly = True
            try:
                for j in self.build_gpu():
                    print(f"   [{j['kind']}] {j['name']}: ({j.get('cwd')}) {' '.join(j['argv'])[:300]}")
            except Exception as exc:
                print(f"   (cannot expand the GPU jobs yet: {exc})")
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
            print(f"{e['ts']} [{e.get('phase')}] {e['kind']}: {e['msg']}")

    def submit(self, req):
        """Apply a control request now (no supervisor running) or queue it for the running one."""
        if self.acquire_lock(blocking_exit=False):
            try:
                try:
                    self.apply_request(req)
                except Stop:
                    pass
                except RequestError as exc:
                    sys.exit(f"rejected: {exc}")
                self.save()
                self.render()
            finally:
                self.release_lock()
            print(f"applied: {req}")
        else:
            rq = self.cycle_dir / "requests"
            rq.mkdir(exist_ok=True)
            R.write_json(rq / f"{R.stamp()}_{req['cmd']}.json", req)
            print(f"queued for the running supervisor: {req}")

    def approve(self, levers):
        pl = self.st.get("plan")
        if not pl:
            sys.exit("no agreed plan yet (the gate opens after the debate)")
        lv = [x.strip() for x in levers.split(",") if x.strip()] if levers else None
        if lv:
            bad = [x for x in lv if x not in pl["levers"]]
            if bad:
                sys.exit(f"unknown lever(s) {bad}; plan levers: {pl['levers']}")
        R.write_json(self.cycle_dir / "approval.json", {"plan_sha": pl["sha"], "levers": lv, "ts": R.now_ts(), "by": "owner"})
        print(f"approved plan {pl['sha'][:12]} levers {lv or 'default (all agreed, not contested, no missing dependency)'}")
        return 0

    def unblock(self, reset_attempts, accept_manifest):
        if self.st.get("blocked"):
            self.event("unblock", f"cleared: {self.st['blocked'].get('reason', '')[:200]}")
        self.st["blocked"] = None
        self.st["needs_user"] = []
        if self.st.get("status") in ("BLOCKED", "STOPPED", "INITIALISED", "NEEDS_USER", "PAUSED"):
            self.st["status"] = "RESUMING"
        if reset_attempts:
            self.st["attempts"] = {}
            if self.st.get("debate"):
                self.st["debate"]["lint_failures"] = 0
        if accept_manifest:
            man = R.build_manifest(self.repo, self.cycle, self.rel(self.cycle_dir))
            R.write_json(self.cycle_dir / "isolation" / "manifest.json", man)
            self.event("manifest", f"re-baselined by the owner: {len(man['files'])} files")
        self.save()
        self.render()

    def verify(self):
        sha = R.code_sha(self.repo, self.cycle)
        tested = (self.st.get("passes") or {}).get("gpu")
        man = R.read_json(self.cycle_dir / "isolation" / "manifest.json")
        v = R.check_manifest(self.repo, man, self.cycle, self.rel(self.cycle_dir)) if man else ["no manifest"]
        ok = sha == tested and not v and self.st.get("status") in ("AWAITING_USER", "HANDED_OFF")
        print(f"module {self.lay['module_dir']} sha {sha[:12]}; tested sha {str(tested)[:12]}; status {self.st.get('status')}; "
              f"isolation {'ok' if not v else str(len(v)) + ' violation(s): ' + str(v[:4])}")
        print("VERIFIED: the module is exactly the code that passed regression + smoke" if ok else "NOT VERIFIED")
        return 0 if ok else 1


# ---------------------------------------------------------------------------------- helpers

def json_from_text(text):
    for block in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)):
        try:
            return json.loads(block)
        except ValueError:
            continue
    return None


def cmd_new(a):
    cdir = Path(a.cycle).resolve()
    name = cdir.name
    if (cdir / "cycle.json").exists() and not a.force:
        sys.exit(f"{cdir}/cycle.json exists (use --force to overwrite)")
    tpl = (HERE / "templates" / "cycle.json").read_text(encoding="utf-8")
    m = re.match(r"^[A-Za-z]*(\d+)", name)
    module = a.module or (f"b{m.group(1)}wave" if m else f"{name.lower()}wave")
    prefix = a.prefix or (module[:2] + "_" if re.match(r"^b\d", module) else module + "_")
    from_wave = a.from_wave or (R.latest_wave(R.REPO) or {}).get("wave") or ""
    cyc = json.loads(tpl.replace("@CYCLE@", name).replace("@MODULE@", module).replace("@PREFIX@", prefix)
                     .replace("@FROM_WAVE@", from_wave).replace("@DESCRIPTION@", f"rdloop cycle {name}: new levers in SLS_setup/{module}"))
    cyc["repo"] = str(R.REPO)
    if a.wait_for is not None:
        cyc["wait_for_waves"] = a.wait_for
    else:
        cyc["wait_for_waves"] = [w["wave"] for w in R.exploop_waves(R.REPO) if w["state"].get("status") not in ("DONE", "STOPPED")]
    if a.auto_gate:
        cyc["gates"]["plan"] = "auto"
    if a.levers_max:
        cyc["levers"]["max"] = a.levers_max
        cyc["levers"]["min"] = min(cyc["levers"]["min"], a.levers_max)
    if a.levers_min:
        cyc["levers"]["min"] = a.levers_min
    cyc["focus"] = a.focus or []
    cyc["exclude"] = a.exclude or []
    if a.budget:
        cyc["claude"]["max_total_usd"] = a.budget
    if (R.REPO / "SLS_setup" / module).exists():
        sys.exit(f"SLS_setup/{module} already exists: pass --module <new name>")
    cdir.mkdir(parents=True, exist_ok=True)
    R.write_json(cdir / "cycle.json", cyc)
    brief = a.brief or (Path(a.brief_file).read_text(encoding="utf-8") if a.brief_file else "")
    R.write_text(cdir / "brief.md", f"# Owner brief for rdloop cycle {name}\n\n_{R.now_ts()}_\n\n{brief.strip()}\n")
    print(f"wrote {R.rel(R.REPO, cdir / 'cycle.json')} (module {module}, prefix {prefix}, base from wave {from_wave or '?'}, "
          f"waits for {cyc['wait_for_waves']}, gate {cyc['gates']['plan']}, levers {cyc['levers']['min']}-{cyc['levers']['max']}, "
          f"budget ${cyc['claude']['max_total_usd']}) and brief.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["new", "init", "run", "status", "plan", "tail", "approve", "gate", "reopen", "park",
                                    "unblock", "verify", "handed-off", "alive"])
    ap.add_argument("arg", nargs="?", help="gate: owner|auto")
    ap.add_argument("--cycle", required=True, help="cycle dir, e.g. rdloop/B4")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--levers", default=None)
    ap.add_argument("--at", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--wave", default=None)
    ap.add_argument("--reset-attempts", action="store_true")
    ap.add_argument("--accept-manifest", action="store_true")
    ap.add_argument("--brief", default=None)
    ap.add_argument("--brief-file", default=None)
    ap.add_argument("--auto-gate", action="store_true")
    ap.add_argument("--from-wave", default=None)
    ap.add_argument("--module", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--wait-for", nargs="*", default=None)
    ap.add_argument("--levers-max", type=int, default=None)
    ap.add_argument("--levers-min", type=int, default=None)
    ap.add_argument("--focus", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "new":
        return cmd_new(a)
    s = Supervisor(a.cycle)
    if a.cmd == "run":
        sys.exit(s.run())
    if a.cmd == "init":
        return s.init(force=a.force)
    if not s.st:
        sys.exit(f"cycle {s.name} is not initialised (rdctl.sh init {s.name})")
    if a.cmd == "status":
        sys.exit(s.status())
    if a.cmd == "alive":
        sys.exit(0 if s.health()["running"] else 1)
    if a.cmd == "plan":
        sys.exit(s.plan_cmd())
    if a.cmd == "tail":
        return s.tail(a.n)
    if a.cmd == "approve":
        sys.exit(s.approve(a.levers))
    if a.cmd == "gate":
        if a.arg not in ("owner", "auto"):
            sys.exit("gate owner|auto")
        return s.submit({"cmd": "gate", "mode": a.arg})
    if a.cmd == "reopen":
        return s.submit({"cmd": "reopen", "at": a.at, "note": a.note})
    if a.cmd == "park":
        return s.submit({"cmd": "park"})
    if a.cmd == "unblock":
        if not s.acquire_lock(blocking_exit=False):
            sys.exit("a supervisor is running for this cycle; stop it first (rdctl.sh stop) or use reopen / gate")
        s.unblock(a.reset_attempts, a.accept_manifest)
        s.release_lock()
        return None
    if a.cmd == "verify":
        sys.exit(s.verify())
    if a.cmd == "handed-off":
        if not a.wave:
            sys.exit("handed-off --wave <W>")
        s.st.update(status="HANDED_OFF", handed_off={"wave": a.wave, "ts": R.now_ts(), "code_sha": R.code_sha(s.repo, s.cycle)},
                    next=f"exploop wave {a.wave} screens the levers; the module is frozen while it runs")
        s.event("handed_off", f"handed off to exploop wave {a.wave}")
        s.save()
        s.render()
        print(f"cycle {s.name} handed off to {a.wave}")
    return None


if __name__ == "__main__":
    main()
