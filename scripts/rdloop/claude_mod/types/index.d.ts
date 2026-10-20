// The JSON `scripts/rdloop/view.py snapshot` prints (the fields this mod reads), and the mod's $.state.

export type RdloopEvent = {
  seq: number
  ts: string | null
  phase: string | null
  kind: string
  msg: string
}

export type RdloopRecent = { tool: string | null; target: string | null; ts: string | null }

export type RdloopAgent = {
  agent: string
  label: string
  role: string
  state: string
  step: string | null
  headline: string | null
  cost: number
  attempt: number | null
  session_id: string | null
  elapsed_s: number | null
  tool_uses?: number
  last_tool?: string | null
  last_target?: string | null
  last_text?: string | null
  recent?: RdloopRecent[]
  queries?: string[]
  fetched?: string[]
  files_written?: string[]
}

export type RdloopObjection = { id: string; severity: string; lever_id: string; claim: string; required_change?: string }

export type RdloopRound = {
  v: number
  levers: string[]
  counts: { blocker?: number; major?: number; minor?: number; resolved?: number }
  agreed: boolean
  critic_verdict?: string | null
  ts?: string
}

export type RdloopDebate = {
  version: number
  rounds: RdloopRound[]
  outcome: string | null
  open: RdloopObjection[]
  counts: { blocker?: number; major?: number; minor?: number; resolved?: number }
  lint_failures: number
  responses: { id: string; action: string; detail: string }[]
  levers_in_play: string[]
}

export type RdloopLever = {
  id: string
  title: string | null
  category?: string | null
  status: string
  flags: string[]
  gap_ids: string[]
  dropped_reason: string | null
  smoke_state: string | null
  smoke_note: string | null
  peak_gib: number | null
  time_frac: number | null
  eta_full_h: number | null
}

export type RdloopJob = {
  name: string
  kind: string
  state: string
  elapsed_s: number | null
  note: string
  lever: string | null
  peak_gib: number | null
  budget_s: number | null
}

export type RdloopTests = {
  jobs: RdloopJob[]
  done: boolean
  ok: boolean | null
  code_sha: string
  smoke_spent_s: number
  smoke_budget_s: number
}

export type RdloopAction = { kind: string; text: string; command: string; file: string }

export type RdloopSnapshot = {
  ok: boolean
  cycle: string | null
  error: string | null
  taken?: string
  status?: string | null
  phase?: string | null
  phase_key?: string | null
  phase_index?: number | null
  phases?: string[]
  gate_mode?: string | null
  module?: string | null
  base?: { label: string | null; run_dir: string | null; from_wave: string | null }
  cost_usd?: number
  max_total_usd?: number | null
  supervisor?: {
    state: string
    pid: number | null
    heartbeat_age_s: number | null
    container_restarted: boolean
    paused: boolean
    stop_requested: boolean
  }
  action?: RdloopAction | null
  waiting?: string | null
  blocked?: { phase?: string; reason?: string; hint?: string } | null
  needs_user?: string[]
  next?: string | null
  debate_round?: number
  debate_max?: number
  team?: RdloopAgent[]
  debate?: RdloopDebate
  levers?: RdloopLever[]
  tests?: { static?: RdloopTests; gpu?: RdloopTests; compat?: RdloopTests }
  review?: { n: number; open: { id: string; severity: string; lever_id?: string; issue?: string; file?: string }[] }
  fix_rounds?: number
  events_total?: number
  events?: RdloopEvent[]
  files?: { [k: string]: string }
}

export type RdloopCursor = { cycle: string; seq: number }

declare module 'claude-code' {
  interface PluginState {
    'rdloop-view': {
      snap: RdloopSnapshot | null
      cycle: string | null
      cursor: RdloopCursor | null
      error: string | null
      focus: string | null
    }
  }
}
