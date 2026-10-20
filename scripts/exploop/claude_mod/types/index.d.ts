// The JSON `scripts/exploop/view.py snapshot` prints (the fields this mod reads), and the mod's $.state.

export type ExploopEvent = {
  seq: number
  ts: string | null
  iter: number | null
  phase: string | null
  kind: string
  msg: string
  data?: { epoch: number | null; delta_vs_best: number | null; wf1: number | null }
}

export type ExploopCompareRow = {
  epoch: number
  wf1: number | null
  d_wf1: number | null
  best_wf1: number | null
  f1_noisy: number | null
  d_f1_noisy: number | null
  eer_noisy: number | null
  d_eer_noisy: number | null
  dev_loss: number | null
  d_dev_loss: number | null
}

export type ExploopCompare = {
  best_label: string | null
  best_merge: number | null
  best_final: number | null
  best_registry_ids: { [k: string]: string }
  rows: ExploopCompareRow[]
  last: ExploopCompareRow | null
  epoch_now: number | null
  num_epochs: number
  target_now: number | null
  best_peak: { epoch: number; wf1: number } | null
  current_peak: { epoch: number; wf1: number } | null
  ahead: number
  judged: number
  pace_s: number | null
  best_pace_s: number | null
  spark: string
  best_spark: string
}

export type ExploopProgress = {
  epoch: number
  num_epochs: number
  epochs_done: number
  substep: string | null
  i: number | null
  n: number | null
  pct: number | null
  rate_per_s: number | null
  epoch_s: number | null
  remaining_s: number | null
  eta_h: number | null
  log_age_s: number | null
  validate?: { i: number; n: number; pct: number }
}

export type ExploopEvalStep = { name: string; gpu: boolean; optional: boolean; state: string }

export type ExploopEval = {
  steps: ExploopEvalStep[]
  done: number
  total: number
  running: { name: string; pid: number; elapsed_s: number; log: string | null } | null
}

export type ExploopCurrent = {
  k: number | null
  label: string | null
  track?: string | null
  phase: string
  phase_index: number | null
  change?: string | null
  run_name?: string | null
  num_epochs?: number
  progress?: ExploopProgress | null
  eval?: ExploopEval
}

export type ExploopClaude = {
  step: string
  agent: string | null
  session_id: string | null
  elapsed_s: number
  tool_uses: number
  last_tool: string | null
  last_target: string | null
}

export type ExploopIteration = {
  k: number
  label: string | null
  change: string | null
  phase: string | null
  done: boolean
  merge: number | null
  merge_delta: number | null
  ci_low: number | null
  ci_high: number | null
  final: number | null
  decision: string | null
  new_best: boolean
}

export type ExploopSnapshot = {
  ok: boolean
  wave: string | null
  error: string | null
  taken?: string
  status?: string | null
  waiting?: string | null
  blocked?: { phase?: string; reason?: string; hint?: string } | null
  max_iterations?: number
  phases?: string[]
  cost_usd?: number
  max_total_usd?: number | null
  best?: { label: string | null; merge_value: number | null; final_value: number | null }
  supervisor?: {
    state: string
    pid: number | null
    heartbeat_age_s: number | null
    container_restarted: boolean
    paused: boolean
    stop_requested: boolean
  }
  status_page?: string
  summary_report?: string | null
  wave_phase?: string
  events_total?: number
  events?: ExploopEvent[]
  iterations?: ExploopIteration[]
  current?: ExploopCurrent | null
  compare?: ExploopCompare
  claude?: ExploopClaude
}

export type ExploopCursor = { wave: string; seq: number }

declare module 'claude-code' {
  interface PluginState {
    'exploop-view': {
      snap: ExploopSnapshot | null
      wave: string | null
      cursor: ExploopCursor | null
      error: string | null
    }
  }
}
