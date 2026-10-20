// exploop-view: a live view of the running exploop wave (scripts/exploop) inside Claude Code.
//
// Every 10 s it runs `scripts/exploop/view.py snapshot` (read-only: never touches the supervisor's lock or
// state) and keeps the JSON in $.state; the /exploop pane draws from it, and new lines of the wave's
// events.jsonl become toasts (phase change, epoch vs best, new best) or notifications (BLOCKED, DONE,
// supervisor down). It does nothing in a directory without scripts/exploop/view.py.
import { atom, read, update } from 'claude-code'
import type { Elements, EngineInterface, Register } from 'claude-code'

import type { ExploopCompareRow, ExploopEvent, ExploopSnapshot } from '../types'

const PANE = 'exploop'
const POLL_MS = 10_000
const snapAtom = atom({ plugin: 'exploop-view', key: 'snap' } as const, null)
const waveAtom = atom({ plugin: 'exploop-view', key: 'wave' } as const, null)
const cursorAtom = atom({ plugin: 'exploop-view', key: 'cursor' } as const, null)
const errorAtom = atom({ plugin: 'exploop-view', key: 'error' } as const, null)

const TOAST_KINDS = new Set(['phase', 'epoch', 'new_best', 'iteration_done', 'early_check_failed', 'train_died',
  'eval_failed', 'design_rejected', 'redesign', 'container_restart', 'supervisor_exit', 'unblock'])
const ALERT_KINDS = new Set(['blocked', 'done'])
const MAX_TOASTS_PER_POLL = 4

type El = Pick<Elements['terminal'], 'Box' | 'Text' | 'Button'>

let root: string | null = null
let timer: { cancel: () => void } | null = null
let polling = false

// ------------------------------------------------------------------------------------ formatting

const fmt = (x: number | null | undefined, nd = 2) => (x === null || x === undefined ? '–' : x.toFixed(nd))
const sgn = (x: number | null | undefined, nd = 2) =>
  x === null || x === undefined ? '–' : `${x >= 0 ? '+' : ''}${x.toFixed(nd)}`
const pad = (s: string, n: number) => (s.length >= n ? s : ' '.repeat(n - s.length) + s)
const padEnd = (s: string, n: number) => (s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length))

function dur(s: number | null | undefined): string {
  if (s === null || s === undefined) return '?'
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

function bar(pct: number | null | undefined, width = 20): string {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const full = Math.round((p / 100) * width)
  return '▕' + '█'.repeat(full) + '░'.repeat(width - full) + '▏'
}

// higher is better for WF1 / F1, lower for EER / loss
function deltaColor(d: number | null | undefined, higherIsBetter = true): 'success' | 'error' | 'subtle' {
  if (d === null || d === undefined || d === 0) return 'subtle'
  return d > 0 === higherIsBetter ? 'success' : 'error'
}

function labelOf(snap: ExploopSnapshot, iter: number | null): string {
  return snap.iterations?.find(i => i.k === iter)?.label ?? (iter ? `iter ${iter}` : '')
}

function isLive(snap: ExploopSnapshot | null): boolean {
  return !!snap?.ok && snap.status !== 'DONE' && snap.supervisor?.state !== 'exited'
}

export function summaryLine(snap: ExploopSnapshot | null, error: string | null): string {
  if (!snap) return `exploop: no snapshot yet${error ? ` (${error})` : ''}`
  if (!snap.ok) return `exploop ${snap.wave ?? ''}: ${snap.error ?? 'no live wave'}`.trim()
  const c = snap.current
  const parts = [`exploop ${snap.wave} ${snap.status}`]
  if (c?.k) parts.push(`iteration ${c.k}/${snap.max_iterations} ${c.label} phase ${c.phase.toUpperCase()}`)
  const p = c?.progress
  if (p) parts.push(`epoch ${p.epoch}/${p.num_epochs} ${p.pct ?? 0}%` + (p.eta_h !== null ? `, ETA ${dur(p.eta_h * 3600)}` : ''))
  const last = snap.compare?.last
  if (last) parts.push(`ep${last.epoch} LocalWF1 ${fmt(last.wf1)} (${sgn(last.d_wf1)} vs ${snap.compare?.best_label}@${last.epoch})`)
  if (snap.best?.label) parts.push(`best ${snap.best.label} half_B ${fmt(snap.best.merge_value)}`)
  if (snap.supervisor?.state === 'down') parts.push('supervisor DOWN')
  return parts.join(' · ')
}

export function eventText(snap: ExploopSnapshot, e: ExploopEvent): string {
  const w = snap.wave ?? ''
  if (e.kind === 'phase') return `${w}: ${e.msg.replace(' -> ', ' → ')}`
  if (e.kind === 'epoch' && e.data && e.data.wf1 !== null) {
    const ep = e.data.epoch
    const best = snap.best?.label ?? 'best'
    const d = e.data.delta_vs_best
    return `${w} ${labelOf(snap, e.iter)} epoch ${ep}: LocalWF1 ${fmt(e.data.wf1)}` + (d !== null ? ` (${sgn(d)} vs ${best}@${ep})` : '')
  }
  return `${w} ${e.kind.replace(/_/g, ' ')}: ${e.msg.slice(0, 160)}`
}

// ------------------------------------------------------------------------------------- polling

async function alert($: EngineInterface, text: string) {
  $.ui.toast(text, { timeoutMs: 15_000 })
  try {
    await $.ui.notify(text, { title: 'exploop' })
  } catch {
    // the toast already said it
  }
}

async function announce($: EngineInterface, prev: ExploopSnapshot | null, snap: ExploopSnapshot) {
  const wave = snap.wave ?? ''
  const total = snap.events_total ?? 0
  const cur = await read($, cursorAtom)
  if (!cur || cur.wave !== wave || total < cur.seq) {
    // first sight of this wave (or its events were rewritten): baseline, so a restart replays nothing
    await update($, cursorAtom, () => ({ wave, seq: total }))
    return
  }
  const fresh = (snap.events ?? []).filter(e => e.seq > cur.seq)
  const alerts = fresh.filter(e => ALERT_KINDS.has(e.kind))
  const toasts = fresh.filter(e => TOAST_KINDS.has(e.kind))
  for (const e of alerts) await alert($, eventText(snap, e))
  const shown = toasts.slice(-MAX_TOASTS_PER_POLL)
  if (toasts.length > shown.length) $.ui.toast(`${wave}: ${toasts.length - shown.length} more events (/exploop)`)
  for (const e of shown) $.ui.toast(eventText(snap, e), { timeoutMs: e.kind === 'phase' ? 6000 : 10_000 })
  if (total !== cur.seq) await update($, cursorAtom, () => ({ wave, seq: total }))
  const was = prev?.ok && prev.wave === snap.wave ? prev.supervisor?.state : undefined
  if (was && was !== 'down' && snap.supervisor?.state === 'down' && snap.status !== 'DONE') {
    await alert($, `${wave}: supervisor DOWN - run scripts/exploop/loopctl.sh resume ${wave}`)
  }
}

export async function poll($: EngineInterface): Promise<void> {
  if (!root || polling) return
  polling = true
  try {
    const wave = await read($, waveAtom)
    const argv = ['python3', `${root}/scripts/exploop/view.py`, 'snapshot', ...(wave ? ['--wave', wave] : [])]
    const r = await $.process.run(argv, { cwd: root, timeoutMs: 20_000 })
    let snap: ExploopSnapshot | null = null
    try {
      snap = JSON.parse(r.stdout) as ExploopSnapshot
    } catch {
      snap = null
    }
    if (!snap) {
      await update($, errorAtom, () => `view.py exit ${r.exitCode}: ${r.stderr.trim().slice(-200)}`)
      return
    }
    const fresh = snap
    const prev = await read($, snapAtom)
    await update($, snapAtom, () => fresh)
    await update($, errorAtom, () => (fresh.ok ? null : fresh.error))
    if (fresh.ok) await announce($, prev, fresh)
  } catch (err) {
    await update($, errorAtom, () => `poll failed: ${String(err).slice(0, 200)}`)
  } finally {
    polling = false
  }
}

async function findRoot($: EngineInterface, cwd: string): Promise<string | null> {
  let dir = cwd.replace(/\/+$/, '')
  for (let i = 0; i < 6 && dir; i++) {
    const hit = await $.fs.stat(`${dir}/scripts/exploop/view.py`).catch(() => undefined)
    if (hit?.kind === 'file') return dir
    dir = dir.slice(0, dir.lastIndexOf('/'))
  }
  return null
}

// ---------------------------------------------------------------------------------------- pane

function header(el: El, snap: ExploopSnapshot) {
  const { Box, Text } = el
  const sup = snap.supervisor
  const stateColor = { alive: 'success', stale: 'warning', down: 'error', exited: 'inactive' }[sup?.state ?? ''] ?? 'subtle'
  const statusColor =
    { RUNNING: 'success', DONE: 'success', BLOCKED: 'error', WAITING: 'warning', PAUSED: 'warning', STOPPED: 'warning' }[
      snap.status ?? ''
    ] ?? 'subtle'
  return (
    <Box flexDirection="column">
      <Text wrap="truncate">
        <Text bold>{snap.wave}</Text> <Text bold color={statusColor}>{snap.status}</Text>
        <Text color="subtle">{'  supervisor '}</Text>
        <Text color={stateColor}>● {sup?.state}</Text>
        <Text color="subtle">
          {sup?.heartbeat_age_s !== null && sup?.heartbeat_age_s !== undefined ? ` (heartbeat ${dur(sup.heartbeat_age_s)} ago)` : ''}
          {`  Claude $${fmt(snap.cost_usd)} / $${snap.max_total_usd ?? '?'}`}
        </Text>
      </Text>
      {sup?.container_restarted && snap.status !== 'DONE' && (
        <Text color="error" bold>
          Container restarted since the supervisor started: scripts/exploop/loopctl.sh resume {snap.wave}
        </Text>
      )}
      {sup?.state === 'down' && snap.status !== 'DONE' && (
        <Text color="error" bold>
          Supervisor DOWN: scripts/exploop/loopctl.sh resume {snap.wave}
        </Text>
      )}
      {sup?.paused && <Text color="warning">PAUSED: no new GPU job starts (loopctl.sh unpause {snap.wave})</Text>}
      {sup?.stop_requested && snap.status !== 'STOPPED' && <Text color="warning">Stop requested</Text>}
    </Box>
  )
}

function pipeline(el: El, snap: ExploopSnapshot) {
  const { Box, Text } = el
  const c = snap.current
  if (!c?.k) {
    return <Text color="subtle">{snap.wave_phase === 'summary' ? 'All iterations done: SUMMARY step' : snap.wave_phase ?? ''}</Text>
  }
  const idx = c.phase_index
  const phases = snap.phases ?? []
  return (
    <Box flexDirection="column">
      <Text wrap="truncate">
        <Text>{`iteration ${c.k}/${snap.max_iterations} `}</Text>
        <Text bold>{c.label}</Text>
        <Text>{'  '}</Text>
        {phases.map((p, i) => (
          <Text
            key={p}
            bold={i === idx}
            color={idx === null ? 'subtle' : i < idx ? 'success' : i === idx ? 'suggestion' : 'inactive'}
          >
            {(i === idx ? `[${p.toUpperCase()}]` : p) + (i < phases.length - 1 ? ' › ' : '')}
          </Text>
        ))}
      </Text>
      {c.change && (
        <Text color="subtle" wrap="truncate">
          change: {c.change}
        </Text>
      )}
    </Box>
  )
}

function stepCard(el: El, snap: ExploopSnapshot) {
  const { Box, Text } = el
  const c = snap.current
  const out = []
  if (snap.status === 'BLOCKED' && snap.blocked) {
    out.push(
      <Text key="blocked" color="error" bold>
        BLOCKED in {snap.blocked.phase}: {snap.blocked.reason}
      </Text>,
      <Text key="hint" color="warning">
        hint: {snap.blocked.hint ?? ''} - then scripts/exploop/loopctl.sh resume {snap.wave}
      </Text>,
    )
  }
  if (snap.waiting) out.push(<Text key="waiting" color="warning">waiting: {snap.waiting}</Text>)
  if (snap.claude) {
    const a = snap.claude
    out.push(
      <Text key="claude" wrap="truncate">
        <Text color="claude" bold>{a.agent ?? a.step}</Text>
        {` running ${dur(a.elapsed_s)} · ${a.tool_uses} tool calls`}
        {a.last_tool ? ` · last: ${a.last_tool} ${a.last_target ?? ''}` : ''}
      </Text>,
      <Text key="resume" color="subtle" wrap="truncate">
        claude --resume {a.session_id ?? '?'}
      </Text>,
    )
  }
  const p = c?.progress
  if (c?.phase === 'train' && p) {
    const where =
      p.substep === 'validate' && p.validate
        ? `validating ${p.validate.pct}% (${p.validate.i}/${p.validate.n})`
        : p.substep === 'local eval'
          ? 'local eval of the epoch'
          : `training ${p.i ?? '?'}/${p.n ?? '?'} batches${p.rate_per_s ? `, ${p.rate_per_s.toFixed(2)} batch/s` : ''}`
    out.push(
      <Text key="epoch" wrap="truncate">
        <Text bold>{`epoch ${p.epoch}/${p.num_epochs} `}</Text>
        <Text color="suggestion">{bar(p.pct)}</Text>
        {` ${p.pct ?? 0}%  ${where}`}
      </Text>,
      <Text key="eta" color="subtle" wrap="truncate">
        {`epoch ends in ~${dur(p.substep === 'train' ? p.remaining_s : 0)} · run ETA ~${p.eta_h !== null ? dur(p.eta_h * 3600) : '?'}`}
        {p.log_age_s !== null && p.log_age_s > 300 ? ` · log silent for ${dur(p.log_age_s)}` : ''}
        {c.run_name ? ` · ${c.run_name}` : ''}
      </Text>,
    )
  }
  if (c?.eval) {
    const ev = c.eval
    out.push(
      <Text key="evalhead" bold>
        {`eval ${ev.done}/${ev.total}`}
        {ev.running ? ` - running ${ev.running.name} for ${dur(ev.running.elapsed_s)}` : ''}
      </Text>,
      <Text key="evalsteps" wrap="wrap">
        {ev.steps.map(s => (
          <Text
            key={s.name}
            color={s.state === 'done' ? 'success' : s.state === 'running' ? 'suggestion' : s.state === 'skipped' ? 'warning' : 'inactive'}
          >
            {`${s.state === 'done' ? '✓' : s.state === 'running' ? '▶' : s.state === 'skipped' ? '–' : '·'} ${s.name}${s.gpu ? ' (GPU)' : ''}   `}
          </Text>
        ))}
      </Text>,
    )
  }
  if (snap.status === 'DONE') {
    out.push(
      <Text key="done" color="success" bold>
        Wave done.{snap.summary_report ? ` Summary: ${snap.summary_report}` : ''}
      </Text>,
    )
  }
  return out.length ? <Box flexDirection="column">{out}</Box> : null
}

function cell(el: El, v: number | null, d: number | null, nd: number, higherIsBetter: boolean, w: number) {
  const { Text } = el
  return [
    <Text key="v">{pad(fmt(v, nd), w)}</Text>,
    <Text key="d" color={deltaColor(d, higherIsBetter)}>{pad(sgn(d, nd), w)}</Text>,
  ]
}

function compareCard(el: El, snap: ExploopSnapshot) {
  const { Box, Text } = el
  const cmp = snap.compare
  if (!cmp) return null
  const label = snap.current?.label ?? 'current'
  const bl = cmp.best_label ?? 'best'
  const ids = Object.values(cmp.best_registry_ids ?? {}).join(', ')
  const p = snap.current?.progress
  const row = (r: ExploopCompareRow) => (
    <Text key={`ep${r.epoch}`}>
      <Text>{pad(String(r.epoch), 3)}</Text>
      {cell(el, r.wf1, r.d_wf1, 2, true, 9)}
      {cell(el, r.f1_noisy, r.d_f1_noisy, 2, true, 9)}
      {cell(el, r.eer_noisy, r.d_eer_noisy, 2, false, 9)}
      {cell(el, r.dev_loss, r.d_dev_loss, 4, false, 9)}
    </Text>
  )
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="subtle" paddingX={1}>
      <Text wrap="truncate">
        <Text bold>{`${label} vs best ${bl}`}</Text>
        <Text color="subtle">{'   best half_B WF1 (objective, after EVAL): '}</Text>
        <Text bold>{`merge ${fmt(cmp.best_merge)} · final ${fmt(cmp.best_final)}`}</Text>
        <Text color="subtle">{ids ? `  (${ids})` : ''}</Text>
      </Text>
      {p && cmp.target_now !== null && (
        <Text wrap="truncate">
          {`epoch ${cmp.epoch_now} in progress - to beat: ${bl}@ep${cmp.epoch_now} LocalWF1 `}
          <Text bold>{fmt(cmp.target_now)}</Text>
          <Text color="subtle">
            {cmp.best_peak ? `   (${bl} peak ${fmt(cmp.best_peak.wf1)} @ep${cmp.best_peak.epoch})` : ''}
            {cmp.pace_s ? `   pace ${dur(cmp.pace_s)}/epoch` : ''}
            {cmp.best_pace_s ? ` (${bl} ${dur(cmp.best_pace_s)})` : ''}
          </Text>
        </Text>
      )}
      {cmp.rows.length > 0 ? (
        <Box flexDirection="column">
          <Text color="subtle">
            {padEnd(' ep', 3) +
              pad('LocalWF1', 9) + pad(`Δ ${bl}`, 9) +
              pad('F1noisy', 9) + pad('Δ', 9) +
              pad('EERnoisy', 9) + pad('Δ', 9) +
              pad('DevLoss', 9) + pad('Δ', 9)}
          </Text>
          {cmp.rows.map(row)}
        </Box>
      ) : (
        <Text color="subtle">No finished epoch of {label} yet; its LocalWF1 lands at the end of epoch 1.</Text>
      )}
      <Text wrap="truncate">
        <Text color="subtle">LocalWF1 </Text>
        <Text>{`${label} `}</Text>
        <Text color="suggestion">{cmp.spark || '·'}</Text>
        <Text>{`   ${bl} `}</Text>
        <Text color="subtle">{cmp.best_spark || '·'}</Text>
        <Text color={cmp.judged ? (cmp.ahead * 2 >= cmp.judged ? 'success' : 'warning') : 'subtle'}>
          {cmp.judged ? `   ahead of ${bl} on ${cmp.ahead}/${cmp.judged} finished epochs` : ''}
        </Text>
      </Text>
      <Text color="subtle" wrap="truncate">
        LocalWF1 = per-epoch dev mini sets (train.log); not comparable with half_B.
      </Text>
    </Box>
  )
}

function iterationsTable(el: El, snap: ExploopSnapshot) {
  const { Box, Text } = el
  const its = snap.iterations ?? []
  if (!its.length) return null
  return (
    <Box flexDirection="column">
      <Text color="subtle">{' #  run    status    merge       Δ  CI               decision  change'}</Text>
      {its.map(i => {
        const ci = i.ci_low === null || i.ci_high === null ? '–' : `[${sgn(i.ci_low)}, ${sgn(i.ci_high)}]`
        const status = i.done ? 'done' : (i.phase ?? '').toUpperCase()
        return (
          <Text key={`it${i.k}`} wrap="truncate">
            <Text>{pad(String(i.k), 2)}  </Text>
            <Text bold={i.new_best}>{padEnd(i.label ?? '', 6)} </Text>
            <Text color={i.done ? 'success' : 'suggestion'}>{padEnd(status, 9)}</Text>
            <Text>{pad(fmt(i.merge), 6)}</Text>
            <Text color={deltaColor(i.merge_delta)}>{pad(sgn(i.merge_delta), 8)}</Text>
            <Text>{'  ' + padEnd(ci, 17)}</Text>
            <Text color={i.decision === 'promote' ? 'success' : i.decision === 'drop' ? 'error' : 'subtle'}>
              {padEnd(i.decision ?? '–', 10)}
            </Text>
            <Text color="subtle">{i.change ?? ''}</Text>
          </Text>
        )
      })}
    </Box>
  )
}

function eventsList(el: El, snap: ExploopSnapshot, n: number) {
  const { Box, Text } = el
  const evs = (snap.events ?? []).filter(e => e.kind !== 'job_start').slice(-n)
  return (
    <Box flexDirection="column">
      {evs.map(e => (
        <Text key={`ev${e.seq}`} wrap="truncate" color={ALERT_KINDS.has(e.kind) ? 'error' : undefined}>
          <Text color="subtle">{`${(e.ts ?? '').slice(5, 16)} ${e.iter ? `[${labelOf(snap, e.iter)}] ` : ''}`}</Text>
          <Text bold={TOAST_KINDS.has(e.kind)}>{e.kind}</Text>
          <Text>{`: ${e.msg}`}</Text>
        </Text>
      ))}
    </Box>
  )
}

// -------------------------------------------------------------------------------------- hooks

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    timer?.cancel()
    timer = null
    root = await findRoot($, e.cwd)
    if (root) {
      await $.command.register({
        name: 'exploop',
        description: 'Live view of the exploop wave: phase, current epoch vs the best run, eval steps, events',
        argumentHint: '[wave | auto | close]',
        immediate: true,
      })
      timer = $.clock.every(POLL_MS, () => void poll($))
      $.clock.after(1, () =>
        void poll($).then(async () => {
          const snap = await read($, snapAtom)
          if (e.isInteractive && isLive(snap)) $.ui.toast(`${summaryLine(snap, null)} - /exploop for the live view`, { timeoutMs: 8000 })
        }),
      )
    }
    return next(e)
  })

  on('command.run', { command: 'exploop' }, async ($, e) => {
    const arg = e.args.trim()
    if (arg === 'close' || arg === 'off') {
      await $.ui.close({ id: PANE })
      return { text: 'exploop pane closed.' }
    }
    if (arg) {
      await update($, waveAtom, () => (arg === 'auto' ? null : arg))
      await update($, cursorAtom, () => null)
    }
    await poll($)
    const snap = await read($, snapAtom)
    await $.ui.open({ id: PANE, title: snap?.wave ? `exploop ${snap.wave}` : 'exploop' })
    return { text: summaryLine(snap, await read($, errorAtom)) }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const el: El = $.ui.resolve(e)
    const { Box, Text, Button } = el
    const snap = await read($, snapAtom)
    const error = await read($, errorAtom)
    const buttons = (
      <Box gap={2} key="buttons">
        <Button key="refresh" label="Refresh" hotkey="r" onPress={() => poll($)} />
        <Button key="close" label="Close" hotkey="c" role="dismiss" onPress={() => $.ui.close({ id: PANE })} />
        <Text color="subtle">{snap?.taken ? `updated ${snap.taken.slice(11)} · every ${POLL_MS / 1000}s` : ''}</Text>
      </Box>
    )
    if (!snap || !snap.ok) {
      return (
        <Box flexDirection="column">
          <Text color={error ? 'error' : 'subtle'}>{error ?? snap?.error ?? 'Reading the wave...'}</Text>
          <Text color="subtle">/exploop &lt;wave&gt; picks a wave under exploop/; /exploop auto follows the live one.</Text>
          {buttons}
        </Box>
      )
    }
    const rows = e.props.scroll?.bodyRows ?? 40
    return (
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column">
          {header(el, snap)}
          {pipeline(el, snap)}
        </Box>
        {stepCard(el, snap)}
        {compareCard(el, snap)}
        {iterationsTable(el, snap)}
        {eventsList(el, snap, rows > 40 ? 12 : 6)}
        {error && <Text color="warning">{error}</Text>}
        {buttons}
      </Box>
    )
  })
}
