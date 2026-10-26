// rdloop-view: a live view of the running rdloop research cycle (scripts/rdloop) inside Claude Code.
//
// Every 10 s it runs `scripts/rdloop/view.py snapshot` (read-only: never touches the supervisor's lock or
// state) and keeps the JSON in $.state; the /rdloop pane draws from it: the phase strip, what each agent of
// the team is doing (from its live session transcript), the debate rounds and open objections, the levers,
// the regression / smoke checks and the events. Hotkeys 1-6 drill into one agent, d into the debate, o back
// to the overview. New lines of the cycle's events.jsonl become toasts; statuses that need the owner
// (plan gate, final review, needs-user, blocked, supervisor down) become notifications. It does nothing in
// a directory without scripts/rdloop/view.py.
import { atom, read, update } from 'claude-code'
import type { Elements, EngineInterface, Register } from 'claude-code'

import type { RdloopAgent, RdloopEvent, RdloopJob, RdloopSnapshot, RdloopTests } from '../types'

const PANE = 'rdloop'
const POLL_MS = 10_000
const snapAtom = atom({ plugin: 'rdloop-view', key: 'snap' } as const, null)
const cycleAtom = atom({ plugin: 'rdloop-view', key: 'cycle' } as const, null)
const cursorAtom = atom({ plugin: 'rdloop-view', key: 'cursor' } as const, null)
const errorAtom = atom({ plugin: 'rdloop-view', key: 'error' } as const, null)
const focusAtom = atom({ plugin: 'rdloop-view', key: 'focus' } as const, null)

const TOAST_KINDS = new Set(['phase', 'base', 'scout', 'lint_failed', 'critique', 'debate_exhausted', 'plan', 'approved',
  'scaffold', 'implemented', 'lever_dropped', 'review', 'fixed', 'static_start', 'gpu_start', 'compat_start', 'step_failed',
  'container_restart', 'reopened', 'gate_mode', 'docs', 'integrity_warning', 'request_rejected'])
const ALERT_KINDS = new Set(['awaiting_approval', 'awaiting_user', 'needs_user', 'blocked'])
const MAX_TOASTS_PER_POLL = 4
const AGENT_KEYS = ['1', '2', '3', '4', '5', '6']

type El = Pick<Elements['terminal'], 'Box' | 'Text' | 'Button'>

let root: string | null = null
let timer: { cancel: () => void } | null = null
let polling = false

// ------------------------------------------------------------------------------------ formatting

const pad = (s: string, n: number) => (s.length >= n ? s : ' '.repeat(n - s.length) + s)
const padEnd = (s: string, n: number) => (s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length))
const money = (x: number | null | undefined) => (x === null || x === undefined ? '–' : `$${x.toFixed(2)}`)

function dur(s: number | null | undefined): string {
  if (s === null || s === undefined) return '?'
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

function bar(frac: number, width = 16): string {
  const p = Math.max(0, Math.min(1, frac))
  const full = Math.round(p * width)
  return '▕' + '█'.repeat(full) + '░'.repeat(width - full) + '▏'
}

const stateColor = (s: string | null | undefined) =>
  ({ running: 'suggestion', done: 'success', pass: 'success', failed: 'error', fail: 'error', interrupted: 'warning',
     idle: 'inactive', pending: 'inactive' } as Record<string, string>)[s ?? ''] ?? 'subtle'
const stateMark = (s: string | null | undefined) =>
  ({ running: '▶', done: '✓', pass: '✓', failed: '✗', fail: '✗', interrupted: '!', idle: '·', pending: '·' } as Record<string, string>)[s ?? ''] ?? '·'

function isLive(snap: RdloopSnapshot | null): boolean {
  return !!snap?.ok && !['PARKED', 'HANDED_OFF'].includes(snap.status ?? '')
}

function phaseLabel(snap: RdloopSnapshot): string {
  const k = (snap.phase_key ?? '').toUpperCase()
  if (k === 'DEBATE') {
    const r = Math.max(1, (snap.debate_round ?? 0) + (snap.phase === 'critique' ? 0 : 1))
    return `DEBATE r${Math.min(r, snap.debate_max ?? r)}/${snap.debate_max ?? '?'}`
  }
  return k
}

export function summaryLine(snap: RdloopSnapshot | null, error: string | null): string {
  if (!snap) return `rdloop: no snapshot yet${error ? ` (${error})` : ''}`
  if (!snap.ok) return `rdloop ${snap.cycle ?? ''}: ${snap.error ?? 'no live cycle'}`.trim()
  const parts = [`rdloop ${snap.cycle} ${snap.status}`, phaseLabel(snap)]
  const run = (snap.team ?? []).find(t => t.state === 'running')
  if (run) parts.push(`${run.label} ${dur(run.elapsed_s)}${run.last_tool ? ` · last: ${run.last_tool} ${run.last_target ?? ''}` : ''}`)
  if (snap.action) parts.push(`${snap.action.text} -> ${snap.action.command.split('  ')[0]}`)
  if (snap.supervisor?.state === 'down') parts.push('supervisor DOWN')
  parts.push(`${money(snap.cost_usd)} / $${snap.max_total_usd ?? '?'}`)
  return parts.join(' · ')
}

export function eventText(snap: RdloopSnapshot, e: RdloopEvent): string {
  const c = snap.cycle ?? ''
  if (e.kind === 'phase') return `${c}: ${e.msg.replace(' -> ', ' → ')}`
  if (e.kind === 'awaiting_approval') return `${c}: plan ready for you - /rd-cycle approve ${c} (${e.msg.slice(0, 120)})`
  if (e.kind === 'awaiting_user') return `${c}: cycle done - /rd-cycle review ${c}`
  return `${c} ${e.kind.replace(/_/g, ' ')}: ${e.msg.slice(0, 160)}`
}

// ------------------------------------------------------------------------------------- polling

async function alert($: EngineInterface, text: string) {
  $.ui.toast(text, { timeoutMs: 15_000 })
  try {
    await $.ui.notify(text, { title: 'rdloop' })
  } catch {
    // the toast already said it
  }
}

async function announce($: EngineInterface, prev: RdloopSnapshot | null, snap: RdloopSnapshot) {
  const cycle = snap.cycle ?? ''
  const total = snap.events_total ?? 0
  const cur = await read($, cursorAtom)
  if (!cur || cur.cycle !== cycle || total < cur.seq) {
    // first sight of this cycle (or its events were rewritten): baseline, so a restart replays nothing
    await update($, cursorAtom, () => ({ cycle, seq: total }))
    return
  }
  const fresh = (snap.events ?? []).filter(e => e.seq > cur.seq)
  const alerts = fresh.filter(e => ALERT_KINDS.has(e.kind))
  const toasts = fresh.filter(e => TOAST_KINDS.has(e.kind) || (e.kind === 'check' && e.msg.includes('FAIL')))
  for (const e of alerts) await alert($, eventText(snap, e))
  const shown = toasts.slice(-MAX_TOASTS_PER_POLL)
  if (toasts.length > shown.length) $.ui.toast(`${cycle}: ${toasts.length - shown.length} more events (/rdloop)`)
  for (const e of shown) $.ui.toast(eventText(snap, e), { timeoutMs: e.kind === 'phase' ? 6000 : 10_000 })
  if (total !== cur.seq) await update($, cursorAtom, () => ({ cycle, seq: total }))
  const was = prev?.ok && prev.cycle === snap.cycle ? prev.supervisor?.state : undefined
  if (was && was !== 'down' && snap.supervisor?.state === 'down') {
    await alert($, `${cycle}: supervisor DOWN - run scripts/rdloop/rdctl.sh resume ${cycle}`)
  }
}

export async function poll($: EngineInterface): Promise<void> {
  if (!root || polling) return
  polling = true
  try {
    const cycle = await read($, cycleAtom)
    const argv = ['python3', `${root}/scripts/rdloop/view.py`, 'snapshot', ...(cycle ? ['--cycle', cycle] : [])]
    const r = await $.process.run(argv, { cwd: root, timeoutMs: 20_000 })
    let snap: RdloopSnapshot | null = null
    try {
      snap = JSON.parse(r.stdout) as RdloopSnapshot
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
    const hit = await $.fs.stat(`${dir}/scripts/rdloop/view.py`).catch(() => undefined)
    if (hit?.kind === 'file') return dir
    dir = dir.slice(0, dir.lastIndexOf('/'))
  }
  return null
}

// ---------------------------------------------------------------------------------------- pane

function header(el: El, snap: RdloopSnapshot) {
  const { Box, Text } = el
  const sup = snap.supervisor
  const supColor = ({ alive: 'success', stale: 'warning', down: 'error', exited: 'inactive' } as Record<string, string>)[sup?.state ?? ''] ?? 'subtle'
  const statusColor = ({ RUNNING: 'success', AWAITING_USER: 'warning', AWAITING_APPROVAL: 'warning', BLOCKED: 'error',
    NEEDS_USER: 'error', WAITING: 'warning', PAUSED: 'warning', STOPPED: 'warning' } as Record<string, string>)[snap.status ?? ''] ?? 'subtle'
  return (
    <Box flexDirection="column">
      <Text wrap="truncate">
        <Text bold>{`rdloop ${snap.cycle}`}</Text> <Text bold color={statusColor}>{snap.status}</Text>
        <Text color="subtle">{'  supervisor '}</Text>
        <Text color={supColor}>● {sup?.state}</Text>
        <Text color="subtle">
          {sup?.heartbeat_age_s !== null && sup?.heartbeat_age_s !== undefined ? ` (heartbeat ${dur(sup.heartbeat_age_s)} ago)` : ''}
          {`  module ${snap.module ?? '?'} · base ${snap.base?.label ?? '?'} · gate ${snap.gate_mode ?? '?'} · Claude ${money(snap.cost_usd)} / $${snap.max_total_usd ?? '?'}`}
        </Text>
      </Text>
      {sup?.container_restarted && !['AWAITING_USER', 'PARKED', 'HANDED_OFF'].includes(snap.status ?? '') && (
        <Text color="error" bold>Container restarted since the supervisor started: scripts/rdloop/rdctl.sh resume {snap.cycle}</Text>
      )}
      {sup?.state === 'down' && <Text color="error" bold>Supervisor DOWN: scripts/rdloop/rdctl.sh resume {snap.cycle}</Text>}
      {sup?.paused && <Text color="warning">PAUSED: no GPU job starts (rdctl.sh unpause {snap.cycle})</Text>}
    </Box>
  )
}

function phaseStrip(el: El, snap: RdloopSnapshot) {
  const { Text } = el
  const idx = snap.phase_index ?? null
  const phases = snap.phases ?? []
  return (
    <Text wrap="wrap">
      {phases.map((p, i) => (
        <Text key={p} bold={i === idx} color={idx === null ? 'subtle' : i < idx ? 'success' : i === idx ? 'suggestion' : 'inactive'}>
          {(i === idx ? `[${p === 'DEBATE' ? phaseLabel(snap) : p}]` : p) + (i < phases.length - 1 ? ' › ' : '')}
        </Text>
      ))}
    </Text>
  )
}

function actionBanner(el: El, snap: RdloopSnapshot) {
  const { Box, Text } = el
  const out = []
  const a = snap.action
  if (a) {
    const color = a.kind === 'blocked' || a.kind === 'needs_user' ? 'error' : 'warning'
    out.push(
      <Box key="action" flexDirection="column" borderStyle="round" borderColor={color} paddingX={1}>
        <Text bold color={color}>{`⚠ ${snap.status}: ${a.text}`}</Text>
        <Text>{`→ ${a.command}`}</Text>
        <Text color="subtle">{`read: ${a.file}`}</Text>
      </Box>,
    )
  }
  if (snap.waiting) out.push(<Text key="waiting" color="warning">waiting: {snap.waiting}</Text>)
  return out.length ? <Box flexDirection="column">{out}</Box> : null
}

function teamCard(el: El, snap: RdloopSnapshot) {
  const { Box, Text } = el
  const team = snap.team ?? []
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="subtle" paddingX={1}>
      <Text bold>Team</Text>
      {team.map((t, i) => (
        <Box key={t.agent} flexDirection="column">
          <Text wrap="truncate">
            <Text color="subtle">{`${AGENT_KEYS[i] ?? ' '} `}</Text>
            <Text color={stateColor(t.state)}>{`${stateMark(t.state)} `}</Text>
            <Text bold={t.state === 'running'}>{padEnd(t.label, 19)}</Text>
            <Text color={stateColor(t.state)}>{padEnd(t.state, 12)}</Text>
            <Text>{padEnd(t.step ?? '', 18)}</Text>
            <Text color="subtle">{pad(t.elapsed_s !== null ? dur(t.elapsed_s) : '', 5)} {pad(t.cost ? money(t.cost) : '', 7)}  </Text>
            <Text>{t.headline ?? ''}</Text>
          </Text>
          {t.state === 'running' && (
            <Text color="subtle" wrap="truncate">
              {`      ${t.tool_uses ?? 0} tool calls${t.last_tool ? ` · last: ${t.last_tool} ${t.last_target ?? ''}` : t.last_target ? ` · ${t.last_target}` : ''}`}
            </Text>
          )}
          {t.state === 'running' && t.last_text && (
            <Text color="claude" wrap="truncate">{`      “${t.last_text}”`}</Text>
          )}
        </Box>
      ))}
    </Box>
  )
}

function agentDetail(el: El, t: RdloopAgent) {
  const { Box, Text } = el
  const list = (title: string, xs: string[] | undefined) =>
    xs && xs.length ? (
      <Box key={title} flexDirection="column">
        <Text bold>{title}</Text>
        {xs.map((x, i) => <Text key={`${title}${i}`} wrap="truncate" color="subtle">{`  ${x}`}</Text>)}
      </Box>
    ) : null
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="suggestion" paddingX={1}>
      <Text bold>{`${t.agent} - ${t.role}`}</Text>
      <Text>{`state ${t.state} · step ${t.step ?? '–'} · attempt ${t.attempt ?? '–'} · ${t.elapsed_s !== null ? dur(t.elapsed_s) : ''} · ${money(t.cost)}`}</Text>
      {t.headline && <Text>{`result: ${t.headline}`}</Text>}
      {t.session_id && <Text color="subtle">{`claude --resume ${t.session_id}`}</Text>}
      {t.last_text && <Text color="claude" wrap="wrap">{`last said: “${t.last_text}”`}</Text>}
      {list('recent tool calls', (t.recent ?? []).map(r => `${r.tool ?? ''} ${r.target ?? ''}`))}
      {list('web searches', t.queries)}
      {list('pages fetched', t.fetched)}
      {list('files written', t.files_written)}
    </Box>
  )
}

function debateCard(el: El, snap: RdloopSnapshot, full: boolean) {
  const { Box, Text } = el
  const d = snap.debate
  if (!d || (!d.rounds.length && !d.open.length)) return null
  const sev = (s: string) => (s === 'blocker' ? 'error' : s === 'major' ? 'warning' : 'subtle')
  const open = full ? d.open : d.open.slice(0, 4)
  return (
    <Box flexDirection="column" borderStyle="round" borderColor={full ? 'suggestion' : 'subtle'} paddingX={1}>
      <Text bold>
        {`Debate · proposal v${d.version} · ${d.rounds.length} critique round(s) of ${snap.debate_max ?? '?'}`}
        {d.outcome ? ` · ${d.outcome}` : ''}
        {d.lint_failures ? ` · ${d.lint_failures} lint failure(s)` : ''}
      </Text>
      <Text color="subtle">{' round  levers                         blocker major minor resolved  critic'}</Text>
      {d.rounds.map(r => (
        <Text key={`v${r.v}`} wrap="truncate">
          <Text>{padEnd(` v${r.v}`, 7)}</Text>
          <Text>{padEnd(r.levers.join(', '), 31)}</Text>
          <Text color={(r.counts.blocker ?? 0) > 0 ? 'error' : 'subtle'}>{pad(String(r.counts.blocker ?? 0), 7)}</Text>
          <Text color={(r.counts.major ?? 0) > 0 ? 'warning' : 'subtle'}>{pad(String(r.counts.major ?? 0), 6)}</Text>
          <Text color="subtle">{pad(String(r.counts.minor ?? 0), 6)}{pad(String(r.counts.resolved ?? 0), 9)}  </Text>
          <Text color={r.agreed ? 'success' : 'warning'}>{r.agreed ? 'agreed' : r.critic_verdict ?? 'revise'}</Text>
        </Text>
      ))}
      {open.length > 0 && <Text bold>open objections</Text>}
      {open.map(o => (
        <Text key={o.id} wrap={full ? 'wrap' : 'truncate'}>
          <Text color={sev(o.severity)}>{`  ${o.id} [${o.severity}] `}</Text>
          <Text color="subtle">{`${o.lever_id}: `}</Text>
          <Text>{o.claim}</Text>
          {full && o.required_change ? <Text color="subtle">{` → ${o.required_change}`}</Text> : null}
        </Text>
      ))}
      {!full && d.open.length > open.length && <Text color="subtle">{`  +${d.open.length - open.length} more (d)`}</Text>}
      {full && d.responses.length > 0 && <Text bold>architect's responses (latest proposal)</Text>}
      {full && d.responses.map(r => (
        <Text key={`r${r.id}`} wrap="wrap" color="subtle">{`  ${r.id}: ${r.action} - ${r.detail}`}</Text>
      ))}
    </Box>
  )
}

function leversCard(el: El, snap: RdloopSnapshot) {
  const { Box, Text } = el
  const lv = snap.levers ?? []
  if (!lv.length) return null
  const color = (s: string) => ({ implemented: 'success', approved: 'suggestion', implementing: 'suggestion', dropped: 'error',
    contested: 'warning' } as Record<string, string>)[s] ?? 'subtle'
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="subtle" paddingX={1}>
      <Text bold>Levers</Text>
      {lv.map(l => (
        <Text key={l.id} wrap="truncate">
          <Text bold>{padEnd(l.id, 18)}</Text>
          <Text color={color(l.status)}>{padEnd(l.status, 13)}</Text>
          <Text color="subtle">{padEnd(l.flags.join(' '), 34)}</Text>
          <Text color="subtle">{padEnd(l.gap_ids.join(','), 8)}</Text>
          {l.smoke_state && (
            <Text color={stateColor(l.smoke_state)}>
              {`smoke ${stateMark(l.smoke_state)}${l.time_frac ? ` ${l.time_frac.toFixed(2)}x` : ''}${l.peak_gib ? ` ${l.peak_gib} GiB` : ''}${l.eta_full_h ? ` ETA ${l.eta_full_h} h` : ''}`}
            </Text>
          )}
          {l.dropped_reason && <Text color="error">{` ${l.dropped_reason}`}</Text>}
        </Text>
      ))}
    </Box>
  )
}

function testsCard(el: El, snap: RdloopSnapshot) {
  const { Box, Text } = el
  const t = snap.tests ?? {}
  const blocks: [string, RdloopTests | undefined][] = [['STATIC', t.static], ['GPU (regression + smoke)', t.gpu], ['COMPAT (exploop dry run)', t.compat]]
  const present = blocks.filter(([, b]) => b && b.jobs.length)
  if (!present.length) return null
  const row = (j: RdloopJob) => (
    <Text key={j.name} wrap="truncate">
      <Text color={stateColor(j.state)}>{`  ${stateMark(j.state)} `}</Text>
      <Text>{padEnd(j.name, 24)}</Text>
      <Text color="subtle">{pad(j.elapsed_s !== null ? dur(j.elapsed_s) : '', 5)}{j.budget_s ? `/${dur(j.budget_s)}` : ''}  </Text>
      <Text color={j.state === 'fail' ? 'error' : 'subtle'}>{j.note}</Text>
    </Text>
  )
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="subtle" paddingX={1}>
      {present.map(([title, b]) => (
        <Box key={title} flexDirection="column">
          <Text bold>
            {`${title}${b!.done ? (b!.ok ? ' - passed' : ' - FAILED') : ''}`}
            <Text color="subtle">{` · code ${b!.code_sha}`}</Text>
            {title.startsWith('GPU') && (
              <Text color="subtle">{`  smoke ${bar(b!.smoke_spent_s / Math.max(1, b!.smoke_budget_s))} ${dur(b!.smoke_spent_s)} / ${dur(b!.smoke_budget_s)}`}</Text>
            )}
          </Text>
          {b!.jobs.map(row)}
        </Box>
      ))}
    </Box>
  )
}

function eventsList(el: El, snap: RdloopSnapshot, n: number) {
  const { Box, Text } = el
  const evs = (snap.events ?? []).slice(-n)
  return (
    <Box flexDirection="column">
      {evs.map(e => (
        <Text key={`ev${e.seq}`} wrap="truncate" color={ALERT_KINDS.has(e.kind) ? 'error' : undefined}>
          <Text color="subtle">{`${(e.ts ?? '').slice(5, 16)} `}</Text>
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
        name: 'rdloop',
        description: 'Live view of the rdloop research cycle: phase, what each agent is doing, debate, levers, tests, events',
        argumentHint: '[cycle | auto | close]',
        immediate: true,
      })
      timer = $.clock.every(POLL_MS, () => void poll($))
      $.clock.after(1, () =>
        void poll($).then(async () => {
          const snap = await read($, snapAtom)
          if (e.isInteractive && isLive(snap)) $.ui.toast(`${summaryLine(snap, null)} - /rdloop for the live view`, { timeoutMs: 8000 })
        }),
      )
    }
    return next(e)
  })

  on('command.run', { command: 'rdloop' }, async ($, e) => {
    const arg = e.args.trim()
    if (arg === 'close' || arg === 'off') {
      await $.ui.close({ id: PANE })
      return { text: 'rdloop pane closed.' }
    }
    if (arg) {
      await update($, cycleAtom, () => (arg === 'auto' ? null : arg))
      await update($, cursorAtom, () => null)
    }
    await poll($)
    const snap = await read($, snapAtom)
    await $.ui.open({ id: PANE, title: snap?.cycle ? `rdloop ${snap.cycle}` : 'rdloop' })
    return { text: summaryLine(snap, await read($, errorAtom)) }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const el: El = $.ui.resolve(e)
    const { Box, Text, Button } = el
    const snap = await read($, snapAtom)
    const error = await read($, errorAtom)
    const focus = await read($, focusAtom)
    const team = snap?.team ?? []
    const setFocus = (f: string | null) => update($, focusAtom, () => f)
    const buttons = (
      <Box gap={2} key="buttons" flexWrap="wrap">
        {team.map((t, i) => (
          <Button key={`a${t.agent}`} label={t.label.split(' ')[0]!} hotkey={AGENT_KEYS[i]} plain onPress={() => setFocus(t.agent)} />
        ))}
        <Button key="debate" label="debate" hotkey="d" plain onPress={() => setFocus('debate')} />
        <Button key="overview" label="overview" hotkey="o" plain onPress={() => setFocus(null)} />
        <Button key="refresh" label="Refresh" hotkey="r" onPress={() => poll($)} />
        <Button key="close" label="Close" hotkey="c" role="dismiss" onPress={() => $.ui.close({ id: PANE })} />
        <Text color="subtle">{snap?.taken ? `updated ${snap.taken.slice(11)} · every ${POLL_MS / 1000}s` : ''}</Text>
      </Box>
    )
    if (!snap || !snap.ok) {
      return (
        <Box flexDirection="column">
          <Text color={error ? 'error' : 'subtle'}>{error ?? snap?.error ?? 'Reading the cycle...'}</Text>
          <Text color="subtle">/rdloop &lt;cycle&gt; picks a cycle under rdloop/; /rdloop auto follows the live one.</Text>
          {buttons}
        </Box>
      )
    }
    const rows = e.props.scroll?.bodyRows ?? 40
    const focused = focus && focus !== 'debate' ? team.find(t => t.agent === focus) : undefined
    return (
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column">
          {header(el, snap)}
          {phaseStrip(el, snap)}
        </Box>
        {actionBanner(el, snap)}
        {focused ? agentDetail(el, focused) : null}
        {focus === 'debate' ? debateCard(el, snap, true) : null}
        {teamCard(el, snap)}
        {focus !== 'debate' ? debateCard(el, snap, false) : null}
        {leversCard(el, snap)}
        {testsCard(el, snap)}
        {eventsList(el, snap, rows > 40 ? 12 : 6)}
        {error && <Text color="warning">{error}</Text>}
        {buttons}
      </Box>
    )
  })
}
