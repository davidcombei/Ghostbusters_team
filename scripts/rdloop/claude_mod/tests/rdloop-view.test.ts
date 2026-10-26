import { describe, expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import type { RdloopEvent, RdloopSnapshot } from '../types'

const ROOT = '/repo'

const agent = (agent: string, label: string, extra: object = {}) => ({
  agent, label, role: `${label} role`, state: 'idle', step: null, headline: null, cost: 0, attempt: null, session_id: null,
  elapsed_s: null, ...extra,
})

function snapshot(events: RdloopEvent[], extra: Partial<RdloopSnapshot> = {}): RdloopSnapshot {
  return {
    ok: true, cycle: 'B4', error: null, taken: '2026-10-12 10:00:00', status: 'RUNNING', phase: 'critique', phase_key: 'debate',
    phase_index: 3, phases: ['WAIT', 'AUDIT', 'SCOUT', 'DEBATE', 'GATE', 'IMPLEMENT', 'STATIC', 'REVIEW', 'GPU', 'COMPAT', 'SUMMARY', 'REVIEW-BY-OWNER'],
    gate_mode: 'owner', module: 'b4wave', base: { label: 'B3a', run_dir: 'SLS_setup/exp/B3a', from_wave: 'B3' }, cost_usd: 41.2,
    max_total_usd: 150, debate_round: 1, debate_max: 3, action: null, waiting: null, blocked: null, needs_user: [], next: null,
    supervisor: { state: 'alive', pid: 1, heartbeat_age_s: 4, container_restarted: false, paused: false, stop_requested: false },
    team: [
      agent('research-auditor', 'auditor', { state: 'done', step: 'audit a', headline: '5 gaps · top: noisy bona FA', cost: 3.1 }),
      agent('research-scout', 'scout', { state: 'done', step: 'scout a', headline: '9 ideas · 14/15 sources fetched', cost: 8.4,
        queries: ['codec robust anti-spoofing'], fetched: ['https://arxiv.org/abs/2604.23742'] }),
      agent('research-architect', 'architect', { state: 'done', step: 'propose v2', headline: 'proposal v2: 3 levers', cost: 6 }),
      agent('research-critic', 'critic', { state: 'running', step: 'critique v2', elapsed_s: 250, tool_uses: 23, last_tool: 'WebFetch',
        last_target: 'https://arxiv.org/abs/2406.0001', last_text: 'Checking whether the PLC paper reports EER on Opus.', session_id: 'sid-1',
        recent: [{ tool: 'Read', target: 'rdloop/B4/debate/proposal_v2.md', ts: 't' }] }),
      agent('research-engineer', 'engineer'),
      agent('supervisor', 'supervisor (tests)'),
    ],
    debate: {
      version: 2, outcome: null, lint_failures: 0, levers_in_play: ['aug_plc', 'head_asp', 'loss_ocs'],
      rounds: [{ v: 1, levers: ['aug_plc', 'head_asp', 'loss_ocs'], counts: { blocker: 1, major: 2, minor: 1, resolved: 0 }, agreed: false, critic_verdict: 'revise' }],
      open: [{ id: 'O1.2', severity: 'major', lever_id: 'head_asp', claim: 'ASP changes the state_dict without an eval path', required_change: 'add the eval script' }],
      counts: { blocker: 0, major: 1, minor: 1, resolved: 2 },
      responses: [{ id: 'O1.1', action: 'accept', detail: 'dropped the reserved G.726 profile' }],
    },
    levers: [{ id: 'aug_plc', title: 'packet loss', status: 'proposed', flags: ['--aug_plc=False'], gap_ids: ['G1'], dropped_reason: null,
               smoke_state: null, smoke_note: null, peak_gib: null, time_frac: null, eta_full_h: null }],
    tests: {}, events_total: events.length ? events[events.length - 1]!.seq : 0, events, ...extra,
  }
}

const ev = (seq: number, kind: string, msg: string): RdloopEvent => ({ seq, ts: '2026-10-12 10:00:00', phase: 'critique', kind, msg })

function world(on: On, snaps: RdloopSnapshot[], { inRepo = true } = {}) {
  const seen = { toasts: [] as string[], notes: [] as string[], opened: [] as string[], commands: [] as string[], runs: 0 }
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('fs.stat', (_$, e) => {
    if (inRepo && e.path === `${ROOT}/scripts/rdloop/view.py`) return { value: { kind: 'file', size: 1, mtimeMs: 0, isLink: false } }
    return { deny: `ENOENT ${e.path}` }
  })
  on('process.run', (_$, e) => {
    expect(e.argv.slice(1, 3)).toEqual([`${ROOT}/scripts/rdloop/view.py`, 'snapshot'])
    const s = snaps[Math.min(seen.runs, snaps.length - 1)]!
    seen.runs += 1
    return { value: { exitCode: 0, stdout: JSON.stringify(s), stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('command.register', (_$, e) => {
    seen.commands.push(e.name)
    return { value: { command: e.name } }
  })
  on('ui.toast', (_$, e) => {
    seen.toasts.push(e.text)
    return { value: undefined }
  })
  on('ui.notify', (_$, e) => {
    seen.notes.push(e.text)
    return { value: { isSent: true, channel: 'terminal_bell' } }
  })
  on('ui.open', (_$, e) => {
    seen.opened.push(e.id)
    return { value: { isPlaced: true } }
  })
  on('ui.close', () => ({ value: undefined }))
  const clock = mock.clock(on)
  return { seen, clock }
}

const mount = ($: Parameters<Parameters<typeof test>[1]>[0], surface: 'terminal' | 'desktop' = 'terminal') =>
  $.ui.mount({
    plugin: 'rdloop-view', surface, component: 'Pane', requestId: 'rdloop',
    props: { title: 'rdloop B4', isFocused: true, bodyColumns: 150, placement: 'dock', scroll: { offset: 0, bodyRows: 70 }, view: {} },
  })

describe('rdloop-view', () => {
  test('does nothing outside the repo', async ($, on) => {
    const { seen, clock } = world(on, [snapshot([])], { inRepo: false })
    await $.session.start({ cwd: '/elsewhere', surface: 'terminal', isInteractive: true })
    await clock.advance(30_000)
    expect(seen.commands).toEqual([])
    expect(seen.runs).toBe(0)
  })

  test('first poll is a baseline; new events toast; the plan gate notifies', async ($, on) => {
    const old = [ev(1, 'phase', 'SCOUT -> PROPOSE'), ev(2, 'critique', 'v1: 1 blocker / 2 major / 1 minor open; revise')]
    const next = [...old, ev(3, 'phase', 'CRITIQUE -> PROPOSE (revise)'), ev(4, 'check', 'gpu S_aug_plc: PASS ok')]
    const gate = [...next, ev(5, 'awaiting_approval', "plan ready for the owner: levers ['aug_plc']")]
    const { seen, clock } = world(on, [snapshot(old), snapshot(next), snapshot(gate, { status: 'AWAITING_APPROVAL' })])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    expect(seen.commands).toEqual(['rdloop'])
    await clock.advance(1)
    expect(seen.toasts).toHaveLength(1)                        // the intro line only, no replay
    expect(seen.toasts[0]).toContain('/rdloop')
    await clock.advance(10_000)
    expect(seen.toasts.slice(1)).toEqual(['B4: CRITIQUE → PROPOSE (revise)'])   // a passing check is not toasted
    expect(seen.notes).toEqual([])
    await clock.advance(10_000)
    expect(seen.notes).toEqual(['B4: plan ready for you - /rd-cycle approve B4 (plan ready for the owner: levers [\'aug_plc\'])'])
  })

  test('supervisor going down notifies once', async ($, on) => {
    const down = snapshot([], { supervisor: { state: 'down', pid: 1, heartbeat_age_s: 900, container_restarted: false, paused: false, stop_requested: false } })
    const { seen, clock } = world(on, [snapshot([]), down, down])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    await clock.advance(1)
    await clock.advance(10_000)
    await clock.advance(10_000)
    expect(seen.notes).toEqual(['B4: supervisor DOWN - run scripts/rdloop/rdctl.sh resume B4'])
  })

  test('/rdloop opens the pane: phase strip, who is doing what, debate, drill-down per agent and into the debate', async ($, on) => {
    const { seen } = world(on, [snapshot([ev(1, 'phase', 'PROPOSE -> CRITIQUE')])])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    const out = await $.command.run({ command: 'rdloop', args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 160 } } as never)
    expect(out.text).toContain('rdloop B4 RUNNING · DEBATE r1/3 · critic 4m')
    expect(seen.opened).toContain('rdloop')
    const ui = await mount($)
    expect(await ui.find({ text: '[DEBATE r1/3]' })).toBeDefined()
    expect(await ui.find({ text: /23 tool calls · last: WebFetch/ })).toBeDefined()
    expect(await ui.find({ text: /Checking whether the PLC paper/ })).toBeDefined()
    expect(await ui.find({ text: /O1\.2 \[major\]/ })).toBeDefined()
    expect(await ui.find({ text: /9 ideas/ })).toBeDefined()
    await ui.press({ key: 'aresearch-scout' })
    expect(await ui.find({ text: /research-scout - scout role/ })).toBeDefined()
    expect(await ui.find({ text: /codec robust anti-spoofing/ })).toBeDefined()
    await ui.press({ key: 'debate' })
    expect(await ui.find({ text: /dropped the reserved G\.726 profile/ })).toBeDefined()
    await ui.unmount()
  })

  test('owner action banner', async ($, on) => {
    const snap = snapshot([], { status: 'AWAITING_USER', action: { kind: 'review', text: 'cycle finished: implemented, regression + smoke passed',
      command: '/rd-cycle review B4', file: 'rdloop/B4/FINAL_SUMMARY.md' } })
    world(on, [snap])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    await $.command.run({ command: 'rdloop', args: 'B4', origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 160 } } as never)
    for (const surface of ['terminal', 'desktop'] as const) {
      const ui = await mount($, surface)
      expect(await ui.find({ text: /⚠ AWAITING_USER: cycle finished/ })).toBeDefined()
      expect(await ui.find({ text: /→ \/rd-cycle review B4/ })).toBeDefined()
      await ui.unmount()
    }
  })
})
