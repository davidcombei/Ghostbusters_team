import { describe, expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import type { ExploopEvent, ExploopSnapshot } from '../types'

const ROOT = '/repo'

function snapshot(events: ExploopEvent[], extra: Partial<ExploopSnapshot> = {}): ExploopSnapshot {
  return {
    ok: true, wave: 'B3', error: null, taken: '2026-10-10 15:40:00', status: 'RUNNING', waiting: null, blocked: null,
    max_iterations: 5, phases: ['design', 'render', 'train', 'eval', 'verdict', 'log', 'analyze', 'decide'],
    cost_usd: 5.65, max_total_usd: 150,
    best: { label: 'B3a', merge_value: 93.056, final_value: 92.94 },
    supervisor: { state: 'alive', pid: 1, heartbeat_age_s: 5, container_restarted: false, paused: false, stop_requested: false },
    events_total: events.length ? events[events.length - 1]!.seq : 0, events,
    iterations: [
      { k: 1, label: 'B3a', change: '+ --consistency js', phase: 'decide', done: true, merge: 93.056, merge_delta: 1.08,
        ci_low: 0.5, ci_high: 1.6, final: 92.94, decision: 'promote', new_best: true },
      { k: 2, label: 'B3b', change: '+ --sampler balanced', phase: 'train', done: false, merge: null, merge_delta: null,
        ci_low: null, ci_high: null, final: null, decision: null, new_best: false },
    ],
    current: {
      k: 2, label: 'B3b', phase: 'train', phase_index: 2, change: '+ --sampler balanced', run_name: 'B3b_run', num_epochs: 8,
      progress: { epoch: 3, num_epochs: 8, epochs_done: 2, substep: 'train', i: 3753, n: 9474, pct: 40, rate_per_s: 2.86,
                  epoch_s: 3496, remaining_s: 1998, eta_h: 5.6, log_age_s: 1 },
    },
    compare: {
      best_label: 'B3a', best_merge: 93.056, best_final: 92.94, best_registry_ids: { merge: 'B3a-m' },
      rows: [
        { epoch: 1, wf1: 90.6, d_wf1: 0.43, best_wf1: 90.17, f1_noisy: 87.9, d_f1_noisy: 0.62, eer_noisy: 12.8, d_eer_noisy: -0.34, dev_loss: 0.095, d_dev_loss: -0.004 },
        { epoch: 2, wf1: 91.2, d_wf1: -0.44, best_wf1: 91.64, f1_noisy: 88.7, d_f1_noisy: -0.54, eer_noisy: 11.9, d_eer_noisy: 0.34, dev_loss: 0.118, d_dev_loss: -0.005 },
      ],
      last: null, epoch_now: 3, num_epochs: 8, target_now: 91.39, best_peak: { epoch: 6, wf1: 92.87 },
      current_peak: { epoch: 2, wf1: 91.2 }, ahead: 1, judged: 2, pace_s: 3540, best_pace_s: 3534, spark: '▃▆', best_spark: '▃▇▆▆▇█▇▆',
    },
    ...extra,
  }
}

const ev = (seq: number, kind: string, msg: string, data?: ExploopEvent['data']): ExploopEvent =>
  ({ seq, ts: '2026-10-10 15:40:00', iter: 2, phase: 'train', kind, msg, ...(data ? { data } : {}) })

function world(on: On, snaps: ExploopSnapshot[], { inRepo = true } = {}) {
  const seen = { toasts: [] as string[], notes: [] as string[], opened: [] as string[], commands: [] as string[], runs: 0 }
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('fs.stat', (_$, e) => {
    if (inRepo && e.path === `${ROOT}/scripts/exploop/view.py`) return { value: { kind: 'file', size: 1, mtimeMs: 0, isLink: false } }
    return { deny: `ENOENT ${e.path}` }
  })
  on('process.run', (_$, e) => {
    expect(e.argv.slice(1, 3)).toEqual([`${ROOT}/scripts/exploop/view.py`, 'snapshot'])
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

describe('exploop-view', () => {
  test('does nothing outside the repo', async ($, on) => {
    const { seen, clock } = world(on, [snapshot([])], { inRepo: false })
    await $.session.start({ cwd: '/elsewhere', surface: 'terminal', isInteractive: true })
    await clock.advance(30_000)
    expect(seen.commands).toEqual([])
    expect(seen.runs).toBe(0)
  })

  test('first poll is a baseline; new events become toasts, BLOCKED a notification', async ($, on) => {
    const old = [ev(1, 'phase', 'B3b -> TRAIN'), ev(2, 'launch', 'bash scripts/run_B3b.sh')]
    const next = [...old, ev(3, 'epoch', 'B3b epoch 3/8', { epoch: 3, wf1: 91.5, delta_vs_best: 0.11 }), ev(4, 'phase', 'B3b -> EVAL')]
    const blocked = [...next, ev(5, 'blocked', 'eval: merge_eval failed 2 times')]
    const { seen, clock } = world(on, [snapshot(old), snapshot(next), snapshot(blocked, { status: 'BLOCKED' })])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    expect(seen.commands).toEqual(['exploop'])
    await clock.advance(1)
    expect(seen.runs).toBe(1)
    expect(seen.toasts).toHaveLength(1)                       // the intro line only, no replay of old events
    expect(seen.toasts[0]).toContain('/exploop')
    await clock.advance(10_000)
    expect(seen.toasts.slice(1)).toEqual(['B3 B3b epoch 3: LocalWF1 91.50 (+0.11 vs B3a@3)', 'B3: B3b → EVAL'])
    expect(seen.notes).toEqual([])
    await clock.advance(10_000)
    expect(seen.notes).toEqual(['B3 blocked: eval: merge_eval failed 2 times'])
  })

  test('supervisor going down raises a notification once', async ($, on) => {
    const down = snapshot([], { supervisor: { state: 'down', pid: 1, heartbeat_age_s: 900, container_restarted: false, paused: false, stop_requested: false } })
    const { seen, clock } = world(on, [snapshot([]), down, down])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    await clock.advance(1)
    await clock.advance(10_000)
    await clock.advance(10_000)
    expect(seen.notes).toEqual(['B3: supervisor DOWN - run scripts/exploop/loopctl.sh resume B3'])
  })

  test('/exploop opens the pane, which shows the phase and the current epoch vs the best', async ($, on) => {
    const { seen } = world(on, [snapshot([ev(1, 'phase', 'B3b -> TRAIN')])])
    await $.session.start({ cwd: ROOT, surface: 'terminal', isInteractive: true })
    for (const surface of ['terminal', 'desktop'] as const) {
      const out = await $.command.run({ command: 'exploop', args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 160 } } as never)
      expect(out.text).toContain('iteration 2/5 B3b phase TRAIN')
      expect(seen.opened).toContain('exploop')
      const ui = await $.ui.mount({
        plugin: 'exploop-view', surface, component: 'Pane', requestId: 'exploop',
        props: { title: 'exploop B3', isFocused: false, bodyColumns: 140, placement: 'dock', scroll: { offset: 0, bodyRows: 60 }, view: {} },
      })
      expect(await ui.find({ text: '[TRAIN]' })).toBeDefined()
      expect(await ui.find({ text: /B3b vs best B3a/ })).toBeDefined()
      expect(await ui.find({ text: /to beat: B3a@ep3 LocalWF1 91\.39/ })).toBeDefined()
      expect(await ui.find({ text: /merge 93\.06 · final 92\.94/ })).toBeDefined()
      expect(await ui.find({ text: /epoch 3\/8/ })).toBeDefined()
      expect(await ui.find({ text: '-0.44' })).toBeDefined()
      expect(await ui.find({ text: /ahead of B3a on 1\/2 finished epochs/ })).toBeDefined()
      expect(await ui.findAll({ type: 'Button' })).toHaveLength(2)
      await ui.unmount()
    }
  })
})
