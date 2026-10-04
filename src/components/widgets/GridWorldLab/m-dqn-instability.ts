/**
 * dqn-instability (deep-rl step 3): the same 1,000 random-walk transitions, the same start weights and the same Adam
 * learning rate train a network two ways: towards its own bootstrapped target (no replay, no target network) and
 * towards fixed answers Q* from value iteration. Two log charts: largest |Q| anywhere and the Bellman error.
 * Measured (plans/day-11.md §6, init seed 17, stream seed 17): Adam 0.1 moving target peaks at |Q| 218, fixed fit stays at 5.3.
 * Props: seed (17), lr (0.1 or 0.01, a control), updates (1000), hidden (16).
 */
import { ACTIONS, cellName, makeGrid } from '../../../lib/rl/gridworld';
import { randomStream, type DQNResult } from '../../../lib/rl/dqn';
import { optimal } from '../../../lib/rl/tabular';
import { int, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, fmt, line, lineChart, text, type Env } from '../_rl/svg';
import { DIAG, GAMMA, dqnGrid } from './dqn-common';

// time -> update: the first 50 updates (where the run-away starts) get 3 s, updates 50-300 get 4 s, the rest 5 s
const KNOTS: [number, number][] = [[0, 0], [3, 50], [7, 300], [12, 1000]], HOLD = 2;
interface Scene { xs: number[]; moving: DQNResult; fixed: DQNResult; n: number; qmax: number; lr: number; ymax: number; bmax: number }
interface F extends Frame { k: number }
const kAt = (t: number, n: number) => {
  for (let i = 1; i < KNOTS.length; i++) if (t <= KNOTS[i][0]) { const [t0, k0] = KNOTS[i - 1], [t1, k1] = KNOTS[i]; return Math.max(1, Math.round(((k0 + ((t - t0) / (t1 - t0)) * (k1 - k0)) * n) / 1000)); }
  return n;
};
const up10 = (v: number) => 10 ** Math.ceil(Math.log10(v));

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit: Kit) {
    const g = dqnGrid(o), seed = int(o.seed, 17), n = int(o.updates, 1000), lr = num(o.lr, 0.1), stream = randomStream(g, n, seed);
    const base = { optimizer: 'adam', lr, gamma: GAMMA, hidden: int(o.hidden, 16), replay: false, target: false, seed, stream, diag: DIAG, updates: n, evalEvery: 100000 };
    const [moving, fixed] = await Promise.all([kit.job<DQNResult>('trainDQN', g, base), kit.job<DQNResult>('trainDQN', g, { ...base, targetKind: 'qstar' })]);
    const Q = optimal({ ...makeGrid({ gamma: GAMMA }) }, GAMMA).Q, qmax = Math.max(...Array.from(Q, Math.abs));
    const hi = (r: DQNResult, f: keyof DQNResult) => Math.max(...Array.from(r[f] as Float64Array));
    return { xs: Array.from({ length: n }, (_, i) => i + 1), moving, fixed, n, qmax, lr, ymax: up10(Math.max(hi(moving, 'maxAbsQ'), hi(fixed, 'maxAbsQ'))), bmax: up10(Math.max(hi(moving, 'bellman'), hi(fixed, 'bellman'))) };
  },
  timeline: (s) => ({ duration: KNOTS[KNOTS.length - 1][0] + HOLD, markers: [{ at: 0, label: 'Updates 1–50' }, { at: 3, label: 'Updates 51–300' }, { at: 7, label: 'Updates 301–1000' }] }),
  frameAt(t, s) {
    const k = kAt(Math.min(t, KNOTS[KNOTS.length - 1][0]), s.n), i = k - 1, m = s.moving, x = s.fixed;
    const blow = s.lr >= 0.1;
    return {
      k,
      status: `After ${k} update${k > 1 ? 's' : ''}, the network that chases its own target has a largest |Q| of ${fmt(m.maxAbsQ[i])} and a Bellman error of ${fmt(m.bellman[i])}; the one fitting fixed answers has ${fmt(x.maxAbsQ[i])} and ${fmt(x.bellman[i])}. The true largest Q is ${fmt(s.qmax)}.`,
      detail: k >= s.n ? (blow ? 'Same data, same optimiser and learning rate: only the answers being fitted differ.' : 'At this smaller rate the moving target drifts, but it stays bounded.') : undefined,
      readouts: [['Update', `${k} of ${s.n}`], [`Target y for ${cellName(DIAG.s)} ${ACTIONS[DIAG.a]}`, fmt(m.diagY[i])], ['Own target: largest |Q|', fmt(m.maxAbsQ[i])], ['Fixed answers: largest |Q|', fmt(x.maxAbsQ[i])]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, cw = nar ? E.W - 56 : (E.W - 126) / 2, ch = nar ? (E.H - 150) / 2 : E.H - 90;
    const boxes = nar ? [{ x: 44, y: 30, w: cw, h: ch }, { x: 44, y: 30 + ch + 74, w: cw, h: ch }] : [{ x: 46, y: 34, w: cw, h: ch }, { x: 46 + cw + 56, y: 34, w: cw, h: ch }];
    const xa = { lo: 0, hi: s.n, ticks: [0, s.n / 2, s.n], label: 'Update' };
    const ser = (a: DQNResult, b: DQNResult, fld: 'maxAbsQ' | 'bellman') => [
      { ys: a[fld], xs: s.xs, color: 'var(--viz-cat-2)', width: 2.2 },
      { ys: b[fld], xs: s.xs, color: 'var(--viz-cat-1)', width: 2.2 },
    ];
    const ya1 = { lo: 0.1, hi: s.ymax, log: true, label: 'Largest |Q| over all cells and actions' }, ya2 = { lo: 0.1, hi: s.bmax, log: true, label: 'Bellman error' };
    const c1 = lineChart(E, boxes[0], xa, ya1, ser(s.moving, s.fixed, 'maxAbsQ'), f.k);
    const c2 = lineChart(E, boxes[1], xa, ya2, ser(s.moving, s.fixed, 'bellman'), f.k);
    const qy = c1.sy(s.qmax);
    line(E, boxes[0].x, qy, boxes[0].x + boxes[0].w, qy, { stroke: MUTED, sw: 1.25, dash: '5 4' });
    text(E, `true largest ${fmt(s.qmax)}`, boxes[0].x + boxes[0].w - 4, qy - 5, { anchor: 'end', size: 11, fill: MUTED });
    for (const [b, c] of [[boxes[0], c1], [boxes[1], c2]] as const) { const px = c.sx(f.k); line(E, px, b.y, px, b.y + b.h, { stroke: TEXT, sw: 1, op: 0.45 }); }
  },
  label: (s) => `Two log-scale charts over ${s.n} updates comparing a network trained on its own bootstrapped targets with one fitting fixed answers: the largest Q-value and the Bellman error.`,
};
export default mode;
