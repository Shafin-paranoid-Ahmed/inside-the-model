/**
 * dqn-compare (deep-rl step 6): the four DQN variants (neither, replay only, target only, both) trained for 500 episodes
 * with 10 runs each, sharing initial weights and environment seeds run by run. Curves are the exact return of the greedy
 * policy from the start cell (no exploration), measured every 10 episodes. Runs in the worker (4-9 s).
 * Measured (plans/day-11.md §6, Adam, V* 0.351), final mean (sd): lr 0.001 neither 0.260 (0.220), replay 0.299 (0.102),
 * target 0.285 (0.139), both 0.331 (0.019); lr 0.01 neither 0.331 (0.014), replay 0.183, target 0.279, both 0.190.
 * Props: lr (0.001, a control), measure (return | error), episodes (500), runs (10), seed (23), epsilon (0.1).
 */
import { int, num, str, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, fmt, line, lineChart, rect, text, type Env, type Series } from '../_rl/svg';
import { optimal } from '../../../lib/rl/tabular';
import { GAMMA, VARIANT_COLOR, VARIANT_NAME, dqnGrid } from './dqn-common';

const PLAY = 10, HOLD = 2.5, VARIANTS = ['naive', 'replay', 'target', 'both'] as const;
type Measure = 'return' | 'error';
interface Curve { variant: string; evalAt: Int32Array; evalReturn: Float64Array[]; qErr: Float64Array[] }
interface Stat { mean: Float64Array; sd: Float64Array; runs: Float64Array[] }
interface Scene { stats: Record<Measure, Record<string, Stat>>; at: Int32Array; episodes: number; runs: number; vstar: number; measure: Measure; lr: number }
interface F extends Frame { ep: number; i: number }

const statOf = (runs: Float64Array[]): Stat => {
  const n = runs[0].length, mean = new Float64Array(n), sd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (const r of runs) m += r[i];
    m /= runs.length;
    let v = 0;
    for (const r of runs) v += (r[i] - m) ** 2;
    mean[i] = m; sd[i] = Math.sqrt(v / runs.length);
  }
  return { mean, sd, runs };
};

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit: Kit) {
    const g = dqnGrid(o), episodes = int(o.episodes, 500), runs = int(o.runs, 10), lr = num(o.lr, 0.001);
    const res = await kit.job<Curve[]>('compareDQN', g, { variants: [...VARIANTS], runs, episodes, epsilon: num(o.epsilon, 0.1), lr, hidden: int(o.hidden, 16), capacity: int(o.capacity, 256), batch: int(o.batch, 16), copyEvery: int(o.copyEvery ?? o.targetEvery, 50), seed: int(o.seed, 23), gamma: GAMMA, evalEvery: 10 });
    const stats = { return: {}, error: {} } as Scene['stats'];
    for (const c of res) { stats.return[c.variant] = statOf(c.evalReturn); stats.error[c.variant] = statOf(c.qErr); }
    return { stats, at: res[0].evalAt, episodes, runs, lr, vstar: optimal(g, GAMMA).V[0], measure: str(o.measure, ['return', 'error'] as const, 'return') };
  },
  timeline: (s) => ({ duration: PLAY + HOLD, markers: [{ at: 0, label: 'Early training' }, { at: PLAY * 0.2, label: 'Middle' }, { at: PLAY * 0.6, label: 'Late' }, { at: PLAY, label: `All ${s.episodes} episodes` }] }),
  frameAt(t, s) {
    const ep = Math.min(s.episodes, Math.round((t / PLAY) * s.episodes)), i = Math.min(s.at.length - 1, Math.floor(ep / 10)), st = s.stats[s.measure];
    const rank = VARIANTS.map((v) => ({ v, m: st[v].mean[i] })).sort((a, b) => (s.measure === 'return' ? b.m - a.m : a.m - b.m));
    const what = s.measure === 'return' ? 'average return of the greedy policy' : 'average largest error against Q*';
    const status = ep >= s.episodes
      ? `After ${ep} episodes the ${what} is ${rank.map((r) => `${VARIANT_NAME[r.v].toLowerCase()} ${fmt(r.m)}`).join(', ')}${s.measure === 'return' ? `; the best possible is ${fmt(s.vstar)}` : ''}.`
      : `At episode ${s.at[i]}, ${VARIANT_NAME[rank[0].v].toLowerCase()} leads on ${what} (${fmt(rank[0].m)}); ${VARIANT_NAME[rank[3].v].toLowerCase()} is last (${fmt(rank[3].m)}).`;
    return { ep, i, status, readouts: [['Episode', `${s.at[i]} of ${s.episodes}`], ['Runs per variant', String(s.runs)], ['Learning rate', String(s.lr)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, st = s.stats[s.measure], ret = s.measure === 'return';
    const cb = nar ? { x: 44, y: 24, w: E.W - 58, h: E.H * 0.5 } : { x: 48, y: 26, w: E.W * 0.6 - 48, h: E.H - 84 };
    const all = VARIANTS.flatMap((v) => st[v].runs.flatMap((r) => Array.from(r)));
    const ya = ret ? { lo: Math.floor(Math.min(...all, 0) * 5) / 5, hi: Math.ceil(Math.max(...all, s.vstar) * 5) / 5, label: 'Return of the greedy policy' } : { lo: 0, hi: Math.ceil(Math.max(...all) * 2) / 2, label: 'Largest |Q − Q*|' };
    const ser: Series[] = [];
    for (const v of VARIANTS) for (const r of st[v].runs) ser.push({ ys: r, xs: s.at, color: VARIANT_COLOR[v], width: 1, op: 0.16 });
    for (const v of VARIANTS) ser.push({ ys: st[v].mean, xs: s.at, color: VARIANT_COLOR[v], width: 2.8 });
    const { sy } = lineChart(E, cb, { lo: 0, hi: s.episodes, label: 'Episode', ticks: [0, 100, 200, 300, 400, 500].filter((x) => x <= s.episodes) }, ya, ser, f.ep);
    if (ret) { line(E, cb.x, sy(s.vstar), cb.x + cb.w, sy(s.vstar), { stroke: MUTED, sw: 1.25, dash: '5 4' }); text(E, `best possible ${fmt(s.vstar)}`, cb.x + cb.w - 4, sy(s.vstar) - 5, { anchor: 'end', size: 11, fill: MUTED }); }
    const tx = nar ? 12 : cb.x + cb.w + 28, ty = nar ? cb.y + cb.h + 64 : 40, rh = nar ? 24 : 30, w = nar ? E.W - 24 : E.W - tx - 12;
    text(E, `Mean at episode ${s.at[f.i]}, ± spread of ${s.runs} runs`, tx, ty - 16, { fill: MUTED, size: 12 });
    VARIANTS.forEach((v, j) => {
      const y = ty + j * rh;
      rect(E, tx, y - 9, 16, 4, { fill: VARIANT_COLOR[v] });
      text(E, VARIANT_NAME[v], tx + 24, y, { size: 12 });
      text(E, `${fmt(st[v].mean[f.i])} ± ${fmt(st[v].sd[f.i])}`, tx + w, y, { anchor: 'end', size: 12, weight: 600 });
    });
    void TEXT;
  },
  label: (s) => `Learning curves of four DQN variants over ${s.episodes} episodes, ${s.runs} runs each, with the mean and spread of ${s.measure === 'return' ? 'the greedy policy return' : 'the largest error against the optimal Q-values'}.`,
  restTime: () => PLAY + HOLD,
};
export default mode;
