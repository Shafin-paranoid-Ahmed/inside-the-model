/**
 * explore (rl-basics step 6): greedy vs ε-greedy vs UCB. Left: one featured ε-greedy run (estimates, true means,
 * pull counts). Right: average reward and % optimal action over `runs` bandits, drawn left to right.
 * Props: arms (5), epsilons ("0,0.1"), ucb (true), runs (50), steps (500), seed (7).
 * Measured (seed 7): % optimal over the last 100 steps (WIN) greedy 74.0, ε 0.1 89.3, UCB 91.0; greedy stuck 13/50.
 */
import { compareStrategies, makeBandit, runBandit, strategyName, type Bandit, type BanditRun, type StrategyCurve } from '../../../lib/rl/bandit';
import { int, key, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, lineChart, text, type Env } from '../_rl/svg';
import { armBars } from './draw';
import { strategiesOf, strategyColor } from './sim';

const PLAY = 12, HOLD = 2.5, WIN = 100, SMOOTH = 20;
/** Trailing mean over `w` steps, so the drawn curves are readable (the per-step averages over 50 bandits are noisy). */
const trail = (ys: ArrayLike<number>, w: number, scale = 1) => { const out = new Array<number>(ys.length); let sum = 0; for (let i = 0; i < ys.length; i++) { sum += ys[i]; if (i >= w) sum -= ys[i - w]; out[i] = (scale * sum) / Math.min(i + 1, w); } return out; };
interface Scene { curves: StrategyCurve[]; bandit: Bandit; run: BanditRun; steps: number; runs: number; k: number; featured: string }
interface F extends Frame { n: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit: Kit) {
    const k = int(o.arms, 5), runs = int(o.runs, 50), steps = int(o.steps, 500), seed = int(o.seed, 7), strategies = strategiesOf(o);
    const curves = kit.memo(key('bandit-compare', k, runs, steps, seed, strategies), () => compareStrategies({ k, runs, steps, seed, strategies }));
    const feat = strategies.find((s) => s.kind === 'epsilon') ?? strategies[0], bandit = makeBandit(k, seed * 1000);
    return { curves, bandit, run: runBandit(bandit, feat, steps, seed * 7919, { record: true }), steps, runs, k, featured: strategyName(feat) };
  },
  timeline: (s) => ({
    duration: PLAY + HOLD,
    markers: [{ at: 0, label: 'First pulls' }, { at: (PLAY * 50) / s.steps, label: 'Early' }, { at: PLAY * 0.4, label: 'Later' }, { at: PLAY, label: `All ${s.steps} steps` }],
  }),
  frameAt(t, s) {
    const n = Math.max(1, Math.min(s.steps, Math.round((t / PLAY) * s.steps)));
    const pct = (c: StrategyCurve) => { const a = Math.max(0, n - WIN); let m = 0; for (let i = a; i < n; i++) m += c.optimal[i]; return (100 * m) / (n - a); };
    const done = n >= s.steps, g = s.curves.find((c) => c.strategy.kind === 'greedy');
    const status = done
      ? `After ${s.steps} steps, over the last ${WIN}: ` + s.curves.map((c) => `${c.name} picks the best arm ${pct(c).toFixed(0)}% of the time`).join(', ') + '.' + (g ? ` Greedy settled on a worse arm in ${g.stuck} of ${s.runs} runs.` : '')
      : `Step ${n}: averaged over ${s.runs} bandits, ` + s.curves.map((c) => `${c.name} ${pct(c).toFixed(0)}%`).join(', ') + ` best-arm picks (last ${Math.min(n, WIN)} steps).`;
    return { n, status, detail: `Left: one ${s.featured} run on bandit 1. Bars are its estimates, ticks the true means.`, readouts: [['Step', `${n} of ${s.steps}`], ...s.curves.map((c) => [c.name, `${pct(c).toFixed(0)}%`] as [string, string])] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, i = f.n - 1, k = s.k;
    const est = s.run.est!.subarray(i * k, i * k + k), counts = s.run.counts!.subarray(i * k, i * k + k);
    const lo = Math.min(-0.5, ...Array.from(s.bandit.means), ...Array.from(est)) - 0.2, hi = Math.max(0.5, ...Array.from(s.bandit.means), ...Array.from(est)) + 0.3;
    const left = nar ? { x: 40, y: 26, w: E.W - 56, h: E.H * 0.27 } : { x: 44, y: 30, w: E.W * 0.36, h: E.H - 100 };
    text(E, `One ${s.featured} run`, left.x - 34, left.y - 12, { weight: 600 });
    armBars(E, left, { est, truth: s.bandit.means, counts, lo, hi, pulled: s.run.arms[i], best: s.bandit.best });
    const rx = nar ? 44 : left.x + left.w + 70, rw = E.W - rx - 14;
    const top = nar ? left.y + left.h + 78 : 30, ch = nar ? (E.H - top - 80) / 2 : (E.H - 120) / 2;
    const xa = { lo: 0, hi: s.steps, label: nar ? undefined : 'Step' };
    const sr = s.curves.map((c, j) => ({ ys: trail(c.avgReward, SMOOTH), color: strategyColor(j) })), so = s.curves.map((c, j) => ({ ys: trail(c.optimal, SMOOTH, 100), color: strategyColor(j) }));
    const yr = sr.flatMap((c) => c.ys);
    lineChart(E, { x: rx, y: top, w: rw, h: ch }, { ...xa, label: undefined }, { lo: Math.floor(Math.min(0, ...yr) * 2) / 2, hi: Math.ceil(Math.max(...yr) * 2) / 2, label: `Average reward (${s.runs} bandits, ${SMOOTH}-step mean)` }, sr, f.n);
    lineChart(E, { x: rx, y: top + ch + (nar ? 48 : 58), w: rw, h: ch }, xa, { lo: 0, hi: 100, ticks: [0, 50, 100], label: `% best arm chosen (${SMOOTH}-step mean)` }, so, f.n);
    if (nar) text(E, 'Step', rx + rw / 2, E.H - 6, { anchor: 'middle', fill: MUTED });
  },
  label: (s) => `Bandit comparison: estimated and true means of ${s.k} arms for one run, and the average reward and percentage of best-arm choices of ${s.curves.map((c) => c.name).join(', ')} over ${s.runs} bandits and ${s.steps} steps.`,
  restTime: () => PLAY + HOLD,
};
export default mode;
