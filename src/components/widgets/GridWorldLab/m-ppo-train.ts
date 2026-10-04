/**
 * ppo-train (ppo step 7): the whole PPO loop on the 4x3 grid, 100 iterations with a fresh batch each time (worker job).
 * Left: the policy's action probabilities as arrows. Right: the exact value of the start cell under the current
 * stochastic policy (dashed line = best possible), and the critic's RMS error against the exact values.
 * Props: seed (37), iterations (100), gamma (0.9), lambda (0.95), epsilon (0.2), batch (64), epochs (4), minibatch (16), lr (0.03).
 */
import { START } from '../../../lib/rl/gridworld';
import { optimal, softmaxPolicy } from '../../../lib/rl/tabular';
import type { PPOResult } from '../../../lib/rl/ppo';
import { int, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, axes, circle, clamp, fmt, lerp, line, path, signFmt, text, type Env } from '../_rl/svg';
import { drawGrid, drawPolicy, gridIn } from './draw';
import { gridOf } from './sim';

const DT = 0.1, HOLD = 2.5;
interface Scene { res: PPOResult; iters: number; probs: Float64Array[]; best: number; eps: number; batch: number }
interface F extends Frame { n: number; x: number; p: Float64Array }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const g = gridOf(o, 0.9), iters = int(o.iterations, 100), eps = num(o.epsilon, 0.2), batch = int(o.batch, 64);
    const res = await kit.job<PPOResult>('trainPPO', g, { iterations: iters, seed: int(o.seed, 37), gamma: g.gamma, lambda: num(o.lambda, 0.95), eps, batch, epochs: int(o.epochs, 4), minibatch: int(o.minibatch, 16), lr: num(o.lr, 0.03) });
    return { res, iters, probs: res.thetas.map((th) => softmaxPolicy(th)), best: optimal(g, g.gamma).V[START], eps, batch };
  },
  timeline(s) {
    const at = (k: number) => Math.min(s.iters, k) * DT;
    return { duration: s.iters * DT + HOLD, markers: [{ at: 0, label: 'Early gains' }, { at: at(20), label: 'Slowing down' }, { at: at(50), label: 'Near the limit' }] };
  },
  frameAt(t, s) {
    const x = clamp(t / DT, 0, s.iters), n = Math.floor(x), k = Math.min(s.iters - 1, n), m = x - k;
    const p = s.probs[k].map((v, i) => lerp(v, s.probs[Math.min(s.iters, k + 1)][i], x >= s.iters ? 1 : m)), r = s.res;
    const val = r.evalReturn[n], err = r.criticErr[Math.min(n, s.iters - 1)];
    const eps = n ? r.iterations[n - 1].batch.episodeReturns : [], mean = eps.length ? eps.reduce((a, b) => a + b, 0) / eps.length : NaN;
    const status = n === 0
      ? `Iteration 0: the policy is uniform, so every action has probability 0.25. Its start cell is worth ${fmt(val, 3)} under this policy; the best possible is ${fmt(s.best, 3)}.`
      : `Iteration ${n}: PPO collected ${s.batch} steps with the policy it had, trained four epochs on them, and discarded the batch. The start cell is now worth ${fmt(val, 3)} (best ${fmt(s.best, 3)}). The last batch ended ${eps.length} episode${eps.length === 1 ? '' : 's'}${eps.length ? ` with mean return ${signFmt(mean, 2)}` : ''}.`;
    return { n, x, p, status, readouts: [['Iteration', `${n} of ${s.iters}`], ['Start-cell value', fmt(val, 3)], ['Critic error', fmt(err, 3)], ['KL(old ‖ new), last batch', n ? fmt(r.kl[n - 1], 4) : '—']] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H;
    const G = gridIn(nw ? { x: 10, y: 28, w: W - 20, h: H * 0.27 } : { x: 10, y: 34, w: W * 0.36, h: H - 80 });
    text(E, 'Policy after this iteration', G.x0, nw ? 18 : 24, { size: 12, fill: MUTED });
    drawGrid(E, G, {});
    drawPolicy(E, G, { probs: f.p, color: 'var(--viz-forward)' });
    const left = nw ? 46 : G.x0 + G.w + 60, w = W - left - 16, top = nw ? G.y0 + G.h + 40 : 40;
    const h1 = nw ? (H - top - 120) * 0.55 : (H - top - 120) * 0.55, h2 = nw ? (H - top - 120) * 0.45 : (H - top - 120) * 0.45;
    const ticks = [0, 20, 40, 60, 80, 100].filter((v) => v <= s.iters);
    const R = s.res, up = Math.floor(f.x);
    const A = axes(E, { x: left, y: top, w, h: h1 }, { lo: 0, hi: s.iters, ticks }, { lo: -0.5, hi: 0.4, ticks: [-0.4, 0, 0.4], label: 'Value of the start cell' });
    line(E, left, A.sy(s.best), left + w, A.sy(s.best), { stroke: MUTED, sw: 1.25, dash: '5 4' });
    let d = '';
    for (let i = 0; i <= up; i++) d += `${d ? 'L' : 'M'}${A.sx(i).toFixed(1)} ${A.sy(R.evalReturn[i]).toFixed(1)}`;
    path(E, d, { stroke: 'var(--viz-pos)', sw: 2.25 });
    circle(E, A.sx(up), A.sy(R.evalReturn[up]), 4.5, { fill: ACCENT, stroke: BG, sw: 2 });
    const top2 = top + h1 + 52;
    const B = axes(E, { x: left, y: top2, w, h: h2 }, { lo: 0, hi: s.iters, ticks, label: 'Iteration' }, { lo: 0, hi: 0.5, ticks: [0, 0.25, 0.5], label: 'Critic error (RMS)' });
    let d2 = '';
    for (let i = 0; i <= Math.min(up, s.iters - 1); i++) d2 += `${d2 ? 'L' : 'M'}${B.sx(i).toFixed(1)} ${B.sy(R.criticErr[i]).toFixed(1)}`;
    path(E, d2, { stroke: 'var(--viz-neg)', sw: 2.25 });
    circle(E, B.sx(Math.min(up, s.iters - 1)), B.sy(R.criticErr[Math.min(up, s.iters - 1)]), 4.5, { fill: ACCENT, stroke: BG, sw: 2 });
  },
  label: (s) => `PPO training on the 4 by 3 grid for ${s.iters} iterations: policy arrows, the value of the start cell rising from ${fmt(s.res.evalReturn[0], 2)} to ${fmt(s.res.evalReturn[s.iters], 2)}, and the critic error falling.`,
};
export default mode;
