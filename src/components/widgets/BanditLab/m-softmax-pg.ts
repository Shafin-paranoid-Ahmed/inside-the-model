/**
 * softmax-pg (policy-gradients step 1): a softmax policy over bandit arms learns by REINFORCE. Each pull samples an arm,
 * earns a reward r, and pushes every logit by lr * r * (onehot(arm) - pi). The first six pulls play slowly (sample,
 * reward, push, new probabilities); the rest stream past while the probability curves draw.
 * Props: means ("1,2,3"), lr (0.1), steps (200), seed (5). Measured (seed 5): P(best arm) 0.333, 0.835, 0.964 at pulls 0, 50, 200.
 */
import { banditPG, makeBandit, type BanditPGRun } from '../../../lib/rl/bandit';
import { int, nums, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, ease, fmt, lerp, lineChart, line, rect, text, type Env } from '../_rl/svg';

const SLOW_N = 6, SLOW = 1.4, FAST = 9, HOLD = 2.2;
interface Scene { run: BanditPGRun; means: number[]; best: number; k: number; steps: number; lr: number; series: Float64Array[] }
interface F extends Frame { n: number; i: number; u: number; slow: boolean; p: Float64Array; pulled: boolean; done: boolean }

const col = (a: number) => `var(--viz-cat-${a + 1})`;
const at = (run: BanditPGRun, k: number, n: number) => run.probs.subarray(n * k, n * k + k);

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const means = nums(o.means, [1, 2, 3]), steps = int(o.steps, 200), lr = num(o.lr, 0.1), k = means.length;
    const bandit = makeBandit(k, 0, { means }), run = banditPG(bandit, { lr, steps, seed: int(o.seed, 5) });
    const series = Array.from({ length: k }, (_, a) => Float64Array.from({ length: steps + 1 }, (_, n) => run.probs[n * k + a]));
    return { run, means, best: bandit.best, k, steps, lr, series };
  },
  timeline(s) {
    const t50 = SLOW_N * SLOW + (FAST * (50 - SLOW_N)) / (s.steps - SLOW_N), end = SLOW_N * SLOW + FAST;
    return { duration: end + HOLD, markers: [{ at: 0, label: `First ${SLOW_N} pulls` }, { at: SLOW_N * SLOW, label: `Pulls ${SLOW_N + 1} to 50` }, { at: t50, label: `Pulls 51 to ${s.steps}` }, { at: end, label: 'All pulls done' }] };
  },
  frameAt(t, s) {
    const { run, k, steps, lr, best } = s, slow = t < SLOW_N * SLOW;
    const x = slow ? t / SLOW : SLOW_N + ((t - SLOW_N * SLOW) / FAST) * (steps - SLOW_N);
    const done = x >= steps, i = done ? steps - 1 : Math.min(steps - 1, Math.floor(x)), u = done ? 1 : slow ? x - i : 1;
    const arm = run.arms[i], r = run.rewards[i], p0 = at(run, k, i), p1 = at(run, k, i + 1);
    const upd = u >= 0.65 || !slow, n = upd ? i + 1 : i;   // pulls whose update is already shown
    const shown = upd ? p1 : p0, tw = slow ? ease((u - 0.65) / 0.35) : 1;
    const p = Float64Array.from(p0, (a, j) => lerp(a, p1[j], tw));
    const name = (a: number) => `arm ${a + 1}`;
    let status: string;
    if (slow && u < 0.3) status = `Pull ${i + 1}: the policy picks ${name(arm)}, which it chose with probability ${fmt(p0[arm], 2)}.`;
    else if (slow && u < 0.5) status = `${name(arm)} pays reward ${fmt(r, 2)}.`;
    else if (slow && u < 0.65) status = `The push on each logit is ${lr} × ${fmt(r, 2)} × (1 for ${name(arm)}, 0 for the others, minus the arm's probability). ${name(arm)} goes ${run.grad[i * k + arm] >= 0 ? 'up' : 'down'}.`;
    else if (slow) status = `After pull ${i + 1} the probabilities are ${Array.from(p1, (v) => fmt(v, 2)).join(', ')}.`;
    else status = done ? `After all ${steps} pulls the best arm (arm ${best + 1}) has probability ${fmt(shown[best], 3)}.` : `Pull ${i + 1}: ${name(arm)}, reward ${fmt(r, 2)}. The best arm now has probability ${fmt(shown[best], 2)}.`;
    return {
      n, i, u, slow, p, pulled: !done && (!slow || u < 1), done, status,
      detail: 'Bars: the policy\'s chance of picking each arm. Right: how this pull moves each logit, and the average move.',
      readouts: [['Pulls', `${n} of ${steps}`], ['Chance of best arm', fmt(shown[best], 3)], ['Reward of last pull', n === 0 ? '–' : fmt(run.rewards[n - 1], 2)]],
    };
  },
  draw(E: Env, f, s) {
    const { run, k, means, best, lr } = s, nar = E.narrow, i = f.i;
    const L = nar ? { x: 10, y: 34, w: E.W - 20, h: E.H * 0.46 } : { x: 10, y: 34, w: E.W * 0.5 - 14, h: E.H - 60 };
    const rh = L.h / k, pw = L.w * 0.5, cxm = L.x + L.w * 0.8, hw = L.w * 0.17, MAXP = 0.3, sz = nar ? 11 : 12;
    text(E, 'Chance of picking the arm', L.x, L.y - 12, { size: sz, fill: MUTED });
    text(E, nar ? 'Logit push' : 'Push on its logit', cxm, L.y - 12, { size: sz, fill: MUTED, anchor: 'middle' });
    for (let a = 0; a < k; a++) {
      const ry = L.y + a * rh, by = ry + 20;
      if (f.pulled && run.arms[i] === a && f.u >= 0) rect(E, L.x - 4, ry + 2, L.w + 8, rh - 6, { fill: 'none', stroke: ACCENT, sw: 2, rx: 4 });
      text(E, `Arm ${a + 1}${a === best ? ' (best)' : ''}, mean ${fmt(means[a], 0)}`, L.x, ry + 15, { size: sz, weight: a === best ? 650 : 400 });
      rect(E, L.x, by, pw, 20, { fill: 'none', stroke: RULE, sw: 1 });
      rect(E, L.x, by, pw * f.p[a], 20, { fill: col(a), op: 0.85 });
      text(E, fmt(f.p[a], 2), L.x + pw + 6, by + 15, { size: sz });
      // sampled push (lr * r * (onehot - pi)) and the average push (lr * exact gradient)
      line(E, cxm, ry + 18, cxm, ry + 46, { stroke: RULE, sw: 1 });
      if (f.pulled && (!f.slow || f.u >= 0.5)) {
        const g = lr * run.grad[i * k + a], w = (Math.max(-MAXP, Math.min(MAXP, g)) / MAXP) * hw;
        rect(E, Math.min(cxm, cxm + w), by, Math.abs(w), 13, { fill: 'var(--viz-gradient)', op: 0.9 });
        text(E, (g > 0 ? '+' : '') + fmt(g, 2), g >= 0 ? cxm - 4 : cxm + 4, by + 11, { size: 11, anchor: g >= 0 ? 'end' : 'start', fill: MUTED });
      }
      const ge = lr * run.exact[Math.min(i, s.steps - 1) * k + a], we = (Math.max(-MAXP, Math.min(MAXP, ge)) / MAXP) * hw;
      rect(E, Math.min(cxm, cxm + we), by + 16, Math.abs(we), 5, { fill: TEXT, op: 0.5 });
    }
    const C = nar ? { x: 44, y: L.y + L.h + 34, w: E.W - 60, h: E.H - L.y - L.h - 34 - 44 } : { x: E.W * 0.5 + 44, y: 34, w: E.W * 0.5 - 62, h: E.H - 90 };
    lineChart(E, C, { lo: 0, hi: s.steps, label: 'Pulls' }, { lo: 0, hi: 1, ticks: [0, 0.5, 1], label: 'Probability of each arm' },
      s.series.map((ys, a) => ({ ys, color: col(a), width: a === best ? 2.5 : 1.75 })), f.n);
  },
  label: (s) => `A softmax policy over ${s.k} bandit arms with means ${s.means.join(', ')} learns by policy gradient: each pull's reward pushes the logits, and the chance of the best arm rises from ${fmt(s.run.probs[s.best], 2)} to ${fmt(s.run.probs[s.steps * s.k + s.best], 2)} over ${s.steps} pulls.`,
};
export default mode;
