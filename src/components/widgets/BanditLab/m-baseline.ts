/**
 * baseline (policy-gradients step 2): one-pull REINFORCE gradient estimates at a frozen uniform policy, without and with a
 * baseline equal to the expected reward. Two histograms of the estimate for one arm's logit fill as samples stream in;
 * both averages settle on the exact gradient, one histogram is far narrower.
 * Props: means ("1,2,3"), samples (400), seed (9), arm (select, 1-based). Measured (seed 9, theta 0, baseline 2): variance
 * without / with, ratio: arm 1 0.727 / 0.232 (3.1x), arm 2 1.268 / 0.274 (4.6x), arm 3 1.788 / 0.305 (5.9x).
 */
import { gradientSamples, makeBandit } from '../../../lib/rl/bandit';
import { int, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, TEXT, axes, binCounts, fmt, line, rect, text, type Env } from '../_rl/svg';

const PLAY = 12, HOLD = 2.5, BINS = 28;
export interface BaselineScene { k: number; arm: number; n: number; plain: Float64Array; based: Float64Array; exact: number; baseline: number; lo: number; hi: number; ymax: number; stats: { mean: Float64Array; vari: Float64Array }[] }
interface F extends Frame { n: number }

/** Running mean and population variance after each prefix of `xs` (index n = first n values). */
function prefix(xs: Float64Array): { mean: Float64Array; vari: Float64Array } {
  const n = xs.length, mean = new Float64Array(n + 1), vari = new Float64Array(n + 1);
  let m = 0, m2 = 0;
  for (let i = 0; i < n; i++) { const d = xs[i] - m; m += d / (i + 1); m2 += d * (xs[i] - m); mean[i + 1] = m; vari[i + 1] = m2 / (i + 1); }
  return { mean, vari };
}
export function baselineScene(o: Opts): BaselineScene {
  const means = nums(o.means, [1, 2, 3]), k = means.length, n = int(o.samples, 400), arm = Math.max(0, Math.min(k - 1, int(o.arm, 3) - 1));
  const g = gradientSamples(new Float64Array(k), makeBandit(k, 0, { means }), n, int(o.seed, 9));
  const plain = Float64Array.from({ length: n }, (_, i) => g.plain[i * k + arm]), based = Float64Array.from({ length: n }, (_, i) => g.based[i * k + arm]);
  const all = [...plain, ...based], lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
  const ymax = Math.max(...binCounts(plain, BINS, lo, hi), ...binCounts(based, BINS, lo, hi));
  return { k, arm, n, plain, based, exact: g.exact[arm], baseline: g.baseline, lo, hi, ymax: Math.ceil((ymax * 1.15) / 10) * 10, stats: [prefix(plain), prefix(based)] };
}

const mode: ModeImpl<BaselineScene, F> = {
  setup: (o: Opts) => baselineScene(o),
  timeline(s) {
    const at = (n: number) => PLAY * Math.sqrt(n / s.n);
    return { duration: PLAY + HOLD, markers: [{ at: 0, label: 'First 20 samples' }, { at: at(20), label: 'Samples 21 to 100' }, { at: at(100), label: `Samples 101 to ${s.n}` }, { at: PLAY, label: 'All samples in' }] };
  },
  frameAt(t, s) {
    const n = Math.max(1, Math.min(s.n, Math.round(s.n * Math.min(1, t / PLAY) ** 2))), [a, b] = s.stats;
    const hdr = `logit of arm ${s.arm + 1}`;
    const status = n >= s.n
      ? `All ${n} estimates of the gradient for the ${hdr}. Without a baseline the variance is ${fmt(a.vari[n], 2)}; with it, ${fmt(b.vari[n], 2)}, which is ${fmt(a.vari[n] / b.vari[n], 1)} times smaller. Both averages are close to the exact gradient ${fmt(s.exact, 2)}.`
      : `${n} estimates so far for the ${hdr}. ${n < 5 ? 'Each one comes from a single pull, so they scatter.' : `Variance without a baseline ${fmt(a.vari[n], 2)}, with it ${fmt(b.vari[n], 2)}.`}`;
    return {
      n, status,
      detail: `The policy is frozen: every arm is equally likely, the baseline is the average reward ${fmt(s.baseline, 1)}.`,
      readouts: [
        ['Estimates', `${n} of ${s.n}`], ['Exact gradient', fmt(s.exact, 3)],
        ['Average, no baseline', fmt(a.mean[n], 3)], ['Average, with baseline', fmt(b.mean[n], 3)],
        ['Variance, no baseline', fmt(a.vari[n], 2)], ['Variance, with baseline', fmt(b.vari[n], 2)],
      ],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, ph = (E.H - (nar ? 108 : 100)) / 2, X = nar ? 40 : 52, W = E.W - X - 16;
    const boxes = [{ x: X, y: 34, w: W, h: ph }, { x: X, y: 34 + ph + (nar ? 40 : 44), w: W, h: ph }];
    const names = ['No baseline', 'With baseline'], cols = ['var(--viz-cat-1)', 'var(--viz-cat-2)'], data = [s.plain, s.based];
    const xa = (last: boolean) => ({ lo: s.lo, hi: s.hi, ticks: last ? undefined : [], label: last ? `Estimate of the gradient for the logit of arm ${s.arm + 1}` : undefined });
    boxes.forEach((b, j) => {
      const { sx, sy } = axes(E, b, xa(j === 1), { lo: 0, hi: s.ymax, ticks: [0, s.ymax / 2, s.ymax] });
      const c = binCounts(data[j], BINS, s.lo, s.hi, f.n), bw = b.w / BINS;
      for (let i = 0; i < BINS; i++) if (c[i]) rect(E, b.x + i * bw + 0.5, sy(c[i]), bw - 1, b.y + b.h - sy(c[i]), { fill: cols[j], op: 0.85 });
      text(E, `${names[j]}: variance ${fmt(s.stats[j].vari[f.n], 2)}`, b.x + 6, b.y + 15, { weight: 650, halo: true });
      const m = s.stats[j].mean[f.n], sd = Math.sqrt(s.stats[j].vari[f.n]), yb = b.y + b.h * 0.34;
      line(E, sx(m - sd), yb, sx(m + sd), yb, { stroke: ACCENT, sw: 2 });
      line(E, sx(m - sd), yb - 4, sx(m - sd), yb + 4, { stroke: ACCENT, sw: 2 });
      line(E, sx(m + sd), yb - 4, sx(m + sd), yb + 4, { stroke: ACCENT, sw: 2 });
      line(E, sx(m), b.y, sx(m), b.y + b.h, { stroke: ACCENT, sw: 2 });
      line(E, sx(s.exact), b.y, sx(s.exact), b.y + b.h, { stroke: TEXT, sw: 1.5, dash: '5 4' });
      text(E, 'Count', b.x - 40, b.y - 8, { size: 11, fill: MUTED });
      if (j === 0) text(E, 'Accent bar: average ± one standard deviation', b.x + b.w, b.y - 8, { size: 11, fill: MUTED, anchor: 'end' });
    });
  },
  label: (s) => `Two histograms of ${s.n} single-pull gradient estimates for the logit of arm ${s.arm + 1}: without a baseline the estimates are spread wide, with a baseline they are narrower, and both average out near the exact gradient ${fmt(s.exact, 2)}.`,
};
export default mode;
