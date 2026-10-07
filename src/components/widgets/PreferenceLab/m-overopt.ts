/**
 * overopt (rlhf step 8): the beta 0 run of `ppo` redrawn against how far the policy has moved: x = square root of KL to the
 * SFT model, y = proxy and gold reward (Gao et al. 2022 plot gold against sqrt KL). The gap between the curves shades in as
 * the run advances; the frames skip updates that barely move on the plot, so the readout always shows a real update
 * number. At the end the end states of the penalised runs (beta 0.5, 0.1, 0.02 from `kl-sweep`) are overlaid.
 * Props: beta (0), iters (400), labels ("human"), seed (6), overlay ("0.5,0.1,0.02").
 * Measured (plan §7): gold peaks at sqrt KL 0.652 (update 19, gold 2.819); final sqrt KL 3.608, proxy minus gold 5.159.
 */
import { int, nums, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, TEXT, circle, clamp, ease, fmt, line, mix, poly, scale, text, type Env } from '../_rl/svg';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { GOLD_INK, PROXY, rewardChart } from '../_align/draw';
import { ppoRun } from './sim';
import { REWARD_Y, peakIndex, runsFor } from './rlhf-common';

const LEAD = 1, PACE = 0.07, FADE = 1.4, HOLD = 3;
interface Scene {
  beta: number; run: AlignPPOResult; iters: number; peak: number;
  /** The kept update numbers and, per kept frame, sqrt KL, proxy and gold. */
  at: Int32Array; xs: Float64Array; proxy: Float64Array; gold: Float64Array; pk: number; xHi: number;
  other: { beta: number; x: number; proxy: number; gold: number }[];
}
interface F extends Frame { j: number; fade: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const beta = num(o.beta, 0), iters = int(o.iters, 400), run = await ppoRun(o, kit, { beta, iters }), peak = peakIndex(run.gold);
    const keep: number[] = [0];
    let lx = 0, ly = run.gold[0], lp = run.proxy[0];
    for (let i = 1; i <= iters; i++) {
      const x = Math.sqrt(run.kl[i]);
      if (i === peak || i === iters || Math.hypot((x - lx) * 3, run.gold[i] - ly, run.proxy[i] - lp) >= 0.05) { keep.push(i); lx = x; ly = run.gold[i]; lp = run.proxy[i]; }
    }
    const betas = nums(o.overlay, [0.5, 0.1, 0.02]), others = await runsFor(o, kit, betas, 200);
    const other = others.map((r, j) => ({ beta: betas[j], x: Math.sqrt(r.kl[r.iters]), proxy: r.proxy[r.iters], gold: r.gold[r.iters] }));
    const xs = Float64Array.from(keep, (i) => Math.sqrt(run.kl[i]));
    return { beta, run, iters, peak, at: Int32Array.from(keep), xs, proxy: Float64Array.from(keep, (i) => run.proxy[i]), gold: Float64Array.from(keep, (i) => run.gold[i]), pk: keep.indexOf(peak), xHi: Math.max(...xs) * 1.08, other };
  },
  timeline(s) {
    const tEnd = LEAD + s.at.length * PACE;
    return { duration: tEnd + FADE + HOLD, markers: [{ at: 0, label: `β = ${s.beta} run` }, { at: tEnd, label: 'Penalised runs' }] };
  },
  frameAt(t, s) {
    const j = clamp(Math.round((t - LEAD) / PACE), 0, s.at.length - 1), tEnd = LEAD + s.at.length * PACE, fade = ease((t - tEnd) / FADE), u = s.at[j];
    const gap = s.proxy[j] - s.gold[j];
    let status = u === 0
      ? `Update 0 is the SFT model: distance 0, proxy ${fmt(s.proxy[0], 2)}, gold ${fmt(s.gold[0], 2)}.`
      : `Update ${u}: the policy is ${fmt(s.xs[j], 2)} from the SFT model (square root of KL). Proxy ${fmt(s.proxy[j], 2)}, gold ${fmt(s.gold[j], 2)}, so the reward model overstates quality by ${fmt(gap, 2)}.`;
    if (u >= s.peak && u > 0) status += ` Gold peaked at distance ${fmt(Math.sqrt(s.run.kl[s.peak]), 2)} (update ${s.peak}).`;
    if (fade > 0.5) status = `The penalised runs end at ${s.other.map((r) => `β = ${r.beta}: distance ${fmt(r.x, 2)}, gold ${fmt(r.gold, 2)}`).join('; ')}. ` + `The β = ${s.beta} run ended at distance ${fmt(s.xs[s.xs.length - 1], 2)} with gold ${fmt(s.gold[s.gold.length - 1], 2)}.`;
    return { j, fade, status, readouts: [['Update', `${u} of ${s.iters}`], ['Distance (√KL)', fmt(s.xs[j], 2)], ['Proxy', fmt(s.proxy[j], 2)], ['Gold', fmt(s.gold[j], 2)], ['Proxy minus gold', fmt(gap, 2)]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, b = { x: nw ? 40 : 56, y: 28, w: E.W - (nw ? 40 : 56) - 16, h: E.H - 28 - 50 }, j = f.j;
    const xa = { lo: 0, hi: s.xHi, ticks: [0, 1, 2, 3, 4].filter((v) => v < s.xHi), label: 'Distance from the SFT model (square root of KL)' };
    const sx = scale(xa, b.x, b.x + b.w), sy = scale(REWARD_Y, b.y + b.h, b.y);
    // the gap between the proxy and gold curves, shaded up to the current frame
    if (j > 0) {
      const top: [number, number][] = [], bot: [number, number][] = [];
      for (let i = 0; i <= j; i++) { top.push([sx(s.xs[i]), clamp(sy(s.proxy[i]), b.y, b.y + b.h)]); bot.unshift([sx(s.xs[i]), clamp(sy(s.gold[i]), b.y, b.y + b.h)]); }
      poly(E, [...top, ...bot], { fill: mix(MUTED, BG, 22) });
    }
    rewardChart(E, b, {
      proxy: s.proxy.subarray(0, j + 1), gold: s.gold.subarray(0, j + 1), xs: s.xs.subarray(0, j + 1), upto: Infinity, x: xa, y: REWARD_Y,
      mark: j >= s.pk ? { i: s.pk, label: `Gold peaks at ${fmt(s.xs[s.pk], 2)}` } : undefined,
    });
    circle(E, sx(s.xs[j]), clamp(sy(s.proxy[j]), b.y, b.y + b.h), 4, { fill: PROXY, stroke: BG, sw: 2 });
    circle(E, sx(s.xs[j]), clamp(sy(s.gold[j]), b.y, b.y + b.h), 4, { fill: GOLD_INK, stroke: BG, sw: 2 });
    if (f.fade > 0) for (const r of s.other) {
      const x = sx(r.x), yp = clamp(sy(r.proxy), b.y, b.y + b.h), yg = clamp(sy(r.gold), b.y, b.y + b.h);
      line(E, x, yp, x, yg, { stroke: MUTED, sw: 1.5, op: f.fade });
      circle(E, x, yp, 4.5, { fill: BG, stroke: PROXY, sw: 2, op: f.fade });
      circle(E, x, yg, 4.5, { fill: BG, stroke: TEXT, sw: 2, op: f.fade });
      text(E, `β = ${r.beta}`, x, yp - 9, { size: 11, anchor: x > b.x + b.w - 40 ? 'end' : 'middle', weight: 650, halo: true, op: f.fade });
    }
  },
  label: (s) => `Proxy and gold reward against the square root of KL to the SFT model for a PPO run with penalty ${s.beta}. Gold peaks at distance ${fmt(Math.sqrt(s.run.kl[s.peak]), 2)} and then falls while the proxy keeps rising.`,
};
export default mode;
