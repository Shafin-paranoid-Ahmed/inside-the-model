/**
 * dpo-train (dpo step 3): DPO (trainDPO, beta 0.1, 300 full-batch Adam steps, lr 0.05 / beta) on the 200 human-labelled pairs,
 * with no reward model and no sampling. Left: the policy on prompt 0 (reference ghost bars, ticks at the closed-form optimum),
 * the DPO loss; right: reward-model score and gold reward (same axes as the PPO figures) and the KL to the reference.
 * Props: beta (0.1), steps (300), seed (4, pair seed). Measured: loss 0.693 -> 0.487, gold 2.819, KL 0.968, held-out accuracy
 * 0.680, max |DPO - pi*| 0.005; gold passes 1.97 at step 100 (the policy visits the rambling answer and returns).
 */
import { PROMPTS, TEXT, TRAIN } from '../../../lib/align/bank';
import { expected, probsInto, scores } from '../../../lib/align/policy';
import { optimalRow } from '../../../lib/align/optimal';
import { pairAccuracy } from '../../../lib/align/reward';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, TEXT as INK, axes, circle, clamp, fmt, path, text, type Env } from '../_rl/svg';
import { candidateBars, charsFor, clip, rewardChart } from '../_align/draw';
import { betaOf, dpoRun } from './dpoKit';
import { prefData } from './sim';

const LEAD = 1.5, RUN = 13, TAIL = 2.5;
interface Scene { beta: number; steps: number; sft: Float64Array; star: Float64Array; rows: Float64Array[]; loss: Float64Array; gold: Float64Array; kl: Float64Array; proxy: Float64Array; acc: Float64Array; klMax: number; gap: number }
interface F extends Frame { n: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const d = prefData(o, kit, int(o.seed, 4)), beta = betaOf(o), steps = int(o.steps, 300), run = await dpoRun(kit, d, beta, steps), rm = scores(d.rm.w);
    const w = new Float64Array(d.sft.length), star = optimalRow(d.sft, d.rm.w, beta, 0).p;
    let gap = 0;
    for (const x of TRAIN) { const a = probsInto(run.thetas[steps], x), b = optimalRow(d.sft, d.rm.w, beta, x).p; for (let y = 0; y < a.length; y++) gap = Math.max(gap, Math.abs(a[y] - b[y])); }
    const acc = new Float64Array(steps + 1), proxy = new Float64Array(steps + 1);
    run.thetas.forEach((th, k) => { for (let j = 0; j < w.length; j++) w[j] = beta * (th[j] - d.sft[j]); acc[k] = pairAccuracy(w, d.held); proxy[k] = expected(th, rm, TRAIN); });
    return { beta, steps, sft: d.sft, star, rows: run.thetas.map((th) => probsInto(th, 0)), loss: run.loss, gold: run.gold, kl: run.kl, proxy, acc, klMax: Math.ceil(Math.max(...run.kl) * 1.05), gap };
  },
  timeline: (s) => ({ duration: LEAD + RUN + TAIL, markers: [{ at: 0, label: 'Reference' }, { at: LEAD, label: 'Training' }, { at: LEAD + RUN, label: `After ${s.steps} steps` }] }),
  frameAt(t, s) {
    const n = clamp(Math.round(((t - LEAD) / RUN) * s.steps), 0, s.steps), p = s.rows[n], ramb = p[4];
    const status = n === 0
      ? 'Before training the policy is the reference, the SFT model. DPO will fit it to the 200 labelled pairs with one supervised-style loss.'
      : `Step ${n} of ${s.steps}: loss ${fmt(s.loss[n], 3)}, gold reward ${fmt(s.gold[n], 2)}, KL to the reference ${fmt(s.kl[n], 2)}. On this prompt the rambling answer has probability ${fmt(ramb, 2)}${n === s.steps ? `, and the policy sits within ${fmt(s.gap, 3)} of the closed-form optimum.` : '.'}`;
    return {
      n, status,
      readouts: [['Step', `${n} of ${s.steps}`], ['Loss', fmt(s.loss[n], 3)], ['Gold reward', fmt(s.gold[n], 2)], ['KL to reference', fmt(s.kl[n], 2)], ['Held-out accuracy', fmt(s.acc[n], 3)]],
    };
  },
  restTime: () => LEAD + RUN + 0.5,
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, n = f.n, lw = nw ? W : W * 0.47;
    text(E, clip(`User: ${PROMPTS[0].q}  Probability of each response`, charsFor(lw - 2 * pad, 12)), pad, 18, { size: 12, fill: MUTED });
    const bh = nw ? 220 : 236;
    candidateBars(E, { x: pad, y: 30, w: lw - 2 * pad - 4, h: bh }, { p: s.rows[n], ref: probsInto(s.sft, 0), ticks: s.star, texts: TEXT[0], highlight: 4 });
    const xa = { lo: 0, hi: s.steps, ticks: [0, s.steps / 3, (2 * s.steps) / 3, s.steps], label: 'Step' };
    let g: { sx: (v: number) => number; sy: (v: number) => number };
    // reward chart (right column, or below the bars on a phone)
    const rb = nw ? { x: 46, y: 30 + bh + 38, w: W - 46 - 16, h: 118 } : { x: lw + 44, y: 40, w: W - lw - 44 - 16, h: 160 };
    g = rewardChart(E, rb, { proxy: s.proxy, gold: s.gold, upto: n, x: xa, y: { lo: -1, hi: 5, ticks: [-1, 0, 1, 2, 3, 4, 5], label: 'Reward (average over prompts)' } });
    circle(E, g.sx(n), clamp(g.sy(s.gold[n]), rb.y, rb.y + rb.h), 4, { fill: INK, stroke: BG, sw: 1.5 });
    // loss and KL
    const small = nw ? 96 : 0, cw = nw ? (W - 2 * pad - 20) / 2 - 42 : 0;
    const lb = nw ? { x: pad + 38, y: rb.y + rb.h + 66, w: cw, h: small } : { x: pad + 38, y: 30 + bh + 40, w: lw - 2 * pad - 44, h: H - (30 + bh + 40) - 50 };
    const kb = nw ? { x: pad + 38 + cw + 20 + 38, y: lb.y, w: cw, h: small } : { x: rb.x, y: rb.y + rb.h + 66, w: rb.w, h: H - (rb.y + rb.h + 66) - 50 };
    const series = (b: { x: number; y: number; w: number; h: number }, ys: Float64Array, ya: { lo: number; hi: number; ticks?: number[]; label: string }, color: string) => {
      const { sx, sy } = axes(E, b, { lo: 0, hi: s.steps, ticks: [0, s.steps / 2, s.steps], label: 'Step' }, ya);
      let d = '';
      for (let i = 0; i <= n; i++) d += `${i ? 'L' : 'M'}${sx(i).toFixed(1)} ${clamp(sy(ys[i]), b.y, b.y + b.h).toFixed(1)}`;
      path(E, d, { stroke: color, sw: 2.25 });
      circle(E, sx(n), clamp(sy(ys[n]), b.y, b.y + b.h), 4, { fill: color, stroke: BG, sw: 1.5 });
    };
    series(lb, s.loss, { lo: 0.45, hi: 0.7, ticks: [0.5, 0.6, 0.7], label: 'DPO loss' }, INK);
    series(kb, s.kl, { lo: 0, hi: s.klMax, label: 'KL to the reference' }, MUTED);
  },
  label: (s) => `DPO trained for ${s.steps} steps on 200 preference pairs at beta ${s.beta}: the policy's probabilities on a prompt moving toward the closed-form optimum, the DPO loss falling from ${fmt(s.loss[0], 2)} to ${fmt(s.loss[s.steps], 2)}, and the reward-model, gold and KL curves.`,
};
export default mode;
