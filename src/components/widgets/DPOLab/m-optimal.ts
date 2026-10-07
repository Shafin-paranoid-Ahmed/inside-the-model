/**
 * optimal (dpo step 1): the closed-form optimum pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x) for the reward model fitted to
 * the 200 pairs (pair seed 4, 200 steps). The timeline sweeps beta on a log scale from 1 down to 0.02, holding at 0.1,
 * where the end state of PPO (beta 0.1, seed 6, 200 iterations) is marked with ticks. Every frame evaluates the formula
 * (optimalRow) for the bars and the exact expectations for the chart. Left: the 8 responses to prompt 0 with the reference
 * as ghost bars; right: reward-model score and gold reward of pi* (all training prompts) against beta.
 * Props: prompt (0), seed (6, PPO). Measured: plans/day-11.md section 7.
 */
import { CANDIDATES, K, TEXT, PROMPTS, TRAIN } from '../../../lib/align/bank';
import { GOLD, expected, kl, probsInto, scores } from '../../../lib/align/policy';
import { optimalRow, optimalTheta } from '../../../lib/align/optimal';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, TEXT as INK, axes, circle, clamp, ease, fmt, line, path, text, type Env } from '../_rl/svg';
import { GOLD_DASH, PROXY, candidateBars, charsFor, clip } from '../_align/draw';
import { name } from './dpoKit';
import { ppoRun, prefData } from './sim';

const A = 4, HOLD = 3, B = 4, TAIL = 2.5, B0 = 0.1, GRID = 64;
const betaAt = (t: number): number => {
  const lg = (a: number, b: number, u: number) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * ease(u));
  return t < A ? lg(1, B0, t / A) : t < A + HOLD ? B0 : t < A + HOLD + B ? lg(B0, 0.02, (t - A - HOLD) / B) : 0.02;
};
interface Scene { sft: Float64Array; w: Float64Array; rm: Float64Array; x: number; ppo: Float64Array; gridBeta: number[]; proxy: number[]; gold: number[]; kls: number[]; gap: number }
interface F extends Frame { beta: number; p: Float64Array; hold: boolean; gold: number; proxy: number; kl: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const d = prefData(o, kit), x = int(o.prompt, 0), w = d.rm.w, rm = scores(w);
    const run = await ppoRun(o, kit, { beta: B0 });
    const ppo = probsInto(run.thetas[run.iters], x), star = optimalRow(d.sft, w, B0, x).p;
    let gap = 0;
    for (let y = 0; y < K; y++) gap = Math.max(gap, Math.abs(ppo[y] - star[y]));
    const gridBeta = Array.from({ length: GRID }, (_, i) => Math.exp(Math.log(1) + ((Math.log(0.02) - Math.log(1)) * i) / (GRID - 1)));
    const th = gridBeta.map((b) => optimalTheta(d.sft, w, b));
    return { sft: d.sft, w, rm, x, ppo, gridBeta, proxy: th.map((t) => expected(t, rm, TRAIN)), gold: th.map((t) => expected(t, GOLD, TRAIN)), kls: th.map((t) => kl(t, d.sft, TRAIN)), gap };
  },
  timeline: () => ({ duration: A + HOLD + B + TAIL, markers: [{ at: 0, label: 'Strong penalty' }, { at: A, label: "PPO's β" }, { at: A + HOLD, label: 'Weak penalty' }] }),
  frameAt(t, s) {
    const beta = betaAt(t), hold = t >= A && t <= A + HOLD, th = optimalTheta(s.sft, s.w, beta), p = optimalRow(s.sft, s.w, beta, s.x).p, ref = probsInto(s.sft, s.x);
    const gold = expected(th, GOLD, TRAIN), proxy = expected(th, s.rm, TRAIN), klv = kl(th, s.sft, TRAIN);
    let top = 0;
    for (let y = 1; y < K; y++) if (p[y] > p[top]) top = y;
    const status = hold
      ? `β = 0.1, the penalty PPO used. The optimum puts ${fmt(p[2], 2)} on a clear correct answer (the reference gave ${fmt(ref[2], 2)}); the ticks show PPO's final policy, never further than ${fmt(s.gap, 3)} from the bars.`
      : `β = ${fmt(beta, beta < 0.1 ? 3 : 2)}: π* puts ${fmt(p[top], 2)} on ${name(top)} (the reference gave it ${fmt(ref[top], 2)}). Across all training prompts it earns gold reward ${fmt(gold, 2)}.`;
    return { beta, p, hold, gold, proxy, kl: klv, status, readouts: [['β', fmt(beta, 3)], ['Gold reward of π*', fmt(gold, 2)], ['KL to the reference', fmt(klv, 2)], ['Rambling answer, this prompt', fmt(p[4], 2)]] };
  },
  restTime: () => A + HOLD / 2,
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12;
    text(E, clip(`User: ${PROMPTS[s.x].q}  Probability of each response`, charsFor((nw ? W : W * 0.5) - 2 * pad, 12)), pad, 18, { size: 12, fill: MUTED });
    const bh = nw ? 232 : H - 56;
    candidateBars(E, { x: pad, y: 30, w: (nw ? W : W * 0.5) - 2 * pad - 4, h: bh }, { p: f.p, ref: probsInto(s.sft, s.x), ticks: f.hold ? s.ppo : undefined, texts: TEXT[s.x], highlight: 4 });
    // reward chart against beta
    const cb = nw ? { x: 46, y: 30 + bh + 40, w: W - 46 - 16, h: H - (30 + bh + 40) - 44 } : { x: W * 0.5 + 44, y: 40, w: W * 0.5 - 44 - 16, h: H - 40 - 52 };
    const { sx, sy } = axes(E, cb, { lo: 0.02, hi: 1, log: true, ticks: [0.02, 0.05, 0.1, 0.2, 0.5, 1], fmt: (v) => String(v), label: 'β, the penalty strength' }, { lo: -1, hi: 5, ticks: [-1, 0, 1, 2, 3, 4, 5], label: 'Reward of π* (all prompts)' });
    const trace = (ys: number[], st: { stroke: string; sw: number; dash?: string }) => path(E, ys.map((v, i) => `${i ? 'L' : 'M'}${sx(s.gridBeta[i]).toFixed(1)} ${clamp(sy(v), cb.y, cb.y + cb.h).toFixed(1)}`).join(''), st);
    trace(s.proxy, { stroke: PROXY, sw: 2.25 });
    trace(s.gold, { stroke: INK, sw: 2, dash: GOLD_DASH });
    const mx = sx(f.beta);
    line(E, mx, cb.y, mx, cb.y + cb.h, { stroke: ACCENT, sw: 1.5, dash: '3 3' });
    circle(E, mx, clamp(sy(f.proxy), cb.y, cb.y + cb.h), 4, { fill: PROXY, stroke: 'var(--bg-raised)', sw: 1.5 });
    circle(E, mx, clamp(sy(f.gold), cb.y, cb.y + cb.h), 4, { fill: INK, stroke: 'var(--bg-raised)', sw: 1.5 });
    text(E, 'Reward model', cb.x + cb.w - 4, sy(s.proxy[0]) - 8, { size: 11, anchor: 'end', fill: PROXY, weight: 650, halo: true });
    text(E, 'Gold', cb.x + cb.w - 4, sy(s.gold[0]) + 16, { size: 11, anchor: 'end', fill: INK, weight: 650, halo: true });
  },
  label: (s) => `The closed-form optimal policy as the KL penalty beta falls from 1 to 0.02: bars for the eight responses to "${PROMPTS[s.x].q}" against the reference, ticks for PPO's end state at beta 0.1, and the reward-model and gold reward of the optimum against beta. ${CANDIDATES.length} candidates.`,
};
export default mode;
