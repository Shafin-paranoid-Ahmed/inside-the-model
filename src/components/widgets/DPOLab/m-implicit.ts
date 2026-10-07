/**
 * implicit (dpo step 4): the DPO policy's implicit reward beta (log pi - log pi_ref) of every candidate (10 prompts x 8)
 * against the separately trained reward model's score, over the 300 DPO steps. Scores are only defined up to a constant per
 * prompt (Bradley-Terry), so both axes are centred within each prompt. Right: the implicit weights beta (theta - theta_ref)
 * (bars) against the reward model's weights (ticks). Props: beta (0.1), steps (300), seed (4, pair seed).
 */
import { FEATURE_LABELS, K, P } from '../../../lib/align/bank';
import { implicitReward } from '../../../lib/align/dpo';
import { scores } from '../../../lib/align/policy';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, axes, circle, clamp, fmt, line, text, type Env } from '../_rl/svg';
import { FORWARD, featureBars } from '../_align/draw';
import { betaOf, centred, dpoRun, pearson } from './dpoKit';
import { prefData } from './sim';

const LEAD = 1.5, RUN = 13, TAIL = 2.5;
interface Scene { beta: number; steps: number; w: Float64Array; rm: Float64Array; impl: Float64Array[]; corr: Float64Array; v: Float64Array[]; lim: number; wlim: number }
interface F extends Frame { n: number; worst: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const d = prefData(o, kit, int(o.seed, 4)), beta = betaOf(o), steps = int(o.steps, 300), run = await dpoRun(kit, d, beta, steps), w = d.rm.w, rm = centred(scores(w));
    const impl = run.thetas.map((th) => centred(implicitReward(th, d.sft, beta)));
    const v = run.thetas.map((th) => Float64Array.from(th, (t, j) => beta * (t - d.sft[j])));
    const corr = Float64Array.from(impl, (a, k) => (k ? pearson(a, rm) : 0));
    const mx = Math.max(...Array.from(rm, Math.abs), ...Array.from(impl[steps], Math.abs));
    return { beta, steps, w, rm, impl, corr, v, lim: Math.ceil(mx), wlim: Math.ceil(Math.max(...Array.from(w, Math.abs), ...Array.from(v[steps], Math.abs))) };
  },
  timeline: (s) => ({ duration: LEAD + RUN + TAIL, markers: [{ at: 0, label: 'Reference' }, { at: LEAD, label: 'Training' }, { at: LEAD + RUN, label: `After ${s.steps} steps` }] }),
  frameAt(t, s) {
    const n = clamp(Math.round(((t - LEAD) / RUN) * s.steps), 0, s.steps), v = s.v[n];
    let worst = 0;
    for (let j = 1; j < v.length; j++) if (Math.abs(v[j] - s.w[j]) > Math.abs(v[worst] - s.w[worst])) worst = j;
    const status = n === 0
      ? 'At the start the policy is the reference, so every implicit reward is 0: the log of π over π_ref is 0 for every answer.'
      : `Step ${n}: the implicit reward of all ${P * K} candidate answers lines up with the reward model's score, correlation ${fmt(s.corr[n], 3)}. The weight that differs most from the reward model's is "${FEATURE_LABELS[worst].toLowerCase()}": ${fmt(v[worst], 2)} against ${fmt(s.w[worst], 2)}.`;
    return { n, worst, status, readouts: [['Step', `${n} of ${s.steps}`], ['Correlation', n ? fmt(s.corr[n], 3) : '–'], ['Largest weight gap', `${FEATURE_LABELS[worst].toLowerCase()} ${fmt(Math.abs(v[worst] - s.w[worst]), 2)}`]] };
  },
  restTime: () => LEAD + RUN + 0.5,
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, L = s.lim;
    const sb = nw ? { x: 44, y: 34, w: W - 44 - 16, h: 232 } : { x: 50, y: 34, w: W * 0.5 - 50 - 14, h: H - 34 - 52 };
    const { sx, sy } = axes(E, sb, { lo: -L, hi: L, label: 'Reward model score (centred within each prompt)' }, { lo: -L, hi: L, label: 'Implicit reward β log(π / π_ref), centred' });
    line(E, sx(-L), sy(-L), sx(L), sy(L), { stroke: MUTED, sw: 1.25, dash: '4 4' });
    const ys = s.impl[f.n];
    for (let i = 0; i < P * K; i++) if (i % K !== 4) circle(E, sx(s.rm[i]), sy(ys[i]), 3.5, { fill: FORWARD, op: 0.7 });
    for (let i = 4; i < P * K; i += K) circle(E, sx(s.rm[i]), sy(ys[i]), 4.5, { fill: ACCENT, stroke: BG, sw: 1 });
    // weights
    const wb = nw ? { x: pad, y: sb.y + sb.h + 62, w: W - 2 * pad, h: H - (sb.y + sb.h + 62) - 12 } : { x: W * 0.5 + 18, y: 52, w: W * 0.5 - 18 - 14, h: H - 52 - 52 };
    text(E, 'Weights: DPO policy (bars) and reward model (ticks)', wb.x, wb.y - 14, { size: 12, fill: MUTED });
    featureBars(E, wb, { v: s.v[f.n], ref: s.w, lim: s.wlim, digits: 2 });
  },
  label: (s) => `Implicit reward of the DPO policy against the reward model's score for all 80 candidate answers, and the six implicit weights against the reward model's weights, over ${s.steps} DPO steps; final correlation ${fmt(s.corr[s.steps], 3)}.`,
};
export default mode;
