/**
 * sl (constitutional-ai step 3): supervised learning on the revisions (SL-CAI). The SFT model writes 64 responses, the
 * critique-revise loop rewrites them, and the SFT model is fine-tuned on the revised responses (Adam, 150 steps) by the
 * same moment matching as the sft lesson. Bars: average probability of each response over the train prompts (ghost
 * outline = before fine-tuning, the terse wrong answer outlined); weight bars: theta against its starting value; chart:
 * expected principles broken per response against the step.
 * Props: n (64), rounds (4), seed (3), steps (150), from ("sft" | "base", default sft), principles.
 * Measured (defaults): 12 of 64 responses break a principle, 7 still do after revision; expected violations 0.207 -> 0.110, gold 2.273 ->
 * 2.508, terse wrong 0.138 -> 0.094, padded 0.005 -> 0.000, continues the list 0.057 -> 0.016.
 */
import { CANDIDATES, THETA_BASE, TRAIN, Y_WRONG, FEATURE_LABELS } from '../../../lib/align/bank';
import { slCAI, violationCount } from '../../../lib/align/constitution';
import { GOLD, expected } from '../../../lib/align/policy';
import { sftTheta } from '../../../lib/align/sft';
import { int, key, str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, clamp, fmt, lineChart, line, text, type Env } from '../_rl/svg';
import { candidateBars, featureBars } from '../_align/draw';
import { avgProbs, expViolations } from './cr-common';
import { principlesOf } from './sim';

const LEAD = 0.8, RUN = 8, HOLD = 2.4;
interface Scene { steps: number; probs: Float64Array[]; viol: Float64Array; gold: Float64Array; thetas: Float64Array[]; broke: number; left: number; n: number }
interface F extends Frame { k: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const act = principlesOf(o), n = int(o.n, 64), rounds = int(o.rounds, 4), seed = int(o.seed, 3), steps = Math.max(1, int(o.steps, 150)), from = str(o.from, ['sft', 'base'] as const, 'sft');
    return kit.memo(key('cr-sl', act, n, rounds, seed, steps, from), () => {
      const sft = sftTheta(), start = from === 'base' ? THETA_BASE : sft, res = slCAI(start, { n, rounds, seed, steps, active: act });
      const thetas = res.run.thetas;
      return {
        steps, thetas, n, broke: res.starts.filter((s) => violationCount(s.x, s.y0, act) > 0).length, left: res.starts.filter((s) => violationCount(s.x, s.y, act) > 0).length,
        probs: thetas.map(avgProbs), viol: Float64Array.from(thetas, (th) => expViolations(th, act)), gold: Float64Array.from(thetas, (th) => expected(th, GOLD, TRAIN)),
      };
    });
  },
  timeline: () => ({ duration: LEAD + RUN + HOLD, markers: [{ at: 0, label: 'Before' }, { at: LEAD, label: 'Fine-tuning' }, { at: LEAD + RUN, label: 'After' }] }),
  frameAt(t, s) {
    const k = Math.floor(clamp((t - LEAD) / RUN, 0, 1) * s.steps + 1e-9), p = s.probs[k], p0 = s.probs[0];
    const name = CANDIDATES[Y_WRONG].label.toLowerCase();
    const status = k === 0 ? `Before fine-tuning: the model breaks ${fmt(s.viol[0], 3)} principles per response on average, and writes the ${name} answer with probability ${fmt(p0[Y_WRONG], 2)}.`
      : k < s.steps ? `Step ${k} of ${s.steps}: the weights move toward the revised responses; principles broken per response ${fmt(s.viol[k], 3)}.`
        : `After ${s.steps} steps: ${fmt(s.viol[k], 3)} principles broken per response (from ${fmt(s.viol[0], 3)}), and the ${name} answer is down to ${fmt(p[Y_WRONG], 2)} (from ${fmt(p0[Y_WRONG], 2)}).`;
    return {
      k, status,
      readouts: [['Step', `${k} of ${s.steps}`], ['Principles broken per response', fmt(s.viol[k], 3)], ['Gold reward', fmt(s.gold[k], 2)], [`Chance of ${name}`, fmt(p[Y_WRONG], 2)]],
      detail: `Training data: ${s.n} responses written by the SFT model; ${s.broke} broke a principle, and ${s.left} still do after revision. Before fine-tuning: ${fmt(s.viol[0], 3)} principles broken, gold ${fmt(s.gold[0], 2)}.`,
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, p = s.probs[f.k];
    text(E, 'Chance of each response', pad, 14, { size: 11, fill: MUTED });
    const barsW = nw ? W - 2 * pad : Math.round((W - 2 * pad) * 0.5), barsH = nw ? 200 : H - 34;
    candidateBars(E, { x: pad + 4, y: 22, w: barsW - 8, h: barsH }, { p, ref: s.probs[0], highlight: Y_WRONG });
    if (nw) {
      lineChart(E, { x: pad + 36, y: 22 + barsH + 34, w: W - 2 * pad - 44, h: H - barsH - 22 - 34 - 56 }, { lo: 0, hi: s.steps, ticks: [0, s.steps / 2, s.steps], label: 'Fine-tuning step' }, { lo: 0, hi: Math.max(0.1, Math.ceil(s.viol[0] * 10) / 10 + 0.05), label: 'Principles broken per response' }, [{ ys: s.viol, color: 'var(--viz-neg)', width: 2.25 }], f.k);
      return;
    }
    const rx = pad + barsW + 24, rw = W - rx - pad, fh = 150;
    featureBars(E, { x: rx, y: 14, w: rw, h: fh }, { v: s.thetas[f.k], ref: s.thetas[0], lim: 7, title: 'Weights (tick = before)', labels: FEATURE_LABELS });
    lineChart(E, { x: rx + 36, y: 14 + fh + 40, w: rw - 44, h: H - fh - 14 - 40 - 42 }, { lo: 0, hi: s.steps, ticks: [0, s.steps / 2, s.steps], label: 'Fine-tuning step' }, { lo: 0, hi: Math.max(0.1, Math.ceil(s.viol[0] * 10) / 10 + 0.05), label: 'Principles broken per response' }, [{ ys: s.viol, color: 'var(--viz-neg)', width: 2.25 }], f.k);
    line(E, rx, 14 + fh + 6, rx + rw, 14 + fh + 6, { stroke: 'var(--rule)', sw: 1 });
  },
  label: (s) => `Fine-tuning the SFT model on ${s.n} revised responses: expected principles broken per response fall from ${fmt(s.viol[0], 3)} to ${fmt(s.viol[s.steps], 3)} and gold reward rises from ${fmt(s.gold[0], 2)} to ${fmt(s.gold[s.steps], 2)}.`,
};
export default mode;
