/**
 * heldout (sft step 5): the SFT run of step 3 (same props), shown only on the two held-out prompts, which are never in the
 * demonstrations. Bars tween from the base model's chances toward the SFT model's as training proceeds; the base model is
 * the outline. Props: demos (24), demoAccuracy (0.8), seed (2), steps (150), lr (0.05), prompts ("8,9").
 * Measured: P(a correct answer) 0.224 -> 0.811 on "How many sides does a hexagon have?" and 0.159 -> 0.679 on "What is
 * 15 - 8?" (peaks of 0.845 and 0.722 near step 50).
 */
import { HELDOUT, PROMPTS } from '../../../lib/align/bank';
import { probsInto } from '../../../lib/align/policy';
import { nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, fmt, lerp, text, type Env } from '../_rl/svg';
import { candidateBars, charsFor, clip } from '../_align/draw';
import { sftRun } from './sim';

const SP = 0.08, FAST = 25, HOLD = 2.5;
interface Scene { xs: number[]; p: Float64Array[][]; steps: number }
interface F extends Frame { k: number; bars: Float64Array[]; vals: Float64Array[] }
const pCorrect = (p: Float64Array) => p[1] + p[2] + p[3] + p[7];

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const { run, steps } = sftRun(o, kit), xs = nums(o.prompts, HELDOUT).filter((x) => x >= 0 && x < PROMPTS.length);
    return { xs, steps, p: xs.map((x) => run.thetas.map((th) => probsInto(th, x))) };
  },
  timeline: (s) => ({ duration: s.steps * SP + HOLD, markers: [{ at: 0, label: 'Rise' }, { at: Math.min(FAST, s.steps) * SP, label: 'Settle' }] }),
  frameAt(t, s) {
    const x = Math.min(s.steps, t / SP), k = Math.floor(x + 1e-9), k1 = Math.min(s.steps, k + 1), w = x - k;
    const vals = s.p.map((rows) => rows[k]), bars = s.p.map((rows) => Float64Array.from(rows[k], (v, y) => lerp(v, rows[k1][y], w)));
    const base = s.xs.map((_, i) => fmt(pCorrect(s.p[i][0]), 2)), now = s.xs.map((_, i) => fmt(pCorrect(vals[i]), 2));
    const status = k === 0
      ? `Step 0, the base model, on prompts that never appear in the demonstrations: the chance of a correct answer is ${base.join(' and ')}.`
      : `Step ${k} of ${s.steps}: on prompts it never trained on, the chance of a correct answer has moved from ${base.join(' and ')} to ${now.join(' and ')}.`;
    return {
      k, bars, vals, status,
      readouts: [['Training step', `${k} of ${s.steps}`], ...s.xs.map((_, i): [string, string] => [`P(correct), prompt ${i + 1}`, `${base[i]} → ${now[i]}`])],
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, n = s.xs.length, gapX = 22;
    const pw = nw ? W - 2 * pad : (W - 2 * pad - gapX * (n - 1)) / n, ph = nw ? (H - 8) / n : H - 8;
    s.xs.forEach((x, i) => {
      const bx = nw ? pad : pad + i * (pw + gapX), by = nw ? 4 + i * ph : 4;
      text(E, clip(`User: ${PROMPTS[x].q}`, charsFor(pw, 13)), bx, by + 14, { size: 13, weight: 600 });
      candidateBars(E, { x: bx, y: by + 24, w: pw, h: ph - 24 - 34 }, { p: f.bars[i], values: f.vals[i], ref: s.p[i][0] });
      text(E, `Chance of a correct answer: ${fmt(pCorrect(s.p[i][0]), 2)} at the base model, ${fmt(pCorrect(f.vals[i]), 2)} now`, bx, by + ph - 12, { size: nw ? 11 : 12, fill: MUTED });
    });
  },
  label: (s) => `The two held-out prompts before and after fine-tuning: the chance of a correct answer rises from ${s.xs.map((_, i) => fmt(pCorrect(s.p[i][0]), 2)).join(' and ')} to ${s.xs.map((_, i) => fmt(pCorrect(s.p[i][s.steps]), 2)).join(' and ')}.`,
};
export default mode;
