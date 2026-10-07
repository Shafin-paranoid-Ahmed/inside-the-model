/**
 * base (sft step 1): the base model's choice among the 8 candidates, prompt by prompt. Bars tween from one train prompt
 * to the next; then one seeded sample drops into the answer slot. The "continues the question list" row is outlined.
 * Props: seed (sampling, default 1), prompts ("0,1,...,7" = the train prompts). Measured: the most likely response is
 * "continues the list" on all 8 train prompts; average P(continue) 0.575, average P(a correct answer) 0.185.
 */
import { CANDIDATES, PROMPTS, TEXT, THETA_BASE, TRAIN, Y_CONTINUE, Y_REFUSAL, F_CORRECT, phi } from '../../../lib/align/bank';
import { expectedFeatures, meanProb, probsInto, sampleFrom } from '../../../lib/align/policy';
import { createRng } from '../../../lib/math/rng';
import { int, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, clamp, ease, fmt, lerp, seg, text, type Env } from '../_rl/svg';
import { candidateBars, charsFor, clip, responseCard } from '../_align/draw';

const PER = 2.6, TWEEN = 0.6, DROP = 1.1, HOLD = 2;
interface Scene { xs: number[]; p: Float64Array[]; sample: number[]; pc: number; pcorrect: number }
interface F extends Frame { k: number; bars: Float64Array; vals: Float64Array; dropped: boolean; u: number }

const correctProb = (p: Float64Array) => p[1] + p[2] + p[3] + p[7];
const top = (p: Float64Array) => p.indexOf(Math.max(...p));

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const xs = nums(o.prompts, TRAIN).filter((x) => x >= 0 && x < PROMPTS.length), rng = createRng(int(o.seed, 1));
    const p = xs.map((x) => probsInto(THETA_BASE, x));
    return { xs, p, sample: p.map((row) => sampleFrom(row, rng.next())), pc: meanProb(THETA_BASE, Y_CONTINUE, xs), pcorrect: expectedFeatures(THETA_BASE, xs)[F_CORRECT] };
  },
  timeline: (s) => ({ duration: s.xs.length * PER + HOLD, markers: s.xs.map((_, i) => ({ at: i * PER, label: `Prompt ${i + 1}` })) }),
  frameAt(t, s) {
    const n = s.xs.length, { i: k, u } = seg(Math.min(t, n * PER - 1e-6), PER, n), w = k === 0 ? 1 : ease(u * PER / TWEEN);
    const prev = s.p[Math.max(0, k - 1)], cur = s.p[k], bars = Float64Array.from(cur, (v, y) => lerp(prev[y], v, w)), dropped = u * PER >= DROP || t >= n * PER;
    const x = s.xs[k], q = PROMPTS[x].q, best = top(cur), onTop = s.p.slice(0, k + 1).filter((r) => top(r) !== Y_CONTINUE && top(r) !== Y_REFUSAL).length;
    const status = dropped
      ? `Sampled: “${CANDIDATES[s.sample[k]].label.toLowerCase()}”. ${s.sample[k] === Y_CONTINUE ? 'Instead of answering, it writes more questions.' : s.sample[k] === Y_REFUSAL ? 'It refuses a harmless question.' : phi(x, s.sample[k])[F_CORRECT] ? 'This time it happens to answer correctly.' : 'It answers, but wrongly.'}`
      : `Prompt ${k + 1} of ${n}: “${q}” The most likely response is “${CANDIDATES[best].label.toLowerCase()}”, with probability ${fmt(cur[best], 2)}.`;
    return {
      k, bars, vals: cur, dropped, u, status,
      readouts: [['Prompt', `${k + 1} of ${n}`], ['P(continues the list) here', fmt(cur[Y_CONTINUE], 2)], ['P(a correct answer) here', fmt(correctProb(cur), 2)], ['Prompts so far with an answer on top', `${onTop} of ${k + 1}`]],
      detail: `Averaged over the ${n} prompts: P(continues the list) ${fmt(s.pc, 3)}, P(a correct answer) ${fmt(s.pcorrect, 3)}.`,
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, x = s.xs[f.k], pad = nw ? 8 : 12;
    responseCard(E, { x: pad, y: 6, w: W - 2 * pad, h: 30 }, { text: `User: ${PROMPTS[x].q}`, size: nw ? 12 : 13 });
    const rowsTop = 48, slotH = nw ? 96 : 64, rowsH = E.H - rowsTop - slotH - 22;
    text(E, 'Chance of each response', W - pad, rowsTop - 2, { size: 11, anchor: 'end', fill: MUTED });
    candidateBars(E, { x: pad + 4, y: rowsTop + 2, w: W - 2 * pad - 8, h: rowsH }, { p: f.bars, values: f.vals, texts: TEXT[x], highlight: Y_CONTINUE });
    const sy = E.H - slotH - 8;
    if (f.dropped) {
      const y = s.sample[f.k], a = clamp((f.u * PER - DROP) / 0.35, 0, 1), dy = (1 - ease(a)) * -18;
      responseCard(E, { x: pad, y: sy + dy, w: W - 2 * pad, h: slotH }, { title: `Sampled response: ${CANDIDATES[y].label.toLowerCase()}`, text: clip(TEXT[x][y], charsFor(W - 2 * pad - 16, 12) * (nw ? 4 : 2)), accent: true, op: 0.3 + 0.7 * ease(a), maxLines: nw ? 4 : 2, size: 12 });
    } else responseCard(E, { x: pad, y: sy, w: W - 2 * pad, h: slotH }, { title: 'Sampled response', text: '…', op: 0.5 });
  },
  label: (s) => `The base model's probabilities for 8 candidate responses on each of ${s.xs.length} prompts: on average it continues the question list with probability ${fmt(s.pc, 2)} and answers correctly with probability ${fmt(s.pcorrect, 2)}.`,
};
export default mode;
