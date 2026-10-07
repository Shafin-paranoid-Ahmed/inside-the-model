/**
 * verify (rlvr step 1): six generated responses to addition problems and a program verifier parsing each. A response
 * types in line by line; the verifier highlights what it reads (strict: the last line; lenient: every number) and a tick
 * or cross appears. Props: verifier ("strict" | "lenient"). Strict verdicts: pass, fail (a dropped carry: 22 for
 * 47 + 85), pass (no carry needed), fail (correct working, but the last line is a list of guesses), fail (533 for
 * 365 + 278), pass. Lenient passes the guess list too.
 */
import { parseLenient, parseStrict, respond, truth, verify, actionOf, type Problem, type VerifierKind } from '../../../lib/align/verifier';
import { str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, RULE, clamp, mix, rect, text, type Env } from '../_rl/svg';
import { charsFor, clip } from '../_align/draw';
import { cross, tick } from './sim';

const EXAMPLES: { p: Problem; action: number }[] = [
  { p: { a: 47, b: 85, digits: 2 }, action: actionOf(2, false) },
  { p: { a: 47, b: 85, digits: 2 }, action: actionOf(0, false) },
  { p: { a: 36, b: 21, digits: 2 }, action: actionOf(0, false) },
  { p: { a: 58, b: 67, digits: 2 }, action: actionOf(2, true) },
  { p: { a: 365, b: 278, digits: 3 }, action: actionOf(1, false) },
  { p: { a: 9, b: 4, digits: 1 }, action: actionOf(1, false) },
];
const TYPE = 1.2, PARSE = 0.5, VERDICT = 0.6, PER = TYPE + PARSE + VERDICT, HOLD = 3;
interface Ex { p: Problem; text: string; lines: string[]; sum: number; pass: boolean; parsed: string }
interface Scene { kind: VerifierKind; ex: Ex[] }
interface F extends Frame { i: number; u: number; done: boolean }

const reason = (e: Ex, kind: VerifierKind) => {
  if (kind === 'lenient') return e.pass ? `${e.sum} appears in the response` : `${e.sum} appears nowhere`;
  const n = parseStrict(e.text);
  return n === null ? 'the last line is not “Answer: N”' : n === e.sum ? `Answer ${n} = ${e.sum}` : `Answer ${n}, but the sum is ${e.sum}`;
};

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const kind = str(o.verifier, ['strict', 'lenient'] as const, 'strict');
    return {
      kind, ex: EXAMPLES.map(({ p, action }) => {
        const t = respond(p, action), sum = truth(p), lines = t.split('\n');
        return { p, text: t, lines, sum, pass: verify(t, sum, kind), parsed: kind === 'strict' ? String(parseStrict(t) ?? 'nothing') : parseLenient(t).join(', ') };
      }),
    };
  },
  timeline: (s) => ({ duration: s.ex.length * PER + HOLD, markers: s.ex.map((_, i) => ({ at: i * PER, label: `Response ${i + 1}` })) }),
  frameAt(t, s) {
    const n = s.ex.length, done = t >= n * PER, i = done ? n - 1 : Math.floor(t / PER), u = done ? PER : t - i * PER, e = s.ex[i];
    const verdicts = s.ex.slice(0, done ? n : i + (u >= TYPE + PARSE ? 1 : 0));
    const status = u < TYPE ? `Response ${i + 1} to ${e.p.a} + ${e.p.b} is being written.`
      : u < TYPE + PARSE ? (s.kind === 'strict' ? `The verifier reads only the last line: “${e.lines[e.lines.length - 1]}”.` : `The lenient verifier collects every number in the response: ${e.parsed}.`)
      : `${e.pass ? 'Pass' : 'Fail'}: ${reason(e, s.kind)}.`;
    return { i, u, done, status, readouts: [['Verifier', s.kind === 'strict' ? 'strict (last line)' : 'lenient (anywhere)'], ['Checked', `${verdicts.length} of ${n}`], ['Passed', String(verdicts.filter((x) => x.pass).length)]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, pad = nw ? 8 : 12, cols = nw ? 1 : 2, gap = 10, n = s.ex.length, rows = Math.ceil(n / cols);
    const cw = (W - 2 * pad - (cols - 1) * gap) / cols, ch = (E.H - 8 - (rows - 1) * gap) / rows, sz = nw ? 11 : 12, lh = sz * 1.4;
    s.ex.forEach((e, j) => {
      const x0 = pad + (j % cols) * (cw + gap), y0 = 4 + Math.floor(j / cols) * (ch + gap), cur = j === f.i && !f.done, future = !f.done && j > f.i;
      const typed = future ? 0 : cur ? clamp(f.u / TYPE, 0, 1) : 1, judged = !future && (!cur || f.u >= TYPE + PARSE), parsing = cur && f.u >= TYPE;
      rect(E, x0, y0, cw, ch, { fill: mix('var(--text)', BG, 4), stroke: cur ? ACCENT : RULE, sw: cur ? 2 : 1, rx: 6, op: future ? 0.45 : 1 });
      text(E, `${e.p.a} + ${e.p.b}`, x0 + 10, y0 + 17, { size: 12, weight: 650, op: future ? 0.5 : 1 });
      if (judged) text(E, `sum ${e.sum}`, x0 + cw - 34, y0 + 17, { size: 11, anchor: 'end', fill: MUTED });
      // the response, typed character by character (monospace, so widths are stable)
      const total = e.text.length, shown = Math.floor(typed * total), maxc = charsFor(cw - 20, sz, true);
      let used = 0;
      e.lines.forEach((l, k) => {
        const vis = l.slice(0, Math.max(0, shown - used)), ly = y0 + 26 + (k + 1) * lh - 4, isLast = k === e.lines.length - 1;
        used += l.length + 1;
        if (parsing && (s.kind === 'strict' ? isLast : true)) rect(E, x0 + 6, ly - sz, Math.min(cw - 12, (s.kind === 'strict' ? l.length : vis.length) * sz * 0.61 + 8), lh, { fill: mix(ACCENT, BG, 22), rx: 3 });
        if (vis) text(E, clip(vis, maxc), x0 + 10, ly, { size: sz, mono: true });
      });
      if (judged) (e.pass ? tick : cross)(E, x0 + cw - 16, y0 + 13, 7);
    });
  },
  label: (s) => `Six generated responses to addition problems checked by a ${s.kind} verifier: ${s.ex.filter((e) => e.pass).length} pass.`,
};
export default mode;
