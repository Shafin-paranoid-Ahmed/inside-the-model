/**
 * benchmark (evaluations step 1): six fictional models answer the same questions in order. Left: one cell per question
 * (blue right, orange wrong). Right: each model's score so far with its 95% interval and a dashed tick at its true score.
 * Props: models (6), items (200), seed (1). The intervals shrink as questions accumulate; neighbours still overlap at 200.
 */
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, RULE, TEXT, circle, fmt, line, text, type Env } from '../_rl/svg';
import { MODEL_NAMES, makeBench, overlaps, wilson, type Bench } from './sim';

const PLAY = 12, HOLD = 3, SHOWN = 6;
interface Scene extends Bench { k: number; seed: number }
interface F extends Frame { n: number; ints: [number, number][]; right: number[]; pairs: [number, number][] }

const cells = (E: Env, d: string, fill: string) => { if (d) E.out.push({ k: 'path', d, fill, stroke: 'none' }); };
export const pairName = ([a, b]: [number, number]) => `${MODEL_NAMES[a].slice(-1)} and ${MODEL_NAMES[b].slice(-1)}`;

const mode: ModeImpl<Scene, F> = {
  setup: (o: Opts) => { const k = Math.min(SHOWN, int(o.models, 6)), seed = int(o.seed, 1), items = int(o.items, 200); const b = makeBench(seed, items); return { ...b, k, seed }; },
  timeline: (s) => ({
    duration: PLAY + HOLD,
    markers: [{ at: 0, label: 'First 20 questions' }, { at: (PLAY * 20) / s.n, label: 'Up to 100' }, { at: (PLAY * 100) / s.n, label: 'Up to 200' }],
  }),
  frameAt(t, s) {
    const n = Math.max(1, Math.min(s.n, Math.round((t / PLAY) * s.n)));
    const right = Array.from({ length: s.k }, (_, m) => s.cum[m][n]);
    const ints = right.map((r) => wilson(r, n));
    const pairs: [number, number][] = [];
    for (let a = 0; a < s.k; a++) for (let b = a + 1; b < s.k; b++) if (overlaps(ints[a], ints[b])) pairs.push([a, b]);
    const total = (s.k * (s.k - 1)) / 2, sc = right.map((r) => r / n), best = sc.indexOf(Math.max(...sc)), worst = sc.indexOf(Math.min(...sc));
    const ov = pairs.length === total ? 'every pair of intervals overlaps' : `${pairs.length} of ${total} pairs of intervals overlap`;
    const status = n >= s.n && pairs.length <= 8 && pairs.length > 0
      ? `After all ${n} questions, ${ov}: ${pairs.map(pairName).join(', ')}. Scores run from ${fmt(sc[worst], 2)} (${MODEL_NAMES[worst]}) to ${fmt(sc[best], 2)} (${MODEL_NAMES[best]}).`
      : `After ${n} of ${s.n} questions the scores run from ${fmt(sc[worst], 2)} (${MODEL_NAMES[worst]}) to ${fmt(sc[best], 2)} (${MODEL_NAMES[best]}), and ${ov}.`;
    return { n, ints, right, pairs, status, readouts: [['Questions', `${n}`], ...right.map((r, m) => [MODEL_NAMES[m], `${r}/${n} = ${fmt(r / n, 2)} (${fmt(ints[m][0], 2)} to ${fmt(ints[m][1], 2)})`] as [string, string])] };
  },
  draw(E, f, s) {
    const nar = E.narrow, top = 26, rh = (E.H - top - 38) / s.k, labelW = nar ? 66 : 92, gx = labelW, gw = nar ? 108 : E.W * 0.33, fx = gx + gw + (nar ? 20 : 34), fw = E.W - fx - 12;
    const rowsN = nar ? 10 : 5, cols = Math.ceil(s.n / rowsN), cw = gw / cols, ch = Math.min(cw * 1.4, (rh - 8) / rowsN);
    const lo = 0.1, hi = 1, sx = (v: number) => fx + ((v - lo) / (hi - lo)) * fw;
    text(E, 'Questions answered', gx, top - 10, { size: 11, fill: MUTED });
    text(E, nar ? 'Share right' : 'Share answered right, with 95% interval', fx, top - 10, { size: 11, fill: MUTED });
    for (const v of [0.2, 0.4, 0.6, 0.8, 1]) {
      line(E, sx(v), top, sx(v), top + s.k * rh, { stroke: RULE, sw: 1, dash: '2 4' });
      text(E, fmt(v, 1), sx(v), top + s.k * rh + 15, { size: 11, anchor: 'middle', fill: MUTED });
    }
    let pos = '', neg = '';
    for (let m = 0; m < s.k; m++) {
      const yc = top + (m + 0.5) * rh, y0 = yc - (rowsN * ch) / 2;
      text(E, MODEL_NAMES[m], 2, yc - 3, { weight: 600, size: 12 });
      text(E, `${f.right[m]}/${f.n}`, 2, yc + 12, { size: 11, fill: MUTED });
      for (let i = 0; i < f.n; i++) {
        const c = `M${(gx + (i % cols) * cw + 0.4).toFixed(1)} ${(y0 + Math.floor(i / cols) * ch + 0.4).toFixed(1)}h${(cw - 0.8).toFixed(1)}v${(ch - 0.8).toFixed(1)}h${(0.8 - cw).toFixed(1)}z`;
        if (s.results[m][i]) pos += c; else neg += c;
      }
      const [a, b] = f.ints[m], p = f.right[m] / f.n;
      line(E, sx(s.pop[m]), yc - 11, sx(s.pop[m]), yc + 11, { stroke: TEXT, sw: 1.75, dash: '2 2' });
      line(E, sx(Math.max(lo, a)), yc, sx(Math.min(hi, b)), yc, { stroke: MUTED, sw: 4.5, op: 0.6 });
      circle(E, sx(Math.min(hi, Math.max(lo, p))), yc, 4.5, { fill: TEXT, stroke: BG, sw: 1.5 });
    }
    cells(E, pos, 'var(--viz-pos)'); cells(E, neg, 'var(--viz-neg)');
    const w = Math.max(...f.ints.map(([a, b]) => b - a));
    if (!nar) text(E, `Widest interval now: ${fmt(w, 2)} wide`, fx + fw, top + s.k * rh + 31, { size: 11, anchor: 'end', fill: MUTED });
  },
  label: (s) => `Six fictional models answering ${s.n} questions: a cell per answer, and each model's score with a 95% interval against its true score.`,
};
export default mode;
