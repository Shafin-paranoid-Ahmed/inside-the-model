/**
 * saturation (evaluations step 5): eight model generations, each a little more skilled than the last, take an easy test and
 * a hard test of 200 questions each. A few answer keys are wrong (labelErrors), so no model can score above 1 minus that share.
 * Props: generations (8), items (200), labelErrors (0.03), seed (6).
 */
import { int, key, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, TEXT, axes, circle, ease, fmt, line, seg, text, type Env } from '../_rl/svg';
import { createRng, difficulties, overlaps, sigmoid, wilson } from './sim';

export const STEP = 1.0, START = 0.2, EASY = { mu: -1.5, sd: 1 }, HARD = { mu: 3.8, sd: 1.2 };
const PER = 1.6, HOLD = 2;
export interface Bench2 { right: Int32Array; ints: [number, number][]; sep: boolean[] }
export interface Scene { gens: number; items: number; ceiling: number; wrongKeys: number; easy: Bench2; hard: Bench2 }

function run(gens: number, items: number, wrongKeys: number, seed: number, mu: number, sd: number): Bench2 {
  const rng = createRng(seed), d = difficulties(items, rng, mu, sd), right = new Int32Array(gens);
  for (let g = 0; g < gens; g++) { let k = 0; for (let i = wrongKeys; i < items; i++) k += rng.next() < sigmoid(START + STEP * g - d[i]) ? 1 : 0; right[g] = k; }
  const ints = Array.from(right, (k) => wilson(k, items)), sep = ints.map((iv, g) => g === 0 || !overlaps(iv, ints[g - 1]));
  return { right, ints, sep };
}
export function makeSaturation(gens: number, items: number, labelErrors: number, seed: number): Scene {
  const wrongKeys = Math.round(items * labelErrors);
  return { gens, items, wrongKeys, ceiling: 1 - wrongKeys / items, easy: run(gens, items, wrongKeys, seed, EASY.mu, EASY.sd), hard: run(gens, items, wrongKeys, seed + 100, HARD.mu, HARD.sd) };
}
/** First generation (1-based) from which every later generation overlaps its predecessor on this test, or 0 if none does. */
export const lastSeparated = (b: Bench2) => { let last = 0; b.sep.forEach((ok, g) => { if (ok) last = g + 1; }); return last; };

interface F extends Frame { g: number; u: number }
const mode: ModeImpl<Scene, F> = {
  setup: (o: Opts, kit: Kit) => { const a = [int(o.generations, 8), int(o.items, 200), num(o.labelErrors, 0.03), int(o.seed, 6)] as const; return kit.memo(key('saturation', ...a), () => makeSaturation(...a)); },
  timeline: (s) => ({ duration: PER * s.gens + HOLD, markers: Array.from({ length: s.gens }, (_, g) => ({ at: g * PER, label: `Gen ${g + 1}` })) }),
  frameAt(t, s) {
    const { i, u } = seg(t, PER, s.gens), g = i + 1, e = s.easy.right[i] / s.items, h = s.hard.right[i] / s.items;
    const sepE = s.easy.sep[i], sepH = s.hard.sep[i];
    const status = `Generation ${g} scores ${fmt(e, 3)} on the easy test (best possible ${fmt(s.ceiling, 2)}) and ${fmt(h, 3)} on the hard test. ` + (g === 1 ? 'Nothing to compare with yet.' : `Against generation ${g - 1}, the easy-test intervals ${sepE ? 'do not overlap' : 'overlap'} and the hard-test intervals ${sepH ? 'do not overlap' : 'overlap'}.`);
    return { g, u, status, readouts: [['Generation', `${g}`], ['Easy test', fmt(e, 3)], ['Hard test', fmt(h, 3)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, W = E.W, pw = nar ? W - 56 : (W - 120) / 2, ph = nar ? (E.H - 150) / 2 : E.H - 80;
    const panels = [{ name: 'Easy test', b: s.easy, x: nar ? 44 : 50, y: 34 }, { name: 'Hard test', b: s.hard, x: nar ? 44 : 50 + pw + 60, y: nar ? 34 + ph + 66 : 34 }];
    for (const pn of panels) {
      const box = { x: pn.x, y: pn.y, w: pw, h: ph };
      text(E, pn.name, pn.x, pn.y - 14, { weight: 600, size: 12 });
      const { sx, sy } = axes(E, box, { lo: 0.5, hi: s.gens + 0.5, label: nar && pn.name === 'Easy test' ? undefined : 'Generation', ticks: Array.from({ length: s.gens }, (_, i) => i + 1) }, { lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => fmt(v, 2) });
      line(E, box.x, sy(s.ceiling), box.x + box.w, sy(s.ceiling), { stroke: TEXT, sw: 1.5, dash: '5 4' });
      text(E, `best possible ${fmt(s.ceiling, 2)}`, box.x + 6, sy(s.ceiling) - 5, { size: 11, fill: MUTED });
      for (let g = 0; g < s.gens; g++) {
        if (g > f.g - 1) break;
        const iv = pn.b.ints[g], p = pn.b.right[g] / s.items, a = g === f.g - 1 ? ease(f.u * 3) : 1, col = pn.b.sep[g] ? 'var(--viz-forward)' : ACCENT, x = sx(g + 1);
        line(E, x, sy(iv[0]), x, sy(iv[1]), { stroke: col, sw: 4.5, op: 0.6 * a });
        circle(E, x, sy(p), 5, { fill: col, stroke: BG, sw: 1.5, op: a });
      }
    }
  },
  label: (s) => `Scores of ${s.gens} model generations with 95% intervals on an easy and a hard test of ${s.items} questions; the easy test levels off near ${fmt(s.ceiling, 2)}.`,
};
export default mode;
