/**
 * contamination (evaluations step 3): one model (true score about 0.67) takes a public test of 400 questions, some of which
 * leaked into its training data (it then answers a leaked question right 97% of the time), and a fresh test of 400 new
 * questions of the same kind. The leak share sweeps from 0 to the `leakPct` control; the public score climbs, the fresh one does not.
 * Props: items (400), leakPct (20), seed (1).
 */
import { int, key, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, TEXT, axes, circle, fmt, line, path, signFmt, text, type Env } from '../_rl/svg';
import { createRng, difficulties, sigmoid } from './sim';

const SKILL = 0.9, MEMO = 0.97, SWEEP = 8, HOLD = 3;
export interface Scene {
  n: number; nat: Uint8Array; mem: Uint8Array; fresh: number; order: Int32Array; leakRank: Int32Array;
  /** pub[L] = right answers on the public test when the first L questions (in leak order) have leaked. */
  pub: Int32Array;
}
export function makeContamination(items: number, seed: number): Scene {
  const rng = createRng(seed), d = difficulties(items, rng), nat = new Uint8Array(items), mem = new Uint8Array(items);
  for (let i = 0; i < items; i++) { nat[i] = rng.next() < sigmoid(SKILL - d[i]) ? 1 : 0; mem[i] = rng.next() < MEMO ? 1 : 0; }
  const order = Int32Array.from({ length: items }, (_, i) => i);
  for (let i = items - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const leakRank = new Int32Array(items); order.forEach((q, r) => { leakRank[q] = r; });
  const fd = difficulties(items, rng); let fresh = 0; for (let i = 0; i < items; i++) fresh += rng.next() < sigmoid(SKILL - fd[i]) ? 1 : 0;
  const pub = new Int32Array(items + 1); pub[0] = nat.reduce((a, b) => a + b, 0);
  for (let L = 0; L < items; L++) pub[L + 1] = pub[L] - nat[order[L]] + mem[order[L]];
  return { n: items, nat, mem, fresh, order, leakRank, pub };
}
/** Leaked count at a whole percent. */
export const leakedAt = (n: number, pct: number) => Math.round((n * pct) / 100);

interface Sc extends Scene { max: number }
interface F extends Frame { pct: number; L: number; pubScore: number; freshScore: number }

const mode: ModeImpl<Sc, F> = {
  setup(o: Opts, kit: Kit) { const n = int(o.items, 400), seed = int(o.seed, 1); return { ...kit.memo(key('contam', n, seed), () => makeContamination(n, seed)), max: Math.max(0, Math.min(50, int(num(o.leakPct, 20), 20))) }; },
  timeline: (s) => ({ duration: SWEEP + HOLD, markers: s.max ? [{ at: 0, label: 'No leak' }, { at: 0.8, label: 'Leak grows' }, { at: SWEEP, label: `Leak at ${s.max}%` }] : [{ at: 0, label: 'No leak' }] }),
  frameAt(t, s) {
    const pct = Math.floor(s.max * Math.max(0, Math.min(1, t / SWEEP)) + 1e-9), L = leakedAt(s.n, pct), pub = s.pub[L], pubScore = pub / s.n, freshScore = s.fresh / s.n, gap = pubScore - freshScore;
    const status = pct === 0
      ? `Nothing has leaked. The model scores ${fmt(pubScore, 3)} on the public test and ${fmt(freshScore, 3)} on fresh questions of the same kind: a gap of ${signFmt(gap, 3)}, and with no leak the two tests agree up to chance.`
      : `With ${pct}% of the public test leaked (${L} questions), the model scores ${fmt(pubScore, 3)} on the public test but ${fmt(freshScore, 3)} on fresh questions: a gap of ${signFmt(gap, 3)}.`;
    return { pct, L, pubScore, freshScore, status, readouts: [['Leaked', `${pct}% (${L} of ${s.n})`], ['Public test', fmt(pubScore, 3)], ['Fresh test', fmt(freshScore, 3)], ['Gap', signFmt(gap, 3)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gs = nar ? Math.min(E.W - 24, 200) : Math.min(E.W * 0.34, E.H - 70), gx = nar ? (E.W - gs) / 2 : 8, gy = 30, cols = Math.ceil(Math.sqrt(s.n)), cell = gs / cols;
    text(E, nar ? 'Public test (ringed = leaked)' : 'Public test: one cell per question (ringed and dotted = leaked)', gx, gy - 12, { size: 11, fill: MUTED });
    let pos = '', neg = '', ring = ''; const dots: [number, number][] = [];
    for (let i = 0; i < s.n; i++) {
      const leaked = s.leakRank[i] < f.L, right = leaked ? s.mem[i] : s.nat[i], x = gx + (i % cols) * cell, y = gy + Math.floor(i / cols) * cell;
      const c = `M${(x + 0.5).toFixed(1)} ${(y + 0.5).toFixed(1)}h${(cell - 1).toFixed(1)}v${(cell - 1).toFixed(1)}h${(1 - cell).toFixed(1)}z`;
      if (right) pos += c; else neg += c;
      if (leaked) { ring += `M${(x + 1.3).toFixed(1)} ${(y + 1.3).toFixed(1)}h${(cell - 2.6).toFixed(1)}v${(cell - 2.6).toFixed(1)}h${(2.6 - cell).toFixed(1)}z`; dots.push([x + cell / 2, y + cell / 2]); }
    }
    E.out.push({ k: 'path', d: pos, fill: 'var(--viz-pos)', op: 0.9 }, { k: 'path', d: neg, fill: 'var(--viz-neg)', op: 0.9 });
    if (ring) E.out.push({ k: 'path', d: ring, fill: 'none', stroke: BG, sw: 3.4 }, { k: 'path', d: ring, fill: 'none', stroke: TEXT, sw: 1.7 });
    if (dots.length) E.out.push({ k: 'path', d: dots.map(([x, y]) => `M${(x - 1.6).toFixed(1)} ${y.toFixed(1)}a1.6 1.6 0 1 0 3.2 0a1.6 1.6 0 1 0 -3.2 0`).join(''), fill: BG, stroke: TEXT, sw: 0.8 });
    // score against leak share
    const b = nar ? { x: 48, y: gy + gs + 52, w: E.W - 48 - 60, h: E.H - gy - gs - 52 - 56 } : { x: gx + gs + 62, y: 34, w: E.W - (gx + gs + 62) - 60, h: E.H - 34 - 54 };
    const maxPct = Math.max(10, s.max), series = (p: number) => s.pub[leakedAt(s.n, p)] / s.n, fresh = s.fresh / s.n;
    let ylo = fresh, yhi = fresh; for (let p = 0; p <= maxPct; p++) { ylo = Math.min(ylo, series(p)); yhi = Math.max(yhi, series(p)); }
    const lo = Math.floor((ylo - 0.03) * 20) / 20, hi = Math.ceil((yhi + 0.03) * 20) / 20;
    const yt = Array.from({ length: Math.round((hi - lo) / 0.05) + 1 }, (_, i) => Math.round((lo + i * 0.05) * 100) / 100);
    const { sx, sy } = axes(E, b, { lo: 0, hi: maxPct, label: 'Share of the public test that leaked (%)', ticks: [0, 10, 20, 30, 40, 50].filter((v) => v <= maxPct) }, { lo, hi, label: 'Score', ticks: yt, fmt: (v) => fmt(v, 2) });
    line(E, b.x, sy(fresh), b.x + b.w, sy(fresh), { stroke: 'var(--viz-cat-7)', sw: 2.5 });
    let d = ''; for (let p = 0; p <= f.pct; p++) d += `${d ? 'L' : 'M'}${sx(p).toFixed(1)} ${sy(series(p)).toFixed(1)}`;
    path(E, d, { stroke: 'var(--viz-cat-3)', sw: 2.75 });
    const x = sx(f.pct);
    line(E, x, sy(f.pubScore), x, sy(f.freshScore), { stroke: ACCENT, sw: 2.5 });
    circle(E, x, sy(f.pubScore), 5, { fill: 'var(--viz-cat-3)', stroke: BG, sw: 2 });
    circle(E, x, sy(f.freshScore), 5, { fill: 'var(--viz-cat-7)', stroke: BG, sw: 2 });
    text(E, 'Fresh', b.x + b.w + 6, sy(fresh) + 4, { size: 11, fill: TEXT, weight: 600 });
    text(E, 'Public', b.x + b.w + 6, sy(series(maxPct)) + 4, { size: 11, fill: TEXT, weight: 600 });
    if (f.pct > 0) text(E, `gap ${signFmt(f.pubScore - f.freshScore, 2)}`, x > b.x + b.w * 0.7 ? x - 8 : x + 8, (sy(f.pubScore) + sy(f.freshScore)) / 2 + 4, { size: 12, weight: 600, fill: TEXT, halo: true, anchor: x > b.x + b.w * 0.7 ? 'end' : 'start' });
  },
  label: (s) => `A model's score on a public test of ${s.n} questions with up to ${s.max}% leaked into training, against its steady score on fresh questions.`,
};
export default mode;
