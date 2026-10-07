/**
 * goodhart (evaluations step 4): fifty variants of one system, all with the same true skill (true score about 0.60), are
 * scored on the same 200-question public test. Choosing the best public score inflates it; the chosen variant's score on a
 * private test of 1,000 other questions stays near the truth.
 * Props: submissions (50), items (200), privateItems (1000), seed (2).
 */
import { int, key, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, TEXT, axes, circle, fmt, line, path, text, type Env } from '../_rl/svg';
import { createRng, difficulties, popAcc, sigmoid } from './sim';

export const SKILL = 0.5, PLAY = 12, HOLD = 3;
export interface Scene {
  tries: number; items: number; privItems: number; truth: number;
  pub: Int32Array; priv: Int32Array;
  /** best[j] = index of the best public score among tries 0..j (earliest wins ties). */
  best: Int32Array;
}
export function makeGoodhart(tries: number, items: number, privItems: number, seed: number): Scene {
  const rng = createRng(seed), dPub = difficulties(items, rng), dPriv = difficulties(privItems, rng), count = (d: Float64Array) => { let k = 0; for (let i = 0; i < d.length; i++) k += rng.next() < sigmoid(SKILL - d[i]) ? 1 : 0; return k; };
  const pub = new Int32Array(tries), priv = new Int32Array(tries), best = new Int32Array(tries);
  for (let j = 0; j < tries; j++) {
    pub[j] = count(dPub); priv[j] = count(dPriv);
    best[j] = j && pub[best[j - 1]] >= pub[j] ? best[j - 1] : j;
  }
  return { tries, items, privItems, truth: popAcc(SKILL), pub, priv, best };
}

interface F extends Frame { n: number; b: number }
const mode: ModeImpl<Scene, F> = {
  setup: (o: Opts, kit: Kit) => { const a = [int(o.submissions, 50), int(o.items, 200), int(o.privateItems, 1000), int(o.seed, 2)] as const; return kit.memo(key('goodhart', ...a), () => makeGoodhart(...a)); },
  timeline: (s) => ({ duration: PLAY + HOLD, markers: [{ at: 0, label: 'Tries 1 to 10' }, { at: (PLAY * 10) / s.tries, label: 'Tries 11 to 30' }, { at: (PLAY * 30) / s.tries, label: 'Tries 31 to 50' }] }),
  frameAt(t, s) {
    const n = Math.max(1, Math.min(s.tries, Math.floor((t / PLAY) * s.tries) + 1)), b = s.best[n - 1], pubS = s.pub[b] / s.items, privS = s.priv[b] / s.privItems;
    const status = `After ${n} ${n === 1 ? 'try' : 'tries'}, the best public score is ${fmt(pubS, 3)} (try ${b + 1}). That same try scores ${fmt(privS, 3)} on the private test of ${s.privItems.toLocaleString('en-US')} questions, and its true skill is ${fmt(s.truth, 3)}.`;
    return { n, b, status, readouts: [['Tries', `${n}`], ['Best public score', fmt(pubS, 3)], ['Its private score', fmt(privS, 3)], ['True skill', fmt(s.truth, 3)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, b = nar ? { x: 44, y: 28, w: E.W - 56, h: E.H - 76 } : { x: 56, y: 30, w: E.W - 150, h: E.H - 78 };
    const all = [...Array.from(s.pub, (k) => k / s.items), ...Array.from(s.priv, (k) => k / s.privItems), s.truth];
    const lo = Math.floor((Math.min(...all) - 0.02) * 20) / 20, hi = Math.ceil((Math.max(...all) + 0.02) * 20) / 20;
    const yt = Array.from({ length: Math.round((hi - lo) / 0.05) + 1 }, (_, i) => Math.round((lo + i * 0.05) * 100) / 100);
    const { sx, sy } = axes(E, b, { lo: 0.5, hi: s.tries + 0.5, label: 'Try number', ticks: [1, 10, 20, 30, 40, 50].filter((v) => v <= s.tries) }, { lo, hi, label: 'Score', ticks: yt, fmt: (v) => fmt(v, 2) });
    line(E, b.x, sy(s.truth), b.x + b.w, sy(s.truth), { stroke: TEXT, sw: 1.75, dash: '6 4' });
    if (nar) text(E, `True skill ${fmt(s.truth, 2)}`, b.x + b.w - 4, sy(s.truth) + 15, { size: 11, fill: TEXT, anchor: 'end' });
    else text(E, `True skill ${fmt(s.truth, 2)}`, b.x + b.w + 6, sy(s.truth) + 4, { size: 11, fill: TEXT });
    for (let j = 0; j < f.n; j++) circle(E, sx(j + 1), sy(s.pub[j] / s.items), 2.8, { fill: 'var(--viz-cat-1)', op: 0.55 });
    let dBest = '', dPriv = '';
    for (let j = 0; j < f.n; j++) {
      const x = sx(j + 1), yb = sy(s.pub[s.best[j]] / s.items), yp = sy(s.priv[s.best[j]] / s.privItems), x0 = j ? sx(j) : x;
      dBest += `${dBest ? 'L' + x0.toFixed(1) + ' ' + (sy(s.pub[s.best[j - 1]] / s.items)).toFixed(1) : 'M' + x.toFixed(1) + ' ' + yb.toFixed(1)}L${x.toFixed(1)} ${yb.toFixed(1)}`;
      dPriv += `${dPriv ? 'L' + x0.toFixed(1) + ' ' + (sy(s.priv[s.best[j - 1]] / s.privItems)).toFixed(1) : 'M' + x.toFixed(1) + ' ' + yp.toFixed(1)}L${x.toFixed(1)} ${yp.toFixed(1)}`;
    }
    path(E, dBest, { stroke: ACCENT, sw: 2.75 });
    path(E, dPriv, { stroke: 'var(--viz-cat-2)', sw: 2.75 });
    const x = sx(f.n);
    circle(E, x, sy(s.pub[f.b] / s.items), 5.5, { fill: ACCENT, stroke: BG, sw: 2 });
    circle(E, x, sy(s.priv[f.b] / s.privItems), 5.5, { fill: 'var(--viz-cat-2)', stroke: BG, sw: 2 });
    if (!nar) {
      text(E, 'Best public', b.x + b.w + 6, sy(s.pub[f.b] / s.items) + 4, { size: 11, fill: TEXT, weight: 600 });
      text(E, 'Its private score', b.x + b.w + 6, sy(s.priv[f.b] / s.privItems) + 4, { size: 11, fill: TEXT, weight: 600 });
    }
  },
  label: (s) => `${s.tries} variants with the same true skill scored on one public test: the best public score climbs while the chosen variant's private score stays near the true skill.`,
};
export default mode;
