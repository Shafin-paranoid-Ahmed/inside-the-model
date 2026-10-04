/** The reach figure: its frame function and its canvas drawing. */
import { ease, type Scene, REACH_HOP, reachCount, reachEnds, type ReachFrame, type Opts, pre, REACH_WORD, type Frame } from './core';
import { type Env, rgba, seqColor, txt, grid } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T } = pre(t, s, o);
  const { from, to } = reachEnds(T, s.words), rc = reachCount(T, s.words), hops = to - from;
  const part = tt < rc.pathEnd ? 'path' : 'cost';
  const pos = Math.min(hops, Math.max(0, (tt - rc.lead) / REACH_HOP));
  const rnnHops = Math.floor(pos + 1e-9), attnU = ease((tt - 0.3) / 0.7);
  const Tn = part === 'path' ? T : Math.min(T, 1 + Math.floor((tt - rc.pathEnd) / REACH_WORD));
  const status = part === 'path'
    ? `From “${s.words[from]}” to “${s.words[to]}”: ${rnnHops} of ${hops} hops through the recurrent chain, ${attnU >= 1 ? 1 : 0} of 1 through attention.`
    : `${Tn} token${Tn === 1 ? '' : 's'}${Tn > 1 ? ` (${Tn - 1} word${Tn === 2 ? '' : 's'} plus [start])` : ''} need ${Tn} × ${Tn} = ${Tn * Tn} scores.`;
  return { mode: 'reach', part, from, to, rnnPos: from + pos, rnnHops, attnU, attnHops: attnU >= 1 ? 1 : 0, Tn, scores: Tn * Tn, status, phase };
}

function drawReach(E: Env, f: ReachFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, M = narrow ? 8 : 14, cost = f.part === 'cost';
  const areaW = narrow ? W - 2 * M : 520, sp = areaW / T, xs = (i: number) => M + sp * (i + 0.5), right = M + areaW;
  const yR = narrow ? 62 : 74, yA = narrow ? 204 : 204, r = narrow ? 11 : 14, sz = narrow ? 9 : 10.5, hs = narrow ? 10 : 12;
  const vis = (i: number) => (i < f.Tn ? 1 : 0.2), cnt = (n: number) => `${n} hop${n === 1 ? '' : 's'}`;
  const node = (x: number, y: number, on: boolean, a: number) => {
    c.fillStyle = on ? rgba(seqColor(P, 0.7), a) : rgba(P.bg, 1); c.strokeStyle = rgba(P.muted, a); c.lineWidth = 1.5;
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); c.stroke();
  };
  txt(E, narrow ? 'Recurrent network: word by word' : 'Recurrent network: through every word in between', M, 18, { sz: hs, bold: true });
  if (!cost) txt(E, cnt(f.rnnHops), right, 18, { sz: narrow ? 12 : 15, bold: true, mono: true, al: 'right', col: P.accent });
  for (let i = 0; i < T; i++) {
    if (i > 0) { c.strokeStyle = rgba(P.rule, Math.min(vis(i), vis(i - 1))); c.lineWidth = 1.5; c.beginPath(); c.moveTo(xs(i - 1) + r, yR); c.lineTo(xs(i) - r, yR); c.stroke(); }
    node(xs(i), yR, !cost && i >= f.from && i <= f.rnnPos + 1e-9, vis(i));
    txt(E, s.words[i], xs(i), yR + r + 15 + (narrow && i % 2 ? 14 : 0), { sz, al: 'center', a: vis(i), bold: !cost && (i === f.from || i === f.to) });
  }
  if (!cost) { c.fillStyle = rgba(P.accent); c.beginPath(); c.arc(xs(0) + f.rnnPos * sp, yR, r * 0.5, 0, Math.PI * 2); c.fill(); }
  txt(E, narrow ? 'Self-attention: straight to the word' : 'Self-attention: straight to any earlier word', M, yA - 82, { sz: hs, bold: true });
  if (!cost) txt(E, cnt(f.attnHops), right, yA - 82, { sz: narrow ? 12 : 15, bold: true, mono: true, al: 'right', col: P.accent });
  for (let i = 0; i < T; i++) {
    node(xs(i), yA, !cost && (i === f.from || (i === f.to && f.attnU >= 1)), vis(i));
    txt(E, s.words[i], xs(i), yA + r + 15 + (narrow && i % 2 ? 14 : 0), { sz, al: 'center', a: vis(i), bold: !cost && (i === f.from || i === f.to) });
  }
  const top = yA - r, arc = (i: number, j: number, hgt: number, upto = 1) => {
    const x1 = xs(i), x2 = xs(j); c.beginPath(); c.moveTo(x1, top);
    const N = 32; for (let k = 1; k <= Math.round(N * upto); k++) { const t = k / N, it = 1 - t, h = top - hgt * 1.33; c.lineTo(it ** 3 * x1 + 3 * it * it * t * x1 + 3 * it * t * t * x2 + t ** 3 * x2, it ** 3 * top + 3 * it * it * t * h + 3 * it * t * t * h + t ** 3 * top); }
    c.stroke();
  };
  if (cost) {
    c.strokeStyle = rgba(P.seq[2], 0.55); c.lineWidth = 1;
    for (let i = 1; i < f.Tn; i++) for (let j = 0; j < i; j++) arc(i, j, Math.min(64, 8 + (i - j) * 8));
  } else {
    c.strokeStyle = rgba(P.accent, 0.22); c.lineWidth = 2; arc(f.to, f.from, 62);
    if (f.attnU > 0) { c.strokeStyle = rgba(P.accent); c.lineWidth = 3; arc(f.from, f.to, 62, f.attnU); }
  }
  // one score for every pair of words
  const cs = narrow ? 12 : 16, gx = narrow ? (W - T * cs) / 2 : W - M - T * cs - 20, gy = narrow ? yA + 78 : 54;
  txt(E, 'Scores for every pair', narrow ? M : gx - 4, gy - 16, { sz: narrow ? 10 : 11.5, bold: true });
  grid(E, gx, gy, cs, cs, T, T, () => 0.6, (v) => seqColor(P, v), { vis: (i, j) => (i < f.Tn && j < f.Tn ? 1 : 0) });
  txt(E, `${f.Tn} \u00d7 ${f.Tn} = ${f.scores}`, gx + (T * cs) / 2, gy + T * cs + 22, { sz: narrow ? 13 : 16, bold: true, mono: true, al: 'center' });
}

export const draw = drawReach;
