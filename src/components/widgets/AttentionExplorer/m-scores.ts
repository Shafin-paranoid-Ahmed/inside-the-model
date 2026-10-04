/** The scores figure: its frame function and its canvas drawing. */
import { ease, sgn, type Scene, headData, SC_D, scoreTimes, type ScoresFrame, type Opts, pre, SC_F, SC_G, type Frame } from './core';
import { type Env, signed, txt, box, short, grid, strip, maxAbs, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T, dk, qs } = pre(t, s, o);
  const { S, q, k } = headData(s), tm = scoreTimes(T, qs);
  const vis = new Float64Array(T * T), val = new Float64Array(T * T);
  let detail: ScoresFrame['detail'] = null;
  const start = (j: number) => (j <= qs ? j * SC_D : (qs + 1) * SC_D + (j - qs - 1) * SC_F);
  for (let j = 0; j < T; j++) {
    const st = start(j), dur = j <= qs ? SC_D : SC_F, u = (tt - st) / dur, c = qs * T + j;
    if (j <= qs) {
      if (tt >= st + dur) { vis[c] = 1; val[c] = S[c]; }
      else if (u >= 0) {
        const pr = ease((u - 0.2) / 0.4), shown = pr * dk, products = new Float64Array(dk);
        let run = 0;
        for (let d = 0; d < dk; d++) { products[d] = q[qs * dk + d] * k[j * dk + d]; if (d + 1 <= shown) run += products[d]; else if (d < shown) run += products[d] * (shown - d); }
        detail = { j, u, products, shown, sum: run };
        vis[c] = ease((u - 0.25) / 0.1); val[c] = run;
      }
    } else if (u >= 0) { const e = ease(u); vis[c] = e; val[c] = S[c] * e; }
  }
  let n = 0;
  for (let i = 0; i < T; i++) { if (i === qs) continue; for (let j = 0; j < T; j++, n++) {
    const u = (tt - (tm.rowEnd + n * SC_G)) / 0.2, c = i * T + j;
    if (u > 0) { const e = ease(u); vis[c] = e; val[c] = S[c] * e; }
  } }
  const status = detail ? `Row “${s.words[qs]}”, key “${s.words[detail.j]}”: query · key = ${sgn(detail.sum)}` :
    tt < tm.rowEnd ? `Row “${s.words[qs]}”: the remaining cells are future words. They get scores too, and are masked later.` :
    tt < tm.allEnd ? 'Every other row is the same sum with a different query.' : 'The full grid of scores, one row per query word.';
  return { mode: 'scores', qi: qs, vis, val, detail, status, phase };
}

function drawScores(E: Env, f: ScoresFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, dk = s.dk, { S, q, k } = headData(s), cs = narrow ? 24 : 30;
  const gx = narrow ? 54 : 88, gy = narrow ? 58 : 66, vmax = maxAbs(S), col = signed(P, vmax), qi = f.qi;
  txt(E, 'Key (column)', gx, gy - (narrow ? 44 : 50), { sz: 10.5, col: P.muted });
  txt(E, narrow ? 'Query' : 'Query (row)', 4, gy - (narrow ? 44 : 50) + 12, { sz: 10.5, col: P.muted });
  for (let j = 0; j < T; j++) {
    c.save(); c.translate(gx + (j + 0.5) * cs - 2, gy - 6); c.rotate(-Math.PI / 3);
    txt(E, short(E, s.words[j], 7), 0, 0, { sz: narrow ? 9 : 10.5, bold: f.detail?.j === j, col: f.detail?.j === j ? P.accent : P.text }); c.restore();
  }
  for (let i = 0; i < T; i++) txt(E, short(E, s.words[i], 7), gx - 6, gy + (i + 0.5) * cs + 4, { sz: narrow ? 9.5 : 11, al: 'right', bold: i === qi, col: i === qi ? P.accent : P.text });
  grid(E, gx, gy, cs, cs, T, T, (i, j) => f.val[i * T + j], col, {
    vis: (i, j) => f.vis[i * T + j], masked: (i, j) => j > i,
    num: (i, j) => (i === qi && f.vis[i * T + j] > 0.9 ? sgn(S[i * T + j], narrow ? 0 : 1) : null),
    tip: (i, j) => `${quote(s.words[i])} · ${quote(s.words[j])}: score ${sgn(S[i * T + j])}${j > i ? ' (masked later)' : ''}`,
  });
  box(E, gx - 1, gy + qi * cs - 1, T * cs + 1, cs + 1, P.accent, 1.5, 2);
  if (f.detail) box(E, gx + f.detail.j * cs - 1, gy + qi * cs - 1, cs + 1, cs + 1, P.accent, 2.5, 2);
  txt(E, narrow ? 'Hatched: later words, masked next.' : 'Hatched cells are later words: they get scores here, and are masked in the next step.', gx, gy + T * cs + 16, { sz: narrow ? 9 : 10, col: P.muted });
  // detail panel
  const px = narrow ? 14 : gx + T * cs + 34, py = narrow ? gy + T * cs + 40 : gy - 4, pc = narrow ? 30 : 30, pw = dk * pc;
  const d = f.detail;
  if (!d) { txt(E, 'Every cell is a sum like the one shown while the row fills.', px, py + 14, { sz: 11, col: P.muted }); return; }
  txt(E, `${quote(s.words[qi])} · ${quote(s.words[d.j])}`, px, py + 6, { sz: 12, bold: true });
  txt(E, `query of ${quote(s.words[qi])}`, px, py + 28, { sz: 10.5, col: P.muted });
  strip(E, px, py + 34, pc, 22, q.subarray(qi * dk, qi * dk + dk), 4, { nums: true });
  txt(E, `key of ${quote(s.words[d.j])}`, px, py + 78, { sz: 10.5, col: P.muted });
  strip(E, px, py + 84, pc, 22, k.subarray(d.j * dk, d.j * dk + dk), 4, { nums: true });
  txt(E, 'multiply number by number', px, py + 128, { sz: 10.5, col: P.muted });
  strip(E, px, py + 134, pc, 22, d.products, 8, { nums: true, vis: (kk) => (kk < d.shown ? 1 : 0.12) });
  txt(E, 'add them up', px, py + 182, { sz: 10.5, col: P.muted });
  txt(E, `= ${sgn(d.sum)}`, px, py + 206, { sz: 20, bold: true, mono: true, col: d.sum >= 0 ? P.pos : P.neg });
  void pw; void W;
}

export const draw = drawScores;
