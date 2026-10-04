/** The split figure: its frame function and its canvas drawing. */
import { ease, sgn, type Scene, weightMatrix, fullVector, SPLIT, type SplitFrame, type Opts, pre, type Frame } from './core';
import { type Env, HEAD_CAT, rgba, signed, txt, grid, strip, maxAbs, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase } = pre(t, s, o);
  const hs = [1, 2, 4], idx = tt < SPLIT[1] ? 0 : tt < SPLIT[2] ? 1 : 2, h = hs[idx], prev = hs[Math.max(0, idx - 1)];
  const e = idx === 0 ? 1 : ease((tt - SPLIT[idx]) / 1.0), gaps = new Float64Array(s.d - 1);
  for (let b = 1; b < s.d; b++) gaps[b - 1] = b % (s.d / prev) === 0 && prev > 1 ? 1 : b % (s.d / h) === 0 && h > 1 ? e : 0;
  const total = 4 * s.d * s.d, count = Math.round(total * ease(tt / 1.6));
  const status = `${h} head${h === 1 ? '' : 's'}: each takes ${s.d / h} of the ${s.d} columns. The four matrices hold ${count.toLocaleString('en-US')} weights${count === total ? ' however many heads there are' : ''}.`;
  return { mode: 'split', h, gaps, count, dkHead: s.d / h, status, phase };
}

function drawSplit(E: Env, f: SplitFrame) {
  const { c, P, s, narrow, W } = E, d = s.d, qi = 3 < s.T ? 3 : s.T - 1, cs = narrow ? 5.2 : 9.5, gpx = narrow ? 4 : 7, maxGap = gpx * 3;
  const names = ['q', 'k', 'v'] as const, vecW = narrow ? 7 : 10, groupW = vecW + 6 + d * cs + maxGap, gap = (W - 3 * groupW) / 4;
  const y0 = narrow ? 56 : 62, x = s.trace.layers[s.layer].ln1.subarray(qi * d, qi * d + d);
  const gapAt = (j: number) => (j > 0 ? f.gaps[j - 1] * gpx : 0);
  names.forEach((n, g) => {
    const gx = gap + g * (groupW + gap), M = weightMatrix(s.w, s.layer, n), bx = gx + vecW + 6;
    txt(E, `W_${n.toUpperCase()}`, bx, y0 - 22, { sz: narrow ? 10.5 : 12.5, bold: true }); if (!narrow) txt(E, `${d} × ${d}`, bx + 34, y0 - 22, { sz: 10.5, col: P.muted });
    txt(E, 'x', gx + vecW / 2, y0 - 6, { sz: narrow ? 9.5 : 11, al: 'center', col: P.muted });
    strip(E, gx, y0, vecW, cs, x, 2.5, { vert: true, tip: (k) => `${quote(s.words[qi])} vector · number ${k + 1}: ${sgn(x[k])}` });
    grid(E, bx, y0, cs, cs, d, d, (i, j) => M[i * d + j], signed(P, maxAbs(M)), { gap: gapAt, tip: (i, j) => `W_${n.toUpperCase()} · in ${i + 1}, out ${j + 1}: ${sgn(M[i * d + j], 3)}` });
    const out = fullVector(s, s.layer, n, qi), oy = y0 + d * cs + 8;
    txt(E, n, bx - 4, oy + (narrow ? 8 : 11), { sz: narrow ? 9.5 : 11, al: 'right', col: P.muted });
    strip(E, bx, oy, cs, narrow ? 8 : 12, out, 4, { gap: gapAt, tip: (k) => `${n.toUpperCase()} · ${quote(s.words[qi])} · number ${k + 1}: ${sgn(out[k])}` });
    const per = d / f.h;
    for (let h = 0; h < f.h; h++) {
      const xa = bx + h * per * cs + (h > 0 ? 0 : 0) + (() => { let a = 0; for (let j = 1; j <= h * per; j++) a += gapAt(j); return a; })(), wd = per * cs;
      c.fillStyle = rgba(P.cat[HEAD_CAT[h]]); c.fillRect(xa, oy + (narrow ? 11 : 15), wd - 1, 3);
    }
  });
  const ly = y0 + d * cs + (narrow ? 44 : 52);
  txt(E, `The word ${quote(s.words[qi])} (left strip) times each matrix gives its query, key and value (strip below). Columns of a matrix feed one output number.`, 8, ly, { sz: 10, col: P.muted, a: 0 });
  // legend of head slices
  const lx = narrow ? 8 : 20; let cx = lx;
  for (let h = 0; h < f.h; h++) { c.fillStyle = rgba(P.cat[HEAD_CAT[h]]); c.fillRect(cx, ly - 9, 10, 10); txt(E, narrow ? `head ${h + 1}` : `head ${h + 1}: ${f.dkHead} columns`, cx + 15, ly, { sz: narrow ? 9.5 : 11 }); cx += narrow ? 58 : 118; }
  // parameter counter
  const cy = ly + (narrow ? 34 : 40);
  txt(E, 'Weights in W_Q, W_K, W_V and W_O together', lx, cy, { sz: narrow ? 10 : 11.5, col: P.muted });
  txt(E, f.count.toLocaleString('en-US'), lx, cy + (narrow ? 26 : 30), { sz: narrow ? 22 : 28, bold: true, mono: true });
  txt(E, `= 4 × ${d} × ${d}`, lx + (narrow ? 70 : 92), cy + (narrow ? 26 : 30), { sz: narrow ? 12 : 14, col: P.muted, mono: true });
  txt(E, `each head's own W_Q is ${d} × ${f.dkHead}`, narrow ? 8 : W - 8, cy + (narrow ? 48 : 30), { sz: narrow ? 10 : 11.5, col: P.muted, al: narrow ? 'left' : 'right' });
}

export const draw = drawSplit;
