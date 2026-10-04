/** The concat figure: its frame function and its canvas drawing. */
import { clamp01, ease, sgn, norm, type Scene, rowOf, weightMatrix, CAT, type ConcatFrame, type Opts, pre, type Frame } from './core';
import { type Env, HEAD_CAT, rgba, signed, txt, grid, strip, maxAbs, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, dk, qs } = pre(t, s, o);
  const L = s.trace.layers[s.layer], d = s.d;
  const slide = ease((tt - CAT.join) / 1.4), proj = ease((tt - CAT.proj) / 1.6), add = ease((tt - CAT.add) / 1.4);
  const norms = { input: norm(rowOf(L.input, d, qs)), attn: norm(rowOf(L.attnOut, d, qs)), sum: norm(rowOf(L.resid1, d, qs)) };
  const status = tt < CAT.join ? `Two heads each produced ${dk} numbers for “${s.words[qs]}”.` :
    tt < CAT.proj ? `Stack them into one vector of ${d} numbers.` :
    tt < CAT.add ? `Multiply by W_O (${d} × ${d}) to mix the two heads' results.` :
    `The result (size ${norms.attn.toFixed(2)}) is added to the word's own vector (size ${norms.input.toFixed(2)}).`;
  return { mode: 'concat', qi: qs, slide, proj, add, norms, status, phase };
}

function drawConcat(E: Env, f: ConcatFrame) {
  const { c, P, s, narrow } = E, d = s.d, dk = s.dk, qi = f.qi, L = s.trace.layers[s.layer], cs = narrow ? 7 : 10, M = narrow ? 8 : 16;
  const y0 = narrow ? 54 : 66, vx = M + (narrow ? 44 : 74), small = narrow ? 9.5 : 11, bh = narrow ? 10 : 14;
  const h1 = L.headOut[0].subarray(qi * dk, qi * dk + dk), h2 = L.headOut[1].subarray(qi * dk, qi * dk + dk), vmax = maxAbs(L.concat);
  const x1 = M + 2 + (vx - M - 2) * f.slide, x2 = M + (narrow ? 22 : 58) + (vx - M - (narrow ? 22 : 58)) * f.slide, y2 = y0 + dk * cs * f.slide;
  strip(E, x1, y0, cs, cs, h1, vmax, { vert: true, tip: (k) => `head 1 output \u00b7 number ${k + 1}: ${sgn(h1[k])}` });
  strip(E, x2, y2, cs, cs, h2, vmax, { vert: true, tip: (k) => `head 2 output \u00b7 number ${k + 1}: ${sgn(h2[k])}` });
  c.fillStyle = rgba(P.cat[HEAD_CAT[0]]); c.fillRect(x1 - 4, y0, 2.5, dk * cs); c.fillStyle = rgba(P.cat[HEAD_CAT[1]]); c.fillRect(x2 - 4, y2, 2.5, dk * cs);
  txt(E, narrow ? 'h1' : 'head 1', x1 - 4, y0 + dk * cs + 14, { sz: small, col: P.muted, a: 1 - f.slide }); txt(E, narrow ? 'h2' : 'head 2', x2 - 4, y0 + dk * cs + 14, { sz: small, col: P.muted, a: 1 - f.slide });
  txt(E, narrow ? `${d}` : `${d} numbers`, vx + cs, y0 + d * cs + 14, { sz: small, col: P.muted, a: f.slide, al: 'right' });
  const bx = vx + cs + (narrow ? 12 : 26), M2 = weightMatrix(s.w, s.layer, 'o'), blockA = 0.3 + 0.7 * Math.max(f.slide, f.proj);
  txt(E, `\u00d7 W_O (${d} \u00d7 ${d})`, bx, y0 - 10, { sz: narrow ? 10 : 12, bold: true, a: blockA });
  grid(E, bx, y0, cs, cs, d, d, (i, j) => M2[i * d + j], signed(P, maxAbs(M2)), { vis: () => blockA, tip: (i, j) => `W_O \u00b7 in ${i + 1}, out ${j + 1}: ${sgn(M2[i * d + j], 3)}` });
  const ao = L.attnOut.subarray(qi * d, qi * d + d), oy = y0 + d * cs + 8;
  strip(E, bx, oy, cs, bh, ao, 4, { vis: (k) => clamp01(f.proj * d - k * 0.6), scale: f.proj, tip: (k) => `attention output \u00b7 number ${k + 1}: ${sgn(ao[k])}` });
  if (f.proj > 0.01 && f.proj < 1) { c.fillStyle = rgba(P.accent, 0.8); c.fillRect(bx + f.proj * d * cs - 1, oy - 3, 2, bh + 6); }
  txt(E, 'attention output', bx, oy + bh + 14, { sz: small, col: P.muted, a: f.proj });
  const rx = narrow ? M : bx + d * cs + 60, ry = narrow ? oy + bh + 54 : y0, rw = d * cs, rows = [
    { name: `${quote(s.words[qi])} before attention`, v: L.input.subarray(qi * d, qi * d + d), n: f.norms.input, a: 1 },
    { name: 'attention output', v: ao, n: f.norms.attn, a: f.proj },
    { name: `${quote(s.words[qi])} after attention`, v: L.resid1.subarray(qi * d, qi * d + d), n: f.norms.sum, a: f.add },
  ];
  txt(E, 'Add it to the word\u2019s own vector', rx, ry - 10, { sz: narrow ? 10 : 12, bold: true, a: 0.3 + 0.7 * f.proj });
  rows.forEach((r, k) => {
    const y = ry + k * (narrow ? 50 : 58);
    txt(E, r.name, rx, y + 8, { sz: small, col: P.muted, a: Math.max(0.25, r.a) });
    strip(E, rx, y + 14, cs, bh, r.v, 4, { vis: k === 2 ? r.a : k === 1 ? Math.max(0.25, f.proj) : 1, tip: (kk) => `${r.name} \u00b7 number ${kk + 1}: ${sgn(r.v[kk])}` });
    txt(E, `size ${r.n.toFixed(2)}`, rx + rw + 10, y + 14 + bh - 2, { sz: narrow ? 10 : 11.5, mono: true, bold: true, a: Math.max(0.25, r.a) });
    if (k < 2) txt(E, k === 0 ? '+' : '=', rx - (narrow ? 0 : 14), y + 14 + bh + 22, { sz: 16, bold: true, col: P.muted, a: Math.max(0.3, rows[k + 1].a) });
  });
}

export const draw = drawConcat;
