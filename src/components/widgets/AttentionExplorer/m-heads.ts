/** The heads figure: its frame function and its canvas drawing. */
import { headName, ease, f2, sgn, type Scene, headData, HEADS_ROW, type HeadsFrame, type Opts, argmax, pre, HEADS_OUT, type Frame } from './core';
import { type Env, HEAD_CAT, rgba, seqColor, signed, fmt2, txt, box, short, grid, maxAbs, quote } from './ui';
import { rowDisagreement, previousWordMean } from './derived';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T, dk } = pre(t, s, o);
  const e = T * HEADS_ROW, oEnd = e + T * HEADS_OUT;
  const rowVis = new Float64Array(T), outVis = new Float64Array(T);
  for (let i = 0; i < T; i++) { rowVis[i] = ease((tt - i * HEADS_ROW) / HEADS_ROW); outVis[i] = ease((tt - e - i * HEADS_OUT) / (HEADS_OUT * 2)); }
  const dis = rowDisagreement(s), hiRow = argmax(dis), hiU = ease((tt - oEnd - 0.6) / 0.5);
  const status = tt < e ? 'Both heads fill in their attention rows at the same time, each from its own queries and keys.' :
    tt < oEnd + 0.6 ? `Each head mixes its own values into an ${T} × ${dk} output.` :
    `“${s.words[hiRow]}” is where the two heads disagree most: ${f2(dis[hiRow])} of the weight sits on different words.`;
  return { mode: 'heads', rowVis, outVis, hiU, hiRow, status, phase };
}

function drawHeads(E: Env, f: HeadsFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, dk = s.dk, cs = narrow ? 14 : 26, lab = narrow ? 34 : 64, M = narrow ? 6 : 14, panelW = lab + T * cs;
  const x0s = [narrow ? M : (W / 2 - panelW) / 2 + 4, narrow ? W / 2 + M / 2 : W / 2 + (W / 2 - panelW) / 2 - 4];
  const gy = narrow ? 96 : 78, oy = gy + T * cs + (narrow ? 38 : 50), oh = narrow ? 8 : 12, ocw = (T * cs) / dk;
  for (let h = 0; h < 2; h++) {
    const { A, out } = headData(s, s.layer, h), x0 = x0s[h], gx = x0 + lab, col = P.cat[HEAD_CAT[h]];
    c.fillStyle = rgba(col); c.fillRect(x0, 14, 10, 10); txt(E, headName(s.layer, h), x0 + 15, 23, { sz: narrow ? 10 : 12, bold: true });
    txt(E, narrow ? `prev. word ${fmt2(previousWordMean(s, s.layer, h))}` : `mean weight on the previous word ${fmt2(previousWordMean(s, s.layer, h))}`, x0, 40, { sz: narrow ? 9 : 10.5, col: P.muted });
    for (let j = 0; j < T; j++) { c.save(); c.translate(gx + (j + 0.5) * cs - 2, gy - 5); c.rotate(narrow ? -Math.PI / 2 : -Math.PI / 3); txt(E, short(E, s.words[j], 6), 0, 0, { sz: narrow ? 8 : 10 }); c.restore(); }
    for (let i = 0; i < T; i++) txt(E, short(E, s.words[i], narrow ? 5 : 7), gx - 5, gy + (i + 0.5) * cs + 3.5, { sz: narrow ? 8.5 : 10.5, al: 'right' });
    grid(E, gx, gy, cs, cs, T, T, (i, j) => A[i * T + j], (v) => seqColor(P, v), {
      vis: (i) => (f.rowVis[i] > 0.02 ? f.rowVis[i] : 0), masked: (i, j) => j > i, num: (i, j) => (!narrow && j <= i && A[i * T + j] >= 0.1 ? fmt2(A[i * T + j]).replace(/^0/, '') : null),
      tip: (i, j) => `${headName(s.layer, h)}: ${quote(s.words[i])} → ${quote(s.words[j])}: ${j > i ? '0 (masked)' : fmt2(A[i * T + j])}`,
    });
    txt(E, narrow ? 'output (8 numbers)' : `output of the head: ${T} × ${dk}`, x0, oy - 8, { sz: narrow ? 9 : 10.5, col: P.muted });
    if (!narrow) for (let i = 0; i < T; i++) txt(E, s.words[i], gx - 5, oy + i * oh + oh - 2, { sz: 9, al: 'right', col: P.muted });
    grid(E, gx, oy, ocw, oh, T, dk, (i, j) => out[i * dk + j], signed(P, maxAbs(headData(s, s.layer, 0).out) > maxAbs(headData(s, s.layer, 1).out) ? maxAbs(headData(s, s.layer, 0).out) : maxAbs(headData(s, s.layer, 1).out)), {
      vis: (i) => (f.outVis[i] > 0.02 ? 1 : 0), tip: (i, j) => `head ${h + 1} output · ${s.words[i]} · number ${j + 1}: ${sgn(out[i * dk + j])}`,
    });
    if (f.hiU > 0.02) { c.globalAlpha = f.hiU; box(E, gx - 1, gy + f.hiRow * cs - 1, T * cs + 1, cs + 1, P.accent, 2.2, 2); c.globalAlpha = 1; }
  }
  if (f.hiU > 0.02) txt(E, `${quote(s.words[f.hiRow])}: the two rows differ most`, W / 2, oy + T * oh + 22, { sz: narrow ? 10 : 11.5, al: 'center', bold: true, a: f.hiU, col: P.accent });
}

export const draw = drawHeads;
export { ablationMasses, ablatedMass, sampleMass, sampleStats, headStats } from './stats';
