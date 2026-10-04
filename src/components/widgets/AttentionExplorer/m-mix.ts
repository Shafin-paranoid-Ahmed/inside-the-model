/** The mix figure: its frame function and its canvas drawing. */
import { clamp01, ease, f2, sgn, type Scene, headData, rowOf, type MixFrame, type Opts, pre, MIX_SEG, type Frame } from './core';
import { type Env, rgba, seqColor, fmt2, txt, hatch, box, short, strip, maxAbs, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T, dk, qs } = pre(t, s, o);
  const { A, v } = headData(s), a = Float64Array.from(rowOf(A, T, qs));
  const done = tt >= (qs + 1) * MIX_SEG - 1e-9, j = done ? qs : Math.min(qs, Math.floor(tt / MIX_SEG));
  const u = done ? 1 : (tt - j * MIX_SEG) / MIX_SEG, take = ease((u - 0.3) / 0.6);
  const acc = new Float64Array(dk), contrib = new Float64Array(dk);
  for (let m = 0; m <= qs; m++) {
    const share = done || m < j ? 1 : m === j ? take : 0;
    for (let d = 0; d < dk; d++) acc[d] += share * a[m] * v[m * dk + d];
    if (!done && m === j) for (let d = 0; d < dk; d++) contrib[d] = a[m] * v[m * dk + d];
  }
  const status = done ? `The head's output for “${s.words[qs]}” is the weighted sum of the ${qs + 1} value rows above.` :
    `Adding ${f2(a[j])} × the value of “${s.words[j]}” to the running total.`;
  return { mode: 'mix', qi: qs, j: done ? -1 : j, u, flow: done ? 1 : clamp01(u / 0.7), a, acc, contrib, done, status, phase };
}

function drawMix(E: Env, f: MixFrame) {
  const { c, P, s, narrow } = E, T = s.T, dk = s.dk, qi = f.qi, { v } = headData(s), vmax = maxAbs(v);
  const rowH = narrow ? 26 : 32, y0 = 34, cw = narrow ? 22 : 24, M = narrow ? 6 : 14;
  const labW = narrow ? 50 : 66, barX = M + labW + 4, barW = narrow ? 40 : 60, vx = barX + barW + 36 + (narrow ? 0 : 6);
  txt(E, 'weight', barX, y0 - 12, { sz: 10.5, col: P.muted }); txt(E, 'value (8 numbers)', vx, y0 - 12, { sz: 10.5, col: P.muted });
  const outX = narrow ? M + 4 : vx + dk * cw + 80, outY = narrow ? y0 + T * rowH + 40 : y0 + 12, ow = narrow ? 36 : 30;
  for (let j = 0; j < T; j++) {
    const y = y0 + j * rowH, masked = j > qi, active = j === f.j, a = masked ? 0.35 : 1;
    txt(E, short(E, s.words[j], 7), M + labW, y + rowH / 2 + 3.5, { sz: narrow ? 9.5 : 11, al: 'right', bold: active, col: active ? P.accent : P.text, a });
    c.fillStyle = rgba(P.rule, 0.5); c.fillRect(barX, y + rowH / 2 - 5, barW, 10);
    c.fillStyle = rgba(seqColor(P, 0.5 + 0.5 * f.a[j]), a); c.fillRect(barX, y + rowH / 2 - 5, barW * f.a[j], 10);
    txt(E, masked ? '0' : fmt2(f.a[j]), barX + barW + 5, y + rowH / 2 + 3.5, { sz: 9.5, mono: true, col: P.muted, a });
    strip(E, vx, y + 4, cw, rowH - 8, v.subarray(j * dk, j * dk + dk), vmax, { vis: a, tip: (kk) => `value of ${quote(s.words[j])} · number ${kk + 1}: ${sgn(v[j * dk + kk])}` });
    if (masked) hatch(E, vx, y + 4, dk * cw, rowH - 8, 0.25);
    if (active) box(E, vx - 2, y + 2, dk * cw + 3, rowH - 4, P.accent, 1.5, 3);
  }
  txt(E, f.done ? `${quote(s.words[qi])}: the head's output` : 'running total (adds up to the output)', outX, outY - 12, { sz: narrow ? 10 : 11.5, bold: true });
  strip(E, outX, outY, ow, 30, f.acc, vmax, { nums: true, tip: (kk) => `output so far · number ${kk + 1}: ${sgn(f.acc[kk])}` });
  if (f.done) box(E, outX - 2, outY - 2, dk * ow + 3, 34, P.accent, 2, 3);
  // flow of the active value into the total
  if (!f.done && f.j >= 0 && f.flow > 0 && f.flow < 1) {
    const y = y0 + f.j * rowH + rowH / 2, x1 = vx + dk * cw, y2 = outY + 15, n = Math.max(2, Math.round(f.a[f.j] * 12));
    c.strokeStyle = rgba(P.accent, 0.3); c.lineWidth = 1 + 5 * f.a[f.j]; c.beginPath(); c.moveTo(x1, y);
    c.bezierCurveTo(x1 + 40, y, outX - 40, y2, outX, y2); c.stroke();
    for (let k = 0; k < n; k++) {
      const u = f.flow - k * 0.05; if (u < 0 || u > 1) continue;
      const it = 1 - u, mx1 = x1 + 40, mx2 = outX - 40;
      const px = it ** 3 * x1 + 3 * it * it * u * mx1 + 3 * it * u * u * mx2 + u ** 3 * outX, py = it ** 3 * y + 3 * it * it * u * y + 3 * it * u * u * y2 + u ** 3 * y2;
      c.fillStyle = rgba(P.accent, 1 - k / n * 0.7); c.beginPath(); c.arc(px, py, 2.6 + 3 * f.a[f.j], 0, Math.PI * 2); c.fill();
    }
  }
  // value of the heaviest word, for comparison once the sum is complete
  const top = f.a.reduce((b, x, j) => (j <= qi && x > f.a[b] ? j : b), 0), cy = outY + 66;
  txt(E, `value of ${quote(s.words[top])}, the heaviest word`, outX, cy - 6, { sz: narrow ? 9.5 : 11, col: P.muted });
  strip(E, outX, cy, ow, 22, v.subarray(top * dk, top * dk + dk), vmax, { vis: f.done ? 1 : 0.35, nums: f.done });
}

export const draw = drawMix;
