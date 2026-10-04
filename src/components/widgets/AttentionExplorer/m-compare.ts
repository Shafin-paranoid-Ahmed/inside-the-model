/** The compare figure: its frame function and its canvas drawing. */
import { headName, ease, f2, type Scene, headData, rowOf, type CompareFrame, type Opts, argmax, seg, pre, CMP_SEG, type Frame } from './core';
import { type Env, HEAD_CAT, rgba, fmt2, txt, bars, wordsRow, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, T } = pre(t, s, o);
  const { i, u } = o.pin != null ? { i: o.pin, u: 1 } : seg(tt, CMP_SEG, T);
  const fade = ease(u / 0.12), prevs = [0, 1].map((h) => (i > 0 ? Float64Array.from(rowOf(headData(s, s.layer, h).A, T, i - 1)) : new Float64Array(T)));
  const targets = [0, 1].map((h) => Float64Array.from(rowOf(headData(s, s.layer, h).A, T, i))), am = targets.map((r) => argmax(r));
  const status = `“${s.words[i]}”: head 1 looks mostly at “${s.words[am[0]]}” (${f2(targets[0][am[0]])}), head 2 at “${s.words[am[1]]}” (${f2(targets[1][am[1]])}).`;
  return { mode: 'compare', q: i, prevs, targets, fade, argmax: am, status, phase: s.words[i] };
}

function drawCompare(E: Env, f: CompareFrame) {
  const { c, P, s, narrow, W } = E, M = narrow ? 8 : 14, x0 = M + (narrow ? 18 : 26), w = W - x0 - M, ph = narrow ? 96 : 104;
  for (let h = 0; h < 2; h++) {
    const y = 40 + h * (ph + 52), col = P.cat[HEAD_CAT[h]];
    c.fillStyle = rgba(col); c.fillRect(M, y - 22, 10, 10);
    txt(E, `${headName(s.layer, h)}`, M + 16, y - 13, { sz: narrow ? 10.5 : 12, bold: true });
    if (f.fade < 1 && f.q > 0) { c.save(); c.globalAlpha = 1 - f.fade; bars(E, x0, y, w, ph, f.prevs[h], { vmax: 1, q: f.q - 1, cols: () => col }); c.restore(); }
    c.save(); c.globalAlpha = f.fade;
    bars(E, x0, y, w, ph, f.targets[h], { vmax: 1, q: f.q, cols: () => col, mark: f.argmax[h], label: (j) => (f.targets[h][j] >= 0.08 ? fmt2(f.targets[h][j]) : null), tip: (j) => `${headName(s.layer, h)}: ${quote(s.words[f.q])} → ${quote(s.words[j])}: ${fmt2(f.targets[h][j])}` }); c.restore();
    txt(E, '1', x0 - 4, y + 10, { sz: 9, al: 'right', col: P.muted, mono: true }); txt(E, '0', x0 - 4, y + ph, { sz: 9, al: 'right', col: P.muted, mono: true });
    wordsRow(E, x0, y + ph + 15, w, { hi: f.q, dim: (j) => (j > f.q ? 0.45 : 1), sz: narrow ? 9 : 11 });
  }
}

export const draw = drawCompare;
