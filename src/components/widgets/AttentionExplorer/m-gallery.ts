/** The gallery figure: its frame function and its canvas drawing. */
import { HEADS, headName, f2, type Scene, headData, rowOf, type GalleryFrame, type Opts, argmax, seg, pre, GAL_SEG, type Frame } from './core';
import { type Env, seqColor, fmt2, txt, box, short, grid, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, T } = pre(t, s, o);
  const { i } = o.pin != null ? { i: o.pin } : seg(tt, GAL_SEG, T);
  const tops = HEADS.map(([l, h], head) => { const { A } = headData(s, l, h), row = rowOf(A, T, i), b = argmax(row); return { head, word: b, weight: row[b] }; });
  const status = `“${s.words[i]}” looks at ` + tops.map((x, k) => `“${s.words[x.word]}” ${f2(x.weight)} (${headName(...HEADS[k])})`).join(', ') + '.';
  return { mode: 'gallery', q: i, tops, status, phase: s.words[i] };
}

function drawGallery(E: Env, f: GalleryFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, cs = narrow ? Math.max(10, Math.min(14, Math.floor((W / 2 - 52) / T))) : 22, lab = narrow ? 46 : 56, M = narrow ? 6 : 14, panelW = lab + T * cs;
  const colX = [narrow ? M : (W / 2 - panelW) / 2, narrow ? W / 2 + 2 : W / 2 + (W / 2 - panelW) / 2], hdr = narrow ? 46 : 40, panelH = 54 + hdr + T * cs + (narrow ? 10 : 18);
  HEADS.forEach(([l, h], k) => {
    const x0 = colX[k % 2], y0 = 10 + Math.floor(k / 2) * panelH, gx = x0 + lab, gy = y0 + 50 + hdr, { A } = headData(s, l, h), st = E.stats?.[k];
    txt(E, headName(l, h), x0, y0 + 11, { sz: narrow ? 10 : 12, bold: true });
    const wd = f.tops[k];
    txt(E, narrow ? `${quote(s.words[f.q])} → ${quote(short(E, s.words[wd.word], 6))} ${fmt2(wd.weight)}` : `${quote(s.words[f.q])} looks mostly at ${quote(s.words[wd.word])} (${fmt2(wd.weight)})`, x0, y0 + 26, { sz: narrow ? 9 : 10.5, col: P.muted });
    const q3 = (v: number) => v.toFixed(2).replace(/^0/, '');
    txt(E, st ? (narrow ? `prev ${q3(st.previous)} det ${q3(st.determiner)} verb ${q3(st.verb)}` : `200 sentences: previous ${q3(st.previous)} · determiner ${q3(st.determiner)} · verb ${q3(st.verb)}`) : '200 sentences: calculating…', x0, y0 + 42, { sz: narrow ? 9 : 10.5, col: P.muted });
    for (let j = 0; j < T; j++) { c.save(); c.translate(gx + (j + 0.5) * cs - 2, gy - 4); c.rotate(narrow ? -Math.PI / 2 : -Math.PI / 3); txt(E, short(E, s.words[j], 6), 0, 0, { sz: narrow ? 8 : 9.5 }); c.restore(); }
    for (let i = 0; i < T; i++) txt(E, short(E, s.words[i], narrow ? 5 : 7), gx - 5, gy + (i + 0.5) * cs + 3.5, { sz: narrow ? 8.5 : 10, al: 'right', bold: i === f.q, col: i === f.q ? P.accent : P.text });
    grid(E, gx, gy, cs, cs, T, T, (i, j) => A[i * T + j], (v) => seqColor(P, v), { masked: (i, j) => j > i, tip: (i, j) => `${headName(l, h)}: ${quote(s.words[i])} → ${quote(s.words[j])}: ${j > i ? '0 (masked)' : fmt2(A[i * T + j])}`, row: (i) => i });
    box(E, gx - 1, gy + f.q * cs - 1, T * cs + 1, cs + 1, P.accent, 2, 2);
  });
}

export const draw = drawGallery;
export { ablationMasses, ablatedMass, sampleMass, sampleStats, headStats } from './stats';
