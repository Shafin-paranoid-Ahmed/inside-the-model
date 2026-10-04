/** The ablate figure: its frame function and its canvas drawing. */
import { HEADS, headName, sum, type Scene, AB_SEG, ABLATE_LABELS, type AblateFrame, type Opts, seg, pre, type Frame } from './core';
import { type Env, rgba, fmt2, txt, short, quote } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, T } = pre(t, s, o);
  const mass = o.mass ?? [], P = T, base = Float64Array.from(mass[0] ?? new Array(P).fill(NaN));
  const state = o.toggles && o.customMass ? -1 : seg(tt, AB_SEG, 5).i;
  const target = state < 0 ? o.customMass! : (mass[state] ?? base);
  // Bars, narration and averages switch together at the segment boundary (no in-between state to mis-state).
  const cur = Float64Array.from(target);
  const valid = Array.from(cur).filter((x) => !Number.isNaN(x)), validBase = Array.from(base).filter((x) => !Number.isNaN(x));
  const avg = valid.length ? sum(valid) / valid.length : NaN, avgFull = validBase.length ? sum(validBase) / validBase.length : NaN;
  const off = HEADS.map((_, k) => (state < 0 ? !!o.toggles![k] : state === k + 1));
  const names = HEADS.filter((_, k) => off[k]).map(([l, h]) => headName(l, h));
  const status = !names.length ? `Full model: average probability on grammatical next words ${avgFull.toFixed(3)}.` :
    `${names.join(' and ')} switched off: ${avg.toFixed(3)} against ${avgFull.toFixed(3)} for the full model.`;
  return { mode: 'ablate', state, mass: cur, base, avg, avgFull, off, status, phase: state < 0 ? 'your toggles' : ABLATE_LABELS[state] };
}

function drawAblate(E: Env, f: AblateFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, M = narrow ? 8 : 14, x0 = M + (narrow ? 20 : 30), w = W - x0 - M, y0 = 56, ph = narrow ? 150 : 190, n = T, bw = w / n;
  txt(E, narrow ? 'Probability on grammatical next words' : 'Probability the model gives to grammatical next words, after each word', M, 18, { sz: narrow ? 10 : 12, bold: true });
  txt(E, f.off.some(Boolean) ? 'switched off: ' + HEADS.filter((_, k) => f.off[k]).map(([l, h]) => `layer ${l + 1} head ${h + 1}`).join(' and ') : 'all four heads on', M, 36, { sz: narrow ? 10 : 11.5, col: f.off.some(Boolean) ? P.accent : P.muted, bold: true });
  c.strokeStyle = rgba(P.rule); c.lineWidth = 1;
  for (const v of [0, 0.5, 1]) { const y = y0 + ph - v * (ph - 6); c.beginPath(); c.moveTo(x0, y + 0.5); c.lineTo(x0 + w, y + 0.5); c.stroke(); txt(E, String(v), x0 - 4, y + 3, { sz: 9, al: 'right', col: P.muted, mono: true }); }
  for (let i = 0; i < n; i++) {
    const m = f.mass[i], b = f.base[i], bx = x0 + i * bw + bw * 0.14, bwid = bw * 0.36;
    if (Number.isNaN(m)) { txt(E, 'n/a', x0 + (i + 0.5) * bw, y0 + ph - 8, { sz: 9, al: 'center', col: P.muted }); continue; }
    const hb = b * (ph - 6), hm = m * (ph - 6);
    c.fillStyle = rgba(P.muted, 0.45); c.fillRect(bx, y0 + ph - hb, bwid, hb);
    c.fillStyle = rgba(P.accent); c.fillRect(bx + bwid + 2, y0 + ph - hm, bwid, hm);
    if (b - m > 0.05) txt(E, fmt2(m), bx + bwid * 1.5 + 2, y0 + ph - hm - 3, { sz: narrow ? 8.5 : 9.5, al: 'center', mono: true, bold: true });
    E.hits.push({ x: x0 + i * bw, y: y0, w: bw, h: ph, text: `after ${quote(s.words[i])}: full model ${fmt2(b)}, this setting ${fmt2(m)}` });
    txt(E, short(E, s.words[i], narrow ? 5 : 8), x0 + (i + 0.5) * bw, y0 + ph + 15, { sz: narrow ? 9 : 11, al: 'center' });
  }
  const ly = y0 + ph + 36, ay = narrow ? ly + 18 : ly;
  c.fillStyle = rgba(P.muted, 0.45); c.fillRect(M, ly - 9, 10, 10); txt(E, 'full model', M + 15, ly, { sz: 10.5 });
  c.fillStyle = rgba(P.accent); c.fillRect(M + 100, ly - 9, 10, 10); txt(E, 'this setting', M + 115, ly, { sz: 10.5 });
  txt(E, `average: ${f.avg.toFixed(3)} (full model ${f.avgFull.toFixed(3)})`, narrow ? M : W - M, ay, { sz: narrow ? 10.5 : 12, bold: true, al: narrow ? 'left' : 'right', mono: true });
}

export const draw = drawAblate;
export { ablationMasses, ablatedMass, sampleMass, sampleStats, headStats } from './stats';
