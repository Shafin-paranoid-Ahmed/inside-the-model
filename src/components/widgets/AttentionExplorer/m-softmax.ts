/** The softmax figure: its frame function and its canvas drawing. */
import { ease, lerp, f2, sgn, sum, type Scene, headData, rowOf, SM, type SoftmaxFrame, type Opts, argmax, pre, type Frame } from './core';
import { type Env, fmt2, txt, maxAbs, bars, wordsRow, quote } from './ui';
import { maskedSoftmax } from './frames-a';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T, dk, qs } = pre(t, s, o);
  const { S } = headData(s), raw = Float64Array.from(rowOf(S, T, qs)), kk = 1 / Math.sqrt(dk);
  const un = maskedSoftmax(raw, qs, T), sc = maskedSoftmax(raw.map((x) => x * kk), qs, T);
  const noU = ease((tt - SM.noscale) / 1.5), maskU = ease((tt - SM.mask) / 1.0), p3 = ease((tt - SM.soft) / 1.4);
  const fShown = tt < SM.scale ? 1 : tt < SM.noscale ? lerp(1, kk, ease((tt - SM.scale) / 1.6)) : lerp(kk, 1, ease((tt - SM.noscale) / 1.2));
  const scaled = new Float64Array(T), w = new Float64Array(T);
  for (let j = 0; j < T; j++) {
    scaled[j] = raw[j] * fShown * (j > qs ? 1 - maskU : 1);
    w[j] = p3 * lerp(sc[j], un[j], noU);
  }
  const alpha: [number, number, number] = [ease((tt - 0.0) / 0.9), ease((tt - SM.scale) / 0.4), ease((tt - SM.soft + 0.1) / 0.4)];
  const mx = argmax(w), total = sum(w);
  const status = tt < SM.scale ? `Raw scores of “${s.words[qs]}” against every word, including words that come later.` :
    tt < SM.mask ? `Each score is divided by √${dk} = ${(1 / kk).toFixed(2)}.` :
    tt < SM.soft ? 'Later words are masked: their weight will be exactly 0.' :
    tt < SM.noscale ? (p3 < 0.999 ? `Softmax: positive weights appear. Their sum so far is ${total.toFixed(3)}.` : `Softmax: the weights are positive and add up to ${total.toFixed(3)}. The largest, ${f2(w[mx])}, is on “${s.words[mx]}”.`) :
    `Without the division the largest weight is ${f2(w[mx])} on “${s.words[mx]}” (scaled: ${f2(sc[argmax(sc)])}).`;
  return { mode: 'softmax', qi: qs, raw, scaled, w, f: fShown, maskU, alpha, sum: total, maxW: w[mx], maxIdx: mx, status, phase };
}

function drawSoftmax(E: Env, f: SoftmaxFrame) {
  const { c, P, s, narrow, W } = E, qi = f.qi, M = narrow ? 8 : 14, x0 = M + (narrow ? 22 : 34), w = W - x0 - M, ph = narrow ? 92 : 104;
  const vmax = maxAbs(f.raw), y1 = 24, y2 = y1 + ph + 26, y3 = y2 + ph + 26;
  const names = [narrow ? `1. Raw scores of ${quote(s.words[qi])}` : `1. Raw scores of ${quote(s.words[qi])} against each word`, `2. Divided by √${s.dk} = ${(Math.sqrt(s.dk)).toFixed(2)}` + (f.f > 0.999 ? (narrow ? ' (undone)' : ' (division undone)') : ''), narrow ? '3. Softmax: weights add up to 1' : '3. Softmax: positive weights that add up to 1'];
  const ys = [y1, y2, y3];
  ys.forEach((y, k) => txt(E, names[k], M, y - 8, { sz: narrow ? 10 : 12, bold: true, a: f.alpha[k] > 0.05 || k === 0 ? 1 : 0.25 }));
  const lab = (v: number) => sgn(v, narrow ? 1 : 2);
  bars(E, x0, y1, w, ph, f.raw, { signed: true, vmax, q: qi, grow: f.alpha[0], label: (j) => lab(f.raw[j]), tip: (j) => `${quote(s.words[j])}: raw score ${sgn(f.raw[j])}` });
  c.globalAlpha = 1;
  if (f.alpha[1] > 0.01) { c.save(); c.globalAlpha = f.alpha[1]; bars(E, x0, y2, w, ph, f.scaled, { signed: true, vmax, q: qi, fade: f.maskU, label: (j) => (j > qi && f.maskU > 0.5 ? null : lab(f.scaled[j])), tip: (j) => `${quote(s.words[j])}: scaled score ${sgn(f.scaled[j])}` }); c.restore(); }
  c.save(); c.globalAlpha = Math.max(0.0, f.alpha[2]);
  bars(E, x0, y3, w, ph, f.w, { vmax: 1, q: qi, mark: f.alpha[2] > 0.5 ? f.maxIdx : undefined, label: (j) => (f.w[j] > 0.015 ? fmt2(f.w[j]) : null), tip: (j) => `${quote(s.words[j])}: weight ${fmt2(f.w[j])}` });
  c.restore();
  txt(E, '+' + sgn(vmax, 1), x0 - 4, y1 + 10, { sz: 9, al: 'right', col: P.muted, mono: true });
  txt(E, '−' + sgn(vmax, 1), x0 - 4, y1 + ph, { sz: 9, al: 'right', col: P.muted, mono: true });
  txt(E, '1', x0 - 4, y3 + 10, { sz: 9, al: 'right', col: P.muted, mono: true });
  txt(E, '0', x0 - 4, y3 + ph, { sz: 9, al: 'right', col: P.muted, mono: true });
  if (f.alpha[2] > 0.05) txt(E, `sum = ${f.sum.toFixed(3)}`, W - M, y3 - 8, { sz: narrow ? 10 : 12, al: 'right', bold: true, mono: true, a: f.alpha[2] });
  wordsRow(E, x0, y3 + ph + 15, w, { hi: undefined, dim: (j) => (j > qi ? 0.45 : 1), sz: narrow ? 9 : 11 });
}

export const draw = drawSoftmax;
