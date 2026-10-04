/** The explorer figure: its frame function and its canvas drawing. */
import { ease, f2, sgn, type Scene, headData, type ExplorerFrame, type Opts, seg, easedRow, topWord, pre, EXP_SEG, type Frame } from './core';
import { type Env, seqColor, fmt2, txt, box, short, grid, strip, maxAbs, bars, quote } from './ui';
import { paintArcs } from './arcs';
import { pipelineRow } from './frames-a';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, T } = pre(t, s, o);
  const { i, u } = o.pin != null ? { i: o.pin, u: 1 } : seg(tt, EXP_SEG, T);
  const pipe = pipelineRow(s, i), prev = i > 0 ? easedRow(s, i - 1, 1) : new Float64Array(T), fade = ease(u / 0.12);
  const stage: ExplorerFrame['stage'] = o.pin != null ? [1, 1, 1, 1] : [ease(u / 0.2), ease((u - 0.25) / 0.2), ease((u - 0.5) / 0.2), ease((u - 0.75) / 0.2)];
  const top = topWord(s, pipe.w);
  const step = o.pin != null || u >= 0.95 ? 'mixes the values' : u < 0.25 ? 'scores every word' : u < 0.5 ? 'scales the scores' : u < 0.75 ? 'turns them into weights' : 'mixes the values';
  const status = `“${s.words[i]}” ${step}; most weight (${f2(top.w)}) lands on “${top.word}”.`;
  return { mode: 'explorer', q: i, prev, fade, stage, pipe, status, phase: s.words[i] };
}

function drawExplorer(E: Env, f: ExplorerFrame) {
  const { c, P, s, narrow, W } = E, T = s.T, qi = f.q, M = narrow ? 8 : 14;
  const arcsH = narrow ? 224 : 200;
  paintArcs(E, { x0: M, sp: (W - 2 * M) / T, yTok: arcsH - 44, top: 14 }, qi, f.prev, f.pipe.w, f.fade);
  const cs = narrow ? 22 : 26, gy = arcsH + (narrow ? 46 : 52), gx = narrow ? 56 : 84, { A } = headData(s);
  txt(E, 'Attention weights, one row per word', 4, arcsH + 8, { sz: narrow ? 10 : 12, bold: true });
  for (let j = 0; j < T; j++) { c.save(); c.translate(gx + (j + 0.5) * cs - 2, gy - 5); c.rotate(-Math.PI / 3); txt(E, short(E, s.words[j], 7), 0, 0, { sz: narrow ? 9 : 10.5 }); c.restore(); }
  for (let i = 0; i < T; i++) txt(E, short(E, s.words[i], 7), gx - 6, gy + (i + 0.5) * cs + 4, { sz: narrow ? 9.5 : 11, al: 'right', bold: i === qi, col: i === qi ? P.accent : P.text });
  grid(E, gx, gy, cs, cs, T, T, (i, j) => A[i * T + j], (v) => seqColor(P, v), {
    masked: (i, j) => j > i, num: (i, j) => (!narrow && A[i * T + j] >= 0.1 && j <= i ? fmt2(A[i * T + j]).replace(/^0/, '') : null),
    tip: (i, j) => `${quote(s.words[i])} → ${quote(s.words[j])}: weight ${j > i ? '0 (masked)' : fmt2(A[i * T + j])}`, row: (i) => i,
  });
  box(E, gx - 1, gy + qi * cs - 1, T * cs + 1, cs + 1, P.accent, 2, 2);
  // pipeline for the chosen word
  const px = narrow ? 8 : gx + T * cs + 36, py = narrow ? gy + T * cs + 30 : arcsH + 8, pw = W - px - M, ph = narrow ? 52 : 56, p = f.pipe;
  const vmax = maxAbs(f.pipe.raw), rows = [
    { name: `1. Score the query of ${quote(s.words[qi])} against each key`, a: f.stage[0] },
    { name: `2. Divide by √${s.dk} = ${Math.sqrt(s.dk).toFixed(2)}`, a: f.stage[1] },
    { name: '3. Softmax: weights that add up to 1', a: f.stage[2] },
  ];
  rows.forEach((r, k) => {
    const y = py + 18 + k * (ph + 24);
    txt(E, r.name, px, y - 5, { sz: narrow ? 9.5 : 11, bold: true, a: 0.35 + 0.65 * r.a });
    bars(E, px, y, pw, ph, k === 0 ? p.raw : k === 1 ? p.scaled : p.w, { signed: k < 2, vmax: k < 2 ? vmax : 1, q: qi, grow: r.a, fade: 1, label: (j) => (k < 2 ? (j <= qi ? sgn(k === 0 ? p.raw[j] : p.scaled[j], 1) : null) : p.w[j] > 0.05 ? fmt2(p.w[j]) : null), mark: k === 2 && r.a > 0.9 ? p.w.reduce((b, x, j) => (x > p.w[b] ? j : b), 0) : undefined });
  });
  const y4 = py + 18 + 3 * (ph + 24), { v } = headData(s), vm = maxAbs(v), cw = Math.min(30, (pw - 4) / s.dk);
  txt(E, `4. Mix the values with those weights: output of ${quote(s.words[qi])}`, px, y4 - 5, { sz: narrow ? 9.5 : 11, bold: true, a: 0.35 + 0.65 * f.stage[3] });
  strip(E, px, y4, cw, 22, p.out, vm, { scale: f.stage[3], vis: f.stage[3] > 0.02 ? 1 : 0.12, tip: (kk) => `output · number ${kk + 1}: ${sgn(p.out[kk])}` });
}

export const draw = drawExplorer;
