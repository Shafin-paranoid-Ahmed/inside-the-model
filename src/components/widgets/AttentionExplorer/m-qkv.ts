/** The qkv figure: its frame function and its canvas drawing. */
import { clamp01, ease, sgn, type Scene, weightMatrix, type QkvFrame, type Opts, seg, pre, QKV_SEG, type Frame } from './core';
import { type Env, rgba, signed, txt, box, short, grid, maxAbs } from './ui';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, phase, T, dk } = pre(t, s, o);
  const done = tt >= T * QKV_SEG - 1e-9;
  const { i, u } = o.pin != null ? { i: o.pin, u: 1 } : seg(Math.min(tt, T * QKV_SEG - 1e-9), QKV_SEG, T);
  const reveal = new Float64Array(T);
  for (let r = 0; r < T; r++) reveal[r] = done ? 1 : r < i ? 1 : r === i ? ease((u - 0.3) / 0.55) : 0;
  const status = done ? `All ${T} words now have a query, a key and a value: three ${T} × ${dk} grids.` :
    `“${s.words[i]}”: its ${s.d} numbers become a query, a key and a value of ${dk} numbers each.`;
  return { mode: 'qkv', i: done ? T - 1 : i, u: done ? 1 : u, flow: done ? 1 : clamp01(u / 0.65), reveal, active: !done && u < 0.95, status, phase };
}

/** Phone layout: X on top, the three results side by side, the 16 x 8 matrices reduced to the "× W" label. */
function drawQkvNarrow(E: Env, f: QkvFrame) {
  const { c, P, s, W } = E, T = s.T, d = s.d, dk = s.dk, L = s.trace.layers[s.layer], X = L.ln1;
  const rowH = 14, cwX = 14, lab = 54, xX = lab + 2, y0 = 36, cwO = 9, rowO = 11, gw = dk * cwO, gap = (W - 3 * gw) / 4, yO = y0 + T * rowH + 52;
  const names = ['Q', 'K', 'V'] as const, outs = names.map((n) => L[n.toLowerCase() as 'q'][s.head]);
  txt(E, `Word vectors (${T} × ${d})`, xX, 16, { sz: 12, bold: true });
  for (let i = 0; i < T; i++) txt(E, s.words[i], xX - 5, y0 + (i + 0.5) * rowH + 4, { sz: 11.5, al: 'right', bold: f.i === i && f.active, col: f.i === i && f.active ? P.accent : P.text });
  grid(E, xX, y0, cwX, rowH, T, d, (i, j) => X[i * d + j], signed(P, 2.5), { tip: (i, j) => `x · ${s.words[i]} · number ${j + 1}: ${sgn(X[i * d + j])}` });
  box(E, xX - 1, y0 + f.i * rowH - 1, d * cwX + 1, rowH + 1, P.accent, 1.5, 2);
  const p0 = { x: xX + d * cwX, y: y0 + (f.i + 0.5) * rowH };
  names.forEach((n, g) => {
    const xO = gap + g * (gw + gap), o = outs[g];
    grid(E, xO, yO, cwO, rowO, T, dk, (i, j) => o[i * dk + j] * f.reveal[i], signed(P, 4), { vis: (i) => (f.reveal[i] > 0.02 ? 1 : 0), tip: (i, j) => `${n} · ${s.words[i]} · number ${j + 1}: ${sgn(o[i * dk + j])}` });
    box(E, xO - 1, yO + f.i * rowO - 1, gw + 1, rowO + 1, P.accent, 1.5, 2);
    if (f.active && f.flow > 0 && f.flow < 1) {
      const p3 = { x: xO + gw / 2, y: yO + (f.i + 0.5) * rowO }, my = (p0.y + p3.y) / 2 + 20;
      c.strokeStyle = rgba(P.accent, 0.3); c.lineWidth = 1; c.beginPath(); c.moveTo(p0.x, p0.y); c.bezierCurveTo(p0.x + 14, my, p3.x, yO - 28, p3.x, yO - 24); c.stroke();
      for (let k = 0; k < 4; k++) {
        const u = f.flow - k * 0.07; if (u < 0 || u > 1) continue; const it = 1 - u;
        const px = it ** 3 * p0.x + 3 * it * it * u * (p0.x + 14) + 3 * it * u * u * p3.x + u ** 3 * p3.x, py = it ** 3 * p0.y + 3 * it * it * u * my + 3 * it * u * u * (yO - 28) + u ** 3 * (yO - 24);
        c.fillStyle = rgba(P.accent, 1 - k * 0.2); c.beginPath(); c.arc(px, py, 3.2 - k * 0.4, 0, Math.PI * 2); c.fill();
      }
    }
  });
  names.forEach((n, g) => {
    const xO = gap + g * (gw + gap);
    c.fillStyle = rgba(P.bg, 1); c.fillRect(xO - 2, yO - 34, gw + 4, 32);
    txt(E, `${n} = X·W_${n}`, xO, yO - 20, { sz: 11.5, bold: true });
    txt(E, `${T} × ${dk}`, xO, yO - 6, { sz: 11.5, col: P.muted });
  });
}

function drawQkv(E: Env, f: QkvFrame) {
  if (E.narrow) return drawQkvNarrow(E, f);
  const { c, P, s, narrow } = E, T = s.T, d = s.d, dk = s.dk, L = s.trace.layers[s.layer], X = L.ln1;
  const lab = narrow ? 46 : 76, rowH = narrow ? 14 : 15, cwX = narrow ? 14 : 10, cwO = narrow ? 14 : 10, cwW = narrow ? 6 : 5, y0 = narrow ? 34 : 66;
  const xX = lab + 2, wideStep = 14 + 16 * 0 + 8 * cwW + 10 + 8 * cwO;
  const names = ['Q', 'K', 'V'] as const, hh = 16 * cwW;
  const xv = 2.5, vq = 4;
  txt(E, `Word vectors, after layer norm (${T} × ${d})`, xX, 16, { sz: narrow ? 10 : 11.5, bold: true });
  for (let i = 0; i < T; i++) txt(E, short(E, s.words[i], narrow ? 6 : 9), xX - 5, y0 + (i + 0.5) * rowH + 3.5, { sz: narrow ? 9 : 10.5, al: 'right', bold: f.i === i && f.active, col: f.i === i && f.active ? P.accent : P.text });
  grid(E, xX, y0, cwX, rowH, T, d, (i, j) => X[i * d + j], signed(P, xv), {
    tip: (i, j) => `x · ${s.words[i]} · number ${j + 1}: ${sgn(X[i * d + j])}`,
  });
  if (f.active || f.i >= 0) box(E, xX - 1, y0 + f.i * rowH - 1, d * cwX + 1, rowH + 1, P.accent, 1.5, 2);
  const mats = names.map((n) => weightMatrix(s.w, s.layer, n.toLowerCase() as 'q'));
  const outs = names.map((n) => s.trace.layers[s.layer][n.toLowerCase() as 'q'][s.head]);
  names.forEach((n, g) => {
    const xW = narrow ? xX : xX + d * cwX + 14 + g * wideStep, yW = narrow ? y0 + T * rowH + 36 + g * (Math.max(T * rowH, hh) + 28) : y0;
    const xO = xW + 8 * cwW + (narrow ? 16 : 10), M = mats[g], yO = yW;
    txt(E, `× W_${n}`, xW, yW - 17, { sz: narrow ? 10 : 11.5, bold: true });
    txt(E, `${d} × ${dk}`, xW, yW - 6, { sz: narrow ? 9 : 10, col: P.muted });
    grid(E, xW, yW, cwW, cwW, d, dk, (i, j) => M[i * d + s.head * dk + j], signed(P, maxAbs(M)), { tip: (i, j) => `W_${n} · in ${i + 1}, out ${j + 1}: ${sgn(M[i * d + s.head * dk + j], 3)}` });
    txt(E, `= ${n}`, xO, yO - 17, { sz: narrow ? 10 : 11.5, bold: true });
    txt(E, `${T} × ${dk}`, xO, yO - 6, { sz: narrow ? 9 : 10, col: P.muted });
    grid(E, xO, yO, cwO, rowH, T, dk, (i, j) => outs[g][i * dk + j] * f.reveal[i], signed(P, vq), {
      vis: (i) => (f.reveal[i] > 0.02 ? 1 : 0), tip: (i, j) => `${n} · ${s.words[i]} · number ${j + 1}: ${sgn(outs[g][i * dk + j])}`,
    });
    if (f.reveal[f.i] > 0 || f.active) box(E, xO - 1, yO + f.i * rowH - 1, dk * cwO + 1, rowH + 1, P.accent, 1.5, 2);
    if (f.active && f.flow > 0 && f.flow < 1) {
      const p0 = { x: xX + d * cwX, y: y0 + (f.i + 0.5) * rowH }, p1 = { x: xW, y: yW + hh / 2 }, p2 = { x: xW + dk * cwW, y: yW + hh / 2 }, p3 = { x: xO, y: yO + (f.i + 0.5) * rowH };
      const bez = (a: typeof p0, b: typeof p0, u: number) => { const mx = (a.x + b.x) / 2; const it = 1 - u; return { x: it ** 3 * a.x + 3 * it * it * u * mx + 3 * it * u * u * mx + u ** 3 * b.x, y: it ** 3 * a.y + 3 * it * it * u * a.y + 3 * it * u * u * b.y + u ** 3 * b.y }; };
      c.strokeStyle = rgba(P.accent, 0.35); c.lineWidth = 1;
      for (const [a, b] of [[p0, p1], [p2, p3]] as const) { c.beginPath(); c.moveTo(a.x, a.y); const mx = (a.x + b.x) / 2; c.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y); c.stroke(); }
      for (let k = 0; k < 4; k++) {
        const u = f.flow - k * 0.07; if (u < 0 || u > 1) continue;
        const p = u < 0.5 ? bez(p0, p1, u * 2) : bez(p2, p3, u * 2 - 1);
        c.fillStyle = rgba(P.accent, 1 - k * 0.2); c.beginPath(); c.arc(p.x, p.y, 3.2 - k * 0.4, 0, Math.PI * 2); c.fill();
      }
    }
  });
}

export const draw = drawQkv;
