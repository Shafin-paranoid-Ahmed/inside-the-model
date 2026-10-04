/** The arc diagram shared by the arcs and explorer figures. */
import { type Env, classOf, rgba, seqColor, fmt2, MIN_PX, txt, box, quote } from './ui';

export interface ArcGeo { x0: number; sp: number; yTok: number; top: number }
export function paintArcs(E: Env, g: ArcGeo, q: number, prev: ArrayLike<number>, target: ArrayLike<number>, fade: number) {
  if (fade < 1 && q > 0) arcLayer(E, g, q - 1, prev, 1 - fade, false);
  arcLayer(E, g, q, target, fade, true);
  arcTokens(E, g, q);
}
export function arcLayer(E: Env, g: ArcGeo, q: number, row: ArrayLike<number>, alpha: number, labels: boolean) {
  const { c, P, s } = E, T = s.T, xs = (i: number) => g.x0 + g.sp * (i + 0.5), y0 = g.yTok - 20, avail = y0 - g.top - 6;
  const order = Array.from({ length: q + 1 }, (_, j) => j).sort((a, b) => row[a] - row[b]);
  c.save(); c.globalAlpha = alpha;
  c.lineCap = 'round';
  for (const j of order) {
    const w = row[j]; if (w < 0.004) continue;
    const x1 = xs(q), x2 = xs(j);
    c.strokeStyle = rgba(seqColor(P, 0.25 + 0.75 * w), 0.4 + 0.6 * w); c.lineWidth = 1.2 + 11 * w;
    c.beginPath();
    let apex = y0;
    if (j === q) { const r = Math.min(avail * 0.4, 15 + 10 * w); c.arc(x1, y0 - r, r, 0, Math.PI * 2); apex = y0 - 2 * r; }
    else { const hgt = Math.min(avail, 16 + Math.abs(q - j) * (avail - 18) / Math.max(1, T - 1)); c.moveTo(x1, y0); c.bezierCurveTo(x1, y0 - hgt * 1.33, x2, y0 - hgt * 1.33, x2, y0); apex = y0 - hgt; }
    c.stroke();
    if (labels && w >= 0.08) {
      const mx = j === q ? Math.min(x1 + 22, E.W - 16) : (x1 + x2) / 2, my = j === q ? apex + 12 : apex - 3;
      c.font = `600 ${E.narrow ? MIN_PX : 10.5}px ${P.code}`; const tw = c.measureText(fmt2(w)).width + 6;
      c.fillStyle = rgba(P.bg, 0.85); c.fillRect(mx - tw / 2, my - 10, tw, 13);
      txt(E, fmt2(w), mx, my, { sz: E.narrow ? 9 : 10.5, al: 'center', bold: true, mono: true });
    }
  }
  c.restore();
}
export function arcTokens(E: Env, g: ArcGeo, q: number) {
  const { c, P, s } = E, T = s.T, xs = (i: number) => g.x0 + g.sp * (i + 0.5), stagger = E.narrow && g.sp < 46;
  for (let i = 0; i < T; i++) {
    const a = i > q ? 0.4 : 1, cx = xs(i), k = classOf(s.ids[i]);
    E.hits.push({ x: cx - g.sp / 2, y: g.yTok - 16, w: g.sp, h: 36, text: `look from ${quote(s.words[i])}`, row: i });
    if (i === q) { c.fillStyle = rgba(P.accent, 0.12); c.beginPath(); c.roundRect(cx - g.sp / 2 + 2, g.yTok - 16, g.sp - 4, 30 + (stagger && i % 2 ? 15 : 0), 4); c.fill(); box(E, cx - g.sp / 2 + 2, g.yTok - 16, g.sp - 4, 30 + (stagger && i % 2 ? 15 : 0), P.accent, 1.5, 4); }
    const off = stagger && i % 2 ? 15 : 0;
    txt(E, s.words[i], cx, g.yTok + off, { sz: E.narrow ? MIN_PX : 11.5, al: 'center', bold: i === q, a });
    if (k >= 0) { c.fillStyle = rgba(P.cat[k], a); c.fillRect(cx - g.sp * 0.3, g.yTok + 5 + off, g.sp * 0.6, 3); }
  }
  if (q < T - 1) {
    const x1 = xs(q + 1) - g.sp * 0.4, x2 = xs(T - 1) + g.sp * 0.4;
    c.strokeStyle = rgba(P.muted, 0.6); c.lineWidth = 1; c.beginPath(); c.moveTo(x1, g.yTok + 18 + (stagger ? 15 : 0)); c.lineTo(x2, g.yTok + 18 + (stagger ? 15 : 0)); c.stroke();
    txt(E, E.narrow ? 'later words: masked' : 'later words: masked, not yet read', (x1 + x2) / 2, g.yTok + 31 + (stagger ? 15 : 0), { sz: E.narrow ? MIN_PX : 10.5, al: 'center', col: P.muted });
  }
}
