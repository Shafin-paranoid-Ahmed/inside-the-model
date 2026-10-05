<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, clamp, type Box } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    DEG, DIMS, HOLD, makeModel, makeScree, rotateFrame, eigenFrame, projectFrame, screeFrame, keepOne, recon, varAt, residAt, duration, markers,
    type Mode, type Pt,
  } from './sim';

  interface Props {
    title?: string;
    /** rotate: a line turns through 180° and the shadow variance / error curves trace out. eigen: the covariance matrix and its eigenvectors. project: points slide onto a component. scree: explained variance and keep-k on six measurements. */
    mode?: Mode;
    /** Tilt of the 2D cloud in degrees (eigen and project let the learner change it). */
    tilt?: number;
  }
  let { title = 'Find the direction of most variance', mode = 'rotate', tilt: tilt0 = 28 }: Props = $props();
  const init = untrack(() => ({ mode, tilt: tilt0 }));
  const MODE = init.mode;
  const Tdur = duration(MODE), MK = markers(MODE);
  const scree = makeScree();

  let tilt = $state(init.tilt);
  let comp = $state<0 | 1>(0);
  let over = $state<number | null>(null); // rotate: angle set by hand
  let keepK = $state<number | null>(null); // scree: k set by hand
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {}, dragging = false;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const model = $derived(makeModel(tilt));
  const start = $derived(model.th1 - Math.PI / 2);
  const fR = $derived(MODE === 'rotate' ? rotateFrame(uiT, model, over) : null);
  const fE = $derived(MODE === 'eigen' ? eigenFrame(uiT, model) : null);
  const fP = $derived(MODE === 'project' ? projectFrame(uiT) : null);
  const fS = $derived(MODE === 'scree' ? screeFrame(uiT, scree, keepK) : null);
  const one = $derived(keepOne(model, comp));
  const focus = $derived(recon(model.pts[model.focus], one.u));
  const degOf = (rad: number) => { const d = ((rad / DEG + 90) % 180 + 180) % 180 - 90; return d; }; // (-90, 90]

  const narration = $derived.by(() => {
    if (fR) {
      if (over !== null) return 'You set the angle by hand. Press Play to watch the sweep again.';
      return fR.phase === 'worst axis' ? 'The line points across the cloud. The shadows bunch together and the points sit far from the line: little variance kept, a large error.'
        : fR.phase === 'sweep' ? 'The line turns. The teal bar (the spread of the shadows) swells and shrinks, and the curves trace the variance and the error for every angle.'
        : fR.phase === 'settle' ? 'Gliding back to the angle where the variance curve peaked.'
        : 'Best axis: the shadows spread out as far as they can and the points sit as close to the line as they can. Both curves single out the same angle.';
    }
    if (fE) {
      return fE.phase === 'probe' ? 'v is a unit vector (the short arrow); the teal arrow is C·v, what the covariance matrix does to it. They point in different directions.'
        : fE.phase === 'turn to PC1' ? 'Turning v toward the long axis of the cloud. Watch the teal arrow swing into line with it.'
        : fE.phase === 'PC1' ? `v sits on PC1 and C·v points the same way, ${f2(model.values[0])} times as long. That stretch factor is the eigenvalue λ₁.`
        : fE.phase === 'turn to PC2' ? 'Turning v a quarter turn to the short axis.'
        : `v sits on PC2 and C·v is parallel to it again, but only ${f2(model.values[1])} times as long: λ₂.`;
    }
    if (fP) {
      const name = comp === 0 ? 'PC1' : 'PC2';
      return fP.phase === 'cloud' ? `The cloud and the ${name} axis. Every point is described by two numbers.`
        : fP.phase === 'project' ? 'Every point slides straight (perpendicularly) onto the line.'
        : fP.phase === 'compressed' ? 'Now each point needs one number: its position z along the line. Two numbers have become one.'
        : comp === 0 ? 'Rebuilding a point from its one number puts it on the line. The magenta segments are the error: what was thrown away. They are short.'
        : 'Rebuilding a point from its one number puts it on the line. The magenta segments are the error: what was thrown away. Keeping the weak axis throws away almost everything.';
    }
    const k = fS!.k;
    if (keepK !== null) return `You keep ${k} of ${DIMS} components. Press Play to watch the components being dropped one by one.`;
    return k === DIMS ? 'All six components kept: nothing is lost. Each bar is the share of the total variance one principal component carries.'
      : k >= 3 ? 'Dropping a weak component: the variance kept and the error hardly move, because these components carry very little.'
      : k === 2 ? 'Two components left: the scatter on the right still shows the full picture, and most of the variance is kept.'
      : 'Only PC1 left: the points flatten onto a line and the variance of every other component is gone.';
  });

  $effect(() => { if (playing) { over = null; keepK = null; } speed; tilt; comp; wake(); });

  const minus = (v: number | string) => String(v).replace('-', '−');
  const pct = (v: number) => (100 * v).toFixed(v < 0.1 ? 1 : 0) + '%';
  const fv = (v: number[]) => `(${v.map((x) => f2(x)).join(', ')})`;
  const replay = (c: 0 | 1) => { comp = c; t = reducedMotion() ? Tdur : 0; playing = !reducedMotion(); wake(); };
  const setAngle = (deg: number) => { over = clamp(deg * DEG, start, start + Math.PI); playing = false; wake(); };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const tt = () => Math.min(t, Tdur);
    if (reducedMotion()) t = Tdur;

    function fieldView(b: Box, X: number, Y: number) {
      const u = Math.min(b.w / (2 * X), b.h / (2 * Y)), cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      return { u, cx, cy, b, px: (x: number) => cx + x * u, py: (y: number) => cy - y * u };
    }
    type View = ReturnType<typeof fieldView>;
    function arrow(x0: number, y0: number, x1: number, y1: number, col: string, w: number, dash = false, label = '', lo = 12) {
      const { ctx } = stage;
      if (Math.hypot(x1 - x0, y1 - y0) < 2) return;
      const ang = Math.atan2(y1 - y0, x1 - x0);
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4)); ctx.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      if (label) { ctx.font = stage.mono(12, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, x1 + lo * Math.cos(ang) + 8 * Math.sin(ang), y1 + lo * Math.sin(ang) - 8 * Math.cos(ang)); }
    }
    function axes(V: View) {
      const { ctx, c } = stage, b = V.b;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(b.x, V.py(0)); ctx.lineTo(b.x + b.w, V.py(0)); ctx.moveTo(V.px(0), b.y); ctx.lineTo(V.px(0), b.y + b.h); ctx.stroke();
    }
    const clip = (b: Box) => { const { ctx } = stage; ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip(); };
    function dots(V: View, pts: Pt[], col: string, r: number, al = 1) {
      const { ctx } = stage; ctx.fillStyle = col; ctx.globalAlpha = al; ctx.beginPath();
      for (const p of pts) { ctx.moveTo(V.px(p[0]) + r, V.py(p[1])); ctx.arc(V.px(p[0]), V.py(p[1]), r, 0, 6.2832); }
      ctx.fill(); ctx.globalAlpha = 1;
    }
    function text(s: string, x: number, y: number, col: string, font: string, align: CanvasTextAlign = 'left', base: CanvasTextBaseline = 'middle') {
      const { ctx } = stage; ctx.font = font; ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(s, x, y);
    }

    // ---------- rotate: shadows on a turning line, and the two curves
    function renderRotate(now: number) {
      const { ctx, w: W, h: H, c } = stage, m = model, f = rotateFrame(now, m, over), P = splitPanels(W, H, 0.5);
      ctx.clearRect(0, 0, W, H);
      const V = fieldView(P.main, 3.8, 3.3), cs = Math.cos(f.theta), sn = Math.sin(f.theta), sd = Math.sqrt(f.variance);
      clip(P.main); axes(V);
      const L = 12;
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(V.px(-L * cs), V.py(-L * sn)); ctx.lineTo(V.px(L * cs), V.py(L * sn)); ctx.stroke();
      ctx.strokeStyle = c.fwd.hex; ctx.globalAlpha = 0.4; ctx.lineWidth = 9; ctx.lineCap = 'butt'; ctx.beginPath(); ctx.moveTo(V.px(-sd * cs), V.py(-sd * sn)); ctx.lineTo(V.px(sd * cs), V.py(sd * sn)); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.strokeStyle = c.grad.hex; ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.beginPath();
      const feet: Pt[] = [];
      for (const p of m.pts) { const z = p[0] * cs + p[1] * sn, fx = z * cs, fy = z * sn; ctx.moveTo(V.px(p[0]), V.py(p[1])); ctx.lineTo(V.px(fx), V.py(fy)); feet.push([fx, fy]); }
      ctx.stroke(); ctx.globalAlpha = 1;
      dots(V, m.pts, c.muted.hex, 2.8, 0.85);
      dots(V, feet, c.fwd.hex, 2.4);
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(V.px(0), V.py(0), 2.5, 0, 6.2832); ctx.fill();
      ctx.restore();
      text('data, its shadows on the line, and the distances', P.main.x + 6, P.main.y + 10, c.muted.hex, stage.body(12, true));

      // the curves
      const S = P.side, o = { l: S.x + 42, r: S.x + S.w - 10, t: S.y + 30, b: S.y + S.h - 36 };
      const cx = scale(0, 180, o.l, o.r), cy = scale(0, m.trace * 1.12, o.b, o.t), start0 = m.th1 / DEG - 90;
      text('Every direction of the line', S.x + 2, S.y + 10, c.text.hex, stage.body(12.5));
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex; ctx.strokeRect(o.l, o.t, o.r - o.l, o.b - o.t);
      for (let a = Math.ceil(start0 / 30) * 30; a <= start0 + 180; a += 30) { const x = cx(a - start0); ctx.beginPath(); ctx.moveTo(x, o.b); ctx.lineTo(x, o.b + 4); ctx.stroke(); text(minus(a), x, o.b + 6, c.muted.hex, stage.mono(11), 'center', 'top'); }
      text('angle of the line (degrees)', o.r, S.y + S.h - 1, c.muted.hex, stage.body(12, true), 'right', 'bottom');
      for (const y of [0, 1, 2]) { text(String(y), o.l - 6, cy(y), c.muted.hex, stage.mono(11), 'right'); ctx.beginPath(); ctx.moveTo(o.l - 3, cy(y)); ctx.lineTo(o.l, cy(y)); ctx.stroke(); }
      ctx.save(); ctx.translate(S.x + 8, (o.t + o.b) / 2); ctx.rotate(-Math.PI / 2); text('mean square', 0, 0, c.muted.hex, stage.body(12, true), 'center', 'top'); ctx.restore();
      ctx.setLineDash([3, 3]); ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(o.l, cy(m.trace)); ctx.lineTo(o.r, cy(m.trace)); ctx.stroke();
      ctx.strokeStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(cx(90), o.t); ctx.lineTo(cx(90), o.b); ctx.stroke(); ctx.setLineDash([]);
      text('PC1', cx(90) + 4, o.t + 8, c.accent.hex, stage.mono(11, 700));
      text(`total ${f2(m.trace)}`, o.r - 3, cy(m.trace) - 7, c.muted.hex, stage.mono(11), 'right');
      const curve = (fn: (th: number) => number, col: string, upTo: number) => {
        ctx.lineJoin = 'round'; ctx.strokeStyle = col; ctx.globalAlpha = 0.28; ctx.lineWidth = 1.5; ctx.beginPath();
        for (let i = 0; i <= 90; i++) { const r = i * 2, x = cx(r), y = cy(fn(start0 * DEG + r * DEG)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(cx(0), cy(fn(start0 * DEG)));
        for (let r = 2; r < upTo / DEG; r += 2) ctx.lineTo(cx(r), cy(fn(start0 * DEG + r * DEG)));
        ctx.lineTo(cx(upTo / DEG), cy(fn(start0 * DEG + upTo))); ctx.stroke();
      };
      curve((th) => varAt(m, th), c.fwd.hex, f.reach); curve((th) => residAt(m, th), c.grad.hex, f.reach);
      for (const [v, col] of [[f.variance, c.fwd.hex], [f.resid, c.grad.hex]] as const) { ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx(f.rel / DEG), cy(v), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke(); }
    }

    // ---------- eigen: the covariance ellipse, eigenvector arrows and the probe
    function renderEigen(now: number) {
      const { ctx, w: W, h: H, c } = stage, m = model, f = eigenFrame(now, m);
      ctx.clearRect(0, 0, W, H);
      const V = fieldView({ x: 0, y: 0, w: W, h: H }, W < 540 ? 3.9 : 4.6, 3.6);
      clip(V.b); axes(V);
      dots(V, m.pts, c.muted.hex, 2.8, 0.7);
      const [s1, s2] = [Math.sqrt(m.values[0]), Math.sqrt(m.values[1])], [v1, v2] = m.vectors;
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]); ctx.beginPath();
      for (let i = 0; i <= 72; i++) { const a = (i / 72) * 6.2832, x = s1 * Math.cos(a) * v1[0] + s2 * Math.sin(a) * v2[0], y = s1 * Math.cos(a) * v1[1] + s2 * Math.sin(a) * v2[1]; if (i) ctx.lineTo(V.px(x), V.py(y)); else ctx.moveTo(V.px(x), V.py(y)); }
      ctx.stroke(); ctx.setLineDash([]);
      arrow(V.px(0), V.py(0), V.px(s1 * v1[0]), V.py(s1 * v1[1]), c.s5.hex, 3.4, false, 'PC1', 14);
      arrow(V.px(0), V.py(0), V.px(s2 * v2[0]), V.py(s2 * v2[1]), c.s3.hex, 3.4, false, 'PC2', 14);
      arrow(V.px(0), V.py(0), V.px(f.Cv[0]), V.py(f.Cv[1]), c.fwd.hex, 3, false, 'C·v', 14);
      arrow(V.px(0), V.py(0), V.px(f.v[0]), V.py(f.v[1]), c.accent.hex, 3, true, 'v', 12);
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(V.px(0), V.py(0), 3, 0, 6.2832); ctx.fill();
      ctx.restore();
      text('PC arrows have length √λ (the spread of the data along them)', 8, 10, c.muted.hex, stage.body(12, true));
    }

    // ---------- project: points slide onto the component line
    function renderProject(now: number) {
      const { ctx, w: W, h: H, c } = stage, m = model, f = projectFrame(now), { u } = keepOne(m, comp);
      ctx.clearRect(0, 0, W, H);
      const V = fieldView({ x: 0, y: 0, w: W, h: H }, W < 540 ? 3.9 : 4.6, 3.6);
      clip(V.b); axes(V);
      const L = 12;
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(V.px(-L * u[0]), V.py(-L * u[1])); ctx.lineTo(V.px(L * u[0]), V.py(L * u[1])); ctx.stroke();
      const cur: Pt[] = [], hats: Pt[] = [];
      ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 1.3; ctx.setLineDash([3, 3]); ctx.globalAlpha = f.err * 0.9; ctx.beginPath();
      for (const p of m.pts) {
        const { hat } = recon(p, u);
        cur.push([p[0] + (hat[0] - p[0]) * f.s, p[1] + (hat[1] - p[1]) * f.s]); hats.push(hat);
        ctx.moveTo(V.px(hat[0]), V.py(hat[1])); ctx.lineTo(V.px(p[0]), V.py(p[1]));
      }
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      if (f.s > 0.02) dots(V, m.pts, c.muted.hex, 2.6, 0.35 * Math.min(1, f.s * 3));
      dots(V, cur, c.muted.hex, 2.9, 1 - f.s);
      dots(V, cur, c.fwd.hex, 2.9, f.s);
      // the tracked point
      const p = m.pts[m.focus], fp = cur[m.focus];
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.arc(V.px(fp[0]), V.py(fp[1]), 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.arc(V.px(fp[0]), V.py(fp[1]), 2.3, 0, 6.2832); ctx.fill();
      if (f.s > 0.98) {
        const { z } = recon(p, u), hx = V.px(hats[m.focus][0]), hy = V.py(hats[m.focus][1]), side = z >= 0 ? 1 : -1;
        const lab = `z = ${f2(z)}`; ctx.font = stage.mono(12, 700); const tw = ctx.measureText(lab).width; ctx.fillStyle = c.raised.hex; ctx.globalAlpha = 0.9; ctx.fillRect(hx - 16 - tw, hy + 16 * side - 9, tw + 8, 18); ctx.globalAlpha = 1;
        text(lab, hx - 12, hy + 16 * side, c.accent.hex, stage.mono(12, 700), 'right');
      }
      ctx.restore();
      text(`the line is ${comp === 0 ? 'PC1' : 'PC2'}; the ringed point is tracked below`, 8, 10, c.muted.hex, stage.body(12, true));
    }

    // ---------- scree: explained-variance bars, cumulative line, and the PC1/PC2 scores
    function renderScree(now: number) {
      const { ctx, w: W, h: H, c } = stage, f = screeFrame(now, scree, keepK), P = splitPanels(W, H, 0.52);
      ctx.clearRect(0, 0, W, H);
      const B = P.main, o = { l: B.x + 38, r: B.x + B.w - 8, t: B.y + 26, b: B.y + B.h - 26 };
      const y = scale(0, 1, o.b, o.t), slot = (o.r - o.l) / DIMS, bw = slot * 0.62;
      text('Explained variance', B.x + 2, B.y + 10, c.text.hex, stage.body(12.5));
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex;
      for (const v of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(o.l, y(v)); ctx.lineTo(o.r, y(v)); ctx.stroke(); text(`${v * 100}%`, o.l - 5, y(v), c.muted.hex, stage.mono(11), 'right'); }
      ctx.setLineDash([4, 4]); ctx.strokeStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(o.l, y(0.9)); ctx.lineTo(o.r, y(0.9)); ctx.stroke(); ctx.setLineDash([]);
      text('90%', o.r, y(0.9) + 9, c.accent.hex, stage.mono(11, 700), 'right');
      for (let j = 0; j < DIMS; j++) {
        const x0 = o.l + j * slot + (slot - bw) / 2, h = scree.ratios[j] * (o.b - o.t), kept = f.w[j];
        ctx.fillStyle = c.fwd.hex; ctx.globalAlpha = 0.18 + 0.82 * kept; ctx.fillRect(x0, o.b - h, bw, h); ctx.globalAlpha = 1;
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.4; ctx.strokeRect(x0 + 0.7, o.b - h + 0.7, bw - 1.4, Math.max(h - 1.4, 0));
        text(pct(scree.ratios[j]), x0 + bw / 2, j === 0 ? o.b - h + 13 : o.b - h - 7, j === 0 && kept > 0.5 ? c.raised.hex : kept > 0.5 ? c.text.hex : c.muted.hex, stage.mono(11, 600), 'center');
        text(`PC${j + 1}`, o.l + (j + 0.5) * slot, o.b + 13, kept > 0.5 ? c.text.hex : c.muted.hex, stage.mono(11), 'center');
      }
      ctx.strokeStyle = c.s5.hex; ctx.lineWidth = 2; ctx.beginPath();
      scree.cum.forEach((v, j) => { const x = o.l + (j + 0.5) * slot; if (j) ctx.lineTo(x, y(v)); else ctx.moveTo(x, y(v)); }); ctx.stroke();
      scree.cum.forEach((v, j) => { ctx.fillStyle = f.w[j] > 0.5 ? c.s5.hex : c.raised.hex; ctx.strokeStyle = c.s5.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(o.l + (j + 0.5) * slot, y(v), 3.6, 0, 6.2832); ctx.fill(); ctx.stroke(); });

      // scores of PC1 against PC2, the plane you are left with when k is 2 or more
      const S = P.side, q = { l: S.x + 30, r: S.x + S.w - 6, t: S.y + 26, b: S.y + S.h - 24 };
      text('The data seen through PC1 and PC2', S.x + 2, S.y + 10, c.text.hex, stage.body(12.5));
      const R1 = 1.05 * Math.max(...scree.scores.map((r) => Math.abs(r[0]))), R2 = 1.05 * Math.max(...scree.scores.map((r) => Math.abs(r[1])));
      const sx = scale(-R1, R1, q.l, q.r), sy = scale(-R2, R2, q.b, q.t);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(q.l, q.t, q.r - q.l, q.b - q.t);
      ctx.beginPath(); ctx.moveTo(q.l, sy(0)); ctx.lineTo(q.r, sy(0)); ctx.moveTo(sx(0), q.t); ctx.lineTo(sx(0), q.b); ctx.stroke();
      ctx.fillStyle = c.fwd.hex; ctx.globalAlpha = 0.6; ctx.beginPath();
      for (const r of scree.scores) { const px = sx(r[0] * f.w[0]), py = sy(r[1] * f.w[1]); ctx.moveTo(px + 2.6, py); ctx.arc(px, py, 2.6, 0, 6.2832); }
      ctx.fill(); ctx.globalAlpha = 1;
      text('PC1 score', q.r, S.y + S.h - 1, c.muted.hex, stage.body(12, true), 'right', 'bottom');
      ctx.save(); ctx.translate(S.x + 8, (q.t + q.b) / 2); ctx.rotate(-Math.PI / 2); text('PC2 score', 0, 0, c.muted.hex, stage.body(12, true), 'center', 'top'); ctx.restore();
    }

    function render() {
      const v = tt();
      (MODE === 'rotate' ? renderRotate : MODE === 'eigen' ? renderEigen : MODE === 'project' ? renderProject : renderScree)(v);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: Tdur, loopFrom: 0, loopTo: Tdur + HOLD }).t;
      const v = tt();
      render(); tl.paint(v);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = v; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    uiT = tt();

    // rotate: drag in the scatter to aim the line
    const aim = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), P = splitPanels(stage.w, stage.h, 0.5), cx = P.main.x + P.main.w / 2, cy = P.main.y + P.main.h / 2;
      let a = Math.atan2(-(e.clientY - r.top - cy), e.clientX - r.left - cx);
      while (a < start) a += Math.PI;
      while (a > start + Math.PI) a -= Math.PI;
      over = a; playing = false; wake();
    };
    const inMain = (e: PointerEvent) => e.clientX - cv.getBoundingClientRect().left < splitPanels(stage.w, stage.h, 0.5).main.w + 4 && (!splitPanels(stage.w, stage.h, 0.5).stacked || e.clientY - cv.getBoundingClientRect().top < splitPanels(stage.w, stage.h, 0.5).main.h);
    const down = (e: PointerEvent) => { if (MODE === 'rotate' && inMain(e)) { dragging = true; cv.setPointerCapture(e.pointerId); aim(e); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (MODE !== 'rotate') return; if (dragging) aim(e); else cv.style.cursor = inMain(e) ? 'crosshair' : ''; };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });

  const legend = MODE === 'rotate'
    ? [{ label: 'the line', color: '--accent', shape: 'line' as const }, { label: 'shadow variance', color: '--viz-forward', shape: 'line' as const }, { label: 'squared distance to the line', color: '--viz-gradient', shape: 'line' as const }]
    : MODE === 'eigen'
      ? [{ label: 'PC1', color: '--viz-seq-5', shape: 'line' as const }, { label: 'PC2', color: '--viz-seq-3', shape: 'line' as const }, { label: 'C·v', color: '--viz-forward', shape: 'line' as const }, { label: 'v', color: '--accent', shape: 'line' as const }]
      : MODE === 'project'
        ? [{ label: 'the component', color: '--accent', shape: 'line' as const }, { label: 'projected points', color: '--viz-forward' }, { label: 'error', color: '--viz-gradient', shape: 'line' as const }]
        : [{ label: 'variance share', color: '--viz-forward' }, { label: 'running total', color: '--viz-seq-5', shape: 'line' as const }, { label: '90% line', color: '--accent', shape: 'line' as const }];
  const figLabel = MODE === 'rotate'
    ? 'Left: a tilted elongated cloud of points with a line through the middle that turns through half a circle, each point dropping a perpendicular shadow on the line. Right: two curves against the angle of the line, the variance of the shadows which peaks along the cloud and the squared distance to the line which is lowest at the same angle.'
    : MODE === 'eigen'
      ? 'The same cloud with its covariance ellipse, two eigenvector arrows along the long and short axes, and a turning unit vector v with the arrow C times v. The two arrows point the same way only when v lies on an eigenvector.'
      : MODE === 'project'
        ? 'The cloud with one principal axis drawn as a line. The points slide perpendicularly onto the line and dashed segments then show the reconstruction error for each point.'
        : 'Left: bars showing the share of total variance carried by each of six principal components, with a running-total line and a 90 percent line. Right: the data plotted by its first two principal scores, which flattens onto a line when only one component is kept.';
  const aspect = MODE === 'rotate' || MODE === 'scree' ? '720 / 400' : '16 / 9';
  const narrowAspect = MODE === 'rotate' || MODE === 'scree' ? '360 / 600' : '4 / 3';
</script>

<FigureCard {title} role={MODE === 'rotate' ? 'group' : 'img'} {aspect} {narrowAspect} label={figLabel} {legend}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={Tdur} markers={MK} label="Timeline: {MODE === 'rotate' ? 'turn the line' : MODE === 'eigen' ? 'probe vector' : MODE === 'project' ? 'project and rebuild' : 'drop components'}"
      onSeek={(to) => { over = null; keepK = null; t = to; wake(); }} onStep={(to) => { over = null; keepK = null; t = to >= Tdur ? 0 : to; wake(); }} onReset={() => { over = null; keepK = null; t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if fR}
      <p class="vals num">line at <b>{f2(degOf(fR.theta), 1)}°</b> · variance of the shadows = <b class="fwd">{f2(fR.variance)}</b> · mean squared distance to the line = <b class="grad">{f2(fR.resid)}</b> · together {f2(fR.variance + fR.resid)} = total variance<br />
        <span class="base">PC1 is at {f2(degOf(model.th1), 1)}° with variance λ₁ = {f2(model.values[0])}; the perpendicular direction has λ₂ = {f2(model.values[1])}.</span></p>
    {:else if fE}
      <p class="vals num">C = [{f2(model.cov[0][0])} {f2(model.cov[0][1])} ; {f2(model.cov[1][0])} {f2(model.cov[1][1])}] · v = {fv(fE.v)} → C·v = <b class="fwd">{fv(fE.Cv)}</b> · angle between them <b>{f2(fE.off, 1)}°</b><br />
        {#each [0, 1] as i (i)}<span class="base">PC{i + 1}: λ = {f2(model.values[i])}, v = {fv(model.vectors[i])}: C·v = {fv([model.cov[0][0] * model.vectors[i][0] + model.cov[0][1] * model.vectors[i][1], model.cov[1][0] * model.vectors[i][0] + model.cov[1][1] * model.vectors[i][1]])}, λ·v = {fv(model.vectors[i].map((x) => model.values[i] * x))}{i ? '' : ' · '}</span>{/each}</p>
    {:else if fP}
      <p class="vals num">keep {comp === 0 ? 'PC1' : 'PC2'}: variance kept <b>{pct(one.kept)}</b> · mean squared error <b class="grad">{f2(one.mse, 3)}</b> (= the other eigenvalue) · 2 numbers per point become 1<br />
        <span class="base">ringed point {fv(model.pts[model.focus])} → z = {f2(focus.z)} → rebuilt {fv(focus.hat)}</span></p>
    {:else if fS}
      <p class="vals num">keep k = <b>{fS.k}</b> of {DIMS} · variance kept <b class="fwd">{pct(fS.varKept)}</b> · mean squared reconstruction error <b class="grad">{f2(fS.mse, 3)}</b> of total variance {f2(scree.total)}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if MODE === 'rotate'}
        <F1Slider label="angle" value={(fR?.theta ?? start) / DEG} min={Math.floor(start / DEG)} max={Math.ceil((start + Math.PI) / DEG)} step={0.5} digits={1} oninput={setAngle} />
        <span class="hint">You can also drag inside the scatter to aim the line.</span>
      {:else if MODE === 'eigen'}
        <F1Slider label="tilt" value={tilt} min={-60} max={60} step={1} digits={0} oninput={(v) => (tilt = v)} />
        <span class="hint">Turn the cloud and watch C and its eigenvectors follow.</span>
      {:else if MODE === 'project'}
        <span class="grp" role="group" aria-label="Component to keep">
          <button type="button" class="btn sm" aria-pressed={comp === 0} onclick={() => replay(0)}>Keep PC1</button>
          <button type="button" class="btn sm" aria-pressed={comp === 1} onclick={() => replay(1)}>Keep PC2</button>
        </span>
        <F1Slider label="tilt" value={tilt} min={-60} max={60} step={1} digits={0} oninput={(v) => (tilt = v)} />
      {:else}
        <F1Slider label="keep k" value={fS?.k ?? DIMS} min={1} max={DIMS} step={1} digits={0} oninput={(v) => { keepK = v; playing = false; wake(); }} />
        <span class="hint">Drag to choose how many components to keep; Play resumes the animation.</span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.6 var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 700; }
  .fwd { color: var(--viz-forward); }
  .grad { color: var(--viz-gradient); }
  .base { color: var(--text-muted); font-size: 0.8rem; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .grp { display: inline-flex; flex-wrap: wrap; gap: var(--s-2); }
  .sm { padding: 0.3em 0.7em; font-size: var(--step--1); }
  .sm[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); color: var(--text); font-weight: 600; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
