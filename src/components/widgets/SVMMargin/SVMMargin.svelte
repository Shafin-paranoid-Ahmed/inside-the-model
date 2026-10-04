<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour } from '../F1Shared/plot';
  import {
    DOM, GAMMAS, LIFT_Y, ID_R, T_MARGIN, T_SOFT, T_LIFT, T_KERNEL, T_ID,
    MARKERS_MARGIN, MARKERS_SOFT, MARKERS_LIFT, MARKERS_KERNEL, MARKERS_ID,
    blobs, fanLines, startLine, solveLinear, supportIdx, trainLinear, linearFrameAt, fmargin, marginWidth,
    trainLift, liftFrameAt, bestThreshold, kernelData, trainKernel, kernelFrameAt, kfield, identityFrameAt,
    type Mode, type KData, type Pt, type Cand, type LinRun, type LiftRun, type KernelRun,
    type LinFrame, type LiftFrame, type KernelFrame, type IdFrame,
  } from './sim';

  interface Props {
    title?: string;
    /** 'margin': fan of lines then the widest one. 'soft': C slider + hinge loss. 'lift': 1-D to 2-D. 'kernel': RBF with a gamma slider. 'identity': k(x, z) = phi(x) . phi(z). */
    mode?: Mode;
    /** Kernel mode only: the starting data. */
    dataset?: KData;
  }
  let { title = 'A linear SVM finding the widest street', mode = 'margin', dataset = 'rings' }: Props = $props();
  const cfg = untrack(() => ({ mode, dataset }));
  const m = cfg.mode;

  const CS = [0.1, 0.3, 1, 3, 10, 100];
  let svOnly = $state(false), ci = $state(2), gi = $state(2), kd = $state<KData>(cfg.dataset);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let lf = $state.raw<LinFrame | null>(null), kf = $state.raw<KernelFrame | null>(null), xf = $state.raw<IdFrame | null>(null), ff = $state.raw<LiftFrame | null>(null);
  let t = 0, wake = () => {}, dragged = false;
  let zo: [number, number] | null = null;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const dur = $derived(m === 'margin' ? (svOnly ? T_SOFT : T_MARGIN) : m === 'soft' ? T_SOFT : m === 'lift' ? T_LIFT : m === 'kernel' ? T_KERNEL : T_ID);
  const markers = $derived(m === 'margin' ? (svOnly ? MARKERS_SOFT : MARKERS_MARGIN) : m === 'soft' ? MARKERS_SOFT : m === 'lift' ? MARKERS_LIFT : m === 'kernel' ? MARKERS_KERNEL : MARKERS_ID);

  // model state (plain variables: the canvas reads them every frame)
  let pts: Pt[] = [], run: LinRun | null = null, cands: Cand[] | null = null, curve: Float64Array = new Float64Array(0);
  let all = $state.raw<Pt[]>([]), svCount = $state(0), lrun = $state.raw<LiftRun | null>(null), thr = $state(0), krun = $state.raw<KernelRun | null>(null);

  function build() {
    if (m === 'margin') {
      all = blobs(false);
      const fan = fanLines(all), sv = supportIdx(all, solveLinear(all, 100));
      svCount = sv.length;
      pts = svOnly ? sv.map((i) => all[i]) : all;
      cands = svOnly ? null : fan;
      run = trainLinear(pts, 100, startLine(fan));
    } else if (m === 'soft') {
      all = pts = blobs(true); cands = null;
      run = trainLinear(pts, CS[ci]);
    } else if (m === 'lift') {
      lrun = trainLift(); thr = bestThreshold(lrun.data);
    } else if (m === 'kernel') {
      all = pts = kernelData(kd);
      krun = trainKernel(pts, GAMMAS[gi]);
    }
    if (run) {
      const n = 140; curve = new Float64Array(n + 1);
      for (let i = 0; i <= n; i++) curve[i] = linearFrameAt((i / n) * dur, run, cands).width;
    }
  }
  build();

  const restart = () => { build(); t = 0; playing = true; wake(); };

  $effect(() => { playing; speed; wake(); });
  $effect(() => { if (playing && zo) { zo = null; wake(); } });

  const fmt = (v: number) => (Number.isFinite(v) ? f2(v, 2) : '∞');

  const narration = $derived.by(() => {
    if (m === 'margin' && lf) {
      if (svOnly) return lf.phase === 'settled' ? `Retrained on only the ${pts.length} support vectors: the margin is ${fmt(lf.width)}, the same as with all ${all.length} points.` : `The ${all.length - pts.length} other points are gone. Training starts again from the narrow line.`;
      if (lf.phase === 'many lines') return `Line ${lf.hi + 1} of ${cands?.length ?? 0} separates the classes with a margin of ${fmt(lf.width)}. Every line in the fan gets all the points right.`;
      if (lf.phase === 'widening') return 'Training pushes the narrowest line away from its nearest points. The street between the dashed lines widens.';
      return `Settled. The widest street has width ${fmt(lf.width)}, held in place by ${lf.nsv} support vectors (ringed).`;
    }
    if (m === 'soft' && lf) {
      if (lf.phase === 'start') return 'Start at w = 0: every point sits at margin 0 and pays a hinge loss of 1.';
      if (lf.phase === 'fitting') return 'The line tilts and the street narrows or widens until the regularizer and the hinge losses balance.';
      return `Settled with C = ${CS[ci]}: margin width ${fmt(lf.width)}, ${lf.viol} points inside or past the margin.`;
    }
    if (m === 'lift' && ff && lrun) {
      if (ff.phase === 'line') return `Class 1 sits in the middle of the line and class 0 on both sides. The best single cut gets only ${Math.round(thr * 100)}% right.`;
      if (ff.phase === 'lift') return 'Give every point a second coordinate, x squared. The middle points barely move; the outer points rise onto the parabola.';
      if (ff.phase === 'flat cut') return `In the lifted plane one straight line separates the classes. The SVM picks the widest one, with margin ${fmt(lrun.width)}.`;
      return `Where the flat cut meets the parabola, at x = ${lrun.roots.map((r) => f2(r)).join(' and x = ')}, it becomes two cuts on the original line.`;
    }
    if (m === 'kernel' && kf) {
      if (kf.phase === 'start') return 'No support vectors yet: every spot on the plane scores zero.';
      if (kf.phase === 'fitting') return 'Each round of training gives some points more weight (alpha). Their smooth bumps add up to the decision regions.';
      return `Settled with gamma = ${GAMMAS[gi]}: ${kf.nsv} support vectors, ${Math.round(kf.acc * 100)}% of the training points on the right side.`;
    }
    if (m === 'identity' && xf) return `Computing the kernel directly takes one dot product and a square. The lifted route needs three products and their sum. Both give ${f2(xf.direct, 4)}.`;
    return '';
  });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const N = 56, grid = new Float32Array(N * N), off = document.createElement('canvas');
    off.width = N; off.height = N;
    const octx = off.getContext('2d')!, img = octx.createImageData(N, N), px = new Uint32Array(img.data.buffer);
    let lastUi = 0, lastKey = "";
    const keyOf = () => { const a = cur.lf, k = cur.kf, x = cur.xf, l = cur.ff; return a ? `${a.width.toFixed(2)}|${a.nsv}|${a.viol}|${a.hi}|${a.phase}|${Math.round(a.acc * 100)}|${a.reg.toFixed(2)}|${a.hinge.toFixed(2)}` : k ? `${k.nsv}|${Math.round(k.acc * 100)}|${k.phase}` : x ? `${x.direct.toFixed(4)}|${x.z[0].toFixed(2)}|${x.z[1].toFixed(2)}|${x.phase}` : `${l?.phase}`; };
    const cur: { lf?: LinFrame; kf?: KernelFrame; ff?: LiftFrame; xf?: IdFrame } = {};
    const minus = (v: number) => String(v).replace('-', '−');

    const lay = () => {
      const hasSide = m !== 'lift';
      const { main, side, stacked } = hasSide ? splitPanels(stage.w, stage.h, 0.6) : { main: { x: 0, y: 0, w: stage.w, h: stage.h }, side: { x: 0, y: 0, w: 0, h: 0 }, stacked: false };
      if (m === 'lift') {
        const l = main.x + 34, r = main.x + main.w - 12, tp = main.y + 10, bt = main.y + main.h - 26;
        return { main, side, stacked, px0: l, py0: tp, s: r - l, sh: bt - tp, gx: scale(-DOM, DOM, l, r), gy: scale(-0.12, LIFT_Y, bt, tp), sd: { l: 0, r: 0, t: 0, b: 0 } };
      }
      const s = Math.max(40, Math.min(main.w - 40, main.h - 30));
      const px0 = main.x + 28 + (main.w - 28 - s) / 2, py0 = main.y + 6;
      const sd = { l: side.x + 44, r: side.x + side.w - 12, t: side.y + 34, b: side.y + side.h - 34 };
      return { main, side, stacked, px0, py0, s, sh: s, sd, gx: scale(-DOM, DOM, px0, px0 + s), gy: scale(-DOM, DOM, py0 + s, py0) };
    };
    type L = ReturnType<typeof lay>;

    const mark = (p: { x: number; y: number; c: 0 | 1 }, X: number, Y: number, a = 1) => {
      const { ctx, c } = stage;
      ctx.globalAlpha = a; ctx.fillStyle = p.c ? c.pos.hex : c.neg.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (p.c) ctx.arc(X, Y, 4.6, 0, 6.2832); else { ctx.moveTo(X, Y - 5.6); ctx.lineTo(X + 5.6, Y); ctx.lineTo(X, Y + 5.6); ctx.lineTo(X - 5.6, Y); ctx.closePath(); }
      ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
    };
    const ring = (X: number, Y: number, a = 1) => {
      const { ctx, c } = stage;
      ctx.globalAlpha = a; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X, Y, 9.5, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
    };
    const title2 = (txt: string, x: number, y: number) => {
      const { ctx, c } = stage;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(txt, x, y);
    };
    const axisBox = (x: number, y: number, w: number, h: number) => { const { ctx, c } = stage; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h); };

    /** Line w0 x + w1 y + b = off, drawn across the plane (clip does the rest). */
    const wline = (Lo: L, w: ArrayLike<number>, off: number) => {
      const n = Math.hypot(w[0], w[1]);
      if (n < 1e-9) return;
      const nx = w[0] / n, ny = w[1] / n, c0 = (off - w[2]) / n, dx = -ny, dy = nx, ox = nx * c0, oy = ny * c0, S = 8;
      const { ctx } = stage;
      ctx.beginPath(); ctx.moveTo(Lo.gx(ox - dx * S), Lo.gy(oy - dy * S)); ctx.lineTo(Lo.gx(ox + dx * S), Lo.gy(oy + dy * S)); ctx.stroke();
    };
    const band = (Lo: L, w: ArrayLike<number>) => {
      const n = Math.hypot(w[0], w[1]);
      if (n < 1e-9) return;
      const { ctx, c } = stage, nx = w[0] / n, ny = w[1] / n, dx = -ny, dy = nx, S = 8;
      const a = -w[2] / n - 1 / n, b = -w[2] / n + 1 / n;
      ctx.fillStyle = c.soft.hex; ctx.globalAlpha = 0.75; ctx.beginPath();
      ctx.moveTo(Lo.gx(nx * a - dx * S), Lo.gy(ny * a - dy * S)); ctx.lineTo(Lo.gx(nx * a + dx * S), Lo.gy(ny * a + dy * S));
      ctx.lineTo(Lo.gx(nx * b + dx * S), Lo.gy(ny * b + dy * S)); ctx.lineTo(Lo.gx(nx * b - dx * S), Lo.gy(ny * b - dy * S));
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    };
    const planeTicks = (Lo: L) => {
      const { ctx, c } = stage;
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-1, 0, 1]) ctx.fillText(minus(v), Lo.gx(v), Lo.py0 + Lo.s + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [-1, 0, 1]) ctx.fillText(minus(v), Lo.px0 - 5, Lo.gy(v));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('x₁', Lo.px0 + Lo.s, Lo.main.y + Lo.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('x₂', Lo.px0 - 26, Lo.py0 - 4);
    };

    function drawLinear(Lo: L, f: LinFrame) {
      const { ctx, c } = stage;
      axisBox(Lo.px0, Lo.py0, Lo.s, Lo.s); planeTicks(Lo);
      ctx.save(); ctx.beginPath(); ctx.rect(Lo.px0, Lo.py0, Lo.s, Lo.s); ctx.clip();
      band(Lo, f.w);
      if (cands && f.fan > 0) {
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.globalAlpha = 0.5 * f.fan;
        for (const cd of cands) wline(Lo, cd.w, 0);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      wline(Lo, f.w, 1); wline(Lo, f.w, -1); ctx.setLineDash([]);
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.6; wline(Lo, f.w, 0);
      const svs = new Set(supportIdx(pts, f.w));
      if (svOnly) for (const p of all) if (!pts.includes(p)) mark(p, Lo.gx(p.x), Lo.gy(p.y), 0.28);
      pts.forEach((p, i) => { mark(p, Lo.gx(p.x), Lo.gy(p.y)); if (svs.has(i)) ring(Lo.gx(p.x), Lo.gy(p.y)); });
      ctx.restore();
    }

    function drawWidthPanel(Lo: L, f: LinFrame) {
      const { ctx, c } = stage, sd = Lo.sd;
      let mx = 0.1;
      for (const v of curve) if (Number.isFinite(v)) mx = Math.max(mx, v);
      mx = Math.ceil(mx * 10.5) / 10;
      const sx = scale(0, dur, sd.l, sd.r), sy = scale(0, mx, sd.b, sd.t);
      title2('Margin width (street width)', sd.l - 34, Lo.side.y + 4);
      axisBox(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const step = mx > 0.8 ? 0.5 : 0.2;
      for (let v = 0; v < mx; v += step) ctx.fillText(f2(v, 1), sd.l - 6, sy(v));
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let v = 0; v <= dur; v += 2) ctx.fillText(String(v), sx(v), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('time (s)', sd.r, Lo.side.y + Lo.side.h - 1);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      const n = curve.length - 1, upto = Math.min(n, Math.floor((t / dur) * n));
      for (let i = 0; i <= upto; i++) { const X = sx((i / n) * dur), Y = sy(Math.min(mx, curve[i])); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.lineTo(sx(t), sy(Math.min(mx, f.width))); ctx.stroke();
      ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx(t), sy(Math.min(mx, f.width)), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    function drawHingePanel(Lo: L, f: LinFrame) {
      const { ctx, c } = stage, sd = Lo.sd, lo = -2.5, hiM = 3.5, ymax = 3.5;
      const sx = scale(lo, hiM, sd.l, sd.r), sy = scale(0, ymax, sd.b, sd.t);
      title2('Hinge loss of each point', sd.l - 34, Lo.side.y + 4);
      axisBox(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 1, 2, 3]) ctx.fillText(String(v), sd.l - 6, sy(v));
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-2, -1, 0, 1, 2, 3]) ctx.fillText(minus(v), sx(v), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('margin of the point, y·f(x)', sd.r, Lo.side.y + Lo.side.h - 1);
      ctx.strokeStyle = c.rule.hex; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(sx(1), sd.t); ctx.lineTo(sx(1), sd.b); ctx.moveTo(sx(0), sd.t); ctx.lineTo(sx(0), sd.b); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(sx(lo), sy(1 - lo)); ctx.lineTo(sx(1), sy(0)); ctx.lineTo(sx(hiM), sy(0)); ctx.stroke();
      for (const p of pts) {
        const mg = fmargin(f.w, p), X = sx(Math.max(lo, Math.min(hiM, mg))), Y = sy(Math.max(0, 1 - mg));
        ctx.globalAlpha = 0.85; ctx.fillStyle = p.c ? c.pos.hex : c.neg.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(X, Y, 3.8, 0, 6.2832); ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
      }
    }

    function drawLift(Lo: L, f: LiftFrame) {
      const { ctx, c } = stage, r = lrun!, X0 = Lo.px0, X1 = Lo.px0 + Lo.s;
      axisBox(X0, Lo.py0, Lo.s, Lo.sh);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-1, 0, 1]) ctx.fillText(minus(v), Lo.gx(v), Lo.py0 + Lo.sh + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 1, 2]) ctx.fillText(String(v), X0 - 5, Lo.gy(v));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', X1, Lo.main.y + Lo.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('x²', X0 + 4, Lo.py0 + 3);
      ctx.save(); ctx.beginPath(); ctx.rect(X0, Lo.py0, Lo.s, Lo.sh); ctx.clip();
      // the number line
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X0, Lo.gy(0)); ctx.lineTo(X1, Lo.gy(0)); ctx.stroke();
      // regions on the line, after the cut has come back down
      if (f.back > 0) {
        const n = 160;
        for (let i = 0; i < n; i++) {
          const x = -DOM + ((i + 0.5) / n) * 2 * DOM, z = r.w[0] * x + r.w[1] * x * x + r.w[2];
          ctx.globalAlpha = 0.5 * f.back; ctx.fillStyle = z > 0 ? c.pos.hex : c.neg.hex;
          ctx.fillRect(Lo.gx(x - DOM / n), Lo.gy(0) - 5, (Lo.s / n) + 0.6, 10);
        }
        ctx.globalAlpha = 1;
      }
      // parabola guide
      if (f.s > 0) {
        ctx.strokeStyle = c.muted.hex; ctx.globalAlpha = 0.6 * f.s; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath();
        for (let i = 0; i <= 80; i++) { const x = -DOM + (i / 80) * 2 * DOM; if (i === 0) ctx.moveTo(Lo.gx(x), Lo.gy(f.s * x * x)); else ctx.lineTo(Lo.gx(x), Lo.gy(f.s * x * x)); }
        ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      // the flat cut and its margin, in the lifted plane: w0 x + w1 u + b = off  ->  u = (off - b - w0 x) / w1
      if (f.cut > 0 && Math.abs(r.w[1]) > 1e-9) {
        const uAt = (x: number, off: number) => (off - r.w[2] - r.w[0] * x) / r.w[1];
        const seg = (off: number) => { ctx.beginPath(); ctx.moveTo(X0, Lo.gy(uAt(-DOM, off))); ctx.lineTo(X1, Lo.gy(uAt(DOM, off))); ctx.stroke(); };
        ctx.globalAlpha = f.cut;
        ctx.fillStyle = c.soft.hex; ctx.beginPath(); ctx.moveTo(X0, Lo.gy(uAt(-DOM, 1))); ctx.lineTo(X1, Lo.gy(uAt(DOM, 1))); ctx.lineTo(X1, Lo.gy(uAt(DOM, -1))); ctx.lineTo(X0, Lo.gy(uAt(-DOM, -1))); ctx.closePath(); ctx.globalAlpha = 0.75 * f.cut; ctx.fill(); ctx.globalAlpha = f.cut;
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); seg(1); seg(-1); ctx.setLineDash([]);
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.6; seg(0);
        ctx.globalAlpha = 1;
        // where the cut meets the parabola: slide the marks down to the number line
        if (f.back > 0) for (const x of r.roots) {
          const y0 = x * x, yy = y0 * (1 - f.back);
          ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(Lo.gx(x), Lo.gy(y0)); ctx.lineTo(Lo.gx(x), Lo.gy(0)); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(Lo.gx(x), Lo.gy(yy), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        }
      }
      const svs = f.cut > 0 ? new Set(supportIdx(r.pts, r.w)) : null;
      r.pts.forEach((p, i) => {
        const X = Lo.gx(p.x), Y = Lo.gy(f.s * p.x * p.x);
        mark(p, X, Y);
        if (svs?.has(i)) ring(X, Y, f.cut);
      });
      ctx.restore();
    }

    function drawKernel(Lo: L, f: KernelFrame) {
      const { ctx, c } = stage, r = krun!;
      kfield(r, f.a, N, grid);
      const [pr, pg, pb] = c.pos.rgb, [nr, ng, nb] = c.neg.rgb, [br, bg, bb] = c.raised.rgb;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const v = grid[j * N + i], a = Math.min(0.6, Math.abs(Math.tanh(v)) * 0.75), [rr, gg, b2] = v > 0 ? [pr, pg, pb] : [nr, ng, nb];
        px[(N - 1 - j) * N + i] = 0xff000000 | (Math.round(bb + (b2 - bb) * a) << 16) | (Math.round(bg + (gg - bg) * a) << 8) | Math.round(br + (rr - br) * a);
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(off, Lo.px0, Lo.py0, Lo.s, Lo.s);
      axisBox(Lo.px0, Lo.py0, Lo.s, Lo.s); planeTicks(Lo);
      ctx.save(); ctx.beginPath(); ctx.rect(Lo.px0, Lo.py0, Lo.s, Lo.s); ctx.clip();
      const cell = Lo.s / N, at = (i: number, j: number) => grid[j * N + i];
      const stroke = (level: number, width: number, color: string, dash: number[], alpha: number) => {
        const seg = contour(at, N, N, level);
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.globalAlpha = alpha; ctx.beginPath();
        for (let k = 0; k < seg.length; k += 4) {
          ctx.moveTo(Lo.px0 + (seg[k] + 0.5) * cell, Lo.py0 + Lo.s - (seg[k + 1] + 0.5) * cell);
          ctx.lineTo(Lo.px0 + (seg[k + 2] + 0.5) * cell, Lo.py0 + Lo.s - (seg[k + 3] + 0.5) * cell);
        }
        ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      };
      stroke(1, 1, c.muted.hex, [3, 4], 0.8); stroke(-1, 1, c.muted.hex, [3, 4], 0.8);
      stroke(0, 2.6, c.text.hex, [], 1);
      r.pts.forEach((p, i) => { mark(p, Lo.gx(p.x), Lo.gy(p.y)); if (f.a[i] > 1e-6) ring(Lo.gx(p.x), Lo.gy(p.y)); });
      ctx.restore();
    }

    function drawBump(Lo: L) {
      const { ctx, c } = stage, sd = Lo.sd, g = GAMMAS[gi], dmax = 2;
      const sx = scale(0, dmax, sd.l, sd.r), sy = scale(0, 1, sd.b, sd.t);
      title2('How far one point reaches', sd.l - 34, Lo.side.y + 4);
      axisBox(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.5, 1]) ctx.fillText(f2(v, 1), sd.l - 6, sy(v));
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 1, 2]) ctx.fillText(String(v), sx(v), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('distance d between two points', sd.r, Lo.side.y + Lo.side.h - 1);
      const half = Math.sqrt(Math.LN2 / g);
      ctx.strokeStyle = c.rule.hex; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(sx(half), sd.t); ctx.lineTo(sx(half), sd.b); ctx.moveTo(sd.l, sy(0.5)); ctx.lineTo(sd.r, sy(0.5)); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let i = 0; i <= 100; i++) { const d = (i / 100) * dmax, X = sx(d), Y = sy(Math.exp(-g * d * d)); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.stroke();
      ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx(Math.min(dmax, half)), sy(0.5), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    function drawIdentity(Lo: L, f: IdFrame) {
      const { ctx, c } = stage;
      axisBox(Lo.px0, Lo.py0, Lo.s, Lo.s); planeTicks(Lo);
      ctx.save(); ctx.beginPath(); ctx.rect(Lo.px0, Lo.py0, Lo.s, Lo.s); ctx.clip();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(Lo.gx(-DOM), Lo.gy(0)); ctx.lineTo(Lo.gx(DOM), Lo.gy(0)); ctx.moveTo(Lo.gx(0), Lo.gy(-DOM)); ctx.lineTo(Lo.gx(0), Lo.gy(DOM)); ctx.stroke();
      const rr = Lo.gx(ID_R) - Lo.gx(0);
      ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(Lo.gx(0), Lo.gy(0), rr, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
      const arrow = (v: readonly number[], col: string, label: string) => {
        const X = Lo.gx(v[0]), Y = Lo.gy(v[1]), X0 = Lo.gx(0), Y0 = Lo.gy(0), a = Math.atan2(Y - Y0, X - X0);
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X, Y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X - 10 * Math.cos(a - 0.4), Y - 10 * Math.sin(a - 0.4)); ctx.lineTo(X - 10 * Math.cos(a + 0.4), Y - 10 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
        ctx.font = stage.body(14, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, X + 14 * Math.cos(a), Y + 14 * Math.sin(a));
      };
      arrow(f.x, c.fwd.hex, 'x'); arrow(f.z, c.grad.hex, 'z');
      ctx.restore();
    }

    function drawBars(Lo: L, f: IdFrame) {
      const { ctx, c } = stage, sd = Lo.sd, rows = [['x₁²', 0], ['√2·x₁x₂', 1], ['x₂²', 2]] as const;
      title2('Lifted features φ = (x₁², √2·x₁x₂, x₂²)', sd.l - 34, Lo.side.y + 4);
      const mid = (sd.l + sd.r) / 2 + 8, half = (sd.r - sd.l) / 2 - 22, sc = half / 1.05, gh = (sd.b - sd.t - 34) / 3;
      const series = [['φ(x)', f.px, c.fwd.hex], ['φ(z)', f.pz, c.grad.hex], ['product', f.prod, c.text.hex]] as const;
      ctx.font = stage.mono(10.5);
      rows.forEach(([name, i], g) => {
        const y0 = sd.t + g * gh;
        ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(name, sd.l - 30, y0);
        series.forEach(([, vec, col], s) => {
          const v = vec[i], y = y0 + 14 + s * 11;
          ctx.fillStyle = col; ctx.globalAlpha = s === 2 ? 1 : 0.75;
          ctx.fillRect(Math.min(mid, mid + v * sc), y, Math.abs(v * sc), 8); ctx.globalAlpha = 1;
        });
      });
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(mid, sd.t + 12); ctx.lineTo(mid, sd.b - 36); ctx.stroke();
      // key
      ctx.textBaseline = 'middle'; let kx = sd.l - 30; const ky = sd.b - 18;
      series.forEach(([nm, , col]) => {
        ctx.fillStyle = col; ctx.fillRect(kx, ky - 4, 9, 8); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.fillText(nm, kx + 13, ky);
        kx += 22 + nm.length * 6.4;
      });
    }

    function render() {
      const { ctx, w: W, h: H } = stage, Lo = lay();
      ctx.clearRect(0, 0, W, H);
      if ((m === 'margin' || m === 'soft') && run) {
        const f = linearFrameAt(t, run, cands); cur.lf = f;
        drawLinear(Lo, f);
        if (m === 'margin') drawWidthPanel(Lo, f); else drawHingePanel(Lo, f);
      } else if (m === 'lift') { const f = liftFrameAt(t); cur.ff = f; drawLift(Lo, f); }
      else if (m === 'kernel' && krun) { const f = kernelFrameAt(t, krun); cur.kf = f; drawKernel(Lo, f); drawBump(Lo); }
      else if (m === 'identity') { const f = identityFrameAt(t, zo); cur.xf = f; drawIdentity(Lo, f); drawBars(Lo, f); }
    }
    const read = () => { if (cur.lf) { lf = cur.lf; } if (cur.kf) kf = cur.kf; if (cur.ff) ff = cur.ff; if (cur.xf) xf = cur.xf; lastKey = keyOf(); };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: dur }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing || keyOf() !== lastKey) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const toData = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), Lo = lay(), x = e.clientX - r.left, y = e.clientY - r.top;
      const dx = -DOM + ((x - Lo.px0) / Lo.s) * 2 * DOM, dy = DOM - ((y - Lo.py0) / Lo.s) * 2 * DOM;
      return Math.abs(dx) <= DOM && Math.abs(dy) <= DOM ? ([dx, dy] as [number, number]) : null;
    };
    const place = (e: PointerEvent) => {
      const p = toData(e);
      if (!p) return;
      const n = Math.hypot(p[0], p[1]) || 1, k = Math.min(1, ID_R * 1.3 / n);
      zo = [p[0] * k, p[1] * k]; playing = false; wake();
    };
    const down = (e: PointerEvent) => { if (m !== 'identity') return; dragged = true; cv.setPointerCapture(e.pointerId); place(e); };
    const move = (e: PointerEvent) => { if (m === 'identity' && dragged) place(e); };
    const up = () => { dragged = false; };
    if (m === 'identity') cv.style.cursor = 'crosshair';
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });

  const labelOf = {
    margin: 'Left: two classes of points on a plane with a solid boundary line and two dashed margin lines, and the support vectors ringed. Right: the margin width growing over time.',
    soft: 'Left: two overlapping classes with a boundary, dashed margin lines and ringed support vectors. Right: the hinge loss curve with every point placed on it.',
    lift: 'Points on a number line, lifted onto the parabola x squared, where one flat line separates the two classes.',
    kernel: 'Left: a plane coloured by an RBF-kernel SVM, with the decision boundary as a black curve and support vectors ringed. Right: how the kernel value falls with distance.',
    identity: 'Left: two vectors x and z on a plane. Right: bars for the lifted features of x, of z and their products, which sum to the same number as the squared dot product.',
  } as const;
  const legends = {
    margin: [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'boundary', color: '--text', shape: 'line' as const }, { label: 'support vector', color: '--accent' }, { label: 'width', color: '--viz-forward', shape: 'line' as const }],
    soft: [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'boundary', color: '--text', shape: 'line' as const }, { label: 'support vector', color: '--accent' }, { label: 'hinge loss', color: '--viz-gradient', shape: 'line' as const }],
    lift: [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'flat cut', color: '--text', shape: 'line' as const }, { label: 'support vector', color: '--accent' }],
    kernel: [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'boundary', color: '--text', shape: 'line' as const }, { label: 'support vector', color: '--accent' }, { label: 'kernel value', color: '--viz-forward', shape: 'line' as const }],
    identity: [{ label: 'x', color: '--viz-forward', shape: 'line' as const }, { label: 'z', color: '--viz-gradient', shape: 'line' as const }],
  };
</script>

<FigureCard {title} role="group" aspect="720 / 400" narrowAspect={m === 'lift' ? '360 / 400' : '360 / 640'} label={labelOf[m]} legend={legends[m]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={dur} {markers} label="Timeline: SVM figure"
      onSeek={(to) => { t = to; if (m === 'identity') zo = null; wake(); }}
      onStep={(to) => { t = to >= dur ? 0 : to; wake(); }}
      onReset={() => { t = 0; zo = null; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if (m === 'margin' || m === 'soft') && lf}
        margin width <b>{fmt(lf.width)}</b> · support vectors <b>{lf.nsv}</b> · inside or past the margin <b>{lf.viol}</b> · accuracy <b>{Math.round(lf.acc * 100)}%</b>
        {#if m === 'soft'}<span class="base">· ½‖w‖² = {f2(lf.reg, 2)}, C·Σ hinge = {f2(CS[ci] * lf.hinge, 2)} (C = {CS[ci]})</span>{/if}
      {:else if m === 'lift' && ff && lrun}
        single cut on the line <b>{Math.round(thr * 100)}%</b> · lifted margin <b>{f2(lrun.width, 2)}</b> · cut points <b>{lrun.roots.map((r) => f2(r)).join(', ')}</b>
      {:else if m === 'kernel' && kf}
        γ <b>{GAMMAS[gi]}</b> · support vectors <b>{kf.nsv}</b> of {krun?.pts.length} · training accuracy <b>{Math.round(kf.acc * 100)}%</b> · half-reach <b>{f2(Math.sqrt(Math.LN2 / GAMMAS[gi]), 2)}</b>
      {:else if m === 'identity' && xf}
        x = ({f2(xf.x[0])}, {f2(xf.x[1])}) · z = ({f2(xf.z[0])}, {f2(xf.z[1])}) · k(x, z) = (x·z)² = <b>{f2(xf.direct, 4)}</b> · φ(x)·φ(z) = <b>{f2(xf.lifted, 4)}</b>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if m === 'margin'}
      <div class="ctl">
        <button type="button" class="btn" aria-pressed={svOnly} onclick={() => { svOnly = !svOnly; restart(); }}>{svOnly ? 'Bring back all points' : 'Keep only the support vectors'}</button>
        <span class="hint">{svOnly ? `Only ${svCount} of ${all.length} points remain. The boundary and the margin do not change.` : `${lf?.nsv ?? svCount} of ${all.length} points sit on the margin at this moment; ${svCount} at the widest street.`}</span>
      </div>
    {:else if m === 'soft'}
      <div class="ctl">
        <label>Penalty C
          <input type="range" min="0" max={CS.length - 1} step="1" value={ci} aria-valuetext="C = {CS[ci]}" oninput={(e) => { ci = +e.currentTarget.value; restart(); }} />
          <span class="num val">{CS[ci]}</span>
        </label>
        <span class="hint">Small C: cheap violations, wide street. Large C: violations are costly, narrow street.</span>
      </div>
    {:else if m === 'kernel'}
      <div class="ctl">
        <label>γ (sharpness)
          <input type="range" min="0" max={GAMMAS.length - 1} step="1" value={gi} aria-valuetext="gamma = {GAMMAS[gi]}" oninput={(e) => { gi = +e.currentTarget.value; restart(); }} />
          <span class="num val">{GAMMAS[gi]}</span>
        </label>
        <label>Data
          <select value={kd} onchange={(e) => { kd = e.currentTarget.value as KData; restart(); }}>
            <option value="rings">Rings</option><option value="xor">XOR</option>
          </select>
        </label>
        <span class="hint">Each set has two mislabelled points.</span>
      </div>
    {:else if m === 'identity'}
      <div class="ctl"><span class="hint">Drag on the plane to place z yourself. Press Play to let it circle x again.</span></div>
    {:else}
      <div class="ctl"><span class="hint">Scrub the timeline to lift the points by hand.</span></div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: 0.6em; font: var(--step--1) var(--font-ui); }
  .ctl input[type='range'] { width: 11rem; accent-color: var(--accent); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .val { min-width: 3ch; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
