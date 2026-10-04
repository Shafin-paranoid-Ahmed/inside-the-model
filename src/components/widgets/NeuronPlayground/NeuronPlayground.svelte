<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 as fmt } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { POINTS, DOM, ACTS, Z_RANGE, FINAL, PROBE, activate, shade, accuracy, duration, markers, frameAt, type Mode, type Act, type Params, type Frame } from './sim';

  interface Props {
    title?: string;
    /** 'weighted-sum': a test point circles the plane and the diagram adds up its products. 'dot-product': w rotates, the boundary turns with it. 'activation': none, step and sigmoid take turns. 'playground': w1, w2, b move through a script and you can grab them. */
    mode?: Mode;
  }
  let { title = '', mode = 'weighted-sum' }: Props = $props();
  const init = untrack(() => ({ mode, title: title || 'One neuron with two inputs' }));
  const MK = markers(init.mode), T = duration(init.mode);
  const isAct = init.mode === 'activation';

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let over = $state<Params | null>(null);
  let userAct = $state<Act>('sigmoid');
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let values: HTMLParagraphElement;
  const cur = (): Frame => frameAt(t, init.mode, over ?? undefined, userAct);
  const snap = () => { const f = cur(); return { f, acc: Math.round(accuracy(f) * POINTS.length) }; };
  let ui = $state.raw(snap());

  const label = $derived(over ? 'manual'
    : (init.mode === 'weighted-sum' || init.mode === 'dot-product') && Math.abs(ui.f.z) < 1e-8 ? 'boundary'
    : (markerAt(uiT, MK)?.label ?? ''));
  const narration = $derived(
    label === 'manual' ? 'You set the weights and the bias by hand. Press play to run the script again from here.'
      : label === 'boundary' ? 'The score is zero: the input sits on the decision boundary. In the dot-product view the two vectors are perpendicular here.'
      : label === 'negative' && init.mode === 'weighted-sum' ? 'Each input is multiplied by its weight, the products are added, and the bias is added on top. The sum is below zero here, so the neuron leans to the orange class.'
      : label === 'positive' && init.mode === 'weighted-sum' ? 'The sum is above zero: the neuron leans to the blue class. The point has crossed the boundary, the line where the sum is exactly zero.'
      : label === 'positive' && init.mode === 'dot-product' ? 'The weight vector and the input point roughly the same way, so the dot product is positive and the neuron leans to the blue class.'
      : label === 'negative' ? 'The weight vector points away from the input, so the dot product is negative. The input is on the orange side of the boundary.'
      : label === 'no activation' ? 'No activation: the output is the sum z itself. It can be any size; colour intensity shows its magnitude up to the display scale limit.'
      : label === 'step' ? 'A step activation outputs 0 or 1 and nothing in between: the colour jumps at the boundary and the output flips as the point crosses it.'
      : label === 'sigmoid' ? 'A sigmoid is a smooth step: the output moves gently through 0.5 at the boundary and approaches 0 or 1 far from it.'
      : label === 'start' ? 'Start: the boundary cuts through the wrong region, so many points get the wrong colour.'
      : label === 'turn w' ? 'The weights change. The boundary turns, because it is always perpendicular to the weight vector.'
      : label === 'slide b' ? 'The bias changes. The boundary keeps its direction and slides, because the bias only shifts the sum.'
      : label === 'separated' ? 'Every point sits on the correct side of the boundary. Now drag the sliders and watch the accuracy fall.'
      : '',
  );
  $effect(() => { if (playing) over = null; speed; userAct; wake(); });

  const f2 = (v: number, d = 2) => fmt(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);
  const sg = (v: number) => (v < 0 ? ' − ' : ' + ') + f2(Math.abs(v));
  const deg = (r: number) => Math.round((r * 180) / Math.PI);
  const lr = (f: Frame) => Math.hypot(f.x1, f.x2);
  function valueText(f: Frame, acc: number): string {
    if (init.mode === 'weighted-sum') return `z = w₁x₁ + w₂x₂ + b ≈ ${f2(f.p1)}${sg(f.p2)}${sg(f.b)} ≈ ${f2(f.z)}\nx = (${f2(f.x1)}, ${f2(f.x2)}), w = (${f2(f.w1)}, ${f2(f.w2)}) · displayed terms rounded independently`;
    if (init.mode === 'dot-product') return `w·x = |w| |x| cos θ ≈ ${f2(Math.hypot(f.w1, f.w2))} × ${f2(lr(f))} × ${f2(Math.cos(f.theta))} ≈ ${f2(f.z)}\nθ = ${deg(f.theta)}° · w₁x₁ + w₂x₂ ≈ ${f2(f.p1)}${sg(f.p2)}`;
    if (init.mode === 'activation') return `activation = ${f.act} · z = ${f2(f.z)} → output = ${f2(f.out)}\n${f.act === 'none' ? 'output = z' : f.act === 'step' ? 'output = 1 if z > 0, else 0' : 'output = 1 / (1 + e⁻ᶻ)'}`;
    return `w₁ = ${f2(f.w1)} · w₂ = ${f2(f.w2)} · b = ${f2(f.b)} · accuracy = ${acc}/${POINTS.length}\ntest point (${PROBE.x1}, ${PROBE.x2}): z = ${f2(f.z)}, output = ${f2(f.out)}`;
  }
  const initialValues = untrack(() => valueText(ui.f, ui.acc));

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const off = document.createElement('canvas'), N = 110;
    off.width = off.height = N;
    const octx = off.getContext('2d')!, img = octx.createImageData(N, N);

    const lay = () => {
      const W = stage.w, H = stage.h, wide = W >= 540;
      let dia, plot = null as null | { x: number; y: number; w: number; h: number }, pl;
      if (wide) {
        const lw = Math.round(W * 0.46), dh = isAct ? Math.round(H * 0.6) : H;
        dia = { x: 0, y: 0, w: lw, h: dh };
        if (isAct) plot = { x: 0, y: dh + 6, w: lw, h: H - dh - 6 };
        pl = { x: lw + 8, y: 0, w: W - lw - 8, h: H };
      } else {
        const dh = 200, ph = isAct ? 150 : 0;
        dia = { x: 0, y: 0, w: W, h: dh };
        pl = { x: 0, y: dh + 6, w: W, h: H - dh - 6 - (ph ? ph + 6 : 0) };
        if (isAct) plot = { x: 0, y: H - ph, w: W, h: ph };
      }
      const side = Math.max(60, Math.min(pl.w - 44, pl.h - 36)), px0 = pl.x + 34 + (pl.w - 44 - side) / 2, py0 = pl.y + 8;
      return { wide, dia, plot, pl, side, px0, py0, X: scale(-DOM, DOM, px0, px0 + side), Y: scale(-DOM, DOM, py0 + side, py0) };
    };

    const arrow = (c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, head = 8) => {
      const a = Math.atan2(y1 - y0, x1 - x0);
      c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4)); c.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4)); c.closePath(); c.fill();
    };
    const actGlyph = (c: CanvasRenderingContext2D, a: Act, cx: number, cy: number, w: number, h: number) => {
      c.beginPath();
      if (a === 'none') { c.moveTo(cx - w, cy + h); c.lineTo(cx + w, cy - h); }
      else if (a === 'step') { c.moveTo(cx - w, cy + h); c.lineTo(cx, cy + h); c.lineTo(cx, cy - h); c.lineTo(cx + w, cy - h); }
      else for (let i = 0; i <= 20; i++) { const u = i / 20, x = cx + w * (2 * u - 1), y = cy + h - 2 * h / (1 + Math.exp(-(12 * u - 6))); if (i) c.lineTo(x, y); else c.moveTo(x, y); }
      c.stroke();
    };

    function drawDiagram(f: Frame, B: { x: number; y: number; w: number; h: number }, flow: number) {
      const { ctx: c, c: col } = stage;
      const cx = B.y + B.h / 2, spread = Math.min(B.h * 0.27, 78);
      const ix = B.x + 24, y1 = cx - spread + 8, y2 = cx + spread + 8, ym = cx + 8;
      const sx = B.x + B.w * 0.55, ax = B.x + B.w * 0.76, ox = B.x + B.w - 24, R = 15, RS = 17;
      c.font = stage.mono(11); c.textBaseline = 'middle'; c.lineCap = 'round';
      const wEdge = (y: number, w: number, prod: number, lab: string, up: number) => {
        const x0 = ix + R, x1 = sx - RS, ys = y, ye = ym;
        const colr = w < 0 ? col.neg : col.pos;
        c.strokeStyle = colr.hex; c.lineWidth = 1 + 5 * Math.min(1, Math.abs(w) / 3); c.beginPath(); c.moveTo(x0, ys); c.lineTo(x1, ye); c.stroke();
        if (flow >= 0) { c.fillStyle = col.fwd.hex; for (let k = 0; k < 3; k++) { const u = (flow * 0.4 + k / 3) % 1; c.beginPath(); c.arc(x0 + (x1 - x0) * u, ys + (ye - ys) * u, 2.4, 0, 6.2832); c.fill(); } }
        c.textAlign = 'center'; c.fillStyle = colr.hex;
        const at = (u: number) => [x0 + (x1 - x0) * u, ys + (ye - ys) * u + up * 16];
        c.fillText(lab + ' ' + f2(w), ...(at(0.3) as [number, number]));
        // Narrow diagrams need the product farther from the sum's centered z label.
        c.fillStyle = col.text.hex; c.fillText(f2(prod), ...(at(stage.w < 540 ? 0.58 : 0.78) as [number, number]));
      };
      wEdge(y1, f.w1, f.p1, 'w₁', -1);
      wEdge(y2, f.w2, f.p2, 'w₂', 1);
      // inputs
      for (const [y, v, n] of [[y1, f.x1, 'x₁'], [y2, f.x2, 'x₂']] as const) {
        c.fillStyle = col.raised.hex; c.strokeStyle = col.text.hex; c.lineWidth = 1.5; c.beginPath(); c.arc(ix, y, R, 0, 6.2832); c.fill(); c.stroke();
        c.fillStyle = col.text.hex; c.textAlign = 'center'; c.fillText(n, ix, y);
        c.fillStyle = col.muted.hex; c.fillText(f2(v), ix, y + (y === y1 ? -R - 8 : R + 9));
      }
      // bias
      const by = Math.max(B.y + 12, ym - 62);
      c.strokeStyle = (f.b < 0 ? col.neg : col.pos).hex; c.lineWidth = 1 + 5 * Math.min(1, Math.abs(f.b) / 3);
      c.beginPath(); c.moveTo(sx, by + 9); c.lineTo(sx, ym - RS); c.stroke();
      c.fillStyle = (f.b < 0 ? col.neg : col.pos).hex; c.textAlign = 'center'; c.fillText('b ' + f2(f.b), sx, by);
      // sum
      c.fillStyle = col.raised.hex; c.strokeStyle = col.text.hex; c.lineWidth = 1.5; c.beginPath(); c.arc(sx, ym, RS, 0, 6.2832); c.fill(); c.stroke();
      c.font = stage.body(16); c.fillStyle = col.text.hex; c.fillText('Σ', sx, ym + 1);
      c.font = stage.mono(11, 650); c.fillStyle = col.fwd.hex; c.fillText('z ' + f2(f.z), sx, ym + RS + 12);
      // activation
      c.strokeStyle = col.rule.hex; c.lineWidth = 2; c.beginPath(); c.moveTo(sx + RS, ym); c.lineTo(ax - 20, ym); c.moveTo(ax + 20, ym); c.lineTo(ox - 20, ym); c.stroke();
      if (flow >= 0) { c.fillStyle = col.fwd.hex; for (const [a, b] of [[sx + RS, ax - 20], [ax + 20, ox - 20]]) { const u = (flow * 0.4) % 1; c.beginPath(); c.arc(a + (b - a) * u, ym, 2.4, 0, 6.2832); c.fill(); } }
      c.fillStyle = col.raised.hex; c.strokeStyle = col.text.hex; c.lineWidth = 1.5; c.beginPath(); c.roundRect(ax - 20, ym - 17, 40, 34, 5); c.fill(); c.stroke();
      c.strokeStyle = col.fwd.hex; c.lineWidth = 2; actGlyph(c, f.act, ax, ym, 11, 9);
      c.font = stage.mono(10.5); c.fillStyle = col.muted.hex; c.fillText(f.act, ax, ym + 30);
      // output
      c.fillStyle = col.raised.hex; c.strokeStyle = col.fwd.hex; c.lineWidth = 2; c.beginPath(); c.arc(ox, ym, 20, 0, 6.2832); c.fill(); c.stroke();
      c.font = stage.mono(11, 650); c.fillStyle = col.fwd.hex; c.fillText(f2(f.out), ox, ym);
      c.font = stage.mono(10.5); c.fillStyle = col.muted.hex; c.fillText('output', ox, ym + 32);
    }

    function drawPlane(f: Frame, L: ReturnType<typeof lay>) {
      const { ctx: c, c: col } = stage, { X, Y, px0, py0, side } = L;
      // field: colour = which class the neuron leans to, depth = how sure
      const pos = col.pos.rgb, neg = col.neg.rgb, bg = col.raised.rgb, d = img.data;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const x1 = -DOM + ((i + 0.5) / N) * 2 * DOM, x2 = DOM - ((j + 0.5) / N) * 2 * DOM, z = f.w1 * x1 + f.w2 * x2 + f.b;
        const sv = shade(f.act, z), a = 0.5 * Math.abs(sv), cc = sv >= 0 ? pos : neg, k = (j * N + i) * 4;
        d[k] = bg[0] + (cc[0] - bg[0]) * a; d[k + 1] = bg[1] + (cc[1] - bg[1]) * a; d[k + 2] = bg[2] + (cc[2] - bg[2]) * a; d[k + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      c.imageSmoothingEnabled = true; c.drawImage(off, px0, py0, side, side);
      c.strokeStyle = col.rule.hex; c.lineWidth = 1; c.strokeRect(px0, py0, side, side);
      c.beginPath(); c.moveTo(X(0), py0); c.lineTo(X(0), py0 + side); c.moveTo(px0, Y(0)); c.lineTo(px0 + side, Y(0)); c.globalAlpha = 0.6; c.stroke(); c.globalAlpha = 1;
      c.font = stage.mono(10.5); c.fillStyle = col.muted.hex; c.textAlign = 'center'; c.textBaseline = 'top';
      for (const v of [-2, 0, 2]) c.fillText(f2(v, 0), X(v), py0 + side + 4);
      c.textAlign = 'right'; c.textBaseline = 'middle';
      for (const v of [-2, 0, 2]) c.fillText(f2(v, 0), px0 - 5, Y(v));
      c.font = stage.body(12, true); c.textAlign = 'right'; c.textBaseline = 'bottom'; c.fillText('x₁', px0 + side, L.pl.y + L.pl.h - 1);
      c.textAlign = 'left'; c.textBaseline = 'top'; c.fillText('x₂', px0 + 4, py0 + 3);
      c.save(); c.beginPath(); c.rect(px0, py0, side, side); c.clip();
      const nw = Math.hypot(f.w1, f.w2);
      if (nw > 1e-6) {
        const qx = (-f.b * f.w1) / (nw * nw), qy = (-f.b * f.w2) / (nw * nw), dx = -f.w2 / nw, dy = f.w1 / nw;
        c.setLineDash([6, 5]); c.strokeStyle = col.accent.hex; c.lineWidth = 2;
        c.beginPath(); c.moveTo(X(qx - 12 * dx), Y(qy - 12 * dy)); c.lineTo(X(qx + 12 * dx), Y(qy + 12 * dy)); c.stroke(); c.setLineDash([]);
        if (init.mode === 'dot-product' || init.mode === 'playground') {
          const ox = init.mode === 'dot-product' ? 0 : qx, oy = init.mode === 'dot-product' ? 0 : qy;
          c.strokeStyle = col.text.hex; c.fillStyle = col.text.hex; c.lineWidth = 2.4;
          arrow(c, X(ox), Y(oy), X(ox + f.w1), Y(oy + f.w2), 10);
          c.font = stage.body(13, true); c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText('w', X(ox + f.w1) + 6, Y(oy + f.w2) - 6);
        }
      }
      for (const p of POINTS) {
        c.fillStyle = (p.y ? col.pos : col.neg).hex; c.strokeStyle = col.raised.hex; c.lineWidth = 1.5;
        c.globalAlpha = init.mode === 'dot-product' ? 0.45 : 1;
        c.beginPath();
        if (p.y) c.arc(X(p.x1), Y(p.x2), 4.6, 0, 6.2832); else { const r = 5.6, x = X(p.x1), y = Y(p.x2); c.moveTo(x, y - r); c.lineTo(x + r, y); c.lineTo(x, y + r); c.lineTo(x - r, y); c.closePath(); }
        c.fill(); c.stroke(); c.globalAlpha = 1;
      }
      if (init.mode === 'dot-product') {
        const wl = Math.hypot(f.w1, f.w2), ux = f.w1 / wl, uy = f.w2 / wl, pr = f.x1 * ux + f.x2 * uy, fx = pr * ux, fy = pr * uy;
        c.strokeStyle = col.fwd.hex; c.lineWidth = 4; c.lineCap = 'butt'; c.beginPath(); c.moveTo(X(0), Y(0)); c.lineTo(X(fx), Y(fy)); c.stroke(); c.lineCap = 'round';
        c.setLineDash([3, 3]); c.strokeStyle = col.muted.hex; c.lineWidth = 1.3; c.beginPath(); c.moveTo(X(f.x1), Y(f.x2)); c.lineTo(X(fx), Y(fy)); c.stroke(); c.setLineDash([]);
        c.strokeStyle = col.accent.hex; c.fillStyle = col.accent.hex; c.lineWidth = 2; arrow(c, X(0), Y(0), X(f.x1), Y(f.x2), 9);
        const a1 = Math.atan2(f.x2, f.x1), r = 0.6;
        if (Math.abs(f.theta) > 0.05) {
          c.strokeStyle = col.text.hex; c.lineWidth = 1.4; c.beginPath(); c.arc(X(0), Y(0), r * (side / (2 * DOM)), -a1, -(a1 + f.theta), f.theta > 0); c.stroke();
          const m = a1 + f.theta / 2; c.font = stage.mono(11, 650); c.fillStyle = col.text.hex; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('θ', X(0.95 * Math.cos(m)), Y(0.95 * Math.sin(m)));
        }
      }
      if (init.mode !== 'dot-product') {
        c.fillStyle = col.accent.hex; c.strokeStyle = col.raised.hex; c.lineWidth = 2; c.beginPath(); c.arc(X(f.x1), Y(f.x2), 7, 0, 6.2832); c.fill(); c.stroke();
      } else {
        c.fillStyle = col.accent.hex; c.strokeStyle = col.raised.hex; c.lineWidth = 2; c.beginPath(); c.arc(X(f.x1), Y(f.x2), 5.5, 0, 6.2832); c.fill(); c.stroke();
        c.font = stage.body(13, true); c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText('x', X(f.x1) + 8, Y(f.x2) - 8);
      }
      c.restore();
    }

    function drawPlot(f: Frame, B: { x: number; y: number; w: number; h: number }) {
      const { ctx: c, c: col } = stage, l = B.x + 36, r = B.x + B.w - 10, tp = B.y + 20, bt = B.y + B.h - 22, YR = 2.2;
      const X = scale(-Z_RANGE, Z_RANGE, l, r), Y = scale(-YR, YR, bt, tp);
      c.font = stage.body(12); c.fillStyle = col.text.hex; c.textAlign = 'left'; c.textBaseline = 'top'; c.fillText('Activation curve', B.x + 2, B.y + 1);
      c.strokeStyle = col.rule.hex; c.lineWidth = 1; c.strokeRect(l, tp, r - l, bt - tp);
      c.beginPath(); c.moveTo(X(0), tp); c.lineTo(X(0), bt); c.moveTo(l, Y(0)); c.lineTo(r, Y(0)); c.moveTo(l, Y(1)); c.lineTo(r, Y(1)); c.globalAlpha = 0.55; c.stroke(); c.globalAlpha = 1;
      c.font = stage.mono(10.5); c.fillStyle = col.muted.hex; c.textAlign = 'right'; c.textBaseline = 'middle';
      for (const v of [-2, 0, 1, 2]) c.fillText(f2(v, 0), l - 5, Y(v));
      c.textAlign = 'center'; c.textBaseline = 'top';
      for (const v of [-2, 0, 2]) c.fillText(f2(v, 0), X(v), bt + 4);
      c.textAlign = 'right'; c.font = stage.body(12, true); c.textBaseline = 'bottom'; c.fillText('z', r, B.y + B.h - 1);
      c.save(); c.beginPath(); c.rect(l, tp, r - l, bt - tp); c.clip(); c.lineJoin = 'round';
      for (const a of ACTS) {
        const on = a === f.act;
        c.strokeStyle = on ? col.fwd.hex : col.muted.hex; c.globalAlpha = on ? 1 : 0.4; c.lineWidth = on ? 3 : 1.4; c.beginPath();
        for (let i = 0; i <= 200; i++) { const z = -Z_RANGE + (2 * Z_RANGE * i) / 200; if (a === 'step' && Math.abs(z) < 0.0126) { c.lineTo(X(0), Y(0)); c.moveTo(X(0), Y(1)); } const y = activate(a, z); if (i && !(a === 'step' && Math.abs(z) < 0.0126)) c.lineTo(X(z), Y(y)); else c.moveTo(X(z), Y(y)); }
        c.stroke();
      }
      c.globalAlpha = 1;
      const zc = Math.max(-Z_RANGE, Math.min(Z_RANGE, f.z)), yc = f.out;
      c.setLineDash([3, 3]); c.strokeStyle = col.text.hex; c.lineWidth = 1; c.beginPath(); c.moveTo(X(zc), bt); c.lineTo(X(zc), Y(yc)); c.lineTo(l, Y(yc)); c.stroke(); c.setLineDash([]);
      c.fillStyle = col.accent.hex; c.strokeStyle = col.raised.hex; c.lineWidth = 2; c.beginPath(); c.arc(X(zc), Y(yc), 6, 0, 6.2832); c.fill(); c.stroke();
      c.restore();
    }

    function render() {
      const current = snap();
      // Paint the static caption's text node at display rate, preserving Svelte's DOM node.
      if (values?.firstChild) values.firstChild.nodeValue = valueText(current.f, current.acc);
      const { ctx: c, w: W, h: H } = stage, L = lay(), f = current.f;
      c.clearRect(0, 0, W, H);
      drawDiagram(f, L.dia, playing && !reducedMotion() ? t : -1);
      drawPlane(f, L);
      if (L.plot) drawPlot(f, L.plot);
    }

    let lastUi = 0;
    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      const current = snap();
      // Playground sliders expose real changing parameters. Update only when their displayed
      // values change; other DOM state is sampled, or updated immediately at a phase boundary.
      const knobsChanged = init.mode === 'playground' && ['w1', 'w2', 'b'].some(k => f2(current.f[k as keyof Params]) !== f2(ui.f[k as keyof Params]));
      if (!playing || now - lastUi > 100 || current.f.phase !== ui.f.phase || knobsChanged) {
        ui = current; uiT = t; lastUi = now;
      }
      render(); tl.paint(t);
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });

  const place = (p: Partial<Params>) => { const f = cur(); over = { w1: f.w1, w2: f.w2, b: f.b, ...p }; playing = false; ui = snap(); wake(); };
  const replay = () => { over = null; t = 0; uiT = 0; playing = !reducedMotion(); ui = snap(); wake(); };
  const fm = $derived(ui.f);
</script>

<div class="neuron">
<FigureCard title={init.title} role="group" aspect={isAct ? '800 / 440' : '800 / 400'} narrowAspect={isAct ? '360 / 720' : '360 / 580'}
  label="Left: a neuron drawn as a diagram. Two inputs x1 and x2 are multiplied by weights w1 and w2, added together with a bias b to give the sum z, passed through an activation, and the output is shown. Right: the plane of inputs, coloured blue where the neuron leans to class 1 and orange where it leans to class 0, with a dashed boundary where the sum is zero and two classes of points."
  legend={[{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'boundary z = 0 and test point', color: '--accent', shape: 'line' }, { label: 'sum and output', color: '--viz-forward', shape: 'line' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: one neuron"
      onSeek={(to) => { over = null; t = to; uiT = to; ui = snap(); wake(); }} onReset={replay} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num" bind:this={values}>{initialValues}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if init.mode === 'playground'}
      <div class="ctl">
        <F1Slider label="w₁" value={fm.w1} min={-3} max={3} step={0.05} oninput={(v) => place({ w1: v })} />
        <F1Slider label="w₂" value={fm.w2} min={-3} max={3} step={0.05} oninput={(v) => place({ w2: v })} />
        <F1Slider label="b" value={fm.b} min={-3} max={3} step={0.05} oninput={(v) => place({ b: v })} />
        <label class="act">Activation <select bind:value={userAct}>{#each ACTS as a (a)}<option value={a}>{a}</option>{/each}</select></label>
        <button type="button" class="btn" onclick={replay}>Replay the script</button>
        <span class="hint">Target: w = ({FINAL.w1}, {FINAL.w2}), b = {f2(FINAL.b, 1)}. Dragging pauses the script.</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>
</div>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.6 var(--font-code); font-variant-numeric: tabular-nums; white-space: pre-line; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .act { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .act select { font: 0.85rem var(--font-code); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
  @media (max-width: 540px) {
    .neuron :global(.tick:last-child) { transform: translateX(-100%); }
    .neuron :global(.tick:last-child::before) { left: auto; right: 0; }
  }
</style>
