<script lang="ts">
  import { onMount } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar } from '../_shared';
  import { DURATION, MARKERS, NONCONVEX_STEPS, SADDLE_STEPS, frameAt, nonconvexLoss, saddleLoss, type Mode } from './sim';

  interface Props {
    mode?: Mode;
    title?: string;
  }
  let { mode = 'nonconvex', title = 'A loss landscape' }: Props = $props();
  const uid = $props.id();
  let startSide = $state<'left' | 'right'>('right'), view = $state<'slice' | 'map'>('slice');
  let coordinateScale = $state(4);
  let alphaMax = $state(0.5);
  let playing = $state(false);
  let speed = $state(0.5);
  let uiT = $state(0);
  let t = 0;
  let wake = () => {};
  let cv!: HTMLCanvasElement;
  let tl: ReturnType<typeof Timeline>;
  const frame = $derived(frameAt(uiT, mode, { scale: coordinateScale, alpha: alphaMax, side: startSide }));
  // Dimension markers show only the number so they fit at 320 px; the phase text keeps the full name.
  const markers = $derived(mode === 'dimension' ? MARKERS[mode].map((m) => ({ ...m, label: m.label.split(' ')[0] })) : MARKERS[mode]);
  const f3 = (n: number) => n.toFixed(3);
  const legends = $derived(mode === 'nonconvex'
    ? view === 'slice'
      ? [{ label: 'Current weights', color: '--viz-forward', shape: 'dot' as const }, { label: 'Loss along x', color: '--text-muted', shape: 'line' as const }]
      : [{ label: 'Path of the weights', color: '--viz-forward', shape: 'line' as const }, { label: 'Equal-loss lines', color: '--text-muted', shape: 'line' as const }]
    : mode === 'saddle'
      ? view === 'slice'
        ? [{ label: 'Displaced start', color: '--viz-forward', shape: 'dot' as const }, { label: 'Exact center', color: '--text-muted', shape: 'dot' as const }]
        : [{ label: 'Displaced start', color: '--viz-forward', shape: 'line' as const }, { label: 'Exact center', color: '--text-muted', shape: 'dot' as const }, { label: 'Loss below the center', color: '--viz-neg', shape: 'line' as const }]
      : mode === 'coordinates'
        ? [{ label: 'Original units θ', color: '--text-muted', shape: 'line' as const }, { label: 'Stretched units θ′ = cθ', color: '--viz-forward', shape: 'line' as const }]
        : mode === 'dimension'
          ? [{ label: 'All up: minimum', color: '--viz-pos', shape: 'dot' as const }, { label: 'All down: maximum', color: '--viz-neg', shape: 'dot' as const }, { label: 'Mixed: saddle', color: '--text-muted', shape: 'dot' as const }]
          : [{ label: 'Plain network', color: '--text-muted', shape: 'line' as const }, { label: 'With skip connections', color: '--viz-gradient', shape: 'line' as const }, { label: 'Direct path alone (0.5)', color: '--rule', shape: 'line' as const }]);
  const explain = $derived.by(() => {
    const p = frame.phase, side = startSide === 'right' ? 'right' : 'left';
    if (mode === 'nonconvex') return p === 'Downhill'
      ? `The weights start just ${side} of the hilltop, so every downhill step pushes x further ${side}, into the ${side}-hand well.`
      : `x has reached the well near ${side === 'right' ? '1' : '−1'}. The remaining updates mostly shrink y, which is why the whole curve keeps sinking toward zero.`;
    if (mode === 'saddle') return p === 'Near center'
      ? 'The gray dot sits exactly on the center, where the slope is zero, so gradient descent never moves it. The teal dot starts 0.06 away along y, where the slope is tiny but not zero.'
      : p === 'Escaping'
        ? 'Along y the surface curves down, so the slope grows as the dot moves away and each step is longer than the last.'
        : 'The teal dot has reached a valley where loss is about −1. The gray dot is still on the center at loss 0.';
    if (mode === 'dimension') return `Each square is one way the ${frame.dimension} directions could curve: + for up, − for down. Only the top-left square, every direction up, is a minimum.`;
    if (mode === 'coordinates') return p === 'Unscaled'
      ? 'Both curves are the same function measured in the same units, so they lie on top of each other.'
      : p === 'Stretching'
        ? 'Now measure the parameter in smaller units, θ′ = cθ. Each point slides outward by c but keeps its height.'
        : 'Same losses, same minimum, same predictions. Only the units changed, yet the bowl looks much flatter.';
    return p === 'Off'
      ? 'With α = 0 each residual block passes its input straight through, so the output is 0.5 × x and the sensitivity is 0.5 at every input.'
      : p === 'Growing'
        ? 'The branch α·tanh(w·h) is added on top of the direct path. Its slope adds to the 1 that the direct path contributes.'
        : `At x = 1 the plain network passes on ${f3(frame.tinyNet.plainJacobian[4])} of a small input change; with skip connections it passes on ${f3(frame.tinyNet.residualJacobian[4])}.`;
  });

  function updateSettings() {
    playing = false;
    t = 0;
    uiT = 0;
    wake();
  }

  function seek(to: number) {
    playing = false;
    t = to;
    uiT = to;
    wake();
  }

  function canvasSize() {
    if (!cv) return { width: 0, height: 0 };
    const rect = cv.getBoundingClientRect(), dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const width = Math.max(1, Math.round(rect.width * dpr)), height = Math.max(1, Math.round(rect.height * dpr));
    if (cv.width !== width || cv.height !== height) { cv.width = width; cv.height = height; }
    return { width, height };
  }

  type Colors = ReturnType<typeof readColors>;
  type Box = { x0: number; x1: number; y0: number; y1: number };

  function draw(ctx: CanvasRenderingContext2D, colors: Colors) {
    const { width, height } = canvasSize();
    if (!width || !height) return;
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const w = width / dpr, h = height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg.hex;
    ctx.fillRect(0, 0, w, h);
    ctx.textBaseline = 'alphabetic';
    if (mode === 'nonconvex' || mode === 'saddle') { if (view === 'slice') drawSlice(ctx, colors, w, h); else drawLandscape(ctx, colors, w, h); }
    else if (mode === 'coordinates') drawCoordinates(ctx, colors, w, h);
    else if (mode === 'dimension') drawDimension(ctx, colors, w, h);
    else drawResidual(ctx, colors, w, h);
  }

  function axes(ctx: CanvasRenderingContext2D, colors: Colors, b: Box) {
    ctx.strokeStyle = colors.rule.hex; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(b.x0, b.y1); ctx.lineTo(b.x1, b.y1); ctx.stroke();
  }

  function ticks(ctx: CanvasRenderingContext2D, colors: Colors, b: Box, xs: number[], sx: (v: number) => number, ys: number[], sy: (v: number) => number, xLabel: string, h: number) {
    ctx.fillStyle = colors.muted.hex; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center';
    for (const v of xs) ctx.fillText(String(v).replace('-', '−'), sx(v), b.y1 + 15);
    ctx.fillText(xLabel, (b.x0 + b.x1) / 2, h - 6);
    ctx.textAlign = 'right';
    for (const v of ys) ctx.fillText(String(v).replace('-', '−'), b.x0 - 6, sy(v) + 4);
    ctx.textAlign = 'start';
  }

  function curve(ctx: CanvasRenderingContext2D, points: { x: number; y: number }[], sx: (v: number) => number, sy: (v: number) => number) {
    ctx.beginPath();
    points.forEach((p, i) => i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y)));
    ctx.stroke();
  }

  function clipTo(ctx: CanvasRenderingContext2D, b: Box) {
    ctx.save(); ctx.beginPath(); ctx.rect(b.x0, b.y0 - 6, b.x1 - b.x0 + 6, b.y1 - b.y0 + 6); ctx.clip();
  }

  function drawLandscape(ctx: CanvasRenderingContext2D, colors: Colors, w: number, h: number) {
    const box = { x0: 44, x1: w - 14, y0: 14, y1: h - 40 }, extent = frame.plotExtent;
    axes(ctx, colors, box);
    const sx = (x: number) => box.x0 + (x + extent) / (2 * extent) * (box.x1 - box.x0);
    const sy = (y: number) => box.y1 - (y + extent) / (2 * extent) * (box.y1 - box.y0);
    ctx.lineWidth = 0.9;
    for (const segment of frame.contours) {
      ctx.strokeStyle = segment.negative ? colors.neg.hex : colors.muted.hex;
      ctx.globalAlpha = 0.7;
      ctx.beginPath(); ctx.moveTo(sx(segment.a.x), sy(segment.a.y)); ctx.lineTo(sx(segment.b.x), sy(segment.b.y)); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ticks(ctx, colors, box, [-1, 0, 1], sx, [-1, 0, 1], sy, 'x weight  (vertical: y weight)', h);
    ctx.strokeStyle = colors.fwd.hex; ctx.lineWidth = 2.2;
    if (mode === 'nonconvex') {
      curve(ctx, frame.path, sx, sy);
      dot(ctx, sx(frame.point.x), sy(frame.point.y), 5, colors.fwd.hex);
    } else {
      const still = frame.saddleStill, escape = frame.saddleEscape;
      curve(ctx, [{ x: 0, y: 0.06 }, escape], sx, sy);
      dot(ctx, sx(still.x), sy(still.y), 5, colors.muted.hex);
      dot(ctx, sx(escape.x), sy(escape.y), 5, colors.fwd.hex);
    }
  }

  function drawSlice(ctx: CanvasRenderingContext2D, colors: Colors, w: number, h: number) {
    const saddle = mode === 'saddle', box = { x0: 44, x1: w - 14, y0: saddle ? 58 : 44, y1: h - 40 };
    const extent = saddle ? 1.7 : 1.45;
    const low = saddle ? -1.15 : -0.08, high = saddle ? 1.25 : 1.9;
    const sx = (x: number) => box.x0 + (x + extent) / (2 * extent) * (box.x1 - box.x0);
    const sy = (y: number) => box.y1 - (y - low) / (high - low) * (box.y1 - box.y0);
    const along = (fn: (a: number) => number) => Array.from({ length: 121 }, (_, i) => { const x = -extent + 2 * extent * i / 120; return { x, y: fn(x) }; });
    ctx.font = '12px system-ui, sans-serif'; ctx.fillStyle = colors.text.hex;
    ctx.fillText(saddle ? 'Loss along each weight direction through the center' : `Loss along x, with y held at its current value ${frame.point.y.toFixed(2)}`, 12, 22);
    axes(ctx, colors, box);
    ctx.strokeStyle = colors.rule.hex; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(box.x0, sy(0)); ctx.lineTo(box.x1, sy(0)); ctx.stroke(); ctx.setLineDash([]);
    ticks(ctx, colors, box, [-1, 0, 1], sx, saddle ? [-1, 0, 1] : [0, 0.5, 1, 1.5], sy, saddle ? 'distance from the center' : 'x weight', h);
    ctx.fillStyle = colors.muted.hex; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'start';
    ctx.fillText('loss', 12, box.y0 - 4);
    clipTo(ctx, box);
    ctx.strokeStyle = colors.muted.hex; ctx.lineWidth = 2;
    if (saddle) {
      ctx.setLineDash([6, 4]); curve(ctx, along((a) => saddleLoss({ x: a, y: 0 })), sx, sy); ctx.setLineDash([]);
      curve(ctx, along((a) => saddleLoss({ x: 0, y: a })), sx, sy);
    } else curve(ctx, along((a) => nonconvexLoss({ x: a, y: frame.point.y })), sx, sy);
    ctx.restore();
    ctx.fillStyle = colors.muted.hex; ctx.font = '11px system-ui, sans-serif';
    if (saddle) {
      ctx.textAlign = 'start'; ctx.fillText('Dashed: along x, curves up · Solid: along y, curves down', 12, 38);
      dot(ctx, sx(0), sy(0), 5, colors.muted.hex);
      const y = frame.saddleEscape.y;
      dot(ctx, sx(y), sy(saddleLoss({ x: 0, y })), 5.5, colors.fwd.hex);
    } else {
      ctx.textAlign = 'center';
      ctx.fillText('hill', sx(0), sy(nonconvexLoss({ x: 0, y: frame.point.y })) - 10);
      dot(ctx, sx(frame.point.x), sy(frame.loss), 5.5, colors.fwd.hex);
    }
    ctx.textAlign = 'start';
  }

  function drawCoordinates(ctx: CanvasRenderingContext2D, colors: Colors, w: number, h: number) {
    const b = { x0: 44, x1: w - 14, y0: 24, y1: h - 40 }, span = 4.3;
    const sx = (x: number) => b.x0 + (x + span) / (2 * span) * (b.x1 - b.x0);
    const sy = (y: number) => b.y1 - y / 1.1 * (b.y1 - b.y0);
    axes(ctx, colors, b);
    ticks(ctx, colors, b, [-4, -3, -2, -1, 0, 1, 2, 3, 4], sx, [0, 0.5, 1], sy, 'parameter value', h);
    ctx.fillStyle = colors.muted.hex; ctx.font = '11px system-ui, sans-serif'; ctx.fillText('loss', 12, b.y0 - 8);
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = colors.muted.hex; curve(ctx, frame.originalCurve, sx, sy);
    ctx.strokeStyle = colors.fwd.hex; curve(ctx, frame.rescaledCurve, sx, sy);
    const y = sy(frame.originalLoss), a = sx(frame.theta), c = sx(frame.mappedThetaPrime);
    ctx.strokeStyle = colors.rule.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(sx(0), y); ctx.lineTo(Math.max(a, c) + 8, y); ctx.stroke(); ctx.setLineDash([]);
    dot(ctx, a, y, 5, colors.muted.hex);
    dot(ctx, c, y, 5, colors.fwd.hex);
    ctx.fillStyle = colors.text.hex; ctx.textAlign = 'right';
    ctx.textAlign = 'start'; ctx.fillText(`same loss ${frame.originalLoss.toFixed(2)}`, Math.max(a, c) + 12, y + 4);
    ctx.textAlign = 'start';
  }

  function drawDimension(ctx: CanvasRenderingContext2D, colors: Colors, w: number, h: number) {
    const d = frame.dimension, cols = 2 ** Math.ceil(d / 2), rows = 2 ** Math.floor(d / 2);
    const cell = Math.min(140, (w - 24) / cols, (h - 24) / rows), gap = cell > 6 ? 1 : 0;
    const x0 = (w - cell * cols) / 2, y0 = (h - cell * rows) / 2, last = cols * rows - 1;
    const labels = cell >= d * 8 + 10;
    ctx.globalAlpha = frame.cellReveal;
    ctx.font = `${Math.min(20, cell / 4)}px ${readVar('--font-code') || 'monospace'}`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= last; i++) {
      const x = x0 + (i % cols) * cell, y = y0 + Math.floor(i / cols) * cell;
      const extreme = i === 0 || i === last;
      ctx.fillStyle = i === 0 ? colors.pos.hex : i === last ? colors.neg.hex : colors.muted.hex;
      ctx.globalAlpha = frame.cellReveal * (extreme ? 1 : 0.28);
      ctx.fillRect(x + gap, y + gap, cell - 2 * gap, cell - 2 * gap);
      if (labels) {
        ctx.globalAlpha = frame.cellReveal;
        ctx.fillStyle = extreme ? colors.bg.hex : colors.text.hex;
        ctx.fillText(i.toString(2).padStart(d, '0').replace(/0/g, '+').replace(/1/g, '−'), x + cell / 2, y + cell / 2);
      }
    }
    ctx.globalAlpha = 1; ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }

  function drawResidual(ctx: CanvasRenderingContext2D, colors: Colors, w: number, h: number) {
    const b = { x0: 44, x1: w - 14, y0: 24, y1: h - 40 }, data = frame.tinyNet, { low, high } = frame.jacobianExtent;
    const sx = (x: number) => b.x0 + 10 + (x + 1) / 2 * (b.x1 - b.x0 - 20);
    const sy = (v: number) => b.y1 - (v - low) / (high - low) * (b.y1 - b.y0);
    axes(ctx, colors, b);
    const yTicks = [0, 0.5, 1, 1.5, 2].filter((v) => v <= high);
    ticks(ctx, colors, b, data.inputs, sx, low < 0 ? [low < -0.5 ? -0.5 : 0, ...yTicks.filter((v) => v > 0)] : yTicks, sy, 'input x', h);
    ctx.fillStyle = colors.muted.hex; ctx.font = '11px system-ui, sans-serif'; ctx.fillText('output change per unit of input change', 12, b.y0 - 8);
    ctx.strokeStyle = colors.rule.hex; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(b.x0, sy(0.5)); ctx.lineTo(b.x1, sy(0.5)); ctx.stroke();
    for (const [values, color] of [[data.plainJacobian, colors.muted.hex], [data.residualJacobian, colors.grad.hex]] as const) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2;
      curve(ctx, values.map((v, i) => ({ x: data.inputs[i], y: v })), sx, sy);
      values.forEach((v, i) => dot(ctx, sx(data.inputs[i]), sy(v), 3.6, color));
    }
  }

  function dot(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, color: string) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
  }

  onMount(() => {
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const colorsNow = () => readColors({
      bg: '--bg-raised', rule: '--rule', text: '--text', muted: '--text-muted',
      fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg',
    });
    let colors = colorsNow();
    const stage = cv.closest('[data-figure]') ?? cv;
    const resize = new ResizeObserver(() => { draw(ctx, colors); });
    resize.observe(stage);
    draw(ctx, colors);
    const loop = createLoop(stage, (dt) => {
      if (playing) {
        t = advance({ t, playing, speed }, dt, { duration: DURATION }).t;
        uiT = t;
      }
      draw(ctx, colors);
      tl?.paint(t);
      return playing;
    }, { onView: (_visible, first) => { if (first && !prefersReducedMotion().matches) playing = true; } });
    wake = loop.wake;
    const off = onThemeChange(() => { colors = colorsNow(); wake(); });
    const motion = prefersReducedMotion();
    const onMotionChange = (event: MediaQueryListEvent) => { if (event.matches) playing = false; wake(); };
    motion.addEventListener('change', onMotionChange);
    return () => { resize.disconnect(); loop.destroy(); off(); motion.removeEventListener('change', onMotionChange); };
  });

  $effect(() => { void [mode, coordinateScale, alphaMax, startSide, view, playing, speed]; wake(); });
</script>

<FigureCard {title} label="Interactive loss-landscape figure" role="group" aspect="16 / 10" narrowAspect="1 / 1" legend={legends}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={DURATION} {markers} label="Loss landscape timeline"
      onSeek={(to) => seek(to)} onStep={(to) => seek(to)} onReset={() => seek(0)} />
  {/snippet}
  {#snippet readout()}
    <div class="readout">
      <p class="explain" aria-live="polite">{explain}</p>
      {#if mode === 'nonconvex'}
        <p>Update <span class="num">{frame.update}</span> of {NONCONVEX_STEPS} · weights (x, y) = <span class="num">({f3(frame.point.x)}, {f3(frame.point.y)})</span> · loss <span class="num">{f3(frame.loss)}</span></p>
      {:else if mode === 'saddle'}
        <p>Update <span class="num">{frame.saddleUpdate}</span> of {SADDLE_STEPS} · teal dot y = <span class="num">{f3(frame.saddleEscape.y)}</span>, loss <span class="num">{f3(frame.saddleEscape.loss).replace('-', '−')}</span> · gray dot (0, 0), loss <span class="num">0</span></p>
      {:else if mode === 'dimension'}
        <p><span class="num">{frame.dimension}</span> directions · <span class="num">2<sup>{frame.dimension}</sup> = {frame.patterns}</span> sign patterns: 1 minimum, 1 maximum, <span class="num">{frame.patterns - 2}</span> saddles · chance of a minimum <span class="num">1/{frame.patterns} = {Number(frame.allPositiveProbability.toPrecision(3))}</span></p>
      {:else if mode === 'coordinates'}
        <p>c = <span class="num">{frame.coordinateScale.toFixed(2)}</span> · matched points θ = <span class="num">0.70</span> and θ′ = <span class="num">{frame.mappedThetaPrime.toFixed(2)}</span> both have loss <span class="num">{f3(frame.rescaledLoss)}</span></p>
        <p>Curvature (how sharply the bowl bends): original <span class="num">2</span> · stretched <span class="num">2/c² = {frame.rescaledCurvature.toFixed(3)}</span></p>
      {:else}
        {@const n = frame.tinyNet}
        <div class="blocks" aria-label="Sensitivity calculation for input 1">
          <strong>Follow input x = 1 (α = {n.alpha.toFixed(2)})</strong>
          <p>Plain: block slopes <span class="num">{f3(n.plainBlocks[0].derivative)} × {f3(n.plainBlocks[1].derivative)}</span> × readout 0.5 = <strong class="num">{f3(n.plainJacobian[4])}</strong></p>
          {#each n.blocks as block, i (i)}<p>Skip block {i + 1}: <span class="num">{f3(block.input)} + branch {f3(block.branch)} = {f3(block.output)}</span>; slope <span class="num">1 + {f3(block.derivative - 1)} = {f3(block.derivative)}</span></p>{/each}
          <p>With skips: <span class="num">{f3(n.blocks[0].derivative)} × {f3(n.blocks[1].derivative)}</span> × readout 0.5 = <strong class="num">{f3(n.residualJacobian[4])}</strong></p>
        </div>
      {/if}
    </div>
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {mode === 'nonconvex' ? 'Switch Starting side to the left and replay: the same rule, started a mirror-image hair left of the hilltop, ends in the other well. Then switch View to the contour map to see both weights at once.' : mode === 'saddle' ? 'Switch View to the contour map. The center sits between two valleys (the colored lines are below its height) and two rising sides.' : mode === 'dimension' ? 'Drag the timeline back to 2 directions and read the four patterns. Then step to 12 directions and find the single minimum square in the top-left corner.' : mode === 'coordinates' ? 'Set c to 2 and replay. The stretched curvature becomes 2/2² = 0.5, and the matched dots still sit at the same height.' : 'Set α to −0.8 and replay. The branch now cancels the direct path near x = 0, and the skip-connection line drops toward 0.'}</p>
    {#if mode === 'nonconvex' || mode === 'saddle'}<div class="settings">
      {#if mode === 'nonconvex'}<label for="{uid}-side">Starting side<select id="{uid}-side" bind:value={startSide} onchange={updateSettings}><option value="right">Right of the hilltop</option><option value="left">Left of the hilltop</option></select></label>{/if}
      <label for="{uid}-view">View<select id="{uid}-view" bind:value={view}><option value="slice">Cross section</option><option value="map">Contour map (both weights)</option></select></label>
    </div>{/if}
    {#if mode === 'coordinates'}
      <div class="settings">
        <label for="{uid}-scale">Unit stretch c <output>{coordinateScale}</output>
          <input id="{uid}-scale" type="range" min="1" max="4" step="0.5" bind:value={coordinateScale} oninput={updateSettings} />
        </label>
      </div>
    {:else if mode === 'residual'}
      <div class="settings">
        <label for="{uid}-alpha">Branch strength α <output>{alphaMax.toFixed(2)}</output>
          <input id="{uid}-alpha" type="range" min="-1.2" max="1.2" step="0.05" bind:value={alphaMax} oninput={updateSettings} />
        </label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .blocks { padding: var(--s-3); margin-bottom: var(--s-2); background: var(--bg-sunken); border-left: 2px solid var(--viz-gradient); }
  .blocks p { margin: var(--s-2) 0 0; line-height: 1.6; }
  .explain { line-height: 1.6; }
  .try { padding: var(--s-3) var(--s-4) 0; margin: 0; line-height: 1.6; }
  .readout p { margin: 0 0 var(--s-2); }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; }
  .settings { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr)); gap: var(--s-3); padding: var(--s-3) var(--s-4); }
  .settings label { display: grid; gap: var(--s-1); font-weight: 600; }
  .settings select { min-height: 2.5rem; padding: 0.35rem 0.5rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); font: inherit; }
  .settings input { width: 100%; accent-color: var(--accent); }
  .settings output { float: right; font-family: var(--font-code); font-weight: 400; }
</style>
