<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { POINTS, BEST, START, W_DOM, B_DOM, X_DOM, LR, STEPS, PATH, loss, sigmoid, accuracy, grad, gdPath, probabilityCalculation, duration, markers, frameAt, T_GUESS, type Mode, type PathPt } from './sim';

  interface Props {
    title?: string;
    /** 'train': gradient descent on log-loss. 'sweep': a probe slides along x and reads off score and probability. 'steepen': the curve sharpens with the boundary fixed. 'knobs': slide b, then tilt w. */
    mode?: Mode;
    /** Draw the dashed decision boundary (where p = 0.5). */
    boundary?: boolean;
    /** Draw the downhill (negative gradient) arrow on the map and write the update out in the readout. */
    showGrad?: boolean;
  }
  let { title = '', mode = 'train', boundary = true, showGrad = false }: Props = $props();
  const init = untrack(() => ({ mode, boundary, showGrad, title: title || 'A line, squashed into a probability' }));
  const T = duration(init.mode), MK = markers(init.mode);
  const LOSS_MIN = BEST.loss;
  const sweep = init.mode === 'sweep';

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  /** Hand-set knobs in the demo modes; cleared when the timeline plays again. */
  let over = $state<{ w: number; b: number } | null>(null);
  /** Train mode: true after the reader placed a guess, until Play. The guess is path[0], so every replay starts there. */
  let manual = $state(false);
  let path = $state.raw<PathPt[]>(PATH);
  let ui = $state({ ...START, loss: loss(START.w, START.b), acc: accuracy(START.w, START.b), probe: null as null | { x: number; z: number; p: number } });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const phase = $derived(over || manual ? 'manual' : (markerAt(uiT, MK)?.label ?? ''));
  const g = $derived(grad(ui.w, ui.b));
  const arithmetic = $derived(ui.probe ? probabilityCalculation(ui.w, ui.b, ui.probe.x) : null);
  const precise = (v: number) => v !== 0 && Math.abs(v) < 0.000001 ? v.toExponential(3) : f2(v, 6);
  const narration = $derived(
    phase === 'manual' ? (init.mode === 'train' ? 'You placed a new guess. Press Play (or Descend from here) and training starts from this spot.' : 'You set w and b by hand. Press Play to resume the demo.')
      : phase === 'slide b' ? 'Only b changes: the curve slides left and right, and the dashed boundary at x = −b/w slides with it. The steepness stays the same.'
      : phase === 'tilt w' ? 'Only w changes: the curve gets steeper or flatter. Steep means confident far from the boundary; flat means unsure everywhere.'
      : phase === 'a guess' ? 'A first guess: the curve is nearly flat and pushed to the right, so points of both classes get poor probabilities.'
      : phase === 'descend' ? 'Each step nudges (w, b) downhill on the log-loss. The curve shifts and steepens; the dot on the map slides to the minimum.'
      : phase === 'settled' && init.mode === 'train' ? (ui.loss - LOSS_MIN > 0.005 ? `All ${STEPS} steps used. The model is still above the minimum and needs more updates.` : 'Forty updates complete: close to the minimum. More updates can still reduce the loss.')
      : phase === 'far left' ? 'Far to the left the score is very negative, so the probability is close to 0.'
      : phase === 'near the boundary' ? 'Near the boundary the score is close to 0 and the probability passes through 0.5.'
      : phase === 'far right' ? 'Far to the right the score is large and positive, so the probability is close to 1.'
      : phase === 'too flat' ? 'Too flat: the model is never sure, so even correct points get probabilities near one half.'
      : phase === 'best fit' ? 'The fitted steepness: the lowest log-loss on this data.'
      : phase === 'sharpening' ? 'Steeper than the fit: confident predictions start to pay for the points they get wrong.'
      : phase === 'overconfident' ? 'Overconfident: wrong points now sit at probabilities near 0 or 1, and the log-loss climbs again.'
      : '',
  );
  $effect(() => { if (playing) { over = null; manual = false; } speed; wake(); });

  const frame = () => {
    if (over) return { w: over.w, b: over.b, loss: loss(over.w, over.b), k: 0, probe: sweep ? probabilityCalculation(over.w, over.b, frameAt(t, 'sweep').probe!.x) : null };
    return frameAt(t, init.mode, path);
  };
  const snap = () => { const f = frame(); return { w: f.w, b: f.b, loss: f.loss, acc: accuracy(f.w, f.b), probe: f.probe }; };
  const place = (w: number, b: number) => {
    const s = { w: clamp(w, W_DOM[0], W_DOM[1]), b: clamp(b, B_DOM[0], B_DOM[1]) };
    // In train mode the guess becomes the start of the descent, so Play, Step and scrubbing all run from it.
    if (init.mode === 'train') { path = gdPath(s, LR, STEPS); t = 0; uiT = 0; manual = true; }
    else over = s;
    playing = false; ui = snap(); wake();
  };
  function launch() { manual = false; over = null; t = T_GUESS; playing = true; wake(); }
  function reset() { path = PATH; over = null; manual = false; t = 0; playing = !reducedMotion(); wake(); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let dragging = false;
    const NX = 64, NY = 56;
    const grid = new Float64Array(NX * NY);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) grid[j * NX + i] = loss(W_DOM[0] + ((W_DOM[1] - W_DOM[0]) * i) / (NX - 1), B_DOM[0] + ((B_DOM[1] - B_DOM[0]) * j) / (NY - 1));
    const LEVELS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5].map((d) => LOSS_MIN + d);
    let cache = { key: '', paths: [] as Path2D[] };
    const lay = () => {
      const panels = splitPanels(stage.w, stage.h, 0.55);
      const { side, stacked } = panels;
      const main = sweep ? { x: 0, y: 0, w: stage.w, h: stage.h } : panels.main;
      const d = { l: main.x + 34, r: main.x + main.w - 8, t: main.y + 14, b: main.y + main.h - 26 };
      const s = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 22, b: side.y + side.h - 30 };
      return { main, side, stacked, d, s, dx: scale(X_DOM[0] - (sweep ? 0.15 : 0), X_DOM[1] + (sweep ? 0.15 : 0), d.l, d.r), dy: scale(-0.1, 1.1, d.b, d.t), cw: scale(W_DOM[0], W_DOM[1], s.l, s.r), cb: scale(B_DOM[0], B_DOM[1], s.b, s.t) };
    };
    function contours(L: ReturnType<typeof lay>) {
      const key = `${L.s.l}|${L.s.r}|${L.s.t}|${L.s.b}`;
      if (cache.key === key) return cache.paths;
      const gx = (i: number) => L.s.l + (i / (NX - 1)) * (L.s.r - L.s.l), gy = (j: number) => L.s.b + (j / (NY - 1)) * (L.s.t - L.s.b);
      cache = { key, paths: LEVELS.map((lv) => {
        const seg = contour((i, j) => grid[j * NX + i], NX, NY, lv), p = new Path2D();
        for (let k = 0; k < seg.length; k += 4) { p.moveTo(gx(seg[k]), gy(seg[k + 1])); p.lineTo(gx(seg[k + 2]), gy(seg[k + 3])); }
        return p;
      }) };
      return cache.paths;
    }
    const minus = (v: number | string) => String(v).replace('-', '−');

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frame();
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      const { d } = L;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [-4, -2, 0, 2, 4]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(minus(x), L.dx(x), d.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of [0, 0.5, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(d.l, L.dy(y)); ctx.lineTo(d.r, L.dy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y), d.l - 6, L.dy(y)); }
      ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, d.r - d.l, d.b - d.t); ctx.clip();
      if (!sweep && init.boundary && Math.abs(f.w) > 1e-3) {
        const xb = -f.b / f.w;
        ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.8;
        ctx.beginPath(); ctx.moveTo(L.dx(xb), d.t); ctx.lineTo(L.dx(xb), d.b); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      // the sigmoid curve
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) { const x = X_DOM[0] + ((X_DOM[1] - X_DOM[0]) * i) / 120, y = sigmoid(f.w * x + f.b); if (i) ctx.lineTo(L.dx(x), L.dy(y)); else ctx.moveTo(L.dx(x), L.dy(y)); }
      ctx.stroke();
      // each point: stem length = how far its probability is from the truth
      for (const p of sweep ? [] : POINTS) {
        const q = sigmoid(f.w * p.x + f.b), ty = p.y, wrongness = Math.abs(ty - q);
        ctx.strokeStyle = c.grad.hex; ctx.globalAlpha = 0.25 + 0.65 * wrongness; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(L.dx(p.x), L.dy(ty)); ctx.lineTo(L.dx(p.x), L.dy(q)); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(q), 3, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      for (const p of sweep ? [] : POINTS) {
        const col = p.y ? c.pos : c.neg;
        ctx.fillStyle = col.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(p.y), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      if (f.probe) {
        const { x, p } = f.probe;
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(L.dx(x), d.b); ctx.lineTo(L.dx(x), L.dy(p)); ctx.lineTo(d.l, L.dy(p)); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(L.dx(x), L.dy(p), 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', d.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('P(y = 1) = σ(w·x + b)', d.l + 4, L.main.y + 1);
      if (sweep) return;

      const s = L.s;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Log-loss over every (w, b)', s.l - 30, L.side.y + 2);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const w of [-1, 0, 1, 2, 3, 4]) ctx.fillText(minus(w), L.cw(w), s.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const b of [-3, -2, -1, 0, 1, 2, 3]) ctx.fillText(minus(b), s.l - 6, L.cb(b));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('w (steepness)', s.r, L.side.y + L.side.h - 1);
      ctx.save(); ctx.translate(L.side.x + 11, s.t); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('b (offset)', 0, 0); ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(s.l, s.t, s.r - s.l, s.b - s.t);
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t, s.r - s.l, s.b - s.t); ctx.clip();
      contours(L).forEach((p, i) => { ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.35 + 0.07 * i; ctx.lineWidth = 1.3; ctx.stroke(p); });
      ctx.globalAlpha = 1;
      if (!over && init.mode === 'train') {
        const k = f.k;
        ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(L.cw(path[0].w), L.cb(path[0].b));
        for (let i = 1; i <= Math.floor(k); i++) ctx.lineTo(L.cw(path[i].w), L.cb(path[i].b));
        ctx.lineTo(L.cw(f.w), L.cb(f.b)); ctx.stroke();
        for (let i = 0; i <= Math.floor(k); i++) { ctx.fillStyle = c.grad.hex; ctx.beginPath(); ctx.arc(L.cw(path[i].w), L.cb(path[i].b), 2.5, 0, 6.2832); ctx.fill(); }
      }
      if (init.mode === 'steepen' && !over) {
        ctx.setLineDash([4, 4]); ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 1.6; ctx.beginPath();
        ctx.moveTo(L.cw(BEST.w * 0.1), L.cb(BEST.b * 0.1)); ctx.lineTo(L.cw(BEST.w * 3), L.cb(BEST.b * 3)); ctx.stroke(); ctx.setLineDash([]);
      }
      if (init.showGrad) {
        // the step gradient descent is about to take: -lr * gradient, drawn from the dot
        const [gw, gb] = grad(f.w, f.b), x0 = L.cw(f.w), y0 = L.cb(f.b), x1 = L.cw(f.w - LR * gw), y1 = L.cb(f.b - LR * gb);
        if (Math.hypot(x1 - x0, y1 - y0) > 2) {
          const ang = Math.atan2(y1 - y0, x1 - x0);
          ctx.strokeStyle = c.grad.hex; ctx.fillStyle = c.grad.hex; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 8 * Math.cos(ang - 0.45), y1 - 8 * Math.sin(ang - 0.45)); ctx.lineTo(x1 - 8 * Math.cos(ang + 0.45), y1 - 8 * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fill();
        }
      }
      ctx.restore();
      const bx = L.cw(BEST.w), by = L.cb(BEST.b);
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx, by - 7); ctx.lineTo(bx + 7, by); ctx.lineTo(bx, by + 7); ctx.lineTo(bx - 7, by); ctx.closePath(); ctx.stroke();
      ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(L.cw(clamp(f.w, W_DOM[0], W_DOM[1])), L.cb(clamp(f.b, B_DOM[0], B_DOM[1])), 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    const loop = createLoop(cv, (dt) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      ui = snap(); uiT = t;
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const inSide = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top; return { L, x, y, ok: !sweep && x >= L.s.l - 8 && x <= L.s.r + 8 && y >= L.s.t - 8 && y <= L.s.b + 8 }; };
    const put = (e: PointerEvent) => { const { L, x, y } = inSide(e); place(W_DOM[0] + ((x - L.s.l) / (L.s.r - L.s.l)) * (W_DOM[1] - W_DOM[0]), B_DOM[0] + ((y - L.s.b) / (L.s.t - L.s.b)) * (B_DOM[1] - B_DOM[0])); };
    const down = (e: PointerEvent) => { if (inSide(e).ok) { dragging = true; cv.setPointerCapture(e.pointerId); put(e); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (dragging) put(e); else cv.style.cursor = inSide(e).ok ? 'crosshair' : ''; };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect={sweep ? '720 / 320' : '720 / 400'} narrowAspect={sweep ? '360 / 300' : '360 / 620'}
  label={sweep ? 'Probability curve with a movable input x. Dotted guides connect x to its probability; the calculation below expands every operation.' : 'Left: sixteen labeled points and the sigmoid probability curve. Right: contour map of log-loss over w and b, with the current curve marked by a dot and the minimum by a diamond.'}
  legend={sweep ? [{ label: 'probability curve', color: '--viz-forward', shape: 'line' }, { label: 'selected input', color: '--accent' }] : [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'P(y=1) curve', color: '--viz-forward', shape: 'line' }, { label: init.showGrad ? 'error (stem) / downhill step' : 'error (stem)', color: '--viz-gradient', shape: 'line' }, { label: 'minimum / boundary', color: '--accent' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: the sigmoid model"
      onSeek={(to) => { over = null; manual = false; t = to; wake(); }} onStep={(to) => { over = null; manual = false; t = to >= T ? 0 : to; wake(); }} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    {#if sweep && arithmetic}
      <div class="arithmetic">
        <p><strong>1. Multiply:</strong> w × x ≈ ({f2(ui.w, 6)}) × ({f2(arithmetic.x, 4)}) ≈ {f2(arithmetic.product, 6)}.</p>
        <p><strong>2. Add b:</strong> z ≈ {f2(arithmetic.product, 6)} + ({f2(ui.b, 6)}) ≈ <b>{f2(arithmetic.z, 6)}</b>.</p>
        <p><strong>3. Flip the sign and exponentiate:</strong> −z ≈ {f2(arithmetic.exponent, 6)}; e<sup>−z</sup> ≈ <b>{precise(arithmetic.exponential)}</b>.</p>
        <p><strong>4. Add 1, then divide:</strong> p = 1 / (1 + e<sup>−z</sup>) ≈ 1 / {f2(arithmetic.denominator, 6)} ≈ <b>{precise(arithmetic.p)}</b>.</p>
        <p><strong>As a percentage:</strong> p × 100 ≈ <b>{f2(arithmetic.percent, 3)}%</b> probability of class 1.</p>
        <p class="say">e ≈ 2.71828. Values are rounded; ≈ means approximately equal.</p>
      </div>
    {:else}
    <p class="vals num">w = <b>{f2(ui.w)}</b> · b = <b>{f2(ui.b)}</b> · log-loss = <b>{f2(ui.loss, 3)}</b> · accuracy = <b>{Math.round(ui.acc * 16)}/16</b>
      {#if ui.probe}<span class="base">· at x = {f2(ui.probe.x, 1)}: score z = {f2(ui.probe.z)}, P(y=1) = {f2(ui.probe.p)}</span>
      {:else}<span class="base">· minimum: w = {f2(BEST.w)}, b = {f2(BEST.b)}, log-loss = {f2(LOSS_MIN, 3)}</span>{/if}</p>
    {#if init.showGrad}<p class="vals num grad">∂L/∂w = <b>{f2(g[0], 3)}</b> · ∂L/∂b = <b>{f2(g[1], 3)}</b>
      <span class="base">· next step: w → {f2(ui.w - LR * g[0])}, b → {f2(ui.b - LR * g[1])} (step size {LR})</span></p>{/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    {/if}
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="w" value={ui.w} min={W_DOM[0]} max={W_DOM[1]} step={0.01} oninput={(v) => place(v, ui.b)} />
      <F1Slider label="b" value={ui.b} min={B_DOM[0]} max={B_DOM[1]} step={0.01} oninput={(v) => place(ui.w, v)} />
      {#if sweep}<F1Slider label="input x" value={ui.probe?.x ?? -4} min={-4} max={4} step={0.01} oninput={(v) => { t = v + 4.5; playing = false; ui = snap(); uiT = t; wake(); }} />{/if}
      {#if init.mode === 'train'}<button type="button" class="btn" onclick={launch}>Descend from here</button>{/if}
      <span class="hint">{sweep ? 'Change input x and follow its calculation above.' : `You can also click or drag on the contour map. ${Math.abs(ui.w) < 1e-9 ? 'w is zero: probability is the same for every input.' : `Decision boundary at x = ${f2(-ui.b / ui.w, 2)}.`}`}</span>
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .arithmetic { font: 0.95rem / 1.6 var(--font-ui); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .arithmetic p + p { margin-top: var(--s-2); }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .vals.grad b { color: var(--viz-gradient); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
