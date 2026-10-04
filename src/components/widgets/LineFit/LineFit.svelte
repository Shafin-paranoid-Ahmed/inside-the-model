<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { POINTS, BEST, START, W_DOM, B_DOM, LR, STEPS, PATH, loss, gdPath, duration, markers, frameAt, T_GUESS, type Mode, type PathPt } from './sim';

  interface Props {
    title?: string;
    /** 'iterative': gradient descent walks to the answer. 'closed': the formula jumps straight there. */
    mode?: Mode;
    /** Show the residual squares on the data. */
    squares?: boolean;
  }
  let { title = '', mode = 'iterative', squares = true }: Props = $props();
  const init = untrack(() => ({ mode, squares, title: title || (mode === 'iterative' ? 'Fitting a line by walking downhill' : 'Fitting a line with a formula') }));
  const T = duration(init.mode), MK = markers(init.mode);
  const LOSS_MIN = loss(BEST.w, BEST.b);

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let over = $state<{ w: number; b: number } | null>(null);
  let path = $state.raw<PathPt[]>(PATH);
  let ui = $state({ w: START.w, b: START.b, loss: loss(START.w, START.b), k: 0 });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const phase = $derived(over ? 'manual' : (markerAt(uiT, MK)?.label ?? ''));
  const narration = $derived(
    phase === 'manual' ? 'You placed the line. Press the button below to start the fit from this spot.'
      : phase === 'a guess' ? 'A first guess: the line misses the points, so the loss is high.'
      : phase === 'descend' ? 'Each step moves (w, b) a little downhill on the loss surface. The line rotates and slides toward the points.'
      : phase === 'solve' ? 'The formula gives the bottom of the bowl directly. No steps needed.'
      : 'Settled at the minimum of the bowl: the least-squares line.',
  );
  $effect(() => { if (playing) over = null; speed; wake(); });

  const frame = () => {
    if (over) return { w: over.w, b: over.b, loss: loss(over.w, over.b), k: 0 };
    return frameAt(t, init.mode, path);
  };
  const place = (w: number, b: number) => { over = { w: clamp(w, W_DOM[0], W_DOM[1]), b: clamp(b, B_DOM[0], B_DOM[1]) }; playing = false; ui = { ...frame() }; wake(); };
  function launch() {
    const s = over ?? { w: ui.w, b: ui.b };
    path = init.mode === 'closed' ? [{ ...s, loss: loss(s.w, s.b) }] : gdPath(s, LR, STEPS);
    over = null; t = T_GUESS; playing = true; wake();
  }
  function reset() { path = PATH; over = null; t = 0; playing = true; wake(); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, dragging = false;
    const NX = 64, NY = 56;
    const grid = new Float64Array(NX * NY);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) grid[j * NX + i] = loss(W_DOM[0] + ((W_DOM[1] - W_DOM[0]) * i) / (NX - 1), B_DOM[0] + ((B_DOM[1] - B_DOM[0]) * j) / (NY - 1));
    const LEVELS = [0.4, 1, 2, 3.5, 5.5, 8, 12, 17, 24].map((d) => LOSS_MIN + d);
    let cache = { key: '', paths: [] as Path2D[] };
    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, 0.5);
      const d = { l: main.x + 34, r: main.x + main.w - 8, t: main.y + 10, b: main.y + main.h - 26 };
      const s = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 22, b: side.y + side.h - 30 };
      return { main, side, stacked, d, s, dx: scale(-3.6, 3.6, d.l, d.r), dy: scale(-4.5, 5, d.b, d.t), cw: scale(W_DOM[0], W_DOM[1], s.l, s.r), cb: scale(B_DOM[0], B_DOM[1], s.b, s.t) };
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

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frame();
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      const { d } = L;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [-3, 0, 3]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(x).replace('-', '−'), L.dx(x), d.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of [-4, -2, 0, 2, 4]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(d.l, L.dy(y)); ctx.lineTo(d.r, L.dy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y).replace('-', '−'), d.l - 6, L.dy(y)); }
      ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, d.r - d.l, d.b - d.t); ctx.clip();
      const ppu = Math.abs(L.dy(1) - L.dy(0));
      ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.7;
      ctx.beginPath(); ctx.moveTo(L.dx(-3.6), L.dy(BEST.w * -3.6 + BEST.b)); ctx.lineTo(L.dx(3.6), L.dy(BEST.w * 3.6 + BEST.b)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      for (const p of POINTS) {
        const yl = f.w * p.x + f.b, r = p.y - yl, col = r >= 0 ? c.pos : c.neg;
        if (init.squares) {
          const s = Math.abs(r) * ppu;
          ctx.fillStyle = `rgba(${col.rgb.join(',')},0.2)`; ctx.strokeStyle = `rgba(${col.rgb.join(',')},0.7)`; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.rect(L.dx(p.x), Math.min(L.dy(p.y), L.dy(yl)), s, s); ctx.fill(); ctx.stroke();
        }
        ctx.strokeStyle = col.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(L.dx(p.x), L.dy(p.y)); ctx.lineTo(L.dx(p.x), L.dy(yl)); ctx.stroke();
      }
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(L.dx(-3.6), L.dy(f.w * -3.6 + f.b)); ctx.lineTo(L.dx(3.6), L.dy(f.w * 3.6 + f.b)); ctx.stroke();
      for (const p of POINTS) { ctx.fillStyle = c.text.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(p.y), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', d.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y = w·x + b', d.l + 4, d.t + 2);

      const s = L.s;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('MSE over every (w, b)', s.l - 30, L.side.y + 2);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const w of [-1, 0, 1, 2, 3]) ctx.fillText(String(w).replace('-', '−'), L.cw(w), s.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const b of [-3, -2, -1, 0, 1, 2, 3]) ctx.fillText(String(b).replace('-', '−'), s.l - 6, L.cb(b));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('w (slope)', s.r, L.side.y + L.side.h - 1);
      ctx.save(); ctx.translate(L.side.x + 11, s.t); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('b (intercept)', 0, 0); ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(s.l, s.t, s.r - s.l, s.b - s.t);
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t, s.r - s.l, s.b - s.t); ctx.clip();
      contours(L).forEach((p, i) => { ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.35 + 0.07 * i; ctx.lineWidth = 1.3; ctx.stroke(p); });
      ctx.globalAlpha = 1;
      if (!over) {
        const k = init.mode === 'iterative' ? f.k : 0;
        ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(L.cw(path[0].w), L.cb(path[0].b));
        if (init.mode === 'iterative') { for (let i = 1; i <= Math.floor(k); i++) ctx.lineTo(L.cw(path[i].w), L.cb(path[i].b)); }
        ctx.lineTo(L.cw(f.w), L.cb(f.b)); ctx.stroke();
        if (init.mode === 'iterative') for (let i = 0; i <= Math.floor(k); i++) { ctx.fillStyle = c.grad.hex; ctx.beginPath(); ctx.arc(L.cw(path[i].w), L.cb(path[i].b), 2.5, 0, 6.2832); ctx.fill(); }
      }
      ctx.restore();
      const bx = L.cw(BEST.w), by = L.cb(BEST.b);
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx, by - 7); ctx.lineTo(bx + 7, by); ctx.lineTo(bx, by + 7); ctx.lineTo(bx - 7, by); ctx.closePath(); ctx.stroke();
      ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(L.cw(f.w), L.cb(f.b), 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; ui = { ...frame() }; uiT = t; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const inSide = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top; return { L, x, y, ok: x >= L.s.l - 8 && x <= L.s.r + 8 && y >= L.s.t - 8 && y <= L.s.b + 8 }; };
    const put = (e: PointerEvent) => { const { L, x, y } = inSide(e); place(W_DOM[0] + ((x - L.s.l) / (L.s.r - L.s.l)) * (W_DOM[1] - W_DOM[0]), B_DOM[0] + ((y - L.s.b) / (L.s.t - L.s.b)) * (B_DOM[1] - B_DOM[0])); };
    const down = (e: PointerEvent) => { if (inSide(e).ok) { dragging = true; cv.setPointerCapture(e.pointerId); put(e); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (dragging) put(e); else cv.style.cursor = inSide(e).ok ? 'crosshair' : ''; };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 400" narrowAspect="360 / 620"
  label="Left: twelve points with a line and squares showing each error. Right: contour map of the mean squared error over slope w and intercept b, with a dot for the current line, a diamond at the exact minimum, and the path taken."
  legend={[{ label: 'current line (w, b)', color: '--viz-forward', shape: 'line' }, { label: 'path', color: '--viz-gradient', shape: 'line' }, { label: 'exact minimum', color: '--accent' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: fitting the line"
      onSeek={(to) => { over = null; t = to; wake(); }} onStep={(to) => { over = null; t = to >= T ? 0 : to; wake(); }} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">w = <b>{f2(ui.w)}</b> · b = <b>{f2(ui.b)}</b> · MSE = <b>{f2(ui.loss, 3)}</b>
      <span class="base">· exact minimum: w = {f2(BEST.w)}, b = {f2(BEST.b)}, MSE = {f2(LOSS_MIN, 3)}</span></p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="w" value={ui.w} min={W_DOM[0]} max={W_DOM[1]} step={0.01} oninput={(v) => place(v, ui.b)} />
      <F1Slider label="b" value={ui.b} min={B_DOM[0]} max={B_DOM[1]} step={0.01} oninput={(v) => place(ui.w, v)} />
      <button type="button" class="btn" onclick={launch}>{init.mode === 'iterative' ? 'Descend from here' : 'Solve from here'}</button>
      <span class="hint">You can also click or drag on the contour map.</span>
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
