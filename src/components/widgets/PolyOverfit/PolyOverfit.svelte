<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { BASE, GX, GT, N_GHOST, D_MAX, BIAS2, VAR, NOISE, T_TOTAL, MARKERS, SEED0, fitModel, timeOfDegree, frameAt, frameAtDegree, ghostAt, type Model } from './sim';

  interface Props {
    title?: string;
    /** Draw bias² and variance (over many datasets) next to train and test error. */
    decompose?: boolean;
    /** Start with the fits from 20 other datasets drawn faintly behind the current one. */
    ghosts?: boolean;
    /** Start paused at this degree (1 to 12) instead of sweeping. */
    degree?: number;
  }
  let { title = 'Polynomial fits of growing degree', decompose = false, ghosts = false, degree }: Props = $props();
  const init = untrack(() => ({ decompose, degree, title }));

  let t = untrack(() => (init.degree != null ? timeOfDegree(init.degree) : 0));
  let playing = $state(false), speed = $state(1), uiT = $state(untrack(() => t));
  let showGhosts = $state(untrack(() => ghosts));
  let over = $state<number | null>(init.degree ?? null);
  let seed = $state(SEED0);
  let ui = $state(untrack(() => (init.degree != null ? frameAtDegree(init.degree) : frameAt(0))));
  let model: Model = BASE;
  let wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const phase = $derived(over != null ? ui.phase : (markerAt(uiT, MARKERS)?.label ?? 'Underfit'));
  const narration = $derived(
    ui.phase === 'Underfit' ? 'Too rigid: the curve cannot bend enough, so it misses the pattern. Train and test error are both high (bias).'
      : ui.phase === 'Good fit' ? 'About the right flexibility: it follows the trend without chasing individual points. Test error is near its lowest.'
      : 'Too flexible: it threads through the training points and swings wildly between them. Train error keeps falling, test error explodes (variance).',
  );
  $effect(() => { if (playing) over = null; speed; showGhosts; wake(); });

  const frame = () => (over != null ? frameAtDegree(over, model) : frameAt(t, model));
  const setDegree = (d: number) => { over = clamp(d, 1, D_MAX); playing = false; ui = frame(); wake(); };
  function resample() { seed += 1; model = fitModel(seed); ui = frame(); wake(); }
  function reset() { seed = SEED0; model = BASE; over = null; t = 0; playing = true; wake(); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, dragging = false;
    const lay = () => {
      const { main, side } = splitPanels(stage.w, stage.h, 0.56);
      const d = { l: main.x + 30, r: main.x + main.w - 6, t: main.y + 10, b: main.y + main.h - 24 };
      const s = { l: side.x + 40, r: side.x + side.w - 12, t: side.y + 26, b: side.y + side.h - 32 };
      const lg = (v: number) => Math.log10(clamp(v, 0.001, 100));
      const cy = scale(-3, 2, s.b, s.t);
      return { main, side, d, s, dx: scale(-1, 1, d.l, d.r), dy: scale(-1.7, 1.7, d.b, d.t), cx: scale(1, D_MAX, s.l, s.r), cy: (v: number) => cy(lg(v)) };
    };
    const curvePath = (L: ReturnType<typeof lay>, y: Float64Array) => {
      const p = new Path2D();
      for (let g = 0; g < GX.length; g++) { const X = L.dx(GX[g]), Y = L.dy(y[g]); if (g) p.lineTo(X, Y); else p.moveTo(X, Y); }
      return p;
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frame(), { d, s } = L;
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // main panel: data and fit
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [-1, 0, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(x).replace('-', '−'), L.dx(x), d.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of [-1, 0, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(d.l, L.dy(y)); ctx.lineTo(d.r, L.dy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y).replace('-', '−'), d.l - 6, L.dy(y)); }
      ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, d.r - d.l, d.b - d.t); ctx.clip();
      if (showGhosts) {
        ctx.strokeStyle = c.s3.hex; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.3;
        for (let k = 0; k < N_GHOST; k++) ctx.stroke(curvePath(L, ghostAt(k, f.d)));
        ctx.globalAlpha = 1;
      }
      ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.8; ctx.stroke(curvePath(L, GT)); ctx.setLineDash([]); ctx.globalAlpha = 1;
      const yAt = (x: number) => { const u = (x + 1) * 60, i = clamp(Math.floor(u), 0, 119), fr = u - i; return f.curve[i] * (1 - fr) + f.curve[i + 1] * fr; };
      for (let i = 0; i < model.xs.length; i++) {
        const x = model.xs[i], y = model.ys[i], yl = yAt(x);
        ctx.strokeStyle = (y >= yl ? c.pos : c.neg).hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(L.dx(x), L.dy(y)); ctx.lineTo(L.dx(x), L.dy(yl)); ctx.stroke();
      }
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.stroke(curvePath(L, f.curve));
      for (let i = 0; i < model.xs.length; i++) { ctx.fillStyle = c.text.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(L.dx(model.xs[i]), L.dy(model.ys[i]), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', d.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`degree ${f2(f.d, 1)} polynomial · 14 points`, d.l + 4, d.t + 2);

      // side panel: error vs degree (log scale)
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Error (mean squared, log scale) vs degree', s.l - 30, L.side.y + 2);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const k of [1, 3, 5, 7, 9, 12]) ctx.fillText(String(k), L.cx(k), s.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const [v, lb] of [[0.001, '.001'], [0.01, '.01'], [0.1, '0.1'], [1, '1'], [10, '10'], [100, '100']] as [number, string][]) {
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(s.l, L.cy(v)); ctx.lineTo(s.r, L.cy(v)); ctx.stroke(); ctx.fillText(lb, s.l - 6, L.cy(v));
      }
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('degree', s.r, L.side.y + L.side.h - 1);
      const line = (vals: number[], col: string, w: number, dash: number[] = []) => {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.lineJoin = 'round'; ctx.beginPath();
        vals.forEach((v, i) => (i ? ctx.lineTo(L.cx(i + 1), L.cy(v)) : ctx.moveTo(L.cx(i + 1), L.cy(v)))); ctx.stroke(); ctx.setLineDash([]);
      };
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t, s.r - s.l, s.b - s.t); ctx.clip();
      ctx.globalAlpha = 0.85; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(s.l, L.cy(NOISE)); ctx.lineTo(s.r, L.cy(NOISE)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      if (init.decompose) { line(BIAS2, c.s3.hex, 1.6, [6, 4]); line(VAR, c.s5.hex, 1.6, [2, 3]); }
      line(model.train, c.fwd.hex, 2.4); line(model.test, c.grad.hex, 2.4);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.cx(f.d), s.t); ctx.lineTo(L.cx(f.d), s.b); ctx.stroke();
      for (const [v, col] of [[f.train, c.fwd.hex], [f.test, c.grad.hex]] as [number, string][]) { ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(L.cx(f.d), L.cy(v), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(s.l, s.t, s.r - s.l, s.b - s.t);
      ctx.font = stage.body(11.5); ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
      ctx.fillStyle = c.fwd.hex; ctx.textAlign = 'right'; ctx.fillText('train', L.cx(D_MAX) - 6, L.cy(model.train[D_MAX - 1]) - 6);
      ctx.fillStyle = c.grad.hex; ctx.fillText('test', L.cx(D_MAX) - 6, s.t + 14);
      ctx.fillStyle = c.accent.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('noise floor', L.cx(D_MAX) - 6, L.cy(NOISE) + 3);
      if (init.decompose) {
        ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
        ctx.fillStyle = c.s5.hex; ctx.fillText('variance', L.cx(D_MAX - 1) - 6, L.cy(VAR[D_MAX - 2]) - 4);
        ctx.fillStyle = c.s3.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('bias²', L.cx(1) + 6, L.cy(BIAS2[0]) + 4);
      }
    }

    const loop = createLoop(cv, (dt, now) => {
      if (over == null) t = advance({ t, playing, speed }, dt, { duration: T_TOTAL }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; ui = frame(); uiT = t; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion() && init.degree == null) playing = true; } });
    wake = loop.wake;

    const inSide = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top; return { L, x, ok: x >= L.s.l - 8 && x <= L.s.r + 8 && y >= L.s.t - 8 && y <= L.s.b + 8 }; };
    const put = (e: PointerEvent) => { const { L, x } = inSide(e); setDegree(1 + ((x - L.s.l) / (L.s.r - L.s.l)) * (D_MAX - 1)); };
    const down = (e: PointerEvent) => { if (inSide(e).ok) { dragging = true; cv.setPointerCapture(e.pointerId); put(e); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (dragging) put(e); else cv.style.cursor = inSide(e).ok ? 'ew-resize' : ''; };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 400" narrowAspect="360 / 620"
  label="Left: fourteen noisy points, the true curve dashed, and a fitted polynomial whose degree changes. Right: training and test error against degree; training error keeps falling while test error is U-shaped."
  legend={[{ label: 'fitted polynomial / train error', color: '--viz-forward', shape: 'line' }, { label: 'test error', color: '--viz-gradient', shape: 'line' }, { label: 'true curve / noise floor', color: '--accent', shape: 'line' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T_TOTAL} markers={MARKERS} label="Timeline: raising the degree"
      onSeek={(to) => { over = null; t = to; wake(); }} onStep={(to) => { over = null; t = to >= T_TOTAL ? 0 : to; wake(); }} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">degree = <b>{f2(ui.d, 1)}</b> · train = <b>{f2(ui.train, 3)}</b> · test = <b class="te">{f2(ui.test, 3)}</b>
      {#if init.decompose}<span class="base">· bias² = {f2(ui.bias2, 3)} · variance = {f2(ui.variance, 3)} · noise = {f2(NOISE, 3)}</span>{/if}</p>
    <p class="say" role="status" aria-live="polite">{phase}: {narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="degree" value={ui.d} min={1} max={D_MAX} step={0.05} digits={1} oninput={setDegree} />
      <button type="button" class="btn" onclick={resample}>New data</button>
      <label class="chk"><input type="checkbox" bind:checked={showGhosts} /> Show fits from 20 other datasets</label>
      <span class="hint">Click or drag on the error chart to pick a degree.</span>
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .vals b.te { color: var(--viz-gradient); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); font-size: 0.85rem; }
  .chk input { accent-color: var(--accent); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
