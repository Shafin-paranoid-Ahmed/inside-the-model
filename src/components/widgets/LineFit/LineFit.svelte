<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { POINTS, BEST, START, W_DOM, B_DOM, LR, STEPS, PATH, loss, calculation, gdPath, duration, markers, frameAt, T_GUESS, type Mode, type PathPt } from './sim';

  interface Props {
    title?: string;
    /** 'iterative': gradient descent walks to the answer. 'closed': the formula jumps straight there. 'tour': the dot visits many lines. */
    mode?: Mode;
    /** Show the residual squares on the data. */
    squares?: boolean;
  }
  let { title = '', mode = 'iterative', squares = true }: Props = $props();
  const init = untrack(() => ({ mode, squares, title: title || (mode === 'iterative' ? 'Fitting a line by walking downhill' : mode === 'tour' ? 'Every point on the map is a line' : 'Fitting a line with a formula') }));
  const T = duration(init.mode), MK = markers(init.mode);
  const LOSS_MIN = loss(BEST.w, BEST.b);
  const simple = init.mode === 'prediction' || init.mode === 'errors';

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  /** True after the user placed a guess, until the timeline plays again. */
  let manual = $state(false);
  /** path[0] is the start: the default guess, or wherever the user put the dot. Every mode replays from it. */
  let path = $state.raw<PathPt[]>(PATH);
  let ui = $state({ w: START.w, b: START.b, loss: loss(START.w, START.b), k: 0 });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  const calc = $derived(calculation(ui.w, ui.b));
  const example = $derived(calc.rows[7]);
  const before = $derived(calculation(path[0].w, path[0].b));

  const phase = $derived(manual ? 'manual' : (markerAt(uiT, MK)?.label ?? ''));
  const narration = $derived(
    phase === 'manual' ? 'Your chosen line. The calculations below use these same w and b values.'
      : simple ? (phase === 'start' ? (init.mode === 'prediction' ? 'Read the starting prediction. The outlined point is example 8.' : 'Each bar is one squared error. The dashed line is their average: MSE.') : phase === 'raise b' ? 'Only b increases: every prediction rises by the same amount.' : `b has increased by ${f2(ui.b - path[0].b)}, so every prediction is ${f2(ui.b - path[0].b)} higher.`)
      : init.mode === 'update' ? (uiT < 4 ? 'Read both MSE slopes at the starting line, then calculate the two new parameters below.' : uiT < 8 ? 'One update: the line moves from the old parameters to the new ones.' : 'One update finished. Compare the new MSE with the starting MSE below.')
      : phase === 'a bad line' ? 'The dot starts far from the diamond, so its line misses the points badly. Then it heads to other corners of the map.'
      : phase === 'other lines' ? 'The dot visits other corners of the map. Each spot is a different line, with its own MSE.'
      : phase === 'approach best' ? (uiT < 10 ? 'Approaching the diamond: the squares shrink as the line improves.' : 'At the diamond: the line with the smallest MSE of all.')
      : phase === 'a guess' ? 'A first guess: the line misses the points, so the loss is high.'
      : phase === 'descend' ? 'Each step moves (w, b) a little downhill on the loss surface. The line rotates and slides toward the points.'
      : phase === 'solve' ? 'The formula gives the bottom of the bowl directly. No steps needed.'
      : 'Near the minimum: compare the current MSE with the exact minimum.',
  );
  $effect(() => { if (playing) manual = false; speed; wake(); });

  const frame = () => frameAt(t, init.mode, path);
  /** A new guess becomes the start of every replay: recompute the descent from it and rewind to the guess. */
  const place = (w: number, b: number) => {
    const s = { w: clamp(w, W_DOM[0], W_DOM[1]), b: clamp(b, B_DOM[0], B_DOM[1]) };
    path = init.mode === 'iterative' ? gdPath(s, LR, STEPS) : [{ ...s, loss: loss(s.w, s.b) }];
    t = 0; manual = true; playing = false; ui = { ...frame() }; uiT = 0; wake();
  };
  function launch() { manual = false; t = init.mode === 'tour' || init.mode === 'update' ? 0 : T_GUESS; playing = true; wake(); }
  function reset() { path = PATH; manual = false; t = 0; playing = !reducedMotion(); wake(); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let dragging = false;
    const NX = 64, NY = 56;
    const grid = new Float64Array(NX * NY);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) grid[j * NX + i] = loss(W_DOM[0] + ((W_DOM[1] - W_DOM[0]) * i) / (NX - 1), B_DOM[0] + ((B_DOM[1] - B_DOM[0]) * j) / (NY - 1));
    const LEVELS = [0.4, 1, 2, 3.5, 5.5, 8, 12, 17, 24].map((d) => LOSS_MIN + d);
    let cache = { key: '', paths: [] as Path2D[] };
    const lay = () => {
      const panels = splitPanels(stage.w, stage.h, 0.5);
      const { side, stacked } = panels;
      const main = simple ? { x: 0, y: 0, w: stage.w, h: stage.h } : panels.main;
      const d = { l: main.x + 34, r: main.x + main.w - 8, t: main.y + 10, b: main.y + main.h - 26 };
      const s = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 22, b: side.y + side.h - 30 };
      const f = frame(), selectedY = f.w * POINTS[7].x + f.b;
      const yMin = simple ? Math.min(-4.5, Math.floor(selectedY - 0.5)) : -4.5;
      const yMax = simple ? Math.max(5, Math.ceil(selectedY + 0.5)) : 5;
      return { main, side, stacked, d, s, yMin, yMax, dx: scale(-3.6, 3.6, d.l, d.r), dy: scale(yMin, yMax, d.b, d.t), cw: scale(W_DOM[0], W_DOM[1], s.l, s.r), cb: scale(B_DOM[0], B_DOM[1], s.b, s.t) };
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
      if (init.mode === 'errors') {
        const values = calculation(f.w, f.b), initial = calculation(path[0].w, path[0].b);
        const top = Math.ceil(Math.max(...initial.rows.map((p) => p.squared), ...values.rows.map((p) => p.squared)) / 10) * 10 || 1;
        const left = 38, right = W - 12, floor = H - 32, ceiling = 28;
        const y = scale(0, top, floor, ceiling), cell = (right - left) / POINTS.length;
        ctx.font = stage.body(12); ctx.fillStyle = c.text.hex;
        ctx.textAlign = 'left'; ctx.fillText('Squared error for each example', left, 15);
        for (const v of [0, top / 2, top]) {
          ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(right, y(v)); ctx.stroke();
          ctx.textAlign = 'right'; ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), left - 5, y(v) + 4);
        }
        values.rows.forEach((row, i) => {
          const x = left + i * cell + 3;
          ctx.fillStyle = i === 7 ? c.accent.hex : c.fwd.hex;
          ctx.fillRect(x, y(row.squared), Math.max(2, cell - 6), floor - y(row.squared));
          ctx.textAlign = 'center'; ctx.fillStyle = c.text.hex; ctx.fillText(String(i + 1), x + (cell - 6) / 2, floor + 17);
        });
        ctx.setLineDash([4, 4]); ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(left, y(values.mse)); ctx.lineTo(right, y(values.mse)); ctx.stroke(); ctx.setLineDash([]);
        ctx.textAlign = 'right'; ctx.fillStyle = c.grad.hex; ctx.fillText(`MSE ${f2(values.mse, 3)}`, right, y(values.mse) - 7);
        return;
      }
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      const { d } = L;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [-3, 0, 3]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(x).replace('-', '−'), L.dx(x), d.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of simple ? [Math.ceil(L.yMin), 0, L.yMax] : [-4, -2, 0, 2, 4]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(d.l, L.dy(y)); ctx.lineTo(d.r, L.dy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y).replace('-', '−'), d.l - 6, L.dy(y)); }
      ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, d.r - d.l, d.b - d.t); ctx.clip();
      const ppu = Math.abs(L.dy(1) - L.dy(0));
      ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.globalAlpha = simple ? 0 : 0.7;
      ctx.beginPath(); ctx.moveTo(L.dx(-3.6), L.dy(BEST.w * -3.6 + BEST.b)); ctx.lineTo(L.dx(3.6), L.dy(BEST.w * 3.6 + BEST.b)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      for (const p of POINTS) {
        if (init.mode === 'prediction' && p !== POINTS[7]) continue;
        const yl = f.w * p.x + f.b, r = p.y - yl, col = r >= 0 ? c.pos : c.neg;
        if (init.squares && init.mode !== 'prediction') {
          const s = Math.abs(r) * ppu;
          ctx.fillStyle = `rgba(${col.rgb.join(',')},0.2)`; ctx.strokeStyle = `rgba(${col.rgb.join(',')},0.7)`; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.rect(L.dx(p.x), Math.min(L.dy(p.y), L.dy(yl)), s, s); ctx.fill(); ctx.stroke();
        }
        ctx.strokeStyle = col.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(L.dx(p.x), L.dy(p.y)); ctx.lineTo(L.dx(p.x), L.dy(yl)); ctx.stroke();
      }
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(L.dx(-3.6), L.dy(f.w * -3.6 + f.b)); ctx.lineTo(L.dx(3.6), L.dy(f.w * 3.6 + f.b)); ctx.stroke();
      for (const p of POINTS) { ctx.fillStyle = c.text.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(p.y), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      if (simple) {
        const p = POINTS[7];
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(p.y), 8, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = c.fwd.hex;
        ctx.beginPath(); ctx.arc(L.dx(p.x), L.dy(f.w * p.x + f.b), 5, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', d.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y = w·x + b', d.l + 4, d.t + 2);

      if (simple) return;

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
      if (init.mode !== 'tour' && t >= T_GUESS) {
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

    const loop = createLoop(cv, (dt) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      ui = { ...frame() }; uiT = t;
      render(); tl.paint(t);
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const inSide = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top; return { L, x, y, ok: !simple && x >= L.s.l - 8 && x <= L.s.r + 8 && y >= L.s.t - 8 && y <= L.s.b + 8 }; };
    const put = (e: PointerEvent) => { const { L, x, y } = inSide(e); place(W_DOM[0] + ((x - L.s.l) / (L.s.r - L.s.l)) * (W_DOM[1] - W_DOM[0]), B_DOM[0] + ((y - L.s.b) / (L.s.t - L.s.b)) * (B_DOM[1] - B_DOM[0])); };
    const down = (e: PointerEvent) => { if (inSide(e).ok) { dragging = true; cv.setPointerCapture(e.pointerId); put(e); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (dragging) put(e); else cv.style.cursor = inSide(e).ok ? 'crosshair' : ''; };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect={simple ? '720 / 330' : '720 / 400'} narrowAspect={simple ? '360 / 360' : '360 / 620'}
  label={init.mode === 'errors' ? 'Twelve squared-error bars, one per example. A dashed horizontal line marks their average, the MSE.' : simple ? 'Twelve observed points and the current prediction line. Example 8 is outlined; its prediction is marked on the line.' : 'Left: data and prediction line. Right: MSE contour map over slope w and intercept b, with the current line marked by a dot and the exact minimum by a diamond.'}
  legend={init.mode === 'errors' ? [{ label: 'squared error', color: '--viz-forward' }, { label: 'example 8', color: '--accent' }, { label: 'MSE', color: '--viz-gradient', shape: 'line' }] : simple ? [{ label: 'prediction line', color: '--viz-forward', shape: 'line' }, { label: 'example 8 (outlined)', color: '--accent' }] : [{ label: 'current line (w, b)', color: '--viz-forward', shape: 'line' }, { label: 'path', color: '--viz-gradient', shape: 'line' }, { label: 'exact minimum', color: '--accent' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: fitting the line"
      onSeek={(to) => { manual = false; t = to; wake(); }} onStep={(to) => { manual = false; t = to >= T ? 0 : to; wake(); }} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">w = <b>{f2(ui.w)}</b> · b = <b>{f2(ui.b)}</b>{#if init.mode !== 'prediction'} · MSE = <b>{f2(ui.loss, 3)}</b>{/if}
      {#if !simple}<span class="base">· exact minimum: w = {f2(BEST.w)}, b = {f2(BEST.b)}, MSE = {f2(LOSS_MIN, 3)}</span>{/if}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
    {#if simple}
      <div class="worked">
        <p><strong>Example 8:</strong> input x ≈ {f2(example.x, 6)}; observed y ≈ {f2(example.y, 6)}.</p>
        <p>Prediction = w × x + b<br />≈ ({f2(ui.w, 6)}) × ({f2(example.x, 6)}) + ({f2(ui.b, 6)})<br />≈ <strong>{f2(example.prediction, 6)}</strong>.</p>
        {#if init.mode === 'errors'}
          <p>Error = prediction − observed<br />≈ {f2(example.prediction, 6)} − ({f2(example.y, 6)})<br />≈ <strong>{f2(example.error, 6)}</strong>.</p>
          <p>Squared error ≈ ({f2(example.error, 6)})² ≈ <strong>{f2(example.squared, 3)}</strong>.</p>
          <p><strong>MSE</strong> = sum of all 12 squared errors ÷ 12<br />≈ {f2(calc.sum, 3)} ÷ 12 ≈ <strong>{f2(calc.mse, 3)}</strong>.</p>
          <details><summary>See all 12 predictions and squared errors</summary>
            <div class="table-wrap"><table><thead><tr><th scope="col">x</th><th scope="col">y</th><th scope="col">Prediction</th><th scope="col">Error²</th></tr></thead><tbody>
              {#each calc.rows as row, i (i)}<tr class:selected={i === 7}><td>{f2(row.x, 3)}</td><td>{f2(row.y, 3)}</td><td>{f2(row.prediction, 3)}</td><td>{f2(row.squared, 3)}</td></tr>{/each}
            </tbody></table></div>
          </details>
        {/if}
        <p class="rounding">Displayed values are rounded; ≈ means approximately equal.</p>
      </div>
    {:else if init.mode === 'update'}
      <div class="worked">
        <p><strong>One update from the starting line</strong> (learning rate {LR})</p>
        <p>Slope of MSE with respect to w:<br />2 ÷ 12 × ({f2(before.sumErrorX, 3)}) ≈ <strong>{f2(before.gw, 3)}</strong>.</p>
        <p>Slope of MSE with respect to b:<br />2 ÷ 12 × ({f2(before.sumError, 3)}) ≈ <strong>{f2(before.gb, 3)}</strong>.</p>
        <p>New w = {f2(path[0].w, 3)} − {LR} × ({f2(before.gw, 3)}) ≈ <strong>{f2(before.next.w, 3)}</strong>.</p>
        <p>New b = {f2(path[0].b, 3)} − {LR} × ({f2(before.gb, 3)}) ≈ <strong>{f2(before.next.b, 3)}</strong>.</p>
        <p>Starting MSE: <strong>{f2(before.mse, 3)}</strong>. MSE after this update: <strong>{f2(before.next.loss, 3)}</strong>.</p>
        <p class="rounding">The arithmetic previews the destination. The plot and current MSE above follow the movement.</p>
      </div>
    {:else if init.mode === 'iterative'}
      <p class="say">Updates completed: {Math.floor(ui.k)} / {STEPS}. Same data, same MSE, new w and b.</p>
    {/if}
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="w" value={ui.w} min={W_DOM[0]} max={W_DOM[1]} step={0.01} oninput={(v) => place(v, ui.b)} />
      <F1Slider label="b" value={ui.b} min={B_DOM[0]} max={B_DOM[1]} step={0.01} oninput={(v) => place(ui.w, v)} />
      {#if !simple && init.mode !== 'tour'}<button type="button" class="btn" onclick={launch}>{init.mode === 'closed' ? 'Solve from here' : init.mode === 'update' ? 'One update from here' : 'Descend from here'}</button>{/if}
      <span class="hint">{simple ? 'Change w or b to try a different line.' : 'Click or drag on the contour map to place your own guess.'}</span>
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
  .worked { margin-top: var(--s-3); font: var(--step--1) / 1.6 var(--font-ui); font-variant-numeric: tabular-nums; }
  .worked p + p { margin-top: var(--s-2); }
  .rounding { color: var(--text-muted); font-size: 0.8rem; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th, td { padding: 0.3em; text-align: right; border-bottom: 1px solid var(--rule); }
  .selected { background: var(--bg-sunken); font-weight: 650; }
  summary { cursor: pointer; margin-block: var(--s-2); }
</style>
