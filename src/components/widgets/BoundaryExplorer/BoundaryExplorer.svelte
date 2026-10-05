<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { createRng } from '../../../lib/math/rng';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour } from '../F1Shared/plot';
  import {
    DOM, STEPS, T, MARKERS, makeDataset, train, frameAt, field, embed, evaluate, prob, score, signedDistance,
    type Dataset, type Model, type Pt, type Run,
  } from './sim';

  interface Props {
    title?: string;
    dataset?: Dataset;
    /** 'linear': features x, y. 'poly': adds x², xy, y². */
    model?: Model;
    /** Show the dataset / model pickers and the add-point tools. */
    editable?: boolean;
    /** Draw labelled score contours z = −2…2 instead of the 12% / 88% probability contours. */
    scores?: boolean;
  }
  let { title = 'A classifier learning where to cut the plane', dataset = 'linear', model = 'linear', editable = true, scores = false }: Props = $props();
  const init = untrack(() => ({ dataset, model }));

  let ds = $state<Dataset>(init.dataset), mdl = $state<Model>(init.model), cls = $state<0 | 1>(1);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let ui = $state({ k: 0, loss: Math.LN2, acc: 0.5, phase: 'start', extra: 0 });
  let hov = $state<{ x: number; y: number; p: number; z: number; d: number } | null>(null);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  let pts: Pt[] = makeDataset(init.dataset), run: Run = train(pts, init.model), extra = 0;
  let hover: { x: number; y: number } | null = null;

  const narration = $derived(
    ui.phase === 'start' ? 'Weights start at zero, so every spot on the plane gets probability 0.5.'
      : ui.phase === 'learning' ? 'Each gradient step tilts the probability field. The black contour, where the probability is 0.5, is the decision boundary.'
      : `Settled: ${Math.round(ui.acc * 100)}% of the points are on the correct side.`,
  );
  $effect(() => { playing; speed; wake(); });

  const restart = (w0?: ArrayLike<number>) => { run = train(pts, mdl, w0); t = 0; playing = true; wake(); };
  function setData(d: Dataset) { ds = d; pts = makeDataset(d); extra = 0; run = train(pts, mdl); t = 0; playing = true; wake(); }
  function setModel(m: Model) { mdl = m; restart(embed(frameAt(t, run).w, m)); }
  function addPoint(x: number, y: number, c: 0 | 1) { pts = [...pts, { x, y, c }]; extra++; restart(frameAt(t, run).w); }
  function addRandom() { const r = createRng(101 + extra); addPoint(r.uniform(-1.3, 1.3), r.uniform(-1.3, 1.3), cls); }
  function reset() { setData(ds); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const N = 72, grid = new Float32Array(N * N), off = document.createElement('canvas');
    off.width = N; off.height = N;
    const octx = off.getContext('2d')!, img = octx.createImageData(N, N), px = new Uint32Array(img.data.buffer);
    let lastUi = 0;
    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, 0.6);
      const s = Math.max(40, Math.min(main.w - 40, main.h - 30));
      const px0 = main.x + 28 + (main.w - 28 - s) / 2, py0 = main.y + 6;
      const sd = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 30, b: side.y + side.h - 30 };
      return { main, side, stacked, px0, py0, s, sd, gx: scale(-DOM, DOM, px0, px0 + s), gy: scale(-DOM, DOM, py0 + s, py0) };
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frameAt(t, run);
      ctx.clearRect(0, 0, W, H);
      field(run.model, f.w, N, grid);
      const [pr, pg, pb] = c.pos.rgb, [nr, ng, nb] = c.neg.rgb, [br, bg, bb] = c.raised.rgb;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const p = grid[j * N + i], a = Math.min(0.62, Math.abs(2 * p - 1) * 0.7), [r, g, b] = p > 0.5 ? [pr, pg, pb] : [nr, ng, nb];
        px[(N - 1 - j) * N + i] = 0xff000000 | (Math.round(bb + (b - bb) * a) << 16) | (Math.round(bg + (g - bg) * a) << 8) | Math.round(br + (r - br) * a);
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, L.px0, L.py0, L.s, L.s);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(L.px0, L.py0, L.s, L.s);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-1, 0, 1]) ctx.fillText(String(v).replace('-', '−'), L.gx(v), L.py0 + L.s + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [-1, 0, 1]) ctx.fillText(String(v).replace('-', '−'), L.px0 - 5, L.gy(v));

      ctx.save(); ctx.beginPath(); ctx.rect(L.px0, L.py0, L.s, L.s); ctx.clip();
      const cell = L.s / N, at = (i: number, j: number) => grid[j * N + i];
      const stroke = (level: number, width: number, color: string, dash: number[], alpha: number) => {
        const seg = contour(at, N, N, level);
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.globalAlpha = alpha; ctx.beginPath();
        for (let k = 0; k < seg.length; k += 4) {
          ctx.moveTo(L.px0 + (seg[k] + 0.5) * cell, L.py0 + L.s - (seg[k + 1] + 0.5) * cell);
          ctx.lineTo(L.px0 + (seg[k + 2] + 0.5) * cell, L.py0 + L.s - (seg[k + 3] + 0.5) * cell);
        }
        ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        return seg;
      };
      const labels: [number, number, string][] = [];
      if (scores) {
        for (const z of [-2, -1, 0, 1, 2]) {
          const seg = z ? stroke(1 / (1 + Math.exp(-z)), 1.2, c.muted.hex, [], 0.9) : stroke(0.5, 2.6, c.text.hex, [], 1);
          // label the top end of each line, away from the data in the middle
          let k = -1;
          for (let q = 0; q < seg.length; q += 4) if (k < 0 || seg[q + 1] > seg[k + 1]) k = q;
          if (k >= 0) labels.push([
            Math.min(L.px0 + L.s - 26, Math.max(L.px0 + 26, L.px0 + (seg[k] + 0.5) * cell)),
            Math.max(L.py0 + 14, L.py0 + L.s - (seg[k + 1] + 0.5) * cell + 14), 'z = ' + String(z).replace('-', '−')]);
        }
      } else {
        stroke(0.12, 1, c.muted.hex, [3, 4], 0.7); stroke(0.88, 1, c.muted.hex, [3, 4], 0.7);
        stroke(0.5, 2.6, c.text.hex, [], 1);
      }
      for (const p of pts) {
        const x = L.gx(p.x), y = L.gy(p.y), wrong = (score(run.model, f.w, p.x, p.y) > 0 ? 1 : 0) !== p.c;
        ctx.fillStyle = p.c ? c.pos.hex : c.neg.hex; ctx.strokeStyle = wrong ? c.text.hex : c.raised.hex; ctx.lineWidth = wrong ? 2 : 1.5;
        ctx.beginPath();
        if (p.c) ctx.arc(x, y, 4.6, 0, 6.2832); else { ctx.moveTo(x, y - 5.6); ctx.lineTo(x + 5.6, y); ctx.lineTo(x, y + 5.6); ctx.lineTo(x - 5.6, y); ctx.closePath(); }
        ctx.fill(); ctx.stroke();
      }
      ctx.font = stage.mono(11, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const [x, y, txt] of labels) {
        const w = ctx.measureText(txt).width + 8;
        ctx.fillStyle = c.raised.hex; ctx.globalAlpha = 0.85; ctx.fillRect(x - w / 2, y - 8, w, 16); ctx.globalAlpha = 1;
        ctx.fillStyle = c.text.hex; ctx.fillText(txt, x, y);
      }
      if (hover) {
        const x = L.gx(hover.x), y = L.gy(hover.y);
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.2832); ctx.stroke();
      }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('x', L.px0 + L.s, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y', L.px0 - 24, L.py0 - 4);

      // side panel: loss over training steps, with the playhead
      const sd = L.sd, mx = Math.max(0.75, run.loss[0]) * 1.05;
      const sx = scale(0, STEPS, sd.l, sd.r), sy = scale(0, mx, sd.b, sd.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Loss while training', sd.l - 30, L.side.y + 4);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.25, 0.5, 0.75]) if (v < mx) ctx.fillText(f2(v), sd.l - 6, sy(v));
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 100, 200]) ctx.fillText(String(v), sx(v), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('gradient step', sd.r, L.side.y + L.side.h - 1);
      ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let k = 0; k <= f.k; k++) { const X = sx(k), Y = sy(run.loss[k]); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.lineTo(sx(f.k), sy(f.loss)); ctx.stroke();
      ctx.fillStyle = c.grad.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx(f.k), sy(f.loss), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    const read = () => {
      const f = frameAt(t, run), e = evaluate(run.model, f.w, pts);
      ui = { k: f.k, loss: f.loss, acc: e.acc, phase: f.phase, extra };
      if (hover) { const z = score(run.model, f.w, hover.x, hover.y); hov = { ...hover, z, p: prob(run.model, f.w, hover.x, hover.y), d: signedDistance(f.w, hover.x, hover.y) }; } else hov = null;
    };
    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const toData = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top;
      const dx = -DOM + ((x - L.px0) / L.s) * 2 * DOM, dy = DOM - ((y - L.py0) / L.s) * 2 * DOM;
      return Math.abs(dx) <= DOM && Math.abs(dy) <= DOM ? { x: dx, y: dy } : null;
    };
    const move = (e: PointerEvent) => { hover = toData(e); cv.style.cursor = hover && editable ? 'crosshair' : ''; wake(); };
    const leave = () => { hover = null; wake(); };
    const down = (e: PointerEvent) => { const p = editable ? toData(e) : null; if (p) addPoint(p.x, p.y, cls); };
    cv.addEventListener('pointermove', move); cv.addEventListener('pointerleave', leave); cv.addEventListener('pointerdown', down);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerleave', leave); cv.removeEventListener('pointerdown', down); };
  });
</script>

<FigureCard {title} role="group" aspect="720 / 400" narrowAspect="360 / 640"
  label="Left: a plane of blue circles (class 1) and orange diamonds (class 0), coloured by the classifier's probability, with a black line where the probability is 0.5. Right: the training loss falling over gradient steps."
  legend={[{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: scores ? 'boundary, z = 0' : 'boundary, P = 0.5', color: '--text', shape: 'line' }, ...(scores ? [{ label: 'score z = ±1, ±2', color: '--text-muted', shape: 'line' as const }] : []), { label: 'loss', color: '--viz-gradient', shape: 'line' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline: training the classifier"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">step <b>{Math.round(ui.k)}</b> · loss <b>{f2(ui.loss, 3)}</b> · accuracy <b>{Math.round(ui.acc * 100)}%</b>
      {#if hov}<span class="base">· at ({f2(hov.x)}, {f2(hov.y)}): P(class 1) = {f2(hov.p, 3)}, score z = {f2(hov.z)}{#if mdl === 'linear'}, distance to the line = {f2(hov.d)}{/if}</span>{/if}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if editable}
      <div class="ctl">
        <label>Data
          <select value={ds} onchange={(e) => setData(e.currentTarget.value as Dataset)}>
            <option value="linear">Two blobs</option><option value="xor">XOR</option><option value="circles">Rings</option>
          </select>
        </label>
        <label>Features
          <select value={mdl} onchange={(e) => setModel(e.currentTarget.value as Model)}>
            <option value="linear">x, y (a line)</option><option value="poly">x, y, x², xy, y² (a curve)</option>
          </select>
        </label>
        <label>Click adds
          <select value={cls} onchange={(e) => (cls = +e.currentTarget.value as 0 | 1)}>
            <option value={1}>class 1 (blue circle)</option><option value={0}>class 0 (orange diamond)</option>
          </select>
        </label>
        <button type="button" class="btn" onclick={addRandom}>Add a random point</button>
        <button type="button" class="btn" onclick={reset}>Clear my points</button>
        <span class="hint">Click the plane to add a point; the classifier keeps training from where it was.</span>
      </div>
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
  .ctl label { display: inline-flex; align-items: center; gap: 0.5em; font: var(--step--1) var(--font-ui); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
