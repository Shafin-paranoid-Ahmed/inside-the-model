<script module lang="ts">
  import type { FromWorker, ToWorker } from './worker';
  import type { NetId, NetRun } from './sim';
  const workers = new Map<NetId, Worker>();
  const pending = new Map<string, Promise<NetRun>>();
  /** Train one network in its own worker, once per (network, seed) per page. */
  function load(id: NetId, seed: number): Promise<NetRun> {
    const key = `${id}:${seed}`;
    let p = pending.get(key);
    if (!p) {
      let w = workers.get(id);
      if (!w) { w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }); workers.set(id, w); }
      const ww = w;
      p = new Promise<NetRun>((res) => {
        const on = (e: MessageEvent<FromWorker>) => { if (e.data.seed === seed) { ww.removeEventListener('message', on); res(e.data.run); } };
        ww.addEventListener('message', on);
        ww.postMessage({ id, seed } satisfies ToWorker);
      });
      pending.set(key, p);
    }
    return p;
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, clamp } from '../F1Shared/plot';
  import {
    NETS, STEPS, GRID, EXTENT, LOSS_FLOOR, DEG_T, DEG_MARKERS, BLOCK_D, BLOCK_T, BLOCK_MARKERS, ARCHS, HIST_T, HIST_MARKERS,
    DEFAULT_SEED, dataset, degradationFrame, makeBlock, blockAt, historyAt, type NetId, type NetRun,
  } from './sim';

  interface Props {
    title?: string;
    /** 'history': four landmark networks, their depth and ImageNet error. 'degradation': train three networks. 'block': one residual block. */
    mode?: 'history' | 'degradation' | 'block';
  }
  let { title = '', mode = 'degradation' }: Props = $props();
  const init = untrack(() => ({ mode, title }));
  const T = init.mode === 'history' ? HIST_T : init.mode === 'block' ? BLOCK_T : DEG_T;
  const MK = init.mode === 'history' ? HIST_MARKERS : init.mode === 'block' ? BLOCK_MARKERS : DEG_MARKERS;
  const SEEDS = [DEFAULT_SEED, 1, 3];

  let playing = $state(false), speed = $state(1), uiT = $state(0), ready = $state(init.mode !== 'degradation');
  let seedIx = $state(0), done = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let runs: Partial<Record<NetId, NetRun>> = {};
  const block = makeBlock();
  const data = dataset();

  const phase = $derived(markerAt(uiT, MK)?.label ?? '');
  const pct = (v: number) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1)).replace('e-', 'e−');

  const ui = $derived.by(() => {
    if (init.mode === 'history') {
      const k = Math.min(ARCHS.length - 1, Math.floor(uiT / 3)), a = ARCHS[k];
      return { a: `${a.name} (${a.year}): ${a.layers} layers`, b: a.err === null ? 'no ImageNet result (it read handwritten digits)' : `ImageNet top-5 error ${a.err}%`, c: a.note };
    }
    if (init.mode === 'block') {
      const f = blockAt(uiT, block);
      return {
        a: `Branch weights multiplied by ${f2(f.w)}`,
        b: `Distance from the input: plain layer ${f2(f.plainGap)}, residual block ${f2(f.residGap)}`,
        c: f.w < 0.005 ? 'With zero weights the residual block returns its input exactly; the plain layer returns zeros.' : '',
      };
    }
    if (!ready) return { a: '', b: '', c: '' };
    const f = degradationFrame(uiT, runs);
    return { a: `Step ${f.step} of ${STEPS}`, b: NETS.map((n, i) => `${n.short}: ${pct(f.loss[i])}`).join(' · '), c: '' };
  });
  const narration = $derived(
    init.mode === 'history' ? 'Each network is drawn with its layer count (top) and its ImageNet top-5 error (bottom). From 2012 to 2015 depth grew about nineteenfold while the error fell from 15.3% to 3.57%.'
    : init.mode === 'block' ? (phase === 'Flow' ? 'The input travels down two roads: through the branch F, and along the identity road that skips it. The residual block adds them.'
        : phase === 'Shrink' ? 'The branch weights shrink. The plain layer’s output collapses toward zero, while the residual block’s output slides back to the input.'
        : 'Weights at zero: the plain layer has erased the signal; the residual block is the identity.')
    : !ready ? `Training three networks in background threads (${done} of 3 done)…`
    : 'Same 120 points, same optimizer, same learning rate, same 800 steps. The deeper plain network ends with a higher training loss than the shallow one; the skip network goes lowest.',
  );

  function reseed() { seedIx = (seedIx + 1) % SEEDS.length; start(); }
  function start() {
    const seed = SEEDS[seedIx]; ready = false; done = 0; runs = {}; t = 0; uiT = 0; playing = false;
    for (const n of NETS) void load(n.id, seed).then((r) => { if (SEEDS[seedIx] !== seed) return; runs[n.id] = r; done++; if (done === 3) { ready = true; playing = !reducedMotion(); } wake(); });
    wake();
  }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const rgba = (c: { rgb: number[] }, a: number) => `rgba(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]},${a})`;

    // ---------------- degradation ----------------
    function degradation() {
      const { ctx, w: W, h: H, c } = stage, R = runs, f = degradationFrame(t, R), narrow = W < 540;
      const side = Math.min((W - 24) / 3, H * 0.4), gap = (W - 8 - 3 * side) / 2, py = H - side - 4;
      const cb = { l: narrow ? 44 : 52, r: W - 10, t: 22, b: py - 52 };
      const X = scale(0, STEPS, cb.l, cb.r), lo = Math.log10(LOSS_FLOOR), hi = Math.log10(3), Y = scale(lo, hi, cb.b, cb.t);
      const yv = (v: number) => Y(Math.log10(Math.max(LOSS_FLOOR, v)));
      ctx.lineWidth = 1; ctx.font = stage.mono(narrow ? 10 : 11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let e = -5; e <= 0; e++) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(cb.l, Y(e)); ctx.lineTo(cb.r, Y(e)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(e === 0 ? '1' : `1e−${-e}`, cb.l - 5, Y(e)); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const s of [0, 200, 400, 600, 800]) { ctx.fillStyle = c.muted.hex; ctx.fillText(String(s), X(s), cb.b + 5); }
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.fillText('training steps', cb.r, cb.b + 20);
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText('training loss (log scale, lower is better)', cb.l - 38, cb.t - 6);
      if (!ready) { ctx.font = stage.body(13); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(`Training three networks in the background… ${done} of 3 done`, (cb.l + cb.r) / 2, (cb.t + cb.b) / 2); }
      // coin-flip level
      const ln2 = Math.LN2; ctx.setLineDash([5, 5]); ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cb.l, yv(ln2)); ctx.lineTo(cb.r, yv(ln2)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.muted.hex; ctx.font = stage.body(11); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('0.693: guessing 50/50', cb.r - 4, yv(ln2) - 3);
      const cols = [c.fwd, c.text, c.accent];
      NETS.forEach((n, i) => {
        const run = R[n.id]; if (!run) return;
        const L = run.loss; ctx.strokeStyle = cols[i].hex; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let s = 0; s <= f.step; s++) { const x = X(s), y = yv(L[s]); if (s) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.stroke();
        ctx.fillStyle = cols[i].hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(f.step), yv(L[f.step]), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      });
      // decision maps
      NETS.forEach((n, i) => {
        const px = 4 + i * (side + gap), S = R[n.id]?.snaps, a = f.snap * GRID * GRID, b = (f.snap + 1) * GRID * GRID, cell = side / GRID;
        if (!S) {
          ctx.fillStyle = c.sunken.hex; ctx.fillRect(px, py, side, side); ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('training…', px + side / 2, py + side / 2);
        } else for (let j = 0; j < GRID; j++) for (let k = 0; k < GRID; k++) {
          const q = S[a + j * GRID + k] * (1 - f.f) + S[b + j * GRID + k] * f.f, conf = Math.abs(2 * q - 1);
          ctx.fillStyle = rgba(q >= 0.5 ? c.pos : c.neg, 0.1 + 0.45 * conf);
          const x0 = Math.round(px + k * cell), y0 = Math.round(py + side - (j + 1) * cell);
          ctx.fillRect(x0, y0, Math.round(px + (k + 1) * cell) - x0, Math.round(py + side - j * cell) - y0);
        }
        for (let p = 0; p < (S ? data.n : 0); p++) {
          ctx.fillStyle = (data.y[p] ? c.pos : c.neg).hex; ctx.beginPath();
          ctx.arc(px + ((data.X[2 * p] + EXTENT) / (2 * EXTENT)) * side, py + side - ((data.X[2 * p + 1] + EXTENT) / (2 * EXTENT)) * side, narrow ? 1.6 : 2.2, 0, 6.2832); ctx.fill();
        }
        ctx.strokeStyle = cols[i].hex; ctx.lineWidth = 2.5; ctx.strokeRect(px, py, side, side);
        ctx.font = stage.body(narrow ? 11 : 12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(narrow ? n.short : n.label, px, py - 4);
      });
    }

    // ---------------- block ----------------
    function bars(x0: number, y0: number, w: number, h: number, v: number[], col: { hex: string; rgb: number[] }, ref?: number[]) {
      const { ctx, c } = stage, D = BLOCK_D, bw = w / D, mid = y0 + h / 2, k = h / 2 / 2; // values in [-2, 2]
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x0 + w, mid); ctx.stroke();
      for (let j = 0; j < D; j++) {
        const val = clamp(v[j], -2, 2); ctx.fillStyle = rgba(col, 0.85); ctx.fillRect(x0 + j * bw + 1.5, val >= 0 ? mid - val * k : mid, bw - 3, Math.abs(val) * k);
      }
      if (ref) for (let j = 0; j < D; j++) { ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2; ctx.beginPath(); const yy = mid - clamp(ref[j], -2, 2) * k; ctx.moveTo(x0 + j * bw, yy); ctx.lineTo(x0 + (j + 1) * bw, yy); ctx.stroke(); }
    }
    function blockFig() {
      const { ctx, w: W, h: H, c } = stage, f = blockAt(t, block), narrow = W < 540, rowH = H / 2, bw = narrow ? 0.2 * W : 0.17 * W, bh = rowH * 0.6;
      const xIn = 0.03 * W, xBox = narrow ? 0.34 * W : 0.37 * W, boxW = narrow ? 0.18 * W : 0.14 * W, xOut = W - bw - 0.03 * W, xPlus = narrow ? 0.64 * W : 0.66 * W;
      const rows = [{ y: 0, resid: false, y_out: f.plain, label: 'Plain layer: y = F(x)' }, { y: rowH, resid: true, y_out: f.resid, label: 'Residual block: y = x + F(x)' }];
      const anim = !reducedMotion();
      rows.forEach((r) => {
        const top = r.y + 26, cy = top + bh / 2;
        ctx.font = stage.body(narrow ? 12 : 13.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(r.label, xIn, r.y + 4);
        bars(xIn, top, bw, bh, f.x, c.fwd); bars(xOut, top, bw, bh, r.y_out, c.fwd, f.x);
        ctx.font = stage.mono(10.5); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText('x', xIn + bw / 2, top + bh + 3); ctx.fillText('y', xOut + bw / 2, top + bh + 3);
        // roads
        const x1 = xIn + bw, x2 = r.resid ? xPlus - 12 : xOut, bx = xBox, ex = xBox + boxW;
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x1, cy); ctx.lineTo(bx, cy); ctx.moveTo(ex, cy); ctx.lineTo(x2, cy); ctx.stroke();
        const byY = top + bh * 0.04;
        if (r.resid) { ctx.beginPath(); ctx.moveTo(x1 + 6, cy); ctx.lineTo(x1 + 6, byY); ctx.lineTo(xPlus, byY); ctx.lineTo(xPlus, cy - 12); ctx.moveTo(xPlus + 12, cy); ctx.lineTo(xOut, cy); ctx.stroke(); }
        // branch box, shaded by the weight scale
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.5 + 1.5 * f.w; ctx.beginPath(); ctx.roundRect(bx, cy - bh * 0.3, boxW, bh * 0.6, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c.text.hex; ctx.font = stage.mono(narrow ? 11 : 13, 650); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('F', bx + boxW / 2, cy - 7);
        ctx.font = stage.mono(narrow ? 9 : 10.5); ctx.fillStyle = c.muted.hex; ctx.fillText(narrow ? `×${f2(f.w)}` : `weights ×${f2(f.w)}`, bx + boxW / 2, cy + 9);
        if (r.resid) {
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(xPlus, cy, 12, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.fillStyle = c.accent.hex; ctx.font = stage.mono(16, 700); ctx.fillText('+', xPlus, cy + 1);
          if (!narrow) { ctx.font = stage.body(11.5); ctx.fillStyle = c.muted.hex; ctx.fillText('identity road (skips F)', (x1 + xPlus) / 2 + 6, byY - 9); }
        }
        // particles
        if (anim) {
          const dot = (x: number, y: number, a: number) => { ctx.fillStyle = rgba(c.fwd, a); ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.2832); ctx.fill(); };
          for (let k = 0; k < 2; k++) {
            const u = (f.flow + k * 0.5) % 1; // along the branch road
            const xb = x1 + (x2 - x1) * u; dot(xb, cy, xb < bx ? 1 : 0.15 + 0.85 * f.w);
            if (r.resid) { // identity road: along the arc
              const L1 = byY - cy, L2 = xPlus - (x1 + 6), L3 = cy - 12 - byY, tot = Math.abs(L1) + L2 + L3; let d = u * tot;
              let px = x1 + 6, py = cy; const s1 = Math.abs(L1);
              if (d < s1) py = cy - d; else if (d < s1 + L2) { py = byY; px = x1 + 6 + (d - s1); } else { px = xPlus; py = byY + (d - s1 - L2); }
              dot(px, py, 1);
            }
          }
        }
      });
    }

    // ---------------- history ----------------
    function history() {
      const { ctx, w: W, h: H, c } = stage, p = historyAt(t), narrow = W < 540, n = ARCHS.length;
      const L = 12, Rr = W - 12, colW = (Rr - L) / n, bw = colW * 0.56;
      const top1 = { t: 26, b: H * 0.5 - 8 }, top2 = { t: H * 0.5 + 34, b: H - 38 };
      const Yl = scale(0, 168, top1.b, top1.t), Ye = scale(0, 19, top2.b, top2.t);
      ctx.font = stage.body(narrow ? 12 : 13); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('Layers with learned weights', L, top1.t - 6); ctx.fillText(narrow ? 'ImageNet top-5 error, % (lower is better)' : 'ImageNet top-5 test error, % (lower is better)', L, top2.t - 6);
      ARCHS.forEach((a, i) => {
        const cx = L + (i + 0.5) * colW, x0 = cx - bw / 2, q = p[i];
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; if (i === 0) { for (const box of [top1, top2]) { ctx.beginPath(); ctx.moveTo(L, box.b); ctx.lineTo(Rr, box.b); ctx.stroke(); } }
        const hL = (top1.b - Yl(a.layers)) * q;
        ctx.fillStyle = rgba(c.fwd, 0.85); ctx.fillRect(x0, top1.b - hL, bw, hL);
        ctx.globalAlpha = q; ctx.fillStyle = c.text.hex; ctx.font = stage.mono(narrow ? 11 : 13, 650); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(String(a.layers), cx, top1.b - hL - 3);
        if (a.err !== null) {
          const hE = (top2.b - Ye(a.err)) * q; ctx.fillStyle = rgba(c.neg, 0.85); ctx.fillRect(x0, top2.b - hE, bw, hE);
          ctx.fillStyle = c.text.hex; ctx.fillText(a.err + '%', cx, top2.b - hE - 3);
        } else { ctx.font = stage.body(narrow ? 10.5 : 12); ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'bottom'; ctx.fillText('digits only', cx, top2.b - 4); }
        ctx.globalAlpha = 1;
        ctx.font = stage.body(narrow ? 11.5 : 13); ctx.fillStyle = c.text.hex; ctx.textBaseline = 'top'; ctx.textAlign = 'center'; ctx.fillText(a.name, cx, top2.b + 6);
        ctx.font = stage.mono(10.5); ctx.fillStyle = c.muted.hex; ctx.fillText(String(a.year), cx, top2.b + 22);
      });
    }

    function render() {
      const { ctx, w, h } = stage; ctx.clearRect(0, 0, w, h);
      if (init.mode === 'degradation') return degradation();
      if (init.mode === 'block') return blockFig();
      history();
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T, loopTo: T + (init.mode === 'degradation' ? 1.5 : 0) }).t; // degradation holds on step 800 before looping
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && init.mode !== 'degradation' && !reducedMotion()) playing = true; if (first && init.mode === 'degradation') start(); } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });
  $effect(() => { playing; speed; wake(); });
</script>

<FigureCard title={init.title || 'Residual networks'} role="group" aspect={init.mode === 'degradation' ? '720 / 520' : init.mode === 'block' ? '720 / 380' : '720 / 400'}
  narrowAspect={init.mode === 'degradation' ? '360 / 490' : init.mode === 'block' ? '360 / 420' : '360 / 440'}
  label={init.mode === 'degradation' ? 'Top: training loss on a log scale against training steps for three networks. Bottom: the decision map each network has learned on two interleaved spirals.'
    : init.mode === 'block' ? 'A plain layer and a residual block side by side. As the branch weights shrink to zero, the plain layer outputs zeros and the residual block outputs its input.'
    : 'Bar charts of layer count and ImageNet top-5 error for LeNet-5, AlexNet, VGG-19 and ResNet-152.'}
  legend={init.mode === 'degradation' ? [{ label: '4 layers, plain', color: '--viz-forward', shape: 'line' }, { label: '20 layers, plain', color: '--text', shape: 'line' }, { label: '20 layers, skips', color: '--accent', shape: 'line' }]
    : init.mode === 'block' ? [{ label: 'signal values', color: '--viz-forward' }, { label: 'tick: the input x', color: '--text', shape: 'line' }]
    : [{ label: 'layers', color: '--viz-forward' }, { label: 'top-5 error', color: '--viz-neg' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK}
      label={init.mode === 'degradation' ? 'Timeline: training steps' : 'Timeline'}
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = !reducedMotion() && ready; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{ui.a}</p>
    <p class="base num">{ui.b}</p>
    {#if ui.c}<p class="base">{ui.c}</p>{/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if init.mode === 'degradation'}
        <button type="button" class="btn" onclick={reseed}>Train again from a different random start</button>
        <span class="hint">Start {seedIx + 1} of {SEEDS.length}</span>
      {:else if init.mode === 'block'}
        <span class="hint">Drag the timeline to set the weight scale yourself: 1.00 at 3 s, 0.00 from 8 s.</span>
      {:else}
        <span class="hint">Drag the timeline to move from 1998 to 2015.</span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.92rem var(--font-code); font-variant-numeric: tabular-nums; }
  .base { font: 0.8rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
