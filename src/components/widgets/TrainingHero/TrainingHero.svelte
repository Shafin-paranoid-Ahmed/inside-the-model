<script lang="ts">
  import { onMount } from 'svelte';
  import { SIZES, EXTENT, KINDS, wIndex, makeDataset, field, type Kind, type Dataset } from './sim';
  import type { Snapshot } from './trainer';
  import type { ToWorker } from './worker';

  const SEED = 1, RES = 48, CYCLE = 1000, FWD_END = 350, BWD_START = 420, BWD_END = 770;
  const LABEL: Record<Kind, string> = { moons: 'Moons', circle: 'Circle', xor: 'XOR', spiral: 'Spiral' };

  let kind = $state<Kind>('moons');
  let playing = $state(true);
  let cls = $state<1 | -1>(1);
  let lossText = $state('1.0000');
  let stepText = $state('0');
  let live = $state('');

  let plotEl: HTMLCanvasElement, netEl: HTMLCanvasElement, lossEl: HTMLCanvasElement, root: HTMLElement;

  onMount(() => {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const html = document.documentElement;
    const heat = document.createElement('canvas');
    heat.width = heat.height = RES;
    const hctx = heat.getContext('2d')!, himg = hctx.createImageData(RES, RES), fld = new Float32Array(RES * RES);
    const pctx = plotEl.getContext('2d')!, nctx = netEl.getContext('2d')!, lctx = lossEl.getContext('2d')!;

    let data: Dataset = makeDataset(kind, SEED);
    let snap: Snapshot | null = null;
    let history: number[] = [];
    let maxGrad = 1e-9;
    let inView = false, alive = false, raf = 0, lastStart = -1e9, lastField = 0, lastText = 0;
    let dirtyPlot = true, dirtyNet = true, dirtyLoss = true;
    let c = readColors();

    function css(n: string) { return getComputedStyle(html).getPropertyValue(n).trim(); }
    function rgb(n: string): number[] {
      const h = css(n).replace('#', '');
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    function readColors() {
      return {
        bg: rgb('--bg-raised'), pos: rgb('--viz-pos'), neg: rgb('--viz-neg'),
        posS: css('--viz-pos'), negS: css('--viz-neg'), fwd: css('--viz-forward'), grad: css('--viz-gradient'),
        text: css('--text'), muted: css('--text-muted'), rule: css('--rule'), raised: css('--bg-raised'),
        font: css('--font-code'),
      };
    }

    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    const post = (m: ToWorker) => worker.postMessage(m);
    worker.onmessage = (e: MessageEvent<Snapshot>) => {
      snap = e.data;
      history.push(snap.loss);
      if (history.length > 400) history.shift();
      let g = 1e-9;
      for (let i = 0; i < snap.grads.length; i++) g = Math.max(g, Math.abs(snap.grads[i]));
      maxGrad = g;
      dirtyPlot = dirtyNet = dirtyLoss = true;
      const now = performance.now();
      if (now - lastText > 100) {
        lastText = now;
        lossText = snap.loss.toFixed(4);
        stepText = String(snap.step);
      }
      wake();
    };

    function restart() {
      history = [];
      snap = null;
      post({ type: 'reset', seed: SEED, xs: data.xs, ys: data.ys });
      dirtyPlot = true;
      wake();
    }

    function syncRun() {
      alive = inView && !document.hidden;
      post({ type: 'run', on: alive && playing });
      if (alive) wake();
    }

    // ---- drawing -------------------------------------------------------------------------
    const dpr = () => Math.min(2, window.devicePixelRatio || 1);

    function drawPlot() {
      const W = plotEl.width, H = plotEl.height, d = dpr();
      pctx.fillStyle = c.raised;
      pctx.fillRect(0, 0, W, H);
      if (snap) {
        field(snap.weights, RES, fld);
        lastField = performance.now();
        for (let i = 0; i < fld.length; i++) {
          const v = fld[i], t = v > 0 ? c.pos : c.neg, a = 0.5 * Math.min(1, Math.abs(v));
          himg.data[4 * i] = c.bg[0] + (t[0] - c.bg[0]) * a;
          himg.data[4 * i + 1] = c.bg[1] + (t[1] - c.bg[1]) * a;
          himg.data[4 * i + 2] = c.bg[2] + (t[2] - c.bg[2]) * a;
          himg.data[4 * i + 3] = 255;
        }
        hctx.putImageData(himg, 0, 0);
      }
      if (snap) {
        pctx.imageSmoothingEnabled = true;
        pctx.imageSmoothingQuality = 'high';
        pctx.drawImage(heat, 0, 0, W, H);
      }
      pctx.strokeStyle = c.rule;
      pctx.lineWidth = d;
      pctx.globalAlpha = 0.7;
      pctx.beginPath();
      pctx.moveTo(W / 2, 0); pctx.lineTo(W / 2, H);
      pctx.moveTo(0, H / 2); pctx.lineTo(W, H / 2);
      pctx.stroke();
      pctx.globalAlpha = 1;
      const k = W / (2 * EXTENT);
      pctx.lineWidth = 1.4 * d;
      pctx.strokeStyle = c.raised;
      for (let i = 0; i < data.ys.length; i++) {
        pctx.fillStyle = data.ys[i] > 0 ? c.posS : c.negS;
        pctx.beginPath();
        pctx.arc((data.xs[2 * i] + EXTENT) * k, (EXTENT - data.xs[2 * i + 1]) * k, 3.6 * d, 0, 6.2832);
        pctx.fill();
        pctx.stroke();
      }
    }

    const nodeX = (l: number, W: number) => 34 * dpr() + (l * (W - 68 * dpr())) / (SIZES.length - 1);
    const nodeY = (l: number, j: number, H: number) => {
      const gap = (H - 28 * dpr()) / 5;
      return H / 2 + (j - (SIZES[l] - 1) / 2) * gap;
    };

    function drawNet(t: number) {
      const W = netEl.width, H = netEl.height, d = dpr();
      nctx.clearRect(0, 0, W, H);
      const w = snap?.weights;
      const seg = (l: number, j: number, i: number, u0: number, u1: number) => {
        const x0 = nodeX(l, W), y0 = nodeY(l, i, H), x1 = nodeX(l + 1, W), y1 = nodeY(l + 1, j, H);
        nctx.beginPath();
        nctx.moveTo(x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0);
        nctx.lineTo(x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1);
        nctx.stroke();
      };
      nctx.lineCap = 'round';
      if (w) {
        for (let l = 0; l < SIZES.length - 1; l++)
          for (let j = 0; j < SIZES[l + 1]; j++)
            for (let i = 0; i < SIZES[l]; i++) {
              const v = w[wIndex(l, j, i)];
              nctx.strokeStyle = v > 0 ? c.posS : c.negS;
              nctx.globalAlpha = 0.75;
              nctx.lineWidth = (0.4 + 2.6 * Math.min(1, Math.abs(v) / 1.5)) * d;
              seg(l, j, i, 0, 1);
            }
      }
      if (w && snap && t >= 0) {
        const L = SIZES.length - 1, tail = 0.3;
        nctx.lineWidth = 2.4 * d;
        const fp = t < FWD_END ? (t / FWD_END) * L : -1;
        const bp = t >= BWD_START && t < BWD_END ? (1 - (t - BWD_START) / (BWD_END - BWD_START)) * L : -1;
        for (let l = 0; l < L; l++)
          for (let j = 0; j < SIZES[l + 1]; j++)
            for (let i = 0; i < SIZES[l]; i++) {
              const idx = wIndex(l, j, i);
              if (fp >= 0 && fp - l > 0 && fp - l < 1 + tail) {
                nctx.strokeStyle = c.fwd;
                nctx.globalAlpha = 0.25 + 0.7 * Math.min(1, Math.abs(w[idx]) / 1.2);
                seg(l, j, i, Math.max(0, fp - l - tail), Math.min(1, fp - l));
              }
              if (bp >= 0 && bp - l > -tail && bp - l < 1) {
                nctx.strokeStyle = c.grad;
                nctx.globalAlpha = 0.2 + 0.75 * Math.sqrt(Math.abs(snap.grads[idx]) / maxGrad);
                seg(l, j, i, Math.max(0, bp - l), Math.min(1, bp - l + tail));
              }
            }
      }
      nctx.globalAlpha = 1;
      nctx.lineWidth = 1.3 * d;
      nctx.strokeStyle = c.muted;
      nctx.fillStyle = c.raised;
      for (let l = 0; l < SIZES.length; l++)
        for (let j = 0; j < SIZES[l]; j++) {
          nctx.beginPath();
          nctx.arc(nodeX(l, W), nodeY(l, j, H), 5.5 * d, 0, 6.2832);
          nctx.fill();
          nctx.stroke();
        }
      nctx.fillStyle = c.muted;
      nctx.font = `${11 * d}px ${c.font}`;
      nctx.textBaseline = 'middle';
      nctx.textAlign = 'right';
      nctx.fillText('x', nodeX(0, W) - 11 * d, nodeY(0, 0, H));
      nctx.fillText('y', nodeX(0, W) - 11 * d, nodeY(0, 1, H));
      nctx.textAlign = 'left';
      nctx.fillText('out', nodeX(3, W) + 11 * d, nodeY(3, 0, H));
    }

    function drawLoss() {
      const W = lossEl.width, H = lossEl.height, d = dpr();
      lctx.clearRect(0, 0, W, H);
      lctx.strokeStyle = c.rule;
      lctx.lineWidth = d;
      lctx.beginPath();
      lctx.moveTo(0, H - d / 2); lctx.lineTo(W, H - d / 2);
      lctx.stroke();
      if (history.length < 2) return;
      // log scale: a 10x drop reads the same early and late
      const lg = history.map((v) => Math.log(v + 1e-4)), hi = Math.max(...lg), lo = Math.min(...lg, hi - 1), pad = 3 * d;
      lctx.strokeStyle = c.text;
      lctx.lineWidth = 1.5 * d;
      lctx.lineJoin = 'round';
      lctx.beginPath();
      lg.forEach((v, i) => {
        const x = pad + (i / (lg.length - 1)) * (W - 2 * pad), y = pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad - d);
        i ? lctx.lineTo(x, y) : lctx.moveTo(x, y);
      });
      lctx.stroke();
    }

    function frame(now: number) {
      raf = 0;
      if (!alive) return;
      const training = playing;
      let again = false;
      if (dirtyPlot) {
        if (reduced.matches && snap && now - lastField < 250) again = true; // calmer refresh rate
        else { drawPlot(); dirtyPlot = false; }
      }
      if (dirtyLoss) { drawLoss(); dirtyLoss = false; }
      if (!reduced.matches && training && snap && now - lastStart >= CYCLE) lastStart = now;
      const t = now - lastStart;
      if (!reduced.matches && training && t < CYCLE) {
        drawNet(t);
        dirtyNet = true;
        again = true;
      } else if (dirtyNet) {
        drawNet(-1);
        dirtyNet = false;
      }
      if (again) raf = requestAnimationFrame(frame);
    }
    function wake() { if (alive && !raf) raf = requestAnimationFrame(frame); }

    // ---- environment ---------------------------------------------------------------------
    function fit(cv: HTMLCanvasElement) {
      const r = cv.getBoundingClientRect(), d = dpr();
      const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    }
    const ro = new ResizeObserver(() => {
      [plotEl, netEl, lossEl].forEach(fit);
      dirtyPlot = dirtyNet = dirtyLoss = true;
      lastField = 0;
      wake();
    });
    [plotEl, netEl, lossEl].forEach((cv) => ro.observe(cv));

    const io = new IntersectionObserver(([e]) => { inView = e.isIntersecting; syncRun(); });
    io.observe(root);
    const onVis = () => syncRun();
    document.addEventListener('visibilitychange', onVis);

    const onTheme = () => { c = readColors(); dirtyPlot = dirtyNet = dirtyLoss = true; lastField = 0; wake(); };
    const mo = new MutationObserver(onTheme);
    mo.observe(html, { attributes: true, attributeFilter: ['data-theme'] });
    const dark = matchMedia('(prefers-color-scheme: dark)');
    dark.addEventListener('change', onTheme);
    reduced.addEventListener('change', () => { dirtyPlot = dirtyNet = true; wake(); });

    // announce loss occasionally, never faster than every 5 s
    const liveTimer = setInterval(() => {
      if (alive && playing && snap) live = `Step ${snap.step}, loss ${snap.loss.toFixed(4)}.`;
    }, 5000);

    // ---- user actions (exposed to the template) ------------------------------------------
    actions.toggle = () => { playing = !playing; syncRun(); wake(); };
    actions.reset = (k?: Kind) => {
      if (k) kind = k;
      data = makeDataset(kind, SEED);
      restart();
    };
    actions.add = (x: number, y: number, label: number) => {
      const n = data.ys.length, xs = new Float32Array(2 * n + 2), ys = new Float32Array(n + 1);
      xs.set(data.xs); ys.set(data.ys);
      xs[2 * n] = x; xs[2 * n + 1] = y; ys[n] = label;
      data = { xs, ys };
      post({ type: 'data', xs, ys });
      dirtyPlot = true;
      wake();
    };

    post({ type: 'reset', seed: SEED, xs: data.xs, ys: data.ys });

    return () => {
      worker.terminate();
      cancelAnimationFrame(raf);
      clearInterval(liveTimer);
      ro.disconnect(); io.disconnect(); mo.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      dark.removeEventListener('change', onTheme);
    };
  });

  const actions = {
    toggle: () => {}, reset: (_k?: Kind) => {}, add: (_x: number, _y: number, _l: number) => {},
  };

  function onPlotClick(e: MouseEvent) {
    const r = plotEl.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 2 * EXTENT - EXTENT;
    const y = EXTENT - ((e.clientY - r.top) / r.height) * 2 * EXTENT;
    actions.add(x, y, e.shiftKey ? -cls : cls);
  }
  function addRandom() {
    actions.add((Math.random() * 2 - 1) * (EXTENT - 0.2), (Math.random() * 2 - 1) * (EXTENT - 0.2), cls);
  }
</script>

<figure class="hero-viz" bind:this={root}>
  <div class="grid">
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <div class="plot" onclick={onPlotClick}>
      <canvas
        bind:this={plotEl}
        role="img"
        aria-label="Scatter plot of two classes of points, blue and orange, over a heat map of the network's current prediction. The boundary between the tints moves as the network trains."
      ></canvas>
    </div>
    <div class="side">
      <div class="net">
        <canvas
          bind:this={netEl}
          role="img"
          aria-label="Diagram of a neural network with 2 inputs, two hidden layers of 6 units and 1 output. Blue connections have positive weights, orange negative."
        ></canvas>
      </div>
      <div class="loss">
        <dl class="readout num">
          <div><dt>loss</dt><dd>{lossText}</dd></div>
          <div><dt>step</dt><dd>{stepText}</dd></div>
        </dl>
        <div class="spark">
          <canvas bind:this={lossEl} role="img" aria-label="Line chart of training loss over recent steps, falling."></canvas>
        </div>
      </div>
    </div>
  </div>

  <div class="controls ui">
    <div class="group">
      <button class="btn" type="button" onclick={() => actions.toggle()}>{playing ? 'Pause' : 'Play'}</button>
      <button class="btn" type="button" onclick={() => actions.reset()}>Reset</button>
    </div>
    <div class="group seg" role="group" aria-label="Dataset">
      {#each KINDS as k}
        <button class="btn" type="button" aria-pressed={kind === k} onclick={() => actions.reset(k)}>{LABEL[k]}</button>
      {/each}
    </div>
    <div class="group seg" role="group" aria-label="Class of the point to add">
      <span class="lbl">Add</span>
      <button class="btn" type="button" aria-pressed={cls === 1} onclick={() => (cls = 1)}><i class="dot pos"></i>Blue</button>
      <button class="btn" type="button" aria-pressed={cls === -1} onclick={() => (cls = -1)}><i class="dot neg"></i>Orange</button>
      <button class="btn" type="button" onclick={addRandom}>Random point</button>
    </div>
  </div>

  <figcaption>
    Blue connections push the output up, orange push it down. Teal pulses are the forward pass, magenta the gradients
    flowing back. Click the plot to add a point (shift-click for the other color) and watch the boundary bend.
  </figcaption>
  <p class="sr-only" aria-live="polite">{live}</p>
</figure>

<style>
  .hero-viz { margin: 0; padding: var(--s-3); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-2); }
  .grid { display: grid; gap: var(--s-3); }
  .plot { aspect-ratio: 1; border: 1px solid var(--rule); cursor: crosshair; position: relative; }
  canvas { display: block; width: 100%; height: 100%; }
  .side { display: flex; flex-direction: column; gap: var(--s-3); min-width: 0; }
  .net { aspect-ratio: 16 / 10; position: relative; }
  .net canvas { position: absolute; inset: 0; }
  .loss { display: flex; gap: var(--s-4); align-items: stretch; border-top: 1px solid var(--rule); padding-top: var(--s-3); }
  .readout { margin: 0; display: grid; gap: var(--s-1); font-size: var(--step--1); flex: none; min-width: 7.5ch; }
  .readout div { display: flex; flex-direction: column; }
  dt { font: 0.75rem var(--font-ui); color: var(--text-muted); }
  dd { margin: 0; font-family: var(--font-code); font-variant-numeric: tabular-nums; min-width: 6ch; }
  .spark { flex: 1; height: 3.6rem; min-width: 0; }
  .controls { display: flex; flex-wrap: wrap; gap: var(--s-3) var(--s-5); margin-top: var(--s-3); padding-top: var(--s-3); border-top: 1px solid var(--rule); align-items: center; }
  .group { display: flex; flex-wrap: wrap; gap: var(--s-2); align-items: center; }
  .group .btn { padding: 0.3rem 0.7rem; font-size: 0.8125rem; display: inline-flex; align-items: center; gap: 0.4rem; }
  .btn[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); color: var(--text); }
  .lbl { font-size: 0.75rem; color: var(--text-muted); }
  .dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; display: inline-block; }
  .dot.pos { background: var(--viz-pos); }
  .dot.neg { background: var(--viz-neg); }
  figcaption { margin-top: var(--s-3); font: 0.8125rem / 1.5 var(--font-ui); color: var(--text-muted); max-width: 60ch; }
  @media (min-width: 48em) {
    .hero-viz { padding: var(--s-4); }
    .grid { grid-template-columns: minmax(0, 24rem) minmax(0, 1fr); gap: var(--s-4); }
    .net { aspect-ratio: auto; flex: 1; min-height: 0; }
  }
  @media (prefers-reduced-motion: reduce) { .btn { transition: none; } }
</style>
