<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, reducedMotion, advance } from '../_shared';
  import { createRng } from '../../../lib/math/rng';
  import {
    mixture, sampleData, score, forwardAt, reversePath, pathAt, tauAt, alphaBar, epsHat, T, T_FWD, T_HOLD, MARKERS,
    type Dataset, type Sampler, type Path, type Phase,
  } from './sim';

  interface Props {
    /** 'both' loops noise-in then denoise; 'forward' or 'reverse' loops just that half. */
    mode?: 'both' | 'forward' | 'reverse';
    dataset?: Dataset;
    /** Reverse process: 'ode' (deterministic probability flow) or 'sde' (adds fresh noise each step). */
    sampler?: Sampler;
    /** Draw the score / denoising-direction arrows. */
    showScore?: boolean;
    /** Show the sampler / arrows controls. */
    showControls?: boolean;
    /** Number of points. */
    n?: number;
    title?: string;
  }
  let { mode = 'both', dataset = 'spiral', sampler = 'ode', showScore = true, showControls = true, n = 600, title = 'Diffusion on a spiral' }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ mode, dataset, n }));
  let smp = $state<Sampler>(untrack(() => sampler));
  let arrows = $state(untrack(() => showScore));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let uiTau = $state(0), uiPhase = $state<Phase>('data');
  let cursorText = $state('');
  let narration = $state('Points from the data distribution are gradually buried in Gaussian noise.');

  const mix = mixture(init.dataset);
  const data = sampleData(mix, init.n, 7);
  const eps = (() => { const r = createRng(8); return Float32Array.from({ length: 2 * init.n }, () => r.normal()); })();
  const xT = new Float32Array(2 * init.n); forwardAt(1, data.x, eps, xT);
  const paths: Partial<Record<Sampler, Path>> = {};
  const pathFor = (k: Sampler) => (paths[k] ??= reversePath(mix, xT, k, 240, 5));

  const [lo, hi] = init.mode === 'forward' ? [0, T_FWD + T_HOLD] : init.mode === 'reverse' ? [T_FWD + T_HOLD - 0.5, T] : [0, T];
  let t = init.mode === 'reverse' ? T_FWD + T_HOLD : 0;
  uiT = t;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, wake = () => {};
  let cursor: [number, number] | null = null;
  $effect(() => { playing; speed; smp; arrows; wake(); });

  const TEXT: Record<Phase, string> = {
    data: 'Clean data: 600 samples from a spiral-shaped distribution.',
    forward: 'Forward process: every point shrinks toward the origin while fresh Gaussian noise is mixed in.',
    noise: 'Almost pure noise: the spiral is gone. The score arrows now point toward the origin.',
    reverse: 'Reverse process: follow the score field back. Points move toward regions of higher density.',
    clean: 'Denoised: the points have settled on the spiral again.',
  };

  onMount(() => {
    const ctx = cv.getContext('2d')!;
    const NAMES = { grad: '--viz-gradient', fwd: '--viz-forward', text: '--text', muted: '--text-muted', rule: '--rule', s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5', raised: '--bg-raised' };
    let c = readColors(NAMES), font = readVar('--font-code'), lastUi = 0, lastKey = '';
    const pts = new Float32Array(2 * init.n), sc = [0, 0];
    const ramp = () => { const s = [c.s2, c.s3, c.s4, c.s5]; return (u: number) => s[Math.min(3, Math.floor(u * 3.999))].hex; };
    let color = ramp();
    let W = 720, H = 420, S = 1, ox = 0, oy = 0;
    const X = (x: number) => ox + x * S, Y = (y: number) => oy - y * S;

    const draw = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), r = cv.getBoundingClientRect();
      W = r.width; H = r.height;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const wide = W >= 540;
      const sw = wide ? W * 0.66 : W, sh = wide ? H : H * 0.76;
      S = Math.min(sw, sh) / 3.5; ox = sw / 2; oy = sh / 2;
      const { tau, phase } = tauAt(t), ab = alphaBar(tau);
      if (phase === 'reverse' || phase === 'clean') pathAt(pathFor(smp), tau, pts); else forwardAt(tau, data.x, eps, pts);
      // frame
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(ox - 1.7 * S, oy - 1.7 * S, 3.4 * S, 3.4 * S);
      // score arrows
      if (arrows) {
        const stp = 0.34; ctx.strokeStyle = c.grad.hex; ctx.fillStyle = c.grad.hex; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.75;
        for (let gx = -1.5; gx <= 1.51; gx += stp) for (let gy = -1.5; gy <= 1.51; gy += stp) {
          score(mix, Math.max(tau, 0.002), gx, gy, sc);
          const m = Math.hypot(sc[0], sc[1]); if (m < 1e-9) continue;
          const len = 0.3 * Math.tanh(m / 2.5) * S, x1 = X(gx), y1 = Y(gy), ux = sc[0] / m, uy = -sc[1] / m;
          const x2 = x1 + ux * len, y2 = y1 + uy * len;
          ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - ux * 5 - uy * 3, y2 - uy * 5 + ux * 3); ctx.lineTo(x2 - ux * 5 + uy * 3, y2 - uy * 5 - ux * 3); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
      // points
      const u = phase === 'reverse' || phase === 'clean' ? data.u : data.u;
      for (let i = 0; i < init.n; i++) { ctx.fillStyle = color(u[i]); ctx.fillRect(X(pts[2 * i]) - 1.5, Y(pts[2 * i + 1]) - 1.5, 3, 3); }
      // cursor
      if (cursor) {
        score(mix, Math.max(tau, 0.002), cursor[0], cursor[1], sc);
        const m = Math.hypot(sc[0], sc[1]), len = Math.min(60, 0.3 * Math.tanh(m / 2.5) * S * 2.2);
        const x1 = X(cursor[0]), y1 = Y(cursor[1]), ux = m ? sc[0] / m : 0, uy = m ? -sc[1] / m : 0;
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 + ux * len, y1 + uy * len); ctx.stroke();
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.text.hex; ctx.beginPath(); ctx.arc(x1, y1, 4, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      // signal / noise plot
      const px = wide ? sw + 24 : 30, py = wide ? 60 : sh + 24, pw = wide ? W - sw - 44 : W - 60, ph = wide ? 130 : H - sh - 40;
      ctx.font = `600 11px ${font}`; ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left';
      ctx.fillText('signal and noise share', px, py - 12);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px, py, pw, ph);
      for (const [f, col] of [[(a: number) => Math.sqrt(a), c.s5.hex], [(a: number) => Math.sqrt(1 - a), c.fwd.hex]] as const) {
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        for (let k = 0; k <= 60; k++) { const tt = k / 60, v = f(alphaBar(tt)); k ? ctx.lineTo(px + tt * pw, py + ph - v * ph) : ctx.moveTo(px, py + ph - v * ph); }
        ctx.stroke();
      }
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px + tau * pw, py); ctx.lineTo(px + tau * pw, py + ph); ctx.stroke();
      ctx.fillStyle = c.s5.hex; ctx.fillText('√ᾱ data', px + 4, py + 12);
      ctx.fillStyle = c.fwd.hex; ctx.fillText('√(1−ᾱ) noise', px + pw * 0.4, py + ph - 6);
      ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.fillText('τ →', px + pw, py + ph + 13);
      tl?.paint(t);
      return { tau, phase, ab };
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: hi, loopFrom: lo, loopTo: hi }).t;
      const f = draw();
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t; uiTau = f.tau; uiPhase = f.phase;
        if (f.phase !== lastKey) { lastKey = f.phase; narration = TEXT[f.phase]; }
        if (cursor) { score(mix, Math.max(f.tau, 0.002), cursor[0], cursor[1], sc); cursorText = `at (${cursor[0].toFixed(2)}, ${cursor[1].toFixed(2)}): score (${sc[0].toFixed(1)}, ${sc[1].toFixed(1)}), predicted noise ε̂ (${epsHat(f.tau, sc[0]).toFixed(2)}, ${epsHat(f.tau, sc[1]).toFixed(2)})`; } else cursorText = '';
      }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const ro = new ResizeObserver(() => wake()); ro.observe(cv);
    const off = onThemeChange(() => { c = readColors(NAMES); font = readVar('--font-code'); color = ramp(); wake(); });
    const mv = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); cursor = [(((e.clientX - r.left) / r.width) * W - ox) / S, -((((e.clientY - r.top) / r.height) * H) - oy) / S]; wake(); };
    const lv = () => { cursor = null; wake(); };
    cv.addEventListener('pointermove', mv); cv.addEventListener('pointerleave', lv);
    document.fonts.ready.then(() => wake());
    return () => { loop.destroy(); ro.disconnect(); off(); cv.removeEventListener('pointermove', mv); cv.removeEventListener('pointerleave', lv); };
  });
</script>

<FigureCard {title} aspect="720 / 420" narrowAspect="360 / 520"
  label="A cloud of points shaped like a spiral. As the timeline runs the points dissolve into round Gaussian noise, then the reverse process gathers them back into the spiral. Arrows show the denoising direction."
  legend={[{ label: 'data position along the curve', color: '--viz-seq-4' }, { label: 'score (denoising direction)', color: '--viz-gradient', shape: 'line' }, { label: 'noise share', color: '--viz-forward', shape: 'line' }]}
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Diffusion time"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = lo === 0 ? 0 : lo; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next num">τ = {uiTau.toFixed(2)} · ᾱ = {alphaBar(uiTau).toFixed(3)} · signal-to-noise {(alphaBar(uiTau) / Math.max(1e-9, 1 - alphaBar(uiTau))).toFixed(2)}{cursorText ? ' · ' + cursorText : ' · hover the plot to read the score'}</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls}
      <div class="ctl">
        <fieldset class="seg"><legend class="sr-only">Reverse sampler</legend>
          <label><input type="radio" name="s-{uid}" value="ode" bind:group={smp} /><span>Deterministic (ODE)</span></label>
          <label><input type="radio" name="s-{uid}" value="sde" bind:group={smp} /><span>Stochastic (SDE)</span></label>
        </fieldset>
        <label><input type="checkbox" bind:checked={arrows} /> score arrows</label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  canvas { touch-action: pan-y; }
  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
