<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import { ACTS, T, MARKERS, makeBase, profile, frameAt, type Act } from './sim';

  interface Props {
    /** Hidden activation. */
    activation?: Act;
    /** Weight scale s: W = (s / sqrt(width)) * N(0,1). 1 is Glorot/LeCun, sqrt 2 is He. */
    scale?: number;
    /** Residual connections: h_l = h_{l-1} + phi(z_l). */
    skip?: boolean;
    /** Number of layers. */
    depth?: number;
    title?: string;
    /** Show the activation switch and scale slider. */
    controls?: boolean;
  }
  let { activation = 'sigmoid', scale = 1, skip: skipProp = false, depth = 30, title = 'Gradient size at every layer, log scale', controls: showControls = true }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ depth, title }));
  const base = makeBase(init.depth);
  let act = $state<Act>(untrack(() => activation));
  let s = $state(untrack(() => scale));
  let skip = $state(untrack(() => skipProp));
  const prof = $derived(profile(base, { act, scale: s, skip }));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let vertical = $state(false);
  let narration = $state('');
  let phase = '';

  const LO = -22, HI = 12, SPAN = HI - LO;
  const exp10 = (x: number) => {
    if (x <= -99) return '0';
    const e = Math.floor(x), m = 10 ** (x - e);
    return `${m.toFixed(1)}e${e < 0 ? '−' : ''}${Math.abs(e)}`;
  };
  const LEGEND = [{ label: 'forward activations', color: '--viz-forward' }, { label: '‖∂L/∂h‖ (gradient size)', color: '--viz-gradient' }];
  const summary = $derived(
    `‖∂L/∂h₀‖ = ${exp10(prof.log[0])} after ${init.depth} layers · geometric-mean gain per layer ×${prof.gain.toFixed(3)} · mean φ′ = ${(prof.meanDeriv.reduce((a, b) => a + b, 0) / init.depth).toFixed(2)}`,
  );
  const verdict = $derived(prof.log[0] < -4 ? 'vanishing' : prof.log[0] > 4 ? 'exploding' : 'stable');

  let t = 0;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, poke = () => {};
  $effect(() => { act; s; skip; vertical; playing; speed; poke(); });
  $effect(() => { verdict; phase = ''; });

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const NAMES = { fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft' };
    let c = readColors(NAMES), code = readVar('--font-code'), body = readVar('--font-body');
    let lastUi = 0;
    if (reduced.matches) t = T;
    const D = init.depth;

    const lay = () => vertical
      ? { w: 360, h: 470, x: 50, y: 62, pw: 296, ph: 350, fs: 11 }
      : { w: 720, h: 360, x: 70, y: 48, pw: 632, ph: 268, fs: 12 };

    const text = (str: string, x: number, y: number, col: string, size: number, align: CanvasTextAlign = 'left', a = 1, mono = true) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = mono ? `600 ${size}px ${code}` : `italic ${size}px ${body}`;
      ctx.fillText(str, x, y);
    };

    function render(dt: number, now: number) {
      // Reveal the actual computed profile, without blending unrelated runs.
      const disp = prof.log;
      const L = lay(), sc = cv.width / L.w;
      ctx.setTransform(sc, 0, 0, sc, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      const fr = frameAt(t, D);
      const colW = L.pw / (D + 1), by = L.y + (HI / SPAN) * L.ph;
      const Y = (lg: number) => by - (Math.max(LO, Math.min(HI, lg)) / SPAN) * L.ph;
      const X = (l: number) => L.x + (l + 0.5) * colW;

      // grid + ticks
      ctx.lineWidth = 1;
      for (const e of [10, 0, -10, -20]) {
        ctx.globalAlpha = e === 0 ? 1 : 0.6; ctx.strokeStyle = c.rule.hex; ctx.setLineDash(e === 0 ? [] : [2, 4]);
        ctx.beginPath(); ctx.moveTo(L.x, Y(e)); ctx.lineTo(L.x + L.pw, Y(e)); ctx.stroke(); ctx.setLineDash([]);
        text(e === 0 ? '1' : `1e${e < 0 ? '−' : ''}${Math.abs(e)}`, L.x - 6, Y(e) + 4, c.muted.hex, L.fs, 'right');
      }
      text('exploding', L.x + 4, L.y + 12, c.muted.hex, L.fs + 1, 'left', 1, false);
      text('vanishing', L.x + 4, L.y + L.ph - 6, c.muted.hex, L.fs + 1, 'left', 1, false);
      // forward strip
      text('forward', L.x - 6, L.y - 26, c.fwd.hex, L.fs, 'right', 1, false);
      for (let l = 0; l <= D; l++) {
        const on = Math.min(1, Math.max(0, fr.fwd - l));
        ctx.globalAlpha = 0.25 + 0.75 * on; ctx.fillStyle = on > 0.5 ? c.fwd.hex : c.rule.hex;
        ctx.beginPath(); ctx.arc(X(l), L.y - 30, Math.max(2, colW * 0.22), 0, 6.2832); ctx.fill();
      }
      // bars (a white cap marks values beyond the axis)
      for (let l = 0; l <= D; l++) {
        const r = fr.reveal[l];
        if (r <= 0.005) continue;
        const v = disp[l] * r, yy = Y(v), bw = Math.max(2, colW * 0.62);
        ctx.globalAlpha = 0.9; ctx.fillStyle = c.grad.hex;
        ctx.fillRect(X(l) - bw / 2, Math.min(by, yy), bw, Math.max(1.5, Math.abs(yy - by)));
        if (v <= LO || v >= HI) { ctx.fillStyle = c.text.hex; ctx.globalAlpha = 1; ctx.fillRect(X(l) - bw / 2, v <= LO ? yy - 3 : yy, bw, 3); }
      }
      // backward front
      if (fr.phase === 'backward') {
        const xf = L.x + (D + 1 - fr.bwd + 0.5) * colW;
        ctx.globalAlpha = 0.9; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(xf, L.y - 6); ctx.lineTo(xf, L.y + L.ph); ctx.stroke(); ctx.setLineDash([]);
      }
      // x axis
      for (let l = 0; l <= D; l += 5) text(String(l), X(l), L.y + L.ph + 16, c.muted.hex, L.fs, 'center');
      text('layer l   (0 = input, ' + D + ' = output and loss)', L.x + L.pw / 2, L.y + L.ph + 34, c.muted.hex, L.fs + 1, 'center', 1, false);
      ctx.globalAlpha = 1;

      tl.paint(t);
      if (fr.phase !== phase) {
          phase = fr.phase;
          narration = phase === 'forward' ? 'Forward: the input passes through every layer.'
            : phase === 'backward' ? `Backward: at each layer the gradient is multiplied by ${skip ? 'I + ' : ''}Wᵀ diag(φ′). Watch the bars reveal the completed gradient profile toward the input.`
            : `Result: ${verdict}. The gradient at the first layer is ${exp10(prof.log[0])} times the size at the output.`;
      }
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return false;
    }

    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      return render(dt, now) || playing;
    }, { onView: (inView, first) => { if (inView && first && !reduced.matches) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = Math.min(2, Math.max(1, devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { vertical = e.contentRect.width < 540; fit(); render(0, performance.now()); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(NAMES); code = readVar('--font-code'); body = readVar('--font-body'); poke(); });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });
</script>

<FigureCard title={init.title} legend={LEGEND} role="img" aspect="720 / 360" narrowAspect="360 / 470"
  label="Bar chart of the gradient size at each of {init.depth} layers on a log scale, growing or shrinking toward the input as the backward wave passes.">
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline of the forward and backward waves"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to >= T ? 0 : to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="num muted">{summary}</p>
    <p class="muted">Readout: completed backward pass. Bars progressively reveal those final norms, not intermediate gradient estimates.</p>
    <details><summary>Numerical gradient profile</summary><ol>{#each prof.norms as norm, layer (layer)}<li>Layer {layer}: {norm.toExponential(3)}</li>{/each}</ol></details>
    <p class="math" aria-hidden="true">∂L/∂h₀ = J₁ᵀ J₂ᵀ ⋯ J{init.depth}ᵀ · ∂L/∂h{init.depth}, &nbsp; Jₗᵀ = {skip ? 'I + ' : ''}Wₗᵀ diag(φ′(zₗ))</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls}
      <div class="ctl">
        <fieldset class="seg">
          <legend class="sr-only">Activation function</legend>
          {#each ACTS as a (a)}
            <label><input type="radio" name="act-{uid}" value={a} bind:group={act} /><span>{a === 'relu' ? 'ReLU' : a}</span></label>
          {/each}
        </fieldset>
        <label>weight scale s
          <input type="range" min="0.3" max="3.5" step="0.05" bind:value={s} aria-label="Weight scale" />
          <output class="num">{s.toFixed(2)}</output>
        </label>
        <label class="chk"><input type="checkbox" bind:checked={skip} /> skip connections</label>
        <span class="presets">
          <button type="button" class="btn" onclick={() => (s = 1)}>Glorot s = 1</button>
          <button type="button" class="btn" onclick={() => (s = Math.SQRT2)}>He s = √2</button>
        </span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say, .muted, .math { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .muted { color: var(--text-muted); font-size: var(--step--1); }
  .math { color: var(--text-muted); font: 0.86rem var(--font-body); font-style: italic; margin-top: 0.2em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 10rem; accent-color: var(--accent); }
  .ctl output { min-width: 2.8rem; color: var(--text); }
  .presets { display: inline-flex; gap: var(--s-2); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
