<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar } from '../_shared';
  import {
    BASE, DEPTH, WIDTH, SIGNAL_MARKERS, SYMMETRY_MARKERS, T, frameAt, initialPreset,
    presetScale, signalStats, signalSizes, symmetryExperiment,
    type Act, type Mode, type Preset, type SymmetryCase,
  } from './sim';

  interface Props {
    mode?: Mode;
    title?: string;
    controls?: boolean;
  }
  let { mode = 'signals', title = '', controls: showControls = true }: Props = $props();
  const init = untrack(() => ({ mode, title: title || defaultTitle(mode) }));

  let act = $state<Act>(init.mode === 'he' ? 'relu' : 'tanh');
  let preset = $state<Preset>(initialPreset(init.mode));
  let scale = $state(presetScale(initialPreset(init.mode)));
  let symmetryCase = $state<SymmetryCase>('matched');
  let playing = $state(false);
  let speed = $state(0.5);
  let uiT = $state(0);
  let phase = $state('');
  let narration = $state('The displayed statistics come from a completed seeded forward and backward pass.');

  let cv: HTMLCanvasElement;
  let tl: ReturnType<typeof Timeline>;
  const uid = $props.id();
  let t = 0;
  let poke = () => {};
  const markers = init.mode === 'symmetry' ? SYMMETRY_MARKERS : SIGNAL_MARKERS;
  const stats = $derived(signalStats(BASE, act, scale));
  const sizes = $derived(signalSizes(stats));
  const reveal = $derived(frameAt(uiT, init.mode));
  const symmetryCurrent = $derived(symmetryExperiment(symmetryCase, frameAt(uiT, 'symmetry').updateProgress));
  const presetNames: Record<Preset, string> = {
    zero: 'All zero', tiny: 'Tiny', xavier: 'Xavier', he: 'He', large: 'Large', custom: 'Custom scale',
  };

  function defaultTitle(mode: Mode): string {
    if (mode === 'symmetry') return 'When do identical neurons stay identical?';
    if (mode === 'xavier') return 'Xavier scale through a 12-layer tanh network';
    if (mode === 'he') return 'He scale through a 12-layer ReLU network';
    return 'Follow activation variance and gradient size across layers';
  }

  const ratio = (v: number) => v === 0 ? '0' : v >= 0.01 && v < 100 ? v.toFixed(2) : v.toExponential(1);

  function selectPreset(value: Preset) {
    preset = value;
    if (value !== 'custom') scale = presetScale(value);
    poke();
  }

  function renderSignals(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, width: number, height: number, colors: ReturnType<typeof readColors>, code: string, body: string) {
    const layers = [0, 2, 4, 6, 8, 10, 12];
    const left = 30, right = width - 30;
    const x = (i: number) => left + i / (layers.length - 1) * (right - left);
    const label = (value: string, px: number, py: number, color: string, size = 11, align: CanvasTextAlign = 'center') => {
      ctx.fillStyle = color; ctx.font = `500 ${size}px ${code}`; ctx.textAlign = align; ctx.fillText(value, px, py);
    };
    label('Forward → signal size', 15, 24, colors.text.hex, 12, 'left');
    label('1× = size at the input · log scale', 15, 42, colors.muted.hex, 10, 'left');
    label('Backward ← gradient size', 15, 222, colors.text.hex, 12, 'left');
    label('1× = size at the output · log scale', 15, 240, colors.muted.hex, 10, 'left');
    // Log scale around 1×: growth rises above the dashed line, shrinkage hangs below it.
    const perDecade = 16, up = 2, down = 4;
    for (const [values, reveal, one, color, reverse] of [
      [sizes.forward, fr.forwardReveal, 100, colors.fwd.hex, false],
      [sizes.backward, fr.backwardReveal, 292, colors.grad.hex, true],
    ] as const) {
      ctx.fillStyle = colors.rule.hex;
      for (let k = -down; k <= up; k++) if (k) { ctx.globalAlpha = 0.45; ctx.fillRect(15, one - k * perDecade, width - 30, 0.6); }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = colors.muted.hex; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(15, one); ctx.lineTo(width - 15, one); ctx.stroke(); ctx.setLineDash([]);
      label('1×', width - 15, one - 3, colors.muted.hex, 9, 'right');
      layers.forEach((layer, i) => {
        const visible = reveal * DEPTH >= (reverse ? DEPTH - layer : layer);
        if (visible) {
          const value = values[layer];
          const decades = value > 0 ? Math.log10(value) : -Infinity;
          const shown = Math.max(-down, Math.min(up, decades));
          const h = shown * perDecade;
          ctx.fillStyle = color; ctx.fillRect(x(i) - 9, h >= 0 ? one - h : one, 18, Math.max(1, Math.abs(h)));
          if (decades > up) { ctx.beginPath(); ctx.moveTo(x(i), one - up * perDecade - 7); ctx.lineTo(x(i) - 5, one - up * perDecade); ctx.lineTo(x(i) + 5, one - up * perDecade); ctx.fill(); }
          if (decades < -down) { ctx.beginPath(); ctx.moveTo(x(i), one + down * perDecade + 7); ctx.lineTo(x(i) - 5, one + down * perDecade); ctx.lineTo(x(i) + 5, one + down * perDecade); ctx.fill(); }
          const number = value === 0 ? '0' : value < 0.01 || value >= 100 ? value.toExponential(0) : value.toFixed(2);
          label(number + '×', x(i), one + down * perDecade + 20, color, width < 500 ? 9 : 11);
        }
        label(layer === 0 ? 'Input' : layer === DEPTH ? 'Out 12' : 'L' + layer, x(i), one + down * perDecade + 35, colors.muted.hex, 10);
      });
    }
    label('Each gridline is a factor of 10. Triangles: beyond 100× or below 0.0001×.', width / 2, height - 6, colors.muted.hex, width < 500 ? 8 : 10);
  }

  function renderSymmetry(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, width: number, height: number, colors: ReturnType<typeof readColors>, code: string) {
    const exp = symmetryExperiment(symmetryCase, fr.updateProgress);
    const inputX = 58, hiddenX = width * 0.5, outputX = width - 66;
    const inputY = [height * 0.35, height * 0.68];
    const hiddenY = [height * 0.24, height * 0.5, height * 0.76];
    const node = (x: number, y: number, label: string, value: string, color: string) => {
      ctx.fillStyle = colors.raised.hex; ctx.strokeStyle = colors.rule.hex; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.roundRect(x - 31, y - 25, 62, 50, 5); ctx.fill(); ctx.stroke();
      ctx.fillStyle = colors.muted.hex; ctx.textAlign = 'center'; ctx.font = `10px ${code}`; ctx.fillText(label, x, y - 6);
      ctx.fillStyle = color; ctx.font = `600 12px ${code}`; ctx.fillText(value, x, y + 12);
    };
    const weight = (i: number) => exp.initialWeights[i] * (1 - fr.updateProgress) + exp.updatedWeights[i] * fr.updateProgress;
    ctx.lineWidth = 1;
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 2; i++) {
        const v = weight(j * 2 + i);
        ctx.strokeStyle = v > 0 ? colors.pos.hex : v < 0 ? colors.neg.hex : colors.muted.hex;
        ctx.globalAlpha = 0.35 + 0.65 * (symmetryCase === 'matched' ? fr.forwardReveal : 1);
        ctx.lineWidth = Math.min(4, 1 + Math.abs(v) * 4);
        ctx.beginPath(); ctx.moveTo(inputX + 31, inputY[i]); ctx.lineTo(hiddenX - 32, hiddenY[j]); ctx.stroke();
      }
      const v = weight(6 + j);
      ctx.strokeStyle = v > 0 ? colors.pos.hex : v < 0 ? colors.neg.hex : colors.muted.hex;
      ctx.globalAlpha = 0.35 + 0.65 * fr.forwardReveal;
      ctx.lineWidth = Math.min(4, 1 + Math.abs(v) * 4);
      ctx.beginPath(); ctx.moveTo(hiddenX + 32, hiddenY[j]); ctx.lineTo(outputX - 32, height * 0.5); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    node(inputX, inputY[0], 'x₁', '1.00', colors.fwd.hex);
    node(inputX, inputY[1], 'x₂', '0.50', colors.fwd.hex);
    hiddenY.forEach((y, j) => node(hiddenX, y, `h${j + 1}`, exp.shown.hidden[j].toFixed(3), colors.fwd.hex));
    node(outputX, height * 0.5, 'prediction', exp.shown.prediction.toFixed(3), colors.fwd.hex);
    // Reveal the already-computed reverse pass; only geometry/opacity changes.
    if (fr.backwardReveal > 0) {
      ctx.strokeStyle = colors.grad.hex; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      const segment = (ax: number, ay: number, bx: number, by: number, progress: number) => {
        const p = Math.min(1, Math.max(0, progress));
        if (!p) return;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + (bx - ax) * p, ay + (by - ay) * p); ctx.stroke();
      };
      for (let j = 0; j < 3; j++) {
        segment(outputX - 32, height * 0.5, hiddenX + 32, hiddenY[j], fr.backwardReveal * 2);
        for (let i = 0; i < 2; i++) segment(hiddenX - 32, hiddenY[j], inputX + 31, inputY[i], fr.backwardReveal * 2 - 1);
        ctx.fillStyle = colors.grad.hex; ctx.textAlign = 'center'; ctx.font = `10px ${code}`;
        ctx.globalAlpha = Math.min(1, fr.backwardReveal * 2);
        ctx.fillText(`dL/dz${j + 1} ${exp.before.hiddenGradient[j * 2].toFixed(3)}`, hiddenX, hiddenY[j] + 39);
        ctx.globalAlpha = 1;
      }
      ctx.setLineDash([]);
    }
    ctx.fillStyle = colors.muted.hex; ctx.textAlign = 'center'; ctx.font = `11px ${code}`;
    ctx.fillText('one shared input example', width / 2, 25);
    ctx.fillText(`target = ${0.8.toFixed(2)}   ·   loss = ${exp.shown.loss.toFixed(4)}`, width / 2, height - 17);
  }

  function render(dt: number, now: number) {
    const rect = cv.getBoundingClientRect();
    if (!rect.width || !rect.height) return false;
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext('2d');
    if (!ctx) return false;
    const logicalWidth = rect.width < 500 ? 360 : 720;
    const logicalHeight = init.mode === 'symmetry' ? 410 : 410;
    ctx.setTransform(cv.width / logicalWidth, 0, 0, cv.height / logicalHeight, 0, 0);
    ctx.clearRect(0, 0, logicalWidth, logicalHeight);
    uiT = t;
    const fr = frameAt(t, init.mode);
    if (init.mode === 'symmetry') renderSymmetry(ctx, fr, logicalWidth, logicalHeight, colors, codeFont);
    else renderSignals(ctx, fr, logicalWidth, logicalHeight, colors, codeFont, bodyFont);
    tl?.paint(t);
    if (fr.phase !== phase) {
      phase = fr.phase;
      narration = init.mode === 'symmetry'
        ? fr.phase === 'forward'
          ? symmetryCase === 'matched' ? 'Forward pass: identical hidden units receive identical inputs and weights.'
            : symmetryCase === 'perturbed' ? 'Forward pass: one hidden unit has a slightly different starting row.'
              : 'Forward pass: with both layers zero, every hidden activation and the prediction are zero.'
          : fr.phase === 'backward'
            ? symmetryCase === 'matched' ? 'Backward pass: equal outputs and activations give each copy the same gradient.'
              : symmetryCase === 'perturbed' ? 'Backward pass: the perturbed unit receives a different gradient.'
                : 'Backward pass: zero activations and zero output weights give every weight zero gradient.'
            : 'SGD applies the computed update. Inspect the numerical gradients and updated weights below.'
        : fr.phase === 'forward' ? 'Forward: watch the signal travel from the input to layer 12. Bars below the dashed line mean the signal has shrunk; each gridline is another factor of 10.'
          : 'Backward: watch the gradient travel from layer 12 back to the input. Bars below the dashed line mean early layers get a weaker learning signal.';
    }
    return playing;
  }

  let colors: ReturnType<typeof readColors>;
  let codeFont = '';
  let bodyFont = '';

  $effect(() => { act; preset; scale; symmetryCase; playing; speed; poke(); });

  onMount(() => {
    const reduced = prefersReducedMotion();
    colors = readColors({
      fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg',
      text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent',
    });
    codeFont = readVar('--font-code');
    bodyFont = readVar('--font-body');
    const drawOnce = () => render(0, performance.now());
    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      return render(dt, now);
    }, { onView: (inView, first) => { if (inView && first && !reduced.matches) playing = true; } });
    poke = loop.wake;
    const offTheme = onThemeChange(() => {
      colors = readColors({
        fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg',
        text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent',
      });
      codeFont = readVar('--font-code'); bodyFont = readVar('--font-body'); poke();
    });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    const ro = new ResizeObserver(drawOnce);
    ro.observe(cv.parentElement!);
    document.fonts.ready.then(() => poke());
    render(0, performance.now());
    return () => { loop.destroy(); offTheme(); ro.disconnect(); reduced.removeEventListener('change', onReduce); };
  });

  function seek(to: number) { playing = false; t = to; uiT = to; phase = ''; poke(); }
  function reset() { playing = false; seek(0); }
  function step(to: number) { seek(to); }
</script>

<FigureCard
  title={init.title}
  legend={init.mode === 'symmetry'
    ? [{ label: 'activation / prediction', color: '--viz-forward' }, { label: 'positive weight', color: '--viz-pos' }, { label: 'negative weight', color: '--viz-neg' }]
    : [{ label: 'Signal size (forward)', color: '--viz-forward', shape: 'line' }, { label: 'Gradient size (backward)', color: '--viz-gradient', shape: 'line' }]}
  role="group"
  aspect="720 / 410"
  narrowAspect="360 / 410"
  label={init.mode === 'symmetry'
    ? 'A two input, three hidden unit, one output tanh network showing equal hidden unit weights, activations and an SGD update.'
    : 'Layer bars compare signal spread with the input and gradient size with the output, sampled every two layers.'}
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers}
      label="Timeline of forward pass, backward pass and update" onSeek={(to) => seek(to)} onStep={step} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    {#if init.mode === 'symmetry'}
      {@const exp = symmetryCurrent}
      <p class="readout"><strong>{symmetryCase === 'matched' ? 'Matched nonzero weights' : symmetryCase === 'perturbed' ? 'One hidden row perturbed' : 'Both layers all zero'}</strong> · prediction {exp.shown.prediction.toFixed(4)} · loss {exp.shown.loss.toFixed(4)}</p>
      <p class="readout">Hidden activations: ({exp.shown.hidden.map((v) => v.toFixed(4)).join(', ')}).</p>
      <p class="readout">Hidden weight gradients: {exp.shown.hiddenGradient.map((v) => v.toFixed(4)).join(', ')}. Output weight gradients: {exp.shown.outputGradient.map((v) => v.toFixed(4)).join(', ')}.</p>
      <p class="note">After one gradient step the loss is {exp.after.loss.toFixed(4)}.</p>
      <details class="table-wrap">
        <summary>Numeric weights before and after one SGD step</summary>
        <table>
          <caption>Step size 0.20, target 0.80; the table lists all nine weights.</caption>
          <thead><tr><th scope="col">Weight</th><th scope="col">Start</th><th scope="col">Gradient</th><th scope="col">After SGD</th></tr></thead>
          <tbody>
            {#each exp.initialWeights as value, i (i)}
              <tr><th scope="row">{i < 6 ? `w${Math.floor(i / 2) + 1}${i % 2 + 1}` : `v${i - 5}`}</th><td>{value.toFixed(4)}</td><td>{(i < 6 ? exp.before.hiddenGradient[i] : exp.before.outputGradient[i - 6]).toFixed(4)}</td><td>{exp.updatedWeights[i].toFixed(4)}</td></tr>
            {/each}
          </tbody>
        </table>
      </details>
    {:else}
      <p class="readout"><strong>After 12 layers:</strong> the signal is <span class="num">{reveal.forwardReveal >= 1 ? ratio(sizes.forward[DEPTH]) + '×' : '…'}</span> its input size. The gradient reaching the input is <span class="num">{reveal.backwardReveal >= 1 ? ratio(sizes.backward[0]) + '×' : '…'}</span> its size at the output.</p>
      <p class="note">Each bar is one layer. Signal size is the standard deviation of that layer's activations; gradient size is the length of its gradient vector.</p>
      <details class="table-wrap">
        <summary>Show every layer's numbers</summary>
        <p class="note">Weight scale s = {scale.toFixed(2)} · entries use s / √{WIDTH}. Target weight variance {stats.theoreticalWeightVariance.toExponential(2)}; this random draw's first layer has {stats.measuredWeightVariance[0].toExponential(2)}.</p>
        <table>
          <caption>One forward and backward pass through this network</caption>
          <thead><tr><th scope="col">Layer</th><th scope="col">Activation mean</th><th scope="col">Activation variance</th><th scope="col">Gradient norm</th></tr></thead>
          <tbody>
            {#each stats.activationVariance as variance, layer (layer)}
              <tr><th scope="row">{layer}</th><td>{stats.activationMean[layer].toExponential(2)}</td><td>{variance.toExponential(2)}</td><td>{stats.gradientNorm[layer].toExponential(2)}</td></tr>
            {/each}
          </tbody>
        </table>
      </details>
    {/if}
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {init.mode === 'symmetry' ? 'Compare Equal nonzero with Break symmetry. Watch whether the three hidden units still match after the update.' : init.mode === 'xavier' ? 'Switch the activation from tanh to ReLU and watch the bars sink below the dashed line.' : init.mode === 'he' ? 'Drag Scale s a little below and then above He (1.41) and replay.' : 'Switch Initialization from Tiny to Large, then try ReLU.'}</p>
    {#if showControls}
      <div class="ctl">
        {#if init.mode === 'symmetry'}
          <fieldset class="seg">
            <legend>Initialization example</legend>
            <label><input type="radio" name="symmetry-{uid}" value="matched" bind:group={symmetryCase} /><span>Equal nonzero</span></label>
            <label><input type="radio" name="symmetry-{uid}" value="perturbed" bind:group={symmetryCase} /><span>Break symmetry</span></label>
            <label><input type="radio" name="symmetry-{uid}" value="zero" bind:group={symmetryCase} /><span>All zero</span></label>
          </fieldset>
        {:else}
          <fieldset class="seg">
            <legend>Hidden activation</legend>
            <label><input type="radio" name="activation-{uid}" value="tanh" bind:group={act} /><span>tanh</span></label>
            <label><input type="radio" name="activation-{uid}" value="relu" bind:group={act} /><span>ReLU</span></label>
          </fieldset>
          <label class="preset">Initialization
            <select value={preset} onchange={(event) => selectPreset(event.currentTarget.value as Preset)}>
              <option value="zero">All zero</option><option value="tiny">Tiny</option><option value="xavier">Xavier</option>
              <option value="he">He</option><option value="large">Large</option><option value="custom">Custom scale</option>
            </select>
          </label>
          <label class="scale">Scale s
            <input type="range" min="0" max="3" step="any" bind:value={scale} oninput={() => (preset = 'custom')} aria-label="Weight scale multiplier" />
            <output class="num">{scale.toFixed(2)}</output>
          </label>
          <span class="formula">Xavier s = 1 · He s = √2</span>
        {/if}
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .try { padding: var(--s-3) var(--s-4) 0; margin: 0; line-height: 1.6; }
  summary { cursor: pointer; }
  .say, .readout, .note { margin: 0 0 var(--s-2); }
  .say { font: 0.92rem / 1.45 var(--font-ui); }
  .readout { font: var(--step--1) / 1.5 var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .note { color: var(--text-muted); font-size: var(--step--1); }
  .table-wrap { margin-top: var(--s-2); }
  table { width: 100%; border-collapse: collapse; font: 0.78rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; }
  caption { text-align: left; color: var(--text-muted); padding: var(--s-2) 0; }
  th, td { text-align: right; padding: 0.25rem 0.35rem; border-bottom: 1px solid var(--rule); }
  th:first-child { text-align: left; }
  thead th { color: var(--text-muted); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-3) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 8.5rem; accent-color: var(--accent); }
  .ctl output { min-width: 3.4rem; color: var(--text); }
  .ctl select { color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3rem 0.45rem; font: inherit; }
  .seg { display: inline-flex; flex-wrap: wrap; gap: var(--s-2); border: 1px solid var(--rule); border-radius: var(--r-2); padding: var(--s-2); margin: 0; }
  .seg legend { padding-inline: var(--s-1); color: var(--text-muted); font-size: 0.8rem; }
  .seg label { position: relative; margin: 0; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { display: block; padding: 0.3rem 0.55rem; cursor: pointer; border-radius: var(--r-1); }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
  .formula { color: var(--text-muted); font: 0.8rem var(--font-code); }
  @media (max-width: 40em) {
    .ctl { gap: var(--s-3); }
    .scale { width: 100%; }
    .ctl input[type='range'] { flex: 1; min-width: 6rem; }
    table { font-size: 0.68rem; }
    th, td { padding: 0.2rem 0.15rem; }
  }
</style>
