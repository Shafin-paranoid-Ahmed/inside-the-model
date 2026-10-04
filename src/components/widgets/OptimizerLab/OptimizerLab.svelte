<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar, reducedMotion } from '../_shared';
  import { DEFAULTS, DOMAIN, DURATION, EXAMPLE_COUNT, OPTIMIZERS, STEPS, contourSegments, frameAt, localLevels, makeRace, objective, teachingFrameAt, type Contour, type Landscape, type Mode, type Optimizer, type Point } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'race', title = 'Optimizer comparison' }: Props = $props();
  const uid = $props.id();
  let optimizer = $state<Optimizer | 'Race'>(untrack(() => mode === 'race' ? 'Race' : mode === 'bias' ? 'Adam' : mode === 'momentum' ? 'Momentum' : mode === 'scaling' ? 'RMSProp' : 'SGD'));
  let landscape = $state<Landscape>('ravine');
  let lr = $state(DEFAULTS.lr);
  let batchSize = $state(untrack(() => mode === 'noise' || mode === 'race' ? DEFAULTS.batchSize : EXAMPLE_COUNT));
  let noise = $state(DEFAULTS.noise);
  let update = $state(untrack(() => mode === 'momentum' || mode === 'scaling' ? 8 : 1));
  const initialTime = 0;
  let playing = $state(false), speed = $state(1), uiT = $state(initialTime);
  let t = initialTime, lastUi = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  const markers = $derived(optimizer === 'Race'
    ? [{ at: 0, label: 'Early' }, { at: DURATION / 2, label: 'Middle' }, { at: 10, label: 'Late' }]
    : [{ at: 0, label: 'Slope' }, { at: 3, label: 'Update' }, { at: 7, label: 'Move' }]);
  const legend = OPTIMIZERS.map((name, i) => ({ label: name, color: ['--viz-forward', '--viz-gradient', '--viz-pos', '--viz-neg'][i], shape: 'line' as const }));
  const runs = $derived.by(() => makeRace({ landscape, lr, batchSize, noise }));
  const frames = $derived.by(() => Object.fromEntries(OPTIMIZERS.map((name) => [name, frameAt(uiT, runs[name], landscape)])) as Record<Optimizer, ReturnType<typeof frameAt>>);
  const currentFrame = $derived(frames[optimizer === 'Race' ? 'SGD' : optimizer]);
  const teaching = $derived(teachingFrameAt(uiT, runs[optimizer === 'Race' ? 'SGD' : optimizer], { landscape, lr, batchSize, noise }, update));
  const fmt = (n: number) => Number.isFinite(n) ? n.toFixed(3) : 'diverged';
  const raceNumber = (n: number) => !Number.isFinite(n) ? 'non-finite' : Math.abs(n) >= 10000 || (n !== 0 && Math.abs(n) < 0.001) ? n.toExponential(2) : n.toFixed(3);
  let ctx: CanvasRenderingContext2D, bgCtx: CanvasRenderingContext2D, bgCanvas: HTMLCanvasElement;
  let colors = {} as ReturnType<typeof readColors>;
  let width = 0, height = 0, cachedLandscape: Landscape | null = null;
  let contours: Contour[] = [];
  let localKey = '', localContours: Contour[] = [], localStartLevel = 0;
  const colorNames = ['fwd', 'grad', 'pos', 'neg'] as const;

  function reset() { playing = false; t = uiT = initialTime; wake(); }
  function settingChanged() { if (optimizer !== 'Race') lr = Math.min(0.16, lr); reset(); }
  function seek(to: number) { t = uiT = to; playing = false; wake(); }
  function colorsNow() {
    colors = readColors({ fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg', rule: '--rule', muted: '--text-muted', accent: '--accent', raised: '--bg-raised', text: '--text' });
  }
  function sizeCanvas() {
    const rect = cv.getBoundingClientRect(), dpr = Math.min(2, Math.max(1, devicePixelRatio || 1));
    width = rect.width; height = rect.height;
    const w = Math.round(width * dpr), h = Math.round(height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; bgCanvas.width = w; bgCanvas.height = h; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cachedLandscape = null;
  }
  const sx = (x: number) => ((x + DOMAIN) / (2 * DOMAIN)) * width;
  const sy = (y: number) => ((DOMAIN - y) / (2 * DOMAIN)) * height;

  function buildContours() {
    const levels = landscape === 'ravine' ? [1.15, 1.25, 1.45, 1.8, 2.5, 4, 7, 13] : [-2, -1, -0.5, 0, 0.5, 1, 2, 4];
    contours = contourSegments((x, y) => objective({ x, y }, landscape), { x0: -DOMAIN, x1: DOMAIN, y0: DOMAIN, y1: -DOMAIN }, levels);
  }

  function drawBackground() {
    if (cachedLandscape === landscape) return;
    cachedLandscape = landscape; buildContours();
    bgCtx.clearRect(0, 0, width, height);
    bgCtx.strokeStyle = colors.rule.hex; bgCtx.lineWidth = 1;
    for (let k = -3; k <= 3; k++) {
      bgCtx.beginPath(); bgCtx.moveTo(sx(k), sy(-DOMAIN)); bgCtx.lineTo(sx(k), sy(DOMAIN)); bgCtx.stroke();
      bgCtx.beginPath(); bgCtx.moveTo(sx(-DOMAIN), sy(k)); bgCtx.lineTo(sx(DOMAIN), sy(k)); bgCtx.stroke();
    }
    bgCtx.globalAlpha = 0.75;
    for (const contour of contours) {
      bgCtx.strokeStyle = contour.level === 0 ? colors.accent.hex : colors.muted.hex;
      bgCtx.lineWidth = contour.level === 0 ? 1.6 : 0.8;
      bgCtx.beginPath(); bgCtx.moveTo(sx(contour.points[0]), sy(contour.points[1])); bgCtx.lineTo(sx(contour.points[2]), sy(contour.points[3])); bgCtx.stroke();
    }
    bgCtx.globalAlpha = 1;
    bgCtx.fillStyle = colors.text.hex; bgCtx.font = `12px ${readVar('--font-code')}`;
    bgCtx.fillText(landscape === 'ravine' ? 'Vertical: y weight · ravine (bowl)' : 'Vertical: y weight · saddle', 8, 16);
    bgCtx.fillText('Horizontal: x weight (−3 to +3)', 8, height - 8);
  }

  function draw(frameT: number) {
    if (!width || !height) return;
    if (optimizer !== 'Race') { drawUpdate(frameT); return; }
    drawBackground();
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(bgCanvas, 0, 0, width, height);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 24, width, Math.max(0, height - 48)); ctx.clip();
    OPTIMIZERS.forEach((name, i) => {
      const run = runs[name], count = Math.min(STEPS, Math.floor(frameT / DURATION * STEPS));
      ctx.strokeStyle = colors[colorNames[i]].hex; ctx.lineWidth = optimizer === 'Race' ? 2 : name === optimizer ? 2.6 : 1;
      ctx.globalAlpha = optimizer === 'Race' || name === optimizer ? 0.88 : 0.18;
      ctx.beginPath();
      for (let step = 0; step <= count; step++) {
        const p = run.states[step].point, x = sx(p.x), y = sy(p.y);
        if (!step) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      const f = frameAt(frameT, run, landscape), x = sx(f.point.x), y = sy(f.point.y);
      if (Number.isFinite(x) && Number.isFinite(y)) ctx.lineTo(x, y);
      ctx.stroke(); ctx.globalAlpha = 1;
      if (x >= 0 && x <= width && y >= 0 && y <= height) {
        ctx.fillStyle = colors[colorNames[i]].hex; ctx.beginPath(); ctx.arc(x, y, name === optimizer ? 5.5 : 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = colors.raised.hex; ctx.lineWidth = 1.5; ctx.stroke();
      }
    });
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function drawUpdate(frameT: number) {
    const f = teachingFrameAt(frameT, runs[optimizer === 'Race' ? 'SGD' : optimizer], { landscape, lr, batchSize, noise }, update);
    ctx.clearRect(0, 0, width, height);
    const origin = { x: width * 0.48, y: height * 0.53 };
    const vectors = [f.plainMove, f.move, ...(mode === 'noise' ? [f.exactMove, ...f.exampleMoves] : [])];
    const scale = Math.min(width * 0.31, height * 0.29) * (lr / 0.16) / Math.max(0.00001, ...vectors.map((p) => Math.hypot(p.x, p.y)));
    const at = (p: Point) => ({ x: origin.x + p.x * scale, y: origin.y - p.y * scale });
    function arrow(v: Point, color: string, dashed = false, widthPx = dashed ? 1.7 : 3) {
      const end = at(v), angle = Math.atan2(end.y - origin.y, end.x - origin.x);
      if (![end.x, end.y].every(Number.isFinite)) return;
      ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = widthPx; ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y); ctx.stroke(); ctx.setLineDash([]);
      if (Math.hypot(end.x - origin.x, end.y - origin.y) < 2) return;
      ctx.beginPath(); ctx.moveTo(end.x, end.y); ctx.lineTo(end.x - 8 * Math.cos(angle - 0.4), end.y - 8 * Math.sin(angle - 0.4)); ctx.lineTo(end.x - 8 * Math.cos(angle + 0.4), end.y - 8 * Math.sin(angle + 0.4)); ctx.closePath(); ctx.fill();
    }
    // Equal-loss curves around the starting weights, in the same magnified coordinates as the arrows.
    const b = f.before.point, key = [landscape, optimizer, update, lr, batchSize, noise, width, height].join('|');
    if (key !== localKey) {
      const box = { x0: b.x - origin.x / scale, x1: b.x + (width - origin.x) / scale, y0: b.y + origin.y / scale, y1: b.y - (height - origin.y) / scale };
      const levels = localLevels(landscape, box, b);
      localStartLevel = levels[levels.length - 1];
      localContours = contourSegments((x, y) => objective({ x, y }, landscape), box, levels, 36);
      localKey = key;
    }
    const px = (x: number, y: number) => at({ x: x - b.x, y: y - b.y });
    for (const c of localContours) {
      const start = c.level === localStartLevel;
      ctx.strokeStyle = start ? colors.accent.hex : colors.muted.hex; ctx.lineWidth = start ? 1.6 : 0.8; ctx.globalAlpha = start ? 0.9 : 0.45;
      const p0 = px(c.points[0], c.points[1]), p1 = px(c.points[2], c.points[3]);
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = `12px ${readVar('--font-ui')}`; ctx.fillStyle = colors.text.hex;
    ctx.fillText(`One ${optimizer} update · update ${f.index}`, 12, 22);
    ctx.fillStyle = colors.muted.hex; ctx.fillText('Gray curves: equal loss', 12, 43);
    if (mode === 'noise') {
      ctx.globalAlpha = 0.35; for (const v of f.exampleMoves) arrow(v, colors.muted.hex, true);
      ctx.globalAlpha = 1; arrow(f.exactMove, colors.pos.hex, true);
    }
    if (f.time < 3) arrow({ x: -f.plainMove.x, y: -f.plainMove.y }, colors.grad.hex, false, 2.2);
    else {
      if (optimizer !== 'SGD') arrow(f.plainMove, colors.grad.hex, true);
      arrow(f.move, colors.fwd.hex);
    }
    ctx.strokeStyle = colors.text.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(origin.x, origin.y, 6, 0, Math.PI * 2); ctx.stroke();
    const p = at({ x: f.move.x * f.progress, y: f.move.y * f.progress });
    ctx.fillStyle = colors.fwd.hex; ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = colors.text.hex; ctx.fillText(f.phase, 12, height - 13);
  }

  onMount(() => {
    bgCanvas = document.createElement('canvas');
    ctx = cv.getContext('2d')!; bgCtx = bgCanvas.getContext('2d')!; colorsNow(); sizeCanvas(); draw(initialTime);
    const figure = cv.closest('[data-figure]') ?? cv;
    const observer = new ResizeObserver(sizeCanvas); observer.observe(figure);
    const loop = createLoop(figure, (dt, now) => {
      if (playing) t = advance({ t, playing, speed }, dt, { duration: DURATION }).t;
      uiT = t;
      draw(t);
      tl?.paint(t);
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const off = onThemeChange(() => { colorsNow(); cachedLandscape = null; wake(); });
    const motion = prefersReducedMotion();
    const onMotionChange = (event: MediaQueryListEvent) => { if (event.matches) playing = false; wake(); };
    motion.addEventListener('change', onMotionChange);
    return () => { observer.disconnect(); loop.destroy(); off(); motion.removeEventListener('change', onMotionChange); };
  });

  $effect(() => { optimizer; landscape; lr; batchSize; noise; update; playing; speed; wake(); });
</script>

<FigureCard {title} label="Optimizer paths or a magnified explanation of one recorded parameter update" role="group" aspect="16 / 9" narrowAspect="1 / 1" legend={optimizer === 'Race' ? legend : [
  { label: 'Rate × slope (uphill)', color: '--viz-gradient', shape: 'line' },
  ...(optimizer !== 'SGD' ? [{ label: 'Plain SGD step (dashed)', color: '--viz-gradient', shape: 'line' as const }] : []),
  { label: mode === 'noise' ? 'Batch update' : 'Update applied', color: '--viz-forward', shape: 'line' },
  ...(mode === 'noise' ? [{ label: 'Full dataset (dashed)', color: '--viz-pos', shape: 'line' as const }] : []),
]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={DURATION} {markers} label="Optimizer race timeline" onSeek={(to) => seek(to)} onStep={(to) => seek(to)} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="phase" aria-live="polite">{optimizer === 'Race' ? `Update ${currentFrame.step} of ${STEPS} · ${landscape}` : `${teaching.phase} · update ${teaching.index}`}</p>
    <p class="explain">{optimizer === 'Race' ? 'Each trail records one optimizer’s moves. Gray curves join positions with equal loss. The horizontal and vertical coordinates are two weights.'
      : teaching.time < 3 ? 'The slope arrow points uphill, toward higher loss. It crosses the highlighted curve, which joins every point with the same loss as the start.'
      : mode === 'noise' ? 'Each thin gray arrow is the step one example asks for. The batch update is their average. The dashed arrow is the step the whole dataset asks for.'
      : optimizer === 'Momentum' ? 'The dashed arrow is what plain SGD would do with this slope. The solid arrow adds the direction remembered from earlier updates.'
      : optimizer === 'RMSProp' ? 'The dashed arrow is what plain SGD would do. RMSProp divides each coordinate by that coordinate’s recent slope size, which changes the arrow’s length and direction.'
      : optimizer === 'Adam' ? 'The dashed arrow is what plain SGD would do. Adam uses an average of recent slopes, divided by their recent size. Follow the calculation below.'
      : 'Flip the slope and you get the update: it points downhill, into lower-loss curves. The filled circle moves the weights along it.'}</p>
    {#if optimizer === 'Race'}
      <ul class="metrics" aria-label="Loss at the displayed position for each optimizer">
        {#each OPTIMIZERS as name, i (name)}
          <li><strong class="optimizer-name"><span class="swatch" style:--line="var({legend[i].color})"></span>{name}</strong><span class="num">Loss {raceNumber(frames[name].loss)}</span><span class="num">Weights ({raceNumber(frames[name].point.x)}, {raceNumber(frames[name].point.y)})</span>{#if frames[name].diverged}<span class="notice">non-finite</span>{:else if Math.abs(frames[name].point.x) > DOMAIN || Math.abs(frames[name].point.y) > DOMAIN}<span class="notice">outside plot</span>{/if}</li>
        {/each}
      </ul>
    {:else}
      <p class="result">Loss at the filled circle <strong>{fmt(teaching.loss)}</strong> · before {fmt(teaching.beforeLoss)} → after {fmt(teaching.afterLoss)}</p>
      <div class="calculation" aria-label="Worked parameter update">
        {#each ['x', 'y'] as axis (axis)}
          {@const key = axis as 'x' | 'y'}
          <div><strong>{axis} weight</strong><span>Start {fmt(teaching.before.point[key])}</span><span>− rate {fmt(lr)} × {optimizer === 'Momentum' ? 'velocity' : optimizer === 'Adam' ? 'average slope' : 'slope'} {fmt(teaching.numerator[key])}{optimizer === 'RMSProp' || optimizer === 'Adam' ? ` ÷ recent slope size ${fmt(teaching.denominator[key])}` : ''}</span><strong>= {fmt(teaching.after.point[key])}</strong></div>
        {/each}
      </div>
      {#if optimizer === 'Momentum'}<p class="explain">Velocity, x part: 0.9 × previous velocity {fmt(teaching.before.momentum.x)} + latest slope {fmt(teaching.after.gradient.x)} = {fmt(teaching.after.momentum.x)}.</p>{/if}
      <details class="update-details"><summary>Show the raw numbers for update {teaching.index}</summary>
      {#if optimizer === 'Adam'}<p class="metric">Raw x average {fmt(teaching.after.first.x)} ÷ correction {teaching.correction1.toFixed(4)} = {fmt(teaching.numerator.x)}. Raw squared x average {teaching.after.second.x.toExponential(3)} ÷ {teaching.correction2.toFixed(4)} = {(teaching.after.second.x / teaching.correction2).toFixed(3)}.</p>{/if}
      <p class="metric">Batch slope ({fmt(teaching.after.gradient.x)}, {fmt(teaching.after.gradient.y)}) · examples {teaching.after.samples.map((i) => i + 1).join(', ')}</p>
      </details>
    {/if}
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {mode === 'noise' ? 'Choose batch size 1, then Full batch. With every example included, the batch update lands exactly on the full-dataset arrow.' : optimizer === 'Momentum' ? 'Set Update number to 1, then 8. Update 1 has no memory yet, so both arrows match; by update 8 they differ.' : optimizer === 'Adam' ? 'Keep Update number at 1 and open the raw numbers: the correction turns the shrunken first average back into the first slope.' : optimizer === 'Race' ? 'Raise the learning rate and replay. Watch which paths zig-zag across the narrow valley and which follow it.' : optimizer === 'RMSProp' ? 'Compare the dashed and solid arrows. The coordinate whose recent slopes were larger is shrunk more.' : 'Drag the learning rate from 0.080 to 0.160 and replay. The arrow doubles in length, and the loss after the step falls to 4.064 instead of 8.775.'}</p>
    <div class="settings">
      {#if optimizer !== 'Race'}<label for="{uid}-update">Update number <output>{update}</output><input id="{uid}-update" type="range" min="1" max="20" step="1" bind:value={update} oninput={settingChanged} /></label>{/if}
      <label for="{uid}-lr">Learning rate <output>{lr.toFixed(3)}</output>
        <input id="{uid}-lr" type="range" min="0.005" max={optimizer === 'Race' ? 1.2 : 0.16} step="0.005" bind:value={lr} oninput={settingChanged} />
      </label>
      {#if mode === 'noise' || mode === 'race'}
      <label for="{uid}-batch">Batch size<select id="{uid}-batch" bind:value={batchSize} onchange={settingChanged}><option value={1}>1</option><option value={4}>4</option><option value={8}>8</option><option value={EXAMPLE_COUNT}>Full batch ({EXAMPLE_COUNT})</option></select></label>
      <label for="{uid}-noise">Spread between examples <output>{noise.toFixed(1)}×</output><input id="{uid}-noise" type="range" min="0" max="2" step="0.1" bind:value={noise} oninput={settingChanged} /></label>
      {/if}
    </div>
    <details class="advanced"><summary>Explore other update rules and landscapes</summary><div class="settings">
      <label for="{uid}-optimizer">Optimizer
        <select id="{uid}-optimizer" bind:value={optimizer} onchange={settingChanged}>
          <option value="Race">Race all four</option>
          {#each OPTIMIZERS as name (name)}<option value={name}>{name}</option>{/each}
        </select>
      </label>
      <label for="{uid}-landscape">Landscape
        <select id="{uid}-landscape" bind:value={landscape} onchange={settingChanged}><option value="ravine">Ravine</option><option value="saddle">Saddle</option></select>
      </label>
    </div><p class="note">Each path is 120 updates from the same start with the same batches. Single-update arrows are magnified to fit; the numbers give their true size. The ravine's lowest loss is 1. The saddle has no lowest point, so paths there can leave the plot.</p></details>
  {/snippet}
</FigureCard>

<style>
  .explain, .try { line-height: 1.6; margin: var(--s-2) 0; }
  .try { padding: var(--s-3) var(--s-4) 0; }
  .result { margin-block: var(--s-3); }
  .result strong { color: var(--viz-forward); font-family: var(--font-code); }
  .calculation { display: grid; gap: var(--s-2); }
  .calculation > div { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s-2); padding: var(--s-2); background: var(--bg-sunken); border-left: 2px solid var(--viz-forward); }
  .calculation span { font-family: var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  summary { cursor: pointer; }
  .advanced { padding: 0 var(--s-4) var(--s-3); }
  .phase, .metric { margin: 0 0 var(--s-2); }
  .phase { color: var(--text-muted); }
  .metrics { list-style: none; padding: 0; margin: 0; display: grid; gap: var(--s-1); }
  .metrics li { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s-1) var(--s-3); padding-block: var(--s-2); border-bottom: 1px solid var(--rule); }
  .optimizer-name { display: flex; align-items: center; gap: var(--s-2); min-width: 6rem; }
  .swatch { width: 0.8rem; height: 0.2rem; background: var(--line); align-self: center; }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .notice { color: var(--warn-text); }
  .settings { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr)); gap: var(--s-3) var(--s-4); padding: var(--s-3) var(--s-4); }
  .settings label { display: grid; gap: var(--s-1); font-weight: 600; }
  .settings select { min-height: 2.5rem; padding: 0.35rem 0.5rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); font: inherit; }
  .settings input { width: 100%; accent-color: var(--accent); }
  .settings output { float: right; font-family: var(--font-code); font-weight: 400; }
  .note { margin: 0; padding: 0 var(--s-4) var(--s-3); color: var(--text-muted); font-size: 0.8rem; }
  @media (max-width: 36rem) {
    .metrics li { flex-direction: column; align-items: flex-start; }
  }
</style>
