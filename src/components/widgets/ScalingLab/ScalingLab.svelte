<script module lang="ts">
  import type { Sweep } from './sim';
  // One fetch per page, shared by every ScalingLab island.
  let sweepPromise: Promise<Sweep> | null = null;
  function loadSweep(): Promise<Sweep> {
    if (!sweepPromise) {
      sweepPromise = fetch('/models/scaling-sweep.json').then((r) => { if (!r.ok) throw new Error('sweep'); return r.json() as Promise<Sweep>; });
      sweepPromise.catch(() => { sweepPromise = null; });
    }
    return sweepPromise;
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { DEFAULTS, PUBLISHED, TITLES, frameAt, makeScene, sci, fmtInt, type Mode, type PubFit, type Scene, type Settings } from './sim';

  interface Props { mode?: Mode; title?: string; fitPoints?: number; floor?: boolean; swap?: boolean; budget?: number; fit?: PubFit }
  let { mode = 'sweep', title, fitPoints = DEFAULTS.fitPoints, floor = DEFAULTS.floor, swap = DEFAULTS.swap, budget = DEFAULTS.budget, fit = DEFAULTS.pubFit }: Props = $props();
  const init = untrack(() => mode);
  const needsData = init !== 'published' && init !== 'overtrain';

  let data = $state.raw<Sweep | null>(null);
  let failed = $state(false);
  let sFloor = $state(untrack(() => floor)), sSwap = $state(untrack(() => swap)), sPoints = $state(untrack(() => fitPoints));
  let sBudget = $state(untrack(() => budget)), sFit = $state<PubFit>(untrack(() => fit));
  let hover = $state(-1);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0), width = $state(0);
  const narrow = $derived(width > 0 && width < 516);

  const settings: Settings = $derived({ floor: sFloor, swap: sSwap, fitPoints: sPoints, budget: sBudget, pubFit: sFit });
  const scene: Scene | null = $derived(makeScene(init, data, settings));
  const duration = $derived(scene?.duration ?? 12);
  const frame = $derived(scene ? frameAt(uiT, scene, hover) : null);

  // Layout in SVG units; the viewBox changes with the container so text stays readable on phones.
  // viewBox = the stage's real pixel size, so SVG text is the same size in CSS px at every width.
  const W = $derived(width > 0 ? width : 720), H = $derived(width > 0 ? Math.round(width * (narrow ? 440 / 360 : 400 / 720)) : 400);
  const M = $derived(narrow ? { l: 54, r: 44, t: 10, b: 50 } : { l: 64, r: 62, t: 12, b: 48 });
  const PW = $derived(W - M.l - M.r), PH = $derived(H - M.t - M.b);
  const uid = $props.id();

  const pos = (v: number, min: number, max: number, log: boolean) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
  const px = (v: number) => (scene ? M.l + pos(v, scene.x.min, scene.x.max, scene.x.log) * PW : 0);
  const py = (v: number) => (scene ? M.t + (1 - pos(v, scene.y.min, scene.y.max, scene.y.log)) * PH : 0);
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`).join('');

  const legend = {
    sweep: [{ label: 'Smaller model', color: '--viz-seq-3' }, { label: 'Larger model', color: '--viz-seq-5' }, { label: 'Lowest loss so far', color: '--accent', shape: 'line' as const }],
    compute: [{ label: 'Best run at each compute', color: '--viz-cat-1' }, { label: 'Fitted power law', color: '--accent', shape: 'line' as const }],
    'params-data': [{ label: untrack(() => swap) ? 'Smaller model' : 'Less data', color: '--viz-seq-3' }, { label: untrack(() => swap) ? 'Larger model' : 'More data', color: '--viz-seq-5' }],
    isoflop: [{ label: '10⁹ FLOP', color: '--viz-cat-1' }, { label: '3×10⁹', color: '--viz-cat-2' }, { label: '10¹⁰', color: '--viz-cat-3' }, { label: '3×10¹⁰', color: '--viz-cat-4' }, { label: 'Your budget', color: '--accent' }],
    published: [{ label: 'Predicted loss level', color: '--viz-seq-3', shape: 'line' as const }, { label: 'Budget', color: '--text', shape: 'line' as const }, { label: 'Chinchilla-style size', color: '--accent', shape: 'line' as const }, { label: 'Kaplan-style size', color: '--viz-neg', shape: 'line' as const }],
    extrapolate: [{ label: 'Used for the fit', color: '--viz-cat-1' }, { label: 'Held out', color: '--viz-neg' }, { label: 'Fitted law', color: '--accent', shape: 'line' as const }],
    overtrain: [{ label: 'Training only', color: '--text-muted', shape: 'line' as const }, { label: 'Training plus serving', color: '--accent', shape: 'line' as const }],
  }[init];

  let tl: ReturnType<typeof Timeline> | undefined, stage: HTMLDivElement | undefined, svg: SVGSVGElement | undefined;
  let t = 0, lastPaint = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; scene; wake(); });
  const seek = (to: number) => { t = to; uiT = to; tl?.paint(to); wake(); };
  /** Restart the animation after a control change. */
  const restart = () => { seek(0); playing = !reducedMotion(); };
  const jumpEnd = () => { playing = false; seek(duration); };

  function nearest(e: PointerEvent) {
    if (!scene?.hoverables || !svg) return;
    const r = svg.getBoundingClientRect(), vx = ((e.clientX - r.left) / r.width) * W, vy = ((e.clientY - r.top) / r.height) * H;
    let best = -1, d2 = 18 * 18;
    for (const h of scene.hoverables) { const dx = px(h.x) - vx, dy = py(h.y) - vy, d = dx * dx + dy * dy; if (d < d2) { d2 = d; best = h.id; } }
    hover = best;
  }

  onMount(() => {
    if (needsData) loadSweep().then((d) => { data = d; }, () => { failed = true; });
    const loop = createLoop(stage?.closest('[data-figure]') ?? stage!, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration }).t;
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    if (reducedMotion()) seek(duration);
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) playing = false; };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  const label = $derived(scene?.label ?? 'Loading the training runs');
  const runOptions = $derived(data ? data.runs.map((r, i) => ({ i, text: `${fmtInt(r.N)} parameters, ${fmtInt(r.D)} characters` })) : []);
</script>

<FigureCard title={title ?? TITLES[init]} {label} aspect="720 / 400" narrowAspect="360 / 440" {legend} role="group">
  <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
    <svg class:wide={!narrow} bind:this={svg} viewBox="0 0 {W} {H}" aria-hidden="true" onpointermove={nearest} onpointerdown={nearest} onpointerleave={() => (hover = -1)}>
      <defs><clipPath id="{uid}-clip"><rect x={M.l} y={M.t} width={PW} height={PH} /></clipPath></defs>
      {#if scene && frame}
        {#each scene.y.ticks as k (k.v)}
          <line x1={M.l} x2={W - M.r} y1={py(k.v)} y2={py(k.v)} class="grid" />
          <text x={M.l - 6} y={py(k.v) + 4} text-anchor="end" class="tick">{k.label}</text>
        {/each}
        {#each scene.x.ticks as k (k.v)}
          <line x1={px(k.v)} x2={px(k.v)} y1={M.t} y2={M.t + PH} class="grid" />
          <text x={px(k.v)} y={M.t + PH + 16} text-anchor="middle" class="tick">{k.label}</text>
        {/each}
        <rect x={M.l} y={M.t} width={PW} height={PH} class="frame" />
        <text x={M.l + PW / 2} y={H - 8} text-anchor="middle" class="axis">{scene.x.label}</text>
        <text transform="translate(12 {M.t + PH / 2}) rotate(-90)" text-anchor="middle" class="axis">{scene.y.label}</text>
        <g clip-path="url(#{uid}-clip)">
          {#each frame.lines as l, i (i)}
            <path d={path(l.pts)} fill="none" style:stroke={l.color} stroke-width={l.w ?? 1.5} stroke-dasharray={l.dash ? '6 4' : undefined} opacity={l.op ?? 1} stroke-linejoin="round" />
          {/each}
        </g>
        {#each frame.dots as d, i (i)}
          {#if d.ring}
            <circle cx={px(d.x)} cy={py(d.y)} r={d.r ?? 5} fill="none" style:stroke={d.color} stroke-width="2" />
          {:else}
            <circle cx={px(d.x)} cy={py(d.y)} r={d.r ?? 4} style:fill={d.hollow ? 'var(--bg-raised)' : d.color} style:stroke={d.color} stroke-width="1.8" />
          {/if}
        {/each}
        {#each frame.labels as l, i (i)}
          <text x={px(l.x) + (l.dx ?? 0)} y={py(l.y) + (l.dy ?? 0)} text-anchor={l.anchor ?? 'start'} class="lab" style:fill={l.color ?? 'var(--text)'}>{l.text}</text>
        {/each}
      {:else}
        <text x={W / 2} y={H / 2} text-anchor="middle" class="axis">{failed ? 'The training runs could not be loaded.' : 'Loading the training runs…'}</text>
      {/if}
    </svg>
  </div>
  {#snippet controls()}
    <div class:stagger={init === 'params-data'}>
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} markers={scene?.markers ?? []} label="{TITLES[init]} timeline" onSeek={seek} onReset={() => seek(0)} />
    </div>
  {/snippet}
  {#snippet readout()}
    <div class="note">
      <p class="status" role="status" aria-live="polite">{frame?.readout ?? (failed ? 'The training runs could not be loaded.' : 'Loading the training runs…')}</p>
      <p class="detail">{frame?.detail ?? ''}</p>
    </div>
  {/snippet}
  {#snippet footer()}
    {#if init === 'sweep' && data}
      <label class="ctl">Highlight a run
        <select value={hover} onchange={(e) => (hover = +(e.currentTarget as HTMLSelectElement).value)}>
          <option value={-1}>None (or hover a curve)</option>
          {#each runOptions as o (o.i)}<option value={o.i}>{o.text}</option>{/each}
        </select>
      </label>
    {:else if init === 'compute'}
      <label class="ctl"><input type="checkbox" bind:checked={sFloor} onchange={restart} /> Allow an irreducible floor E</label>
    {:else if init === 'params-data'}
      <label class="ctl">Plot loss against
        <select bind:value={sSwap} onchange={restart}>
          <option value={false}>parameters, one curve per data amount</option>
          <option value={true}>training characters, one curve per model size</option>
        </select>
      </label>
    {:else if init === 'isoflop'}
      <label class="ctl">Your compute budget: <span class="num">{sci(10 ** sBudget, 2)} FLOP</span>
        <input type="range" min="8.6" max="11" step="0.05" bind:value={sBudget} oninput={jumpEnd} aria-label="Compute budget, log scale" />
      </label>
    {:else if init === 'extrapolate'}
      <label class="ctl">Runs used for the fit: <span class="num">{sPoints} smallest of 15</span>
        <input type="range" min="3" max="12" step="1" bind:value={sPoints} oninput={restart} aria-label="Number of runs used for the fit" />
      </label>
    {:else if init === 'published' || init === 'overtrain'}
      <label class="ctl">Published fit
        <select bind:value={sFit} onchange={restart}>
          <option value="besiroglu">{PUBLISHED.besiroglu.label}</option>
          <option value="hoffmann">{PUBLISHED.hoffmann.label}</option>
        </select>
      </label>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .stagebox { position: absolute; inset: 0; }
  .grid { stroke: var(--rule); stroke-dasharray: 2 4; }
  .frame { fill: none; stroke: var(--rule); }
  .tick { fill: var(--text-muted); font: 12px var(--font-ui); font-variant-numeric: tabular-nums; }
  .wide .tick, .wide .axis, .wide .lab { font-size: 13px; }
  /* Seven or six close markers: alternate them on two rows so labels never collide. */
  .stagger :global(.track) { padding-bottom: 2.5rem; }
  .stagger :global(.ticks) { height: 2.5rem; }
  .stagger :global(.tick:nth-child(even)) { top: 1.2rem; }
  .axis { fill: var(--text-muted); font: 12px var(--font-ui); }
  .lab { font: 600 12px var(--font-ui); font-variant-numeric: tabular-nums; paint-order: stroke; stroke: var(--bg-raised); stroke-width: 3px; }
  /* One block with a reserved height, so the text under it never moves; the spare room sits below the note. */
  .note { min-height: 4.8em; }
  .status { margin: 0; font-family: var(--font-ui); font-variant-numeric: tabular-nums; }
  .detail { margin: var(--s-1) 0 0; color: var(--text-muted); font-size: 0.85rem; }
  @media (max-width: 40em) { .note { min-height: 13em; } }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-3); padding: var(--s-3) var(--s-4); font: 0.9rem var(--font-ui); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; max-width: 100%; }
  .ctl input[type='range'] { accent-color: var(--accent); flex: 1 1 12rem; min-width: 10rem; }
  .ctl input[type='checkbox'] { accent-color: var(--accent); }
  .ctl :global(.num) { font-family: var(--font-code); font-size: 0.85rem; }
</style>
