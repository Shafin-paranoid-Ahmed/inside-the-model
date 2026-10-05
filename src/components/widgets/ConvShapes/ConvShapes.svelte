<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { DEFAULT_SIZE, SIZE_LIMITS, aspectOf, durationOf, frameAt, markersOf, type Mode } from './sim';

  interface Props {
    /** 'channels': 3-channel input, 3x3x3 filters. 'stride' / 'padding': two settings side by side. 'size': the output-size formula. 'pool': max vs average. */
    mode?: Mode;
    title?: string;
  }
  let { mode = 'channels', title = 'Convolution shapes' }: Props = $props();
  const init = untrack(() => mode);
  const uid = $props.id();
  const T = durationOf(init), markers = markersOf(init), aspect = aspectOf(init);
  let boxW = $state(0);
  const narrow = $derived(boxW > 0 && boxW < 540);

  let size = $state({ ...DEFAULT_SIZE });
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let frame = $state.raw(frameAt(0, init, DEFAULT_SIZE));
  let t = 0, wake = $state<() => void>(() => {});
  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline>;
  $effect(() => { playing; speed; narrow; wake(); });

  const paint = () => { frame = frameAt(t, init, size, narrow); tl?.paint(t); uiT = t; };
  const seek = (to: number) => { t = to; wake(); };
  const lineClass = (tone?: string) => (tone ? `ln ${tone}` : 'ln');
  const fontOf = (cell: number, text: string) => Math.min(16, cell * 0.55, (cell * 0.85) / (Math.max(1, text.length) * 0.62));
  const sliders: { key: 'n' | 'k' | 's' | 'p'; label: string }[] = [
    { key: 'n', label: 'Input size n' }, { key: 'k', label: 'Filter size k' },
    { key: 's', label: 'Stride s' }, { key: 'p', label: 'Padding p' },
  ];

  onMount(() => {
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      paint();
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; t = 0; } wake(); };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });
</script>

<div class="conv-shapes">
  <FigureCard {title} label="Grids of numbers for the input and output of a convolution, with the sliding window outlined" aspect={aspect.wide} narrowAspect={aspect.narrow} legend={init === 'pool' ? [
    { label: 'Window', color: '--accent', shape: 'line' }, { label: 'Pooled value', color: '--viz-forward' },
  ] : [
    { label: 'Window', color: '--accent', shape: 'line' }, { label: 'Positive', color: '--viz-pos' }, { label: 'Negative', color: '--viz-neg' },
  ]}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={boxW}>
      <svg viewBox="0 0 {frame.w} {frame.h}" preserveAspectRatio="xMidYMin meet" aria-hidden="true">
        <defs>
          <pattern id="hatch-{uid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" class="hatch" />
          </pattern>
        </defs>
        {#each frame.labels as l, i (i)}
          <text class="lbl {l.tone ?? ''}" x={l.x} y={l.y} text-anchor={l.anchor ?? 'start'}>{l.text}</text>
        {/each}
        {#each frame.grids as g (g.id)}
          <g>
            {#each g.cells as c, i (i)}
              {@const x = g.x + (i % g.cols) * g.cell}
              {@const y = g.y + Math.floor(i / g.cols) * g.cell}
              <rect class="cell {c.kind} {c.sign ?? ''}" {x} {y} width={g.cell} height={g.cell} style:--tone={c.tone} />
              {#if c.kind === 'drop'}<rect {x} {y} width={g.cell} height={g.cell} fill="url(#hatch-{uid})" />{/if}
              {#if c.text}
                <text class="val {c.kind} {c.kind === 'in' && c.tone > 0.55 ? 'hi' : ''}" x={x + g.cell / 2} y={y + g.cell / 2} font-size={fontOf(g.cell, c.text)}>{c.text}</text>
              {/if}
              {#if c.ring}<rect class="ring {c.ring}" x={x + 1} y={y + 1} width={g.cell - 2} height={g.cell - 2} />{/if}
            {/each}
          </g>
        {/each}
        {#each frame.boxes as b, i (i)}
          <rect class="win" x={b.x} y={b.y} width={b.w} height={b.h} />
        {/each}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Window position" onSeek={seek} />
    {/snippet}
    {#snippet readout()}<p class="status" role="status" aria-live="polite">{frame.status}</p>{/snippet}
    {#snippet aside()}
      <h3>{frame.calc.heading}</h3>
      <ul class="calc">{#each frame.calc.lines as l, i (i)}<li class={lineClass(l.tone)}>{l.text}</li>{/each}</ul>
      {#if frame.calc.facts.length}
        <ul class="calc facts">{#each frame.calc.facts as l, i (i)}<li class={lineClass(l.tone)}>{l.text}</li>{/each}</ul>
      {/if}
    {/snippet}
    {#snippet footer()}
      {#if init === 'size'}
        <div class="controls">
          {#each sliders as sl (sl.key)}
            <label class="range">{sl.label} <output class="num">{size[sl.key]}</output>
              <input type="range" min={SIZE_LIMITS[sl.key][0]} max={SIZE_LIMITS[sl.key][1]} step="1" bind:value={size[sl.key]} oninput={wake} />
            </label>
          {/each}
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .conv-shapes :global(.fig) { max-height: 36rem; }
  .conv-shapes :global(.aside) { height: auto; }
  .stagebox { position: absolute; inset: 0; }
  svg { width: 100%; height: 100%; }
  .cell { stroke: var(--rule); stroke-width: 1; fill: var(--bg-raised); }
  .cell.in { fill: color-mix(in srgb, var(--text) calc(var(--tone) * 50%), var(--bg-raised)); }
  .cell.out.pos, .cell.kernel.pos { fill: color-mix(in srgb, var(--viz-pos) calc(var(--tone) * 55%), var(--bg-raised)); }
  .cell.out.neg, .cell.kernel.neg { fill: color-mix(in srgb, var(--viz-neg) calc(var(--tone) * 55%), var(--bg-raised)); }
  .cell.out.fwd { fill: color-mix(in srgb, var(--viz-forward) calc(var(--tone) * 55%), var(--bg-raised)); }
  .cell.pad { fill: var(--bg-sunken); stroke-dasharray: 3 2; }
  .cell.drop { fill: var(--bg-sunken); }
  .cell.empty { fill: none; stroke-dasharray: 2 3; }
  .hatch { stroke: var(--text-muted); stroke-width: 1.5; opacity: 0.55; }
  .val { text-anchor: middle; dominant-baseline: central; fill: var(--text); font-family: var(--font-code); font-variant-numeric: tabular-nums; pointer-events: none; }
  .val.hi { fill: var(--bg-raised); }
  .val.pad { fill: var(--text-muted); }
  .ring { fill: none; pointer-events: none; }
  .ring.cur { stroke: var(--text); stroke-width: 2.5; }
  .ring.winner { stroke: var(--text); stroke-width: 3; stroke-dasharray: 4 3; }
  .win { fill: color-mix(in srgb, var(--accent) 14%, transparent); stroke: var(--accent); stroke-width: 3; pointer-events: none; }
  .lbl { font: 15px var(--font-ui); fill: var(--text-muted); }
  .lbl.strong { fill: var(--text); font-weight: 600; }
  .lbl.pos { fill: var(--viz-pos); font: 600 17px var(--font-code); }
  .lbl.neg { fill: var(--viz-neg); font: 600 17px var(--font-code); }
  .lbl.muted { fill: var(--text-muted); }
  h3 { margin: 0 0 var(--s-2); font: 600 var(--step--1) var(--font-ui); }
  .calc { list-style: none; margin: 0 0 var(--s-3); padding: 0; display: grid; gap: 0.3rem; }
  .ln { font: var(--step--2) / 1.4 var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .ln.pos { color: var(--viz-pos); }
  .ln.neg { color: var(--viz-neg); }
  .ln.strong { font-weight: 700; }
  .ln.muted { color: var(--text-muted); }
  .facts { padding-top: var(--s-2); border-top: 1px solid var(--rule); }
  .facts .ln { font-family: var(--font-ui); }
  .status { margin: 0; font-size: var(--step--1); }
  .controls { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .range { display: flex; flex-direction: column; gap: 0.15rem; font: var(--step--2) var(--font-ui); color: var(--text-muted); }
  .range input { width: 8rem; accent-color: var(--accent); }
  .range output { color: var(--text); }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; }
  @media (max-width: 540px) { .range input { width: min(9rem, 40vw); } }
</style>
