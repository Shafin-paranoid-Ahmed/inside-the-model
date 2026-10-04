<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { APPLY_TEXT, DEFAULT_TEXT, PLAYGROUND_TEXT, defaultModel, durationOf, encode, frameAt, layoutChips, markersOf, tokenLabel, vocabCurve, type Mode } from './sim';

  interface Props { mode?: Mode; text?: string; title?: string }
  let { mode = 'playground', text, title = 'How text becomes token IDs' }: Props = $props();
  const init = untrack(() => mode);
  let input = $state(untrack(() => text ?? (init === 'apply' ? APPLY_TEXT : init === 'granularity' ? DEFAULT_TEXT : PLAYGROUND_TEXT)));
  let width = $state(0), playing = $state(false), speed = $state(1), uiT = $state(0);
  const narrow = $derived(width > 0 && width < 540);
  const svgW = $derived(narrow ? 288 : 720);
  const model = defaultModel();
  const legend = init === 'vocab' ? [{ label: 'Token count', color: '--viz-forward', shape: 'line' as const }, { label: 'Current merge count', color: '--accent' }] : [
    { label: 'Text pieces (· = space)', color: '--viz-seq-2' },
    ...(init === 'granularity' ? [{ label: 'Unknown word', color: '--accent', shape: 'line' as const }] : init === 'merges' || init === 'apply' ? [{ label: 'Pair being joined', color: '--accent', shape: 'line' as const }] : []),
  ];
  const curve = init === 'vocab' ? vocabCurve() : [];
  const duration = $derived(durationOf(init, input));
  const markers = $derived(markersOf(init, input));
  const frame = $derived(frameAt(uiT, init, input));
  const ids = $derived(encode(input, model));
  // Reserve the largest state for this mode, so playback never changes layout height.
  function measureHeight(w: number) {
    let max = init === 'playground' ? 150 : 240;
    const times = init === 'merges' ? markers.flatMap((m) => [m.at, m.at + 2]) : [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => duration * f);
    for (const time of times) {
      let y = init === 'merges' ? 160 : 20;
      const rs = frameAt(time, init, input).rows;
      for (const [i, r] of rs.entries()) {
        y = layoutChips(r, w, y + 44, init === 'merges' ? 30 : 42).bottom + (init === 'merges' ? i === rs.length - 1 ? 4 : 20 : 30);
      }
      max = Math.max(max, y);
    }
    return init === 'vocab' ? (w < 540 ? 300 : 310) : max;
  }
  const wideHeight = $derived(measureHeight(720));
  const narrowHeight = $derived(measureHeight(288));
  const height = $derived(narrow ? narrowHeight : wideHeight);
  const rows = $derived.by(() => {
    let y = init === 'merges' ? 160 : 20;
    return frame.rows.map((r, i) => {
      const top = y;
      const layout = layoutChips(r, svgW, y + 44, init === 'merges' ? 30 : 42, i === 1 ? frame.join : 1);
      y = layout.bottom + (init === 'merges' ? i === frame.rows.length - 1 ? 4 : 20 : 30);
      return { ...r, ...layout, top };
    });
  });
  const curveX = (n: number) => 46 + n / model.merges.length * (svgW - 66);
  const curveY = (value: number) => 35 + (105 - value) / 55 * (height - 90);
  const path = $derived(curve.slice(0, (frame.pointIndex ?? 0) + 1).map((p, i) => `${i ? 'L' : 'M'}${curveX(p.merges)},${curveY(p.per100)}`).join(' '));
  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline>;
  let t = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; input; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  function edit(event: Event) {
    input = Array.from((event.target as HTMLInputElement).value).slice(0, 120).join('');
    t = 0; uiT = 0; playing = !reducedMotion(); wake();
  }
  onMount(() => {
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration }).t;
      // All SVG content and readouts share uiT; updates capped near 30 Hz.
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    let lastPaint = 0;
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });
</script>

<div class="tokenizer" class:merges={init === 'merges'}>
  <FigureCard {title} label={frame.status} aspect="720 / {wideHeight}" narrowAspect="288 / {narrowHeight}" {legend}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {svgW} {height}" aria-hidden="true">
        {#if init === 'vocab' && frame.curve}
          <text x="12" y="18" class="heading">Tokens per 100 characters</text>
          {#each [60, 80, 100] as value (value)}
            <line x1="46" x2={svgW - 20} y1={curveY(value)} y2={curveY(value)} class="gridline" />
            <text x="37" y={curveY(value) + 4} text-anchor="end" class="small">{value}</text>
          {/each}
          <path d={path} class="curve" />
          <circle cx={curveX(frame.curve.merges)} cy={curveY(frame.curve.per100)} r="5" class="cursor" />
          {#each [0, 50, 100, model.merges.length] as value (value)}
            <text x={curveX(value)} y={height - 27} text-anchor={value === model.merges.length ? 'end' : 'middle'} class="small">{value}</text>
          {/each}
          <text x={svgW / 2} y={height - 7} text-anchor="middle" class="small">Merges learned</text>
        {:else}
          {#if init === 'merges' && frame.pair}
            <text x="12" y="19" class="heading">Pair counts across the corpus</text>
            {#each frame.pair.top as pair, i (i)}
              <text x="12" y={43 + i * 28} class="small">{tokenLabel(model.vocab[pair.left])} + {tokenLabel(model.vocab[pair.right])}</text>
              <rect x={svgW * 0.43} y={30 + i * 28} width={pair.count / frame.pair.count * svgW * 0.38} height="18" class:winning={i === 0} class="pairbar" />
              <text x={svgW - 12} y={43 + i * 28} text-anchor="end" class="small">{pair.count}</text>
            {/each}
          {/if}
          {#each rows as r, ri (ri)}
            <text x="12" y={r.top + 14} class="heading">{r.label}: {r.total} {r.total === 1 ? 'piece' : 'pieces'}</text>
            <text x="12" y={r.top + 32} class="small">{narrow && r.detail.length > 35 ? r.detail.slice(0, 33) + '…' : r.detail}</text>
            {#each r.chips as c, i (i)}
              {@const x = c.x}
              {@const y = c.y}
              <rect {x} {y} opacity={c.opacity} width={c.width} height="39" rx="3" class="chip" class:alternate={i % 2 === 1} class:highlight={c.highlight} />
              <text x={x + c.width / 2} y={y + 15} opacity={c.opacity} text-anchor="middle" class="token" data-chip-x={x} data-chip-width={c.width}>{c.label.length > 16 ? c.label.slice(0, 14) + '…' : c.label}{#if c.id !== undefined}<tspan x={x + c.width / 2} y={y + 31} class="id">ID {c.id}</tspan>{/if}</text>
            {/each}
            {#if r.hidden}<text x="12" y={r.bottom + 8} class="small">+ {r.hidden} tokens (all IDs below)</text>{/if}
          {/each}
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} markers={init === 'apply' || init === 'vocab' ? [] : markers} label="Tokenization timeline" onSeek={seek} onReset={() => seek(0)} />
      {#if init === 'apply' || init === 'vocab'}
        <div class="segments" aria-label="Choose a timeline segment">
          {#each markers as marker (marker.at)}
            <button type="button" class:active={marker.label === frame.phase} onclick={() => { playing = false; seek(marker.at); }}>{marker.label}</button>
          {/each}
        </div>
      {/if}
    {/snippet}
    {#snippet readout()}
      <p class="status" role="status" aria-live="polite">{frame.status}</p>
      {#if init === 'merges'}
        <p class="num entries">New vocabulary entries: {model.merges.slice(0, frame.completed).map((m) => `${tokenLabel(model.vocab[m.id])} → ${m.id}`).join(' · ') || 'None yet'}</p>
      {:else if init === 'playground'}
        <p class="num">{Array.from(input).length} characters · {ids.length} token{ids.length === 1 ? '' : 's'} · {ids.length ? (Array.from(input).length / ids.length).toFixed(2) : '0.00'} characters/token</p>
        <p>Try <button class="sample" onclick={() => { input = 'water'; playing = false; seek(duration); }}>water</button> then <button class="sample" onclick={() => { input = 'antidisestablishmentarianism'; playing = false; seek(duration); }}>antidisestablishmentarianism</button>.</p>
        <details><summary>All token IDs</summary><p class="ids num">{ids.join(', ') || 'Empty input: no tokens'}</p></details>
      {:else if init === 'vocab'}
        <p>The curve uses new sentences: “a fox drinks water beside a quiet stream…”</p>
      {/if}
    {/snippet}
    {#snippet footer()}
      {#if init === 'playground'}
        <label class="textinput">Your text (up to 120 Unicode code points)
          <input type="text" value={input} oninput={edit} aria-label="Text to tokenize" />
        </label>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .heading { fill: var(--text); font: 600 14px var(--font-ui); }
  .small { fill: var(--text-muted); font: 12px var(--font-code); }
  .chip { fill: var(--viz-seq-1); stroke: var(--rule); }
  .chip.alternate { fill: var(--viz-seq-2); }
  .chip.highlight { stroke: var(--accent); stroke-width: 2; }
  .token { fill: var(--text); font: 12px var(--font-code); }
  .id { fill: var(--text-muted); font-size: 10px; }
  .gridline { stroke: var(--rule); stroke-dasharray: 3 3; }
  .curve { fill: none; stroke: var(--viz-forward); stroke-width: 2.5; }
  .cursor { fill: var(--accent); }
  .pairbar { fill: var(--viz-seq-2); }
  .pairbar.winning { fill: var(--accent); }
  .status { margin: 0; min-height: 3.6em; }
  /* Keep all 12 named segments attached to the scrubber, with four staggered label lanes. */
  .merges :global(.track) { flex-basis: 100%; padding-bottom: 5rem; }
  .merges :global(.ticks) { height: 4.8rem; }
  .merges :global(.tick) { font-size: 0.68rem; }
  .merges :global(.tick:nth-child(4n + 2)) { top: 1.2rem; }
  .merges :global(.tick:nth-child(4n + 3)) { top: 2.4rem; }
  .merges :global(.tick:nth-child(4n + 4)) { top: 3.6rem; }
  .merges :global(.tick.edge) { transform: translateX(-50%); }
  .merges :global(.tick.edge::before) { left: 50%; right: auto; }
  .segments { display: flex; flex-wrap: wrap; gap: 0.35rem; padding: var(--s-2) var(--s-4); border-top: 1px solid var(--rule); }
  .segments button { font: 0.8rem var(--font-ui); padding: 0.35em 0.55em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .segments button.active { border-color: var(--accent); font-weight: 650; }
  .sample { color: var(--accent); text-decoration: underline; background: none; border: none; padding: 0; font: inherit; cursor: pointer; overflow-wrap: anywhere; }
  .textinput { display: grid; gap: var(--s-2); padding: var(--s-3) var(--s-4); }
  .textinput input { min-width: 0; width: 100%; box-sizing: border-box; padding: 0.6rem; font: 0.9rem var(--font-code); color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .ids, .entries { overflow-wrap: anywhere; }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
