<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { defaultConfig, frameAt, workedExample, PHASES, T, type Config, type Mode, type Norm } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'standardize', title = 'Normalization' }: Props = $props();
  const uid = $props.id();
  let config = $state(defaultConfig(untrack(() => mode)));
  let selectedNorm = $state<Norm>('layer'), methodChanged = $state(false);
  let uiT = $state(0), playing = $state(false), speed = $state(1);
  let t = 0;
  let stage: HTMLDivElement, tl: Timeline, wake = $state<() => void>(() => {});
  const normForMode = (value: Mode): Norm => value === 'batch' || value === 'inference' ? 'batch' : value === 'rms' ? 'rms' : 'layer';
  let activeNorm = $derived(methodChanged ? selectedNorm : normForMode(mode));
  let frame = $derived(frameAt(uiT, { ...config, mode, norm: activeNorm }));
  let groupIndex = $state(0);
  let advancedOpen = $state(untrack(() => mode === 'affine'));
  const markers = PHASES;
  const methods: { id: Norm; label: string }[] = [
    { id: 'batch', label: 'BatchNorm' }, { id: 'layer', label: 'LayerNorm' }, { id: 'rms', label: 'RMSNorm' },
  ];
  const fmt = (x: number) => (Math.abs(x) < 0.0005 ? '0.000' : x.toFixed(3));
  const group = $derived(frame.reductionGroups[Math.min(groupIndex, frame.reductionGroups.length - 1)] ?? []);
  const selectedGroup = $derived(group);
  const example = $derived(workedExample(frame, config, groupIndex));
  const member = (i: number, j: number) => selectedGroup.some(([r, c]) => r === i && c === j);
  const begin = (to: number) => { t = to; uiT = to; wake(); };
  const onSeek = (to: number) => { playing = false; begin(to); };
  const onReset = () => { begin(0); playing = false; };

  function chooseNorm(norm: Norm) {
    selectedNorm = norm;
    methodChanged = true;
    groupIndex = Math.min(groupIndex, 2);
    wake();
  }

  function setNumber(event: Event, key: 'epsilon' | 'gamma' | 'beta', min: number, max: number) {
    const raw = (event.currentTarget as HTMLInputElement).valueAsNumber;
    if (Number.isFinite(raw) && raw >= min && raw <= max) config[key] = raw;
    wake();
  }

  function normalizeNumber(event: Event, key: 'epsilon' | 'gamma' | 'beta', min: number, max: number) {
    const input = event.currentTarget as HTMLInputElement;
    const raw = input.valueAsNumber;
    const value = Number.isFinite(raw) ? Math.min(max, Math.max(min, raw)) : config[key];
    config[key] = value;
    input.value = String(value);
    wake();
  }

  onMount(() => {
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      tl?.paint(t);
      uiT = t;
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => {
      if (motion.matches) { playing = false; t = 0; uiT = 0; }
      wake();
    };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });
  $effect(() => { playing; speed; wake(); });
</script>

<div class="normalization-widget">
<FigureCard {title} label="A matrix of examples and features with normalization groups and computed outputs" aspect="4 / 1" narrowAspect="3 / 4" role="group" legend={[
  { label: 'Values that share statistics', color: '--accent', shape: 'dot' },
  { label: 'Output values', color: '--viz-forward', shape: 'dot' },
]}>
  <div class="lab" bind:this={stage}>
    <div class="matrices">
      <section class="matrix-panel" aria-label="Input activations">
        <h3>Input <span class="caption">rows: examples · columns: features</span></h3>
        <table>
          <caption class="sr-only">Input activations. The selected outline marks one reduction group.</caption>
          <thead><tr><th scope="col">Example</th>{#each [0, 1, 2] as j (j)}<th scope="col">f{j + 1}</th>{/each}</tr></thead>
          <tbody>
            {#each frame.input as row, i (i)}
              <tr><th scope="row">x{i + 1}</th>{#each row as value, j (j)}<td class:member={member(i, j)} class="num">{fmt(value)}</td>{/each}</tr>
            {/each}
          </tbody>
        </table>
      </section>

      <section class="matrix-panel result-panel" aria-label="Current normalization output">
        <h3>{frame.stageLabel}</h3>
        <table>
          <caption class="sr-only">Exact cell computations appear one at a time. An ellipsis marks a cell whose result has not been revealed yet.</caption>
          <thead><tr><th scope="col">Example</th>{#each [0, 1, 2] as j (j)}<th scope="col">f{j + 1}</th>{/each}</tr></thead>
          <tbody>
            {#each frame.visible as row, i (i)}
              <tr><th scope="row">x{i + 1}</th>{#each row as value, j (j)}<td class:member={i === example.row && j === example.column} class="num output">{frame.phase !== 'group' && i * row.length + j < frame.revealCount ? fmt(value) : '…'}</td>{/each}</tr>
            {/each}
          </tbody>
        </table>
      </section>
    </div>

  </div>

  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Normalization stages" onSeek={onSeek} onStep={(to) => begin(to)} onReset={onReset} />
  {/snippet}
  {#snippet readout()}
    <p class="narration" role="status" aria-live="polite">{activeNorm === 'batch' ? 'BatchNorm normalizes each column (one feature) across the examples.' : activeNorm === 'rms' ? 'RMSNorm rescales each row (one example) across its features, without subtracting the mean.' : 'LayerNorm normalizes each row (one example) across its features.'}</p>
    {#if mode === 'inference'}
      <div class="compare" aria-label="Completed BatchNorm training and evaluation outputs">
        <p><strong>Compare the completed outputs for x1</strong></p>
        <p>Current-batch statistics → <span class="num">{frame.training.output[0].map(fmt).join(' · ')}</span></p>
        <p>Fixed evaluation statistics → <span class="eval-out num">{frame.inference.output[0].map(fmt).join(' · ')}</span></p>
        <details><summary>Show both sets of statistics</summary>
          <p class="small">Current column means: {frame.training.statistics.map((s) => fmt(s.mean[0])).join(' · ')}.</p>
          <p class="small">Running means saved during training: {frame.inference.statistics.map((s) => fmt(s.mean[0])).join(' · ')}. Population variances: {frame.inference.statistics.map((s) => fmt(s.variance[0])).join(' · ')}.</p>
        </details>
      </div>
    {/if}
    <div class="worked" aria-label="Worked normalization of one highlighted value">
      <p><strong>Follow x{example.row + 1}, f{example.column + 1}: {fmt(example.input)}</strong>. It shares statistics with the outlined values: {example.values.map(fmt).join(', ')}.</p>
      <ol class="chain">
        <li class:active={frame.phase === 'group'}><strong>1. Group</strong><span>{activeNorm === 'rms' ? 'Root mean square' : 'Mean'} {fmt(activeNorm === 'rms' ? example.divisor : example.mean)}{activeNorm !== 'rms' ? ` · variance ${fmt(example.variance)}` : ''}</span></li>
        <li class:active={frame.phase === 'center'}><strong>2. {activeNorm === 'rms' ? 'Keep the value' : 'Subtract mean'}</strong><span>{activeNorm === 'rms' ? `${fmt(example.input)} (no centering)` : `${fmt(example.input)} − ${fmt(example.mean)} = ${fmt(example.centered)}`}</span></li>
        <li class:active={frame.phase === 'scale'}><strong>3. {activeNorm === 'rms' ? 'Divide by RMS' : 'Divide by standard deviation'}</strong><span>{fmt(example.centered)} ÷ {fmt(example.divisor)} = {fmt(example.scaled)}</span></li>
        <li class:active={frame.phase === 'affine'}><strong>4. Gain and shift</strong><span>{fmt(example.gamma)} × {fmt(example.scaled)} + {fmt(example.beta)} = {fmt(example.output)}</span></li>
      </ol>
      {#if activeNorm !== 'rms'}<p class="small">Standard deviation = √(variance + ε) = √({fmt(example.variance)} + {config.epsilon}) = {fmt(example.divisor)}. The small ε keeps the division safe when a group's values are all equal.</p>{/if}
      <details><summary>Output means and variances</summary><p class="small">After gain and shift, {activeNorm === 'batch' ? 'column' : 'row'} means: {frame.normalizedMean.map(fmt).join(' · ')}. Variances: {frame.normalizedVariance.map(fmt).join(' · ')}.</p></details>
    </div>
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {mode === 'affine' ? 'Open gain and shift, then change gamma or beta. Follow how only the last calculation changes.' : mode === 'inference' ? 'Change example x2’s offset. The current-batch result for x1 changes; its evaluation result uses fixed statistics and stays the same.' : mode === 'batch' ? 'Drag Example x2 offset and watch the worked value for x1 change, even though x1 itself never changed.' : mode === 'layer' ? 'Drag Example x2 offset: x1’s result stays put. Switch to BatchNorm and drag again.' : mode === 'rms' ? 'Switch between RMSNorm and LayerNorm and compare step 2: only LayerNorm subtracts the mean.' : 'Choose a different highlighted example and follow its four steps.'}</p>
    <div class="controls">
      {#if mode !== 'inference'}
        <fieldset class="segmented">
          <legend>Normalization</legend>
        {#each methods as method (method.id)}
            <label><input type="radio" name="normalization-{uid}" value={method.id} checked={activeNorm === method.id} onchange={() => chooseNorm(method.id)} /><span>{method.label}</span></label>
          {/each}
        </fieldset>
      {/if}
      <label>Highlighted {activeNorm === 'batch' ? 'feature' : 'example'}
        <select bind:value={groupIndex} aria-label="Highlighted reduction group" onchange={wake}>
          {#each [0, 1, 2] as index (index)}<option value={index}>{activeNorm === 'batch' ? `Feature f${index + 1}` : `Example x${index + 1}`}</option>{/each}
        </select>
      </label>
      <label class="range">Example x2 scale <output class="num">{config.companionScale.toFixed(1)}</output>
        <input type="range" min="0.5" max="2" step="0.1" bind:value={config.companionScale} aria-label="Companion example scale" oninput={wake} />
      </label>
      <label class="range">Example x2 offset <output class="num">{config.companionOffset.toFixed(1)}</output>
        <input type="range" min="-3" max="3" step="0.5" bind:value={config.companionOffset} aria-label="Companion example offset" oninput={wake} />
      </label>
    </div>
    <details class="advanced" bind:open={advancedOpen}><summary>Gain, shift and numerical stability</summary><div class="controls">
      <label>epsilon (stability) <input class="number num" type="number" min="0.0001" max="1" step="0.01" value={config.epsilon} oninput={(event) => setNumber(event, 'epsilon', 0.0001, 1)} onchange={(event) => normalizeNumber(event, 'epsilon', 0.0001, 1)} /></label>
      <label>gamma (gain) <input class="number num" type="number" min="-3" max="3" step="0.1" value={config.gamma} oninput={(event) => setNumber(event, 'gamma', -3, 3)} onchange={(event) => normalizeNumber(event, 'gamma', -3, 3)} /></label>
      <label>beta (shift) <input class="number num" type="number" min="-3" max="3" step="0.1" value={config.beta} oninput={(event) => setNumber(event, 'beta', -3, 3)} onchange={(event) => normalizeNumber(event, 'beta', -3, 3)} /></label>
      <p class="validation">Here one gain and one shift are shared by every feature; real layers learn one of each per feature.</p>
    </div></details>
  {/snippet}
</FigureCard>
</div>

<style>
  .normalization-widget :global(.fig) { width: 100%; min-height: 12rem; }
  .worked { margin-top: var(--s-3); }
  .worked p { margin: var(--s-2) 0; line-height: 1.6; }
  .chain { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s-2); padding: 0; list-style: none; }
  .chain li { display: grid; gap: var(--s-1); padding: var(--s-2); border: 1px solid var(--rule); }
  .chain li.active { border-color: var(--accent); background: var(--accent-soft); }
  .chain span { font-family: var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .try { padding: var(--s-3) var(--s-4) 0; margin: 0; line-height: 1.6; }
  .advanced { padding: 0 var(--s-4) var(--s-3); }
  summary { cursor: pointer; }
  @media (max-width: 540px) { .chain { grid-template-columns: 1fr; } }
  .lab { width: 100%; padding: var(--s-3); }
  .matrices { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s-4); align-items: start; }
  .matrix-panel { min-width: 0; }
  h3 { display: flex; justify-content: space-between; align-items: baseline; gap: var(--s-2); margin: 0 0 var(--s-2); font: 600 var(--step--1) var(--font-ui); }
  .caption, .small { color: var(--text-muted); font: var(--step--2) var(--font-ui); }
  .lab table { display: table; }
  table { width: 100%; border-collapse: separate; border-spacing: 0.18rem; table-layout: fixed; }
  th, td { text-align: center; padding: 0.35em 0.15em; border: 1px solid var(--rule); border-radius: var(--r-1); font: 0.76rem var(--font-code); font-variant-numeric: tabular-nums; }
  thead th, tbody th { color: var(--text-muted); font-weight: 500; }
  td.member { outline: 2px solid var(--accent); outline-offset: -2px; background: var(--accent-soft); }
  td.output { color: var(--viz-forward); }
  .compare { border-top: 1px solid var(--rule); margin-top: var(--s-3); padding-top: var(--s-2); font: var(--step--2) var(--font-ui); }
  .compare p { margin: 0.25em 0; }
  .eval-out { color: var(--viz-forward); }
  .narration { margin: 0; font-size: var(--step--1); }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .controls label { display: inline-flex; align-items: center; gap: var(--s-2); font: var(--step--2) var(--font-ui); color: var(--text-muted); }
  .range { flex-direction: column; align-items: flex-start !important; gap: 0.15rem !important; }
  .range input { width: 8rem; accent-color: var(--accent); }
  .range output { color: var(--text); }
  .number { width: 4.2rem; padding: 0.25em 0.35em; border: 1px solid var(--rule); border-radius: var(--r-1); background: var(--bg-raised); color: var(--text); }
  .segmented { display: flex; flex-wrap: wrap; gap: 0; padding: 0; border: 1px solid var(--rule); border-radius: var(--r-2); }
  .segmented legend { padding: 0 var(--s-1); color: var(--text-muted); font: var(--step--2) var(--font-ui); }
  .segmented label { position: relative; margin: 0; }
  .segmented input { position: absolute; opacity: 0; inset: 0; }
  .segmented span { display: block; padding: 0.35em 0.6em; cursor: pointer; }
  .segmented input:checked + span { background: var(--accent-soft); color: var(--text); }
  .segmented input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .controls select { max-width: 9rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.25em; }
  .validation { flex-basis: 100%; margin: 0; color: var(--text-muted); font: var(--step--2) var(--font-ui); }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; }
  @media (max-width: 540px) {
    .lab { padding: var(--s-2); }
    .matrices { grid-template-columns: 1fr; gap: var(--s-2); }
    h3 { align-items: flex-start; flex-direction: column; gap: 0.1rem; }
    th, td { padding: 0.28em 0.1em; font-size: 0.72rem; }
    .controls { padding: var(--s-2); }
    .range input { width: min(12rem, 65vw); }
  }
  @media (prefers-reduced-motion: reduce) {
    * { scroll-behavior: auto !important; }
  }
</style>
