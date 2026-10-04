<script module lang="ts">
  import type { Ablations } from './sim';

  // One worker per page: it loads the model once and answers each figure's request separately.
  type Kind = 'rule' | 'random' | 'formats' | 'ablations';
  let worker: Worker | undefined, nextId = 1;
  const pending = new Map<number, [(v: unknown) => void, (e: Error) => void]>();
  const memo = new Map<Kind, Promise<unknown>>();
  function ask<T>(kind: Kind): Promise<T> {
    const known = memo.get(kind);
    if (known) return known as Promise<T>;
    if (!worker) {
      const w = (worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }));
      w.onmessage = ({ data }) => {
        const p = pending.get(data.id);
        pending.delete(data.id);
        if (p) data.error ? p[1](new Error(data.error)) : p[0](data.data);
      };
      w.onerror = () => {
        for (const p of pending.values()) p[1](new Error('Could not run the model'));
        pending.clear(); w.terminate(); worker = undefined; memo.clear();
      };
    }
    const result = new Promise<T>((resolve, reject) => {
      const id = nextId++;
      pending.set(id, [resolve as (v: unknown) => void, reject]);
      worker!.postMessage({ id, kind });
    });
    memo.set(kind, result);
    return result;
  }
  let measured: Promise<[unknown, unknown]> | undefined;
  const measurements = () => (measured ??= Promise.all(['icl-log', 'icl-sweep'].map(async (name) => {
    const r = await fetch(`/models/${name}.json`);
    if (!r.ok) throw new Error('Could not load the training measurements');
    return r.json();
  })) as Promise<[unknown, unknown]>);
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { loadModel } from '../../../lib/transformer/load';
  import type { Weights } from '../../../lib/transformer/model';
  import { createRng } from '../../../lib/math/rng';
  import {
    CHANCE, COUNTS, DEFAULT_PROMPT, DEMO, DURATION, FORMATS, HEADS, LETTERS, METRICS, PHASE_STEPS,
    attentionFrame, beatMarkers, frameAt, headName, makeTask, norm, parsePairs, phaseFrame, predict, shotRows, shift,
    timeOfStep, topPositions, triangle, type Metric, type Mode, type SweepRow, type TrainingLog,
  } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'shots', title = 'Examples in the prompt' }: Props = $props();
  const init = untrack(() => mode);

  // Fetched or computed data: raw state, never deep proxies.
  let weights = $state.raw<Weights | null>(null);
  let log = $state.raw<TrainingLog | null>(null), sizes = $state.raw<SweepRow[]>([]);
  let rule = $state.raw<number[] | null>(null), random = $state.raw<number[] | null>(null), formats = $state.raw<number[] | null>(null);
  let ablation = $state.raw<Ablations | null>(null);
  let failed = $state('');

  // Controls.
  let typedText = $state(''), typedPairs = $state.raw<string[] | null>(null), typedError = $state('');
  let useRandom = $state(false), head = $state(3), headOff = $state(false), metric = $state<Metric>('exact');
  let playing = $state(false), speed = $state(1), uiT = $state(0), width = $state(0);
  const narrow = $derived(width > 0 && width < 540);
  const W = $derived(narrow ? 288 : 720);

  // ---- the examples shown in the first figure ----
  const pairs = $derived(typedPairs ?? DEMO.pairs);
  const rows = $derived(weights && init === 'shots' ? shotRows(weights, pairs) : []);
  const prompts = FORMATS.map((f) => formatPrompt(f.id));
  function formatPrompt(id: string) {
    const ex = DEMO.pairs.slice(0, 4).map((p) => p.split('>')), sep = id === 'equals' ? '=' : '>', gap = id === 'newline' ? '\n' : id === 'packed' ? '' : ' ';
    return ex.map(([a, b], i) => `${a}${sep}${id === 'wrong' && i === 3 ? shift(b, 1) : b}${gap}`).join('') + `${DEMO.query}${sep}`;
  }
  const att = $derived(weights && init === 'attention' ? predict(weights, DEFAULT_PROMPT, headOff ? [HEADS[head]] : []) : null);
  const samples = $derived(COUNTS.map((k) => makeTask(createRng(31 + k), k, { random: useRandom }).text));

  const beats = $derived(init === 'shots' ? Math.max(1, rows.length) : init === 'curve' ? COUNTS.length : init === 'emergence' ? Math.max(1, sizes.length) : init === 'format' ? FORMATS.length : 1);
  const markers = $derived(
    init === 'shots' ? beatMarkers(rows.map((r) => String(r.count)))
    : init === 'curve' ? beatMarkers(COUNTS.map(String))
    : init === 'attention' ? beatMarkers(['Layer 1', 'Layer 2'])
    : init === 'phase' ? [{ at: 0, label: 'Plateau' }, { at: timeOfStep(450), label: 'Sudden drop' }, { at: timeOfStep(1000), label: 'Settling' }]
    : init === 'emergence' ? beatMarkers(sizes.map((r) => `${Math.round(r.N / 1000)}k`))
    : beatMarkers(['Trained', 'Equals', 'Lines', 'No gaps', 'Wrong']));
  const f = $derived(frameAt(uiT, beats));
  const pf = $derived(init === 'phase' && log ? phaseFrame(uiT, log) : null);
  const af = $derived(attentionFrame(uiT));

  const pct = (x: number, d = 1) => (x > 0 && x < 0.5 * 10 ** -(d + 2) ? '0%' : `${(100 * x).toFixed(d)}%`);
  const letter = (i: number) => LETTERS[i] ?? '?';
  const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;
  const lerp = (a: number, b: number, s: number) => a + (b - a) * s;

  // ---- tween of the emergence dots when the metric changes (geometry only) ----
  let tweenFrom = $state.raw<number[]>([]), tweenMix = $state(1);
  const target = $derived(sizes.map((r) => norm(metric, r[metric])));
  const shownY = $derived(target.map((y, i) => lerp(tweenFrom[i] ?? y, y, tweenMix)));
  let tweenBase: number[] = [];
  function pickMetric(next: Metric) {
    if (next === metric) return;
    tweenBase = shownY.length ? [...shownY] : sizes.map((r) => norm(metric, r[metric]));
    metric = next;
    tweenFrom = tweenBase; tweenMix = reducedMotion() ? 1 : 0; wake();
  }

  let tl: ReturnType<typeof Timeline> | undefined, stage: HTMLDivElement;
  let t = 0, wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  const restart = () => { seek(0); playing = !reducedMotion(); };

  function setTyped(text: string) {
    typedText = text;
    const p = parsePairs(text);
    if (p) { typedError = ''; typedPairs = p; restart(); }
    else typedError = text.trim() ? 'Type one to six pairs of letters a to h, like c>f a>d.' : '';
    if (!text.trim()) typedPairs = null;
  }
  const showRandom = () => { useRandom = !useRandom; if (useRandom) ask<number[]>('random').then((v) => (random = v), (e: Error) => (failed = e.message)); restart(); };

  onMount(() => {
    const fail = (e: Error) => (failed = e.message);
    if (init === 'shots' || init === 'attention') loadModel('icl').then((w) => (weights = w), fail);
    if (init === 'curve') ask<number[]>('rule').then((v) => (rule = v), fail);
    if (init === 'format') ask<number[]>('formats').then((v) => (formats = v), fail);
    if (init === 'attention') ask<Ablations>('ablations').then((v) => (ablation = v), fail);
    if (init === 'phase' || init === 'emergence') measurements().then(([l, s]) => { log = l as TrainingLog; sizes = (s as { rows: SweepRow[] }).rows; }, fail);

    let lastPaint = 0;
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      let busy = playing;
      if (playing) t = advance({ t, playing, speed }, dt, { duration: DURATION }).t;
      if (tweenMix < 1) { tweenMix = Math.min(1, tweenMix + dt / 0.5); busy = true; }
      if (!busy || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return busy;
    }, { onView: (_in, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; tweenMix = 1; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  // ---- per-mode labels ----
  const legend = {
    shots: [{ label: 'Probability the model gives each letter', color: '--viz-forward' }, { label: 'Letter the first example implies', color: '--accent', shape: 'line' as const }],
    curve: [{ label: 'Examples follow the rule', color: '--viz-forward' }, { label: 'Random answers', color: '--viz-cat-2' }, { label: 'Chance, 1 in 8', color: '--text-muted', shape: 'line' as const }],
    attention: [{ label: 'Example answers', color: '--viz-cat-3' }, { label: 'Query letter and arrow', color: '--accent' }, { label: 'Darker cell = more attention', color: '--viz-seq-5' }],
    phase: [{ label: 'Loss with 4 examples', color: '--viz-forward' }, { label: 'Loss with no examples', color: '--text-muted', shape: 'line' as const }, { label: 'Attention on the query letter', color: '--accent' }],
    emergence: [{ label: 'One trained model per size', color: '--viz-forward' }, { label: 'Chance', color: '--text-muted', shape: 'line' as const }],
    format: [{ label: 'Share answered right, 500 tasks', color: '--viz-forward' }],
  }[init];
  const label = {
    shots: 'Bars show the probability the model gives to each letter as examples are added to the prompt',
    curve: 'Accuracy over 500 new tasks against the number of examples in the prompt',
    attention: 'Where the last token of the prompt looks, for each of the four attention heads, with attention shown as shaded cells',
    phase: 'Training loss, attention on the query letter and an attention map across training',
    emergence: 'Accuracy of six trained models of different sizes under a chosen scoring rule',
    format: 'Accuracy over 500 tasks for five ways of writing the same examples',
  }[init];
  const aspect = { shots: [300, 305], curve: [320, 340], attention: [285, 285], phase: [350, 585], emergence: [300, 295], format: [265, 320] }[init];
  const mono = (s: string) => s.replaceAll('\n', '↵').replaceAll(' ', '·');

  const status = $derived.by(() => {
    if (failed) return failed;
    if (init === 'shots') {
      const r = rows[f.index];
      if (!r) return 'Loading the model...';
      const guess = `its top guess is ${letter(r.best)} (${pct(r.probs[r.best])})`;
      return r.count === 0
        ? `No examples: ${guess}, and the letter b gets ${pct(r.probs[1])}.`
        : `${plural(r.count, 'example')}: ${guess}; the letter the first example implies, ${letter(r.implied)}, gets ${pct(r.probs[r.implied])}.`;
    }
    if (init === 'curve') {
      if (!rule) return 'Running 500 tasks for each example count...';
      const k = COUNTS[f.index];
      return `${plural(k, 'example')}: ${pct(rule[f.index])} of 500 tasks answered right${useRandom && random ? `; with random answers ${pct(random[f.index])}` : useRandom ? '; random-answer run in progress...' : ''}.`;
    }
    if (init === 'attention') {
      if (!att) return 'Loading the model...';
      const top = topPositions(att.weights[head]).slice(0, 3).map((r) => `${DEFAULT_PROMPT[r.position] === ' ' ? 'a space' : DEFAULT_PROMPT[r.position]} at position ${r.position + 1} (${pct(r.weight, 0)})`);
      const p = att.probs[1];
      return `${headName(head)}${headOff ? ' switched off' : ''}. The last token looks mostly at ${top.join(', ')}. The model gives b ${pct(p)}${ablation && headOff ? `; with this head off it answers ${pct(ablation.accuracyOff[head], 1)} of 500 new tasks right` : ''}.`;
    }
    if (init === 'phase') {
      if (!pf) return 'Loading the training measurements...';
      return `Update ${log!.log[pf.row].step}: loss with 4 examples ${pf.loss4.toFixed(2)}, with none ${pf.loss0.toFixed(2)}; ${pct(pf.acc4, 0)} of 4-example answers right; the query letter gets ${pct(pf.profile.query, 0)} of the attention.`;
    }
    if (init === 'emergence') {
      const r = sizes[f.index];
      if (!r) return 'Loading the model sizes...';
      return `${r.N.toLocaleString('en-US')} parameters: all four letters right ${pct(r.exact)}; each letter right ${pct(r.perLetter)}; log-probability of the answer ${r.meanLogp.toFixed(Math.abs(r.meanLogp) < 0.1 ? 3 : 2)} (chance ${CHANCE.meanLogp.toFixed(2)}).`;
    }
    if (!formats) return 'Running 500 tasks for each format...';
    return `${FORMATS[f.index].label} (${FORMATS[f.index].note}): ${pct(formats[f.index])} of 500 tasks answered right.`;
  });

  // ---- drawing helpers ----
  const cell = (v: number, reveal = 1) => `color-mix(in srgb, var(--viz-seq-5) ${Math.round(100 * Math.min(1, v) * reveal)}%, var(--viz-seq-1))`;
  const px = 46;
  const plot = $derived({ x: px, w: W - px - 14 });
  const ticksK = [0, 500, 1000, 1500, 2000];
  // Token roles in the default prompt (positions of example answers, the query letter and its arrow).
  const ANSWERS = [2, 6, 10, 14];
  const sizeX = (n: number) => { const a = Math.log(sizes[0].N), b = Math.log(sizes[sizes.length - 1].N); return plot.x + 10 + ((Math.log(n) - a) / (b - a)) * (plot.w - 20); };
  const tri = $derived(log && pf ? triangle(log.thumbnails[pf.thumb].weights) : []);
</script>

<div class="lab">
  <FigureCard {title} {label} {legend} aspect="720 / {aspect[0]}" narrowAspect="288 / {aspect[1]}" role={init === 'phase' || init === 'emergence' || init === 'curve' || init === 'format' || init === 'attention' || init === 'shots' ? 'group' : 'img'}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {W} {aspect[narrow ? 1 : 0]}" aria-hidden="true">
        {#if failed}
          <text x={W / 2} y="40" text-anchor="middle" class="small">{failed}</text>
        {:else if init === 'shots'}
          {#if rows[f.index]}
            {@const r = rows[f.index]}
            {@const prev = rows[Math.max(0, f.index - 1)]}
            {@const cw = narrow ? 7.8 : 9}
            {@const bh = 150}
            {@const base = 96 + bh}
            <text x="8" y="20" class="small">Prompt: examples, then the question</text>
            {#each pairs as p, i (i)}
              {#if i < r.count}
                <text x={8 + i * cw * 4} y="50" class="code" style:font-size="{cw / 0.6}px" opacity={i === r.count - 1 && f.index > 0 ? f.settle : 1}>{p}</text>
              {/if}
            {/each}
            <text x={8 + lerp(prev.count, r.count, f.settle) * cw * 4} y="50" class="code query" style:font-size="{cw / 0.6}px">{DEMO.query}&gt;?</text>
            {#each r.probs as p, i (i)}
              {@const slot = (W - 16) / 8}
              {@const h = lerp(prev.probs[i], p, f.settle) * bh}
              <rect x={8 + i * slot + 4} y={base - h} width={slot - 8} height={Math.max(0, h)} class="bar" />
              {#if i === r.implied}<rect x={8 + i * slot + 1} y={base - bh - 22} width={slot - 2} height={bh + 26} class="mark" />{/if}
              <text x={8 + i * slot + slot / 2} y={base - h - 5} text-anchor="middle" class="small">{pct(p)}</text>
              <text x={8 + i * slot + slot / 2} y={base + 18} text-anchor="middle" class="letter">{LETTERS[i]}</text>
            {/each}
            <line x1="8" x2={W - 8} y1={base} y2={base} class="axis" />
            <text x="8" y={base + 44} class="small">The model: 7,740 numbers, the same at every step.</text>
          {:else}
            <text x={W / 2} y="60" text-anchor="middle" class="small">Loading the model...</text>
          {/if}
        {:else if init === 'curve'}
          {@const top = 16}
          {@const ph = narrow ? 190 : 200}
          {@const base = top + ph}
          {@const xs = (i: number) => plot.x + 10 + (COUNTS[i] / 6) * (plot.w - 20)}
          {@const ys = (v: number) => base - v * ph}
          {#each [0, 0.5, 1] as g (g)}
            <line x1={plot.x} x2={plot.x + plot.w} y1={ys(g)} y2={ys(g)} class="grid" />
            <text x={plot.x - 6} y={ys(g) + 4} text-anchor="end" class="small">{pct(g, 0)}</text>
          {/each}
          <line x1={plot.x} x2={plot.x + plot.w} y1={ys(1 / 8)} y2={ys(1 / 8)} class="chance" />
          <text x={plot.x + plot.w} y={ys(1 / 8) - 5} text-anchor="end" class="small">chance 12.5%</text>
          {#snippet curve(values: number[], cls: string)}
            {@const last = f.index}
            {@const pts = values.slice(0, last + 1).map((v, i) => [xs(i), ys(v)])}
            {@const seg = last > 0 ? [lerp(xs(last - 1), xs(last), f.settle), lerp(ys(values[last - 1]), ys(values[last]), f.settle)] : [xs(0), ys(values[0])]}
            <polyline points={[...pts.slice(0, -1), seg].map((p) => p.join(',')).join(' ')} class="line {cls}" />
            {#each pts as p, i (i)}
              <circle cx={p[0]} cy={p[1]} r={i === last ? 5.5 : 4} class="dot {cls}" opacity={i === last ? Math.max(f.settle, last === 0 ? 1 : 0) : 1} />
            {/each}
          {/snippet}
          {#if rule}
            {#if useRandom && random}{@render curve(rule, 'ghost')}{@render curve(random, 'alt')}{:else}{@render curve(rule, 'main')}{/if}
          {:else}
            <text x={plot.x + plot.w / 2} y={top + ph / 2} text-anchor="middle" class="small">Running 500 tasks for each example count...</text>
          {/if}
          {#each [0, 1, 2, 3, 4, 5, 6] as k (k)}
            <text x={plot.x + 10 + (k / 6) * (plot.w - 20)} y={base + 18} text-anchor="middle" class="small" class:strong={k === COUNTS[f.index]}>{k}</text>
          {/each}
          <text x={plot.x + plot.w / 2} y={base + 34} text-anchor="middle" class="small">Examples in the prompt</text>
          <text x="8" y={base + 58} class="small">A prompt with {plural(COUNTS[f.index], 'example')}:</text>
          <text x="8" y={base + 78} class="code" style:font-size="{narrow ? 13 : 14}px">{samples[f.index]}</text>
        {:else if init === 'attention'}
          {#if att}
            {@const n = DEFAULT_PROMPT.length}
            {@const x0 = narrow ? 8 : 124}
            {@const cwid = (W - x0 - 8) / n}
            {@const tokY = narrow ? 66 : 70}
            {@const rowH = narrow ? 18 : 28}
            {@const rowGap = narrow ? 38 : 38}
            {@const rowY = (h: number) => tokY + 44 + h * rowGap + (narrow ? 14 : 0)}
            {@const qx = x0 + (n - 0.5) * cwid}
            {#each topPositions(att.weights[head], 0.03) as a (a.position)}
              {#if a.position !== n - 1}
                {@const ax = x0 + (a.position + 0.5) * cwid}
                {@const lift = Math.min(tokY - 14, 10 + Math.abs(qx - ax) * 0.18)}
                <path d="M {ax} {tokY - 2} Q {(ax + qx) / 2} {tokY - 2 - lift * 2} {qx} {tokY - 2}" class="arc" stroke-width={1 + 4.5 * a.weight} opacity={(0.25 + 0.7 * a.weight) * (head < 2 ? af.layer1 : af.layer2)} />
              {/if}
            {/each}
            {#each [...DEFAULT_PROMPT] as c, i (i)}
              <text x={x0 + (i + 0.5) * cwid} y={tokY + 19} text-anchor="middle" class="tok" class:ans={ANSWERS.includes(i)} class:q={i >= n - 2}>{c === ' ' ? '·' : c}</text>
            {/each}
            {#each [...DEFAULT_PROMPT] as _, i (i)}
              <text x={x0 + (i + 0.5) * cwid} y={tokY + 33} text-anchor="middle" class="small">{i + 1}</text>
            {/each}
            {#each HEADS as _, h (h)}
              {@const reveal = h < 2 ? af.layer1 : af.layer2}
              {@const off = headOff && h === head}
              <g opacity={off ? 0.3 : 1}>
                <text x={narrow ? 8 : 8} y={rowY(h) + (narrow ? -4 : rowH / 2 + 4)} class="small" class:strong={h === head}>{headName(h)}{off ? ' (off)' : ''}</text>
                {#each att.weights[h] as w, i (i)}
                  <rect x={x0 + i * cwid + 1} y={rowY(h)} width={cwid - 2} height={rowH} rx="2" fill={cell(w, reveal)} class="cell" />
                {/each}
                {#if h === head}<rect x={x0 - 1} y={rowY(h) - 2} width={n * cwid + 2} height={rowH + 4} class="mark" />{/if}
              </g>
            {/each}
          {:else}
            <text x={W / 2} y="60" text-anchor="middle" class="small">Loading the model...</text>
          {/if}
        {:else if init === 'phase'}
          {#if pf && log}
            {@const sx = (s: number) => plot.x + (Math.min(s, PHASE_STEPS) / PHASE_STEPS) * (narrow ? plot.w : 380 - px)}
            {@const pw = narrow ? plot.w : 380 - px}
            {@const lossY = (v: number) => 24 + 126 * (1 - v / 2.6)}
            {@const attY = (v: number) => 214 + 70 * (1 - v)}
            {@const shown = log.log.slice(0, pf.row + 1)}
            {#each [0, 1, 2] as g (g)}
              <line x1={plot.x} x2={plot.x + pw} y1={lossY(g)} y2={lossY(g)} class="grid" />
              <text x={plot.x - 6} y={lossY(g) + 4} text-anchor="end" class="small">{g}</text>
            {/each}
            <text x="8" y="14" class="small">Loss (lower is better)</text>
            <polyline points={shown.map((r) => `${sx(r.step)},${lossY(r.loss0)}`).join(' ')} class="line muted" />
            <polyline points={shown.map((r) => `${sx(r.step)},${lossY(r.loss4)}`).join(' ')} class="line main" />
            <text x={plot.x + pw} y={lossY(2.1) - 16} text-anchor="end" class="small">no examples</text>
            <text x={plot.x + pw} y={lossY(0.05) - 16} text-anchor="end" class="small">4 examples</text>
            <line x1={sx(pf.step)} x2={sx(pf.step)} y1="24" y2={narrow ? 284 : 284} class="cursor" />
            <text x="8" y="204" class="small">Share of attention on the query letter</text>
            {#each [0, 0.5, 1] as g (g)}
              <line x1={plot.x} x2={plot.x + pw} y1={attY(g)} y2={attY(g)} class="grid" />
              <text x={plot.x - 6} y={attY(g) + 4} text-anchor="end" class="small">{pct(g, 0)}</text>
            {/each}
            <polyline points={shown.map((r) => `${sx(r.step)},${attY(r.profile.query)}`).join(' ')} class="line acc" />
            {#each ticksK as s (s)}
              <text x={sx(s)} y="302" text-anchor="middle" class="small">{s}</text>
            {/each}
            <text x={plot.x + pw / 2} y="318" text-anchor="middle" class="small">Training updates</text>
            {@const cs = narrow ? 9 : 10}
            {@const tx = narrow ? (W - 18 * cs) / 2 : 420 + (300 - 18 * cs) / 2}
            {@const ty = narrow ? 372 : 56}
            <text x={tx} y={ty - 22} class="small strong">Attention map, update {log.thumbnails[pf.thumb].step}</text>
            <text x={tx} y={ty - 8} class="small">layer 2, head 2, on the example prompt</text>
            {#each tri as row, r (r)}
              {#each row as v, c (c)}
                {#if c <= r}<rect x={tx + c * cs} y={ty + r * cs} width={cs - 0.5} height={cs - 0.5} fill={cell(v)} />{/if}
              {/each}
            {/each}
            <rect x={tx - 1} y={ty + 17 * cs - 1} width={18 * cs + 2} height={cs + 1} class="mark" />
            <text x={tx} y={ty + 18 * cs + 16} class="small">Each row: where one token looks.</text>
            <text x={tx} y={ty + 18 * cs + 30} class="small">Outlined row: the last token, which predicts.</text>
          {:else}
            <text x={W / 2} y="60" text-anchor="middle" class="small">Loading the training measurements...</text>
          {/if}
        {:else if init === 'emergence'}
          {#if sizes.length}
            {@const top = 30}
            {@const ph = narrow ? 190 : 200}
            {@const base = top + ph}
            {@const ys = (v: number) => base - v * ph}
            {@const m = METRICS.find((x) => x.id === metric)!}
            {#each [0, 0.5, 1] as g (g)}
              <line x1={plot.x} x2={plot.x + plot.w} y1={ys(g)} y2={ys(g)} class="grid" />
              <text x={plot.x - 6} y={ys(g) + 4} text-anchor="end" class="small">{metric === 'meanLogp' ? (-9 + 9 * g).toFixed(g === 0.5 ? 1 : 0) : pct(g, 0)}</text>
            {/each}
            <line x1={plot.x} x2={plot.x + plot.w} y1={ys(norm(metric, CHANCE[metric]))} y2={ys(norm(metric, CHANCE[metric]))} class="chance" />
            <text x={(sizeX(sizes[2].N) + sizeX(sizes[3].N)) / 2} y={ys(norm(metric, CHANCE[metric])) - 5} text-anchor="middle" class="small">chance</text>
            <polyline points={shownY.slice(0, f.index + 1).map((y, i) => `${sizeX(sizes[i].N)},${ys(y)}`).join(' ')} class="line main thin" />
            {#each sizes as r, i (r.N)}
              {#if i <= f.index}
                <circle cx={sizeX(r.N)} cy={ys(shownY[i])} r={i === f.index ? 7 : 5} class="dot main" opacity={i === f.index ? Math.max(0.35, f.settle) : 1} />
                {#if i === f.index}<circle cx={sizeX(r.N)} cy={ys(shownY[i])} r="11" class="ring" opacity={f.settle} />{/if}
              {/if}
              <text x={sizeX(r.N)} y={base + 18} text-anchor="middle" class="small" class:strong={i === f.index}>{(r.N / 1000).toFixed(1)}k</text>
              <text x={sizeX(r.N)} y={base + 33} text-anchor="middle" class="small">d {r.d}</text>
            {/each}
            <text x={plot.x + plot.w / 2} y={base + 54} text-anchor="middle" class="small">Parameters (log scale) and width d</text>
            <text x="8" y="12" class="small">{m.label}</text>
          {:else}
            <text x={W / 2} y="60" text-anchor="middle" class="small">Loading the model sizes...</text>
          {/if}
        {:else}
          {#if formats}
            {@const lx = narrow ? 100 : 150}
            {@const bw = W - lx - 56}
            {@const y0 = 72}
            {@const rowH = narrow ? 44 : 36}
            <text x="8" y="18" class="small">Prompt in this format (· space, ↵ new line):</text>
            <text x="8" y="44" class="code" style:font-size="{narrow ? 13 : 15}px">{mono(prompts[f.index])}</text>
            {#each FORMATS as fm, i (fm.id)}
              {@const y = y0 + i * rowH}
              {@const grown = i < f.index ? 1 : i === f.index ? f.settle : 0}
              <text x="8" y={y + 18} class="small" class:strong={i === f.index}>{fm.label}</text>
              <rect x={lx} y={y + 4} width={bw} height="18" class="track" />
              <rect x={lx} y={y + 4} width={Math.max(0, formats[i] * bw * grown)} height="18" class="bar" />
              {#if i === f.index}<rect x={lx - 2} y={y + 2} width={bw + 4} height="22" class="mark" />{/if}
              <text x={lx + bw + 6} y={y + 18} class="small" class:strong={i === f.index}>{i <= f.index ? pct(formats[i]) : ''}</text>
            {/each}
          {:else}
            <text x={W / 2} y="60" text-anchor="middle" class="small">Running 500 tasks for each format...</text>
          {/if}
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={DURATION} {markers}
        label={init === 'phase' ? 'Training progress' : 'Timeline'}
        onSeek={(to) => { t = to; wake(); }} onReset={() => restart()} />
    {/snippet}
    {#snippet readout()}<p class="status" aria-live="polite">{status}</p>{/snippet}
    {#snippet footer()}
      {#if init === 'shots'}
        <div class="field">
          <label for="icl-pairs">Your own examples (up to six pairs, letters a to h)</label>
          <input id="icl-pairs" type="text" class="text" placeholder="c>f a>d e>h" maxlength="30" autocomplete="off" spellcheck="false" value={typedText} oninput={(e) => setTyped(e.currentTarget.value)} />
          <p class="note" class:err={!!typedError}>{typedError || 'The model is asked about g. The first pair sets the rule the bars are compared with.'}</p>
        </div>
      {:else if init === 'curve'}
        <label class="field check"><input type="checkbox" checked={useRandom} onchange={showRandom} /> Give the examples random answers instead of the rule</label>
      {:else if init === 'attention'}
        <div class="field">
          <div class="row" role="group" aria-label="Attention head">
            {#each HEADS as _, h (h)}<button type="button" class="chip" class:on={h === head} aria-pressed={h === head} onclick={() => { head = h; headOff = false; wake(); }}>{headName(h)}</button>{/each}
          </div>
          <label class="check"><input type="checkbox" bind:checked={headOff} /> Switch this head off</label>
        </div>
      {:else if init === 'emergence'}
        <div class="field">
          <div class="row" role="group" aria-label="How to score an answer">
            {#each METRICS as m (m.id)}<button type="button" class="chip" class:on={metric === m.id} aria-pressed={metric === m.id} onclick={() => pickMetric(m.id)}>{m.label}</button>{/each}
          </div>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .small { fill: var(--text-muted); font: 11px var(--font-ui); font-variant-numeric: tabular-nums; }
  .small.strong, .strong { fill: var(--text); font-weight: 650; }
  .code { fill: var(--text); font-family: var(--font-code); }
  .code.query { fill: var(--accent); font-weight: 650; }
  .letter { fill: var(--text); font: 600 14px var(--font-code); }
  .tok { fill: var(--text); font: 13px var(--font-code); }
  .tok.ans { fill: var(--viz-cat-3); font-weight: 700; }
  .tok.q { fill: var(--accent); font-weight: 700; }
  .bar { fill: var(--viz-forward); }
  .track { fill: var(--bg-sunken); }
  .mark { fill: none; stroke: var(--accent); stroke-width: 2; }
  .axis { stroke: var(--text-muted); }
  .grid { stroke: var(--rule); stroke-dasharray: 3 3; }
  .chance { stroke: var(--text-muted); stroke-dasharray: 5 4; fill: none; }
  .cursor { stroke: var(--text-muted); stroke-width: 1; }
  .line { fill: none; stroke-width: 2.5; stroke-linejoin: round; }
  .line.thin { stroke-width: 1.5; opacity: 0.6; }
  .line.main, .dot.main { stroke: var(--viz-forward); }
  .dot.main { fill: var(--viz-forward); }
  .line.alt { stroke: var(--viz-cat-2); }
  .dot.alt { fill: var(--viz-cat-2); }
  .line.ghost { stroke: var(--viz-forward); opacity: 0.35; stroke-dasharray: 5 4; }
  .dot.ghost { fill: var(--viz-forward); opacity: 0.35; }
  .line.muted { stroke: var(--text-muted); stroke-dasharray: 5 4; }
  .line.acc { stroke: var(--accent); }
  .dot { stroke: var(--bg-raised); stroke-width: 1.5; }
  .ring { fill: none; stroke: var(--accent); stroke-width: 2; }
  .arc { fill: none; stroke: var(--accent); stroke-linecap: round; }
  .cell { stroke: none; }
  .status { margin: 0; min-height: 3.4em; font-variant-numeric: tabular-nums; }
  .field { display: grid; gap: var(--s-2); padding: var(--s-3) var(--s-4); font-size: 0.9rem; }
  .field.check { display: flex; align-items: center; gap: var(--s-2); cursor: pointer; }
  .check { display: inline-flex; align-items: center; gap: var(--s-2); cursor: pointer; }
  .row { display: flex; gap: var(--s-2); flex-wrap: wrap; }
  .text { min-width: 0; padding: 0.5rem 0.6rem; font: 0.9rem var(--font-code); color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .note { margin: 0; color: var(--text-muted); font-size: 0.82rem; }
  .note.err { color: var(--text); font-weight: 600; }
  .chip { font: 0.85rem var(--font-ui); padding: 0.35em 0.7em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .chip.on { border-color: var(--accent); font-weight: 650; }
  input[type='checkbox'] { accent-color: var(--accent); }
  input:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
