<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, markerAt, prefersReducedMotion, reducedMotion } from '../_shared';
  import { loadModel } from '../../../lib/transformer/load';
  import { buildScene, flowChips, fmtChange, fmtLoss, frameAt, layoutOf, SENTENCE, type Mode, type Pt, type Scene, type Models } from './sim';

  interface Props { mode?: Mode; sentence?: string; prompt?: string; layer?: number; head?: number; query?: number; editAt?: number; title?: string }
  let { mode = 'mask', sentence = SENTENCE, prompt, layer = 1, head = 1, query = 3, editAt = 4, title = 'Which words can each word look at?' }: Props = $props();
  const init = untrack(() => mode);
  const uid = $props.id();

  let models = $state.raw<Models | null>(null);
  let curves = $state.raw<{ masked: Pt[]; free: Pt[] } | null>(null);
  let failed = $state(false);
  let editPos = $state(untrack(() => editAt));
  let width = $state(0), playing = $state(false), speed = $state(0.5), uiT = $state(0);
  const narrow = $derived(width > 0 && width < 540);
  const wide = layoutOf(init, false), small = layoutOf(init, true);
  const lay = $derived(narrow ? small : wide);

  const scene: Scene | null = $derived.by(() => {
    if (!models) return null;
    try {
      return buildScene(init, models, { sentence, prompt, layer, head, query, editAt: editPos }, curves ?? undefined);
    } catch { return null; }
  });
  const duration = $derived(scene?.duration ?? 10);
  const frame = $derived(scene ? frameAt(uiT, scene) : null);
  const phaseLabel = $derived(scene ? (markerAt(uiT, scene.markers)?.label ?? '') : '');

  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline>;
  let t = 0, wake = $state<() => void>(() => {}), lastPaint = 0;
  $effect(() => { playing; speed; narrow; scene; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  function restart() { t = 0; uiT = 0; playing = !reducedMotion(); wake(); }
  $effect(() => { if (scene) untrack(restart); });

  onMount(() => {
    (async () => {
      try {
        const needFree = init === 'leak' || init === 'edit';
        const [causal, free] = await Promise.all([loadModel('tiny-lm'), needFree ? loadModel('tiny-lm-nomask') : Promise.resolve(undefined)]);
        if (init === 'leak') {
          const [a, b] = await Promise.all(['tiny-lm', 'tiny-lm-nomask'].map((n) => fetch(`/models/${n}.json`).then((r) => r.json())));
          curves = { masked: a.lossCurve, free: b.lossCurve };
        }
        models = { causal, free };
      } catch { failed = true; }
    })();
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration }).t;
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  const legend = {
    leak: [{ label: 'Trained with the mask', color: '--viz-cat-1', shape: 'line' as const }, { label: 'Trained without it', color: '--viz-cat-2', shape: 'line' as const }, { label: 'Word breaks the grammar', color: '--accent' }],
    mask: [{ label: 'Allowed', color: '--viz-seq-3' }, { label: 'Forbidden (a later word)', color: '--text-muted' }],
    encdec: [{ label: 'Can look at', color: '--viz-seq-3' }, { label: 'Cannot look at', color: '--text-muted' }],
    neginf: [{ label: 'Positive score', color: '--viz-pos' }, { label: 'Negative score', color: '--viz-neg' }, { label: 'Weight', color: '--viz-seq-4' }, { label: 'Weight without the mask', color: '--text-muted', shape: 'line' as const }],
    parallel: [{ label: 'Probability of the right next word', color: '--viz-seq-4' }, { label: 'Dots: words this position can use', color: '--viz-seq-3' }],
    edit: [{ label: 'Probability of the top guess', color: '--viz-seq-4' }, { label: 'Row changed by the edit', color: '--accent' }],
  }[init];
  const label = {
    leak: 'Loss curves of two models and the sentences they write',
    mask: 'A grid of which words each word may attend to',
    encdec: 'Two attention grids, encoder and decoder',
    neginf: 'Attention scores before and after masking and softmax',
    parallel: 'Next-word predictions for every position of a sentence',
    edit: 'Next-word predictions before and after changing one word',
  }[init];

  const px = (n: number) => Math.round(n * 10) / 10;
  // ---- leak geometry
  const lk = $derived(narrow ? { x0: 44, x1: 276, y0: 30, y1: 165, gx: 8, gy: 214, gw: 272, cw: 6.6, bh: 118 } : { x0: 52, x1: 340, y0: 34, y1: 292, gx: 384, gy: 0, gw: 328, cw: 7, bh: 138 });
  const logY = (v: number) => lk.y1 - ((Math.log10(Math.max(v, 1e-4)) + 4) / 5) * (lk.y1 - lk.y0);
  const stepX = (s: number) => lk.x0 + (s / 6000) * (lk.x1 - lk.x0);
  const curvePath = (pts: Pt[], k: number) => pts.slice(0, k + 1).map((p, i) => `${i ? 'L' : 'M'}${px(stepX(p.step))},${px(logY(p.loss))}`).join(' ');
  const leakChips = (g: { words: string[] }) => {
    if (scene?.mode !== 'leak') return { labels: [] as string[], chips: [], nPrompt: 0 };
    const labels = [...scene.prompt, ...g.words];
    return { labels, chips: flowChips(labels, lk.gw, lk.cw, 24).chips, nPrompt: scene.prompt.length };
  };
  const leakBlocks = $derived(scene?.mode === 'leak' ? [
    { name: 'Trained with the mask', g: scene.gens.masked, shown: frame?.kind === 'leak' ? frame.shown.masked : 0, ...leakChips(scene.gens.masked) },
    { name: 'Trained without the mask', g: scene.gens.free, shown: frame?.kind === 'leak' ? frame.shown.free : 0, ...leakChips(scene.gens.free) },
  ] : []);

  // ---- grid geometry (mask, encdec)
  const grid = $derived(init === 'mask' ? (narrow ? { cell: 26, gx: 62, gy: 70 } : { cell: 30, gx: 110, gy: 74 }) : (narrow ? { cell: 20, gx: 62, gy: 82 } : { cell: 26, gx: 82, gy: 90 }));
  const idx = (n: number) => Array.from({ length: n }, (_, i) => i);
  const sw = $derived(scene && scene.mode !== 'leak' ? scene.words : ([] as string[]));
  const T = $derived(sw.length);
  const sideChips = (labels: string[], maxW: number) => flowChips(labels, maxW, 7, 22);

  // ---- neginf geometry
  const ng = $derived(narrow ? { x0: 24, x1: 280, bw: 20 } : { x0: 56, x1: 704, bw: 44 });
  const colW = $derived((ng.x1 - ng.x0) / (T || 1));
  const colX = (i: number) => ng.x0 + (i + 0.5) * colW;
  const nTop = $derived(narrow ? { y0: 92, half: 50 } : { y0: 100, half: 70 });
  const nBot = $derived(narrow ? { base: 262, h: 66 } : { base: 312, h: 96 });
  const labelY = $derived(nBot.base + 14);
  const scoreDom = $derived(scene?.mode === 'neginf' ? Math.ceil(Math.max(...scene.scaled.map(Math.abs)) * 1.1 * 2) / 2 : 1);
  const wDom = $derived(scene?.mode === 'neginf' ? Math.ceil(Math.max(...scene.masked, ...scene.unmasked) * 10 + 0.5) / 10 : 1);

  // ---- rows geometry (parallel, edit)
  const rg = $derived(narrow ? { rh: 52, bx: 112, bw: 100, top: 30, right: 280 } : { rh: 44, bx: 230, bw: 290, top: 30, right: 708 });
  const rows = $derived(scene?.mode === 'parallel' ? scene.rows : scene?.mode === 'edit' && frame?.kind === 'edit' ? scene.variants[frame.variant].rows : []);
  const pct = (p: number) => p.toFixed(2);
  const topText = (r: { top: { word: string; p: number }[] }) => r.top.map((x) => `${x.word} ${pct(x.p)}`).join(' · ');
</script>

<div class="cm">
  <FigureCard {title} {label} {legend} aspect="{wide.w} / {wide.h}" narrowAspect="{small.w} / {small.h}">
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {lay.w} {lay.h}" aria-hidden="true" class:nw={narrow}>
        <defs>
          <pattern id="hatch-{uid}" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M-1 7 L7 -1" class="hatchline" /></pattern>
          <clipPath id="clip-{uid}"><rect x="0" y={nTop.y0 - nTop.half - 6} width={lay.w} height={nTop.half * 2 + 12} /></clipPath>
        </defs>
        {#if failed}
          <text x={lay.w / 2} y={lay.h / 2} text-anchor="middle" class="small">The model could not be loaded.</text>
        {:else if !scene || !frame}
          <text x={lay.w / 2} y={lay.h / 2} text-anchor="middle" class="small">Loading the model…</text>
        {:else if scene.mode === 'leak' && frame.kind === 'leak'}
          <text x="8" y="16" class="heading">Training loss (log scale)</text>
          {#each [-4, -3, -2, -1, 0, 1] as e (e)}
            <line x1={lk.x0} x2={lk.x1} y1={logY(10 ** e)} y2={logY(10 ** e)} class="gridline" />
            <text x={lk.x0 - 6} y={logY(10 ** e) + 4} text-anchor="end" class="small">{10 ** e}</text>
          {/each}
          {#each [0, 3000, 6000] as s (s)}<text x={stepX(s)} y={lk.y1 + 16} text-anchor={s === 6000 ? 'end' : s === 0 ? 'start' : 'middle'} class="small">{s}</text>{/each}
          <text x={(lk.x0 + lk.x1) / 2} y={lk.y1 + 32} text-anchor="middle" class="small">Training updates</text>
          <path d={curvePath(scene.curves.masked, frame.k)} class="curve masked" />
          <path d={curvePath(scene.curves.free, frame.k)} class="curve free" />
          <circle cx={stepX(frame.step)} cy={logY(frame.masked)} r="4.5" class="dot masked" />
          <circle cx={stepX(frame.step)} cy={logY(frame.free)} r="4.5" class="dot free" />
          <text x={lk.x1} y={logY(scene.curves.masked.at(-1)!.loss) - 12} text-anchor="end" class="small tag masked">with the mask: {fmtLoss(frame.masked)}</text>
          <text x={lk.x1} y={logY(scene.curves.free.at(-1)!.loss) - 46} text-anchor="end" class="small tag free">without it: {fmtLoss(frame.free)}</text>
          <text x={lk.gx} y={lk.gy + 16} class="heading">Writing a sentence, one word at a time</text>
          {#each leakBlocks as b, bi (bi)}
            {@const oy = lk.gy + 36 + bi * lk.bh}
            <text x={lk.gx} y={oy} class="small tag" class:masked={bi === 0} class:free={bi === 1}>{b.name}</text>
            {#each b.chips as c, i (i)}
              {@const gi = i - b.nPrompt}
              {@const bad = gi >= 0 && b.g.bad[gi]}
              {#if gi < 0 || gi < b.shown}
                <g class="word" class:prompt={gi < 0} class:bad>
                  <rect x={lk.gx + c.x} y={oy + 10 + c.y} width={c.w} height="24" rx="3" class="chip" />
                  <text x={lk.gx + c.x + c.w / 2} y={oy + 26 + c.y} text-anchor="middle" class="token">{b.labels[i]}</text>
                </g>
              {/if}
            {/each}
            {#if b.shown >= b.g.words.length && frame.phase === 'write'}
              {@const nbad = b.g.bad.filter(Boolean).length}
              <text x={lk.gx} y={oy + 12 + 3 * 28 + 6} class="small">{nbad ? `${nbad} of ${b.g.words.length} new words break the grammar` : 'No new word breaks the grammar'}</text>
            {/if}
          {/each}
        {:else if scene.mode === 'mask' && frame.kind === 'mask'}
          {@render gridBlock(0, grid.gx, grid.gy, 'mask', frame.row, frame.alpha)}
          {#if !narrow}
            {@const can = scene.words.slice(0, frame.row + 1)}
            {@const cannot = scene.words.slice(frame.row + 1)}
            <text x="408" y="86" class="heading">Reading: {scene.words[frame.row]}</text>
            <text x="408" y="112" class="small">Can look at ({can.length})</text>
            {#each sideChips(can, 224).chips as c, i (i)}
              <rect x={408 + c.x} y={120 + c.y} width={c.w} height="22" rx="3" class="chip allowed" />
              <text x={408 + c.x + c.w / 2} y={135 + c.y} text-anchor="middle" class="token">{can[i]}</text>
            {/each}
            <text x="408" y="226" class="small">Cannot look at ({cannot.length})</text>
            {#each sideChips(cannot, 224).chips as c, i (i)}
              <rect x={408 + c.x} y={234 + c.y} width={c.w} height="22" rx="3" class="chip forbidden" />
              <text x={408 + c.x + c.w / 2} y={249 + c.y} text-anchor="middle" class="token muted">{cannot[i]}</text>
            {/each}
            <text x="408" y="316" class="count">{frame.count} of {frame.total} cells allowed</text>
          {/if}
        {:else if scene.mode === 'encdec' && frame.kind === 'encdec'}
          {#if narrow}
            {@render gridBlock(0, grid.gx, grid.gy, 'enc', frame.row, 1)}
            {@render gridBlock(1, grid.gx, grid.gy + 262, 'dec', frame.row, 1)}
          {:else}
            {@render gridBlock(0, grid.gx, grid.gy, 'enc', frame.row, 1)}
            {@render gridBlock(1, 382, grid.gy, 'dec', frame.row, 1)}
          {/if}
        {:else if scene.mode === 'neginf' && frame.kind === 'neginf'}
          {@const q = scene.q}
          <rect x={ng.x0 + (q + 1) * colW} y={nTop.y0 - nTop.half - 12} width={(T - q - 1) * colW} height={nBot.base - nTop.y0 + nTop.half + 12} fill="url(#hatch-{uid})" class="band" />
          <text x="8" y="14" class="heading">{narrow ? `Scores of “${scene.words[q]}”` : `Score of “${scene.words[q]}” against each word (layer ${scene.layer}, head ${scene.head})`}</text>
          <line x1={ng.x0 - 8} x2={ng.x1 + 8} y1={nTop.y0} y2={nTop.y0} class="axis" />
          <text x={ng.x0 - 12} y={nTop.y0 + 4} text-anchor="end" class="small">0</text>
          <g clip-path="url(#clip-{uid})">
            {#each scene.scaled as s, i (i)}
              {@const fut = i > q}
              {@const v = s * frame.scoreGrow}
              {@const h = (Math.abs(v) / scoreDom) * nTop.half}
              {@const dy = fut ? frame.slide * (nTop.half * 2 + 20) : 0}
              <g transform="translate(0 {px(dy)})" opacity={fut ? 1 - 0.6 * frame.slide : 1}>
                <rect x={colX(i) - ng.bw / 2} y={v >= 0 ? nTop.y0 - h : nTop.y0} width={ng.bw} height={h} class={v >= 0 ? 'pos' : 'neg'} />
                <text x={colX(i)} y={v >= 0 ? nTop.y0 - h - 4 : nTop.y0 + h + 12} text-anchor="middle" class="val">{v.toFixed(2)}</text>
              </g>
            {/each}
          </g>
          {#each scene.scaled as _, i (i)}
            {#if i > q}<text x={colX(i)} y={nTop.y0 - 8} text-anchor="middle" opacity={frame.inf} class="inf">−∞</text>{/if}
          {/each}
          <text x="8" y={nBot.base - nBot.h - 18} class="heading">Weight after the softmax</text>
          <text x={lay.w - 8} y={nBot.base - nBot.h - 18} text-anchor="end" class="num" opacity={frame.showWeights}>sum = {frame.sum.toFixed(3)}</text>
          <line x1={ng.x0 - 8} x2={ng.x1 + 8} y1={nBot.base} y2={nBot.base} class="axis" />
          {#each scene.unmasked as u, i (i)}
            {@const fut = i > q}
            {@const a = fut ? frame.futureGhost : frame.ghost}
            {@const hh = (u / wDom) * nBot.h * (fut ? frame.futureH : 1)}
            {#if a > 0}
              {#if hh > 0.5}<rect x={colX(i) - ng.bw / 2} y={nBot.base - hh} width={ng.bw} height={hh} opacity={a} class="ghost" />{/if}
              {#if fut}<text x={colX(i)} y={nBot.base - hh - 4} text-anchor="middle" opacity={a} class="val muted">{frame.phase === 'wrong' ? '0.00' : u.toFixed(2)}</text>{/if}
            {/if}
          {/each}
          {#each frame.weights as w, i (i)}
            {#if frame.showWeights > 0.01}
              <rect x={colX(i) - ng.bw / 2} y={nBot.base - (w / wDom) * nBot.h} width={ng.bw} height={(w / wDom) * nBot.h} class="weight" />
              {#if i <= q}<text x={colX(i)} y={nBot.base - (w / wDom) * nBot.h - 4} text-anchor="middle" opacity={frame.showWeights} class="val">{frame.labels[i].toFixed(2)}</text>{/if}
            {/if}
          {/each}
          {#each scene.words as w, i (i)}
            <text transform={narrow ? `translate(${px(colX(i))} ${labelY}) rotate(40)` : ''} x={narrow ? 0 : colX(i)} y={narrow ? 0 : labelY + 4} text-anchor={narrow ? 'start' : 'middle'} class="small" class:muted={i > q} class:query={i === q}>{w}</text>
          {/each}
          {#if !narrow}
            <text x={ng.x0 + ((q + 1) * colW) / 2} y={labelY + 28} text-anchor="middle" class="small">words up to and including “{scene.words[q]}”</text>
            <text x={ng.x0 + (q + 1) * colW + ((T - q - 1) * colW) / 2} y={labelY + 28} text-anchor="middle" class="small muted">later words</text>
          {/if}
        {:else if (scene.mode === 'parallel' && frame.kind === 'parallel') || (scene.mode === 'edit' && frame.kind === 'edit')}
          <text x="12" y="16" class="heading">{scene.mode === 'edit' ? (narrow ? 'In → top guess' : 'Word in → its top guess for the next word') : narrow ? 'In → should predict' : 'Word in → word it should predict'}</text>
          <text x={rg.right} y="16" text-anchor="end" class="heading">{scene.mode === 'edit' ? 'Change' : 'Loss'}</text>
          {#each rows as r, i (i)}
            {@const y = rg.top + i * rg.rh}
            {@const a = frame.kind === 'parallel' ? frame.alpha[i] : 1}
            {@const edited = scene.mode === 'edit' && i === scene.pos}
            {@const moved = scene.mode === 'edit' && frame.kind === 'edit' && r.change > 0}
            <g opacity={a}>
              {#if moved && frame.kind === 'edit'}<rect x="2" y={y + 2} width={lay.w - 4} height={rg.rh - 4} rx="3" class="flash" opacity={0.08 + 0.2 * frame.flash} />{/if}
              {#if frame.kind === 'parallel' && frame.active === i}<rect x="2" y={y + 2} width={lay.w - 4} height={rg.rh - 4} rx="3" class="flash" opacity="0.15" />{/if}
              <text x="12" y={y + 18} class="token" class:edited>{r.input}{scene.mode === 'edit' ? ` → ${r.top[0].word}` : ` → ${r.target}`}{edited && !narrow ? '  (edited)' : ''}</text>
              <rect x={rg.bx} y={y + 6} width={rg.bw} height="14" class="track" />
              <rect x={rg.bx} y={y + 6} width={px(Math.max(r.bar, 0) * rg.bw)} height="14" class="weight" />
              {#if !narrow}<text x={rg.bx + rg.bw + 8} y={y + 18} class="small">p {r.bar.toFixed(3)}</text>{/if}
              <text x={rg.right} y={y + 18} text-anchor="end" class="num" class:changed={moved}>
                {#if scene.mode === 'parallel'}{narrow ? '' : 'loss '}{fmtLoss(r.loss * (frame.kind === 'parallel' ? frame.count : 1))}{:else}{narrow ? '' : 'change '}{fmtChange(r.change)}{/if}
              </text>
              <text x="12" y={y + 36} class="small muted">{narrow ? '' : 'top guesses: '}{topText(r)}</text>
              {#each narrow ? [] : idx(scene.words.length) as j (j)}
                <circle cx={rg.right - (scene.words.length - 1 - j) * 10 - 3} cy={y + 32} r="3" class="use" class:on={j < r.uses} />
              {/each}
              <line x1="0" x2={lay.w} y1={y + rg.rh} y2={y + rg.rh} class="rowline" />
            </g>
          {/each}
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} markers={scene?.markers ?? []} label="{label} timeline" onSeek={seek} onReset={() => seek(0)} />
    {/snippet}
    {#snippet readout()}
      <p class="status">{frame?.status ?? ' '}</p>
      {#if scene?.mode === 'neginf' && frame?.kind === 'neginf' && frame.phase !== 'scores' && frame.phase !== 'mask'}
        <p class="num">Without the mask, {(scene.futureShare * 100).toFixed(1)}% of the weight would land on the {T - scene.q - 1} later words.</p>
      {:else if scene?.mode === 'parallel' && frame?.kind === 'parallel'}
        <p class="num">Forward passes so far: {frame.passes} · Average loss over the positions shown: {frame.avg.toFixed(3)}</p>
      {:else if scene?.mode === 'edit' && frame?.kind === 'edit'}
        {@const v = scene.variants[frame.variant]}
        {#if frame.variant === 0}
          <p class="num">Pick any word to change below, or let the timeline cycle through replacements.</p>
        {:else}
          <p class="num">Largest change at earlier positions: {fmtChange(v.maxBefore)} · from the edited word onward: {fmtChange(v.maxFrom)}{v.freeBefore !== null ? ` · same edit, model trained without the mask, earlier positions: ${fmtChange(v.freeBefore)}` : ''}</p>
        {/if}
      {:else if scene?.mode === 'leak' && frame?.kind === 'leak' && frame.phase === 'write'}
        <p class="num">Update 6000 loss: {fmtLoss(scene.curves.masked.at(-1)!.loss)} with the mask, {fmtLoss(scene.curves.free.at(-1)!.loss)} without it.</p>
      {/if}
      <p class="sr" aria-live="polite">{phaseLabel}</p>
    {/snippet}
    {#snippet footer()}
      {#if init === 'edit' && scene?.mode === 'edit'}
        <label class="pick">Word to change
          <select value={editPos} onchange={(e) => { editPos = +(e.currentTarget as HTMLSelectElement).value; }} aria-label="Word to change">
            {#each scene.words as w, i (i)}{#if i > 0}<option value={i}>{w}</option>{/if}{/each}
          </select>
        </label>
      {/if}
    {/snippet}
  </FigureCard>
</div>

{#snippet gridBlock(which: number, gx: number, gy: number, kind: 'mask' | 'enc' | 'dec', row: number, alpha: number)}
  {@const cell = grid.cell}
  {#if kind !== 'mask'}
    <text x={gx} y={gy - 62} class="heading">{kind === 'enc' ? 'Encoder' : 'Decoder'}</text>
    <text x={gx} y={gy - 48} class="small">{kind === 'enc' ? (narrow ? 'sees every word' : 'every word sees every word') : narrow ? 'sees itself and the past' : 'each word sees itself and the past'}</text>
  {/if}
  {#each idx(T) as j (j)}
    <text transform="translate({px(gx + (j + 0.5) * cell)} {gy - 8}) rotate(-45)" class="small" class:query={j === row}>{sw[j]}</text>
  {/each}
  {#each idx(T) as i (i)}
    <text x={gx - 8} y={gy + (i + 0.5) * cell + 4} text-anchor="end" class="small" class:query={i === row}>{sw[i]}</text>
    {#each idx(T) as j (j)}
      {@const ok = kind === 'enc' || j <= i}
      {@const shown = kind !== 'mask' || i < row || (i === row)}
      {@const active = i === row}
      {#if shown}
        <rect x={gx + j * cell} y={gy + i * cell} width={cell} height={cell} class="cell" class:ok class:active opacity={i === row ? alpha : 1} style:fill={ok ? undefined : `url(#hatch-${uid})`} />
      {:else}
        <rect x={gx + j * cell} y={gy + i * cell} width={cell} height={cell} class="cell empty" />
      {/if}
    {/each}
  {/each}
  <rect x={gx - 1} y={gy + row * cell - 1} width={T * cell + 2} height={cell + 2} class="rowbox" />
  {#if kind !== 'mask'}
    <text x={gx} y={gy + T * cell + 20} class="small">{kind === 'enc' ? (narrow ? 'Fills in a hidden word' : 'Trained by filling in a hidden word') : narrow ? 'Predicts the next word' : 'Trained by predicting the next word'}</text>
  {/if}
{/snippet}

<style>
  .stagebox { position: absolute; inset: 0; }
  .heading { fill: var(--text); font: 600 13px var(--font-ui); }
  .small { fill: var(--text-muted); font: 11px var(--font-code); }
  .small.muted, .muted { fill: var(--text-muted); opacity: 0.8; }
  .small.query { fill: var(--accent); font-weight: 700; }
  .token { fill: var(--text); font: 12px var(--font-code); }
  .nw .token { font-size: 11px; }
  .token.muted { fill: var(--text-muted); }
  .token.edited { fill: var(--accent); font-weight: 700; }
  .num { fill: var(--text); font: 12px var(--font-code); font-variant-numeric: tabular-nums; }
  .num.changed { fill: var(--accent); font-weight: 700; }
  .count { fill: var(--text); font: 600 15px var(--font-code); font-variant-numeric: tabular-nums; }
  .val { fill: var(--text); font: 10.5px var(--font-code); font-variant-numeric: tabular-nums; }
  .val.muted { fill: var(--text-muted); }
  .inf { fill: var(--text); font: 600 16px var(--font-code); }
  .gridline { stroke: var(--rule); stroke-dasharray: 3 3; }
  .axis { stroke: var(--text-muted); stroke-width: 1; }
  .rowline { stroke: var(--rule); }
  .hatchline { stroke: var(--text-muted); stroke-width: 1; opacity: 0.55; }
  .band { opacity: 0.35; }
  .curve { fill: none; stroke-width: 2.5; }
  .curve.masked, .dot.masked, .tag.masked { stroke: var(--viz-cat-1); }
  .curve.free, .dot.free, .tag.free { stroke: var(--viz-cat-2); }
  .dot.masked { fill: var(--viz-cat-1); stroke: none; }
  .dot.free { fill: var(--viz-cat-2); stroke: none; }
  .small.tag { stroke: none; font-weight: 700; }
  .small.tag.masked { fill: var(--viz-cat-1); }
  .small.tag.free { fill: var(--viz-cat-2); }
  .chip { fill: var(--viz-seq-2); stroke: var(--rule); }
  .prompt .chip { fill: var(--viz-seq-1); }
  .chip.allowed { fill: var(--viz-seq-3); }
  .chip.forbidden { fill: var(--bg-raised); stroke: var(--rule); stroke-dasharray: 3 2; }
  .bad .chip { fill: var(--accent); stroke: var(--accent); }
  .bad .token { fill: var(--on-accent, #fff); font-weight: 700; }
  .word { animation: pop 0.3s ease-out; }
  @keyframes pop { from { opacity: 0; } to { opacity: 1; } }
  .cell { stroke: var(--rule); fill: none; }
  .cell.ok { fill: var(--viz-seq-2); }
  .cell.ok.active { fill: var(--viz-seq-4); }
  .cell.empty { fill: none; }
  .rowbox { fill: none; stroke: var(--accent); stroke-width: 2; }
  .pos { fill: var(--viz-pos); }
  .neg { fill: var(--viz-neg); }
  .weight { fill: var(--viz-seq-4); }
  .ghost { fill: none; stroke: var(--text-muted); stroke-dasharray: 4 3; }
  .track { fill: var(--viz-seq-1); }
  .flash { fill: var(--accent); }
  .use { fill: none; stroke: var(--text-muted); stroke-width: 1; }
  .use.on { fill: var(--viz-seq-3); stroke: var(--viz-seq-3); }
  .status { margin: 0; min-height: 2.8em; }
  p.num { margin: 0.3em 0 0; font: inherit; font-variant-numeric: tabular-nums; }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); margin: 0; }
  .pick { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-3); padding: var(--s-3) var(--s-4); }
  .pick select { font: 0.9rem var(--font-code); padding: 0.4rem 0.6rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  select:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  @media (prefers-reduced-motion: reduce) { .word { animation: none; } }
</style>
