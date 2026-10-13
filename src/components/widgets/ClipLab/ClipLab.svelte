<script module lang="ts">
  let shared: Promise<Run> | null = null;
  function trainingRun(): Promise<Run> {
    return shared ??= new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }) => { worker.terminate(); if (data.error) reject(new Error(data.error)); else resolve(data.run); };
      worker.onerror = (event) => { worker.terminate(); shared = null; reject(new Error(event.message)); };
    });
  }
</script>

<script lang="ts">
  import { onMount } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { SHAPES, DURATION, shapeImage, patches, embedPatches, poolPosition, similarities, probabilities, assemble, frameAt, training, type Mode, type Shape, type Run } from './sim';

  let { mode = 'patches', title = 'Image patches and language' }: { mode?: Mode; title?: string } = $props();
  let host: HTMLDivElement, tl: ReturnType<typeof Timeline>;
  let playing = $state(false), speed = $state(0.5), uiT = $state(0), p = $state(4), shape = $state<Shape>('circle'), shuffled = $state(false), prefix = $state(''), pair = $state('order');
  let run = $state.raw<Run | null>(null), error = $state('');
  let t = 0, wake = () => {};
  const image = $derived(shapeImage(shape));
  const pieces = $derived(patches(image, p));
  const frame = $derived(frameAt(uiT, mode, run));
  const tokens = $derived(embedPatches(image));
  const selected = $derived(tokens[5]);
  const positioned = $derived(poolPosition(shuffled ? [...tokens].reverse() : tokens, true));
  const baseline = $derived(poolPosition(tokens, true));
  const change = $derived(Math.sqrt(positioned.reduce((sum, v, i) => sum + (v - baseline[i]) ** 2, 0)));
  const assembled = $derived(assemble(image, p));
  const captions = $derived(SHAPES.map(s => `${prefix}${s}`));
  const scores = $derived(run ? similarities(run.model, shapeImage(shape, true, 1), captions) : []);
  const probs = $derived(probabilities(scores));
  const limitCaptions = $derived(pair === 'order' ? ['circle left of square', 'square left of circle'] : ['large circle', 'small circle']);
  const limitScores = $derived(run ? similarities(run.model, image, limitCaptions) : []);
  const markers = $derived([{ at: 0, label: frame.phase }]);
  $effect(() => { playing; speed; p; shape; shuffled; prefix; pair; wake(); });
  onMount(() => {
    let active = true, last = 0;
    if (['contrastive', 'zeroshot', 'limits'].includes(mode)) trainingRun().then(r => { if (active) { run = r; wake(); } }).catch(e => { if (active) error = String(e); });
    const loop = createLoop(host, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: DURATION }).t;
      tl?.paint(t);
      if (now - last >= 100 || !playing) { uiT = t; last = now; }
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { active = false; loop.destroy(); };
  });
</script>

{#snippet picture(pixels: Float64Array, grid = 4)}
  <svg viewBox="0 0 160 160" role="img" aria-label={`${shape} drawn as 16 by 16 pixels`}>
    {#each pixels as value, i (i)}<rect x={(i % 16) * 10} y={Math.floor(i / 16) * 10} width="10" height="10" fill={value ? 'var(--text)' : 'var(--bg-sunken)'} />{/each}
    {#each Array.from({ length: 16 / grid + 1 }) as _, i (i)}
      <path d={`M${i * grid * 10} 0 V160 M0 ${i * grid * 10} H160`} stroke="var(--rule)" stroke-width="1" />
    {/each}
  </svg>
{/snippet}

<div bind:this={host}>
  <FigureCard {title} label={title} role="group" aspect="auto">
    <div class="stage">
      {#if error}<p role="alert">Training could not load: {error}</p>
      {:else if ['contrastive', 'zeroshot', 'limits'].includes(mode) && !run}<p>Training the image and caption encodersâ€¦</p>
      {:else if mode === 'patches'}
        <div class="two"><div class="image">{@render picture(image, p)}<p>16 Ã— 16 pixels</p></div><div>
          <p>Read patches left to right, then down.</p>
          <div class="patches" style={`--columns:${Math.min(8, pieces.length)}`}>
            {#each pieces as piece, i (i)}<div class:active={i < Math.ceil(frame.progress * pieces.length)}><svg viewBox={`0 0 ${p} ${p}`} role="img" aria-label={`Patch ${i + 1}`}>
              {#each piece as v, j (j)}<rect x={j % p} y={Math.floor(j / p)} width="1" height="1" fill={v ? 'var(--text)' : 'var(--bg-sunken)'} />{/each}
            </svg><span>{i + 1}</span></div>{/each}
          </div>
        </div></div>
        <p class="equation">(16 Ã· {p}) Ã— (16 Ã· {p}) = <strong>{pieces.length} image tokens</strong></p>
        <p>Each token starts with {p} Ã— {p} = {p * p} pixel values.</p>
      {:else if mode === 'embed'}
        <div class="two"><div class="image">{@render picture(image)}<p>Patch 6: row 2, column 2</p></div><div>
          <p>First embedding coordinate: average the patch's pixels.</p>
          <div class="pixels">{#each patches(image)[5] as value, i (i)}<span class:active={i < frame.visible}>{value}</span>{/each}</div>
          <p class="equation">{patches(image)[5].reduce((a, b) => a + b, 0)} lit pixels Ã— 1/16 = <strong>{selected[0].toFixed(4)}</strong></p>
          <p>Add this patch's position value: {selected[0].toFixed(4)} + (âˆ’0.0625) = {(selected[0] - 0.0625).toFixed(4)}.</p>
        </div></div>
        <p>The same 16 â†’ 8 matrix is applied to every patch.</p>
        <p>Reverse patch order: mean alone changes by <strong>0</strong>. Position + tanh + mean changes by <strong>{change.toFixed(5)}</strong>.</p>
        <p class="scope">Position demo: fixed projection and tanh mixing; no transformer attention.</p>
      {:else if mode === 'contrastive' && frame.snapshot}
        {@const snap = frame.snapshot}
        <p>Rows: images 1â€“8. Columns: matching captions 1â€“8. Bright cells mean higher row probability.</p>
        <div class="two"><div>
          <div class="matrix">{#each snap.rows as v, i (i)}<div class:diagonal={Math.floor(i / 8) === i % 8} style={`background:color-mix(in srgb, var(--viz-forward) ${v * 100}%, var(--bg-sunken))`} title={`Image ${Math.floor(i / 8) + 1}, caption ${i % 8 + 1}: ${(100 * v).toFixed(1)}%`}></div>{/each}</div>
        </div><div><ol>{#each training as example, i (i)}<li>{i + 1}: {example.caption}</li>{/each}</ol></div></div>
        <p>Update <strong>{snap.step}</strong> Â· loss <strong>{snap.loss.toFixed(3)}</strong> Â· mean matching probability <strong>{(snap.probability * 100).toFixed(1)}%</strong></p>
        <p>Temperature Ï„ = {(1 / snap.scale).toFixed(3)}; divide cosine scores by Ï„ before softmax.</p>
        <p>For image 1: correct-caption probability {snap.rows[0].toFixed(4)} â†’ âˆ’ln({snap.rows[0].toFixed(4)}) = {(-Math.log(snap.rows[0])).toFixed(3)} row loss.</p>
        <svg class="curve" viewBox="0 0 300 55" role="img" aria-label="Loss falls over training updates"><polyline points={run!.snapshots.filter(s => s.step <= snap.step).map(s => `${s.step / 600 * 290 + 5},${50 - s.loss / 3 * 45}`).join(' ')} fill="none" stroke="var(--viz-forward)" stroke-width="2" /></svg>
        <p class="scope">Trained here: linear image encoder of 16 concatenated patches; mean word embeddings. Eight training pairs.</p>
      {:else if mode === 'zeroshot' && run}
        <div class="two"><div class="image">{@render picture(shapeImage(shape, true, 1))}<p>Held-out: large {shape}, shifted right 1 pixel.</p></div><div>
          {#each scores as score, i (i)}<div class="candidate" class:active={i <= Math.floor(frame.progress * 4)}><p>{captions[i]} <strong>{score.toFixed(3)}</strong></p><div class="track"><span style={`width:${(score + 1) * 50}%`}></span></div><small>{(probs[i] * 100).toFixed(1)}% at temperature 1</small></div>{/each}
        </div></div>
        <p>Choose the highest cosine: <strong>{captions[scores.indexOf(Math.max(...scores))]}</strong>.</p>
        <p>Held-out shape accuracy: <strong>{run.accuracy[['', 'a ', 'large '].indexOf(prefix)] * 16}/16</strong> ({(run.accuracy[['', 'a ', 'large '].indexOf(prefix)] * 100).toFixed(1)}%).</p>
        <p class="scope">Test: four shapes Ã— two sizes Ã— two shifts. These 16 images were excluded from training. Shape names occurred in training captions.</p>
      {:else if mode === 'vlm'}
        <p>Pixels â†’ shared patch map (8 numbers) â†’ fixed projector (12 numbers) â†’ context</p>
        <div class="context">{#each assembled.projected as token, i (i)}<span class="image-token" class:active={i < Math.ceil(frame.progress * assembled.count)} title={`First coordinates: ${Array.from(token.slice(0, 3)).map(v => v.toFixed(3)).join(', ')}`}>I{i + 1}</span>{/each}{#each assembled.text as text, i (i)}<span class:active={assembled.projected.length + i < Math.ceil(frame.progress * assembled.count)}>{text}</span>{/each}</div>
        <p class="equation">{assembled.projected.length} image + {assembled.text.length} text = <strong>{assembled.count} tokens</strong></p>
        <p>Image share: {assembled.projected.length} Ã· {assembled.count} Ã— 100 = <strong>{(assembled.share * 100).toFixed(1)}%</strong>.</p>
        <p class="scope">Token assembly only. Fixed, untrained projector; seven explicit demonstration text tokens; no language model or generated answer.</p>
      {:else if mode === 'limits' && run}
        <div class="two"><div class="image">{@render picture(image)}<p>The same large {shape} image</p></div><div>{#each limitCaptions as caption, i (i)}<div class="candidate" class:active={i <= Math.floor(frame.progress * 2)}><p>â€œ{caption}â€</p><strong>cosine {limitScores[i].toFixed(5)}</strong></div>{/each}</div></div>
        <p>Score difference: <strong>{Math.abs(limitScores[0] - limitScores[1]).toFixed(5)}</strong>.</p>
        <p>Mean word embeddings add the same words in either order. Moving a word cannot change their mean.</p>
        <p class="scope">This encoder ignores word order by construction. The displayed circle does not depict the relational captions.</p>
      {/if}
    </div>
    {#snippet controls()}
      <div class="options">
        {#if ['patches', 'vlm'].includes(mode)}<label>Patch width <select bind:value={p}><option value={2}>2 pixels</option><option value={4}>4 pixels</option><option value={8}>8 pixels</option></select></label>{/if}
        {#if mode === 'zeroshot'}<label>Shape <select bind:value={shape}>{#each SHAPES as s (s)}<option value={s}>{s}</option>{/each}</select></label><label>Caption <select bind:value={prefix}><option value="">circle</option><option value="a ">a circle</option><option value="large ">large circle</option></select></label>{/if}
        {#if mode === 'embed'}<label><input type="checkbox" bind:checked={shuffled} /> Reverse patch order</label>{/if}
        {#if mode === 'limits'}<label>Compare <select bind:value={pair}><option value="order">Word order</option><option value="size">Size words</option></select></label>{/if}
      </div>
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={DURATION} {markers} onSeek={to => { t = to; wake(); }} onReset={() => { t = 0; wake(); }} />
    {/snippet}
    {#snippet readout()}<p aria-live="polite">{frame.phase}</p>{/snippet}
  </FigureCard>
</div>

<style>
  .stage { min-height: 390px; padding: 1rem; font-variant-numeric: tabular-nums; }
  p { margin: 0.55rem 0; line-height: 1.5; }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 1.25rem; align-items: center; }
  .image { max-width: 190px; width: 100%; margin: auto; }
  svg { display: block; width: 100%; }
  .patches { display: grid; grid-template-columns: repeat(var(--columns), minmax(0, 1fr)); gap: 0.3rem; }
  .patches > div { border: 1px solid var(--rule); text-align: center; opacity: 0.3; }
  .patches span { font-size: 0.7rem; }
  .active { opacity: 1 !important; }
  .pixels { display: grid; grid-template-columns: repeat(4, 1fr); max-width: 190px; }
  .pixels span { padding: 0.2rem; text-align: center; border: 1px solid var(--rule); opacity: 0.35; }
  .equation { font-variant-numeric: tabular-nums; }
  .scope, small { color: var(--text-muted); font-size: 0.82rem; }
  .matrix { display: grid; grid-template-columns: repeat(8, 1fr); aspect-ratio: 1; max-width: 260px; gap: 2px; }
  .diagonal { outline: 1px solid var(--text-muted); outline-offset: -1px; }
  ol { padding-left: 0; list-style: none; font-size: 0.8rem; }
  li { line-height: 1.65; }
  .curve { height: 55px; }
  .candidate { margin-bottom: 0.65rem; opacity: 0.3; }
  .candidate p { display: flex; justify-content: space-between; gap: 0.5rem; }
  .track { height: 8px; background: var(--bg-sunken); }
  .track span { display: block; height: 100%; background: var(--viz-forward); }
  .context { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 1.2rem 0; }
  .context span { padding: 0.3rem; border: 1px solid var(--rule); opacity: 0.25; min-width: 2.1rem; text-align: center; }
  .image-token { border-color: var(--viz-forward) !important; }
  .options { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
  label { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  select { background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); padding: 0.3rem; max-width: 100%; }
  input { accent-color: var(--accent); }
  @media (max-width: 540px) { .stage { min-height: 500px; padding: 0.7rem; } .two { grid-template-columns: 1fr; gap: 0.7rem; } .image { max-width: 140px; } .matrix { margin: auto; max-width: 180px; } ol { display: grid; grid-template-columns: 1fr 1fr; } .pixels { margin: auto; } }
</style>
