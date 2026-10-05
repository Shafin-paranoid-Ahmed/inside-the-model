<script module lang="ts">
  import { loadModel } from '../../../lib/transformer/load';
  import { makeData, type Data, type SnapMeta } from './sim';
  let shared: Promise<Data> | undefined;
  /** One fetch of the trained table and the eight checkpoints per page; every figure shares it. */
  function load(): Promise<Data> {
    return (shared ??= (async () => {
      const [w, json, bin] = await Promise.all([loadModel('tiny-lm'), fetch('/models/tiny-lm-snapshots.json'), fetch('/models/tiny-lm-snapshots.bin')]);
      if (!json.ok || !bin.ok) throw new Error('Could not load embedding checkpoints');
      return makeData(w, (await json.json()) as SnapMeta, new Uint8Array(await bin.arrayBuffer()));
    })());
  }
  const forget = () => { shared = undefined; };
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import {
    CLASS_NAMES, CLASS_TOKENS, GROUPS, PRESETS, PROBES, STAGES, buildScene, prepareContext, classOf, clusterWords, fmt, frameAt, idOf, neighbours, project,
    type Metric, type Mode, type Pt,
  } from './sim';
  import { VOCAB } from '../../../lib/transformer/grammar';

  interface Props {
    /** lookup, training, similarity, projection, arithmetic or contextual. */
    mode?: Mode;
    title?: string;
    /** lookup: the sentence (words from the 34-word vocabulary). */
    sentence?: string;
    /** projection: the word whose nearest neighbours are drawn first. */
    select?: string;
    /** arithmetic: show every preset in turn instead of only the first. */
    cycle?: boolean;
  }
  let { mode = 'lookup', title = '', sentence, select = 'sheep', cycle = false }: Props = $props();
  const init = untrack(() => mode);
  const PLACEHOLDER: Record<Mode, string[]> = {
    lookup: ['Look up each token', 'The finished matrix'], training: ['Random', 'Early', 'Middle', 'Trained'], similarity: PROBES,
    projection: ['Rotate', ...GROUPS.map((g) => g.label)], arithmetic: ['Offset', 'Copy', 'Land', 'Nearest'], contextual: ['Embedding', 'Block 1', 'Block 2'],
  };
  const hasAside = init === 'training' || init === 'projection' || init === 'arithmetic' || init === 'contextual';

  let data = $state.raw<Data | null>(null);
  let failed = $state(false);
  let width = $state(0), playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let probeChoice = $state('cycle'), metric = $state<Metric>('cosine'), presetChoice = $state(untrack(() => (cycle ? 'cycle' : '0')));
  let selected = $state(untrack(() => idOf(select))), withThe = $state(false);
  const narrow = $derived(width > 0 && width < 540);
  const sizes: Record<Mode, [[number, number], [number, number]]> = {
    lookup: [[720, 400], [340, 400]], similarity: [[720, 430], [340, 440]], training: [[600, 440], [340, 340]],
    projection: [[600, 440], [340, 340]], arithmetic: [[600, 230], [340, 200]], contextual: [[600, 440], [340, 340]],
  };
  const [[WW, WH], [NW, NH]] = sizes[init];
  const W = $derived(narrow ? NW : WW), H = $derived(narrow ? NH : WH);

  const scene = $derived(data ? buildScene(init, data, {
    sentence, metric,
    probes: probeChoice === 'cycle' ? PROBES : [probeChoice],
    presets: presetChoice === 'cycle' ? PRESETS.map((p) => [...p] as [string, string, string]) : [[...PRESETS[Number(presetChoice)]] as [string, string, string]],
    third: withThe ? 'the' : null,
  }) : null);
  const duration = $derived(scene?.T ?? 1);
  const markers = $derived(scene ? scene.markers : PLACEHOLDER[init].map((label, i) => ({ at: i * 0.1, label })));
  const frame = $derived(scene ? frameAt(uiT, scene) : null);

  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline> | undefined;
  let t = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; scene; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  const restart = () => { t = 0; uiT = 0; wake(); };

  onMount(() => {
    load().then(async (d) => { if (init === 'contextual') await prepareContext(d); data = d; }, () => { failed = true; forget(); });
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      if (!scene) return false;
      t = advance({ t, playing, speed }, dt, { duration: scene.T }).t;
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

  // ---- drawing helpers (all numbers come from `frame`)
  const color = (i: number) => `var(${CLASS_TOKENS[classOf(i)]})`;
  function mapper(box: [number, number, number, number], pad: number, uniform = true) {
    const dx = box[1] - box[0], dy = box[3] - box[2];
    const sx = (W - 2 * pad) / dx, sy = (H - 2 * pad) / dy, k = Math.min(sx, sy);
    const kx = uniform ? k : sx, ky = uniform ? k : sy;
    const ox = (W - kx * dx) / 2, oy = (H - ky * dy) / 2;
    return { x: (v: number) => ox + (v - box[0]) * kx, y: (v: number) => H - oy - (v - box[2]) * ky };
  }
  const lk = $derived(narrow ? { tx: 30, tcw: 6, ty: 44, tch: 9, mx: 222, mcw: 7, my: 64, mch: 22 } : { tx: 96, tcw: 14, ty: 44, tch: 10, mx: 430, mcw: 16, my: 64, mch: 30 });
  const cellOpacity = (v: number, max: number) => Math.min(1, Math.abs(v) / max);
  const sign = (v: number) => (v < 0 ? 'var(--viz-neg)' : 'var(--viz-pos)');
  const trainingMap = $derived(scene?.mode === 'training' ? mapper(scene.box, 26) : null);
  const projMap = $derived(scene?.mode === 'projection' ? mapper(scene.box, 30) : null);
  const arMap = $derived(scene?.mode === 'arithmetic' ? mapper(scene.each[Math.min(frame?.mode === 'arithmetic' ? frame.k : 0, scene.each.length - 1)].box, 30) : null);
  const ctxMap = $derived(mapper([-1.08, 1.08, -1.08, 1.08], 22, false));

  /** Cluster labels for the 2D map: words that sit together share one label; offsets are chosen once on the final layout. */
  const labelOffsets = $derived.by(() => {
    if (scene?.mode !== 'projection' || !projMap) return [];
    const m = projMap, wd = (c: number[]) => clusterText(c).length * 6 + 6;
    const finalPts = scene.table.map((r) => project(scene.basis, r));
    const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
    const hit = (a: { x0: number; x1: number; y0: number; y1: number }) => placed.filter((b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0).length;
    return scene.clusters.map((c) => {
      const cx = m.x(c.reduce((s, i) => s + finalPts[i][0], 0) / c.length), cy = m.y(c.reduce((s, i) => s + finalPts[i][1], 0) / c.length);
      const w = wd(c), spread = Math.max(...c.map((i) => Math.hypot(m.x(finalPts[i][0]) - cx, m.y(finalPts[i][1]) - cy))) + 9;
      const cands: [number, number, 'start' | 'middle' | 'end'][] = [[0, -spread - 3, 'middle'], [0, spread + 11, 'middle'], [spread + 4, 4, 'start'], [-spread - 4, 4, 'end']];
      let best = cands[0], bestHit = Infinity;
      for (const cand of cands) {
        const x0 = cx + cand[0] - (cand[2] === 'middle' ? w / 2 : cand[2] === 'end' ? w : 0);
        const r = { x0, x1: x0 + w, y0: cy + cand[1] - 11, y1: cy + cand[1] + 3 };
        const h = hit(r) + (r.x0 < 2 || r.x1 > W - 2 ? 5 : 0) + (r.y0 < 2 || r.y1 > H - 2 ? 5 : 0);
        if (h < bestHit) { bestHit = h; best = cand; if (!h) break; }
      }
      const x0 = cx + best[0] - (best[2] === 'middle' ? w / 2 : best[2] === 'end' ? w : 0);
      placed.push({ x0, x1: x0 + w, y0: cy + best[1] - 11, y1: cy + best[1] + 3 });
      return { dx: best[0], dy: best[1], anchor: best[2] };
    });
  });
  /** Narrow screens: a long group label shrinks to its first word and a count. */
  const clusterText = (c: number[]) => (narrow && c.length > 2 ? `${VOCAB[c[0]]} +${c.length - 1}` : c.map((i) => VOCAB[i]).join(' '));
  const nn = $derived(data && selected >= 0 ? neighbours(data.table, selected, 3) : []);
  const arrowId = $props.id();

  function pick(i: number) { selected = i; }
  const legendClasses = CLASS_NAMES.map((label, i) => ({ label, color: CLASS_TOKENS[i] }));
  const legend = init === 'contextual'
    ? [{ label: 'Singular nouns', color: '--viz-pos' }, { label: 'Plural nouns', color: '--viz-neg' }, { label: 'Sheep in the sentence', color: '--accent' }]
    : init === 'lookup' ? [{ label: 'Positive number', color: '--viz-pos' }, { label: 'Negative number', color: '--viz-neg' }]
    : init === 'arithmetic' ? [{ label: 'Arrow', color: '--accent', shape: 'line' as const }, { label: 'Animal noun', color: '--viz-cat-3' }, { label: 'Landing point', color: '--text-muted' }]
    : legendClasses;
  const figureLabel: Record<Mode, string> = {
    lookup: 'Table rows copied into a matrix, one per token', training: 'Word points moving as training proceeds', similarity: 'Sorted bars of similarity to a probe word',
    projection: 'Two-dimensional map of the embedding table', arithmetic: 'An offset arrow copied from one word to another', contextual: 'Two sheep vectors separating as blocks are applied',
  };
</script>

{#snippet cells(row: number[], x: number, y: number, w: number, h: number, max: number)}
  {#each row as v, j}
    <rect x={x + j * w} y={y} width={w} height={h} fill={sign(v)} fill-opacity={cellOpacity(v, max) * 0.9} class="cell" />
  {/each}
{/snippet}

{#snippet list(items: { a: string; b: string; on?: boolean; dim?: boolean }[])}
  <ol class="steps num">
    {#each items as it}<li class:on={it.on} class:dim={it.dim}><span>{it.a}</span><b>{it.b}</b></li>{/each}
  </ol>
{/snippet}

{#snippet asideSnip()}
  {#if scene?.mode === 'training' && frame?.mode === 'training'}
    <p class="aside-head">cosine(cat, dog) at each checkpoint</p>
    <ol class="steps two num">
      {#each scene.steps as step, i}
        <li class:on={i === frame.idx}><span>step {step.toLocaleString('en-US')}</span><b>{fmt(scene.cosines[i])}</b></li>
      {/each}
    </ol>
    <p class="note">All checkpoints use the trained table's two directions of greatest spread.</p>
  {:else if scene?.mode === 'projection' && frame?.mode === 'projection' && data}
    <p class="aside-head">The map keeps <b class="num">{(scene.share * 100).toFixed(1)}%</b> of the spread</p>
    <p class="note">The other {(100 - scene.share * 100).toFixed(1)}% of the variance is not on the page.</p>
    {#if selected >= 0}
      <p class="aside-head">Nearest to “{VOCAB[selected]}” in all 16 numbers</p>
      {@render list(nn.map((n) => ({ a: n.word, b: fmt(n.value) })))}
    {:else}<p class="note">Tap a point, or pick a word below, to see its nearest neighbours.</p>{/if}
  {:else if scene?.mode === 'arithmetic' && frame?.mode === 'arithmetic'}
    {@const e = scene.each[frame.k]}
    <p class="aside-head">{e.names[0]} − {e.names[1]} + {e.names[2]}: nearest words</p>
    {@render list(e.near.map((n, i) => ({ a: `${i + 1}. ${i < frame.shown ? n.word : '…'}`, b: i < frame.shown ? fmt(n.value) : '', dim: i >= frame.shown })))}
    <p class="note">Offsets {e.names[0]} − {e.names[1]} and {e.near[0].word} − {e.names[2]}: cosine <b class="num">{fmt(e.offsetCos!, 3)}</b></p>
  {:else if scene?.mode === 'contextual' && frame?.mode === 'contextual'}
    <p class="aside-head">cosine between the two “sheep”</p>
    {@render list(STAGES.map((name, i) => ({ a: name, b: fmt(scene.cos[i][1]), on: i === frame.stage })))}
    {#if withThe}
      <p class="aside-head">“the sheep” after block 2</p>
      <p class="note num">vs “this”: {fmt(scene.cos[2][2])}<br />vs “these”: {fmt(scene.cosThese[2])}</p>
    {/if}
    <p class="aside-head">Table row for “sheep”</p>
    <svg class="strip" viewBox="0 0 160 14" role="img" aria-label="The 16 numbers in the embedding-table row for sheep">
      {#each scene.row as v, j}<rect x={j * 10} y="0" width="10" height="14" fill={sign(v)} fill-opacity={cellOpacity(v, 0.7)} />{/each}
    </svg>
    <p class="note">The same row feeds both sentences. Only the layers that follow tell the two apart.</p>
  {:else}
    <p class="note">Loading…</p>
  {/if}
{/snippet}

<div class="embed">
  <FigureCard {title} label={figureLabel[init]} role="group" aspect="{WW} / {WH}" narrowAspect="{NW} / {NH}" {legend}
    aside={hasAside ? asideSnip : undefined} asideLabel="Numbers shown in the figure">
    {#snippet children()}
      <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
        <svg viewBox="0 0 {W} {H}" aria-hidden="true">
          {#if !frame || !scene}
            <text x={W / 2} y={H / 2} text-anchor="middle" class="muted">{failed ? 'Could not load the trained table. Reload to try again.' : 'Loading the trained embedding table…'}</text>

          {:else if frame.mode === 'lookup' && scene.mode === 'lookup'}
            {@const { tx, tcw, ty, tch, mx, mcw, my, mch } = lk}
                        <text x={tx} y="20" class="heading">Table: {VOCAB.length} words × 16</text>
            <text x={mx} y="20" class="heading">Sentence: {scene.ids.length} × 16</text>
            {#each scene.table as row, i}
              {@render cells(row, tx, ty + i * tch, tcw, tch, scene.maxAbs)}
              {#if scene.ids.includes(i)}
                <text x={tx - 6} y={ty + i * tch + tch - 1.5} text-anchor="end" class="tick" class:on={frame.active >= 0 && scene.ids[frame.active] === i}>{narrow ? i : `${VOCAB[i]} · ${i}`}</text>
              {/if}
            {/each}
            {#each scene.ids as id, k}
              {@const gy = my + k * mch}
              {@const drawn = k < frame.done || (k === frame.active && frame.slide > 0)}
              {#if !drawn}
                <rect x={mx} y={gy} width={mcw * 16} height={mch - 2} class="slot" />
              {/if}
              {#if drawn}
                {@const s = k < frame.done ? 1 : frame.slide}
                {@const cx = tx + (mx - tx) * s}
                {@const cy = ty + id * tch + (gy - (ty + id * tch)) * s}
                {@const cw = tcw + (mcw - tcw) * s}
                {@const ch = tch + (mch - 2 - tch) * s}
                {@render cells(scene.table[id], cx, cy, cw, ch, scene.maxAbs)}
              {/if}
              {#if drawn}
                <text x={mx - 6} y={gy + mch / 2 + 3} text-anchor="end" class="tick on">{VOCAB[id]}</text>
              {/if}
            {/each}
            {#if frame.active >= 0}
              {@const id = scene.ids[frame.active]}
              {@const ay = ty + id * tch + tch / 2}
              <rect x={tx - 1} y={ty + id * tch - 1} width={tcw * 16 + 2} height={tch + 2} class="ring" opacity={frame.pointer} />
              <line x1={tx + tcw * 16 + 2} y1={ay} x2={tx + tcw * 16 + 2 + (mx - tx - tcw * 16 - 10) * frame.pointer} y2={ay + (my + frame.active * mch + mch / 2 - ay) * frame.pointer} class="link" opacity={1 - frame.slide} />
            {/if}

          {:else if frame.mode === 'training' && scene.mode === 'training' && trainingMap}
            {@const m = trainingMap}
            {@const c = frame.points[scene.cat]}
            {@const d = frame.points[scene.dog]}
            <line x1={m.x(c[0])} y1={m.y(c[1])} x2={m.x(d[0])} y2={m.y(d[1])} class="pair" />
            {#each frame.points as p, i}
              <circle cx={m.x(p[0])} cy={m.y(p[1])} r={i === scene.cat || i === scene.dog ? 6.5 : 5} fill={color(i)} fill-opacity={i === scene.cat || i === scene.dog ? 1 : 0.75} />
            {/each}
            <text x={m.x(c[0]) + 9} y={m.y(c[1]) - 8} class="label">cat</text>
            <text x={m.x(d[0]) - 9} y={m.y(d[1]) + 16} text-anchor="end" class="label">dog</text>
            <text x="12" y="20" class="heading num">Step {frame.step.toLocaleString('en-US')} of {scene.steps.at(-1)?.toLocaleString('en-US')}</text>

          {:else if frame.mode === 'similarity' && scene.mode === 'similarity'}
            {@const lab = narrow ? 18 : 26}
            {@const left = narrow ? 74 : 110}
            {@const right = W - (narrow ? 52 : 74)}
            {@const cx = (left + right) / 2}
            {@const half = (right - left) / 2}
            {@const rh = narrow ? 12 : 11.4}
            <text x="12" y="14" class="heading">Closeness of “{VOCAB[frame.probe]}” to every other word ({scene.metric === 'cosine' ? 'cosine' : 'dot product'})</text>
            {#each [-1, 0, 1] as g}
              <line x1={cx + g * half} x2={cx + g * half} y1="34" y2={34 + 33 * rh} class="grid" class:zero={g === 0} />
              <text x={cx + g * half} y="29" text-anchor="middle" class="tick num">{g * scene.domain > 0 ? '+' : ''}{g * scene.domain}</text>
            {/each}
            {#each frame.rows as r}
              {@const y = 34 + r.rank * rh}
              {@const x2 = cx + (r.value / scene.domain) * half}
              <circle cx={lab - 8} cy={y + rh / 2 - 1} r="3" fill={color(r.id)} />
              <text x={lab} y={y + rh - 2.5} class="tick on">{VOCAB[r.id]}</text>
              <rect x={Math.min(cx, x2)} y={y + 1.5} width={Math.abs(x2 - cx)} height={rh - 3} fill={sign(r.value)} fill-opacity="0.85" />
              <text x={r.value >= 0 ? x2 + 4 : x2 - 4} y={y + rh - 2.5} text-anchor={r.value >= 0 ? 'start' : 'end'} class="tick num">{fmt(r.value, 3)}</text>
            {/each}

          {:else if frame.mode === 'projection' && scene.mode === 'projection' && projMap}
            {@const m = projMap}
            {#if selected >= 0}
              {#each nn as n}
                <line x1={m.x(frame.points[selected][0])} y1={m.y(frame.points[selected][1])} x2={m.x(frame.points[n.id][0])} y2={m.y(frame.points[n.id][1])} class="link" opacity={frame.settle} />
              {/each}
            {/if}
            {#each frame.points as p, i}
              {@const on = frame.group < 0 || GROUPS[frame.group].has(i)}
              <circle cx={m.x(p[0])} cy={m.y(p[1])} r={i === selected ? 7 : 5.5} fill={color(i)} fill-opacity={on ? 0.9 : 0.15} stroke={i === selected ? 'var(--accent)' : 'var(--bg-raised)'} stroke-width={i === selected ? 2.5 : 1}
                role="button" tabindex="-1" aria-label={VOCAB[i]} onclick={() => pick(i)} onkeydown={(e) => { if (e.key === 'Enter') pick(i); }} class="pt" />
            {/each}
            {#each scene.clusters as c, ci}
              {@const o = labelOffsets[ci]}
              {#if o}
                {@const cx = m.x(c.reduce((s, i) => s + frame.points[i][0], 0) / c.length)}
                {@const cy = m.y(c.reduce((s, i) => s + frame.points[i][1], 0) / c.length)}
                <text x={cx + o.dx} y={cy + o.dy} text-anchor={o.anchor} class="label halo" class:sel={selected >= 0 && c.includes(selected)} opacity={frame.settle * (frame.group < 0 || c.some((i) => GROUPS[frame.group].has(i)) ? 1 : 0.25)}>{clusterText(c)}</text>
              {/if}
            {/each}

          {:else if frame.mode === 'arithmetic' && scene.mode === 'arithmetic' && arMap}
            {@const m = arMap}
            {@const e = scene.each[frame.k]}
            {@const pts = e.plane}
            <defs><marker id="{arrowId}-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" /></marker></defs>
            {#if frame.slide > 0}
              <line x1={m.x(pts.b[0])} y1={m.y(pts.b[1])} x2={m.x(pts.a[0])} y2={m.y(pts.a[1])} class="ghost" />
            {/if}
            {#if frame.draw > 0.02}
              <line x1={m.x(frame.arrow[0][0])} y1={m.y(frame.arrow[0][1])} x2={m.x(frame.arrow[1][0])} y2={m.y(frame.arrow[1][1])} class="arrow" marker-end="url(#{arrowId}-head)" />
            {/if}
            <circle cx={m.x(pts.land[0])} cy={m.y(pts.land[1])} r="9" class="landing" opacity={frame.ring} />
            {#each e.near.slice(0, 3) as n, i}
              <circle cx={m.x(n.at[0])} cy={m.y(n.at[1])} r="4.5" fill="var(--viz-cat-3)" opacity={i < frame.shown ? 1 : 0} />
            {/each}
            {#each [[pts.b, e.names[1], -1], [pts.a, e.names[0], 1], [pts.c, e.names[2], -1]] as [p, name, side]}
              <circle cx={m.x((p as Pt)[0])} cy={m.y((p as Pt)[1])} r="5.5" fill="var(--viz-cat-3)" stroke="var(--text)" stroke-width="1.5" />
              <text x={m.x((p as Pt)[0]) + (side as number) * 10} y={m.y((p as Pt)[1]) + (name === e.names[1] ? 16 : -9)} text-anchor={(side as number) < 0 ? 'end' : 'start'} class="label">{name}</text>
            {/each}
            {#if frame.ring > 0.3}
              <text x={m.x(pts.land[0])} y={m.y(pts.land[1]) - 15} text-anchor="middle" class="label" opacity={frame.ring}>{e.names[0]} − {e.names[1]} + {e.names[2]}</text>
            {/if}
            {#if frame.shown > 0}
              <text x={m.x(e.near[0].at[0])} y={m.y(Math.min(...e.near.slice(0, Math.min(3, frame.shown)).map((n) => n.at[1]))) + 24} text-anchor="middle" class="label muted">nearest: {e.near.slice(0, Math.min(3, frame.shown)).map((n) => n.word).join(', ')}</text>
            {/if}

          {:else if frame.mode === 'contextual' && scene.mode === 'contextual'}
            {@const m = ctxMap}
            <text x="12" y="20" class="heading">{STAGES[frame.stage]}</text>
            <line x1={m.x(-1.08)} x2={m.x(1.08)} y1={m.y(0)} y2={m.y(0)} class="grid" />
            {#each frame.cloud as p, i}
              <circle cx={m.x(p[0])} cy={m.y(p[1])} r="3" fill={scene.plural[i] ? 'var(--viz-neg)' : 'var(--viz-pos)'} fill-opacity="0.35" />
            {/each}
            {#each frame.sheep as p, d}
              {@const x = m.x(p[0])}
              {@const flip = x > W * 0.62}
              <circle cx={x} cy={m.y(p[1])} r="7" fill={d === 0 ? 'var(--viz-pos)' : d === 1 ? 'var(--viz-neg)' : 'var(--bg-raised)'} stroke="var(--accent)" stroke-width="2.5" />
              <text x={x + (flip ? -12 : 12)} y={m.y(p[1]) + (d === 2 ? 17 : d === 1 ? 4 : -9)} text-anchor={flip ? 'end' : 'start'} class="label halo">{scene.dets[d]} small sheep</text>
            {/each}
          {/if}
        </svg>
      </div>
    {/snippet}
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} label="Embedding timeline" onSeek={seek} onReset={() => seek(0)} />
    {/snippet}
    {#snippet readout()}
      <p class="status" role="status" aria-live="polite">{frame ? frame.status : failed ? 'The trained table could not be loaded.' : 'Loading the trained embedding table…'}</p>
    {/snippet}
    {#snippet footer()}
      {#if init === 'similarity'}
        <div class="controls">
          <label>Probe word
            <select bind:value={probeChoice} onchange={restart} disabled={!data}>
              <option value="cycle">Cycle: {PROBES.join(', ')}</option>
              {#each VOCAB as v}<option value={v}>{v}</option>{/each}
            </select>
          </label>
          <fieldset>
            <legend>Measure</legend>
            <label><input type="radio" bind:group={metric} value="cosine" onchange={restart} /> Cosine</label>
            <label><input type="radio" bind:group={metric} value="dot" onchange={restart} /> Dot product</label>
          </fieldset>
        </div>
      {:else if init === 'projection'}
        <div class="controls">
          <label>Show neighbours of
            <select bind:value={selected} disabled={!data}>
              <option value={-1}>(none)</option>
              {#each VOCAB as v, i}<option value={i}>{v}</option>{/each}
            </select>
          </label>
        </div>
      {:else if init === 'arithmetic'}
        <div class="controls">
          <label>Word sum
            <select bind:value={presetChoice} onchange={restart} disabled={!data}>
              {#each PRESETS as p, i}<option value={String(i)}>{p[0]} − {p[1]} + {p[2]}</option>{/each}
              <option value="cycle">Cycle through all</option>
            </select>
          </label>
        </div>
      {:else if init === 'contextual'}
        <div class="controls">
          <label><input type="checkbox" bind:checked={withThe} onchange={restart} disabled={!data} /> Also show “the small sheep” (“the” does not say singular or plural)</label>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .heading { fill: var(--text); font: 600 13px var(--font-ui); }
  .muted { fill: var(--text-muted); font: 13px var(--font-ui); }
  .label { fill: var(--text); font: 600 11.5px var(--font-ui); }
  .label.muted { fill: var(--text-muted); font-weight: 500; }
  .label.sel { font-weight: 800; }
  .halo { paint-order: stroke; stroke: var(--bg-raised); stroke-width: 3.5px; stroke-linejoin: round; }
  .tick { fill: var(--text-muted); font: 10px var(--font-code); }
  .tick.on { fill: var(--text); }
  .grid { stroke: var(--rule); stroke-dasharray: 3 3; }
  .grid.zero { stroke: var(--text-muted); stroke-dasharray: none; }
  .ring { fill: none; stroke: var(--accent); stroke-width: 2; }
  .link { stroke: var(--accent); stroke-width: 1.5; fill: none; }
  .slot { fill: none; stroke: var(--rule); stroke-dasharray: 3 3; }
  .pair { stroke: var(--text-muted); stroke-dasharray: 4 3; stroke-width: 1.2; }
  .arrow { stroke: var(--accent); stroke-width: 3; }
  .ghost { stroke: var(--text-muted); stroke-width: 1.5; stroke-dasharray: 5 4; }
  .landing { fill: none; stroke: var(--text-muted); stroke-width: 2; }
  .pt { cursor: pointer; }
  .cell { stroke: var(--bg-raised); stroke-width: 0.5; }
  .status { margin: 0; min-height: 3.2em; }
  .aside-head { margin: 0 0 var(--s-1); font: 600 0.85rem var(--font-ui); }
  .note { margin: var(--s-2) 0; font-size: 0.8rem; color: var(--text-muted); }
  .steps { list-style: none; margin: 0 0 var(--s-2); padding: 0; font-size: 0.8rem; }
  @container (max-width: 58rem) { .steps.two { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: repeat(4, auto); grid-auto-flow: column; column-gap: var(--s-4); } }
  .steps li { display: flex; justify-content: space-between; gap: var(--s-3); padding: 0.1rem 0.3rem; border-left: 2px solid transparent; }
  .steps li.on { border-left-color: var(--accent); font-weight: 650; }
  .steps li.dim { color: var(--text-muted); }
  .strip { display: block; width: 100%; max-width: 12rem; height: auto; }
  .controls { display: flex; flex-wrap: wrap; gap: var(--s-3) var(--s-5); align-items: center; padding: var(--s-3) var(--s-4); font: 0.85rem var(--font-ui); }
  .controls label { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
  .controls fieldset { display: flex; gap: var(--s-3); align-items: center; border: 0; margin: 0; padding: 0; }
  .controls legend { float: left; padding: 0; margin-right: 0.5rem; }
  select { max-width: 100%; font: 0.85rem var(--font-ui); padding: 0.3rem 0.4rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
