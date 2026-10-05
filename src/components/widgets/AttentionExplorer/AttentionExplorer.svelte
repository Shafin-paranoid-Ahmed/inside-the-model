<script module lang="ts">
  import type { Weights } from '../../../lib/transformer/model';
  import { loadModel } from '../../../lib/transformer/load';
  /** One fetch per page: every island shares this promise (loadModel also memoises). */
  let pending: Promise<Weights> | null = null;
  const weights = () => (pending ??= loadModel('tiny-lm'));
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar, reducedMotion } from '../_shared';
  import { VOCAB } from '../../../lib/transformer/grammar';
  import { TOKEN_NAMES, type Env, type Hit, type Pal, type RGB } from './ui';
  import { sizeOf } from './size';
  import type { HeadStat } from './stats';
  import {
    DEFAULT_SENTENCE, HEADS, REACH_SENTENCE, headName, makeScene, restTime, startTime, timeline, validate,
    type Frame, type Mode, type Opts, type Scene,
  } from './core';
  // Each figure is its own chunk (m-<mode>.ts), fetched when the island's mode needs it.
  type Impl = { frameAt(t: number, s: Scene, o: Opts): Frame; draw(E: Env, f: Frame): void } & Partial<typeof import('./stats')>;
  const modes = import.meta.glob<Impl>('./m-*.ts');
  const loadImpl = (m: Mode): Promise<Impl> => modes[`./m-${m}.ts`]();

  interface Props {
    /** Which view of attention to show (see the modes in sim.ts). */
    mode?: Mode;
    /** Sentence of vocabulary words; `[start]` is added when missing. Default: these small sheep chase a red fox. */
    sentence?: string;
    /** Layer to show, one-based (1 or 2). */
    layer?: number;
    /** Head to show, one-based (1 or 2). */
    head?: number;
    /** The word the figure follows (modes scores, softmax, mix, concat, split) or opens on (arcs, qkv, compare, explorer, gallery). */
    query?: string | number;
    title?: string;
    /** Show the sentence box (explorer and gallery). */
    editable?: boolean;
  }
  let { mode = 'explorer', sentence, layer = 1, head = 1, query, title = '', editable }: Props = $props();

  const init = untrack(() => ({
    mode, query, title: title || 'Attention figure',
    text: (sentence ?? (mode === 'reach' ? REACH_SENTENCE : DEFAULT_SENTENCE)).replace(/^\[start\]\s*/, ''),
    layer: Math.min(2, Math.max(1, layer)) - 1, head: Math.min(2, Math.max(1, head)) - 1,
    editable: editable ?? (mode === 'explorer' || mode === 'gallery'),
  }));
  // Legend entries by key; each figure lists the keys it needs.
  const KEY: Record<string, [string, string, ('dot' | 'line')?]> = {
    pos: ['Positive', '--viz-pos'], neg: ['Negative', '--viz-neg'], w: ['Weight', '--viz-seq-4'], wd: ['Weight (darker is more)', '--viz-seq-4'],
    det: ['Determiner', '--viz-cat-1'], adj: ['Adjective', '--viz-cat-2'], noun: ['Noun', '--viz-cat-3'], verb: ['Verb', '--viz-cat-4'],
    arc: ['Arc width: weight', '--viz-seq-4', 'line'], in: ['Information has arrived', '--viz-seq-4'], pulse: ['Pulse', '--accent'],
    h1: ['Head 1', '--viz-cat-5'], h2: ['Head 2', '--viz-cat-3'], hatch: ['Hatched: masked', '--text-muted'], full: ['Full model', '--text-muted'], now: ['This setting', '--accent'],
  };
  const LEGENDS: Record<Mode, string> = {
    arcs: 'det adj noun verb arc', qkv: 'pos neg', scores: 'pos neg', softmax: 'pos neg w', mix: 'w pos neg', explorer: 'wd pos neg', reach: 'in pulse',
    compare: 'h1 h2', split: 'pos neg h1 h2', heads: 'wd h1 h2', concat: 'pos neg h1 h2', gallery: 'wd hatch', ablate: 'full now',
  };
  const legend = LEGENDS[init.mode].split(' ').map((k) => ({ label: KEY[k][0], color: KEY[k][1], shape: KEY[k][2] }));
  const SWEEPS: Mode[] = ['arcs', 'qkv', 'explorer', 'compare', 'gallery'];

  // ---- state ----
  let w = $state.raw<Weights | null>(null), failed = $state(false);
  let text = $state(untrack(() => init.text));
  let selLayer = $state(untrack(() => init.layer)), selHead = $state(untrack(() => init.head));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0), width = $state(0);
  let status = $state(''), phase = $state(''), live = $state(''), hoverText = $state('');
  let clickPin = $state<number | null>(null), hoverPin = $state<number | null>(null);
  let toggles = $state([false, false, false, false]), custom = $state(false), offNow = $state([false, false, false, false]);
  let impl = $state.raw<Impl | null>(null);
  let sample = $state.raw<{ heads: HeadStat[]; mass: number[] } | null>(null), customSample = $state<number | null>(null);

  const val = $derived(validate(text));
  let goodIds = $state.raw<number[] | null>(null);
  $effect(() => { if (val.ids) goodIds = val.ids; });
  const scene = $derived<Scene | null>(w && goodIds ? makeScene(w, goodIds, init.mode === 'gallery' || init.mode === 'heads' ? init.layer : selLayer, selHead) : null);
  const narrow = $derived(width > 0 && width < 540);
  // The figure is sized for the current sentence (known at load); editing it may resize the figure.
  const T = $derived((val.ids ?? goodIds)?.length ?? (init.mode === 'reach' ? 9 : 8));
  const wide = $derived(sizeOf(init.mode, T, false)), slim = $derived(sizeOf(init.mode, T, true));
  const size = $derived(narrow ? slim : wide);
  const tl = $derived(timeline(init.mode, scene, init.query));
  const pin = $derived(hoverPin ?? clickPin);
  const masses = $derived(scene && impl?.ablationMasses && init.mode === 'ablate' ? impl.ablationMasses(scene.w, scene.ids) : undefined);
  const customMass = $derived(scene && impl?.ablatedMass && custom && init.mode === 'ablate' ? impl.ablatedMass(scene.w, scene.ids, toggles) : null);
  const sweeps = SWEEPS.includes(init.mode), pinnable = init.mode === 'arcs' || init.mode === 'explorer' || init.mode === 'gallery';

  // ---- canvas + loop ----
  let cv: HTMLCanvasElement, tlc: ReturnType<typeof Timeline>;
  let t = 0, wake = $state<() => void>(() => {}), hits: Hit[] = [], started = false;
  $effect(() => { playing; speed; narrow; scene; impl; pin; custom; toggles.join(); wake(); });
  const seek = (to: number) => { t = Math.min(to, tl.duration); uiT = t; wake(); };
  function opts() {
    return { mode: init.mode, query: init.query, pin: sweeps ? pin : null, mass: masses, toggles: custom ? toggles : null, customMass };
  }
  onMount(() => {
    let alive = true;
    weights().then((m) => { if (alive) w = m; }, () => { if (alive) failed = true; });
    loadImpl(init.mode).then((m) => { if (alive) { impl = m; wake(); } }, () => { if (alive) failed = true; });
    const ctx = cv.getContext('2d')!;
    const read = () => {
      const c = readColors(TOKEN_NAMES);
      const rgb = (k: keyof typeof c): RGB => c[k].rgb;
      return { bg: rgb('bg'), text: rgb('text'), muted: rgb('muted'), rule: rgb('rule'), accent: rgb('accent'), pos: rgb('pos'), neg: rgb('neg'),
        seq: [rgb('s1'), rgb('s2'), rgb('s3'), rgb('s4'), rgb('s5')], cat: [rgb('c1'), rgb('c2'), rgb('c3'), rgb('c4'), rgb('c5')],
        ui: readVar('--font-ui'), code: readVar('--font-code') } as Pal;
    };
    let P = read(), lastUi = 0, sized = '';
    const paint = (f: Frame) => {
      const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1), sz = narrow ? slim : wide;
      const key = `${Math.round(r.width * dpr)}x${Math.round(r.height * dpr)}`;
      if (key !== sized) { sized = key; cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
      const k = (r.width / sz.W) * dpr;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      const E: Env = { c: ctx, P, W: sz.W, H: sz.H, narrow, hits: [], s: scene!, stats: sample?.heads };
      ctx.clearRect(0, 0, sz.W, sz.H); ctx.lineCap = 'butt'; ctx.lineJoin = 'round';
      impl!.draw(E, f); hits = E.hits;
    };
    const loop = createLoop(cv, (dt, now) => {
      if (!scene || !impl) return playing;
      if (!started) { started = true; t = reducedMotion() ? restTime(init.mode, scene, init.query) : sweeps ? startTime(init.mode, scene, init.query) : 0; }
      t = advance({ t: Math.min(t, tl.duration), playing: playing && pin === null, speed }, dt, { duration: tl.duration }).t;
      const f = impl.frameAt(t, scene, opts());
      paint(f);
      tlc?.paint(t);
      if (f.status !== status) status = f.status;
      if (f.mode === 'ablate' && f.off.some((x, k) => x !== offNow[k])) offNow = f.off;
      if (f.phase !== phase) { phase = f.phase; live = `${f.phase}: ${f.status}`; }
      if (!playing || now - lastUi > 100) { lastUi = now; uiT = t; }
      return playing && pin === null;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const off = onThemeChange(() => { P = read(); wake(); });
    document.fonts?.ready.then(() => { P = read(); wake(); });
    const motion = prefersReducedMotion(), onMotion = () => { if (motion.matches) playing = false; };
    motion.addEventListener('change', onMotion);
    return () => { alive = false; loop.destroy(); off(); motion.removeEventListener('change', onMotion); };
  });

  // The 200-sentence numbers are computed in paced slices (never one long task) once the figure is on screen.
  $effect(() => {
    const m = w, im = impl; if (!m || !im || (init.mode !== 'ablate' && init.mode !== 'gallery')) return;
    let ok = true;
    (init.mode === 'gallery' ? im.headStats!(m).then((heads) => ({ heads, mass: [] as number[] })) : im.sampleStats!(m, false)).then((r) => { if (ok) { sample = r; wake(); } });
    return () => { ok = false; };
  });
  $effect(() => {
    const m = w, im = impl, tg = [...toggles], on = custom;
    customSample = null;
    if (!m || !im?.sampleMass || !on || init.mode !== 'ablate') return;
    let ok = true;
    im.sampleMass(m, tg).then((v) => { if (ok) customSample = v; });
    return () => { ok = false; };
  });

  // ---- interaction ----
  function pointer(e: PointerEvent, click = false) {
    const r = cv.getBoundingClientRect(), k = size.W / r.width, x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
    let hit: Hit | undefined;
    for (let i = hits.length - 1; i >= 0; i--) { const h = hits[i]; if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) { hit = h; break; } }
    const txt = hit?.text ?? '';
    if (txt !== hoverText) hoverText = txt;
    const row = hit?.row ?? null;
    if (click) { if (row !== null && pinnable) clickPin = clickPin === row ? null : row; }
    else if (e.pointerType === 'mouse') hoverPin = pinnable ? row : null;
  }
  const leave = () => { hoverPin = null; hoverText = ''; };
  const togglePin = (i: number) => { clickPin = clickPin === i ? null : i; };

  function useWord(index: number, word: string) {
    const words = val.words.slice(1); words[index - 1] = word; text = words.join(' ');
  }
  function toggleHead(k: number) { custom = true; playing = false; toggles[k] = !toggles[k]; }
  const resetTour = () => { custom = false; toggles = [false, false, false, false]; seek(0); playing = !reducedMotion(); };

  // Marker labels are drawn in 1, 2 or 3 lanes, whatever it takes so neighbours never touch (track is about 420 px wide, 290 on a phone).
  const lanes = $derived.by(() => {
    const px = narrow ? 290 : 420, ms = tl.markers, wd = (i: number) => ms[i].label.length * 6.6 + 8;
    const left = (i: number) => (ms[i].at / tl.duration) * px - (ms[i].at / tl.duration > 0.8 ? wd(i) : 0);
    const clash = (k: number) => ms.some((_, i) => i >= k && left(i) < left(i - k) + wd(i - k) + 2);
    return clash(1) ? (clash(2) ? 3 : 2) : 1;
  });
  const label = $derived(`${init.title}. ${status}`);
  const sampleNow = $derived.by(() => {
    if (!sample || init.mode !== 'ablate' || !scene || !impl) return null;
    if (custom) return customSample;
    const f = impl.frameAt(uiT, scene, opts());
    return f.mode === 'ablate' && f.state >= 0 ? sample.mass[f.state] : null;
  });
  const KNOWN = VOCAB.filter((x) => x !== '[start]');
</script>

<div class="ax" class:dense={lanes === 2} class:dense3={lanes === 3}>
  <FigureCard title={init.title} {label} role="group" {legend}
    aspect="{wide.W} / {wide.H}" narrowAspect="{slim.W} / {slim.H}">
    <div class="stagebox" bind:clientWidth={width}>
      <canvas bind:this={cv} aria-hidden="true" onpointermove={(e) => pointer(e)} onpointerdown={(e) => pointer(e, true)} onpointerleave={leave} onpointercancel={leave}></canvas>
      {#if !scene || !impl}
        <p class="loading" role="status">{failed ? 'The model weights could not be loaded.' : 'Loading the trained model (15 KB)…'}</p>
      {/if}
    </div>
    {#snippet controls()}
      <Timeline bind:this={tlc} bind:t={uiT} bind:playing bind:speed duration={tl.duration} markers={tl.markers} label="Timeline: {init.title}" onSeek={(to) => { t = to; clickPin = null; wake(); }} onReset={() => { clickPin = null; seek(0); }} />
    {/snippet}
    {#snippet readout()}
      <p class="status" aria-hidden="true">{status}{#if hoverText}<br /><span class="hover">{hoverText}</span>{/if}</p>
      <p class="sr" role="status" aria-live="polite">{live}</p>
    {/snippet}
    {#snippet footer()}
      {#if scene && (sweeps || init.mode === 'explorer')}
        <div class="row" role="group" aria-label="Choose a word to follow">
          <span class="lab">Pause on a word</span>
          {#each scene.words as word, i (i)}
            <button type="button" class="chip" class:on={clickPin === i} aria-pressed={clickPin === i} onclick={() => togglePin(i)}>{word}</button>
          {/each}
        </div>
      {/if}
      {#if init.mode === 'explorer'}
        <div class="row">
          <label class="sel">Layer <select bind:value={selLayer}><option value={0}>1</option><option value={1}>2</option></select></label>
          <label class="sel">Head <select bind:value={selHead}><option value={0}>1</option><option value={1}>2</option></select></label>
          <span class="lab">Showing {headName(selLayer, selHead)}</span>
        </div>
      {/if}
      {#if init.mode === 'ablate' && scene}
        <div class="row" role="group" aria-label="Switch heads off">
          <span class="lab">Switch off</span>
          {#each HEADS as [l, h], k (k)}
            <label class="chk"><input type="checkbox" checked={custom ? toggles[k] : offNow[k]} onchange={() => { if (!custom) toggles = offNow.slice(); toggleHead(k); }} /> {headName(l, h)}</label>
          {/each}
          {#if custom}<button type="button" class="chip" onclick={resetTour}>Back to the tour</button>{/if}
        </div>
        <p class="note num">
          Over 200 sentences from the grammar:
          {#if sample}full model {sample.mass[0].toFixed(3)}{#if sampleNow !== null}, this setting {sampleNow.toFixed(3)}{/if}{:else}calculating&hellip;{/if}
        </p>
      {/if}
      {#if init.editable}
        <div class="field">
          <label for="s-{init.mode}-{width}">Sentence (words from the 34-word vocabulary)</label>
          <input id="s-{init.mode}-{width}" type="text" bind:value={text} spellcheck="false" autocomplete="off" autocapitalize="off" aria-describedby="m-{init.mode}-{width}" />
          <div id="m-{init.mode}-{width}" class="msg" aria-live="polite">
            {#each val.unknown as u (u.index)}
              <p>&ldquo;{u.word}&rdquo; is not in the vocabulary.
                {#if u.suggestions.length}Try{#each u.suggestions as sg (sg)}{' '}<button type="button" class="chip" onclick={() => useWord(u.index, sg)}>{sg}</button>{/each}{:else}See the list of 34 words below.{/if}
              </p>
            {/each}
            {#if val.tooLong}<p>The model reads at most 10 positions, including [start]. Shorten the sentence.</p>{/if}
            {#if !val.unknown.length && !val.tooLong && val.words.length < 2}<p>Type at least one word.</p>{/if}
            {#if val.ids && val.offGrammar}<p>The model still runs on this, but it only saw sentences like &ldquo;these small sheep chase a red fox&rdquo;. After &ldquo;{val.words[val.offGrammar.index - 1]}&rdquo;, the word &ldquo;{val.offGrammar.word}&rdquo; leaves that grammar.</p>{/if}
          </div>
          <details><summary>The 34 words</summary><p class="words num">{KNOWN.join(' · ')}</p></details>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .loading { position: absolute; inset: 0; display: grid; place-items: center; margin: 0; color: var(--text-muted); font: 0.85rem var(--font-ui); }
  .status { margin: 0; min-height: 3em; font-size: 0.95rem; line-height: 1.5; font-variant-numeric: tabular-nums; }
  .hover { color: var(--text-muted); }
  .sr { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.4rem; padding: var(--s-2) var(--s-4); }
  .row + .row, .row + .field, .row + .note { border-top: 1px solid var(--rule); }
  .lab { color: var(--text-muted); margin-right: 0.3rem; font: 0.8rem var(--font-ui); }
  .chip { font: 0.8rem var(--font-code); padding: 0.25em 0.55em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .chip.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
  .sel { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text-muted); font: 0.85rem var(--font-ui); }
  .sel select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.25em 0.4em; }
  .chk { display: inline-flex; align-items: center; gap: 0.35rem; font: 0.85rem var(--font-ui); cursor: pointer; }
  .note { margin: 0; padding: var(--s-2) var(--s-4); font: 0.82rem var(--font-ui); color: var(--text-muted); }
  .field { display: grid; gap: 0.4rem; padding: var(--s-3) var(--s-4); }
  .field label { color: var(--text-muted); font: 0.8rem var(--font-ui); }
  .field input { min-width: 0; width: 100%; box-sizing: border-box; padding: 0.55rem 0.6rem; font: 0.9rem var(--font-code); color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .msg { min-height: 1.4rem; font: 0.82rem / 1.4 var(--font-ui); color: var(--text-muted); }
  .msg p { margin: 0 0 0.2rem; }
  details { font: 0.8rem var(--font-ui); color: var(--text-muted); }
  .words { margin: 0.3rem 0 0; line-height: 1.6; }
  button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  /* Many markers: two staggered lanes under the scrubber so word labels never touch. */
  /* Word markers keep their own case (the shared Timeline would capitalise the first letter). */
  .ax :global(.tick::first-letter), .ax :global(.phn::first-letter) { text-transform: none; }
  .dense :global(.track) { padding-bottom: 2.7rem; }
  .dense :global(.ticks) { height: 2.6rem; }
  .dense :global(.tick:nth-child(even)) { top: 1.3rem; }
  .dense3 :global(.track) { padding-bottom: 3.8rem; }
  .dense3 :global(.ticks) { height: 3.8rem; }
  .dense3 :global(.tick:nth-child(3n + 1)) { top: 0; }
  .dense3 :global(.tick:nth-child(3n + 2)) { top: 1.25rem; }
  .dense3 :global(.tick:nth-child(3n + 3)) { top: 2.5rem; }
</style>
