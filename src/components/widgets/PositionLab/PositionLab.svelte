<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { loadModel } from '../../../lib/transformer/load';
  import type { Weights } from '../../../lib/transformer/model';
  import {
    CLASS_NAMES, D, DIAGONALS, REL_N, ROPE_K, ROPE_Q, ROPE_THETA, SIN_POSITIONS, TRACE_DIMS, UNTRAINED_ROW, WORDS,
    ablateFrameAt, samplePart, diagonal, durationOf, learnedFrameAt, makeAblate, makeLearned, makeRelative, makeShuffle, makeSin, markersOf,
    relFrameAt, ropeFrameAt, ropeRotate, shuffleFrameAt, sinFrameAt, sinPE, wavelength, type Mode,
  } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'shuffle', title = 'How a transformer sees word order' }: Props = $props();
  const init = untrack(() => mode);
  const needsModel = init === 'shuffle' || init === 'ablate' || init === 'learned';

  let width = $state(0), playing = $state(false), speed = $state(1), uiT = $state(0);
  let weights = $state.raw<Weights | null>(null), failed = $state(false);
  let seed = $state(0);
  let manual = $state<{ m: number; n: number } | null>(null);
  let hover = $state<{ map: 'additive' | 'rope'; m: number; n: number } | null>(null);
  const narrow = $derived(width > 0 && width < 540);
  const duration = durationOf(init), markers = markersOf(init);

  // ---- geometry per mode (viewBox units); the card reserves the tallest state so nothing shifts -------------
  function layout(narrow: boolean) {
    const W = narrow ? 288 : 720;
    switch (init) {
      case 'shuffle': {
        const rh = narrow ? 26 : 30, top = 30, gap = narrow ? 24 : 90, pw = (W - gap) / 2, lw = narrow ? 66 : 92;
        return { W, H: top + 8 * rh + 8, rh, top, gap, pw, lw, cw: (pw - lw - 4) / 16 };
      }
      case 'ablate': {
        const rh = narrow ? 27 : 30, top = 44, lw = narrow ? 66 : 112, cw = narrow ? 4 : 9, barX = lw + 16 * cw + (narrow ? 10 : 24);
        return { W, H: top + 8 * rh + 12, rh, top, lw, cw, barX, barW: W - barX - (narrow ? 46 : 56) };
      }
      case 'sinusoidal': {
        const gl = narrow ? 72 : 110, pw = W - gl - 8, th = narrow ? 34 : 42, hc = narrow ? 8 : 11;
        const heatTop = 6 + 3 * (th + 6) + 6, heatBottom = heatTop + 16 * hc, rs = narrow ? 22 : 24, stripTop = heatBottom + 24;
        return { W, H: stripTop + 3 * rs + 6, gl, pw, col: pw / SIN_POSITIONS, th, hc, heatTop, heatBottom, rs, stripTop, sw: pw / D };
      }
      case 'learned': {
        const rowH = narrow ? 17 : 32, top = 34, labw = narrow ? 14 : 22, cw = narrow ? 4.5 : 14, mx = labw + 16 * cw + (narrow ? 12 : 60);
        return { W, H: top + 10 * rowH + 8, rowH, top, labw, cw, mx };
      }
      case 'rope': {
        const S = narrow ? 240 : 300;
        return narrow ? { W, H: 2 * (S + 40) + 4, S, x0: 24, y0: 28, x1: 24, y1: S + 40 + 28 }
          : { W, H: S + 52, S, x0: 30, y0: 28, x1: 390, y1: 28 };
      }
      case 'relative': {
        const cs = narrow ? 14 : 19, gl = 34, size = REL_N * cs;
        return narrow ? { W, H: 2 * (size + 62) + 6, cs, gl, size, x0: gl, y0: 28, x1: gl, y1: size + 62 + 28 }
          : { W, H: size + 62, cs, gl, size, x0: gl, y0: 28, x1: gl + size + 46, y1: 28 };
      }
    }
  }
  const Lw = layout(false), Ln = layout(true);
  const L = $derived((narrow ? Ln : Lw) as Record<string, number>);

  // ---- data -------------------------------------------------------------------------------------------------
  const sinData = init === 'sinusoidal' ? makeSin() : null;
  const relData = init === 'relative' ? makeRelative() : null;
  const shuf = $derived(weights && init === 'shuffle' ? makeShuffle(weights, seed) : null);
  const abl = $derived(weights && init === 'ablate' ? makeAblate(weights, false) : null);
  // The 200-sentence averages are spread over small tasks so the page never stalls.
  let sample = $state<{ with: number; without: number } | null>(null);
  const learned = $derived(weights && init === 'learned' ? makeLearned(weights) : null);
  const isReady = $derived(!needsModel || !!(shuf || abl || learned));
  const MAPS = ['additive', 'rope'] as const;

  const fSh = $derived(shuf ? shuffleFrameAt(uiT, shuf) : null);
  const fAb = $derived(abl ? ablateFrameAt(uiT, abl) : null);
  const fSi = $derived(sinData ? sinFrameAt(uiT, sinData) : null);
  const fLe = $derived(learned ? learnedFrameAt(uiT, learned) : null);
  const fRo = $derived(init === 'rope' ? ropeFrameAt(uiT, manual) : null);
  const fRe = $derived(relData ? relFrameAt(uiT, relData) : null);
  const status = $derived((fSh ?? fAb ?? fSi ?? fLe ?? fRo ?? fRe)?.status ?? 'Loading the small model.');

  // ---- colour helpers: tokens only; blue = positive, orange = negative -----------------------------------------------
  const sgn = (v: number, max: number) =>
    `color-mix(in srgb, var(${v >= 0 ? '--viz-pos' : '--viz-neg'}) ${Math.round(Math.min(1, Math.abs(v) / max) * 100)}%, var(--bg-raised))`;
  const cat = (cls: number) => `var(--viz-cat-${cls + 1})`;
  const f2 = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
  const signed = (v: number, digits = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(digits);

  // ---- static heat cells -----------------------------------------------------------------------------------------------
  const sinCells = $derived.by(() => {
    if (!sinData) return [];
    const l = L, out: { x: number; y: number; fill: string }[] = [];
    for (let dim = 0; dim < D; dim++) for (let p = 0; p < SIN_POSITIONS; p++) out.push({ x: l.gl + p * l.col, y: l.heatTop + dim * l.hc, fill: sgn(sinData.table[p][dim], 1) });
    return out;
  });
  const traces = $derived.by(() => {
    if (!sinData) return [];
    const l = L;
    return TRACE_DIMS.map((dim, i) => {
      const y0 = 6 + i * (l.th + 6), pair = dim / 2;
      const yOf = (v: number) => y0 + l.th / 2 - v * (l.th / 2 - 3);
      const pts = Array.from({ length: SIN_POSITIONS * 4 - 3 }, (_, k) => { const p = k / 4; return `${l.gl + (p + 0.5) * l.col},${yOf(Math.sin(p / 10000 ** (2 * pair / D)))}`; });
      return { dim, y0, yOf, path: 'M' + pts.join(' L'), pair, wl: wavelength(pair) };
    });
  });
  const relCells = $derived.by(() => {
    if (!relData) return [];
    const out: { map: 'additive' | 'rope'; m: number; n: number; fill: string }[] = [];
    for (const [map, grid, mean, dev] of [['additive', relData.additive, relData.meanAdd, relData.devAdd], ['rope', relData.rope, relData.meanRope, relData.devRope]] as const)
      for (let m = 0; m < REL_N; m++) for (let n = 0; n < REL_N; n++) out.push({ map, m, n, fill: sgn(grid[m][n] - mean, dev) });
    return out;
  });
  const relX = (map: 'additive' | 'rope') => (map === 'additive' ? L.x0 : L.x1);
  const relY = (map: 'additive' | 'rope') => (map === 'additive' ? L.y0 : L.y1);
  const hoverInfo = $derived.by(() => {
    if (!hover || !relData) return null;
    const grid = hover.map === 'rope' ? relData.rope : relData.additive;
    const next = hover.m + 1 < REL_N && hover.n + 1 < REL_N ? grid[hover.m + 1][hover.n + 1] : null;
    return { value: grid[hover.m][hover.n], next };
  });

  // ---- vector arrows for rope -----------------------------------------------------------------------------------------
  const arrow = (cx: number, cy: number, s: number, v: readonly number[]) => {
    const X = cx + v[0] * s, Y = cy - v[1] * s, len = Math.hypot(X - cx, Y - cy) || 1, ux = (X - cx) / len, uy = (Y - cy) / len;
    const bx = X - 9 * ux, by = Y - 9 * uy;
    return { X, Y, head: `${X},${Y} ${bx - 4.5 * uy},${by + 4.5 * ux} ${bx + 4.5 * uy},${by - 4.5 * ux}` };
  };
  const arc = (cx: number, cy: number, s: number, a: readonly number[], b: readonly number[], r = 0.4) => {
    const A = Math.atan2(a[1], a[0]);
    let d = Math.atan2(b[1], b[0]) - A;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    return 'M' + Array.from({ length: 13 }, (_, i) => { const ang = A + (d * i) / 12; return `${cx + Math.cos(ang) * r * s},${cy - Math.sin(ang) * r * s}`; }).join(' L');
  };
  const angleBetween = (a: readonly number[], b: readonly number[]) => {
    const d = Math.atan2(b[1], b[0]) - Math.atan2(a[1], a[0]);
    return Math.atan2(Math.sin(d), Math.cos(d));
  };

  // ---- clock ------------------------------------------------------------------------------------------------------
  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline>;
  let t = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; isReady; wake(); });
  $effect(() => { if (playing) manual = null; });
  const seek = (to: number) => { t = to; uiT = to; manual = null; wake(); };
  const shuffleAgain = () => { seed++; t = 0; uiT = 0; playing = !reducedMotion(); wake(); };
  const setRope = (which: 'm' | 'n', value: number) => {
    const now = fRo!;
    manual = { m: which === 'm' ? value : now.mShown, n: which === 'n' ? value : now.nShown };
    playing = false; wake();
  };

  onMount(() => {
    let dead = false;
    if (needsModel) loadModel('tiny-lm').then((w) => {
      weights = w;
      if (init !== 'ablate') return;
      const acc = [{ total: 0, count: 0 }, { total: 0, count: 0 }];
      const step = (from: number) => {
        if (dead) return;
        if (from >= 200) { sample = { with: acc[0].total / acc[0].count, without: acc[1].total / acc[1].count }; return; }
        [1, 0].forEach((scale, k) => { const p = samplePart(w, scale, from, from + 10); acc[k].total += p.total; acc[k].count += p.count; });
        setTimeout(() => step(from + 10), 0);
      };
      setTimeout(() => step(0), 0);
    }, () => { failed = true; });
    let lastPaint = 0;
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      if (!isReady) return playing;
      t = advance({ t, playing, speed }, dt, { duration }).t;
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { dead = true; loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  const legend = init === 'shuffle' ? [
    ...CLASS_NAMES.map((name, i) => ({ label: name, color: `--viz-cat-${i + 1}` })),
  ] : init === 'rope' ? [{ label: 'query', color: '--viz-cat-1', shape: 'line' as const }, { label: 'key', color: '--viz-cat-2', shape: 'line' as const }]
    : init === 'ablate' ? [{ label: 'probability now', color: '--viz-seq-4' }, { label: 'positive number', color: '--viz-pos' }, { label: 'negative number', color: '--viz-neg' }]
    : init === 'relative' ? [{ label: 'above this map’s average', color: '--viz-pos' }, { label: 'below it', color: '--viz-neg' }, { label: 'current diagonal', color: '--accent', shape: 'line' as const }]
    : [{ label: 'positive number', color: '--viz-pos' }, { label: 'negative number', color: '--viz-neg' }];
  const label = {
    shuffle: 'Two lists of the same eight words, one in order and one shuffled, with the attention output for each word',
    ablate: 'Bars showing, for each word position, how much probability the model puts on grammatical next words as position vectors fade out',
    sinusoidal: 'Sine and cosine position patterns for positions 0 to 31, with three wave traces and a cursor walking through the positions',
    learned: 'The ten learned position vectors and a ten by ten grid of how similar they are',
    rope: 'Two planes showing a query arrow and a key arrow rotated by their positions; the angle between them depends only on the gap',
    relative: 'Two grids of query-key scores over positions, one with added position vectors and one with rotary encoding',
  }[init];
</script>

<div class="lab">
  <FigureCard {title} {label} {legend} aspect="{Lw.W} / {Lw.H}" narrowAspect="{Ln.W} / {Ln.H}">
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      {#if needsModel && !isReady}
        <p class="loading">{failed ? 'The small model could not be loaded. Reload the page to try again.' : 'Loading the small model…'}</p>
      {:else}
        <svg viewBox="0 0 {L.W} {L.H}" aria-hidden="true" role="presentation" onpointerleave={() => (hover = null)}>
          {#if fSh}
            <text x="0" y="18" class="heading">Original order</text>
            <text x={L.pw + L.gap} y="18" class="heading">Shuffled order</text>
            {#each fSh.rows as r (r.i)}
              {@const yl = L.top + r.yL * L.rh}
              {@const yr = L.top + r.yR * L.rh}
              <line x1={L.pw} x2={L.pw + L.gap} y1={yl + L.rh / 2 - 3} y2={yr + L.rh / 2 - 3} stroke={cat(r.cls)} stroke-width="1.5" opacity="0.55" />
              {#each [{ x: 0, y: yl, v: r.left }, { x: L.pw + L.gap, y: yr, v: r.right }] as side, k (k)}
                <rect x={side.x} y={side.y} width={L.lw - 4} height={L.rh - 6} class="chip" />
                <rect x={side.x} y={side.y} width="4" height={L.rh - 6} fill={cat(r.cls)} />
                <text x={side.x + 9} y={side.y + (L.rh - 6) / 2 + 4} class="word">{r.word}</text>
                {#each side.v as v, j (j)}
                  <rect x={side.x + L.lw + j * L.cw} y={side.y} width={L.cw + 0.4} height={L.rh - 6} style:fill={sgn(v, 1.7)} />
                {/each}
              {/each}
            {/each}
          {:else if fAb}
            <text x="0" y="22" class="small">word</text>
            <text x={L.lw} y="22" class="small">position</text>
            <text x={L.lw} y="34" class="small">vector</text>
            <text x={L.barX} y="22" class="small">probability on a</text>
            <text x={L.barX} y="34" class="small">grammatical next word</text>
            {#each WORDS as word, p (p)}
              {@const y = L.top + p * L.rh}
              <text x="0" y={y + L.rh / 2 + 1} class="word"><tspan class="idx">{p}</tspan> {word}</text>
              {#each fAb.strips[p] as v, j (j)}
                <rect x={L.lw + j * L.cw} y={y + 3} width={L.cw + 0.3} height={L.rh - 12} style:fill={sgn(v, 0.45)} />
              {/each}
              <rect x={L.barX} y={y + 3} width={L.barW} height={L.rh - 12} class="track" />
              <rect x={L.barX} y={y + 3} width={Math.max(1, fAb.mass[p] * L.barW)} height={L.rh - 12} class="bar" />
              <line x1={L.barX + fAb.withPos[p] * L.barW} x2={L.barX + fAb.withPos[p] * L.barW} y1={y} y2={y + L.rh - 6} class="ghost" />
              <text x={L.barX + L.barW + 6} y={y + L.rh / 2 + 1} class="small num">{fAb.mass[p].toFixed(3)}</text>
            {/each}
          {:else if fSi}
            {#each traces as tr (tr.dim)}
              <rect x={L.gl} y={tr.y0} width={L.pw} height={L.th} class="plot" />
              <line x1={L.gl} x2={L.gl + L.pw} y1={tr.yOf(0)} y2={tr.yOf(0)} class="axis" />
              <path d={tr.path} class="curve" />
              <circle cx={L.gl + (fSi.p + 0.5) * L.col} cy={tr.yOf(Math.sin(fSi.p / 10000 ** (2 * tr.pair / D)))} r="4" class="dot" />
              <text x={L.gl - 6} y={tr.y0 + 14} text-anchor="end" class="small">dim {tr.dim}</text>
              <text x={L.gl - 6} y={tr.y0 + 26} text-anchor="end" class="tiny">period {tr.wl < 100 ? tr.wl.toFixed(1) : Math.round(tr.wl).toLocaleString('en-US')}</text>
            {/each}
            {#each sinCells as c, i (i)}
              <rect x={c.x} y={c.y} width={L.col + 0.3} height={L.hc + 0.3} style:fill={c.fill} />
            {/each}
            {#each narrow ? [0, 4, 8, 12, 15] : [0, 2, 4, 6, 8, 10, 12, 15] as dim (dim)}
              <text x={L.gl - 6} y={L.heatTop + dim * L.hc + L.hc - 1} text-anchor="end" class="tiny">dim {dim}</text>
            {/each}
            {#each [0, 8, 16, 24, 31] as p (p)}
              <text x={L.gl + (p + 0.5) * L.col} y={L.heatBottom + 13} text-anchor="middle" class="tiny">{p}</text>
            {/each}
            <text x={L.gl - 6} y={L.heatBottom + 13} text-anchor="end" class="tiny">position</text>
            <line x1={L.gl + (fSi.p + 0.5) * L.col} x2={L.gl + (fSi.p + 0.5) * L.col} y1="4" y2={L.heatBottom} class="cursor" />
            <rect x={L.gl + fSi.index * L.col} y={L.heatTop} width={L.col} height={16 * L.hc} class="outline" />
            {#each [{ v: sinData!.word, name: narrow ? 'word' : 'word vector' }, { v: fSi.pe, name: narrow ? `+ pos. ${fSi.index}` : `+ position ${fSi.index}` }, { v: fSi.sum, name: '= sum' }] as row, k (k)}
              <text x={L.gl - 6} y={L.stripTop + k * L.rs + 12} text-anchor="end" class="small">{row.name}</text>
              {#each row.v as v, j (j)}
                <rect x={L.gl + j * L.sw} y={L.stripTop + k * L.rs} width={L.sw - 1} height={L.rs - 5} style:fill={sgn(v, 1.6)} />
              {/each}
            {/each}
          {:else if fLe && learned}
            <text x="0" y="18" class="heading">{narrow ? 'Vectors' : 'Position vectors'}</text>
            <text x={L.mx} y="18" class="heading">{narrow ? 'Similarity' : 'Similarity (cosine)'}</text>
            {#each Array.from({ length: 10 }, (_, c) => c) as c (c)}
              <text x={L.mx + (c + 0.5) * L.rowH} y={L.top - 4} text-anchor="middle" class="tiny" opacity={c === UNTRAINED_ROW ? 0.45 : 1}>{c}</text>
            {/each}
            {#each learned.strips as strip, r (r)}
              {@const y = L.top + r * L.rowH}
              {@const grey = r === UNTRAINED_ROW}
              <g opacity={grey ? 0.35 : 1}>
                <text x={L.labw - 4} y={y + L.rowH / 2 + 4} text-anchor="end" class="small">{r}</text>
                {#each strip as v, j (j)}
                  <rect x={L.labw + j * L.cw} y={y + 1} width={L.cw + 0.3} height={L.rowH - 2} style:fill={sgn(v, 0.45)} />
                {/each}
                {#each learned.cos[r] as v, c (c)}
                  <g opacity={(c === UNTRAINED_ROW ? 0.45 : 1) * fLe.reveal[r]}>
                    <rect x={L.mx + c * L.rowH} y={y + 0.5} width={L.rowH - 1} height={L.rowH - 1} style:fill={sgn(v, 1)} />
                    {#if !narrow}<text x={L.mx + (c + 0.5) * L.rowH} y={y + L.rowH / 2 + 3.5} text-anchor="middle" class="cellnum">{r === c ? '1' : f2(v).replace('-', '−')}</text>{/if}
                  </g>
                {/each}
              </g>
            {/each}
            {#if fLe.row >= 0}
              <rect x={L.labw - 1} y={L.top + fLe.row * L.rowH} width={16 * L.cw + 2} height={L.rowH} class="outline" />
              <rect x={L.mx - 1} y={L.top + fLe.row * L.rowH} width={10 * L.rowH + 1} height={L.rowH} class="outline" />
            {/if}
          {:else if fRo}
            {#each [0, 1] as k (k)}
              {@const px = k ? L.x1 : L.x0}
              {@const py = k ? L.y1 : L.y0}
              {@const s = L.S / 3}
              {@const cx = px + L.S / 2}
              {@const cy = py + L.S / 2}
              {@const q = k ? ROPE_Q : fRo.q}
              {@const kk = k ? ropeRotate(ROPE_K, fRo.n - fRo.m) : fRo.k}
              {@const aq = arrow(cx, cy, s, q)}
              {@const ak = arrow(cx, cy, s, kk)}
              <text x={px} y={py - 10} class="heading">{k ? 'Both turned back by m steps' : 'Arrows at positions m and n'}</text>
              <rect x={px} y={py} width={L.S} height={L.S} class="plot" />
              <line x1={px} x2={px + L.S} y1={cy} y2={cy} class="axis" />
              <line x1={cx} x2={cx} y1={py} y2={py + L.S} class="axis" />
              <circle {cx} {cy} r={Math.hypot(ROPE_Q[0], ROPE_Q[1]) * s} class="guide" />
              <circle {cx} {cy} r={Math.hypot(ROPE_K[0], ROPE_K[1]) * s} class="guide" />
              {#if !k}
                <line x1={cx} y1={cy} x2={cx + ROPE_Q[0] * s} y2={cy - ROPE_Q[1] * s} class="start" stroke="var(--viz-cat-1)" />
                <line x1={cx} y1={cy} x2={cx + ROPE_K[0] * s} y2={cy - ROPE_K[1] * s} class="start" stroke="var(--viz-cat-2)" />
              {/if}
              <path d={arc(cx, cy, s, q, kk)} class="arc" />
              <line x1={cx} y1={cy} x2={aq.X} y2={aq.Y} stroke="var(--viz-cat-1)" stroke-width="3" />
              <polygon points={aq.head} fill="var(--viz-cat-1)" />
              <line x1={cx} y1={cy} x2={ak.X} y2={ak.Y} stroke="var(--viz-cat-2)" stroke-width="3" />
              <polygon points={ak.head} fill="var(--viz-cat-2)" />
              <text x={aq.X + (aq.X >= cx ? 6 : -6)} y={aq.Y - 6} text-anchor={aq.X >= cx ? 'start' : 'end'} class="small">q</text>
              <text x={ak.X + (ak.X >= cx ? 6 : -6)} y={ak.Y + 14} text-anchor={ak.X >= cx ? 'start' : 'end'} class="small">k</text>
              {#if k}<text x={px + L.S - 8} y={py + 18} text-anchor="end" class="heading num">q·k = {signed(fRo.score)}</text>{/if}
            {/each}
          {:else if fRe && relData}
            {#each MAPS as map (map)}
              <text x={relX(map)} y={relY(map) - 10} class="heading">{map === 'rope' ? 'RoPE rotation' : 'Added sinusoidal vectors'}</text>
              <text transform="translate({relX(map) - 24} {relY(map) + L.size / 2}) rotate(-90)" text-anchor="middle" class="tiny">query position m</text>
              {#each [0, 5, 10, 15] as tick (tick)}
                <text x={relX(map) + (tick + 0.5) * L.cs} y={relY(map) + L.size + 12} text-anchor="middle" class="tiny">{tick}</text>
                <text x={relX(map) - 6} y={relY(map) + (tick + 0.5) * L.cs + 3} text-anchor="end" class="tiny">{tick}</text>
              {/each}
              <text x={relX(map) + L.size / 2} y={relY(map) + L.size + 26} text-anchor="middle" class="tiny">key position n</text>
            {/each}
            {#each relCells as c, i (i)}
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <rect x={relX(c.map) + c.n * L.cs} y={relY(c.map) + c.m * L.cs} width={L.cs} height={L.cs} style:fill={c.fill} onpointerenter={() => (hover = { map: c.map, m: c.m, n: c.n })} onpointerdown={() => (hover = { map: c.map, m: c.m, n: c.n })} />
            {/each}
            {#each MAPS as map (map)}
              {#each Array.from({ length: REL_N }, (_, m) => m).filter((m) => m + fRe.c >= 0 && m + fRe.c < REL_N) as m (m)}
                <rect x={relX(map) + (m + fRe.c) * L.cs} y={relY(map) + m * L.cs} width={L.cs} height={L.cs} class="diag" />
              {/each}
            {/each}
            {#if hover}
              <rect x={relX(hover.map) + hover.n * L.cs} y={relY(hover.map) + hover.m * L.cs} width={L.cs} height={L.cs} class="pick" />
              {#if hover.m + 1 < REL_N && hover.n + 1 < REL_N}
                <rect x={relX(hover.map) + (hover.n + 1) * L.cs} y={relY(hover.map) + (hover.m + 1) * L.cs} width={L.cs} height={L.cs} class="pick next" />
              {/if}
            {/if}
          {/if}
        </svg>
      {/if}
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} label={init === 'rope' ? 'Timeline, query position m' : 'Position timeline'} onSeek={seek} onReset={() => seek(0)} />
    {/snippet}
    {#snippet readout()}
      <p class="status" role="status" aria-live="polite">{status}</p>
      {#if fSh}
        <p class="line">Each strip is 16 numbers: blue positive, orange negative.</p>
        <p class="line">Largest difference between the two sets of output strips, once the words are put back in order: <span class="num">{fSh.maxDiff.toExponential(1)}</span>, which is rounding error, so zero.</p>
      {:else if fAb}
        <p class="line">Position vectors at <span class="num">{Math.round(fAb.scale * 100)}%</span> of normal size. This sentence, average over its 8 positions: <span class="num">{fAb.average.toFixed(3)}</span>. Across 200 sample sentences: {#if sample}<span class="num">{sample.with.toFixed(3)}</span> with positions, <span class="num">{sample.without.toFixed(3)}</span> without.{:else}working it out&hellip;{/if}</p>
      {:else if fSi}
        <p class="line">Position {fSi.index}: <span class="num">dim 0 = {signed(fSi.pe[0])}, dim 2 = {signed(fSi.pe[2])}, dim 4 = {signed(fSi.pe[4])}, dim 14 = {signed(fSi.pe[14], 3)}</span></p>
      {:else if fLe}
        <p class="line">
          {#if fLe.row < 0}Every row filled in. Position 9 is greyed: it never appears as an input while the model trains, so it still holds its random starting values.
          {:else if fLe.row === UNTRAINED_ROW}Position 9 is greyed: it never appears as an input while the model trains, so it still holds its random starting values.
          {:else}Position {fLe.row}: average similarity to its immediate neighbours <span class="num">{signed(fLe.neighbour)}</span>, to positions 3 or more away <span class="num">{signed(fLe.far)}</span> (position 9 left out).{/if}
        </p>
      {:else if fRo}
        <p class="line"><span class="num">m = {fRo.mShown}, n = {fRo.nShown}</span>: the query is turned <span class="num">{signed(fRo.mShown * ROPE_THETA, 1)}</span> radians and the key <span class="num">{signed(fRo.nShown * ROPE_THETA, 1)}</span>, so the gap between them is <span class="num">{signed((fRo.nShown - fRo.mShown) * ROPE_THETA, 1)}</span> radians. Dot product: <span class="num">{signed(fRo.score)}</span>.</p>
      {:else if fRe}
        <p class="line">
          <span class="num">{fRe.length}</span> cells on this diagonal. With RoPE their scores are identical (to rounding); with added vectors they differ by up to <span class="num">{f2(fRe.addSpread)}</span>.
          {#if hover && hoverInfo}Cell <span class="num">(m {hover.m}, n {hover.n}) = {signed(hoverInfo.value)}</span>{#if hoverInfo.next !== null}; one step down its diagonal <span class="num">{signed(hoverInfo.next)}</span>{/if}.{/if}
        </p>
      {/if}
    {/snippet}
    {#snippet footer()}
      {#if init === 'shuffle'}
        <div class="row"><button type="button" class="btn" onclick={shuffleAgain}>Shuffle again</button></div>
      {:else if init === 'rope' && fRo}
        <div class="sliders">
          <label>Query position m <input type="range" min="0" max="31" step="1" value={fRo.mShown} oninput={(e) => setRope('m', +e.currentTarget.value)} /><output class="num">{fRo.mShown}</output></label>
          <label>Key position n <input type="range" min="0" max="31" step="1" value={fRo.nShown} oninput={(e) => setRope('n', +e.currentTarget.value)} /><output class="num">{fRo.nShown}</output></label>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .loading { margin: 0; position: absolute; inset: 0; display: grid; place-items: center; color: var(--text-muted); font: 0.9rem var(--font-ui); text-align: center; padding: var(--s-4); }
  .heading { fill: var(--text); font: 600 14px var(--font-ui); }
  .small { fill: var(--text-muted); font: 12px var(--font-ui); }
  .tiny { fill: var(--text-muted); font: 10px var(--font-ui); }
  .word { fill: var(--text); font: 12px var(--font-code); }
  .idx { fill: var(--text-muted); }
  .num { font-variant-numeric: tabular-nums; font-family: var(--font-code); font-size: 0.92em; }
  .cellnum { fill: var(--text); font: 9.5px var(--font-code); }
  .chip { fill: var(--bg-sunken); stroke: var(--rule); }
  .track { fill: var(--bg-sunken); }
  .bar { fill: var(--viz-seq-4); }
  .ghost { stroke: var(--text); stroke-width: 1.5; stroke-dasharray: 2 2; }
  .plot { fill: var(--bg-sunken); stroke: var(--rule); }
  .axis { stroke: var(--rule); }
  .curve { fill: none; stroke: var(--viz-forward); stroke-width: 2; }
  .dot { fill: var(--accent); }
  .cursor { stroke: var(--accent); stroke-width: 1.5; }
  .outline { fill: none; stroke: var(--accent); stroke-width: 2; }
  .guide { fill: none; stroke: var(--rule); stroke-dasharray: 3 3; }
  .start { stroke-width: 1.5; stroke-dasharray: 4 3; opacity: 0.5; }
  .arc { fill: none; stroke: var(--text-muted); stroke-width: 1.5; }
  .diag { fill: none; stroke: var(--accent); stroke-width: 1.5; pointer-events: none; }
  .pick { fill: none; stroke: var(--text); stroke-width: 2; pointer-events: none; }
  .pick.next { stroke-dasharray: 3 2; }
  .status { margin: 0; min-height: 2.6em; }
  .line { margin: 0.25em 0 0; }
  .row, .sliders { padding: var(--s-3) var(--s-4); display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-5); align-items: center; }
  .sliders label { display: flex; align-items: center; gap: var(--s-2); }
  .sliders input { width: 9rem; max-width: 40vw; }
  .sliders output { min-width: 2ch; }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
