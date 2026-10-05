<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { loadModel } from '../../../lib/transformer/load';
  import {
    DEFAULT_SENTENCE, NORM_POST, NORM_PRE, PRESETS, buildData, cellStyle, clamp01, fmtN, fmtP, frameAt, maxAbs, norm2, phaseOf, rowOf, timelineOf,
    type Data, type Mode,
  } from './sim';

  interface Props { mode?: Mode; sentence?: string; layer?: number; query?: string; title?: string }
  let { mode = 'overview', sentence = DEFAULT_SENTENCE, layer = 1, query = 'sheep', title = 'One transformer block' }: Props = $props();
  const init = untrack(() => mode), L1 = untrack(() => layer), qWord = untrack(() => query), sent = untrack(() => sentence);
  const needsModel = init !== 'norm' && init !== 'count';

  let width = $state(0), playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let data = $state<Data | null>(null), failed = $state(false);
  let custom = $state<{ d: number; L: number } | null>(null);
  const narrow = $derived(width > 0 && width < 540);
  const W = $derived(narrow ? 288 : 720);
  const tl = $derived(timelineOf(init, !!custom));
  const frame = $derived(frameAt(uiT, init, data, { layer: L1, custom }));
  const ready = $derived(!needsModel || !!data);

  // [wide height, narrow height] reserved per mode, so playback never moves the page.
  const HEIGHTS: Record<Mode, [number, number]> = { overview: [420, 446], residual: [304, 330], mlp: [310, 262], norm: [340, 450], stream: [350, 360], count: [340, 540] };
  const H = $derived(HEIGHTS[init][narrow ? 1 : 0]);
  const legend = init === 'norm' ? [{ label: 'Pre-norm', color: '--viz-cat-1' }, { label: 'Post-norm', color: '--viz-cat-2' }]
    : init === 'stream' ? [{ label: 'P(chase), plural', color: '--viz-cat-1' }, { label: 'P(chases), singular', color: '--viz-cat-2' }]
    : init === 'residual' ? [{ label: 'Positive', color: '--viz-pos' }, { label: 'Negative', color: '--viz-neg' }, { label: 'Length of x', color: '--viz-seq-4' }, { label: 'Length of update', color: '--viz-seq-2' }]
    : init === 'mlp' ? [{ label: 'Positive (kept)', color: '--viz-pos' }, { label: 'Negative (set to 0)', color: '--viz-neg' }]
    : init === 'overview' ? [{ label: 'A word’s vector', color: '--viz-forward' }, { label: 'Attention weight (mean of 2 heads)', color: '--viz-seq-4', shape: 'line' as const }]
    : [{ label: 'Embeddings', color: '--viz-cat-1' }, { label: 'Attention', color: '--viz-cat-2' }, { label: 'MLP', color: '--viz-cat-3' }, { label: 'Norms and output bias', color: '--viz-cat-4' }];

  let stage: HTMLDivElement, tlc: ReturnType<typeof Timeline>;
  let t = 0, lastPaint = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; data; custom; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  const jump = (at: number) => { playing = false; seek(at); };

  onMount(() => {
    if (needsModel) loadModel('tiny-lm').then((w) => { data = buildData(w, sent, qWord); }).catch(() => { failed = true; });
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: tl.duration }).t;
      if (!playing || now - lastPaint >= 32) { uiT = t; tlc?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  // ---------- shared geometry helpers ----------
  const PAD = 12;
  const f2 = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

  // ---------- overview ----------
  const ov = $derived.by(() => {
    const T = data?.T ?? 8, colW = (W - 2 * PAD) / T, labelH = narrow ? 56 : 34;
    const embB = H - labelH - 4, embT = embB - 26;
    const a1b = embT - 14, a1t = a1b - 70, m1b = a1t - 12, m1t = m1b - 38;
    const a2b = m1t - 36, a2t = a2b - 70, m2b = a2t - 12, m2t = m2b - 38;
    const outB = m2t - 24, outT = outB - 28;
    return { T, colW, labelH, embB, embT, a1b, a1t, m1b, m1t, a2b, a2t, m2b, m2t, outB, outT, colX: (i: number) => PAD + colW * (i + 0.5),
      ys: [H - labelH - 2, a1b, a1t, m1t, a2t, m2t, outB], bands: [{ b: a1b, t: a1t, p: 1 }, { b: a2b, t: a2t, p: 3 }], mlps: [{ b: m1b, t: m1t, p: 2 }, { b: m2b, t: m2t, p: 4 }] };
  });
  const bump = (s: number, p: number) => clamp01(1.2 - 1.6 * Math.abs(s - (p + 0.5)));
  const dotY = (s: number) => { const p = Math.min(5, Math.floor(s)); return lerp(ov.ys[p], ov.ys[p + 1], s - p); };
  interface Arc { j: number; i: number; w: number }
  const arcsOf = (l: number): Arc[] => {
    if (!data) return [];
    const out: Arc[] = [], A = data.blocks[l].arcs, T = data.T;
    for (let i = 1; i < T; i++) for (let j = 0; j < i; j++) if (A[i * T + j] > 0.05) out.push({ j, i, w: A[i * T + j] });
    return out;
  };
  const arcH = (a: Arc) => Math.min(60, 14 + 12 * (a.i - a.j));
  const arcPath = (a: Arc, base: number) => `M${ov.colX(a.j)},${base} Q${(ov.colX(a.j) + ov.colX(a.i)) / 2},${base - arcH(a)} ${ov.colX(a.i)},${base}`;
  const arcPoint = (a: Arc, base: number, u: number) => {
    const x0 = ov.colX(a.j), x1 = ov.colX(a.i), cx = (x0 + x1) / 2, cy = base - arcH(a);
    return { x: (1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, y: (1 - u) ** 2 * base + 2 * (1 - u) * u * cy + u * u * base };
  };
  const ARCS = $derived([arcsOf(0), arcsOf(1)]);

  // ---------- residual ----------
  const rs = $derived.by(() => {
    const cw = (W - 2 * PAD) / 16, ch = narrow ? 24 : 30, gap = narrow ? 14 : 18;
    const rows = [0, 1, 2].map((k) => ({ labelY: 18 + k * (ch + 18 + gap), y: 24 + k * (ch + 18 + gap) }));
    return { cw, ch, rows };
  });
  const rsVec = $derived.by(() => {
    if (!data) return null;
    const l = L1 - 1, B = data.blocks[l], q = data.q, d = data.d;
    const x = rowOf(B.input, q, d), dl = rowOf(B.attn, q, d), s = rowOf(B.r1, q, d);
    return { x, d: dl, s, max: maxAbs(x, dl, s), nx: norm2(x), nd: norm2(dl), ns: norm2(s) };
  });
  const bars = $derived.by(() => {
    if (!data) return null;
    const B = data.blocks[L1 - 1], T = data.T, d = data.d;
    const rows = Array.from({ length: T }, (_, i) => ({ word: data!.words[i], x: norm2(rowOf(B.input, i, d)), u: norm2(rowOf(B.attn, i, d)) }));
    return { rows, max: Math.max(...rows.map((r) => Math.max(r.x, r.u))) };
  });

  // ---------- mlp ----------
  const ml = $derived.by(() => {
    const Lm = narrow ? 58 : 64, R = narrow ? 24 : 78, cw = (W - Lm - R) / 64, rh = narrow ? 18 : 24, y0 = 28;
    const stripY = y0 + 8 * rh + (narrow ? 34 : 38), ocw = (W - 2 * PAD) / 16, och = narrow ? 22 : 26;
    return { Lm, R, cw, rh, y0, stripY, ocw, och };
  });
  const mlMax = $derived(data ? maxAbs(data.blocks[L1 - 1].pre) : 1);

  // ---------- norm ----------
  const nm = $derived.by(() => {
    const cx0 = narrow ? 40 : 318, cw = narrow ? W - 40 - 8 : W - 318 - 10, cy0 = narrow ? 186 : 30, ch = narrow ? H - 186 - 44 : H - 30 - 46;
    return { cx0, cw, cy0, ch, gw: cw / 13 };
  });
  const yLog = (v: number) => nm.cy0 + nm.ch * (1 - Math.log10(Math.max(v, 0.01) / 0.01) / 2);
  const wiring = (kind: 'post' | 'pre') => {
    const x0 = PAD, w = narrow ? W - 2 * PAD : 250, bw = 42, step = (w - bw) / 3;
    const order = kind === 'post' ? ['x', 'F', '+', 'LN'] : ['x', 'LN', 'F', '+'];
    return { x0, bw, step, order, plus: order.indexOf('+'), F: order.indexOf('F') };
  };

  // ---------- stream ----------
  const st = $derived.by(() => {
    const nameW = narrow ? 70 : 118, cw = narrow ? 4 : 6.5, stripW = cw * 16, rh = narrow ? 36 : 38;
    const bx = PAD + nameW + stripW + 10, bwMax = W - PAD - bx - 44, y0 = 34;
    return { nameW, cw, stripW, rh, bx, bwMax, y0, topY: y0 + 5 * rh + 36 };
  });
  const STAGE_NAMES = ['Embedding', '+ attention 1', '+ MLP 1', '+ attention 2', '+ MLP 2'];
  const STAGE_SHORT = ['Embedding', '+ attn 1', '+ MLP 1', '+ attn 2', '+ MLP 2'];
  const logBar = (p: number, max: number) => max * clamp01(Math.log10(Math.max(p, 1e-4) / 1e-4) / 4);

  // ---------- count ----------
  const SWATCH = ['--viz-cat-1', '--viz-cat-2', '--viz-cat-3', '--viz-cat-4'];
  const cnt = $derived.by(() => {
    const colW = narrow ? W - 2 * PAD : 210, ah = narrow ? 170 : H - 70;
    const rx = narrow ? PAD : PAD + colW + 24, rw = W - PAD - rx, ry = narrow ? 24 + ah + 70 : 24;
    return { colW, ah, rx, rw, ry };
  });
  const pct = (v: number, tot: number) => { const x = 100 * v / Math.max(1, tot); return x > 0 && x < 0.1 ? '<0.1%' : x.toFixed(1) + '%'; };
  const cf = $derived(frame.mode === 'count' ? frame : null);
  const cParts = $derived(cf ? [['Embeddings', cf.counts.embeddings], ['Attention', cf.counts.attention], ['MLP', cf.counts.mlp], ['Norms and output bias', cf.counts.norms]] as [string, number][] : []);
  const setSize = (patch: Partial<{ d: number; L: number }>) => {
    const cur = cf?.config ?? PRESETS[1];
    custom = { d: cur.d, L: cur.L, ...patch };
    playing = false; seek(3.5);
  };
  const pick = (at: number) => { if (init === 'count' && custom && at !== -1) { custom = null; } jump(at); };
  const segs = $derived(init === 'count'
    ? [...PRESETS.map((p, i) => ({ at: i * 6, label: p.name })), ...(custom ? [{ at: -1, label: 'Your size' }] : [])]
    : tl.markers);
  const activeSeg = $derived(init === 'count' ? (custom ? 'Your size' : tl.markers[phaseOf(uiT, tl.markers)].label) : tl.markers[phaseOf(uiT, tl.markers)].label);
</script>

<div class="bf">
  <FigureCard {title} label={frame.status} aspect="720 / {HEIGHTS[init][0]}" narrowAspect="288 / {HEIGHTS[init][1]}" {legend} role={init === 'count' ? 'group' : 'img'}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {W} {H}" aria-hidden="true">
        {#if !ready}
          <text x={W / 2} y={H / 2} text-anchor="middle" class="small">{failed ? 'The model could not be loaded.' : 'Loading the model…'}</text>

        {:else if frame.mode === 'overview' && data}
          {#each data.words as _s, i (i)}
            <line x1={ov.colX(i)} x2={ov.colX(i)} y1={ov.ys[0]} y2={ov.outB} class="stream" />
          {/each}
          {#each [0, 1] as l (l)}
            {@const A = ov.bands[l]}
            {@const M = ov.mlps[l]}
            {@const ga = bump(frame.s, A.p)}
            {@const gm = bump(frame.s, M.p)}
            <rect x="3" y={M.t - 20} width={W - 6} height={A.b - M.t + 26} rx="6" class="blockframe" />
            <text x={PAD + 2} y={M.t - 7} class="small">Block {l + 1}</text>
            <rect x={PAD - 4} y={A.t} width={W - 2 * PAD + 8} height={A.b - A.t} rx="4" class="base" />
            <rect x={PAD - 4} y={A.t} width={W - 2 * PAD + 8} height={A.b - A.t} rx="4" class="band" style:stroke-opacity={0.4 + 0.6 * ga} style:fill-opacity={ga * 0.7} />
            <text x={PAD + 2} y={A.t + 13} class="heading">Attention + add</text>
            {#each ARCS[l] as a (`${a.j}-${a.i}`)}
              <path d={arcPath(a, A.b - 6)} class="arc" style:stroke-width={0.6 + 5 * a.w} style:opacity={0.12 + 0.75 * ga * clamp01(a.w * 3)} />
              {#if frame.phase === A.p}
                {@const pt = arcPoint(a, A.b - 6, (frame.local * 1.6 + ((a.i * 7 + a.j * 3) % 10) / 10) % 1)}
                <circle cx={pt.x} cy={pt.y} r="2.6" class="dot" />
              {/if}
            {/each}
            <rect x={PAD - 4} y={M.t} width={W - 2 * PAD + 8} height={M.b - M.t} rx="4" class="base" />
            <rect x={PAD - 4} y={M.t} width={W - 2 * PAD + 8} height={M.b - M.t} rx="4" class="band" style:stroke-opacity={0.4 + 0.6 * gm} style:fill-opacity={gm * 0.7} />
            <text x={PAD + 2} y={M.t + 13} class="heading">MLP + add</text>
            {#each data.words as _w, i (i)}
              <rect x={ov.colX(i) - ov.colW / 2 + 3} y={M.t + 18} width={ov.colW - 6} height={M.b - M.t - 22} rx="2" class="cellbox" style:stroke-opacity={0.4 + 0.6 * gm} />
            {/each}
          {/each}
          <rect x={PAD - 4} y={ov.embT} width={W - 2 * PAD + 8} height={ov.embB - ov.embT} rx="4" class="base" />
          <rect x={PAD - 4} y={ov.embT} width={W - 2 * PAD + 8} height={ov.embB - ov.embT} rx="4" class="band" style:fill-opacity={bump(frame.s, -0.2) * 0.7} />
          <text x={PAD + 2} y={ov.embT + 17} class="heading">Embedding + position</text>
          <rect x={PAD - 4} y={ov.outT} width={W - 2 * PAD + 8} height={ov.outB - ov.outT} rx="4" class="base" />
          <rect x={PAD - 4} y={ov.outT} width={W - 2 * PAD + 8} height={ov.outB - ov.outT} rx="4" class="band" style:fill-opacity={bump(frame.s, 5) * 0.7} />
          <text x={PAD + 2} y={ov.outT + 18} class="heading">{narrow ? 'Final norm, output scores' : 'Final norm and output layer: a score for every next word'}</text>
          {#each data.words as word, i (i)}
            <circle cx={ov.colX(i)} cy={dotY(frame.s)} r="4.5" class="vec" />
            {#if narrow}
              <text transform="translate({ov.colX(i) + 3},{H - 4}) rotate(-90)" class="tok">{word}</text>
            {:else}
              <text x={ov.colX(i)} y={H - 10} text-anchor="middle" class="tok">{word}</text>
            {/if}
          {/each}

        {:else if frame.mode === 'residual' && rsVec && bars && data}
          {#if frame.phase < 3}
            {#each [{ k: 0, vec: rsVec.x, name: `x, the vector entering block ${L1}`, n: rsVec.nx, cells: 16 * frame.xA, a: frame.xA },
              { k: 1, vec: rsVec.d, name: 'update written by attention', n: rsVec.nd, cells: frame.dCells, a: 1 },
              { k: 2, vec: rsVec.s, name: 'x + update', n: rsVec.ns, cells: frame.sCells, a: 1 }] as r (r.k)}
              {@const row = rs.rows[r.k]}
              {#if r.cells > 0}
                <text x={PAD} y={row.labelY} class="heading" style:opacity={r.k === 0 ? frame.xA : 1}>{r.name}</text>
                <text x={W - PAD} y={row.labelY} text-anchor="end" class="small num" style:opacity={r.k === 0 ? frame.xA : clamp01(r.cells - 15)}>length {r.n.toFixed(2)}</text>
              {/if}
              {#each Array.from({ length: 16 }, (_, j) => j) as j (j)}
                {@const s = cellStyle(r.vec[j], rsVec.max)}
                <rect x={PAD + j * rs.cw} y={row.y} width={rs.cw - 1} height={rs.ch} rx="2" class="slot" />
                <rect x={PAD + j * rs.cw} y={row.y} width={rs.cw - 1} height={rs.ch} rx="2" style:fill="var({s.color})" style:opacity={s.a * clamp01(r.cells - j) * (r.k === 0 ? r.a : 1)} />
                {#if !narrow && r.cells - j >= 1}<text x={PAD + j * rs.cw + rs.cw / 2} y={row.y + rs.ch / 2 + 4} text-anchor="middle" class="cellnum">{f2(r.vec[j])}</text>{/if}
              {/each}
            {/each}
          {:else}
            {@const lx = narrow ? 58 : 64}
            {@const bw = W - lx - 52}
            <text x={PAD} y="16" class="heading">Length of x and of the update, block {L1}</text>
            {#each bars.rows as r, i (i)}
              {@const y = 30 + i * (narrow ? 34 : 33)}
              <text x={lx - 6} y={y + 17} text-anchor="end" class="tok" class:strong={i === data.q}>{r.word}</text>
              <rect x={lx} y={y + 2} width={bw * r.x / bars.max * frame.barsA} height="12" class="barx" />
              <rect x={lx} y={y + 16} width={bw * r.u / bars.max * frame.barsA} height="12" class="baru" />
              <text x={lx + bw * r.x / bars.max * frame.barsA + 4} y={y + 12} class="val num" style:opacity={frame.barsA}>{r.x.toFixed(2)}</text>
              <text x={lx + bw * r.u / bars.max * frame.barsA + 4} y={y + 26} class="val num" style:opacity={frame.barsA}>{r.u.toFixed(2)}</text>
            {/each}
          {/if}

        {:else if frame.mode === 'mlp' && data}
          {@const B = data.blocks[L1 - 1]}
          <text x={ml.Lm} y="18" class="heading">{data.ff} hidden units, one row per word</text>
          <text x={W - 2} y="18" text-anchor="end" class="small" style:opacity={frame.countsA}>{narrow ? 'on' : 'units kept'}</text>
          {#each data.words as word, i (i)}
            {@const y = ml.y0 + i * ml.rh}
            <text x={ml.Lm - 5} y={y + ml.rh / 2 + 4} text-anchor="end" class="tok" class:strong={i === data.q || i === data.T - 1}>{word}</text>
            <rect x={ml.Lm} y={y + 1} width={W - ml.Lm - ml.R} height={ml.rh - 2} class="slot" />
            {#each Array.from({ length: data.ff }, (_, j) => j) as j (j)}
              {@const v = B.pre[i * data.ff + j]}
              {@const s = cellStyle(v, mlMax)}
              <rect x={ml.Lm + j * ml.cw} y={y + 1} width={Math.max(0.5, ml.cw - (narrow ? 0.3 : 1))} height={ml.rh - 2} style:fill="var({s.color})" style:opacity={s.a * frame.rowA[i] * (v < 0 ? 1 - frame.relu : 1)} />
            {/each}
            {#if i === data.q || i === data.T - 1}<rect x={ml.Lm - 1} y={y} width={W - ml.Lm - ml.R + 2} height={ml.rh} class="focus" />{/if}
            <text x={W - 2} y={y + ml.rh / 2 + 4} text-anchor="end" class="small num" style:opacity={frame.countsA}>{narrow ? B.active[i] : `${B.active[i]} of ${data.ff}`}</text>
          {/each}
          {@const Q = rowOf(B.mlp, data.q, data.d)}
          {@const qMax = maxAbs(Q)}
          <text x={PAD} y={ml.stripY - 8} class="heading" style:opacity={clamp01(frame.outCells)}>"{data.words[data.q]}": {narrow ? "output, 16 numbers" : "16 numbers added to the stream"}</text>
          <text x={W - PAD} y={ml.stripY - 8} text-anchor="end" class="small num" style:opacity={clamp01(frame.outCells - 15)}>length {norm2(Q).toFixed(2)}</text>
          {#each Array.from({ length: 16 }, (_, j) => j) as j (j)}
            {@const s = cellStyle(Q[j], qMax)}
            <rect x={PAD + j * ml.ocw} y={ml.stripY} width={ml.ocw - 1} height={ml.och} rx="2" class="slot" />
            <rect x={PAD + j * ml.ocw} y={ml.stripY} width={ml.ocw - 1} height={ml.och} rx="2" style:fill="var({s.color})" style:opacity={s.a * clamp01(frame.outCells - j)} />
          {/each}

        {:else if frame.mode === 'norm'}
          {#each ['post', 'pre'] as kind, k (kind)}
            {@const g = wiring(kind as 'post' | 'pre')}
            {@const y = 12 + k * (narrow ? 78 : 90)}
            <text x={g.x0} y={y + 10} class="heading">{kind === 'post' ? 'Post-norm: LN(x + F(x))' : 'Pre-norm: x + F(LN(x))'}</text>
            <path d="M{g.x0 + g.bw / 2},{y + 46} V{y + 54} H{g.x0 + g.step * g.plus + g.bw / 2} V{y + 46}" class="skip" />
            {#each g.order as label, i (i)}
              <rect x={g.x0 + i * g.step} y={y + 20} width={g.bw} height="26" rx="4" class="cellbox" class:norm={label === 'LN'} />
              <text x={g.x0 + i * g.step + g.bw / 2} y={y + 37} text-anchor="middle" class="tok">{label}</text>
              {#if i < 3}<path d="M{g.x0 + i * g.step + g.bw + 2},{y + 33} H{g.x0 + (i + 1) * g.step - 2}" class="skip arrow" />{/if}
            {/each}
          {/each}
          {#each [0.01, 0.1, 1] as v (v)}
            <line x1={nm.cx0} x2={nm.cx0 + nm.cw} y1={yLog(v)} y2={yLog(v)} class="gridline" />
            <text x={nm.cx0 - 5} y={yLog(v) + 4} text-anchor="end" class="small">{v}</text>
          {/each}
          <text x={nm.cx0} y={nm.cy0 - 10} class="heading">Gradient norm (log scale)</text>
          {#each NORM_PRE as pre, i (i)}
            {@const post = NORM_POST[i]}
            {@const x = nm.cx0 + i * nm.gw}
            {@const bw = nm.gw * 0.4}
            {@const k = clamp01(frame.reveal - i)}
            {@const dim = frame.phase === 1 && i === 0 ? 0.35 : 1}
            <rect x={x + nm.gw * 0.08} y={yLog(pre) + (1 - k) * (nm.cy0 + nm.ch - yLog(pre))} width={bw} height={(nm.cy0 + nm.ch - yLog(pre)) * k} style:fill="var(--viz-cat-1)" opacity={dim} />
            <rect x={x + nm.gw * 0.08 + bw + 1} y={yLog(post) + (1 - k) * (nm.cy0 + nm.ch - yLog(post))} width={bw} height={(nm.cy0 + nm.ch - yLog(post)) * k} style:fill="var(--viz-cat-2)" opacity={dim} />
            <text x={x + nm.gw / 2} y={nm.cy0 + nm.ch + 14} text-anchor="middle" class="small">{i}</text>
          {/each}
          <text x={nm.cx0 + nm.cw / 2} y={nm.cy0 + nm.ch + 32} text-anchor="middle" class="small">After block k (0 = embeddings)</text>

        {:else if frame.mode === 'stream' && data}
          {@const k = frame.phase}
          <text x={PAD} y="20" class="heading">{narrow ? 'At' : 'Reading the stream at'} "{data.words[data.q]}"</text>
          <text x={st.bx} y="20" class="small">{narrow ? 'Prob. (log)' : 'Next-word probability (log scale)'}</text>
          {#each frame.rows as r (r.stage)}
            {@const y = st.y0 + r.stage * st.rh}
            {@const vec = data.stages[r.stage].vec}
            {@const m = maxAbs(vec)}
            {@const cur = r.stage === k}
            {#if cur}<rect x="3" y={y - 3} width={W - 6} height={st.rh - 2} rx="4" class="focus" />{/if}
            <text x={PAD} y={y + st.rh / 2 + 1} class="tok" class:strong={cur}>{narrow ? STAGE_SHORT[r.stage] : STAGE_NAMES[r.stage]}</text>
            {#each Array.from(vec) as v, j (j)}
              {@const s = cellStyle(v, m)}
              <rect x={PAD + st.nameW + j * st.cw} y={y + 4} width={st.cw - 0.6} height={st.rh - 14} class="slot" />
              <rect x={PAD + st.nameW + j * st.cw} y={y + 4} width={st.cw - 0.6} height={st.rh - 14} style:fill="var({s.color})" style:opacity={s.a * r.vecA} />
            {/each}
            <rect x={st.bx} y={y + 3} width={logBar(r.chaseBar, st.bwMax)} height="9" style:fill="var(--viz-cat-1)" />
            <text x={st.bx + logBar(r.chaseBar, st.bwMax) + 4} y={y + 11.5} class="small num">{fmtP(r.chase)}</text>
            <rect x={st.bx} y={y + 15} width={logBar(r.chasesBar, st.bwMax)} height="9" style:fill="var(--viz-cat-2)" />
            <text x={st.bx + logBar(r.chasesBar, st.bwMax) + 4} y={y + 23.5} class="small num">{fmtP(r.chases)}</text>
          {/each}
          {#each narrow ? ['1e-4', '0.01', '1'] : ['0.0001', '0.01', '1'] as v, i (v)}
            <text x={st.bx + st.bwMax * i / 2} y={st.y0 + 5 * st.rh + 8} text-anchor={i === 2 ? 'end' : 'start'} class="small">{v}</text>
          {/each}
          <text x={PAD} y={st.topY - 8} class="heading">Top 3 next words at this stage</text>
          {#each frame.top as e, i (e.word)}
            {@const y = st.topY + i * 22}
            <text x={PAD + 2} y={y + 12} class="tok">{e.word}</text>
            <rect x={PAD + 64} y={y + 2} width={Math.max(0, (W - PAD - 64 - PAD - 44) * e.bar)} height="13" class="barx" />
            <text x={PAD + 64 + Math.max(0, (W - PAD - 64 - PAD - 44) * e.bar) + 4} y={y + 12} class="small num">{fmtP(e.p)}</text>
          {/each}

        {:else if frame.mode === 'count'}
          {@const c = frame.config}
          {@const N = c.L}
          {@const bh = Math.min(24, (cnt.ah - 30) / N - (N > 24 ? 1 : 3))}
          {@const gapY = N > 24 ? 1 : 3}
          {@const wf = 0.28 + 0.72 * Math.log(c.d / 16) / Math.log(128)}
          <text x={PAD} y="16" class="heading">{c.L} blocks</text>
          <text x={PAD + cnt.colW} y="16" text-anchor="end" class="small">width = d {fmtN(c.d)}</text>
          {#each Array.from({ length: N }, (_, i) => i) as i (i)}
            {@const w = Math.min(cnt.colW, cnt.colW * wf)}
            {@const a = clamp01(frame.blocks - i)}
            <rect x={PAD + (cnt.colW - w) / 2} y={24 + cnt.ah - 28 - (i + 1) * (bh + gapY)} width={w} height={bh} rx="2" class="blk" style:opacity={a} />
          {/each}
          <rect x={PAD + (cnt.colW - cnt.colW * 0.28) / 2} y={24 + cnt.ah - 20} width={cnt.colW * 0.28} height="18" rx="2" class="emb" />
          <text x={PAD + cnt.colW / 2} y={24 + cnt.ah + 14} text-anchor="middle" class="small">embedding table</text>
          <text x={cnt.rx} y={cnt.ry + 6} class="small">Total parameters</text>
          <text x={cnt.rx} y={cnt.ry + 36} class="big num">{fmtN(frame.counts.total)}</text>
          {#each cParts as p, i (i)}
            {@const before = cParts.slice(0, i).reduce((s, q) => s + q[1], 0)}
            {@const tot = Math.max(1, frame.counts.total)}
            <rect x={cnt.rx + cnt.rw * before / tot} y={cnt.ry + 50} width={Math.max(0, cnt.rw * p[1] / tot)} height="22" style:fill="var({SWATCH[i]})" />
          {/each}
          {#each cParts as p, i (i)}
            {@const y = cnt.ry + 94 + i * (narrow ? 40 : 28)}
            <rect x={cnt.rx} y={y - 11} width="12" height="12" style:fill="var({SWATCH[i]})" />
            <text x={cnt.rx + 20} y={y} class="tok">{p[0]}</text>
            <text x={cnt.rx + cnt.rw} y={y + (narrow ? 14 : 0)} text-anchor="end" class="small num">{fmtN(p[1])} ({pct(p[1], frame.counts.total)})</text>
          {/each}
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tlc} bind:t={uiT} bind:playing bind:speed duration={tl.duration} markers={[]} label="Timeline" onSeek={seek} onReset={() => seek(0)} />
      <div class="segments" aria-label="Choose a timeline segment">
        {#each segs as s (s.label)}
          <button type="button" class:active={s.label === activeSeg} onclick={() => pick(s.at)}>{s.label}</button>
        {/each}
      </div>
    {/snippet}
    {#snippet readout()}
      <p class="status" role="status" aria-live="polite">{frame.status}</p>
    {/snippet}
    {#snippet footer()}
      {#if init === 'count'}
        <div class="sliders">
          <label>Width d: <span class="num">{fmtN(cf?.config.d ?? 0)}</span>
            <input type="range" min="16" max="2048" step="16" value={cf?.config.d ?? 768} oninput={(e) => setSize({ d: +e.currentTarget.value })} aria-label="Model width d" /></label>
          <label>Blocks: <span class="num">{cf?.config.L ?? 0}</span>
            <input type="range" min="1" max="48" step="1" value={cf?.config.L ?? 12} oninput={(e) => setSize({ L: +e.currentTarget.value })} aria-label="Number of blocks" /></label>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .heading { fill: var(--text); font: 600 12.5px var(--font-ui); }
  .small { fill: var(--text-muted); font: 11px var(--font-code); }
  .val { fill: var(--text); font: 11px var(--font-code); }
  .tok { fill: var(--text); font: 12px var(--font-code); }
  .tok.strong { font-weight: 700; }
  .cellnum { fill: var(--text); font: 10px var(--font-code); }
  .big { fill: var(--text); font: 600 26px var(--font-code); }
  .blockframe { fill: none; stroke: var(--rule); stroke-dasharray: 4 3; }
  .base { fill: var(--bg-raised); stroke: none; }
  .band { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1; }
  .cellbox { fill: var(--bg-sunken); stroke: var(--rule); }
  .cellbox.norm { stroke: var(--text-muted); }
  .stream { stroke: var(--rule); stroke-width: 1.5; }
  .vec { fill: var(--viz-forward); stroke: var(--bg-raised); stroke-width: 1.5; }
  .dot { fill: var(--viz-seq-5); }
  .arc { fill: none; stroke: var(--viz-seq-4); stroke-linecap: round; }
  .slot { fill: var(--bg-sunken); stroke: none; }
  .focus { fill: none; stroke: var(--accent); stroke-width: 1.2; }
  .barx { fill: var(--viz-seq-4); }
  .baru { fill: var(--viz-seq-2); stroke: var(--rule); }
  .gridline { stroke: var(--rule); stroke-dasharray: 3 3; }
  .skip { fill: none; stroke: var(--text-muted); stroke-width: 1.3; }
  .blk { fill: var(--viz-seq-3); stroke: var(--bg-raised); }
  .emb { fill: var(--viz-cat-1); }
  .status { margin: 0; min-height: 3.6em; }
  .segments { display: flex; flex-wrap: wrap; gap: 0.35rem; padding: var(--s-2) var(--s-4); border-top: 1px solid var(--rule); }
  .segments button { font: 0.8rem var(--font-ui); padding: 0.35em 0.55em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .segments button.active { border-color: var(--accent); font-weight: 650; }
  .sliders { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .sliders label { display: flex; align-items: center; gap: var(--s-2); font: 0.85rem var(--font-ui); flex: 1 1 14rem; }
  .sliders input { flex: 1; min-width: 6rem; accent-color: var(--accent); }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
