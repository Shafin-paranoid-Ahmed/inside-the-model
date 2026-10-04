<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, markerAt, prefersReducedMotion, reducedMotion, type Marker } from '../_shared';
  import { loadModel, type Weights } from '../../../lib/transformer';
  import { CHARSET, decode } from '../../../lib/lm/charset';
  import {
    PASSAGE, BASE_PROMPT, BASE_N, MIN_EXP, GPT3_BATCH_TOKENS, eng, mant, expo, show, scorePassage, objectiveAt, lossAt, objectiveDuration, lossDuration, textMarkers, OBJ_BEAT, LOSS_BEAT,
    analyse, dataAt, GATES, ladder, computeAt, computeDuration, computeMarkers, learningAt, learningMarkers, stepX, LEARN_T,
    baseJob, baseWork, baseAt, baseDuration, baseMarkers, type Mode, type Log, type Scored, type Lane,
  } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'objective', title = '' }: Props = $props();
  const init = untrack(() => mode);
  const TITLES: Record<Mode, string> = {
    objective: 'Guessing the next character', loss: 'How wrong each guess was', data: 'Cleaning a pile of documents', compute: 'Training compute, from this model to published ones',
    learning: 'What the model knows at each point in training', base: 'A base model continues a prompt',
  };
  const HW: Record<Mode, number> = { objective: 270, loss: 250, data: 366, compute: 300, learning: 380, base: 290 };
  const HN: Record<Mode, number> = { objective: 296, loss: 250, data: 366, compute: 340, learning: 470, base: 420 };

  let width = $state(0), playing = $state(false), speed = $state(1), uiT = $state(0);
  const narrow = $derived(width > 0 && width < 540);
  const W = $derived(narrow ? 288 : 720), H = $derived(narrow ? HN[init] : HW[init]);
  let w = $state.raw<Weights | null>(null), log = $state.raw<Log | null>(null), failed = $state(false);
  let threshold = $state(0.7), logN = $state(9), logD = $state(Math.log10(2e10)), prompt = $state(BASE_PROMPT);
  const needW = init === 'objective' || init === 'loss' || init === 'base', needLog = init === 'compute' || init === 'learning';
  const ready = $derived((!needW || !!w) && (!needLog || !!log));

  const scored: Scored | null = $derived(w ? scorePassage(w, PASSAGE) : null);
  const run = $derived(analyse(threshold));
  const rungs = $derived(log ? ladder(log, 10 ** logN, 10 ** logD) : []);
  let lanes = $state.raw<Lane[]>([]);
  $effect(() => {
    if (init !== 'base' || !w) return;
    const job = baseJob(w, prompt); let dead = false, timer = 0;
    const pump = () => {
      if (dead) return;
      const done = baseWork(job, 8);
      lanes = job.lanes.map((l) => ({ ...l, ids: l.ids.slice(), alts: l.alts.slice() }));
      if (!done) timer = setTimeout(pump, 0);
    };
    pump();
    return () => { dead = true; clearTimeout(timer); };
  });

  const duration = $derived.by(() => {
    if (init === 'objective') return scored ? objectiveDuration(scored) : 10;
    if (init === 'loss') return scored ? lossDuration(scored) : 10;
    if (init === 'data') return run.duration;
    if (init === 'compute') return rungs.length ? computeDuration(rungs) : 10;
    if (init === 'learning') return LEARN_T;
    return baseDuration();
  });
  const markers: Marker[] = $derived.by(() => {
    if (init === 'objective') return scored ? textMarkers(scored, OBJ_BEAT) : [];
    if (init === 'loss') return scored ? textMarkers(scored, LOSS_BEAT) : [];
    if (init === 'data') return run.markers;
    if (init === 'compute') return computeMarkers(rungs);
    if (init === 'learning') return log ? learningMarkers(log) : [];
    return baseMarkers();
  });
  const seg = $derived(markerAt(uiT, markers));

  const fo = $derived(init === 'objective' && scored ? objectiveAt(uiT, scored) : null);
  const fl = $derived(init === 'loss' && scored ? lossAt(uiT, scored) : null);
  const fd = $derived(init === 'data' ? dataAt(uiT, run) : null);
  const fc = $derived(init === 'compute' && rungs.length ? computeAt(uiT, rungs) : null);
  const fn = $derived(init === 'learning' && log ? learningAt(uiT, log) : null);
  const fb = $derived(init === 'base' && lanes.length && lanes[0].ids.length ? { ...baseAt(uiT), typed: Math.min(baseAt(uiT).typed, lanes[0].ids.length) } : null);

  const text = $derived.by(() => {
    if (!ready) return 'Loading the model…';
    if (fo && scored) { const read = scored.text.slice(0, fo.k + 1); return `After “${read.length > 14 ? '…' + read.slice(-13) : read}” the true next character is “${fo.next}”. The model gave it probability ${fo.pTrue.toFixed(2)}.`; }
    if (fl && scored) {
      const h = fl.heights[fl.k], done = uiT >= scored.p.length * LOSS_BEAT;
      return done ? `Average over the whole sentence: ${scored.mean.toFixed(2)} nats per character, perplexity ${scored.ppl.toFixed(2)}. Blind guessing costs ln 38 = ${Math.log(38).toFixed(2)}.`
        : `Next character “${fl.next}”: loss ${h.toFixed(2)} nats. Average so far ${fl.mean.toFixed(2)} nats, perplexity ${fl.ppl.toFixed(2)}.`;
    }
    if (fd) return fd.status + (fd.active && uiT < run.gateStart[3] ? ` Kept so far: ${Math.round(fd.chars).toLocaleString('en-US')} of ${run.totalChars.toLocaleString('en-US')} characters.` : '');
    if (fc) return fc.text;
    if (fn) return fn.text;
    return `${fb ? fb.typed : 0} of ${BASE_N} characters written after the prompt.`;
  });

  /* ---- layout helpers */
  const rowsOf = (s: string, per: number) => Array.from({ length: Math.ceil(s.length / per) }, (_, i) => s.slice(i * per, (i + 1) * per));
  const L = $derived({ x0: narrow ? 12 : 12, per: narrow ? 24 : 48, cw: narrow ? 11.5 : 14 });
  const pFmt = (p: number) => (p >= 0.99 || (p < 0.1 && !narrow) ? p.toFixed(3) : p.toFixed(2));
  const bars = $derived.by(() => {
    const top = narrow ? 82 : 58, h = 150, sw = (W - 36) / 8;
    return { top, h, sw, base: top + 14 + h };
  });
  const lp = $derived({ x0: 40, x1: W - 12, y0: 28, y1: 212, cw: (W - 52) / 46 });
  const ly = (v: number) => lp.y1 - (v / 6) * (lp.y1 - lp.y0);
  const running = $derived.by(() => {
    if (!fl || !scored) return '';
    let s = 0; const pts: string[] = [];
    for (let i = 0; i <= fl.k; i++) {
      s += i === fl.k ? fl.heights[i] : scored.loss[i];
      pts.push(`${lp.x0 + (i + 0.5) * lp.cw},${ly(s / (i + 1))}`);
    }
    return 'M' + pts.join(' L');
  });
  // data
  const nameX = 12, cellX = $derived(narrow ? [170, 208, 246] : [226, 384, 542]), cellW = $derived(narrow ? 30 : 152);
  const ry = (i: number) => 36 + i * 24;
  // compute
  const cx0 = $derived(narrow ? 12 : 142), cx1 = $derived(W - 16), rowH = $derived(narrow ? 56 : 50), axisY = $derived(H - 40);
  const ticks = $derived.by(() => {
    if (!fc) return [] as number[];
    const lo = MIN_EXP, hi = fc.maxExp, step = Math.max(1, Math.ceil((hi - lo) / (narrow ? 4 : 7)));
    return Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, i) => lo + i * step);
  });
  // learning
  const gp = $derived(narrow ? { x0: 40, x1: W - 12, y0: 14, y1: 176 } : { x0: 56, x1: 470, y0: 18, y1: 250 });
  const gy = (l: number) => gp.y1 - ((l - 1.4) / 2.5) * (gp.y1 - gp.y0);
  const gx = (s: number) => (log ? gp.x0 + stepX(log, s) * (gp.x1 - gp.x0) : gp.x0);
  const curve = $derived(fn && log ? 'M' + log.curve.slice(0, fn.idx + 1).map(([s, l]) => `${gx(s)},${gy(l)}`).join(' L') : '');
  const probeBox = $derived(narrow ? { x: 12, y: 232, w: W - 24 } : { x: 506, y: 22, w: 202 });
  const sampleY = $derived(narrow ? 360 : 326);

  /* ---- loop */
  let stage: HTMLDivElement, tl: ReturnType<typeof Timeline>, t = 0, lastPaint = 0;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; threshold; logN; logD; prompt; ready; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  onMount(() => {
    if (needW) loadModel('house').then((m) => (w = m), () => (failed = true));
    if (needLog) fetch('/models/house-log.json').then((r) => r.json()).then((j: Log) => (log = j), () => (failed = true));
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing: playing && ready, speed }, dt, { duration }).t;
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); motion.removeEventListener('change', onMotion); };
  });
  const onPrompt = (e: Event) => {
    const el = e.target as HTMLInputElement, v = [...el.value.toLowerCase()].filter((c) => CHARSET.includes(c) && c !== '\n').join('').slice(0, 40);
    el.value = v; if (v) { prompt = v; seek(0); playing = !reducedMotion(); }
  };
  const legend = {
    objective: [{ label: 'Probability the model gives each character', color: '--viz-seq-3' }, { label: 'The true next character', color: '--accent', shape: 'line' as const }],
    loss: [{ label: 'Loss of one guess', color: '--viz-seq-4' }, { label: 'Biggest miss', color: '--viz-neg' }, { label: 'Average so far', color: '--accent', shape: 'line' as const }, { label: 'Blind guess, ln 38', color: '--text-muted', shape: 'line' as const }],
    data: [{ label: 'Kept', color: '--viz-forward' }, { label: 'Removed', color: '--viz-neg' }],
    compute: [{ label: 'This lesson’s model', color: '--accent' }, { label: 'Your model', color: '--viz-pos' }, { label: 'Published', color: '--viz-forward' }],
    learning: [{ label: 'Validation loss', color: '--viz-forward', shape: 'line' as const }, { label: 'Simple guessing rules', color: '--text-muted', shape: 'line' as const }, { label: 'Saved sample', color: '--accent' }],
    base: [{ label: 'Three likeliest characters at each position; fainter is less likely', color: '--text-muted' }],
  }[init];
  const aria = { objective: 'Bars of the probabilities the model gives each next character as it reads a sentence', loss: 'Columns showing the loss at each character of a sentence', data: 'Twelve documents passing through three cleaning gates', compute: 'Training compute of several models on a log axis', learning: 'Validation loss curve, a text sample and three probe probabilities as training proceeds', base: 'Three continuations of a question prompt, typed one character at a time' }[init];
</script>

<div class="lab">
  <FigureCard title={title || TITLES[init]} label={aria} aspect="720 / {HW[init]}" narrowAspect="288 / {HN[init]}" {legend}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {W} {H}" aria-hidden="true">
        {#if failed}
          <text x="12" y="30" class="small">The model could not be loaded. Reload the page to try again.</text>
        {:else if !ready}
          <text x={W / 2} y={H / 2} text-anchor="middle" class="small">Loading the tiny model…</text>
        {:else if fo && scored}
          <text x="12" y="16" class="small">{narrow ? 'Dark: read so far. Boxed: to guess.' : 'Held-out sentence from the book. The model has read the dark letters and guesses the boxed one.'}</text>
          {#each [...scored.text] as c, i (i)}
            {@const x = L.x0 + (i % L.per) * L.cw}
            {@const y = (narrow ? 50 : 40) + Math.floor(i / L.per) * 22}
            {#if i === fo.k + 1}<rect x={x - 1} y={y - 15} width={L.cw} height="20" rx="3" class="target" />{/if}
            <text x={x + L.cw / 2 - 0.5} {y} text-anchor="middle" class="pass" class:read={i <= fo.k} class:next={i === fo.k + 1}>{show(c)}</text>
          {/each}
          {#each [0, 0.5, 1] as g (g)}
            <line x1="30" x2={W - 6} y1={bars.base - g * bars.h} y2={bars.base - g * bars.h} class="grid" />
            <text x="26" y={bars.base - g * bars.h + 4} text-anchor="end" class="small">{g}</text>
          {/each}
          {#each fo.bars as b, i (b.id)}
            {@const bx = 36 + i * bars.sw}
            <rect x={bx + bars.sw * 0.2} y={bars.base - b.v * bars.h} width={bars.sw * 0.6} height={Math.max(0.5, b.v * bars.h)} class="bar" class:truebar={b.isTrue} />
            <text x={bx + bars.sw / 2} y={bars.base - b.v * bars.h - 5} text-anchor="middle" class="small">{pFmt(b.v)}</text>
            <text x={bx + bars.sw / 2} y={bars.base + 18} text-anchor="middle" class="lab14" class:strong={b.isTrue}>{b.ch}</text>
            {#if b.isTrue}<text x={bx + bars.sw / 2} y={bars.base + 32} text-anchor="middle" class="accent">true</text>{/if}
          {/each}
        {:else if fl && scored}
          {#each [0, 2, 4, 6] as g (g)}
            <line x1={lp.x0} x2={lp.x1} y1={ly(g)} y2={ly(g)} class="grid" />
            <text x={lp.x0 - 5} y={ly(g) + 4} text-anchor="end" class="small">{g}</text>
          {/each}
          <text x="12" y="12" class="small">Loss (nats)</text>
          <line x1={lp.x0} x2={lp.x1} y1={ly(Math.log(38))} y2={ly(Math.log(38))} class="blind" />
          <text x={lp.x0 + 4} y={ly(Math.log(38)) - 8} class="small halo">{narrow ? 'ln 38 = 3.64' : 'blind guess: ln 38 = 3.64'}</text>
          {#each fl.heights as h, i (i)}
            {@const big = i === fl.tallest && i < fl.k}
            <rect x={lp.x0 + i * lp.cw + 1} y={ly(h)} width={Math.max(1, lp.cw - 2)} height={Math.max(0, lp.y1 - ly(h))} class="col" class:big />
            {#if !narrow}<text x={lp.x0 + (i + 0.5) * lp.cw} y={lp.y1 + 16} text-anchor="middle" class="small" class:strong={i === fl.k}>{show(scored.text[i + 1])}</text>{/if}
          {/each}
          {#if fl.tallest < fl.k}<text x={lp.x0 + (fl.tallest + 0.5) * lp.cw} y={ly(scored.loss[fl.tallest]) - 5} text-anchor="middle" class="accent">{show(scored.text[fl.tallest + 1])} {scored.loss[fl.tallest].toFixed(2)}</text>{/if}
          {#if narrow}<text x={lp.x0 + (fl.k + 0.5) * lp.cw} y={lp.y1 + 16} text-anchor="middle" class="accent">{fl.next}</text>{/if}
          <path d={running} class="mean" />
          <text x={lp.x0} y={lp.y1 + (narrow ? 34 : 36)} class="small">{narrow ? 'Column height: −ln p of the true next character' : 'Each column is one position; its height is −ln of the probability given to the true next character.'}</text>
        {:else if fd}
          {#each GATES as g, gi (g)}
            <text x={cellX[gi] + (narrow ? cellW / 2 : 0)} y="22" text-anchor={narrow ? 'middle' : 'start'} class="head">{narrow ? ['Exact', 'Near', 'Rules'][gi] : g}</text>
          {/each}
          <text x={nameX} y="22" class="head">Document</text>
          {#each run.docs as d, i (i)}
            <g opacity={1 - 0.62 * fd.dim[i]}>
              <rect x="6" y={ry(i) - 3} width={W - 12} height="22" rx="3" class="rowbg" class:on={fd.active?.doc === i && fd.gate < 3} />
              <text x={nameX} y={ry(i) + 12} class="mono">{i + 1} {narrow && d.name.length > 20 ? d.name.slice(0, 19) + '…' : d.name}</text>
              {#each d.cells as c, g (g)}
                {#if fd.shown[g][i] && c.state !== 'skip'}
                  {@const mx = cellX[g] + (narrow ? cellW / 2 : 6)}
                  {#if c.state === 'pass'}<circle cx={mx} cy={ry(i) + 8} r="4" class="ok" />{:else}<path d="M{mx - 4},{ry(i) + 4} l8,8 m-8,0 l8,-8" class="no" />{/if}
                  {#if !narrow}<text x={mx + 12} y={ry(i) + 12} class="mono" class:fail={c.state === 'fail'}>{c.note}</text>{/if}
                {:else if c.state === 'skip'}
                  <text x={cellX[g] + (narrow ? cellW / 2 : 6)} y={ry(i) + 12} text-anchor={narrow ? 'middle' : 'start'} class="small">·</text>
                {/if}
              {/each}
            </g>
          {/each}
          <rect x="12" y={ry(12) + 6} width={W - 24} height="14" rx="3" class="rowbg" />
          <rect x="12" y={ry(12) + 6} width={(W - 24) * (fd.chars / run.totalChars)} height="14" rx="3" class="keep" />
          <text x="12" y={ry(12) + 36} class="small">Characters kept: {Math.round(fd.chars).toLocaleString('en-US')} of {run.totalChars.toLocaleString('en-US')}</text>
        {:else if fc}
          {#each ticks as e (e)}
            {@const x = cx0 + ((e - MIN_EXP) / (fc.maxExp - MIN_EXP)) * (cx1 - cx0)}
            <line x1={x} x2={x} y1="6" y2={axisY} class="grid" />
            <text {x} y={axisY + 16} text-anchor="middle" class="small">10<tspan dy="-5" font-size="8">{e}</tspan></text>
          {/each}
          <text x={(cx0 + cx1) / 2} y={H - 4} text-anchor="middle" class="small">{narrow ? 'Compute C = 6ND, FLOP (log scale)' : 'Training compute C = 6ND, in floating-point operations (log scale)'}</text>
          {#each fc.rows as r, i (i)}
            {#if r.grow > 0}
              {@const y = 8 + i * rowH}
              {@const x = cx0 + r.x * r.grow * (cx1 - cx0)}
              {@const col = r.rung.you ? 'you' : i === 0 ? 'house' : 'pub'}
              <g opacity={0.25 + 0.75 * r.grow}>
                <text x="12" y={y + (narrow ? 12 : 32)} class="head">{r.rung.name}</text>
                <text x={narrow ? 12 : cx0 + 4} y={y + (narrow ? 26 : 20)} class="mono">N {eng(r.rung.N)} · D {eng(r.rung.D)} · C {mant(r.rung.C)} × 10<tspan dy="-4" font-size="8">{expo(r.rung.C)}</tspan></text>
                <line x1={cx0} x2={x} y1={y + (narrow ? 40 : 34)} y2={y + (narrow ? 40 : 34)} class="stem {col}" />
                <circle cx={x} cy={y + (narrow ? 40 : 34)} r="5.5" class="dot {col}" />
              </g>
            {/if}
          {/each}
        {:else if fn && log}
          {#each [1.5, 2, 2.5, 3, 3.5] as g (g)}
            <line x1={gp.x0} x2={gp.x1} y1={gy(g)} y2={gy(g)} class="grid" />
            <text x={gp.x0 - 5} y={gy(g) + 4} text-anchor="end" class="small">{g}</text>
          {/each}
          {#each (narrow ? [0, 100, 1600, 6000] : [0, 25, 100, 400, 1600, 6000]) as s (s)}
            <text x={gx(s)} y={gp.y1 + 16} text-anchor="middle" class="small">{s}</text>
          {/each}
          <text x={(gp.x0 + gp.x1) / 2} y={gp.y1 + 32} text-anchor="middle" class="small">Training step (stretched near the start)</text>
          <text x="12" y={gp.y0 - 4 < 10 ? 10 : gp.y0 - 4} class="small">Validation loss</text>
          {#each [[log.baselines.uniformLnV, 'no knowledge (ln 38)'], [log.baselines.unigramValidCrossEntropy, 'letter frequencies'], [log.baselines.bigramValidCrossEntropy, 'letter pairs']] as [v, name] (name)}
            <line x1={gp.x0} x2={gp.x1} y1={gy(+v)} y2={gy(+v)} class="blind" />
            <text x={gp.x1} y={gy(+v) - 4} text-anchor="end" class="small">{name} {(+v).toFixed(2)}</text>
          {/each}
          <path d={curve} class="curvep" />
          {#each log.checkpoints.filter((c) => c.step <= fn.step) as c (c.step)}<circle cx={gx(c.step)} cy={gy(c.validLoss)} r="2.8" class="cp" />{/each}
          <circle cx={gx(fn.step)} cy={gy(fn.loss)} r="5" class="head-dot" />
          <text x={probeBox.x} y={probeBox.y - 4} class="head">Chance of the next character</text>
          {#each log.probes as pr, i (i)}
            {@const y = probeBox.y + 14 + i * (narrow ? 26 : 42)}
            <text x={probeBox.x} y={y + 2} class="mono">“{pr.context}” → {show(pr.next)}</text>
            <rect x={probeBox.x} y={y + 8} width={probeBox.w - 44} height="10" rx="2" class="rowbg" />
            <rect x={probeBox.x} y={y + 8} width={Math.max(0.5, (probeBox.w - 44) * fn.probes[i])} height="10" rx="2" class="keep" />
            <text x={probeBox.x + probeBox.w - 40} y={y + 17} class="mono">{pFmt(fn.probes[i])}</text>
          {/each}
          <text x="12" y={sampleY - 14} class="head">Sample saved at step {log.checkpoints[fn.cp].step}</text>
          {#each rowsOf(fn.sample, narrow ? 38 : 64) as ln, i (i)}
            <text x="12" y={sampleY + 6 + i * 18} class="mono">{ln}</text>
          {/each}
        {:else if fb}
          {#each rowsOf('Prompt: ' + prompt, narrow ? 40 : 60) as ln, i (i)}<text x="12" y={16 + i * 14} class="small">{ln}</text>{/each}
          {#each lanes as lane, li (li)}
            {@const per = narrow ? 24 : 48}
            {@const blockH = narrow ? 112 : 80}
            {@const y0 = (narrow ? 52 : 30) + li * blockH}
            <text x="12" y={y0 + 8} class="head">{lane.label}</text>
            {#each lane.ids.slice(0, fb.typed) as id, i (i)}
              {@const x = 12 + (i % per) * (narrow ? 11.5 : 14)}
              {@const y = y0 + 28 + Math.floor(i / per) * 50}
              {@const o = i === fb.typed - 1 ? fb.fade : 1}
              <g opacity={o}>
                <text x={x + 5} {y} text-anchor="middle" class="lab14 strong">{show(decode([id]))}</text>
                {#each lane.alts[i] as a, j (j)}
                  <text x={x + 5} y={y + 12 + j * 10} text-anchor="middle" class="tiny" opacity={0.2 + 0.8 * Math.min(1, a.p * 1.4)}>{show(decode([a.id]))}</text>
                {/each}
              </g>
            {/each}
          {/each}
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} markers={[]} label="Timeline" onSeek={seek} onReset={() => seek(0)} />
      <div class="segments" aria-label="Choose a timeline segment">
        {#each markers as m (m.at)}
          <button type="button" class:active={m === seg} onclick={() => { playing = false; seek(m.at); }}>{m.label}</button>
        {/each}
      </div>
    {/snippet}
    {#snippet readout()}
      <p class="status">{text}</p>
      <span class="sr" aria-live="polite">{seg?.label ?? ''}</span>
      {#if init === 'compute' && log && ready}
        <p class="side">The house model takes {log.tokensPerStep.toLocaleString('en-US')} tokens per optimizer step; GPT-3 used {eng(GPT3_BATCH_TOKENS)}.</p>
      {/if}
    {/snippet}
    {#snippet footer()}
      {#if init === 'data'}
        <label class="ctl">Near-copy threshold: Jaccard similarity {threshold.toFixed(2)} or more counts as a copy
          <input type="range" min="0.5" max="0.9" step="0.05" bind:value={threshold} aria-label="Near-copy threshold" />
        </label>
        <p class="note">The twelve documents were written for this figure, except the two public-domain openings (Carroll, Austen).</p>
      {:else if init === 'compute'}
        <label class="ctl">Your model’s parameters N: {eng(10 ** logN)}
          <input type="range" min="4" max="12" step="0.05" bind:value={logN} aria-label="Parameters N, log scale" />
        </label>
        <label class="ctl">Your model’s training tokens D: {eng(10 ** logD)}
          <input type="range" min="6" max="14" step="0.05" bind:value={logD} aria-label="Training tokens D, log scale" />
        </label>
      {:else if init === 'base'}
        <label class="ctl">Prompt (lowercase letters and punctuation, up to 40 characters)
          <input type="text" value={prompt} oninput={onPrompt} maxlength="40" aria-label="Prompt" />
        </label>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .small { fill: var(--text-muted); font: 11px var(--font-code); }
  .tiny { fill: var(--text-muted); font: 10px var(--font-code); }
  .mono { fill: var(--text); font: 11px var(--font-code); }
  .mono.fail { fill: var(--viz-neg); }
  .head { fill: var(--text); font: 600 12px var(--font-ui); }
  .lab14 { fill: var(--text); font: 14px var(--font-code); }
  .strong { font-weight: 700; }
  .accent { fill: var(--accent); font: 600 11px var(--font-code); }
  .pass { fill: var(--text-muted); font: 14px var(--font-code); opacity: 0.55; }
  .pass.read { fill: var(--text); opacity: 1; }
  .pass.next { fill: var(--accent); opacity: 1; font-weight: 700; }
  .target { fill: none; stroke: var(--accent); stroke-width: 2; }
  .halo { paint-order: stroke; stroke: var(--bg-raised); stroke-width: 3px; }
  .grid { stroke: var(--rule); stroke-dasharray: 3 3; }
  .blind { stroke: var(--text-muted); stroke-dasharray: 5 4; }
  .bar { fill: var(--viz-seq-3); }
  .bar.truebar { stroke: var(--accent); stroke-width: 2.5; }
  .col { fill: var(--viz-seq-4); }
  .col.big { fill: var(--viz-neg); }
  .mean { fill: none; stroke: var(--accent); stroke-width: 2.2; }
  .rowbg { fill: var(--bg-sunken); }
  .rowbg.on { stroke: var(--accent); stroke-width: 1.5; }
  .ok { fill: var(--viz-forward); }
  .no { fill: none; stroke: var(--viz-neg); stroke-width: 2; }
  .keep { fill: var(--viz-forward); }
  .stem { stroke-width: 2.5; }
  .dot.house, .stem.house { fill: var(--accent); stroke: var(--accent); }
  .dot.you, .stem.you { fill: var(--viz-pos); stroke: var(--viz-pos); }
  .dot.pub, .stem.pub { fill: var(--viz-forward); stroke: var(--viz-forward); }
  .curvep { fill: none; stroke: var(--viz-forward); stroke-width: 2.4; }
  .cp { fill: var(--accent); }
  .head-dot { fill: var(--accent); stroke: var(--bg-raised); stroke-width: 2; }
  .status { margin: 0; min-height: 3.4em; font-variant-numeric: tabular-nums; }
  .side { margin: 0.4rem 0 0; color: var(--text-muted); }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  .segments { display: flex; flex-wrap: wrap; gap: 0.35rem; padding: var(--s-2) var(--s-4); border-top: 1px solid var(--rule); }
  .segments button { font: 0.8rem var(--font-ui); padding: 0.35em 0.55em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .segments button.active { border-color: var(--accent); font-weight: 650; }
  .ctl { display: grid; gap: var(--s-2); padding: var(--s-3) var(--s-4) 0; }
  .ctl:last-of-type { padding-bottom: var(--s-3); }
  .ctl input[type='text'] { min-width: 0; width: 100%; box-sizing: border-box; padding: 0.6rem; font: 0.9rem var(--font-code); color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .note { margin: 0; padding: var(--s-2) var(--s-4) var(--s-3); color: var(--text-muted); }
  input[type='range'] { accent-color: var(--accent); }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
