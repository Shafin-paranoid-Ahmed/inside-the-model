<script module lang="ts">
  import type { Answer, Experiment, Person } from './sim';

  // One worker per page: it loads the model once and answers every island on the page.
  let worker: Worker | undefined, nextId = 1;
  const pending = new Map<number, [(v: unknown) => void, (e: Error) => void]>();
  function rpc<T>(msg: object): Promise<T> {
    if (!worker) {
      const w = (worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }));
      w.onmessage = ({ data }) => {
        const p = pending.get(data.id);
        pending.delete(data.id);
        if (p) data.error ? p[1](new Error(data.error)) : p[0](data.data);
      };
      w.onerror = () => {
        for (const p of pending.values()) p[1](new Error('Could not load the model'));
        pending.clear(); w.terminate(); worker = undefined; experiment = undefined;
      };
    }
    return new Promise((resolve, reject) => { const id = nextId++; pending.set(id, [resolve as (v: unknown) => void, reject]); worker!.postMessage({ id, ...msg }); });
  }
  let experiment: Promise<{ people: Person[]; exp: Experiment }> | undefined;
  const shared = () => (experiment ??= rpc<{ people: Person[]; exp: Experiment }>({ kind: 'experiment' }));
  const askModel = (name: string) => rpc<Answer>({ kind: 'ask', name });
  const statedAnswers = (name: string) => rpc<Answer[]>({ kind: 'stated', name });
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, prefersReducedMotion, reducedMotion } from '../_shared';
  import { ABBR, DURATION, MARKERS, MAX_NAME, MONTHS, SAMPLE_COUNTS, pct, byCount, describe, frameAt, panelOf, pickHook, pickWorked, validName, wrongMonth, type Context, type Mode, type Panel, type Row } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'hook', title = 'A model answers about people' }: Props = $props();
  const init = untrack(() => mode);

  let data = $state.raw<{ people: Person[]; exp: Experiment; unseen: Row[] } | null>(null);
  let failed = $state(''), stated = $state.raw<Answer[] | null>(null), typed = $state.raw<Panel | null>(null);
  let typedText = $state(''), typedError = $state('');
  let unseenOnly = $state(false), k = $state(10), person = $state(-1), picked = $state(1), monthChoice = $state(-1);
  let width = $state(0), playing = $state(false), speed = $state(1), uiT = $state(0);
  const narrow = $derived(width > 0 && width < 540);
  const W = $derived(narrow ? 288 : 720);
  const duration = DURATION[init], markers = MARKERS[init];

  const hookRows = $derived(data ? pickHook(data.exp.rows) : null);
  const worked = $derived.by(() => {
    if (!data || !stated) return undefined;
    const row = pickWorked(data.unseen);
    return { row, stated, month: monthChoice < 0 ? wrongMonth(row.month) : monthChoice };
  });
  const ctx = $derived.by((): Context | null => {
    if (!data || !hookRows || (init === 'retrieval' && !worked)) return null;
    return { exp: data.exp, people: data.people, unseen: data.unseen, panels: [panelOf(hookRows[0]), typed ?? panelOf(hookRows[1])], unseenOnly, k, worked, person };
  });
  const frame = $derived(ctx ? frameAt(uiT, init, ctx) : null);
  const status = $derived(frame && ctx ? describe(frame, ctx) : failed || 'Loading the model and answering 200 questions...');
  const phase = $derived([...markers].reverse().find((m) => m.at <= uiT + 1e-9)?.label ?? '');

  type Rect = { x: number; y: number; w: number; h: number };
  const layout = $derived.by((): { H: number; a: Rect; b: Rect; c: Rect } => {
    const w = W, n = narrow, r = (x: number, y: number, ww: number, h: number): Rect => ({ x, y, w: ww, h });
    const none = r(0, 0, 0, 0);
    switch (init) {
      case 'hook': return n ? { H: 620, a: r(8, 8, 272, 296), b: r(8, 312, 272, 296), c: none } : { H: 316, a: r(12, 8, 336, 300), b: r(372, 8, 336, 300), c: none };
      case 'objective': return { H: n ? 320 : 330, a: r(0, 0, w, n ? 320 : 330), b: none, c: none };
      case 'frequency': return { H: n ? 330 : 310, a: r(0, 0, w, n ? 330 : 310), b: none, c: none };
      case 'calibration': return n ? { H: 540, a: r(8, 8, 272, 262), b: r(8, 278, 272, 254), c: none } : { H: 330, a: r(12, 8, 336, 314), b: r(372, 8, 336, 314), c: none };
      case 'abstain': return n ? { H: 520, a: r(8, 8, 272, 266), b: r(8, 282, 272, 230), c: none } : { H: 330, a: r(12, 8, 336, 314), b: r(372, 8, 336, 314), c: none };
      case 'retrieval': return n ? { H: 590, a: r(8, 8, 272, 300), b: r(8, 316, 272, 266), c: none } : { H: 340, a: r(12, 8, 336, 324), b: r(372, 8, 336, 324), c: none };
      default: return n ? { H: 600, a: r(8, 8, 272, 170), b: r(8, 186, 272, 170), c: r(8, 364, 272, 228) } : { H: 372, a: r(12, 8, 336, 170), b: r(372, 8, 336, 170), c: r(12, 186, 696, 176) };
    }
  });
  const wideH = (() => { const h: Record<Mode, number> = { hook: 316, objective: 330, frequency: 310, calibration: 330, abstain: 330, retrieval: 340, verify: 372 }; return h[init]; })();
  const narrowH = (() => { const h: Record<Mode, number> = { hook: 620, objective: 320, frequency: 330, calibration: 540, abstain: 520, retrieval: 590, verify: 600 }; return h[init]; })();

  const legend = {
    hook: [{ label: 'Probability of each month', color: '--viz-forward' }, { label: 'True month', color: '--accent', shape: 'line' as const }],
    objective: [{ label: 'Model average, unseen people', color: '--viz-forward' }, { label: 'Training frequency', color: '--viz-seq-3' }, { label: 'Chosen person', color: '--accent' }],
    frequency: [{ label: 'Share answered right', color: '--viz-forward' }, { label: 'Pure guessing, 1 in 12', color: '--text-muted', shape: 'line' as const }],
    calibration: [{ label: 'Share actually right', color: '--viz-forward' }, { label: 'Perfectly calibrated', color: '--text-muted', shape: 'line' as const }],
    abstain: [{ label: 'Answered, right', color: '--viz-cat-1' }, { label: 'Answered, wrong', color: '--viz-cat-2' }, { label: 'Not answered', color: '--text-muted' }],
    retrieval: [{ label: 'Probability of each month', color: '--viz-forward' }, { label: 'Month stated in the prompt', color: '--accent', shape: 'line' as const }],
    verify: [{ label: 'Samples', color: '--viz-forward' }, { label: 'Main answer right', color: '--viz-cat-1' }, { label: 'Main answer wrong', color: '--viz-cat-2' }],
  }[init];
  const label = {
    hook: 'Two questions about birth months and the probability the model gives to each month',
    objective: 'Average month probabilities for unseen people next to the month frequencies in training',
    frequency: 'Share of questions answered right, grouped by how often the fact appeared in training',
    calibration: 'Reliability diagram and histogram of the model confidence',
    abstain: 'Error rate against share of questions answered as the confidence threshold rises',
    retrieval: 'Probability of each month with and without a fact in the prompt, and accuracy for each case',
    verify: 'Repeated samples for two people and how often the main answer repeats, for all 200 people',
  }[init];

  // ---- small drawing helpers (geometry only; all numbers come from frameAt) ----
  const slot = (x: number, w: number, i: number, n: number) => ({ x: x + (i * w) / n, w: w / n });
  const niceTop = (v: number) => [0.2, 0.35, 0.5, 1].find((c) => c >= v * 1.05) ?? 1;
  const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
  const typeOut = (s: string, p: number) => s.slice(0, Math.round(p * s.length));

  let tl: ReturnType<typeof Timeline> | undefined, stage: HTMLDivElement;
  let t = 0, wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; narrow; unseenOnly; k; typed; monthChoice; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  const restart = (to = 0) => { seek(to); playing = !reducedMotion(); };

  function submitName(event: Event) {
    event.preventDefault();
    const name = typedText.trim().toLowerCase().replace(/\s+/g, ' ');
    if (!validName(name)) { typedError = `Use up to ${MAX_NAME} letters a to z, with single spaces.`; return; }
    typedError = '';
    askModel(name).then((answer) => {
      const row = data?.exp.rows.find((r) => r.name === name);
      typed = { name, count: row?.count ?? 0, truth: row && row.count > 0 ? row.month : undefined, answer };
      restart(0);
    }, (e: Error) => { typedError = e.message; });
  }
  const sortedBins = $derived(data ? byCount(data.exp.rows) : []);

  onMount(() => {
    shared().then((d) => {
      data = { ...d, unseen: d.exp.rows.filter((r) => r.count === 0) };
      typedText = pickHook(d.exp.rows)[1].name;
      if (init === 'retrieval') statedAnswers(pickWorked(data.unseen).name).then((a) => (stated = a), (e: Error) => (failed = e.message));
    }, (e: Error) => (failed = e.message));
    const loop = createLoop(stage.closest('[data-figure]') ?? stage, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration }).t;
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

{#snippet columns(x: number, y: number, w: number, h: number, series: { v: number[]; cls: string }[], top: number, hi: number, hiOpacity: number)}
  <line x1={x} x2={x + w} y1={y + h} y2={y + h} class="axis" />
  {#each ABBR as ab, i (ab)}
    {@const s = slot(x, w, i, 12)}
    {#each series as sr, j (j)}
      {@const b = slot(s.x + 2, s.w - 4, j, series.length)}
      {@const bh = Math.min(h, (sr.v[i] / top) * h)}
      <rect x={b.x} y={y + h - bh} width={Math.max(1, b.w - (series.length > 1 ? 1 : 0))} height={bh} class={sr.cls} />
    {/each}
    {#if i === hi}<rect x={s.x + 1} y={y - 2} width={s.w - 2} height={h + 4} class="mark" opacity={hiOpacity} />{/if}
    <text x={s.x + s.w / 2} y={y + h + 14} text-anchor="middle" class="small">{ab}</text>
  {/each}
{/snippet}

<div class="lab">
  <FigureCard {title} {label} aspect="720 / {wideH}" narrowAspect="288 / {narrowH}" {legend}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={width}>
      <svg viewBox="0 0 {W} {layout.H}" aria-hidden="true">
        {#if !frame || !ctx}
          <text x={W / 2} y={layout.H / 2} text-anchor="middle" class="small">{failed || 'Loading the model...'}</text>
        {:else if frame.mode === 'hook'}
          {#each ctx.panels as p, i (i)}
            {@const r = i === 0 ? layout.a : layout.b}
            {@const a = p.answer}
            {@const askText = `${p.name} was born in`}
            {@const reply = MONTHS[a.pick]}
            <text x={r.x} y={r.y + 16} class="heading">{i === 0 ? `Trained on: ${p.count} lines` : p.count > 0 ? `Trained on: ${p.count} lines` : 'Never seen in training'}</text>
            <text x={r.x} y={r.y + 40} class="code">{typeOut(askText, Math.min(1, frame.typing * 2))}</text>
            <text x={r.x} y={r.y + 66} class="answer">{frame.typing > 0.5 ? typeOut(reply, (frame.typing - 0.5) * 2) : ''}</text>
            {@render columns(r.x, r.y + 96, r.w, 140, [{ v: a.dist.map((q) => q * frame.reveal), cls: 'bar' }], 1, p.truth ?? -1, frame.mark)}
            {#if frame.showNumbers}
              <text x={r.x} y={r.y + 270} class="small">Top month: {reply}, {pct(a.conf)}</text>
              <text x={r.x} y={r.y + 286} class="small">{p.truth !== undefined && frame.showTruth ? `True month: ${MONTHS[p.truth]}` : p.truth === undefined ? 'No true month: not in the training text' : ''}</text>
            {/if}
          {/each}
        {:else if frame.mode === 'objective'}
          {@const r = layout.a}
          {@const chosen = person >= 0 ? ctx.unseen[person] : null}
          {@const top = niceTop(Math.max(...frame.average, ...frame.freq, ...(chosen?.dist ?? [0])))}
          {@const px = 46}
          {@const py = 22}
          {@const pw = r.w - px - 14}
          {@const ph = r.h - py - 60}
          {#each [0, top / 2, top] as g (g)}
            <line x1={px} x2={px + pw} y1={py + ph - (g / top) * ph} y2={py + ph - (g / top) * ph} class="grid" />
            <text x={px - 6} y={py + ph - (g / top) * ph + 4} text-anchor="end" class="small">{pct(g)}</text>
          {/each}
          {@render columns(px, py, pw, ph, [{ v: frame.average, cls: 'bar' }, { v: frame.freq, cls: 'bar2' }], top, -1, 0)}
          {#if chosen}
            {#each chosen.dist as q, i (i)}
              {@const s = slot(px, pw, i, 12)}
              <circle cx={s.x + s.w / 2} cy={py + ph - Math.min(1, q / top) * ph} r="4" class="dot" />
            {/each}
          {/if}
          <text x={px} y={r.h - 10} class="small">{chosen ? `${chosen.name}: top month ${MONTHS[chosen.pick]}, ${pct(chosen.conf)}` : `People fed in so far: ${frame.n} of ${ctx.unseen.length}`}</text>
        {:else if frame.mode === 'frequency'}
          {@const r = layout.a}
          {@const px = 46}
          {@const py = 22}
          {@const pw = r.w - px - 14}
          {@const ph = r.h - py - 66}
          {#each [0, 0.5, 1] as g (g)}
            <line x1={px} x2={px + pw} y1={py + ph - g * ph} y2={py + ph - g * ph} class="grid" />
            <text x={px - 6} y={py + ph - g * ph + 4} text-anchor="end" class="small">{pct(g)}</text>
          {/each}
          <line x1={px} x2={px + pw} y1={py + ph - ph / 12} y2={py + ph - ph / 12} class="chance" />
          {#each frame.bins as b, i (b.count)}
            {@const s = slot(px, pw, i, 6)}
            {@const bh = b.accuracy * ph}
            <rect x={s.x + s.w * 0.18} y={py + ph - bh} width={s.w * 0.64} height={bh} class="bar" class:picked={picked === i} />
            {#if b.checked}<text x={s.x + s.w / 2} y={py + ph - bh - 6} text-anchor="middle" class="value">{pct(b.accuracy, 1)}</text>{/if}
            <text x={s.x + s.w / 2} y={py + ph + 16} text-anchor="middle" class="small">{b.count}</text>
            <text x={s.x + s.w / 2} y={py + ph + 32} text-anchor="middle" class="small">{b.n} people</text>
          {/each}
          <text x={px + pw / 2} y={r.h - 6} text-anchor="middle" class="small">Times the person's line appears in training</text>
        {:else if frame.mode === 'calibration'}
          {@const a = layout.a}
          {@const b = layout.b}
          {@const px = a.x + 38}
          {@const py = a.y + 30}
          {@const pw = a.w - 50}
          {@const ph = a.h - 30 - 38}
          <text x={a.x} y={a.y + 16} class="heading">Share actually right, by confidence</text>
          {#each [0, 0.5, 1] as g (g)}
            <line x1={px} x2={px + pw} y1={py + ph - g * ph} y2={py + ph - g * ph} class="grid" />
            <text x={px - 6} y={py + ph - g * ph + 4} text-anchor="end" class="small">{pct(g)}</text>
            <text x={px + g * pw} y={py + ph + 16} text-anchor={g === 0 ? 'start' : g === 1 ? 'end' : 'middle'} class="small">{pct(g)}</text>
          {/each}
          <line x1={px} y1={py + ph} x2={px + pw} y2={py} class="chance" />
          {#if frame.n}
            <path d={frame.bins.filter((q) => q.n).map((q, i) => `${i ? 'L' : 'M'}${px + q.confidence * pw},${py + ph - q.accuracy * ph}`).join(' ')} class="curve" />
            {#each frame.bins as q, i (i)}
              {#if q.n}<circle cx={px + q.confidence * pw} cy={py + ph - q.accuracy * ph} r={3 + Math.sqrt(q.n) * 1.1} class="dot2" class:worst={i === frame.worst && frame.done} />{/if}
            {/each}
          {/if}
          <text x={px + pw / 2} y={a.y + a.h - 4} text-anchor="middle" class="small">Confidence in the chosen month</text>
          <text x={b.x} y={b.y + 16} class="heading">Answers per confidence bin</text>
          {@const hy = b.y + 30}
          {@const hh = b.h - 30 - 38}
          {@const hw = b.w - 50}
          {@const hx = b.x + 38}
          {@const top = Math.max(1, frame.maxBin)}
          <line x1={hx} x2={hx + hw} y1={hy + hh} y2={hy + hh} class="axis" />
          <text x={hx - 6} y={hy + 4} text-anchor="end" class="small">{top}</text>
          <text x={hx - 6} y={hy + hh + 4} text-anchor="end" class="small">0</text>
          {#each frame.bins as q, i (i)}
            {@const s = slot(hx, hw, i, 10)}
            <rect x={s.x + 2} y={hy + hh - (q.n / top) * hh} width={s.w - 4} height={(q.n / top) * hh} class="bar" />
          {/each}
          {#each [0, 0.5, 1] as g (g)}<text x={hx + g * hw} y={hy + hh + 16} text-anchor={g === 0 ? 'start' : g === 1 ? 'end' : 'middle'} class="small">{pct(g)}</text>{/each}
          <text x={hx + hw / 2} y={b.y + b.h - 4} text-anchor="middle" class="small">Confidence in the chosen month</text>
        {:else if frame.mode === 'abstain'}
          {@const a = layout.a}
          {@const b = layout.b}
          {@const px = a.x + 38}
          {@const py = a.y + 30}
          {@const pw = a.w - 50}
          {@const ph = a.h - 30 - 40}
          {@const cur = frame.curve[Math.round(frame.threshold * 100)]}
          <text x={a.x} y={a.y + 16} class="heading">Wrong answers among those given</text>
          {#each [0, 0.25, 0.5] as g (g)}
            <line x1={px} x2={px + pw} y1={py + ph - (g / 0.5) * ph} y2={py + ph - (g / 0.5) * ph} class="grid" />
            <text x={px - 6} y={py + ph - (g / 0.5) * ph + 4} text-anchor="end" class="small">{pct(g)}</text>
          {/each}
          {#each [0, 0.5, 1] as g (g)}<text x={px + g * pw} y={py + ph + 16} text-anchor={g === 0 ? 'start' : g === 1 ? 'end' : 'middle'} class="small">{pct(g)}</text>{/each}
          <path d={frame.curve.filter((q) => q.coverage > 0).map((q, i) => `${i ? 'L' : 'M'}${px + q.coverage * pw},${py + ph - (q.error / 0.5) * ph}`).join(' ')} class="curve" />
          {#if frame.answered}<circle cx={px + cur.coverage * pw} cy={py + ph - (cur.error / 0.5) * ph} r="6" class="cursor" />{/if}
          <text x={px + pw / 2} y={a.y + a.h - 4} text-anchor="middle" class="small">Share of questions answered</text>
          <text x={b.x} y={b.y + 16} class="heading">All 200 people, most confident first</text>
          {@const cols = 20}
          {@const cell = (b.w - 4) / cols}
          {#each frame.dots as d, i (i)}
            <circle cx={b.x + 2 + (i % cols) * cell + cell / 2} cy={b.y + 38 + Math.floor(i / cols) * cell + cell / 2} r={cell * 0.36} class={d === 0 ? 'right' : d === 1 ? 'wrong' : 'off'} />
          {/each}
        {:else if frame.mode === 'retrieval'}
          {@const a = layout.a}
          {@const b = layout.b}
          {@const w = ctx.worked!}
          <text x={a.x} y={a.y + 16} class="heading">One person the model never saw</text>
          <text x={a.x} y={a.y + 40} class="code" opacity={frame.seg === 0 ? 0 : frame.seg === 1 ? frame.local : 1}>{w.row.name} was born in {frame.stated === null ? '' : MONTHS[frame.stated]}.</text>
          <text x={a.x} y={a.y + 60} class="code">{w.row.name} was born in <tspan class="answer">{MONTHS[frame.answer.pick]}</tspan></text>
          {@render columns(a.x, a.y + 100, a.w, 150, [{ v: frame.shown, cls: 'bar' }], 1, frame.stated ?? -1, frame.seg === 0 ? 0 : 1)}
          <text x={a.x} y={a.y + 288} class="small">Answer: {MONTHS[frame.answer.pick]}, {pct(frame.answer.conf)} sure</text>
          <text x={b.x} y={b.y + 16} class="heading">Share right, all {frame.unseenN} unseen people</text>
          {@const px = b.x + 38}
          {@const py = b.y + 44}
          {@const pw = b.w - 50}
          {@const ph = b.h - 44 - (narrow ? 58 : 50)}
          {#each [0, 0.5, 1] as g (g)}
            <line x1={px} x2={px + pw} y1={py + ph - g * ph} y2={py + ph - g * ph} class="grid" />
            <text x={px - 6} y={py + ph - g * ph + 4} text-anchor="end" class="small">{pct(g)}</text>
          {/each}
          {#each ['No fact', 'True fact', 'Wrong fact'] as name, i (name)}
            {@const s = slot(px, pw, i, 3)}
            <rect x={s.x + s.w * 0.2} y={py + ph - frame.bars[i] * ph} width={s.w * 0.6} height={frame.bars[i] * ph} class="bar" />
            {#if i <= frame.seg}<text x={s.x + s.w / 2} y={py + ph - frame.bars[i] * ph - 6} text-anchor="middle" class="value">{pct(frame.acc[i])}</text>{/if}
            <text x={s.x + s.w / 2} y={py + ph + 16} text-anchor="middle" class="small">{name}</text>
            <text x={s.x + s.w / 2} y={py + ph + 32} text-anchor="middle" class="small">in the prompt</text>
          {/each}
        {:else}
          {@const c = layout.c}
          {#each frame.panels as p, i (i)}
            {@const r = i === 0 ? layout.a : layout.b}
            <text x={r.x} y={r.y + 16} class="heading">{trunc(`${p.panel.name}`, 22)}: {p.panel.count > 0 ? `${p.panel.count} lines` : 'never seen'}</text>
            {@render columns(r.x, r.y + 30, r.w, 100, [{ v: p.counts.map((n) => n / frame.k), cls: 'bar' }], 1, p.latest ?? -1, p.drawn && frame.n === 0 ? 1 : 0)}
            <text x={r.x} y={r.y + 164} class="small">{p.drawn} of {frame.k} drawn: {pct(p.agreement)} repeat {MONTHS[p.panel.answer.pick]}</text>
          {/each}
          {@const px = c.x + 38}
          {@const py = c.y + 30}
          {@const pw = c.w - 50}
          {@const ph = c.h - 30 - 40}
          <text x={c.x} y={c.y + 16} class="heading">All 200 people, by agreement</text>
          <line x1={px} x2={px + pw} y1={py + ph} y2={py + ph} class="axis" />
          <text x={px - 6} y={py + 4} text-anchor="end" class="small">{frame.maxLevel}</text>
          <text x={px - 6} y={py + ph + 4} text-anchor="end" class="small">0</text>
          {#each frame.levels as lv, i (i)}
            {@const s = slot(px, pw, i, frame.k + 1)}
            {@const hr = (lv.right / frame.maxLevel) * ph}
            {@const hw = (lv.wrong / frame.maxLevel) * ph}
            <rect x={s.x + 1.5} y={py + ph - hr} width={Math.max(1, s.w - 3)} height={hr} class="right" />
            <rect x={s.x + 1.5} y={py + ph - hr - hw} width={Math.max(1, s.w - 3)} height={hw} class="wrong" />
          {/each}
          <line x1={px + (Math.ceil(frame.k / 2) / (frame.k + 1)) * pw} x2={px + (Math.ceil(frame.k / 2) / (frame.k + 1)) * pw} y1={py - 4} y2={py + ph} class="chance" />
          {#each [0, 0.5, 1] as g (g)}<text x={px + ((g * frame.k + 0.5) / (frame.k + 1)) * pw} y={py + ph + 16} text-anchor="middle" class="small">{pct(g)}</text>{/each}
          <text x={px + pw / 2} y={c.y + c.h - 4} text-anchor="middle" class="small">Samples repeating the main answer (dashed: 50%)</text>
        {/if}
      </svg>
    </div>
    {#snippet controls()}
      <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} label="{title} timeline" onSeek={seek} onReset={() => seek(0)} />
    {/snippet}
    {#snippet readout()}
      <p class="status">{status}</p>
      <p class="sr" role="status" aria-live="polite">{phase}</p>
    {/snippet}
    {#snippet footer()}
      {#if init === 'hook'}
        <form class="field" onsubmit={submitName}>
          <label for="name-{init}">Ask about any name (letters and spaces)</label>
          <span class="row"><input id="name-{init}" type="text" bind:value={typedText} maxlength={MAX_NAME} autocomplete="off" spellcheck="false" /><button type="submit" class="btn">Ask</button></span>
          {#if typedError}<span class="err" role="alert">{typedError}</span>{/if}
        </form>
      {:else if init === 'objective' && data}
        <div class="field">
          <label for="pick-{init}">Compare one unseen person with the average</label>
          <select id="pick-{init}" bind:value={person} onchange={() => { playing = false; seek(10); }}>
            <option value={-1}>Nobody</option>
            {#each data.unseen as u, i (u.name)}<option value={i}>{u.name}</option>{/each}
          </select>
        </div>
      {:else if init === 'frequency' && data}
        <div class="field">
          <span id="ex-{init}">Examples from the group seen</span>
          <span class="row" role="group" aria-labelledby="ex-{init}">
            {#each sortedBins as b, i (b.count)}<button type="button" class="chip" class:on={picked === i} aria-pressed={picked === i} onclick={() => (picked = i)}>{b.count}</button>{/each}
            <span class="small-note">times</span>
          </span>
          <ul class="examples">
            {#each sortedBins[picked].examples as e (e.name)}
              <li><span class="mono">{e.name}</span>: model said {MONTHS[e.pick]}, true month {MONTHS[e.month]} <span class:ok={e.correct} class:no={!e.correct}>{e.correct ? 'right' : 'wrong'}</span></li>
            {/each}
          </ul>
        </div>
      {:else if init === 'calibration'}
        <label class="field check"><input type="checkbox" bind:checked={unseenOnly} onchange={() => restart(0)} /> Only people never seen in training</label>
      {:else if init === 'abstain'}
        <div class="field">
          <label for="thr-{init}">Answer only when confidence is at least <span class="mono">{frame && frame.mode === 'abstain' ? frame.threshold.toFixed(2) : '0.00'}</span></label>
          <input id="thr-{init}" class="range" type="range" min="0" max="1" step="0.01" value={frame && frame.mode === 'abstain' ? frame.threshold : 0} oninput={(e) => { playing = false; seek(+(e.currentTarget as HTMLInputElement).value * duration); }} />
        </div>
      {:else if init === 'retrieval' && worked}
        <div class="field">
          <label for="month-{init}">Month written in the third prompt</label>
          <select id="month-{init}" bind:value={monthChoice} onchange={() => restart(7.2)}>
            <option value={-1}>{MONTHS[wrongMonth(worked.row.month)]} (a wrong month)</option>
            {#each MONTHS as m, i (m)}<option value={i}>{m}{i === worked.row.month ? ' (true month)' : ''}</option>{/each}
          </select>
        </div>
      {:else if init === 'verify'}
        <div class="field">
          <label for="k-{init}">Samples per question</label>
          <select id="k-{init}" bind:value={k} onchange={() => restart(0)}>
            {#each SAMPLE_COUNTS as n (n)}<option value={n}>{n}</option>{/each}
          </select>
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  .heading { fill: var(--text); font: 600 13px var(--font-ui); }
  .small { fill: var(--text-muted); font: 11px var(--font-ui); font-variant-numeric: tabular-nums; }
  .value { fill: var(--text); font: 600 11px var(--font-ui); font-variant-numeric: tabular-nums; }
  .code { fill: var(--text); font: 12px var(--font-code); }
  .answer { fill: var(--accent); font: 600 17px var(--font-code); }
  .bar { fill: var(--viz-forward); }
  .bar.picked { stroke: var(--accent); stroke-width: 2; }
  .bar2 { fill: var(--viz-seq-3); }
  .mark { fill: none; stroke: var(--accent); stroke-width: 2; }
  .axis { stroke: var(--text-muted); }
  .grid { stroke: var(--rule); stroke-dasharray: 3 3; }
  .chance { stroke: var(--text-muted); stroke-dasharray: 5 4; fill: none; }
  .curve { fill: none; stroke: var(--viz-forward); stroke-width: 2; }
  .dot, .cursor { fill: var(--accent); stroke: var(--bg-raised); stroke-width: 1.5; }
  .dot2 { fill: var(--viz-forward); fill-opacity: 0.75; stroke: var(--bg-raised); }
  .dot2.worst { stroke: var(--accent); stroke-width: 2.5; }
  .right { fill: var(--viz-cat-1); }
  .wrong { fill: var(--viz-cat-2); }
  .off { fill: var(--rule); }
  .status { margin: 0; min-height: 3.6em; font-variant-numeric: tabular-nums; }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); margin: 0; }
  .field { display: grid; gap: var(--s-2); padding: var(--s-3) var(--s-4); font-size: 0.9rem; }
  .field.check { display: flex; align-items: center; gap: var(--s-2); cursor: pointer; }
  .row { display: flex; gap: var(--s-2); flex-wrap: wrap; align-items: center; }
  .field input[type='text'], .field select { min-width: 0; padding: 0.5rem 0.6rem; font: 0.9rem var(--font-code); color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .field input[type='text'] { flex: 1 1 12rem; }
  .range { accent-color: var(--accent); width: 100%; }
  .btn { padding: 0.45em 0.9em; }
  .chip { font: 0.85rem var(--font-ui); min-width: 2.4rem; padding: 0.3em 0.6em; background: var(--bg-raised); color: var(--text); border: 1px solid var(--rule); border-radius: var(--r-1); cursor: pointer; }
  .chip.on { border-color: var(--accent); font-weight: 650; }
  .small-note { color: var(--text-muted); }
  .examples { margin: 0; padding-left: 1.1rem; min-height: 4.2em; }
  .mono { font-family: var(--font-code); }
  .ok { color: var(--text); font-weight: 600; }
  .no { color: var(--text); font-weight: 600; text-decoration: underline; }
  .err { color: var(--text); }
  input:focus-visible, select:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
