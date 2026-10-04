<script lang="ts">
  // The generic Day 11 figure: loads one mode chunk, runs its scene on the shared timeline and renders the shapes
  // its draw() returns as SVG. GridWorldLab, BanditLab and PPOLab are thin wrappers around this. Contract: ./kit.ts.
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, markerAt, prefersReducedMotion, reducedMotion } from '../_shared';
  import { kit, type Control, type Frame, type ModeImpl, type ModeMeta, type Opts } from './kit';
  import type { Env, Shape } from './svg';

  type Impl = ModeImpl<unknown, Frame>;
  interface Props {
    /** The mode's lazy chunk (from the widget's import.meta.glob) and its eagerly bundled meta. */
    load?: () => Promise<{ default: Impl }>;
    meta?: ModeMeta;
    opts: Opts;
    title?: string;
    aspect?: string;
    narrowAspect?: string;
    /** Widget name, used in messages. */
    name: string;
  }
  const p: Props = $props();
  const init = untrack(() => ({ load: p.load, opts: p.opts, name: p.name, title: p.title, aspect: p.aspect, narrowAspect: p.narrowAspect, meta: p.meta }));
  const meta: ModeMeta = init.meta ?? { title: `${init.name}: unknown mode "${init.opts.mode}"`, legend: [] };
  const ctlDefs: Control[] = meta.controls ?? [];
  const title = init.title ?? meta.title;
  const aspect = init.aspect ?? meta.aspect ?? '720 / 420', narrowAspect = init.narrowAspect ?? meta.narrowAspect ?? '360 / 560';

  const first = (c: Control): unknown => {
    const v = init.opts[c.key];
    if (c.kind === 'range') { const x = Number(v); return Number.isFinite(x) && v !== undefined && v !== '' ? x : c.min; }
    if (c.kind === 'select') return c.options.find((x) => String(x.value) === String(v))?.value ?? c.options[0]?.value;
    return v === true || v === 'true';
  };
  let ctl = $state<Record<string, unknown>>(Object.fromEntries(ctlDefs.map((c) => [c.key, first(c)])));
  const o = $derived<Opts>({ ...init.opts, ...ctl });

  let impl = $state.raw<Impl | null>(null), scene = $state.raw<unknown>(null), failed = $state(false);
  let shapes = $state.raw<Shape[]>([]);
  let status = $state(''), detail = $state(''), readouts = $state.raw<[string, string][]>([]), live = $state(''), hover = $state('');
  let playing = $state(false), speed = $state(1), uiT = $state(0), W = $state(0), H = $state(0), pick = $state<string | null>(null);
  const narrow = $derived(W > 0 && W < 540);
  const tl = $derived(impl && scene ? impl.timeline(scene, o) : { duration: 1, markers: [] });
  const label = $derived(impl && scene ? impl.label(scene, o) : title);

  let tlc: ReturnType<typeof Timeline> | undefined, stage: HTMLDivElement | undefined;
  let t = 0, lastUi = 0, phase: string | null = null, started = false, after: 'restart' | 'end' | 'keep' | null = null;
  let wake = $state<() => void>(() => {});
  $effect(() => { playing; speed; W; H; scene; impl; pick; wake(); });

  // (Re)build the scene whenever the mode chunk arrives or a control changes the options.
  $effect(() => {
    const im = impl, oo = o;
    if (!im) return;
    let ok = true;
    Promise.resolve().then(() => im.setup(oo, kit)).then((s) => {
      if (!ok) return;
      scene = s;
      const d = im.timeline(s, oo).duration;
      if (!started) { started = true; t = reducedMotion() ? Math.min(d, im.restTime?.(s, oo) ?? d) : 0; }
      else if (after === 'restart') { t = 0; playing = !reducedMotion(); }
      else if (after === 'end') { playing = false; t = Math.min(d, im.restTime?.(s, oo) ?? d); }
      else t = Math.min(t, d);
      after = null; uiT = t; wake();
    }, () => { if (ok) failed = true; });
    return () => { ok = false; };
  });

  function paint() {
    if (!impl || !scene || W <= 0) return;
    const f = impl.frameAt(t, scene, o, { pick });
    const E: Env = { W, H, narrow, out: [] };
    impl.draw(E, f, scene, o);
    shapes = E.out;
    if (f.status !== status) status = f.status;
    if ((f.detail ?? '') !== detail) detail = f.detail ?? '';
    const ro = f.readouts ?? [];
    if (ro.length !== readouts.length || ro.some((r, i) => r[0] !== readouts[i][0] || r[1] !== readouts[i][1])) readouts = ro;
    const ph = markerAt(t, tl.markers)?.label ?? '';
    if (ph !== phase) { phase = ph; live = ph ? `${ph}: ${f.status}` : f.status; }
  }

  onMount(() => {
    let alive = true;
    if (init.load) init.load().then((m) => { if (alive) impl = m.default; }, () => { if (alive) failed = true; });
    else failed = true;
    const loop = createLoop(stage?.closest('[data-figure]') ?? stage!, (dt, now) => {
      if (!impl || !scene) return false;
      t = advance({ t: Math.min(t, tl.duration), playing, speed }, dt, { duration: tl.duration }).t;
      paint();
      tlc?.paint(t);
      if (!playing || now - lastUi > 100) { lastUi = now; uiT = t; }
      return playing;
    }, { onView: (_v, firstView) => { if (firstView && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const motion = prefersReducedMotion(), onMotion = () => { if (motion.matches) playing = false; };
    motion.addEventListener('change', onMotion);
    return () => { alive = false; loop.destroy(); motion.removeEventListener('change', onMotion); };
  });

  const seek = (to: number) => { t = Math.max(0, Math.min(to, tl.duration)); uiT = t; wake(); };
  function setCtl(c: Control, v: unknown) { after = c.then ?? 'restart'; ctl[c.key] = v; }
  const shown = (c: Control) => {
    const v = ctl[c.key];
    return c.kind === 'range' ? `${(v as number).toFixed(c.digits ?? 2).replace('-', '−')}${c.unit ?? ''}` : '';
  };
  function target(e: Event): Shape | undefined {
    const i = (e.target as Element | null)?.getAttribute?.('data-i');
    return i === null || i === undefined ? undefined : shapes[+i];
  }
  function onMove(e: PointerEvent) { const tip = target(e)?.tip ?? ''; if (tip !== hover) hover = tip; }
  function onClick(e: PointerEvent) { const id = target(e)?.pick; if (id !== undefined) { pick = pick === id ? null : id; playing = false; } }
  const lanes = meta.lanes ?? 1;
</script>

<div class="rl" class:lanes2={lanes === 2} class:lanes3={lanes === 3}>
  <FigureCard {title} {label} legend={meta.legend} {aspect} {narrowAspect} role={meta.pickable ? 'group' : 'img'}>
    <div class="stagebox" bind:this={stage} bind:clientWidth={W} bind:clientHeight={H}>
      {#if W > 0 && H > 0}
        <svg viewBox="0 0 {W} {H}" aria-hidden="true" onpointermove={onMove} onpointerleave={() => (hover = '')} onpointerdown={onClick} class:pickable={meta.pickable}>
          {#each shapes as s, i (i)}
            {#if s.k === 'rect'}
              <rect x={s.x} y={s.y} width={s.w} height={s.h} rx={s.rx} style:fill={s.fill ?? 'none'} style:stroke={s.stroke} stroke-width={s.sw} stroke-dasharray={s.dash} opacity={s.op} transform={s.tf} data-i={s.tip || s.pick ? i : undefined} />
            {:else if s.k === 'line'}
              <line x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} style:stroke={s.stroke} stroke-width={s.sw ?? 1} stroke-dasharray={s.dash} opacity={s.op} transform={s.tf} stroke-linecap="round" data-i={s.tip || s.pick ? i : undefined} />
            {:else if s.k === 'path'}
              <path d={s.d} style:fill={s.fill ?? 'none'} style:stroke={s.stroke} stroke-width={s.sw ?? 1.5} stroke-dasharray={s.dash} opacity={s.op} transform={s.tf} stroke-linejoin="round" stroke-linecap="round" data-i={s.tip || s.pick ? i : undefined} />
            {:else if s.k === 'circle'}
              <circle cx={s.cx} cy={s.cy} r={s.r} style:fill={s.fill ?? 'none'} style:stroke={s.stroke} stroke-width={s.sw} opacity={s.op} transform={s.tf} data-i={s.tip || s.pick ? i : undefined} />
            {:else if s.k === 'poly'}
              <polygon points={s.pts} style:fill={s.fill ?? 'none'} style:stroke={s.stroke} stroke-width={s.sw} stroke-linejoin="round" opacity={s.op} transform={s.tf} data-i={s.tip || s.pick ? i : undefined} />
            {:else if s.k === 'text'}
              <text x={s.x} y={s.y} text-anchor={s.anchor} dominant-baseline={s.base} font-size={s.size} font-weight={s.weight} class:mono={s.mono} class:halo={s.halo} style:fill={s.fill} opacity={s.op} transform={s.tf} data-i={s.tip || s.pick ? i : undefined}>{s.s}</text>
            {/if}
          {/each}
        </svg>
      {/if}
      {#if !scene}
        <p class="loading" role="status">{failed ? 'This figure could not be computed.' : meta.loading ?? 'Loading the figure…'}</p>
      {/if}
    </div>
    {#snippet controls()}
      <Timeline bind:this={tlc} bind:t={uiT} bind:playing bind:speed duration={tl.duration} markers={tl.markers} label="Timeline: {title}" onSeek={seek} onReset={() => seek(0)} />
    {/snippet}
    {#snippet readout()}
      <div class="note">
        <p class="status" aria-hidden="true">{status}</p>
        <p class="ro">{#each readouts as [k, v], i (i)}<span class="kv"><span class="k">{k}</span> <span class="v">{v}</span></span>{/each}</p>
        <p class="detail">{hover || detail}</p>
        <p class="sr" role="status" aria-live="polite">{live}</p>
      </div>
    {/snippet}
    {#snippet footer()}
      {#if ctlDefs.length}
        <div class="ctls">
          {#each ctlDefs as c (c.key)}
            {#if c.kind === 'range'}
              <label class="ctl">{c.label} <span class="v">{shown(c)}</span>
                <input type="range" min={c.min} max={c.max} step={c.step} value={ctl[c.key] as number} oninput={(e) => setCtl(c, +e.currentTarget.value)} />
              </label>
            {:else if c.kind === 'select'}
              <label class="ctl">{c.label}
                <select onchange={(e) => setCtl(c, c.options[e.currentTarget.selectedIndex].value)}>
                  {#each c.options as op (op.value)}<option value={String(op.value)} selected={String(op.value) === String(ctl[c.key])}>{op.label}</option>{/each}
                </select>
              </label>
            {:else}
              <label class="ctl"><input type="checkbox" checked={ctl[c.key] as boolean} onchange={(e) => setCtl(c, e.currentTarget.checked)} /> {c.label}</label>
            {/if}
          {/each}
        </div>
      {/if}
    {/snippet}
  </FigureCard>
</div>

<style>
  .stagebox { position: absolute; inset: 0; }
  svg { overflow: visible; }
  svg.pickable { cursor: pointer; }
  svg text { font-family: var(--font-ui); font-variant-numeric: tabular-nums; }
  svg text.mono { font-family: var(--font-code); }
  svg text.halo { paint-order: stroke; stroke: var(--bg-raised); stroke-width: 3px; stroke-linejoin: round; }
  .loading { position: absolute; inset: 0; display: grid; place-items: center; margin: 0; color: var(--text-muted); font: 0.85rem var(--font-ui); }
  .note { min-height: 5.2em; }
  .status { margin: 0; min-height: 3em; font-family: var(--font-body); font-size: 1rem; line-height: 1.5; font-variant-numeric: tabular-nums; }
  .ro { display: flex; flex-wrap: wrap; gap: 0.15rem 1.2rem; margin: var(--s-1) 0 0; min-height: 1.4em; font: 0.85rem var(--font-ui); font-variant-numeric: tabular-nums; }
  .kv .k { color: var(--text-muted); }
  .kv .v { font-weight: 600; }
  .detail { margin: var(--s-1) 0 0; min-height: 1.3em; color: var(--text-muted); font: 0.85rem var(--font-ui); font-variant-numeric: tabular-nums; }
  .sr { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .ctls { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-3); font: 0.9rem var(--font-ui); flex: 1 1 16rem; }
  .ctl .v { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 4.5ch; }
  .ctl input[type='range'] { accent-color: var(--accent); flex: 1 1 10rem; min-width: 8rem; }
  .ctl input[type='checkbox'] { accent-color: var(--accent); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; max-width: 100%; }
  input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .lanes2 :global(.track) { padding-bottom: 2.6rem; }
  .lanes2 :global(.ticks) { height: 2.5rem; }
  .lanes2 :global(.tick:nth-child(even)) { top: 1.25rem; }
  .lanes3 :global(.track) { padding-bottom: 3.8rem; }
  .lanes3 :global(.ticks) { height: 3.8rem; }
  .lanes3 :global(.tick:nth-child(3n + 2)) { top: 1.25rem; }
  .lanes3 :global(.tick:nth-child(3n + 3)) { top: 2.5rem; }
</style>
