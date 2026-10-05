<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels } from '../F1Shared/plot';
  import { NG, XS, TOL, SPIKE_AT, SUM_NS, LIM_NS, DEPTHS, durationOf, markersOf, frameAt, sumStages, firstWithin, shallowMin, deepUnits, type Act, type Mode, type StagedFrame } from './sim';

  interface Props {
    title?: string;
    /** bump: two units make a bump. sum: add many bumps. limits: a sharp feature. depth: tent-map composition. */
    mode?: Mode;
    /** Initial unit type for bump, sum and limits. */
    act?: Act;
  }
  let { title = 'Building a bump from two units', mode = 'bump', act: act0 = 'sigmoid' }: Props = $props();
  const init = untrack(() => ({ mode, title, act: act0 }));
  const M = init.mode, T = durationOf(M), MARKERS = markersOf(M);
  const uid = $props.id();

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let act = $state<Act>(init.act);
  let ui = $state.raw(frameAt(M, 0, init.act));
  let t = 0;
  let wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  $effect(() => { playing; speed; act; wake(); });

  const frame = () => frameAt(M, t, act);
  const sf = $derived(ui.kind === 'staged' ? ui : null);
  /** Tolerance facts for the limits figure, from the same stages the chart draws. */
  const tolFacts = $derived.by(() => {
    if (M !== 'limits') return null;
    const sm = firstWithin(sumStages('limitsSmooth', act)), sp = firstWithin(sumStages('limits', act));
    return { sm, sp };
  });

  const RANGE: Record<Mode, [number, number]> = { bump: [-0.08, 1.12], sum: [0, 1], limits: [0, 1.25], depth: [-0.05, 1.1] };
  const LEG: Record<Mode, { label: string; color: string; shape?: 'dot' | 'line' }[]> = {
    bump: [{ label: 'unit A', color: '--viz-pos', shape: 'line' }, { label: 'unit B', color: '--viz-neg', shape: 'line' }, { label: 'A − B (the bump)', color: '--viz-forward', shape: 'line' }],
    sum: [{ label: 'target', color: '--accent', shape: 'line' }, { label: 'sum of bumps', color: '--viz-forward', shape: 'line' }, { label: 'single bumps', color: '--viz-seq-3', shape: 'line' }],
    limits: [{ label: 'target with a spike', color: '--accent', shape: 'line' }, { label: 'sum of N bumps', color: '--viz-forward', shape: 'line' }],
    depth: [{ label: 'output of k tent layers', color: '--viz-forward', shape: 'line' }, { label: 'deep: 2k units', color: '--viz-forward' }, { label: 'one hidden layer: at least 2^k − 1', color: '--viz-neg' }],
  };
  const LABEL: Record<Mode, string> = {
    bump: 'Left to right: a single unit that steps up, a second shifted unit subtracted from it, and the resulting bump growing sharper.',
    sum: 'Left: a smooth target curve and a sum of narrow bumps whose number doubles over time, so the sum hugs the target. Right: the maximum error shrinking as the bump count grows.',
    limits: 'Left: a target with one narrow spike and a sum of bumps whose number grows; the spike is only captured by very narrow bumps. Right: maximum error against bump count for the smooth target and for the spiky one.',
    depth: 'Left: the output of one to five stacked tent layers, whose number of linear pieces doubles with each layer. Right: hidden units needed, which grow slowly for the deep network and exponentially for one hidden layer.',
  };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const lay = () => {
      const { main, side, stacked } = M === 'bump' ? { main: { x: 0, y: 0, w: stage.w, h: stage.h }, side: { x: 0, y: 0, w: 0, h: 0 }, stacked: false } : splitPanels(stage.w, stage.h, 0.58);
      const d = { l: main.x + 30, r: main.x + main.w - 8, t: main.y + 12, b: main.y + main.h - 24 };
      const s = { l: side.x + 44, r: side.x + side.w - 14, t: side.y + 28, b: side.y + side.h - 32 };
      const [y0, y1] = RANGE[M];
      return { main, side, stacked, d, s, dx: scale(0, 1, d.l, d.r), dy: scale(y0, y1, d.b, d.t) };
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frame(), { d } = L;
      ui = f; uiT = t;
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // axes
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [0, 0.5, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(x), L.dx(x), d.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of [0, 0.5, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(d.l, L.dy(y)); ctx.lineTo(d.r, L.dy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y), d.l - 6, L.dy(y)); }
      const path = (y: Float64Array, stride = 1) => {
        const p = new Path2D();
        for (let g = 0; g < NG; g += stride) { const X = L.dx(XS[g]), Y = L.dy(y[g]); if (g) p.lineTo(X, Y); else p.moveTo(X, Y); }
        return p;
      };
      ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, d.r - d.l, d.b - d.t); ctx.clip(); ctx.lineJoin = 'round';

      if (f.kind === 'bump') {
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
        for (const x of f.hasB ? [f.a, f.b] : [f.a]) if (x <= 1.0) { ctx.beginPath(); ctx.moveTo(L.dx(x), d.t); ctx.lineTo(L.dx(x), d.b); ctx.stroke(); }
        ctx.setLineDash([]);
        ctx.strokeStyle = c.pos.hex; ctx.lineWidth = 2; ctx.stroke(path(f.ya));
        if (f.yb) { ctx.strokeStyle = c.neg.hex; ctx.lineWidth = 2; ctx.stroke(path(f.yb)); }
        if (f.hasB) { ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3.5; ctx.stroke(path(f.diff)); }
      } else {
        const sfr = f as StagedFrame;
        if (sfr.mode !== 'depth') {
          ctx.lineWidth = 1.2; ctx.strokeStyle = c.s3.hex;
          ctx.globalAlpha = 0.9 * (1 - sfr.w); for (const b of sfr.bumpsA) ctx.stroke(path(b, 2));
          ctx.globalAlpha = 0.9 * sfr.w; for (const b of sfr.bumpsB) ctx.stroke(path(b, 2));
          ctx.globalAlpha = 1;
          if (sfr.target) { ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.8; ctx.stroke(path(sfr.target)); ctx.setLineDash([]); }
        } else {
          ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1;
          for (const k of sfr.kinks) { ctx.beginPath(); ctx.moveTo(L.dx(k), d.b); ctx.lineTo(L.dx(k), d.b - 7); ctx.stroke(); }
        }
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = sfr.mode === 'depth' ? 2.4 : 2.6;
        // The complete curve always agrees with the displayed count and error.
        // A brighter trace reveals the same curve continuously after each marker.
        ctx.globalAlpha = 0.4; ctx.stroke(path(sfr.curve)); ctx.globalAlpha = 1;
        if (reducedMotion() || !playing) ctx.stroke(path(sfr.curve));
        else {
          ctx.save(); ctx.beginPath(); ctx.rect(d.l, d.t, (d.r - d.l) * sfr.reveal, d.b - d.t); ctx.clip();
          ctx.stroke(path(sfr.curve)); ctx.restore();
        }
      }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x', d.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      if (f.kind === 'bump') {
        ctx.fillStyle = c.pos.hex; ctx.fillText('A', Math.min(d.r - 10, L.dx(f.a) + 4), d.t + 2);
        if (f.hasB && f.b <= 1) { ctx.fillStyle = c.neg.hex; ctx.fillText('B', L.dx(f.b) + 4, d.t + 2); }
      } else if (f.mode === 'depth') ctx.fillText(`${f.pieces} linear pieces`, d.l + 4, d.t + 2);
      else if (f.mode === 'limits') { ctx.fillStyle = c.accent.hex; ctx.textAlign = 'left'; ctx.fillText('narrow spike', Math.min(d.r - 78, L.dx(SPIKE_AT) + 10), d.t + 2); }
      else if (f.mode === 'sum') ctx.fillText(`${f.n} bump${f.n > 1 ? 's' : ''}, ${f.units} units`, d.l + 4, d.t + 2);

      if (M !== 'bump') side(ctx, c, L, f as StagedFrame);
    }

    /** Right-hand chart: max error against N (sum, limits) or hidden units against k (depth). */
    function side(ctx: CanvasRenderingContext2D, c: typeof stage.c, L: ReturnType<typeof lay>, f: StagedFrame) {
      const { s } = L, labels = M === 'sum' ? SUM_NS : M === 'limits' ? LIM_NS : DEPTHS, S = labels.length;
      const cx = (i: number) => s.l + 10 + (i * (s.r - s.l - 20)) / (S - 1);
      const series: { v: number[]; col: string; dash: number[]; name: string }[] = [];
      let lo = 0, hi = 1, log = true, title = '', ticks: [number, string][] = [];
      if (M === 'depth') {
        log = false; lo = 0; hi = 34; title = 'Hidden units needed vs layers k';
        series.push({ v: DEPTHS.map(deepUnits), col: c.fwd.hex, dash: [], name: 'deep: 2k' }, { v: DEPTHS.map(shallowMin), col: c.neg.hex, dash: [], name: 'one layer: ≥ 2^k − 1' });
        ticks = [[0, '0'], [10, '10'], [20, '20'], [30, '30']];
      } else {
        const a = sumStages(M, act).map((x) => x.maxErr), b = M === 'limits' ? sumStages('limitsSmooth', act).map((x) => x.maxErr) : [];
        const mn = Math.min(...a, ...b, TOL);
        lo = 10 ** Math.floor(Math.log10(mn * 0.8)); hi = 1; title = 'Max error vs number of bumps N (log scale)';
        if (M === 'limits') series.push({ v: b, col: c.pos.hex, dash: [], name: 'smooth target' }, { v: a, col: c.neg.hex, dash: [], name: 'with spike' });
        else series.push({ v: a, col: c.fwd.hex, dash: [], name: 'max error' });
        for (let e = Math.round(Math.log10(lo)); e <= 0; e++) ticks.push([10 ** e, e === 0 ? '1' : e === -1 ? '0.1' : '0.' + '0'.repeat(-e - 1) + '1']);
      }
      const yy = log ? scale(Math.log10(lo), Math.log10(hi), s.b, s.t) : scale(lo, hi, s.b, s.t), cy = (v: number) => (log ? yy(Math.log10(Math.max(v, lo))) : yy(v));
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(title, s.l - 40, L.side.y + 2);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      labels.forEach((n, i) => ctx.fillText(String(n), cx(i), s.b + 5));
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const [v, lb] of ticks) { ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(s.l, cy(v)); ctx.lineTo(s.r, cy(v)); ctx.stroke(); ctx.fillText(lb, s.l - 6, cy(v)); }
      ctx.font = stage.body(12, true); ctx.textBaseline = 'bottom'; ctx.fillText(M === 'depth' ? 'k' : 'N', s.r, L.side.y + L.side.h - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t - 4, s.r - s.l, s.b - s.t + 8); ctx.clip();
      if (M === 'limits') { ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(s.l, cy(TOL)); ctx.lineTo(s.r, cy(TOL)); ctx.stroke(); ctx.setLineDash([]); }
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx(f.s), s.t); ctx.lineTo(cx(f.s), s.b); ctx.stroke();
      for (const sr of series) {
        ctx.strokeStyle = sr.col; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
        sr.v.forEach((v, i) => (i ? ctx.lineTo(cx(i), cy(v)) : ctx.moveTo(cx(i), cy(v)))); ctx.stroke();
        ctx.fillStyle = sr.col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx(f.s), cy(sr.v[f.s]), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(s.l, s.t, s.r - s.l, s.b - s.t);
      ctx.font = stage.body(11.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      series.forEach((sr, i) => { ctx.fillStyle = sr.col; ctx.fillText(sr.name, s.l + 8, s.t + 6 + i * 15); });
      if (M === 'limits') { ctx.fillStyle = c.accent.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(`tolerance ${TOL}`, s.r - 6, cy(TOL) - 3); }
    }

    const loop = createLoop(cv, (dt) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });
</script>

<FigureCard title={init.title} role="group" aspect={M === 'bump' ? '720 / 340' : '720 / 400'} narrowAspect={M === 'bump' ? '360 / 300' : '360 / 620'} label={LABEL[M]} legend={LEG[M]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline"
      onSeek={(to) => { t = to; wake(); }} onReset={() => { t = 0; playing = !reducedMotion(); wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if ui.kind === 'bump'}
      <p class="vals num">steepness k = <b>{f2(ui.k, 1)}</b> · centre of A = <b>{f2(ui.a)}</b>{#if ui.hasB} · centre of B = <b>{f2(ui.b)}</b>{/if} · hidden units = <b>{ui.units}</b> · peak = <b>{f2(ui.peak)}</b></p>
    {:else if sf && sf.mode === 'sum'}
      <p class="vals num">N = <b>{sf.n}</b> bumps · hidden units = <b>{sf.units}</b> · sampled max error = <b>{f2(sf.maxErr, 3)}</b> · MSE = <b>{f2(sf.mse, 4)}</b></p>
    {:else if sf && sf.mode === 'limits'}
      <p class="vals num">N = <b>{sf.n}</b> bumps · hidden units = <b>{sf.units}</b> · sampled max error = <b>{f2(sf.maxErr, 3)}</b> (tolerance {TOL})</p>
      {#if tolFacts}<p class="vals num base">Within {TOL}: the smooth target from N = {tolFacts.sm?.n} ({tolFacts.sm?.units} units), the spiky one from N = {tolFacts.sp?.n} ({tolFacts.sp?.units} units).</p>{/if}
    {:else if sf}
      <p class="vals num">k = <b>{sf.n}</b> layers · hidden units = <b>{sf.units}</b> · linear pieces = <b>{sf.pieces}</b> · one hidden layer needs at least <b class="ne">{sf.shallow}</b> units</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{ui.phase}: {ui.say}</p>
  {/snippet}
  {#snippet footer()}
    {#if M === 'depth'}
      <p class="ctl num">One layer is two ReLUs: T(x) = 2·relu(x) − 4·relu(x − 1/2), a tent on [0, 1]. Stacking k of them uses 2k units in total.</p>
    {:else}
      <div class="ctl" role="radiogroup" aria-label="Unit type">
        <span class="nm">Units</span>
        <label class="chk"><input type="radio" name="{uid}-act" value="sigmoid" bind:group={act} /> Sigmoid</label>
        <label class="chk"><input type="radio" name="{uid}-act" value="relu" bind:group={act} /> ReLU (two per step)</label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .vals b.ne { color: var(--viz-neg); }
  .vals.base { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2em; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .nm { color: var(--text-muted); }
  .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text); font-size: 0.85rem; }
  .chk input { accent-color: var(--accent); }
</style>
