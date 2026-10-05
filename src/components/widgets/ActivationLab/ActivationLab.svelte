<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    ACT, LEAK, SAT, HOLD, ZR, PANELS, SEGMENTS, WS, PRODUCT, PROBE, GRID_N, GRID_R, COLLAPSE_NAME,
    flatZones, geluDip, peakSlope, plotFrameAt, collapseFrameAt, collapseResult, collapseNarration, viewOf, duration, markers,
    type Mode, type PlotMode, type Collapse, type Mat2, type PanelSpec,
  } from './sim';

  interface Props {
    title?: string;
    /** saturation, relu-family, gelu, derivatives: a probe z sweeps along curves. collapse: a grid pushed through three layers. */
    mode?: Mode;
    /** collapse only: what sits between the layers at the start. */
    activation?: Collapse;
  }
  let { title = 'Activation functions', mode = 'saturation', activation = 'none' }: Props = $props();
  const init = untrack(() => ({ mode, activation }));
  const MODE = init.mode;
  const collapse = MODE === 'collapse';
  const PM = MODE as PlotMode;

  let act = $state<Collapse>(init.activation);
  let leak = $state(LEAK);
  const Tdur = duration(MODE), MK = markers(MODE);
  const DIP = geluDip();

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let values: HTMLParagraphElement;

  const pf = $derived(collapse ? null : plotFrameAt(uiT, PM, leak));
  const cf = $derived(collapse ? collapseFrameAt(uiT, act) : null);
  const res = $derived(collapseResult(act));
  const narration = $derived(pf ? MODE === 'relu-family' && pf.phase === 0 && leak === 0
    ? 'With leaky slope set to 0, both functions are ReLU: their negative side has output and slope 0.'
    : SEGMENTS[PM][pf.phase].say : collapseNarration(act, cf!.phase));

  // Choosing another activation replays the grid without overriding the learner's pause state.
  function chooseActivation(a: Collapse) { act = a; t = 0; uiT = 0; wake(); }
  $effect(() => { playing; speed; leak; wake(); });

  const fmt = (v: number) => f2(v);
  const sl = (d: number) => `slope ${f2(d)}${Math.abs(d) < SAT ? ', flat' : ''}`;
  const pt = (p: readonly number[]) => `(${f2(p[0])}, ${f2(p[1])})`;
  const mat = (M: Mat2) => `[${f2(M[0])} ${f2(M[1])}; ${f2(M[2])} ${f2(M[3])}]`;
  const same = $derived(Math.hypot(res.actual[0] - res.single[0], res.actual[1] - res.single[1]) < 1e-9);

  function valueText(time: number): string {
    if (collapse) {
      const f = collapseFrameAt(time, act);
      return `p = ${pt(PROBE)} → now at ${pt(f.probe)}\nfinal comparison: one matrix → ${pt(res.single)} · three layers → ${pt(res.actual)}: ${same ? 'identical' : 'different'}`;
    }
    const f = plotFrameAt(time, PM, leak);
    const entries = PANELS[PM].map((p, i) => `${ACT[p.main].name} ${fmt(f.panels[i].f)} (${p.main === 'step' && f.z === 0 ? 'slope undefined at jump' : sl(f.panels[i].d)})`);
    if (MODE === 'gelu') return `z = ${f2(f.z)} · GELU ${fmt(f.panels[0].f)} · ReLU ${fmt(f.panels[0].refs[0])} · SiLU ${fmt(f.panels[0].refs[1])}\nGELU minus ReLU = ${f2(f.panels[0].f - f.panels[0].refs[0])} · lowest GELU value ${f2(DIP.v)} at z = ${f2(DIP.z)}`;
    return `z = ${f2(f.z)} · ${entries.join(' · ')}`;
  }
  const initialValues = untrack(() => valueText(0));

  const flatCache = new Map<string, [number, number][]>();
  const flat = (id: PanelSpec['main'], zr: number, a: number) => {
    const key = `${id}|${zr}|${a}`;
    let v = flatCache.get(key);
    if (!v) { v = flatZones(id, zr, a); flatCache.set(key, v); }
    return v;
  };
  const tick = (lo: number, hi: number) => {
    const span = hi - lo, s = span <= 1.7 ? 0.5 : span <= 4.7 ? 1 : span <= 9 ? 2 : 4, out: number[] = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(Math.round(v * 100) / 100);
    return out;
  };
  const num = (v: number) => String(v).replace('-', '−');

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;

    // ---------- the plot modes
    function renderPlots(now: number) {
      const { ctx, w: W, h: H, c } = stage;
      ctx.clearRect(0, 0, W, H);
      const f = plotFrameAt(now, PM, leak), zr = ZR[PM], specs = PANELS[PM], narrow = W < 540;
      const cols = MODE === 'saturation' ? (narrow ? 1 : 3) : MODE === 'relu-family' ? (narrow ? 1 : 2) : MODE === 'gelu' ? 1 : narrow ? 1 : 2;
      const rows = Math.ceil(specs.length / cols), gap = narrow && MODE === 'derivatives' ? 6 : 12;
      const cw = (W - (cols - 1) * gap) / cols, ch = (H - (rows - 1) * gap) / rows;

      specs.forEach((sp, i) => {
        const bx = (i % cols) * (cw + gap), by = Math.floor(i / cols) * (ch + gap), pfm = f.panels[i];
        const twoLine = cw < 260 || (narrow && rows >= 3);
        const headH = twoLine && sp.deriv ? 32 : 20, xl = 28, bot = 15;
        const inX = bx + xl, inW = cw - xl - 8, inY = by + headH + 2, inH = ch - headH - 2 - bot;
        const hA = sp.deriv ? (inH - 12) * 0.58 : inH, hB = sp.deriv ? inH - 12 - hA : 0;
        const plots = [{ y: inY, h: hA, dom: sp.yDom, kind: 'f' as const }];
        if (sp.deriv) plots.push({ y: inY + hA + 12, h: hB, dom: sp.dDom!, kind: 'd' as const });
        // Six pixels keep the endpoint probe dots inside the clipping rectangle.
        const px = (z: number) => inX + 6 + ((z + zr) / (2 * zr)) * (inW - 12);
        const py = (p: (typeof plots)[number], v: number) => p.y + p.h - ((v - p.dom[0]) / (p.dom[1] - p.dom[0])) * p.h;

        // header: name, then the live values
        ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        ctx.font = `600 ${stage.body(13)}`; ctx.fillStyle = c.text.hex;
        const name = sp.main === 'leaky' ? `Leaky ReLU (slope ${f2(leak)})` : ACT[sp.main].name;
        ctx.fillText(name, bx + 2, by + 9);
        ctx.font = stage.mono(11, 600);
        const fl = `f ${fmt(pfm.f)}`, dl = `f′ ${fmt(pfm.d)}`;
        const valText = sp.deriv ? `${fl}  ${dl}` : fl;
        ctx.fillStyle = c.fwd.hex;
        if (twoLine && sp.deriv) { ctx.textAlign = 'left'; ctx.fillText(fl, bx + 2, by + 24); ctx.fillStyle = c.grad.hex; ctx.fillText(dl, bx + 2 + 62, by + 24); }
        else { ctx.textAlign = 'right'; ctx.fillText(sp.deriv ? fl : valText, bx + cw - (sp.deriv ? 70 : 2), by + 9); if (sp.deriv) { ctx.fillStyle = c.grad.hex; ctx.fillText(dl, bx + cw - 2, by + 9); } }

        for (const p of plots) {
          ctx.save();
          ctx.beginPath(); ctx.rect(inX, p.y, inW, p.h); ctx.clip();
          ctx.fillStyle = c.sunken.hex; ctx.fillRect(inX, p.y, inW, p.h);
          // grid + axes
          ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex; ctx.beginPath();
          for (const z of tick(-zr, zr)) { ctx.moveTo(px(z), p.y); ctx.lineTo(px(z), p.y + p.h); }
          for (const v of tick(p.dom[0], p.dom[1])) { ctx.moveTo(inX, py(p, v)); ctx.lineTo(inX + inW, py(p, v)); }
          ctx.stroke();
          ctx.strokeStyle = c.muted.hex; ctx.globalAlpha = 0.7; ctx.beginPath();
          ctx.moveTo(px(0), p.y); ctx.lineTo(px(0), p.y + p.h); ctx.moveTo(inX, py(p, 0)); ctx.lineTo(inX + inW, py(p, 0)); ctx.stroke(); ctx.globalAlpha = 1;
          // regions where the slope is about zero
          if (MODE !== 'gelu') for (const [a, b] of flat(sp.main, zr, leak)) {
            ctx.fillStyle = c.muted.hex; ctx.globalAlpha = 0.2; ctx.fillRect(px(a), p.y, px(b) - px(a), p.h); ctx.globalAlpha = 1;
            if (p.kind === 'f' && px(b) - px(a) > 62) { ctx.font = stage.body(11, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('slope ≈ 0', (px(a) + px(b)) / 2, p.y + 4); }
          }
          // curves
          const curve = (id: PanelSpec['main'], kind: 'f' | 'd', col: string, w: number, dash: number[] = []) => {
            const fn = kind === 'f' ? ACT[id].f : ACT[id].d;
            ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.lineJoin = 'round'; ctx.beginPath();
            if (id === 'step') {
              ctx.moveTo(px(-zr), py(p, fn(-1))); ctx.lineTo(px(0), py(p, fn(-1)));
              if (kind === 'f') { ctx.moveTo(px(0), py(p, 0)); ctx.lineTo(px(0), py(p, 1)); }
              ctx.moveTo(px(0), py(p, fn(1))); ctx.lineTo(px(zr), py(p, fn(1)));
            } else {
              const n = 240;
              for (let k = 0; k <= n; k++) { const z = -zr + (k / n) * 2 * zr, x = px(z), y = py(p, fn(z, leak)); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
            }
            ctx.stroke(); ctx.setLineDash([]);
          };
          if (p.kind === 'f') {
            for (const r of sp.refs ?? []) curve(r, 'f', r === 'relu' ? c.muted.hex : c.s4.hex, r === 'relu' ? 1.8 : 1.6, r === 'silu' ? [5, 4] : []);
            curve(sp.main, 'f', c.fwd.hex, 2.6);
          } else {
            const pk = peakSlope(sp.main, zr, leak);
            ctx.strokeStyle = c.grad.hex; ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(inX, py(p, pk)); ctx.lineTo(inX + inW, py(p, pk)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
            ctx.font = stage.mono(10, 600); ctx.fillStyle = c.grad.hex; ctx.textAlign = 'left';
            const above = py(p, pk) - 13 > p.y;
            ctx.textBaseline = above ? 'bottom' : 'top';
            ctx.fillText(`max ${f2(pk)}`, inX + 4, py(p, pk) + (above ? -2 : 3));
            curve(sp.main, 'd', c.grad.hex, 2.4);
          }
          // the GELU dip
          if (MODE === 'gelu' && p.kind === 'f') {
            const dx = px(DIP.z), dy = py(p, DIP.v);
            ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(dx, dy, 7, 0, 6.2832); ctx.stroke();
            ctx.font = stage.mono(11, 600); ctx.fillStyle = c.accent.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText(`dip ${f2(DIP.v)} at z = ${f2(DIP.z)}`, dx + 4, dy + 11);
          }
          // the probe
          const x = px(f.z);
          ctx.strokeStyle = c.text.hex; ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
          const dot = (v: number, col: string, r: number) => { ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, py(p, v), r, 0, 6.2832); ctx.fill(); ctx.stroke(); };
          if (p.kind === 'f') { (sp.refs ?? []).forEach((r, k) => dot(pfm.refs[k], r === 'relu' ? c.muted.hex : c.s4.hex, 4)); dot(pfm.f, c.fwd.hex, 5.5); }
          else dot(pfm.d, c.grad.hex, 5);
          ctx.restore();
          // tick labels
          ctx.font = stage.mono(10); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          for (const v of tick(p.dom[0], p.dom[1])) ctx.fillText(num(v), inX - 4, py(p, v));
          if (p === plots[plots.length - 1]) {
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            for (const z of tick(-zr, zr)) ctx.fillText(num(z), px(z), p.y + p.h + 3);
          }
        }
      });
    }

    // ---------- collapse: a grid through three layers
    function renderCollapse(now: number) {
      const { ctx, w: W, h: H, c } = stage;
      ctx.clearRect(0, 0, W, H);
      const f = collapseFrameAt(now, act), V = viewOf(act), u = Math.min(W, H) / (2 * V), cx = W / 2, cy = H / 2;
      const X = (v: number) => cx + v * u, Y = (v: number) => cy - v * u;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      ctx.fillStyle = c.sunken.hex; ctx.fillRect(0, 0, W, H);
      // the untouched grid, faint
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex; ctx.beginPath();
      for (let k = -Math.floor(V); k <= Math.floor(V); k++) { ctx.moveTo(X(k), 0); ctx.lineTo(X(k), H); ctx.moveTo(0, Y(k)); ctx.lineTo(W, Y(k)); }
      ctx.stroke();
      // the single-matrix outline
      if (f.ghost > 0.01) {
        const R = GRID_R, P = PRODUCT, cr: [number, number][] = [[-R, -R], [R, -R], [R, R], [-R, R]];
        ctx.globalAlpha = f.ghost; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.6; ctx.setLineDash([7, 5]); ctx.beginPath();
        cr.forEach(([a, b], i) => { const x = X(P[0] * a + P[1] * b), y = Y(P[2] * a + P[3] * b); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
        const s = res.single;
        ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(s[0]), Y(s[1]), 8, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
      }
      // the grid being transformed: 5 horizontal then 5 vertical lines
      const L = GRID_N + 1;
      for (let line = 0; line < 2 * (2 * GRID_R + 1); line++) {
        const k = (line % (2 * GRID_R + 1)) - GRID_R, horiz = line < 2 * GRID_R + 1;
        ctx.strokeStyle = horiz ? c.fwd.hex : c.s4.hex; ctx.lineWidth = k === 0 ? 2.8 : 1.6; ctx.globalAlpha = 0.95; ctx.beginPath();
        for (let i = 0; i < L; i++) { const j = (line * L + i) * 2, x = X(f.pts[j]), y = Y(f.pts[j + 1]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // the probe point
      ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(f.probe[0]), Y(f.probe[1]), 6, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.font = stage.body(13, true); ctx.fillStyle = c.accent.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('p', X(f.probe[0]) + 9, Y(f.probe[1]) - 5);
      ctx.restore();
      ctx.font = stage.mono(11, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = c.muted.hex;
      ctx.fillText('Grid through three layers', 8, 6);
      if (f.ghost > 0.3) { ctx.fillStyle = c.accent.hex; ctx.textAlign = W < 540 ? 'left' : 'right'; ctx.fillText('dashed: one matrix W3·W2·W1', W < 540 ? 8 : W - 8, W < 540 ? 22 : 6); }
    }

    // Numeric text is painted directly with the canvas; reactive narration only changes at phases.
    function render() {
      const v = Math.min(t, Tdur);
      if (collapse) renderCollapse(v); else renderPlots(v);
      // Keep Svelte's original static text node in place; this caption has no reactive children.
      if (values?.firstChild) values.firstChild.nodeValue = valueText(v);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: Tdur, loopFrom: 0, loopTo: Tdur + HOLD }).t;
      const v = Math.min(t, Tdur);
      const phase = collapse ? collapseFrameAt(v, act).phase : plotFrameAt(v, PM, leak).phase;
      const shownPhase = collapse ? cf!.phase : pf!.phase;
      if (now - lastUi > 100 || !playing || phase !== shownPhase) { lastUi = now; uiT = v; }
      render(); tl.paint(v);
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });

  const legend = $derived.by(() => {
    if (collapse) return [{ label: 'horizontal grid lines', color: '--viz-forward', shape: 'line' as const }, { label: 'vertical grid lines', color: '--viz-seq-4', shape: 'line' as const }, { label: 'point p', color: '--accent' }, { label: 'one matrix W3·W2·W1', color: '--accent', shape: 'line' as const }];
    if (MODE === 'gelu') return [{ label: 'GELU', color: '--viz-forward', shape: 'line' as const }, { label: 'ReLU', color: '--text-muted', shape: 'line' as const }, { label: 'SiLU (smooth sibling)', color: '--viz-seq-4', shape: 'line' as const }];
    if (MODE === 'derivatives') return [{ label: 'f(z)', color: '--viz-forward', shape: 'line' as const }, { label: 'slope f′(z)', color: '--viz-gradient', shape: 'line' as const }, { label: 'slope below 0.05', color: '--text-muted' }];
    return [{ label: 'f(z)', color: '--viz-forward', shape: 'line' as const }, { label: 'slope below 0.05', color: '--text-muted' }];
  });
  const figLabel = collapse
    ? 'A square grid of points pushed through three 2 by 2 weight matrices, with an optional activation after each. The timeline morphs the grid layer by layer.'
    : 'Activation function curves with a probe input z that sweeps from left to right along the timeline, and a dot riding each curve.';
  const timelineLabel = collapse ? 'Timeline: the grid, layer by layer' : `Timeline: z sweeps from −${ZR[PM]} to ${ZR[PM]}`;
  const aspect = collapse ? '16 / 9' : MODE === 'saturation' ? '16 / 7' : MODE === 'derivatives' ? '16 / 10' : '16 / 9';
  const narrowAspect = collapse ? '1 / 1' : MODE === 'saturation' ? '4 / 5' : MODE === 'derivatives' ? '2 / 5' : MODE === 'relu-family' ? '4 / 5' : '4 / 3';
  const CHOICES: Collapse[] = ['none', 'tanh', 'relu'];
</script>

{#snippet matrices()}
  {#if cf}
    <p class="aside-h">Fixed weights, no biases</p>
    <ul class="mats num">
      {#each WS as W, i (i)}
        <li class:on={cf.phase === i + 1}>W{i + 1} = {mat(W)}</li>
      {/each}
      <li class="prod">product = {mat(PRODUCT)}</li>
    </ul>
    <p class="aside-n">Between the layers: <b>{COLLAPSE_NAME[act]}</b>. Without an activation the product matrix is all the three layers amount to.</p>
  {/if}
{/snippet}

<FigureCard {title} role="group" {aspect} {narrowAspect} label={figLabel} {legend} aside={collapse ? matrices : undefined}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={Tdur} markers={MK} label={timelineLabel}
      onSeek={(to) => { t = to; uiT = to; wake(); }} onStep={(to) => { t = to; uiT = to; wake(); }} onReset={() => { t = 0; uiT = 0; playing = !reducedMotion(); wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num" bind:this={values}>{initialValues}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if collapse}
      <div class="ctl">
        <span class="nm">Between the layers</span>
        <span class="grp" role="group" aria-label="Activation between the layers">
          {#each CHOICES as a (a)}
            <button type="button" class="btn sm" aria-pressed={act === a} onclick={() => chooseActivation(a)}>{a === 'none' ? 'None' : COLLAPSE_NAME[a].replace('tanh', 'Tanh')}</button>
          {/each}
        </span>
      </div>
    {:else if MODE === 'relu-family'}
      <div class="ctl"><F1Slider label="Leaky slope" value={leak} min={0} max={0.5} step={0.01} oninput={(v) => (leak = v)} /></div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.6 var(--font-code); font-variant-numeric: tabular-nums; white-space: pre-line; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .nm { color: var(--text-muted); }
  .grp { display: inline-flex; flex-wrap: wrap; gap: var(--s-2); }
  .sm { padding: 0.3em 0.7em; font-size: var(--step--1); }
  .sm[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); color: var(--text); font-weight: 600; }
  .aside-h { font-weight: 600; margin-bottom: var(--s-2); }
  .aside-n { color: var(--text-muted); margin-top: var(--s-3); }
  .mats { list-style: none; margin: 0; padding: 0; font: 0.76rem / 1.9 var(--font-code); font-variant-numeric: tabular-nums; }
  .mats li { padding: 0 0.4em; border-left: 3px solid transparent; }
  .mats li.on { border-left-color: var(--accent); background: var(--accent-soft); }
  .mats li.prod { margin-top: var(--s-2); border-top: 1px solid var(--rule); padding-top: var(--s-1); color: var(--accent); font-weight: 600; }
</style>
