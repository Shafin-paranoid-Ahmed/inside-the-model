<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage } from '../F1Shared/canvas';
  import { scale, smooth } from '../F1Shared/plot';
  import { ROWS, HOODS, FEATURES, PRICE, T_ROW, oneHot, groups, duration, markers, frameAt, type Feature, type Hood } from './sim';

  interface Props {
    title?: string;
    /** Show the neighborhood category and its one-hot encoding, and colour the dots by neighborhood. */
    encode?: boolean;
    /** Show the neighborhood column only (used to check the groups found without labels). */
    hood?: boolean;
    /** Offer the "hide the labels" switch (supervised versus unlabeled data). */
    labelToggle?: boolean;
    /** Feature on the x-axis at first. */
    xFeature?: Feature;
  }
  let { title = 'Rows are examples, columns are features and one label', encode = false, hood = false, labelToggle = true, xFeature = 'area' }: Props = $props();
  const init = untrack(() => ({ encode, hood: hood || encode, labelToggle, x: xFeature }));
  const T = duration(), MK = markers(), N = ROWS.length, uid = $props.id();
  const GROUP = groups(), NG = Math.max(...GROUP) + 1;
  /** Neighborhood colours; the groups found without labels reuse them (newest group first). */
  const HOOD_VAR: Record<Hood, string> = { A: '--viz-cat-1', B: '--viz-cat-2', C: '--viz-cat-3' };
  const GROUP_KEY = ['cat3', 'cat2', 'cat1'] as const, GROUP_VAR = ['--viz-cat-3', '--viz-cat-2', '--viz-cat-1'];
  const HOOD_KEY = { A: 'cat1', B: 'cat2', C: 'cat3' } as const;
  const AGE = FEATURES.find((f) => f.key === 'age')!;

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let feat = $state<Feature>(init.x);
  let hideLabels = $state(false);
  /** True once the unlabeled view has settled and the groups are drawn. */
  let grouped = $state(false);
  let hover = $state(-1);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  /** Without labels the plot shows two features against each other: area across, age up. */
  const fx = $derived<Feature>(hideLabels ? 'area' : feat);
  const F = $derived(FEATURES.find((f) => f.key === fx)!);
  const scanRow = $derived(hover >= 0 ? hover : frameAt(uiT).active);
  const plotted = $derived(frameAt(uiT).arrive.filter((a) => a >= 1).length);
  $effect(() => { if (playing) hover = -1; });
  $effect(() => { fx; hideLabels; hover; speed; playing; wake(); });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const P = { l: 46, r: 16, t: 16, b: 44 };
    const xn = new Float64Array(N).fill(-1);   // tweened normalized x per point (-1 = not placed yet)
    let yMix = 1, gMix = 0, lastUi = 0;          // yMix 1 = price on y, 0 = age on y; gMix fades the groups in
    const geo = () => ({ x0: P.l, x1: stage.w - P.r, y0: stage.h - P.b, y1: P.t });
    const yAt = (i: number, g: ReturnType<typeof geo>) => {
      const sy = scale(PRICE.min, PRICE.max, g.y0, g.y1), sa = scale(AGE.min, AGE.max, g.y0, g.y1);
      return sa(ROWS[i].age) + (sy(ROWS[i].price) - sa(ROWS[i].age)) * yMix;
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, g = geo();
      const f = FEATURES.find((q) => q.key === fx)!;
      const sx = scale(0, 1, g.x0, g.x1), sxd = scale(f.min, f.max, g.x0, g.x1);
      const sy = scale(PRICE.min, PRICE.max, g.y0, g.y1), sa = scale(AGE.min, AGE.max, g.y0, g.y1);
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // x ticks
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const xs = f.key === 'bedrooms' ? [0, 1, 2, 3, 4, 5, 6] : f.key === 'age' ? [0, 10, 20, 30, 40] : [50, 100, 150];
      for (const v of xs) {
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(sxd(v), g.y1); ctx.lineTo(sxd(v), g.y0); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), sxd(v), g.y0 + 6);
      }
      // y ticks: price fades out and age fades in when the labels are hidden
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const [vals, s, a] of [[[100, 200, 300, 400], sy, yMix], [[0, 10, 20, 30, 40], sa, 1 - yMix]] as const) {
        if (a < 0.02) continue;
        ctx.globalAlpha = a;
        for (const v of vals) {
          ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(g.x0, s(v)); ctx.lineTo(g.x1, s(v)); ctx.stroke();
          ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), g.x0 - 6, s(v));
        }
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(g.x0, g.y0); ctx.lineTo(g.x1, g.y0); ctx.stroke();
      ctx.font = stage.body(12.5, true); ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
      ctx.fillStyle = c.fwd.hex; ctx.fillText(`feature: ${f.label.toLowerCase()}${f.unit ? ` (${f.unit})` : ''}`, g.x1, H - 3);
      ctx.save(); ctx.translate(11, g.y1); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      if (yMix > 0.5) { ctx.globalAlpha = yMix; ctx.fillStyle = c.accent.hex; ctx.fillText('label: price ($1000s)', 0, 0); }
      else { ctx.globalAlpha = 1 - yMix; ctx.fillStyle = c.fwd.hex; ctx.fillText('feature: age (yr)', 0, 0); }
      ctx.restore(); ctx.globalAlpha = 1;

      const fr = frameAt(t), act = hover >= 0 ? hover : fr.active;
      const settled = fr.arrive.every((a) => a >= 1);
      // one-hot step: a faint line through each neighborhood's houses (by area) shows the price offset per neighborhood
      if (init.encode && fx === 'area' && settled) {
        for (const h of HOODS) {
          const idx = ROWS.map((r, i) => (r.hood === h ? i : -1)).filter((i) => i >= 0).sort((a, b) => ROWS[a].area - ROWS[b].area);
          ctx.strokeStyle = c[HOOD_KEY[h]].hex; ctx.globalAlpha = 0.45; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.beginPath();
          idx.forEach((i, j) => { const X = xn[i] * (g.x1 - g.x0) + g.x0, Y = sy(ROWS[i].price); j ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
          ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
          const last = idx.at(-1)!;
          ctx.font = stage.mono(11, 600); ctx.fillStyle = c[HOOD_KEY[h]].hex; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(h, Math.min(g.x1 - 10, xn[last] * (g.x1 - g.x0) + g.x0 + 10), sy(ROWS[last].price));
        }
      }
      // unlabeled step: shade each group of nearby houses
      if (gMix > 0.01) {
        for (let k = 0; k < NG; k++) {
          const idx = ROWS.map((_, i) => i).filter((i) => GROUP[i] === k);
          const X = idx.map((i) => xn[i] * (g.x1 - g.x0) + g.x0), Y = idx.map((i) => yAt(i, g));
          const l = Math.min(...X) - 14, r = Math.max(...X) + 14, tp = Math.min(...Y) - 14, bt = Math.max(...Y) + 14;
          const col = c[GROUP_KEY[k]];
          ctx.globalAlpha = gMix; ctx.fillStyle = `rgba(${col.rgb.join(',')},0.12)`; ctx.strokeStyle = col.hex; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.roundRect(l, tp, r - l, bt - tp, 12); ctx.fill(); ctx.stroke();
          ctx.font = stage.body(12, true); ctx.fillStyle = col.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
          ctx.fillText(`group ${k + 1}`, r - 4, tp - 3); ctx.globalAlpha = 1;
        }
      }
      ROWS.forEach((r, i) => {
        const a = smooth(fr.arrive[i]);
        if (a <= 0) return;
        const tx = sx((r[fx] - f.min) / (f.max - f.min)), ty = yAt(i, g);
        const px = xn[i] < 0 ? tx : xn[i] * (g.x1 - g.x0) + g.x0;
        const ox = g.x0 - 30, oy = (g.y0 + g.y1) / 2;
        const X = fr.arrive[i] < 1 ? ox + (px - ox) * a : px, Y = fr.arrive[i] < 1 ? oy + (ty - oy) * a : ty;
        const on = i === act;
        const fill = on ? c.accent.hex : init.encode ? c[HOOD_KEY[r.hood]].hex : gMix > 0.5 ? c[GROUP_KEY[GROUP[i]]].hex : c.text.hex;
        ctx.globalAlpha = Math.min(1, a * 1.5);
        if (on) { ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X, Y, 10, 0, 6.2832); ctx.stroke(); }
        ctx.fillStyle = fill; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X, Y, on ? 6 : 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      // tween the plotted x positions, the price-to-age swap and the group shading toward their targets
      const f = FEATURES.find((q) => q.key === fx)!;
      let moving = false;
      const k = Math.min(1, dt * 7);
      const settled = frameAt(t).arrive.every((a) => a >= 1);
      const yT = hideLabels ? 0 : 1, gT = hideLabels && settled && yMix < 0.05 ? 1 : 0;
      if (Math.abs(yT - yMix) > 0.002) { yMix += (yT - yMix) * k; moving = true; } else yMix = yT;
      if (Math.abs(gT - gMix) > 0.002) { gMix += (gT - gMix) * k; moving = true; } else gMix = gT;
      ROWS.forEach((r, i) => {
        const tgt = (r[fx] - f.min) / (f.max - f.min);
        if (xn[i] < 0) xn[i] = tgt;
        else if (Math.abs(tgt - xn[i]) > 0.001) { xn[i] += (tgt - xn[i]) * k; moving = true; } else xn[i] = tgt;
      });
      if ((gMix > 0.5) !== grouped) grouped = gMix > 0.5;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing || moving;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; else if (first) { t = T; uiT = T; wake(); } } });
    wake = loop.wake;

    const near = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), g = geo(), f = FEATURES.find((q) => q.key === fx)!;
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = -1, bd = 18 * 18;
      ROWS.forEach((row, i) => {
        const X = g.x0 + ((row[fx] - f.min) / (f.max - f.min)) * (g.x1 - g.x0), Y = yAt(i, g);
        const d = (X - mx) ** 2 + (Y - my) ** 2;
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    };
    const move = (e: PointerEvent) => { const b = near(e); if (b !== hover) { hover = b; if (b >= 0) playing = false; } };
    const leave = () => { hover = -1; };
    cv.addEventListener('pointermove', move); cv.addEventListener('pointerleave', leave);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerleave', leave); };
  });
</script>

<FigureCard {title} role="group" aspect="720 / 440" narrowAspect="360 / 720"
  label={init.labelToggle ? 'A table of ten houses next to a plot of the same houses. With the labels hidden, the plot shows area against age and shades the groups of nearby houses.'
    : 'A table of ten houses next to a scatter plot of the same houses: the selected feature on the x-axis, the price label on the y-axis.'}
  legend={init.encode ? HOODS.map((h) => ({ label: `neighborhood ${h}`, color: HOOD_VAR[h] }))
    : [{ label: 'feature (input)', color: '--viz-forward' }, { label: 'label (answer)', color: '--accent' }]}>
  <div class="wrap">
    <div class="tbl">
      <table>
        <caption class="sr-only">Ten houses: features and the price label</caption>
        <thead>
          <tr>
            <th scope="col" class="idx">#</th>
            {#each FEATURES as f (f.key)}<th scope="col" class="feat">{f.label}{f.unit ? ` (${f.unit})` : ''}</th>{/each}
            {#if init.hood}<th scope="col" class="cat">Hood</th>{/if}
            {#if init.encode}
              {#each HOODS as h (h)}<th scope="col" class="feat hot" title="1 if the house is in neighborhood {h}, else 0">is {h}</th>{/each}
            {/if}
            <th scope="col" class="lab">Price ($k)</th>
          </tr>
        </thead>
        <tbody>
          {#each ROWS as r, i (r.id)}
            <tr class:on={i === scanRow} class:seen={i < plotted} tabindex="0" aria-label="House {r.id}"
              onpointerenter={() => { hover = i; playing = false; }} onpointerleave={() => (hover = -1)} onfocus={() => { hover = i; playing = false; }} onblur={() => (hover = -1)}>
              <td class="idx">{r.id}</td>
              {#each FEATURES as f (f.key)}<td class="feat" class:sel={f.key === fx || (hideLabels && f.key === 'age')}>{r[f.key]}</td>{/each}
              {#if init.hood}
                <td class="cat" class:tint={init.encode || grouped}
                  style:--tint={init.encode ? `var(${HOOD_VAR[r.hood]})` : `var(${GROUP_VAR[GROUP[i]]})`}>{r.hood}</td>
              {/if}
              {#if init.encode}
                {#each oneHot(r.hood) as v, k (k)}<td class="feat hot" class:one={v === 1}>{v}</td>{/each}
              {/if}
              <td class="lab">{hideLabels ? '?' : r.price}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <div class="plot"><canvas class="cv" bind:this={cv} aria-hidden="true"></canvas></div>
  </div>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: reading the table row by row"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">
      {#if hover >= 0}House {ROWS[hover].id}: {F.label.toLowerCase()} {ROWS[hover][fx]}{F.unit ? ` ${F.unit}` : ''}{hideLabels ? `, age ${ROWS[hover].age} yr, no label.` : `, price $${ROWS[hover].price}k.`}
      {:else if grouped}No prices, so no right answer to predict. Plotted by area and age, the houses fall into {NG} groups by age. Now compare the groups with the Hood column.
      {:else if hideLabels}The price is gone, so the plot swaps it for a second feature: age.
      {:else if init.encode && plotted === N && fx === 'area'}Color = neighborhood. At a similar area, C houses cost about $100k more than A houses. The one-hot columns are what let a model see that.
      {:else}{plotted} of {N} rows are plotted. Each row becomes one point.{/if}
    </p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <fieldset class="seg" disabled={hideLabels}>
        <legend class="lg">x-axis feature</legend>
        {#each FEATURES as f (f.key)}
          <label><input type="radio" name="xf-{uid}" value={f.key} bind:group={feat} /><span>{f.label}</span></label>
        {/each}
      </fieldset>
      {#if init.labelToggle}
        <label class="chk"><input type="checkbox" bind:checked={hideLabels}
          onchange={() => { if (hideLabels) { t = N * T_ROW; uiT = t; playing = false; } wake(); }} /> Hide the labels (no answers)</label>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  .wrap { position: absolute; inset: 0; display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: var(--s-3); }
  @container (width < 540px) { .wrap { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto minmax(0, 1fr); gap: var(--s-2); } }
  .tbl { overflow: auto; min-width: 0; }
  table { width: 100%; border-collapse: collapse; font: 0.8rem var(--font-code); font-variant-numeric: tabular-nums; }
  th, td { padding: 0.28em 0.5em; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
  th { font: 600 0.7rem var(--font-ui); color: var(--text-muted); vertical-align: bottom; }
  th.feat { color: var(--viz-forward); border-bottom: 2px solid var(--viz-forward); }
  th.lab { color: var(--accent); border-bottom: 2px solid var(--accent); }
  td.lab { color: var(--accent); font-weight: 650; }
  td.feat { color: var(--text); }
  td.feat.sel { background: color-mix(in srgb, var(--viz-forward) 12%, transparent); }
  td.hot { color: var(--text-muted); } td.hot.one { color: var(--viz-forward); font-weight: 650; }
  td.cat, th.cat { text-align: center; color: var(--text-muted); }
  td.cat.tint { color: var(--tint); font-weight: 700; background: color-mix(in srgb, var(--tint) 12%, transparent); }
  td.idx, th.idx { color: var(--text-muted); text-align: center; padding-inline: 0.3em; }
  tbody tr { opacity: 0.45; transition: opacity 0.25s; outline-offset: -2px; }
  tbody tr.seen, tbody tr.on { opacity: 1; }
  tbody tr.on { background: var(--accent-soft); }
  @container (width < 540px) { table { font-size: 0.68rem; } th, td { padding: 0.2em 0.3em; } th { font-size: 0.62rem; } }
  .plot { position: relative; min-width: 0; min-height: 0; }
  .wrap .plot :global(canvas.cv) { position: absolute; inset: 0; width: 100%; height: 100%; }
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.6em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .lg { float: left; margin-right: var(--s-3); padding: 0.3em 0; color: var(--text-muted); }
  .seg { display: inline-flex; align-items: center; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0 0 0 var(--s-3); margin: 0; overflow: hidden; }
  .seg:disabled { opacity: 0.5; }
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); cursor: pointer; }
  .chk input { accent-color: var(--accent); }
</style>
