<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage } from '../F1Shared/canvas';
  import { scale, smooth } from '../F1Shared/plot';
  import { ROWS, HOODS, FEATURES, PRICE, oneHot, duration, markers, frameAt, type Feature } from './sim';

  interface Props {
    title?: string;
    /** Show the neighborhood category and its one-hot encoding. */
    encode?: boolean;
    /** Offer the "hide the labels" switch (supervised versus unlabeled data). */
    labelToggle?: boolean;
    /** Feature on the x-axis at first. */
    xFeature?: Feature;
  }
  let { title = 'Rows are examples, columns are features and one label', encode = false, labelToggle = true, xFeature = 'area' }: Props = $props();
  const init = untrack(() => ({ encode, labelToggle, x: xFeature }));
  const T = duration(), MK = markers(), N = ROWS.length, uid = $props.id();

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let feat = $state<Feature>(init.x);
  let hideLabels = $state(false);
  let hover = $state(-1);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const F = $derived(FEATURES.find((f) => f.key === feat)!);
  const scanRow = $derived(hover >= 0 ? hover : frameAt(uiT).active);
  const plotted = $derived(frameAt(uiT).arrive.filter((a) => a >= 1).length);
  $effect(() => { if (playing) hover = -1; });
  $effect(() => { feat; hideLabels; hover; speed; playing; wake(); });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const P = { l: 46, r: 16, t: 16, b: 44 };
    const xn = new Float64Array(N).fill(-1);   // tweened normalized x per point (-1 = not placed yet)
    let yMix = 1, lastUi = 0;
    const geo = () => ({ x0: P.l, x1: stage.w - P.r, y0: stage.h - P.b, y1: P.t });

    function render() {
      const { ctx, w: W, h: H, c } = stage, g = geo();
      const f = FEATURES.find((q) => q.key === feat)!;
      const sx = scale(0, 1, g.x0, g.x1), sxd = scale(f.min, f.max, g.x0, g.x1);
      const sy = scale(PRICE.min, PRICE.max, g.y0, g.y1), sFlat = g.y0 - 10;
      const yOf = (price: number) => sFlat + (sy(price) - sFlat) * yMix;
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // x ticks
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const xs = f.key === 'bedrooms' ? [0, 1, 2, 3, 4, 5, 6] : f.key === 'age' ? [0, 10, 20, 30, 40] : [50, 100, 150];
      for (const v of xs) {
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(sxd(v), g.y1); ctx.lineTo(sxd(v), g.y0); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), sxd(v), g.y0 + 6);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [100, 200, 300, 400]) {
        ctx.strokeStyle = c.rule.hex; ctx.globalAlpha = yMix; ctx.beginPath(); ctx.moveTo(g.x0, sy(v)); ctx.lineTo(g.x1, sy(v)); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), g.x0 - 6, sy(v)); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(g.x0, g.y0); ctx.lineTo(g.x1, g.y0); ctx.stroke();
      ctx.font = stage.body(12.5, true); ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
      ctx.fillStyle = c.fwd.hex; ctx.fillText(`feature: ${f.label.toLowerCase()}${f.unit ? ` (${f.unit})` : ''}`, g.x1, H - 3);
      ctx.save(); ctx.translate(11, g.y1); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.globalAlpha = yMix; ctx.fillStyle = c.accent.hex; ctx.fillText('label: price ($1000s)', 0, 0); ctx.restore(); ctx.globalAlpha = 1;
      if (yMix < 0.6) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.globalAlpha = 1 - yMix; ctx.fillStyle = c.muted.hex; ctx.fillText('no labels: only positions along one feature', g.x0 + 4, g.y1 + 4); ctx.globalAlpha = 1; }

      const fr = frameAt(t), act = hover >= 0 ? hover : fr.active;
      ROWS.forEach((r, i) => {
        const a = smooth(fr.arrive[i]);
        if (a <= 0) return;
        const tx = sx((r[feat] - f.min) / (f.max - f.min)), ty = yOf(r.price);
        const px = xn[i] < 0 ? tx : xn[i] * (g.x1 - g.x0) + g.x0;
        const ox = g.x0 - 30, oy = (g.y0 + g.y1) / 2;
        const X = fr.arrive[i] < 1 ? ox + (px - ox) * a : px, Y = fr.arrive[i] < 1 ? oy + (ty - oy) * a : ty;
        const on = i === act;
        ctx.globalAlpha = Math.min(1, a * 1.5);
        if (on) { ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X, Y, 10, 0, 6.2832); ctx.stroke(); }
        ctx.fillStyle = on ? c.accent.hex : c.text.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X, Y, on ? 6 : 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      // tween the plotted x positions and the label collapse toward their targets
      const f = FEATURES.find((q) => q.key === feat)!;
      let moving = false;
      const k = Math.min(1, dt * 7);
      const yT = hideLabels ? 0 : 1;
      if (Math.abs(yT - yMix) > 0.002) { yMix += (yT - yMix) * k; moving = true; } else yMix = yT;
      ROWS.forEach((r, i) => {
        const tgt = (r[feat] - f.min) / (f.max - f.min);
        if (xn[i] < 0) xn[i] = tgt;
        else if (Math.abs(tgt - xn[i]) > 0.001) { xn[i] += (tgt - xn[i]) * k; moving = true; } else xn[i] = tgt;
      });
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing || moving;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; else if (first) { t = T; uiT = T; wake(); } } });
    wake = loop.wake;

    const near = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), g = geo(), f = FEATURES.find((q) => q.key === feat)!;
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = -1, bd = 18 * 18;
      ROWS.forEach((row, i) => {
        const X = g.x0 + ((row[feat] - f.min) / (f.max - f.min)) * (g.x1 - g.x0);
        const sy = scale(PRICE.min, PRICE.max, g.y0, g.y1)(row.price), Y = g.y0 - 10 + (sy - (g.y0 - 10)) * yMix;
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
  label="A table of ten houses next to a scatter plot of the same houses: the selected feature on the x-axis, the price label on the y-axis."
  legend={[{ label: 'feature (input)', color: '--viz-forward' }, { label: 'label (answer)', color: '--accent' }]}>
  <div class="wrap">
    <div class="tbl">
      <table>
        <caption class="sr-only">Ten houses: features and the price label</caption>
        <thead>
          <tr>
            <th scope="col" class="idx">#</th>
            {#each FEATURES as f (f.key)}<th scope="col" class="feat">{f.label}{f.unit ? ` (${f.unit})` : ''}</th>{/each}
            {#if init.encode}
              <th scope="col" class="cat">Hood</th>
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
              {#each FEATURES as f (f.key)}<td class="feat" class:sel={f.key === feat}>{r[f.key]}</td>{/each}
              {#if init.encode}
                <td class="cat">{r.hood}</td>
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
      {#if hover >= 0}House {ROWS[hover].id}: {F.label.toLowerCase()} {ROWS[hover][feat]}{F.unit ? ` ${F.unit}` : ''}{hideLabels ? ', no label.' : `, price $${ROWS[hover].price}k.`}
      {:else}{plotted} of {N} rows are plotted. Each row becomes one point.{/if}
    </p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <fieldset class="seg">
        <legend class="lg">x-axis feature</legend>
        {#each FEATURES as f (f.key)}
          <label><input type="radio" name="xf-{uid}" value={f.key} bind:group={feat} /><span>{f.label}</span></label>
        {/each}
      </fieldset>
      {#if init.labelToggle}
        <label class="chk"><input type="checkbox" bind:checked={hideLabels} /> Hide the labels (no answers)</label>
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
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); cursor: pointer; }
  .chk input { accent-color: var(--accent); }
</style>
