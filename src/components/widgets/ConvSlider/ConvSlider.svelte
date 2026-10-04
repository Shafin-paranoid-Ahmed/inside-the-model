<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import * as S from './sim';

  interface Props {
    title?: string;
    /** 'slide': one filter sweeping with the patch x filter = sum breakdown. 'filters': hand-made filters. 'sharing': same nine weights everywhere, parameter counts. 'learn': gradient descent finds the filter. */
    mode?: S.Mode;
  }
  let { title = '', mode = 'slide' }: Props = $props();
  const init = untrack(() => ({ mode, title }));
  const M = init.mode;

  let seed = $state(3);
  let tr = $state.raw<S.Training | null>(M === 'learn' ? S.train(3) : null);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let ui = $state<{ vals: { k: string; v: string }[]; say: string; fi: number }>({ vals: [], say: '', fi: 0 });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const T = M === 'slide' || M === 'sharing' ? S.T_SLIDE : M === 'filters' ? S.T_FILTERS : S.T_LEARN;
  const MK = M === 'slide' || M === 'sharing' ? S.SLIDE_MARKERS : M === 'filters' ? S.FILTER_MARKERS : S.LEARN_MARKERS;
  $effect(() => { playing; speed; wake(); });

  const seekTo = (to: number) => { t = to; playing = false; wake(); };
  function newStart() {
    seed = (seed % 5) + 1;
    tr = S.train(seed);
    t = 0; playing = !reducedMotion(); wake();
  }

  const PC = S.paramCounts();
  const fmt = (n: number) => n.toLocaleString('en-US');
  const wlabel = (w: number) => (Math.abs(w - 1 / 9) < 1e-9 ? '1/9' : f2(w, Number.isInteger(w) ? 0 : 1));
  const VERT = S.FILTERS[0];
  const VOUT = S.applyFilter(VERT.w);
  const VSC = S.outScale(VOUT);
  const FOUT = S.FILTERS.map((f) => S.applyFilter(f.w));

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, lastKey = -1;

    type RGB = [number, number, number];
    const mix = (a: RGB, b: RGB, u: number) => `rgb(${Math.round(a[0] + (b[0] - a[0]) * u)},${Math.round(a[1] + (b[1] - a[1]) * u)},${Math.round(a[2] + (b[2] - a[2]) * u)})`;
    const lum = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    /** Brightness 0..MAXV as grey: bright pixel = the lighter of the two theme inks. */
    const grey = (v: number) => {
      const { text, raised } = stage.c, dark = lum(text.rgb) < lum(raised.rgb) ? text.rgb : raised.rgb, light = dark === text.rgb ? raised.rgb : text.rgb;
      return mix(dark, light, v / S.MAXV);
    };
    const signed = (v: number, sc: number) => {
      const { raised, pos, neg } = stage.c, u = Math.min(1, Math.abs(v) / sc);
      return mix(raised.rgb, v >= 0 ? pos.rgb : neg.rgb, u * 0.92);
    };
    /** Ink colour that reads on a given fill. */
    const inkOn = (fill: string) => {
      const { text, raised } = stage.c, m = fill.match(/\d+/g)!.map(Number) as RGB, l = lum(m);
      return Math.abs(lum(text.rgb) - l) > Math.abs(lum(raised.rgb) - l) ? text.hex : raised.hex;
    };

    function label(txt: string, x: number, y: number, align: CanvasTextAlign = 'left', color = stage.c.text.hex) {
      const { ctx } = stage;
      ctx.font = stage.body(12.5); ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
      ctx.fillText(txt, x, y);
    }
    /** n x n grid of cells. `fill(i)` gives each fill, `text(i)` the number to print (or null). */
    function grid(x: number, y: number, cs: number, n: number, fill: (i: number) => string | null, text?: ((i: number) => string) | null, alpha?: (i: number) => number) {
      const { ctx, c } = stage;
      ctx.lineWidth = 1;
      for (let i = 0; i < n * n; i++) {
        const cx = x + (i % n) * cs, cy = y + Math.floor(i / n) * cs, f = fill(i), a = alpha ? alpha(i) : 1;
        ctx.fillStyle = c.sunken.hex; ctx.fillRect(cx, cy, cs, cs);
        if (f && a > 0) {
          ctx.globalAlpha = a; ctx.fillStyle = f; ctx.fillRect(cx, cy, cs, cs);
          if (text) {
            const s = text(i);
            let px = Math.min(cs * 0.44, 13);
            ctx.font = stage.mono(px, 600);
            while (px > 7 && ctx.measureText(s).width > cs - 3) { px -= 0.5; ctx.font = stage.mono(px, 600); }
            ctx.fillStyle = inkOn(f); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(s, cx + cs / 2, cy + cs / 2 + 0.5);
          }
          ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = c.rule.hex; ctx.strokeRect(cx + 0.5, cy + 0.5, cs - 1, cs - 1);
      }
    }
    const box = (x: number, y: number, w: number, h: number, lw = 2.6) => {
      const { ctx, c } = stage;
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = lw; ctx.strokeRect(x + lw / 2 - 0.5, y + lw / 2 - 0.5, w - lw + 1, h - lw + 1);
    };

    /** Two panels side by side, same cell size. */
    function top(W: number, H: number, nA: number, nB: number, belowCells: number, fixed: number) {
      const gap = Math.max(26, W * 0.05), cs = Math.floor(Math.min((W - 2 * 6 - gap) / (nA + nB), (H - 24 - fixed) / (Math.max(nA, nB) + belowCells)));
      const total = (nA + nB) * cs + gap, ax = Math.round((W - total) / 2);
      return { cs, ax, bx: ax + nA * cs + gap, y: 24, bottom: 24 + Math.max(nA, nB) * cs };
    }
    const numbers = (cs: number) => cs >= 22;

    /** Filled output map with a fading newest cell. */
    function outMap(L: ReturnType<typeof top>, out: ArrayLike<number>, sc: number, sw: S.Sweep, text = true) {
      grid(L.bx, L.y, L.cs, S.O, (i) => signed(out[i], sc), text && numbers(L.cs) ? (i) => String(Math.round(out[i] * 10) / 10).replace('-', '−') : null,
        (i) => (i < sw.idx || sw.done ? 1 : i === sw.idx ? sw.alpha : 0));
    }
    function inImage(L: ReturnType<typeof top>, sw: S.Sweep) {
      grid(L.ax, L.y, L.cs, S.N, (i) => grey(S.IMAGE[i]), numbers(L.cs) ? (i) => String(S.IMAGE[i]) : null);
      box(L.ax + sw.ox * L.cs, L.y + sw.oy * L.cs, 3 * L.cs, 3 * L.cs);
    }
    const outBox = (L: ReturnType<typeof top>, sw: S.Sweep) => box(L.bx + sw.ox * L.cs, L.y + sw.oy * L.cs, L.cs, L.cs, 2.4);

    /** Colour scale bar: negative through zero to positive. */
    function scaleBar(x: number, y: number, w: number, max: number) {
      const { ctx, c } = stage, h = 12, mid = x + w / 2;
      for (let i = 0; i < w; i++) { ctx.fillStyle = signed(((i + 0.5) / w - 0.5) * 2 * max, max); ctx.fillRect(x + i, y, 1, h); }
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillText(f2(-max, max % 1 ? 1 : 0), x, y + h + 4);
      ctx.textAlign = 'center'; ctx.fillText('0', mid, y + h + 4); ctx.textAlign = 'right'; ctx.fillText('+' + f2(max, max % 1 ? 1 : 0), x + w, y + h + 4);
    }

    function filterGrid(x: number, y: number, cs: number, w: ArrayLike<number>, tt?: (i: number) => string) {
      grid(x, y, cs, 3, (i) => signed(w[i], 1.5), tt ?? ((i) => wlabel(w[i])));
    }

    // ---------------------------------------------------------------- slide
    function renderSlide() {
      const { ctx, c, w: W, h: H } = stage, nb = W < 540 ? Math.floor((W - 12 - 40) / 9) : 0, L = top(W, H, S.N, S.O, nb ? 0 : 3.4, 34 + 34 + 34 + 3 * nb), sw = S.sweepAt(S.slideU(t));
      label(`Input image, ${S.N} × ${S.N}`, L.ax, L.y - 8); label(`Output map, ${S.O} × ${S.O}`, L.bx, L.y - 8);
      inImage(L, sw); outMap(L, VOUT, VSC, sw); outBox(L, sw);
      const bd = S.breakdownAt(sw.idx, VERT.w), by = L.bottom + 34, op = 20;
      const cs = nb || Math.round(L.cs * 1.13), gs = 3 * cs, bx0 = Math.round(Math.max(6, (W - (3 * gs + 2 * op + (W - 12 > 3 * gs + 2 * op + 180 ? 160 : 0))) / 2));
      const xs = [bx0, bx0 + gs + op, bx0 + 2 * gs + 2 * op];
      label('Patch', xs[0], by - 8); label('Filter', xs[1], by - 8); label('Products', xs[2], by - 8);
      grid(xs[0], by, cs, 3, (i) => grey(bd.patch[i]), (i) => String(bd.patch[i]));
      filterGrid(xs[1], by, cs, VERT.w, (i) => String(VERT.w[i]).replace('-', '−'));
      grid(xs[2], by, cs, 3, (i) => signed(bd.prods[i], S.MAXV), (i) => String(bd.prods[i]).replace('-', '−'));
      ctx.font = stage.body(20); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('×', xs[0] + gs + op / 2, by + gs / 2); ctx.fillText('=', xs[1] + gs + op / 2, by + gs / 2);
      const sumText = `sum = ${String(bd.sum).replace('-', '−')}`, sx = xs[2] + gs + 14;
      ctx.font = stage.mono(18, 700); ctx.fillStyle = c.text.hex;
      if (sx + ctx.measureText(sumText).width < W - 6) { ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(sumText, sx, by + gs / 2 - 10); label('goes into the boxed output cell', sx, by + gs / 2 + 18, 'left', c.muted.hex); }
      else { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(sumText, W / 2, by + gs + 8); }
    }

    // ---------------------------------------------------------------- filters
    function renderFilters() {
      const { w: W, h: H } = stage, nb = W < 540 ? 30 : 0, L = top(W, H, S.N, S.O, nb ? 0 : 3.4, 34 + 40 + 3 * nb), { fi, u } = S.filtersAt(t), sw = S.sweepAt(u), f = S.FILTERS[fi];
      label(`Input image, ${S.N} × ${S.N}`, L.ax, L.y - 8); label(`Output map: ${f.name.toLowerCase()}`, L.bx, L.y - 8);
      inImage(L, sw); outMap(L, FOUT[fi], S.outScale(FOUT[fi]), sw); outBox(L, sw);
      const by = L.bottom + 34, cs = nb || Math.round(L.cs * 1.13);
      label(`Filter: ${f.name.toLowerCase()}`, L.ax, by - 8);
      filterGrid(L.ax, by, cs, f.w);
      const sc = S.outScale(FOUT[fi]), bw = Math.min(L.cs * S.O, 220);
      label('Output colour scale', L.bx, by - 8); scaleBar(L.bx, by + 2, bw, Math.round(sc * 10) / 10);
    }

    // ---------------------------------------------------------------- sharing
    function renderSharing() {
      const { ctx, c, w: W, h: H } = stage, nb = W < 540 ? 26 : 0, L = top(W, H, S.N, S.O, nb ? 0 : 3.4, 34 + 40 + 3 * nb), sw = S.sweepAt(S.slideU(t));
      label(`Input image, ${S.N} × ${S.N}`, L.ax, L.y - 8); label(`Output map, ${S.O} × ${S.O}`, L.bx, L.y - 8);
      inImage(L, sw); outMap(L, VOUT, VSC, sw, false); outBox(L, sw);
      const by = L.bottom + 34, cs = nb || Math.round(L.cs * 1.13);
      label('Filter: 9 weights', L.ax, by - 8);
      filterGrid(L.ax, by, cs, VERT.w, (i) => String(VERT.w[i]).replace('-', '−'));
      // log-scale bars: weights to learn
      const x0 = Math.max(L.ax + 3 * cs + 24, W * 0.4), x1 = W - 12, sx = scale(0, 5, x0, x1), bh = Math.max(14, Math.min(24, Math.round(cs * 0.7))), y0 = by + 6;
      label(W < 540 ? 'Parameters (log scale)' : 'Weights and biases to learn (log scale)', x0, by - 8);
      const bars: [string, number, string][] = [['Convolution', PC.conv, c.fwd.hex], ['Dense layer', PC.dense, c.grad.hex]];
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let d = 0; d <= 5; d++) {
        const X = sx(d); ctx.beginPath(); ctx.moveTo(X, y0 - 2); ctx.lineTo(X, y0 + 2 * bh + 22); ctx.stroke();
        ctx.fillText(d === 0 ? '1' : d < 3 ? String(10 ** d) : `${10 ** (d - 3)}k`, X, y0 + 2 * bh + 24);
      }
      bars.forEach(([nm, v, col], i) => {
        const y = y0 + i * (bh + 10), X = sx(Math.log10(v));
        ctx.fillStyle = col; ctx.fillRect(x0, y, X - x0, bh);
        ctx.font = stage.mono(12, 650); ctx.textBaseline = 'middle';
        const txt = `${nm} ${fmt(v)}`;
        if (ctx.measureText(txt).width + 12 < X - x0) { ctx.fillStyle = inkOn(col); ctx.textAlign = 'right'; ctx.fillText(txt, X - 6, y + bh / 2 + 0.5); }
        else { ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.fillText(txt, X + 6, y + bh / 2 + 0.5); }
      });
    }

    // ---------------------------------------------------------------- learn
    function renderLearn(R: S.Training) {
      const { ctx, c, w: W, h: H } = stage, gcs = Math.floor(Math.min(46, (W - 12 - 24) / 11)), L = top(W, H, S.O, S.O, 0, 34 + 30 + 3 * gcs), s = S.learnStep(t);
      label(W < 540 ? 'Target map' : 'Target: the vertical-edge map', L.ax, L.y - 8); label(W < 540 ? 'Current map' : 'The learning filter’s map', L.bx, L.y - 8);
      const sc = S.outScale(R.target);
      grid(L.ax, L.y, L.cs, S.O, (i) => signed(R.target[i], sc), numbers(L.cs) ? (i) => String(Math.round(R.target[i])).replace('-', '−') : null);
      grid(L.bx, L.y, L.cs, S.O, (i) => signed(R.out[s][i], sc), numbers(L.cs) ? (i) => String(Math.round(R.out[s][i])).replace('-', '−') : null);
      const by = L.bottom + 34, cs = gcs, dec = cs < 34 ? 1 : 2;
      label('Filter now', L.ax, by - 8);
      grid(L.ax, by, cs, 3, (i) => signed(R.w[s][i], 1.5), (i) => f2(R.w[s][i], dec));
      const tx = L.ax + 3 * cs + 14;
      label('Target', tx, by - 8);
      grid(tx, by, cs, 3, (i) => signed(S.TARGET.w[i], 1.5), (i) => String(S.TARGET.w[i]).replace('-', '−'));
      // loss on a log axis
      const px0 = tx + 3 * cs + 52, px1 = W - 12, py0 = by + 2, py1 = by + 3 * cs - 6;
      if (px1 - px0 > 60) {
        label('Loss (log scale)', px0 - 40, by - 8);
        const lo = 10 ** Math.floor(Math.log10(Math.min(...R.loss))), hi = 10 ** Math.ceil(Math.log10(Math.max(...R.loss)));
        const X = scale(0, S.STEPS, px0, px1), Y = scale(Math.log10(lo), Math.log10(hi), py1, py0);
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px0 + 0.5, py0 + 0.5, px1 - px0, py1 - py0);
        ctx.font = stage.mono(10.5); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let d = Math.log10(lo); d <= Math.log10(hi); d++) {
          ctx.fillText(d >= 0 ? String(10 ** d) : String(10 ** d), px0 - 4, Y(d));
          if (d > Math.log10(lo) && d < Math.log10(hi)) { ctx.beginPath(); ctx.moveTo(px0, Y(d)); ctx.lineTo(px1, Y(d)); ctx.stroke(); }
        }
        ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('0', X(0), py1 + 4); ctx.fillText(String(S.STEPS), X(S.STEPS), py1 + 4);
        ctx.fillText('step', (px0 + px1) / 2, py1 + 4);
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let k = 0; k <= s; k++) { const a = X(k), b = Y(Math.log10(R.loss[k])); if (k) ctx.lineTo(a, b); else ctx.moveTo(a, b); }
        ctx.stroke();
        ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(s), Y(Math.log10(R.loss[s])), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
    }

    function render() {
      stage.ctx.clearRect(0, 0, stage.w, stage.h);
      if (M === 'slide') renderSlide(); else if (M === 'filters') renderFilters(); else if (M === 'sharing') renderSharing(); else if (tr) renderLearn(tr);
    }

    const read = () => {
      const vals: { k: string; v: string }[] = [];
      let say = '', fi = ui.fi;
      if (M === 'slide') {
        const sw = S.sweepAt(S.slideU(t)), bd = S.breakdownAt(sw.idx, VERT.w);
        vals.push({ k: 'window at row', v: `${sw.oy + 1}, column ${sw.ox + 1}` }, { k: 'sum', v: f2(bd.sum, 0) });
        say = bd.sum > 0 ? `Positive: the right column of this patch is brighter than its left column, so the filter reports a rising edge here.`
          : bd.sum < 0 ? `Negative: the left column is brighter than the right, a falling edge.`
            : `Zero: the left and right columns match, so there is no vertical edge under the window.`;
      } else if (M === 'filters') {
        const a = S.filtersAt(t); fi = a.fi;
        const f = S.FILTERS[fi], o = FOUT[fi];
        vals.push({ k: 'filter', v: f.name }, { k: 'largest output', v: f2(Math.max(...o), 1) }, { k: 'smallest', v: f2(Math.min(...o), 1) });
        say = `This filter responds to ${f.finds}.`;
      } else if (M === 'sharing') {
        const sw = S.sweepAt(S.slideU(t)), n = sw.idx + 1;
        vals.push({ k: 'positions visited', v: `${n} of ${S.NPOS}` }, { k: 'multiplications', v: fmt(n * 9) }, { k: 'distinct weights', v: '9' });
        say = `A dense layer from the same ${PC.inputs} pixels to the same ${PC.outputs} outputs has ${fmt(PC.denseWeights)} weights plus ${PC.outputs} biases. The convolution has ${PC.convWeights} weights plus 1 bias.`;
      } else if (tr) {
        const s = S.learnStep(t);
        vals.push({ k: 'step', v: `${s} of ${S.STEPS}` }, { k: 'loss', v: tr.loss[s] >= 10 ? f2(tr.loss[s], 1) : tr.loss[s] >= 0.1 ? f2(tr.loss[s], 2) : f2(tr.loss[s], 3) }, { k: 'bias', v: f2(tr.b[s], 2) });
        say = s < 20 ? 'The random filter starts far from the edge filter, so the loss is large and the first steps are big; the map is already taking shape.'
          : s < 120 ? 'The map is already close to the target; the weights are still settling toward the exact edge filter, so the loss keeps falling slowly.'
            : 'The nine weights have settled close to the vertical-edge filter and the two maps look the same.';
      }
      ui = { vals, say, fi };
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      const key = M === 'learn' ? S.learnStep(t) : M === 'filters' ? S.filtersAt(t).fi : S.sweepAt(S.slideU(t)).idx;
      if (now - lastUi > 100 || !playing || key !== lastKey) { lastUi = now; lastKey = key; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    read();
    return () => { loop.destroy(); stage.destroy(); };
  });

  const cfg = M === 'learn' ? { aspect: '720 / 520', narrow: '360 / 360' } : M === 'filters' ? { aspect: '720 / 470', narrow: '360 / 370' } : M === 'sharing' ? { aspect: '720 / 500', narrow: '360 / 350' } : { aspect: '720 / 500', narrow: '360 / 390' };
  const alt = M === 'slide' ? 'A twelve by twelve grayscale image with a three by three window, the nine products of the patch under the window with the filter and their sum, and the ten by ten output map filling in cell by cell.'
    : M === 'filters' ? 'A grayscale image, the three by three filter currently applied, and the output map in blue for positive and orange for negative values.'
      : M === 'sharing' ? 'The image with its sliding window, the nine shared weights, and a log-scale bar chart comparing the 10 parameters of the convolution with the 14,500 of a dense layer.'
        : 'The target output map beside the map produced by the filter being learned, the filter weights against the target weights, and a log-scale loss curve falling.';
  const legend = M === 'sharing' ? [{ label: 'convolution', color: '--viz-forward' }, { label: 'dense layer', color: '--viz-gradient' }, { label: 'current window', color: '--accent', shape: 'line' as const }]
    : M === 'learn' ? [{ label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }, { label: 'loss', color: '--viz-forward', shape: 'line' as const }]
      : [{ label: 'positive output', color: '--viz-pos' }, { label: 'negative output', color: '--viz-neg' }, { label: 'current window', color: '--accent', shape: 'line' as const }];
  const tlLabel = M === 'learn' ? 'Timeline: training steps' : M === 'filters' ? 'Timeline: one filter after another' : 'Timeline: sliding the window across the image';
</script>

<FigureCard title={init.title || 'A filter sliding across an image'} role="group" aspect={cfg.aspect} narrowAspect={cfg.narrow} label={alt} {legend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label={tlLabel}
      onSeek={(to) => { t = to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{#each ui.vals as r, i (r.k)}{#if i}<span class="sep"> · </span>{/if}<span class="grp">{r.k} <b>{r.v}</b></span>{/each}</p>
    <p class="say" role="status" aria-live="polite">{ui.say}</p>
  {/snippet}
  {#snippet footer()}
    {#if M === 'filters'}
      <div class="ctl">
        <label>Filter
          <select value={ui.fi} onchange={(e) => seekTo(S.filterStart(+e.currentTarget.value))}>
            {#each S.FILTERS as f, i (f.id)}<option value={i}>{f.name}</option>{/each}
          </select>
        </label>
        <span class="hint">Pick a filter, or let the timeline cycle through them.</span>
      </div>
    {:else if M === 'learn'}
      <div class="ctl">
        <button type="button" class="btn" onclick={newStart}>New random start</button>
        <span class="hint">Start {seed} of 5. Each start is a different random filter.</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .sep { color: var(--text-muted); padding: 0 0.5em; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: 0.5em; font: var(--step--1) var(--font-ui); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
