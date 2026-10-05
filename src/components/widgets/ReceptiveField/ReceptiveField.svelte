<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import {
    EFF, EFF_T, EFF_MARKERS, buildScene, buildEffective, centerPick, durationOf, effDepthAt, frameAt, markersOf,
    readoutAt, region, rfTable, sizes, stackRf, unitNear, type Mode,
  } from './sim';
  import type { Reply } from './worker';

  interface Props {
    /** one: a single 3x3 layer. stack: six 3x3 layers. stride: a pooling layer in the middle. effective: gradient heatmap of a real stack. */
    mode?: Mode;
    title?: string;
    /** stride mode: start with the pooling layer replaced by a plain 3x3 conv. */
    plain?: boolean;
  }
  let { mode = 'stack', title = 'What one output unit can see', plain: plainProp = false }: Props = $props();

  const init = untrack(() => ({ mode, title, plain: plainProp }));
  const isEff = init.mode === 'effective';
  let plain = $state(init.plain);
  let sqrtScale = $state(true);
  const sc = $derived(isEff ? null : buildScene(init.mode as Exclude<Mode, 'effective'>, plain));
  const alt = $derived(init.mode === 'stride' ? buildScene('stride', !plain) : null);
  let pickCol = $state(0), pickRow = $state(0);
  $effect.pre(() => { if (sc) { const p = centerPick(sc); pickCol = p.col; pickRow = p.row; } });
  const topN = $derived(sc ? sizes(sc)[sc.specs.length] : 1);
  const T = $derived(sc ? durationOf(sc) : EFF_T);
  const markers = $derived(sc ? markersOf(sc) : EFF_MARKERS);

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let vertical = $state(false);
  let ready = $state(!isEff);
  let say = $state(''), num = $state('');
  let seg = '';
  let t = 0;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, poke = () => {};
  $effect(() => { if (t > T) t = T; });
  $effect(() => { plain; sqrtScale; pickCol; pickRow; vertical; ready; playing; speed; poke(); });

  const LEGEND = isEff
    ? [{ label: 'gradient strength', color: '--viz-gradient' }, { label: 'theoretical field (dashed)', color: '--accent', shape: 'line' as const }, { label: 'holds 90% of the gradient (solid)', color: '--text', shape: 'line' as const }]
    : [{ label: 'chosen unit and what it sees', color: '--accent' }];
  const ARIA = isEff
    ? 'A heatmap of how strongly each input pixel affects one output unit of a random convolution stack, brightest in the middle, with the theoretical square outlined, and a profile of the same gradient across the columns. The depth grows from 1 to 12 layers.'
    : 'Rows of units, one per layer, with the units that can influence one chosen output unit highlighted, widening layer by layer down to the input, and the same patch outlined on the input image.';

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const NAMES = { grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft' };
    let c = readColors(NAMES), code = readVar('--font-code'), body = readVar('--font-body');
    let lastUi = 0;
    let eff: ReturnType<typeof buildEffective> | undefined;
    if (reduced.matches) t = untrack(() => T);

    const text = (s: string, x: number, y: number, col: string, size: number, align: CanvasTextAlign = 'left', mono = true, a = 1) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = mono ? `600 ${size}px ${code}` : `${size}px ${body}`;
      ctx.fillText(s, x, y);
      ctx.globalAlpha = 1;
    };
    const mix = (a: [number, number, number], b: [number, number, number], p: number) =>
      `rgb(${Math.round(a[0] + (b[0] - a[0]) * p)},${Math.round(a[1] + (b[1] - a[1]) * p)},${Math.round(a[2] + (b[2] - a[2]) * p)})`;

    /* ------------------------------------------------------------ layout */
    const layout = () => vertical
      ? { w: 360, h: 540, px: 58, pw: 250, py: 30, ph: 270, fs: 11, cx: 318, ix: 80, iy: 336, is: 200 }
      : { w: 720, h: 360, px: 66, pw: 380, py: 34, ph: 296, fs: 11, cx: 458, ix: 514, iy: 52, is: 194 };

    /* ------------------------------------------------------------ rows of units */
    function drawRows() {
      const S = sc!, L = layout(), nL = S.specs.length, N = S.n;
      const tab = rfTable(S.specs), cnt = sizes(S);
      const reg = region(S, pickCol), fr = frameAt(t, S);
      const cell = Math.min(L.pw / N, 30), xoff = L.px + (L.pw - cell * N) / 2;
      const rowH = L.ph / (nL + 1), ch = Math.min(22, rowH * 0.58);
      const xc = (l: number, i: number) => xoff + (i * tab[l].jump + (tab[l].r - 1) / 2 + 0.5) * cell;
      const cw = (l: number) => Math.max(5, cell * tab[l].jump * 0.82);
      const yc = (l: number) => L.py + (nL - l) * rowH + rowH / 2;

      text('layer', L.px - 8, L.py - 10, c.muted.hex, L.fs, 'right', false);
      text(vertical ? 'cells' : 'cells seen', L.cx, L.py - 10, c.muted.hex, L.fs, 'left', false);

      // connections from the traced cells to the cells they read
      ctx.lineWidth = 1; ctx.strokeStyle = c.accent.hex;
      for (let l = nL; l >= 1; l--) {
        const a = fr.reveal[l - 1];
        if (a <= 0.01) continue;
        ctx.globalAlpha = 0.45 * a;
        const { k, stride } = S.specs[l - 1];
        ctx.beginPath();
        for (let i = reg[l][0]; i <= reg[l][1]; i++)
          for (let q = 0; q < k; q++) { ctx.moveTo(xc(l, i), yc(l) + ch / 2); ctx.lineTo(xc(l - 1, i * stride + q), yc(l - 1) - ch / 2); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      for (let l = 0; l <= nL; l++) {
        const [lo, hi] = reg[l], a = fr.reveal[l], y = yc(l), w = cw(l);
        for (let i = 0; i < cnt[l]; i++) {
          const inReg = i >= lo && i <= hi, top = l === nL && i === pickCol;
          const x = xc(l, i) - w / 2;
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
          ctx.fillRect(x, y - ch / 2, w, ch); ctx.strokeRect(x + 0.5, y - ch / 2 + 0.5, w - 1, ch - 1);
          if (inReg) {
            ctx.fillStyle = c.accent.hex; ctx.globalAlpha = (top ? 1 : 0.5) * (top ? 0.55 + 0.45 * (1 - fr.pulse * 0.6) : a);
            ctx.fillRect(x, y - ch / 2, w, ch);
            ctx.globalAlpha = a; ctx.strokeStyle = c.accent.hex; ctx.strokeRect(x + 0.5, y - ch / 2 + 0.5, w - 1, ch - 1); ctx.globalAlpha = 1;
          }
        }
        text(l === 0 ? 'input' : S.specs[l - 1].label, L.px - 8, y + L.fs / 3, c.muted.hex, L.fs, 'right', false);
        if (a > 0.5) text(String(hi - lo + 1), L.cx, y + L.fs / 3, c.text.hex, L.fs + 1, 'left');
      }
      return { L, N, reg, fr, tab, cell, xoff, rowH, yc };
    }

    /* ------------------------------------------------------------ 2-D input panel */
    function drawImage(g: ReturnType<typeof drawRows>) {
      const S = sc!, { L, N } = g, cs = L.is / N, nL = S.specs.length;
      const colR = g.reg[0], rowR = region(S, pickRow)[0];
      text(`input image, ${N} × ${N}`, L.ix + L.is / 2, L.iy - 8, c.muted.hex, L.fs, 'center', false);
      ctx.fillStyle = c.raised.hex; ctx.fillRect(L.ix, L.iy, L.is, L.is);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i <= N; i++) { ctx.moveTo(L.ix + i * cs, L.iy); ctx.lineTo(L.ix + i * cs, L.iy + L.is); ctx.moveTo(L.ix, L.iy + i * cs); ctx.lineTo(L.ix + L.is, L.iy + i * cs); }
      ctx.globalAlpha = N > 12 ? 0.6 : 1; ctx.stroke(); ctx.globalAlpha = 1;
      const a = g.fr.reveal[0];
      const rx = L.ix + colR[0] * cs, ry = L.iy + rowR[0] * cs, rw = (colR[1] - colR[0] + 1) * cs, rh = (rowR[1] - rowR[0] + 1) * cs;
      ctx.fillStyle = c.accent.hex; ctx.globalAlpha = 0.45 * a; ctx.fillRect(rx, ry, rw, rh);
      ctx.globalAlpha = a; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, rw, rh); ctx.globalAlpha = 1;
      if (alt) { // the other version of the network, centred on the same pixel
        const r2 = rfTable(alt.specs)[alt.specs.length].r, mx = rx + rw / 2, my = ry + rh / 2, s2 = r2 * cs;
        ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; ctx.strokeStyle = c.text.hex; ctx.globalAlpha = a;
        ctx.strokeRect(mx - s2 / 2, my - s2 / 2, s2, s2); ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      const f = rfTable(S.specs)[nL].r;
      if (a > 0.5) text(`${f} × ${f} = ${f * f} pixels`, L.ix + L.is / 2, L.iy + L.is + 18, c.text.hex, L.fs + 1, 'center');
      if (alt) {
        const r2 = rfTable(alt.specs)[alt.specs.length].r;
        text(`dashed: ${r2} × ${r2} ${plain ? 'with pooling' : 'without pooling'}`, L.ix + L.is / 2, L.iy + L.is + 36, c.muted.hex, L.fs, 'center', false);
      }
    }

    function renderRows() {
      const S = sc!, L = layout(), sx = cv.width / L.w;
      ctx.setTransform(sx, 0, 0, sx, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      const g = drawRows();
      drawImage(g);
      const nL = S.specs.length, ro = readoutAt(S, { col: pickCol, row: pickRow }, g.fr);
      const tab = rfTable(S.specs);
      const nextSay = g.fr.phase === 'pick' ? `Chosen: one unit in the output layer, column ${pickCol + 1} of ${topN}.`
        : g.fr.phase === 'trace' ? 'Tracing back: each layer adds the cells that the highlighted cells read.'
        : `Traced to the input: this unit depends on a ${ro.field} × ${ro.field} patch and on nothing else.`;
      if (nextSay !== seg) { seg = nextSay; say = nextSay; }
      if (init.mode === 'stride') {
        const alt1 = rfTable(alt!.specs).slice(1).map((r) => r.r).join(', ');
        num = `field r after each layer: ${tab.slice(1).map((r) => r.r).join(', ')}   (${plain ? 'with pooling' : 'plain 3×3 stack'}: ${alt1})`;
      } else {
        const j = g.fr.back;
        num = `${j} layer${j === 1 ? '' : 's'} back: 1 + ${j} × 2 = ${ro.cells} cell${ro.cells === 1 ? '' : 's'}   ·   field = 1 + ${nL} × 2 = ${stackRf(nL)}`;
      }
      return g;
    }

    /* ------------------------------------------------------------ effective receptive field */
    function renderEff() {
      const L = vertical
        ? { w: 360, h: 560, hx: 15, hy: 30, hs: 330, ox: 46, oy: 398, ow: 300, oh: 110, fs: 10 }
        : { w: 720, h: 360, hx: 14, hy: 30, hs: 314, ox: 398, oy: 52, ow: 296, oh: 230, fs: 11 };
      const sx = cv.width / L.w;
      ctx.setTransform(sx, 0, 0, sx, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      if (!eff) {
        text('Building a random stack and backpropagating from one unit…', L.w / 2, L.h / 2, c.muted.hex, L.fs + 1, 'center', false);
        return;
      }
      const n = EFF.n, mid = n >> 1, d = effDepthAt(t), st = eff.stats[d - 1], map = eff.maps[d - 1], mg = eff.margs[d - 1];
      const cs = L.hs / n;
      text(`gradient reaching each input pixel, ${n} × ${n}`, L.hx + L.hs / 2, L.hy - 10, c.muted.hex, L.fs, 'center', false);
      let peak = 0; for (let i = 0; i < map.length; i++) if (map[i] > peak) peak = map[i];
      const lo = c.raised.rgb, hi = c.grad.rgb;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const v = map[y * n + x] / peak;
        ctx.fillStyle = mix(lo, hi, sqrtScale ? Math.sqrt(v) : v);
        ctx.fillRect(L.hx + x * cs, L.hy + y * cs, cs + 0.5, cs + 0.5);
      }
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(L.hx + 0.5, L.hy + 0.5, L.hs - 1, L.hs - 1);
      const sq = (side: number, col: string, lw: number, dash: number[]) => {
        const o = (mid - (side - 1) / 2) * cs;
        ctx.setLineDash(dash); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.strokeRect(L.hx + o, L.hy + o, side * cs, side * cs); ctx.setLineDash([]);
      };
      sq(st.theory, c.accent.hex, 2, [5, 3]);
      sq(st.core, c.text.hex, 1.5, []);

      // column profile
      let pm = 0; for (let i = 0; i < n; i++) if (mg[i] > pm) pm = mg[i];
      const bw = L.ow / n, Y = (v: number) => L.oy + L.oh - (v / pm) * L.oh;
      text('gradient per input column, relative to the tallest', L.ox, L.oy - 12, c.muted.hex, L.fs, 'left', false);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.ox, L.oy + L.oh + 0.5); ctx.lineTo(L.ox + L.ow, L.oy + L.oh + 0.5); ctx.stroke();
      ctx.fillStyle = c.grad.hex; ctx.globalAlpha = 0.9;
      for (let i = 0; i < n; i++) ctx.fillRect(L.ox + i * bw + 0.5, Y(mg[i]), Math.max(1, bw - 1), (mg[i] / pm) * L.oh);
      ctx.globalAlpha = 1;
      const edge = (side: number, col: string, dash: number[], lw: number) => {
        ctx.setLineDash(dash); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
        for (const e of [mid - (side - 1) / 2, mid + (side - 1) / 2 + 1]) { ctx.moveTo(L.ox + e * bw, L.oy - 2); ctx.lineTo(L.ox + e * bw, L.oy + L.oh); }
        ctx.stroke(); ctx.setLineDash([]);
      };
      edge(st.theory, c.accent.hex, [5, 3], 2);
      edge(st.core, c.text.hex, [], 1.5);
      for (const k of [-16, 0, 16]) text(String(k), L.ox + (mid + k + 0.5) * bw, L.oy + L.oh + 15, c.muted.hex, L.fs, 'center');
      text('pixels from the centre', L.ox + L.ow / 2, L.oy + L.oh + 31, c.muted.hex, L.fs, 'center', false);

      const area = Math.round((st.core * st.core * 100) / (st.theory * st.theory));
      num = `${d} layer${d === 1 ? '' : 's'}: theoretical field ${st.theory} × ${st.theory} = ${st.theory ** 2} pixels · 90% of the gradient lies in the middle ${st.core} × ${st.core} = ${st.core ** 2} pixels (${area}% of the field's area)`;
      const s = d < 4 ? 'a' : d < 8 ? 'b' : d < 12 ? 'c' : 'd';
      if (s !== seg) {
        seg = s;
        say = s === 'a' ? 'With 1 to 3 layers the gradient fills almost the whole square: the effective field matches the theoretical one.'
          : s === 'b' ? 'The square keeps growing by 2 pixels per layer, but the bright core grows more slowly.'
          : s === 'c' ? 'The outer ring of the square is faint: those pixels can change the unit, but barely do.'
          : `At ${d} layers, 90% of the gradient lies in the middle ${st.core} × ${st.core} of a ${st.theory} × ${st.theory} square.`;
      }
    }

    function render(now: number) {
      if (isEff) renderEff(); else renderRows();
      tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
    }

    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: untrack(() => T) }).t;
      render(now);
      return playing;
    }, { onView: (inView, first) => { if (inView && first && !reduced.matches) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = Math.min(2, Math.max(1, devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { vertical = e.contentRect.width < 540; fit(); render(performance.now()); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(NAMES); code = readVar('--font-code'); body = readVar('--font-body'); poke(); });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();

    // clicks choose the output unit
    const onDown = (e: PointerEvent) => {
      if (isEff || !sc) return;
      const L = layout(), r = cv.getBoundingClientRect(), lx = ((e.clientX - r.left) / r.width) * L.w, ly = ((e.clientY - r.top) / r.height) * L.h;
      const N = sc.n, nL = sc.specs.length;
      let col = -1, row = -1;
      if (lx >= L.ix && lx <= L.ix + L.is && ly >= L.iy && ly <= L.iy + L.is) {
        col = unitNear(sc, ((lx - L.ix) / L.is) * N - 0.5); row = unitNear(sc, ((ly - L.iy) / L.is) * N - 0.5);
      } else if (lx >= L.px && lx <= L.px + L.pw && ly >= L.py && ly <= L.py + L.ph / (nL + 1)) {
        const cell = Math.min(L.pw / N, 30), xoff = L.px + (L.pw - cell * N) / 2;
        col = unitNear(sc, (lx - xoff) / cell - 0.5); row = pickRow;
      }
      if (col < 0) return;
      pickCol = col; pickRow = row; t = 1.2; playing = !reduced.matches;
      if (reduced.matches) t = untrack(() => T);
      poke();
    };
    cv.addEventListener('pointerdown', onDown);

    // the random stack is built in a worker (about 150 ms of work)
    let worker: Worker | undefined;
    if (isEff) {
      const done = (r: Reply) => { if ('error' in r) return; eff = r; ready = true; poke(); };
      try {
        worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
        worker.onmessage = (e: MessageEvent<Reply>) => done(e.data);
        worker.onerror = () => { worker?.terminate(); worker = undefined; done(buildEffective()); };
        worker.postMessage(0);
      } catch { done(buildEffective()); }
    }

    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); cv.removeEventListener('pointerdown', onDown); worker?.terminate(); };
  });
</script>

<FigureCard title={init.title} legend={LEGEND} role="img" aspect="720 / 360" narrowAspect={isEff ? '360 / 560' : '360 / 540'} label={ARIA}>
  <canvas bind:this={cv} aria-hidden="true" class:pick={!isEff}></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers}
      label={isEff ? 'Timeline: the network grows from 1 to 12 layers' : 'Timeline: pick a unit, then trace back to the input'}
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to >= T ? 0 : to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{say}</p>
    <p class="num muted">{num}</p>
  {/snippet}
  {#snippet footer()}
    {#if !isEff || ready}
      <div class="ctl">
        {#if isEff}
          <label class="chk"><input type="checkbox" bind:checked={sqrtScale} /> brighten faint pixels (square-root colour scale)</label>
        {:else}
          <label>output unit, column
            <input type="range" min="0" max={topN - 1} step="1" bind:value={pickCol} aria-label="Output unit column" onchange={() => { t = 1.2; playing = true; }} />
          </label>
          <label>row
            <input type="range" min="0" max={topN - 1} step="1" bind:value={pickRow} aria-label="Output unit row" onchange={() => { t = 1.2; playing = true; }} />
          </label>
          {#if init.mode === 'stride'}
            <label class="chk"><input type="checkbox" bind:checked={plain} /> replace the pooling layer with a plain 3×3 layer</label>
          {/if}
          <span class="muted">Click a unit in the top row or on the image to choose it.</span>
        {/if}
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say, .muted { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .num { font-family: var(--font-code); font-size: var(--step--1); font-variant-numeric: tabular-nums; margin: 0; }
  .muted { color: var(--text-muted); font-size: var(--step--1); }
  canvas.pick { cursor: pointer; touch-action: manipulation; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 7rem; accent-color: var(--accent); }
</style>
