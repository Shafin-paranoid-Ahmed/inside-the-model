<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    PRESETS, HOLD, duration, markers, frameAt, squareImage, mul2, apply, det, makeGrid, transposeGrid, crop, matmul,
    shapesDuration, shapesMarkers, shapesFrameAt, shapesValid,
    type Mat2, type Vec2, type Mode, type PresetId,
  } from './sim';

  interface Props {
    title?: string;
    /** transform: the grid warps. dot: same, with each output entry as a row dot x. compose: two matrices in a row. shapes: m×n times n×p. */
    mode?: Mode;
    /** Preset that the (second) matrix A starts as. */
    preset?: PresetId;
    /** Preset that the first matrix B starts as (compose only). */
    firstPreset?: PresetId;
    /** The vector x. */
    x?: Vec2;
    /** Hide the vector x and its image. */
    hideVector?: boolean;
  }
  let { title = 'A matrix bends the plane', mode = 'transform', preset = 'shear', firstPreset = 'rotate', x: x0 = [1.5, 1], hideVector = false }: Props = $props();
  const init = untrack(() => ({ mode, preset, firstPreset, x: x0, hide: hideVector }));
  const MODE = init.mode;
  const plane = MODE !== 'shapes';
  const showVec = plane && MODE !== 'compose' && !init.hide;

  let A = $state<Mat2>([...PRESETS[init.preset].m]);
  let B = $state<Mat2>([...PRESETS[init.firstPreset].m]);
  let order = $state<'AB' | 'BA'>('AB');
  let xv = $state<Vec2>([...init.x]);
  // shapes state: dims of the two factors and their (4×4) entries
  let baseA = $state(makeGrid(11)), baseB = $state(makeGrid(23));
  let dims = $state({ m: 2, n: 3, q: 3, p: 4 });

  const first = $derived<Mat2>(order === 'AB' ? B : A);
  const second = $derived<Mat2>(order === 'AB' ? A : B);
  const Ag = $derived(crop(baseA, dims.m, dims.n)), Bg = $derived(crop(baseB, dims.q, dims.p));
  const valid = $derived(shapesValid(dims));
  const Tdur = $derived(plane ? duration(MODE) : shapesDuration(dims));
  const MK = $derived(plane ? markers(MODE) : shapesMarkers(dims));

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let t = 0, wake = () => {}, dragging = false;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const fr = $derived(plane ? frameAt(uiT, MODE, MODE === 'compose' ? second : A, xv, first) : null);
  const sf = $derived(plane ? null : shapesFrameAt(uiT, Ag, Bg, dims));
  const C = $derived(valid ? matmul(Ag, Bg) : null);

  const narration = $derived.by(() => {
    if (fr) {
      if (MODE === 'compose') {
        return fr.phase === 'start' ? 'Start: the plane is untouched.' : fr.phase === 'apply' ? 'Applying the two matrices one after the other.' : `Done: the grid now sits where the single matrix ${order} puts it.`;
      }
      return fr.phase === 'start' ? 'Start: the identity matrix leaves the plane untouched.' : fr.phase === 'apply' ? 'Morphing: the grid bends as the matrix takes effect.' : Math.abs(fr.det) < 0.02 ? 'Result: the matrix has squashed the whole plane onto a line.' : 'Result: the grid as the matrix leaves it.';
    }
    return valid ? `Computing output row ${sf!.row + 1}, column ${sf!.col + 1}: row ${sf!.row + 1} of the left matrix dotted with column ${sf!.col + 1} of the right one.` : `The inner dimensions ${dims.n} and ${dims.q} differ, so this product is not defined.`;
  });

  // any change of matrices, order or shapes replays from the start
  $effect(() => {
    A[0]; A[1]; A[2]; A[3]; B[0]; B[1]; B[2]; B[3]; order; dims.m; dims.n; dims.q; dims.p; baseA; baseB;
    untrack(() => { t = reducedMotion() ? Tdur : 0; playing = !reducedMotion(); wake(); });
  });
  $effect(() => { speed; wake(); });

  const setEntry = (M: Mat2, i: number, v: number) => { if (Number.isFinite(v)) M[i] = clamp(v, -3, 3); };
  const fm = (M: Mat2) => `[${f2(M[0])}  ${f2(M[1])} ; ${f2(M[2])}  ${f2(M[3])}]`;
  const fv = (v: Vec2) => `(${f2(v[0])}, ${f2(v[1])})`;
  const par = (v: number) => (v < 0 ? `(−${-v})` : String(v));
  const sgn = (v: number) => (v < -0.005 ? 'neg' : v > 0.005 ? 'pos' : '');
  function transposeA() { baseA = transposeGrid(baseA); dims = { ...dims, m: dims.n, n: dims.m }; }
  function transposeB() { baseB = transposeGrid(baseB); dims = { ...dims, q: dims.p, p: dims.q }; }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const tt = () => Math.min(t, Tdur);

    // ---------- plane geometry
    const lay = () => {
      const u = Math.min(stage.w / 10.4, stage.h / 6.2);
      const cx = stage.w / 2, cy = stage.h / 2;
      return { u, cx, cy, px: (v: number) => cx + v * u, py: (v: number) => cy - v * u };
    };
    const cl = (v: number) => clamp(v, -40, 40);
    let taken: number[][] = [];
    function arrow(L: ReturnType<typeof lay>, x1: number, y1: number, col: string, w: number, dash = false, label = '') {
      const { ctx } = stage, X0 = L.px(0), Y0 = L.py(0), X1 = L.px(cl(x1)), Y1 = L.py(cl(y1));
      if (Math.hypot(X1 - X0, Y1 - Y0) < 2) return;
      const ang = Math.atan2(Y1 - Y0, X1 - X0);
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X1 - 9 * Math.cos(ang - 0.4), Y1 - 9 * Math.sin(ang - 0.4)); ctx.lineTo(X1 - 9 * Math.cos(ang + 0.4), Y1 - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      if (label) { ctx.font = stage.body(15, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; const lx = X1 + 10 * Math.cos(ang) + 11 * Math.sin(ang), ly = Y1 + 10 * Math.sin(ang) - 11 * Math.cos(ang); taken.push([lx, ly]); ctx.fillText(label, lx, ly); }
    }
    function polygon(L: ReturnType<typeof lay>, M: Mat2) {
      const { ctx } = stage, P = squareImage(M);
      ctx.beginPath(); P.forEach((p, i) => (i ? ctx.lineTo(L.px(cl(p[0])), L.py(cl(p[1]))) : ctx.moveTo(L.px(cl(p[0])), L.py(cl(p[1]))))); ctx.closePath();
    }

    function renderPlane(now: number) {
      taken = [];
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frameAt(now, MODE, MODE === 'compose' ? second : A, xv, first), M = f.M, N = 12;
      ctx.clearRect(0, 0, W, H);
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      // the untouched grid, faint
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex;
      ctx.beginPath();
      for (let k = -N; k <= N; k++) { ctx.moveTo(L.px(k), 0); ctx.lineTo(L.px(k), H); ctx.moveTo(0, L.py(k)); ctx.lineTo(W, L.py(k)); }
      ctx.stroke();
      // the transformed grid
      const line = (ax: number, ay: number, bx: number, by: number) => { ctx.moveTo(L.px(M[0] * ax + M[1] * ay), L.py(M[2] * ax + M[3] * ay)); ctx.lineTo(L.px(M[0] * bx + M[1] * by), L.py(M[2] * bx + M[3] * by)); };
      ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let k = -N; k <= N; k++) if (k) { line(k, -N, k, N); line(-N, k, N, k); }
      ctx.stroke();
      ctx.globalAlpha = 0.9; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.6; ctx.beginPath(); line(0, -N, 0, N); line(-N, 0, N, 0); ctx.stroke(); ctx.globalAlpha = 1;
      // image of the unit square
      polygon(L, M); ctx.fillStyle = c.s3.hex; ctx.globalAlpha = 0.35; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = c.s5.hex; ctx.lineWidth = 1.5; ctx.stroke();
      // the same matrix computed as one product: appears at the end of the compose timeline
      if (f.ghost > 0.01) {
        polygon(L, mul2(second, first)); ctx.globalAlpha = f.ghost; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      // the vector and its image
      if (showVec) {
        arrow(L, xv[0], xv[1], c.accent.hex, 2, true, 'x');
        if (MODE === 'dot') {
          const px = L.px(cl(f.y[0])), py = L.py(cl(f.y[1]));
          ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, L.py(0)); ctx.moveTo(px, py); ctx.lineTo(L.px(0), py); ctx.stroke(); ctx.setLineDash([]);
          ctx.font = stage.mono(12, 700); ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
          ctx.fillStyle = f.y[0] < 0 ? c.neg.hex : c.pos.hex; ctx.fillText(`y₁ ${f2(f.y[0])}`, px, L.py(0) - (f.y[1] >= 0 ? 6 : -22));
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = f.y[1] < 0 ? c.neg.hex : c.pos.hex; ctx.fillText(`y₂ ${f2(f.y[1])}`, L.px(0) + (f.y[0] >= 0 ? -70 : 8), py - (f.y[0] >= 0 ? 10 : 0));
        }
        arrow(L, f.y[0], f.y[1], c.accent.hex, 3.2, false, MODE === 'dot' ? 'y' : 'Ax');
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.arc(L.px(xv[0]), L.py(xv[1]), 11, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
      }
      // basis vectors
      arrow(L, M[0], M[2], c.s5.hex, 3, false, 'î');
      arrow(L, M[1], M[3], c.fwd.hex, 3, false, 'ĵ');
      // unit ticks
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const free = (x: number, y: number) => taken.every((q) => Math.hypot(q[0] - x, q[1] - y) > 16);
      for (const k of [-3, -2, -1, 1, 2, 3]) if (free(L.px(k), L.py(0) + 10)) ctx.fillText(String(k).replace('-', '−'), L.px(k), L.py(0) + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const k of [-2, -1, 1, 2]) if (free(L.px(0) - 10, L.py(k))) ctx.fillText(String(k).replace('-', '−'), L.px(0) - 5, L.py(k));
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(L.px(0), L.py(0), 3, 0, 6.2832); ctx.fill();
      ctx.restore();
      if (showVec) {
        ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('drag the dashed ring to move x', 8, 6);
      }
      if (MODE === 'compose' && f.ghost > 0.3) { ctx.font = stage.mono(12, 700); ctx.fillStyle = c.accent.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`dashed outline: ${order} computed in one go`, W - 8, 6); }
    }

    // ---------- shapes
    function renderShapes(now: number) {
      const { ctx, w: W, h: H, c } = stage;
      ctx.clearRect(0, 0, W, H);
      const { m, n, q, p } = dims, ok = shapesValid(dims), s = shapesFrameAt(now, Ag, Bg, dims);
      const gap = 1.3, br = 0.4, cols = n + p + p + 2 * gap + 6 * br;
      const cell = Math.min(48, W / cols, (H - 44) / Math.max(m, q)), ymid = H / 2 + 12;
      let x = (W - cols * cell) / 2;
      ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
      const dimLabel = (bx: number, bw: number, top: number, a: string, b: string, ca: string, cb: string) => {
        ctx.font = stage.mono(13, 700);
        const wa = ctx.measureText(a).width, wb = ctx.measureText(b).width, wx = ctx.measureText(' × ').width, x1 = bx + bw / 2 - (wa + wx + wb) / 2;
        ctx.textAlign = 'left'; ctx.fillStyle = ca; ctx.fillText(a, x1, top); ctx.fillStyle = c.muted.hex; ctx.fillText(' × ', x1 + wa, top); ctx.fillStyle = cb; ctx.fillText(b, x1 + wa + wx, top); ctx.textAlign = 'center';
      };
      const bracket = (bx: number, top: number, hh: number, bw: number) => {
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.6; ctx.lineCap = 'butt'; ctx.beginPath();
        ctx.moveTo(bx + 5, top); ctx.lineTo(bx, top); ctx.lineTo(bx, top + hh); ctx.lineTo(bx + 5, top + hh);
        ctx.moveTo(bx + bw - 5, top); ctx.lineTo(bx + bw, top); ctx.lineTo(bx + bw, top + hh); ctx.lineTo(bx + bw - 5, top + hh); ctx.stroke();
      };
      const box = (bx: number, top: number, i: number, j: number, fill: string, al: number) => { ctx.globalAlpha = al; ctx.fillStyle = fill; ctx.fillRect(bx + j * cell + 1.5, top + i * cell + 1.5, cell - 3, cell - 3); ctx.globalAlpha = 1; };
      const text = (bx: number, top: number, i: number, j: number, v: string, col: string) => { ctx.font = stage.mono(Math.round(cell * 0.42), 600); ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.fillText(v, bx + (j + 0.5) * cell, top + (i + 0.5) * cell + 1); };
      const num = (v: number) => String(v).replace('-', '−');
      const pair = s.valid ? Math.min(s.terms, n - 1) : 0;
      const mid = (rows: number) => ymid - (rows * cell) / 2;

      // left matrix A
      x += br * cell; const ax = x, at = mid(m); bracket(ax - br * cell * 0.6, at - 3, m * cell + 6, n * cell + 2 * br * cell * 0.6);
      if (ok) {
        for (let j = 0; j < n; j++) box(ax, at, s.row, j, c.fwd.hex, 0.18);
        box(ax, at, s.row, pair, c.accent.hex, 0.45);
      }
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) text(ax, at, i, j, num(Ag[i][j]), c.text.hex);
      dimLabel(ax, n * cell, at - 16, String(m), String(n), c.muted.hex, ok ? c.fwd.hex : c.neg.hex);
      x += n * cell + br * cell + gap * cell;
      ctx.font = stage.body(Math.round(cell * 0.8)); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.fillText('×', x - (gap * cell) / 2 - br * cell * 0.5, ymid);
      // right matrix B
      x += br * cell; const bx0 = x, bt = mid(q); bracket(bx0 - br * cell * 0.6, bt - 3, q * cell + 6, p * cell + 2 * br * cell * 0.6);
      if (ok) {
        for (let i = 0; i < q; i++) box(bx0, bt, i, s.col, c.fwd.hex, 0.18);
        box(bx0, bt, pair, s.col, c.accent.hex, 0.45);
      } else for (let i = 0; i < q; i++) box(bx0, bt, i, 0, c.neg.hex, 0.12);
      for (let i = 0; i < q; i++) for (let j = 0; j < p; j++) text(bx0, bt, i, j, num(Bg[i][j]), c.text.hex);
      dimLabel(bx0, p * cell, bt - 16, String(q), String(p), ok ? c.fwd.hex : c.neg.hex, c.muted.hex);
      x += p * cell + br * cell + gap * cell;
      ctx.font = stage.body(Math.round(cell * 0.8)); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.fillText('=', x - (gap * cell) / 2 - br * cell * 0.5, ymid);
      // result
      x += br * cell; const cx0 = x, ct = mid(m); bracket(cx0 - br * cell * 0.6, ct - 3, m * cell + 6, p * cell + 2 * br * cell * 0.6);
      if (ok) {
        const Cg = matmul(Ag, Bg);
        for (let k = 0; k < m * p; k++) {
          const i = Math.floor(k / p), j = k % p;
          if (k < s.done) text(cx0, ct, i, j, num(Cg[i][j]), c.text.hex);
          else if (k === s.done) { box(cx0, ct, i, j, c.accent.hex, 0.22); text(cx0, ct, i, j, s.terms ? num(s.partial) : '·', s.terms === n ? c.text.hex : c.muted.hex); }
          else { ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(cx0 + j * cell + 4, ct + i * cell + 4, cell - 8, cell - 8); }
        }
        dimLabel(cx0, p * cell, ct - 16, String(m), String(p), c.muted.hex, c.muted.hex);
      } else {
        ctx.font = stage.mono(Math.round(cell * 0.9), 700); ctx.fillStyle = c.neg.hex; ctx.textAlign = 'center'; ctx.fillText('undefined', cx0 + (p * cell) / 2, ymid);
      }
    }

    function render() { const v = tt(); (plane ? renderPlane : renderShapes)(v); }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: Tdur, loopFrom: 0, loopTo: Tdur + HOLD }).t;
      const v = tt();
      render(); tl.paint(v);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = v; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    // drag the tip of x
    const world = (e: PointerEvent): Vec2 => {
      const r = cv.getBoundingClientRect(), L = lay();
      return [(e.clientX - r.left - L.cx) / L.u, -(e.clientY - r.top - L.cy) / L.u];
    };
    const near = (e: PointerEvent) => { const w = world(e), L = lay(); return Math.hypot((w[0] - xv[0]) * L.u, (w[1] - xv[1]) * L.u) < 20; };
    const down = (e: PointerEvent) => { if (showVec && near(e)) { dragging = true; cv.setPointerCapture(e.pointerId); e.preventDefault(); wake(); } };
    const move = (e: PointerEvent) => {
      if (!showVec) return;
      if (dragging) { const w = world(e); xv = [Math.round(clamp(w[0], -3, 3) * 20) / 20, Math.round(clamp(w[1], -2.5, 2.5) * 20) / 20]; wake(); }
      else cv.style.cursor = near(e) ? 'grab' : '';
    };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });

  const legend = plane
    ? [{ label: 'î', color: '--viz-seq-5', shape: 'line' as const }, { label: 'ĵ', color: '--viz-forward', shape: 'line' as const }, { label: 'grid after the matrix', color: '--viz-seq-4', shape: 'line' as const }, ...(showVec ? [{ label: 'x and Ax', color: '--accent', shape: 'line' as const }] : [])]
    : [{ label: 'row and column being dotted', color: '--viz-forward' }, { label: 'entry being added', color: '--accent' }, { label: 'mismatch', color: '--viz-neg' }];
  const figLabel = plane
    ? 'The plane with a grid and the two basis vectors, redrawn as a 2 by 2 matrix takes effect. The timeline morphs the grid from the identity to the matrix.'
    : 'A matrix multiplication drawn as three grids of numbers. The timeline sweeps over each output cell, highlighting the row of the left matrix and the column of the right matrix that produce it.';
</script>

{#snippet presetBtns(set: (m: Mat2) => void, who: string)}
  <span class="grp" role="group" aria-label="Presets for {who}">
    {#each Object.entries(PRESETS) as [id, p] (id)}
      <button type="button" class="btn sm" onclick={() => set([...p.m])}>{p.name}</button>
    {/each}
  </span>
{/snippet}
{#snippet editor(M: Mat2, who: string)}
  <span class="ed"><span class="nm">{who}</span>
    <span class="mat">
      {#each [0, 1, 2, 3] as i (i)}
        <input type="number" step="0.1" min="-3" max="3" value={+M[i].toFixed(3)} aria-label="{who} row {Math.floor(i / 2) + 1} column {(i % 2) + 1}" oninput={(e) => setEntry(M, i, e.currentTarget.valueAsNumber)} />
      {/each}
    </span>
  </span>
{/snippet}

<FigureCard {title} role="group" aspect={plane ? '16 / 9' : '16 / 7'} narrowAspect="4 / 3" label={figLabel} {legend}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={Tdur} markers={MK} label="Timeline: {plane ? 'from the identity to the matrix' : 'sweep over the output cells'}"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= Tdur ? 0 : to; wake(); }} onReset={() => { t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if fr}
      {#if MODE === 'dot'}
        <p class="vals num">
          y₁ = row 1 · x = {f2(fr.M[0])}·{f2(xv[0])} + {f2(fr.M[1])}·{f2(xv[1])} = <b class={sgn(fr.y[0])}>{f2(fr.y[0])}</b><br />
          y₂ = row 2 · x = {f2(fr.M[2])}·{f2(xv[0])} + {f2(fr.M[3])}·{f2(xv[1])} = <b class={sgn(fr.y[1])}>{f2(fr.y[1])}</b>
        </p>
      {:else if MODE === 'compose'}
        <p class="vals num">first {fm(first)} · second {fm(second)} · {order} = {fm(mul2(second, first))}<br />
          {#if fr.phase === 'result'}apply in order: second(first(x)) = {fv(apply(second, apply(first, xv)))} · one product: {order}·x = {fv(apply(mul2(second, first), xv))}{:else}now acting: {fm(fr.M)}{/if}</p>
      {:else}
        <p class="vals num">î → {fv([fr.M[0], fr.M[2]])} · ĵ → {fv([fr.M[1], fr.M[3]])} · area × <b>{f2(fr.det)}</b><br />
          x = {fv(xv)} → Ax = <b>{fv(fr.y)}</b></p>
      {/if}
    {:else if sf}
      <p class="vals num">
        {#if sf.valid}
          ({dims.m}×{dims.n})({dims.q}×{dims.p}) → {dims.m}×{dims.p}: {dims.n} products per cell, {dims.m * dims.p} cells.<br />
          cell ({sf.row + 1}, {sf.col + 1}) = {#each Array.from({ length: dims.n }, (_, l) => l) as l (l)}{l ? ' + ' : ''}<span class:dim={l >= sf.terms}>{par(Ag[sf.row][l])}·{par(Bg[l][sf.col])}</span>{/each} = <b>{sf.terms ? sf.partial : '…'}</b>
        {:else}
          <b class="neg">{dims.n} ≠ {dims.q}</b>: a row of the left matrix has {dims.n} entries but a column of the right one has {dims.q}, so they cannot be paired up. Make the two inner sizes equal.
        {/if}
      </p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if MODE === 'compose'}
        <span class="grp" role="group" aria-label="Order of the two matrices">
          <button type="button" class="btn sm" aria-pressed={order === 'AB'} onclick={() => (order = 'AB')}>AB: B first, then A</button>
          <button type="button" class="btn sm" aria-pressed={order === 'BA'} onclick={() => (order = 'BA')}>BA: A first, then B</button>
        </span>
        {@render editor(A, 'A')}{@render presetBtns((m) => (A = m), 'A')}
        {@render editor(B, 'B')}{@render presetBtns((m) => (B = m), 'B')}
      {:else if plane}
        {@render editor(A, 'A')}{@render presetBtns((m) => (A = m), 'A')}
        {#if showVec}
          <F1Slider label="x₁" value={xv[0]} min={-3} max={3} step={0.05} oninput={(v) => { xv = [v, xv[1]]; wake(); }} />
          <F1Slider label="x₂" value={xv[1]} min={-2.5} max={2.5} step={0.05} oninput={(v) => { xv = [xv[0], v]; wake(); }} />
        {/if}
      {:else}
        <F1Slider label="m" value={dims.m} min={1} max={4} step={1} digits={0} oninput={(v) => (dims = { ...dims, m: v })} />
        <F1Slider label="n" value={dims.n} min={1} max={4} step={1} digits={0} oninput={(v) => (dims = { ...dims, n: v })} />
        <F1Slider label="q" value={dims.q} min={1} max={4} step={1} digits={0} oninput={(v) => (dims = { ...dims, q: v })} />
        <F1Slider label="p" value={dims.p} min={1} max={4} step={1} digits={0} oninput={(v) => (dims = { ...dims, p: v })} />
        <span class="grp"><button type="button" class="btn sm" onclick={transposeA}>Transpose left</button><button type="button" class="btn sm" onclick={transposeB}>Transpose right</button></span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.6 var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 700; }
  .pos { color: var(--viz-pos); }
  .neg { color: var(--viz-neg); }
  .dim { color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .grp { display: inline-flex; flex-wrap: wrap; gap: var(--s-2); }
  .sm { padding: 0.3em 0.7em; font-size: var(--step--1); }
  .sm[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); color: var(--text); font-weight: 600; }
  .ed { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .mat { display: grid; grid-template-columns: repeat(2, 6.4rem); gap: 4px; padding: 2px 6px; border-left: 2px solid var(--text); border-right: 2px solid var(--text); border-radius: 3px; }
  .mat input { width: 100%; box-sizing: border-box; font: 0.85rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.25em 0.3em; text-align: right; }
  .mat input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
</style>
