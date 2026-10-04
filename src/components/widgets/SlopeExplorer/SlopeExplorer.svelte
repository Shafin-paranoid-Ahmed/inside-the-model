<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    FNS, SECANT_T, secantMarkers, secantAt, slopeCurve, H0, H_MIN,
    bowl, BOWL_D, BOWL_LEVELS, GRAD_START, GRAD_T, N_STEPS, LR, ARROW, gradMarkers, ballPath, gradAt, type FnId,
  } from './sim';

  interface Props {
    title?: string;
    /** 'secant': shrink h and watch the secant become the tangent. 'gradient': partial derivatives on a bowl. */
    mode?: 'secant' | 'gradient';
    /** Curve to start on (secant mode). */
    fn?: FnId;
    /** Where on the curve to measure the slope (secant mode). */
    x0?: number;
    /** Hide the mode switch. */
    lockMode?: boolean;
    /** Hide the curve picker. */
    lockFn?: boolean;
  }
  let { title = 'Slope: from secant to tangent', mode: mode0 = 'secant', fn: fn0 = 'square', x0: x00, lockMode = false, lockFn = false }: Props = $props();
  const init = untrack(() => ({ mode: mode0, fn: fn0, x0: x00, lockMode, lockFn }));
  const uid = $props.id();

  let mode = $state<'secant' | 'gradient'>(init.mode);
  let fnId = $state<FnId>(init.fn);
  const F = $derived(FNS[fnId]);
  let x0 = $state(init.x0 ?? FNS[init.fn].x0);
  let start = $state<[number, number]>([...GRAD_START]);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let t = 0, wake = () => {}, dragging = false;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const T = $derived(mode === 'secant' ? SECANT_T : GRAD_T);
  const MK = $derived(mode === 'secant' ? secantMarkers() : gradMarkers());
  const path = $derived(ballPath(start));
  const sf = $derived(secantAt(uiT, F, x0));
  const gf = $derived(gradAt(uiT, start, path));
  const narration = $derived(
    mode === 'secant'
      ? sf.h > 0.5 ? 'A wide gap: the secant joins two far-apart points, so its slope is only a rough guess.'
        : sf.h > 0.005 ? 'The second point slides toward the first and the secant rotates toward the tangent.'
        : 'The gap is tiny: the secant and the tangent are the same line, and the slope has settled on the derivative.'
      : uiT < 2 ? 'Nudge x only: the partial derivative in x is the slope along the horizontal direction.'
        : uiT < 4 ? 'Nudge y only: the partial derivative in y is the slope along the vertical direction.'
        : uiT < 6 ? 'Put the two partials side by side and the diagonal is the gradient, pointing straight uphill.'
        : 'The ball steps against the gradient, and each step lands lower on the bowl.',
  );

  $effect(() => {
    mode; fnId; x0; start; // any change to what is measured restarts the animation
    untrack(() => { t = reducedMotion() ? T : 0; playing = !reducedMotion(); wake(); });
  });
  $effect(() => { speed; wake(); });

  function pickFn(id: FnId) { fnId = id; x0 = FNS[id].x0; }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, cache = { key: '', paths: [] as Path2D[] };
    const NG = 80;
    const lay = () => {
      const { main, side } = splitPanels(stage.w, stage.h, stage.w < 540 ? 0.66 : 0.62);
      const m = { l: main.x + 8, r: main.x + main.w - 8, t: main.y + 8, b: main.y + main.h - 8 };
      let box = { l: m.l, r: m.r, t: m.t, b: m.b };
      if (mode === 'gradient') {
        const u = Math.min(m.r - m.l, m.b - m.t), mx = (m.l + m.r) / 2, my = (m.t + m.b) / 2;
        box = { l: mx - u / 2, r: mx + u / 2, t: my - u / 2, b: my + u / 2 };
      }
      return { main, side, box };
    };
    type Lay = ReturnType<typeof lay>;
    const X = (L: Lay) => (mode === 'secant' ? scale(F.xd[0], F.xd[1], L.box.l, L.box.r) : scale(BOWL_D[0], BOWL_D[1], L.box.l, L.box.r));
    const Y = (L: Lay) => (mode === 'secant' ? scale(F.yd[0], F.yd[1], L.box.b, L.box.t) : scale(BOWL_D[0], BOWL_D[1], L.box.b, L.box.t));

    function arrow(ctx: CanvasRenderingContext2D, ax: number, ay: number, bx: number, by: number, col: string, w: number, al = 1) {
      if (al <= 0.02 || Math.hypot(bx - ax, by - ay) < 3) return;
      const ang = Math.atan2(by - ay, bx - ax);
      ctx.globalAlpha = al; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - 9 * Math.cos(ang - 0.4), by - 9 * Math.sin(ang - 0.4)); ctx.lineTo(bx - 9 * Math.cos(ang + 0.4), by - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
    const dot = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, fill: string, stroke: string) => {
      ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };

    function contours(L: Lay) {
      const key = `${L.box.l}|${L.box.r}|${L.box.t}|${L.box.b}`;
      if (cache.key === key) return cache.paths;
      const span = BOWL_D[1] - BOWL_D[0], g = (i: number) => BOWL_D[0] + (span * i) / (NG - 1);
      const vals = new Float64Array(NG * NG);
      for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) vals[j * NG + i] = bowl(g(i), g(j));
      const px = X(L), py = Y(L);
      cache = { key, paths: BOWL_LEVELS.map((lv) => {
        const seg = contour((i, j) => vals[j * NG + i], NG, NG, lv), p = new Path2D();
        for (let q = 0; q < seg.length; q += 4) { p.moveTo(px(g(seg[q])), py(g(seg[q + 1]))); p.lineTo(px(g(seg[q + 2])), py(g(seg[q + 3]))); }
        return p;
      }) };
      return cache.paths;
    }

    function sidePanel(L: Lay, heading: string, xlab: string) {
      const { ctx, c } = stage, p = L.side, pad = { l: 12, r: 12, t: 26, b: 30 };
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(heading, p.x + pad.l, p.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x + pad.l, p.y + p.h - pad.b); ctx.lineTo(p.x + p.w - pad.r, p.y + p.h - pad.b); ctx.stroke();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(xlab, p.x + p.w - pad.r, p.y + p.h - 1);
      return { x0: p.x + pad.l, x1: p.x + p.w - pad.r, yb: p.y + p.h - pad.b, yt: p.y + pad.t };
    }

    function renderSecant(L: Lay) {
      const { ctx, c } = stage, px = X(L), py = Y(L), b = L.box, fr = secantAt(t, F, x0);
      ctx.save(); ctx.beginPath(); ctx.rect(b.l, b.t, b.r - b.l, b.b - b.t); ctx.clip();
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
      const N = 160;
      for (let i = 0; i <= N; i++) { const x = F.xd[0] + ((F.xd[1] - F.xd[0]) * i) / N; if (i) ctx.lineTo(px(x), py(F.f(x))); else ctx.moveTo(px(x), py(F.f(x))); }
      ctx.stroke();
      const line = (m: number, col: string, w: number, dash: number[], al: number) => {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.globalAlpha = al; ctx.setLineDash(dash); ctx.beginPath();
        ctx.moveTo(px(F.xd[0]), py(fr.y0 + m * (F.xd[0] - x0))); ctx.lineTo(px(F.xd[1]), py(fr.y0 + m * (F.xd[1] - x0))); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      };
      line(fr.tangent, c.grad.hex, 1.6, [5, 4], 0.85);
      line(fr.secant, c.fwd.hex, 2.4, [], 1);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.setLineDash([2, 3]); ctx.beginPath();
      ctx.moveTo(px(x0), py(fr.y0)); ctx.lineTo(px(fr.x1), py(fr.y0)); ctx.lineTo(px(fr.x1), py(fr.y1)); ctx.stroke(); ctx.setLineDash([]);
      if (fr.h > 0.15) {
        ctx.font = stage.mono(12); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText('h', (px(x0) + px(fr.x1)) / 2, py(fr.y0) + 4);
      }
      dot(ctx, px(fr.x1), py(fr.y1), 5, c.raised.hex, c.fwd.hex);
      dot(ctx, px(x0), py(fr.y0), 6.5, c.accent.hex, c.raised.hex);
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(px(x0), py(fr.y0), 12, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
      ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(b.l, b.t, b.r - b.l, b.b - b.t);
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('drag the ringed point along the curve', b.l + 6, b.t + 5);
      // side: secant slope against h
      const S = sidePanel(L, 'Secant slope as h shrinks', 'h, log scale, shrinking →');
      const cur = slopeCurve(F, x0), ss = cur.map((q) => q.s).concat(fr.tangent), lo = Math.min(...ss), hi = Math.max(...ss), pd = (hi - lo) * 0.15 + 1e-9;
      const sx = (h: number) => S.x0 + ((Math.log(H0) - Math.log(h)) / (Math.log(H0) - Math.log(H_MIN))) * (S.x1 - S.x0), sy = scale(lo - pd, hi + pd, S.yb, S.yt);
      ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(S.x0, sy(fr.tangent)); ctx.lineTo(S.x1, sy(fr.tangent)); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2; ctx.beginPath();
      cur.forEach((q, i) => (i ? ctx.lineTo(sx(q.h), sy(q.s)) : ctx.moveTo(sx(q.h), sy(q.s))));
      ctx.stroke();
      dot(ctx, sx(fr.h), sy(fr.secant), 5, c.fwd.hex, c.raised.hex);
      ctx.font = stage.body(11.5, true); ctx.fillStyle = c.grad.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`true slope ${f2(fr.tangent, 3)}`, S.x0 + 2, sy(fr.tangent) - 4);
    }

    function renderGrad(L: Lay) {
      const { ctx, c } = stage, px = X(L), py = Y(L), b = L.box, fr = gradAt(t, start, path);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(b.l, b.t, b.r - b.l, b.b - b.t);
      ctx.save(); ctx.beginPath(); ctx.rect(b.l, b.t, b.r - b.l, b.b - b.t); ctx.clip();
      contours(L).forEach((p, i, a) => { ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.3 + 0.5 * (1 - i / a.length); ctx.lineWidth = 1.3; ctx.stroke(p); });
      ctx.globalAlpha = 1;
      // pixel offsets of the full-size partial arrows (dy is negative when the gradient points up the screen)
      const ax = px(fr.px), ay = py(fr.py), dx = px(fr.px + ARROW * fr.gx) - ax, dy = py(fr.py + ARROW * fr.gy) - ay;
      const fade = 1 - 0.65 * fr.ball.alpha;
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.globalAlpha = fr.slide * fade; ctx.beginPath();
      ctx.moveTo(ax + dx, ay); ctx.lineTo(ax + dx, ay + dy); ctx.moveTo(ax, ay + dy); ctx.lineTo(ax + dx, ay + dy); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      arrow(ctx, ax, ay, ax + dx * fr.ex, ay, c.fwd.hex, 3, fade);
      arrow(ctx, ax + dx * fr.slide, ay, ax + dx * fr.slide, ay + dy * fr.ey, c.pos.hex, 3, fade);
      arrow(ctx, ax, ay, ax + dx * fr.eg, ay + dy * fr.eg, c.grad.hex, 3.4, fade);
      const bx = px(fr.ball.x), by = py(fr.ball.y);
      if (fr.ball.alpha > 0.02) {
        ctx.strokeStyle = c.text.hex; ctx.globalAlpha = 0.5 * fr.ball.alpha; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px(path[0][0]), py(path[0][1]));
        for (let i = 1; i <= Math.floor(fr.ball.k); i++) ctx.lineTo(px(path[i][0]), py(path[i][1]));
        ctx.lineTo(bx, by); ctx.stroke();
        ctx.globalAlpha = fr.ball.alpha; dot(ctx, bx, by, 7, c.accent.hex, c.raised.hex); ctx.globalAlpha = 1;
      }
      ctx.restore();
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(ax, ay, 11, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('drag the dashed ring to move the point', b.l + 6, b.t + 5);
      ctx.font = stage.mono(12, 600); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillStyle = c.fwd.hex; if (fr.ex > 0.6 && fr.ball.alpha < 0.5) ctx.fillText('∂f/∂x', ax + 4, ay + 16);
      ctx.fillStyle = c.pos.hex; if (fr.ey > 0.6 && fr.slide < 0.5) ctx.fillText('∂f/∂y', ax + 10, ay + dy * fr.ey * 0.6);
      ctx.fillStyle = c.grad.hex; if (fr.eg > 0.7 && fr.ball.alpha < 0.5) ctx.fillText('∇f', ax + dx * 0.55 - 28, ay + dy * 0.55);
      // side: height of the ball against step
      const S = sidePanel(L, 'Height of the ball at each step', 'step');
      const hs = path.map((q) => bowl(q[0], q[1])), sx = scale(0, N_STEPS, S.x0, S.x1), sy = scale(0, hs[0] * 1.05 + 1e-9, S.yb, S.yt);
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2; ctx.beginPath();
      hs.forEach((v, i) => (i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))));
      ctx.stroke();
      ctx.globalAlpha = fr.ball.alpha; dot(ctx, sx(fr.ball.k), sy(fr.ball.f), 5, c.accent.hex, c.raised.hex); ctx.globalAlpha = 1;
    }

    function render() {
      const { ctx, w: W, h: H } = stage, L = lay();
      ctx.clearRect(0, 0, W, H);
      if (mode === 'secant') renderSecant(L); else renderGrad(L);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T, loopFrom: 0, loopTo: T + 1.2 }).t;
      const tt = Math.min(t, T);
      render(); tl.paint(tt);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = tt; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const world = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top, b = L.box;
      const d = mode === 'secant' ? F.xd : BOWL_D;
      return { x, y, L, wx: d[0] + ((x - b.l) / (b.r - b.l)) * (d[1] - d[0]), wy: BOWL_D[1] - ((y - b.t) / (b.b - b.t)) * (BOWL_D[1] - BOWL_D[0]) };
    };
    const handle = (p: ReturnType<typeof world>) =>
      mode === 'secant' ? { x: X(p.L)(x0), y: Y(p.L)(F.f(x0)) } : { x: X(p.L)(start[0]), y: Y(p.L)(start[1]) };
    const down = (e: PointerEvent) => {
      const p = world(e), h = handle(p);
      if (Math.hypot(h.x - p.x, h.y - p.y) < 22) { dragging = true; cv.setPointerCapture(e.pointerId); e.preventDefault(); }
    };
    const move = (e: PointerEvent) => {
      const p = world(e);
      if (dragging) {
        if (mode === 'secant') x0 = clamp(p.wx, F.xd[0] + 0.1, F.xd[1] - 1.5);
        else start = [clamp(p.wx, -2.9, 2.9), clamp(p.wy, -2.9, 2.9)];
      } else { const h = handle(p); cv.style.cursor = Math.hypot(h.x - p.x, h.y - p.y) < 22 ? 'grab' : ''; }
    };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard {title} role="group" aspect="720 / 420" narrowAspect="360 / 640"
  label={mode === 'secant'
    ? 'Left: a curve with a fixed point and a second point sliding toward it, with the secant line through both and the tangent line dashed. Right: the secant slope plotted against the gap h, converging to the true slope.'
    : 'Left: contour map of a bowl with arrows for the two partial derivatives at a point, combining into the gradient arrow that points uphill, and a ball stepping downhill. Right: the height of the ball at each step.'}
  legend={mode === 'secant'
    ? [{ label: 'point', color: '--accent' }, { label: 'secant', color: '--viz-forward', shape: 'line' }, { label: 'tangent', color: '--viz-gradient', shape: 'line' }]
    : [{ label: '∂f/∂x', color: '--viz-forward', shape: 'line' }, { label: '∂f/∂y', color: '--viz-pos', shape: 'line' }, { label: 'gradient', color: '--viz-gradient', shape: 'line' }, { label: 'ball', color: '--accent' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK}
      label={mode === 'secant' ? 'Timeline: shrinking the gap h' : 'Timeline: partials, gradient, descent'}
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if mode === 'secant'}
      <p class="vals num">h = {f2(sf.h, 3)} · secant slope {f2(sf.secant, 3)} · true slope f′({f2(x0)}) = {f2(sf.tangent, 3)} · gap {f2(sf.gap, 3)}</p>
    {:else}
      <p class="vals num">at ({f2(start[0])}, {f2(start[1])}): ∂f/∂x = {f2(gf.gx)} · ∂f/∂y = {f2(gf.gy)} · |∇f| = {f2(Math.hypot(gf.gx, gf.gy))} · ball step {gf.ball.k} of {N_STEPS} (η = {LR}) · height {f2(gf.ball.f, 3)}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if !init.lockMode}
        <fieldset class="seg">
          <legend class="sr-only">View</legend>
          {#each [['secant', 'One variable'], ['gradient', 'Two variables']] as [id, nm] (id)}
            <label><input type="radio" name="mode-{uid}" value={id} checked={mode === id} onchange={() => (mode = id as 'secant' | 'gradient')} /><span>{nm}</span></label>
          {/each}
        </fieldset>
      {/if}
      {#if mode === 'secant'}
        {#if !init.lockFn}
          <fieldset class="seg">
            <legend class="sr-only">Curve</legend>
            {#each Object.values(FNS) as s (s.id)}
              <label><input type="radio" name="fn-{uid}" value={s.id} checked={fnId === s.id} onchange={() => pickFn(s.id)} /><span>{s.name}</span></label>
            {/each}
          </fieldset>
        {/if}
        <F1Slider label="x" value={x0} min={F.xd[0] + 0.1} max={F.xd[1] - 1.5} step={0.05} oninput={(v) => (x0 = v)} />
      {:else}
        <F1Slider label="x" value={start[0]} min={-2.9} max={2.9} step={0.05} oninput={(v) => (start = [v, start[1]])} />
        <F1Slider label="y" value={start[1]} min={-2.9} max={2.9} step={0.05} oninput={(v) => (start = [start[0], v])} />
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.5 var(--font-code); font-variant-numeric: tabular-nums; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
