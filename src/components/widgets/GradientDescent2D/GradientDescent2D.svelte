<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, contour, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { SURFACES, STEPS, duration, markers, trajectory, frameAt, slice, type SurfaceId } from './sim';

  interface Props {
    title?: string;
    /** Loss surface to start on. */
    surface?: SurfaceId;
    /** Learning rate to start with (default: the surface's own). */
    lr?: number;
    /** Start with noisy (SGD-like) gradients. */
    sgd?: boolean;
    /** Hide the surface picker (the lesson step fixes the surface). */
    lockSurface?: boolean;
    /**
     * One job per lesson figure. roll: the map only. starts: also faint paths from other starts. slice: the side-on
     * slice along the gradient. update: loss per step and the update arithmetic. full: slice plus every control.
     */
    view?: 'roll' | 'starts' | 'slice' | 'update' | 'full';
  }
  let { title = 'Gradient descent: a ball rolling downhill', surface = 'bowl', lr: lr0, sgd: sgd0 = false, lockSurface = false, view = 'full' }: Props = $props();
  const init = untrack(() => ({ surface, lr: lr0, sgd: sgd0, lock: lockSurface, view }));
  const V = init.view, hasSide = V === 'slice' || V === 'update' || V === 'full';
  /** Other starting points drawn as faint full paths in the 'starts' view. */
  const OTHER_STARTS: [number, number][] = [[-3.5, -1.8], [-3.2, 1.9], [0.6, -2], [3.6, -0.6]];
  const T = duration(), MK = markers();
  const uid = $props.id();

  let surfId = $state<SurfaceId>(init.surface);
  const S = $derived(SURFACES[surfId]);
  let lr = $state(init.lr ?? SURFACES[init.surface].lr);
  let sgd = $state(init.sgd);
  let start = $state<[number, number]>([...SURFACES[init.surface].start]);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {}, dragging = false;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const traj = $derived(trajectory(S, start, lr, sgd, 3));
  const k = $derived(Math.min(STEPS, Math.floor(uiT / (T / STEPS))));
  const cur = $derived(traj[k]);
  const status = $derived(
    k > 3 && cur.f > 3 * traj[0].f + 5 && cur.f > traj[k - 1].f ? 'diverging' : Math.hypot(cur.tx, cur.ty) < 0.05 ? 'settled' : 'descending',
  );
  const narration = $derived(
    status === 'diverging' ? 'Diverging: every step overshoots the valley and lands higher than it started. The learning rate is too large.'
      : status === 'settled' ? 'Settled: the gradient is almost zero, so the steps are almost zero.'
      : sgd ? 'Descending with noisy gradients: each step points roughly downhill, plus a random kick.' : 'Descending: each step is the learning rate times the gradient, pointing downhill.',
  );
  const crit = $derived(S.hmax ? 2 / S.hmax : null);
  const others = $derived(V === 'starts' ? OTHER_STARTS.map((q) => trajectory(S, q, lr, false)) : []);

  $effect(() => {
    traj; // any change of surface, start, learning rate or noise restarts the descent
    untrack(() => { t = reducedMotion() ? T : 0; playing = !reducedMotion(); wake(); });
  });
  $effect(() => { speed; wake(); });

  function pickSurface(id: SurfaceId) { surfId = id; start = [...SURFACES[id].start]; lr = SURFACES[id].lr; }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, cache = { key: '', paths: [] as Path2D[] };
    const NG = 90;
    const lay = () => {
      const { main, side, stacked } = hasSide ? splitPanels(stage.w, stage.h, stage.w < 540 ? 0.66 : 0.62)
        : { main: { x: 0, y: 0, w: stage.w, h: stage.h }, side: { x: 0, y: 0, w: 0, h: 0 }, stacked: false };
      const xd = S.xd, yd = S.yd, m = { l: main.x + 6, r: main.x + main.w - 6, t: main.y + 6, b: main.y + main.h - 6 };
      const u = Math.min((m.r - m.l) / (xd[1] - xd[0]), (m.b - m.t) / (yd[1] - yd[0]));
      const cx = (m.l + m.r) / 2, cy = (m.t + m.b) / 2, mx = (xd[0] + xd[1]) / 2, my = (yd[0] + yd[1]) / 2;
      const px = (x: number) => cx + (x - mx) * u, py = (y: number) => cy - (y - my) * u;
      const box = { l: px(xd[0]), r: px(xd[1]), t: py(yd[1]), b: py(yd[0]) };
      return { main, side, stacked, px, py, u, box };
    };
    function contours(L: ReturnType<typeof lay>) {
      const key = `${S.id}|${L.box.l}|${L.box.r}|${L.box.t}|${L.box.b}`;
      if (cache.key === key) return cache.paths;
      const gx = (i: number) => S.xd[0] + ((S.xd[1] - S.xd[0]) * i) / (NG - 1), gy = (j: number) => S.yd[0] + ((S.yd[1] - S.yd[0]) * j) / (NG - 1);
      const vals = new Float64Array(NG * NG);
      for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) vals[j * NG + i] = S.f(gx(i), gy(j));
      cache = { key, paths: S.levels.map((lv) => {
        const seg = contour((i, j) => vals[j * NG + i], NG, NG, lv), p = new Path2D();
        for (let q = 0; q < seg.length; q += 4) {
          p.moveTo(L.px(gx(seg[q])), L.py(gy(seg[q + 1]))); p.lineTo(L.px(gx(seg[q + 2])), L.py(gy(seg[q + 3])));
        }
        return p;
      }) };
      return cache.paths;
    }
    const cl = (v: number) => clamp(v, -80, 80);

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), fr = frameAt(t, traj, lr);
      ctx.clearRect(0, 0, W, H);
      const b = L.box;
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex; ctx.strokeRect(b.l, b.t, b.r - b.l, b.b - b.t);
      ctx.save(); ctx.beginPath(); ctx.rect(b.l, b.t, b.r - b.l, b.b - b.t); ctx.clip();
      contours(L).forEach((p, i, a) => { ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.3 + 0.5 * (1 - i / a.length); ctx.lineWidth = 1.3; ctx.stroke(p); });
      ctx.globalAlpha = 1;
      // faint paths from other starts: every one ends at the same bottom
      for (const o of others) {
        ctx.strokeStyle = c.muted.hex; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.3; ctx.setLineDash([4, 3]); ctx.beginPath();
        o.forEach((q, i) => (i ? ctx.lineTo(L.px(cl(q.x)), L.py(cl(q.y))) : ctx.moveTo(L.px(cl(q.x)), L.py(cl(q.y)))));
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = c.muted.hex; ctx.beginPath(); ctx.arc(L.px(o[0].x), L.py(o[0].y), 4, 0, 6.2832); ctx.fill(); ctx.globalAlpha = 1;
      }
      // trail
      ctx.strokeStyle = c.text.hex; ctx.globalAlpha = 0.55; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(L.px(cl(traj[0].x)), L.py(cl(traj[0].y)));
      for (let i = 1; i <= fr.k; i++) ctx.lineTo(L.px(cl(traj[i].x)), L.py(cl(traj[i].y)));
      ctx.lineTo(L.px(fr.x), L.py(fr.y)); ctx.stroke(); ctx.globalAlpha = 1;
      for (let i = 0; i <= fr.k; i++) { ctx.fillStyle = c.text.hex; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.arc(L.px(cl(traj[i].x)), L.py(cl(traj[i].y)), 2.4, 0, 6.2832); ctx.fill(); }
      ctx.globalAlpha = 1;
      // gradient arrows at the current step: faint true gradient (SGD only), bold step actually taken
      const a = fr.arrow, st = traj[fr.k], arrow = (dx: number, dy: number, col: string, w: number, al: number) => {
        if (al <= 0.02 || Math.hypot(dx, dy) < 1e-6) return;
        const x0 = L.px(cl(a.x)), y0 = L.py(cl(a.y)), x1 = L.px(cl(a.x + dx)), y1 = L.py(cl(a.y + dy)), ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.globalAlpha = al; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4)); ctx.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
      };
      if (sgd) arrow(-st.tx * lr, -st.ty * lr, c.muted.hex, 2, a.alpha * 0.8);
      arrow(-a.gx, -a.gy, c.grad.hex, 3, a.alpha);
      // ball
      ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(L.px(fr.x), L.py(fr.y), 7, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.restore();
      // start handle
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(L.px(start[0]), L.py(start[1]), 11, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('drag the dashed ring to move the start', b.l + 6, b.t + 5);
      if (fr.diverged) { ctx.font = stage.mono(13, 700); ctx.fillStyle = c.neg.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('diverged: loss is exploding', (b.l + b.r) / 2, (b.t + b.b) / 2); }

      if (!hasSide) return;
      const p = L.side, pad = { l: 12, r: 12, t: 26, b: 30 };
      if (V === 'update') {
        // ---- loss after each step, log scale ----
        const lo = -4, hi = Math.max(1, Math.ceil(Math.log10(Math.max(...traj.map((q) => q.f), 1))));
        const lx = scale(0, STEPS, p.x + pad.l + 22, p.x + p.w - pad.r), ly = (f: number) => scale(lo, hi, p.y + p.h - pad.b, p.y + pad.t)(clamp(Math.log10(Math.max(f, 1e-9)), lo, hi));
        ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('Loss after each step (log scale)', p.x + pad.l, p.y + 2);
        ctx.font = stage.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let e = lo; e <= hi; e += 2) { ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(lx(0), ly(10 ** e)); ctx.lineTo(lx(STEPS), ly(10 ** e)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(e === 0 ? '1' : `1e${e}`, lx(0) - 4, ly(10 ** e)); }
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        for (const k of [0, 12, 24, 36, 48]) ctx.fillText(String(k), lx(k), p.y + p.h - pad.b + 5);
        ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('step', p.x + p.w - pad.r, p.y + p.h - 1);
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 2; ctx.beginPath(); traj.forEach((q, i) => (i ? ctx.lineTo(lx(i), ly(q.f)) : ctx.moveTo(lx(i), ly(q.f)))); ctx.stroke();
        ctx.strokeStyle = c.grad.hex; ctx.beginPath(); for (let i = 0; i <= fr.k; i++) (i ? ctx.lineTo(lx(i), ly(traj[i].f)) : ctx.moveTo(lx(i), ly(traj[i].f))); ctx.lineTo(lx(fr.k + fr.u), ly(fr.f)); ctx.stroke();
        ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.beginPath(); ctx.arc(lx(fr.k + fr.u), ly(fr.f), 6, 0, 6.2832); ctx.fill(); ctx.stroke();
        return;
      }
      // ---- slice along the downhill direction ----
      const sl = slice(S, a.x, a.y, st.tx, st.ty, 2.5), vmin = Math.min(...sl.v), vmax = Math.max(...sl.v);
      const sx = scale(-2.5, 2.5, p.x + pad.l, p.x + p.w - pad.r), sy = scale(vmin, vmax + (vmax - vmin) * 0.05, p.y + p.h - pad.b, p.y + pad.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Loss along the downhill line', p.x + pad.l, p.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x + pad.l, p.y + p.h - pad.b); ctx.lineTo(p.x + p.w - pad.r, p.y + p.h - pad.b); ctx.stroke();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('distance along −gradient', p.x + p.w - pad.r, p.y + p.h - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(p.x + pad.l - 2, p.y + pad.t - 4, p.w - pad.l - pad.r + 4, p.h - pad.t - pad.b + 8); ctx.clip();
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2; ctx.beginPath();
      sl.s.forEach((q, i) => (i ? ctx.lineTo(sx(q), sy(sl.v[i])) : ctx.moveTo(sx(q), sy(sl.v[i]))));
      ctx.stroke();
      const g = Math.hypot(st.tx, st.ty), f0 = S.f(a.x, a.y);
      ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(sx(-2.5), sy(f0 + 2.5 * g)); ctx.lineTo(sx(2.5), sy(f0 - 2.5 * g)); ctx.stroke(); ctx.setLineDash([]);
      const land = (-a.gx) * sl.dir[0] + (-a.gy) * sl.dir[1], fl = S.f(a.x + sl.dir[0] * land, a.y + sl.dir[1] * land);
      ctx.strokeStyle = c.grad.hex; ctx.globalAlpha = 0.6; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(sx(0), sy(f0)); ctx.lineTo(sx(clamp(land, -2.5, 2.5)), sy(f0)); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx(clamp(land, -2.5, 2.5)), sy(fl), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.beginPath(); ctx.arc(sx(0), sy(f0), 6, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.restore();
      ctx.font = stage.body(11.5, true); ctx.fillStyle = c.grad.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('open dot: where the next step lands', p.x + pad.l, p.y + p.h - pad.b + 6);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T, loopFrom: 0, loopTo: T + 1.2 }).t;
      const tt = Math.min(t, T);
      render(); tl.paint(tt);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = tt; }
      return playing || dragging;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const pt = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top;
      return { x, y, L, wx: S.xd[0] + ((x - L.box.l) / (L.box.r - L.box.l)) * (S.xd[1] - S.xd[0]), wy: S.yd[1] - ((y - L.box.t) / (L.box.b - L.box.t)) * (S.yd[1] - S.yd[0]) };
    };
    const down = (e: PointerEvent) => {
      const p = pt(e);
      if (Math.hypot(p.L.px(start[0]) - p.x, p.L.py(start[1]) - p.y) < 20) { dragging = true; cv.setPointerCapture(e.pointerId); e.preventDefault(); }
    };
    const move = (e: PointerEvent) => {
      const p = pt(e);
      if (dragging) start = [clamp(p.wx, S.xd[0] + 0.05, S.xd[1] - 0.05), clamp(p.wy, S.yd[0] + 0.05, S.yd[1] - 0.05)];
      else cv.style.cursor = Math.hypot(p.L.px(start[0]) - p.x, p.L.py(start[1]) - p.y) < 20 ? 'grab' : '';
    };
    const up = () => { dragging = false; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard {title} role="group" aspect={hasSide ? '720 / 420' : '720 / 380'} narrowAspect={hasSide ? '360 / 640' : '360 / 300'}
  label={`Contour map of a loss surface with a ball descending it, a trail of past steps and an arrow for the current gradient step.${V === 'starts' ? ' Faint dashed paths show descent from four other starts.' : ''}${V === 'update' ? ' Right: the loss after each step on a log scale.' : hasSide ? ' Right: a slice of the loss along the downhill direction, showing where the next step lands.' : ''}`}
  legend={[{ label: 'ball', color: '--accent' }, { label: 'gradient step', color: '--viz-gradient', shape: 'line' }, { label: 'contours (equal loss)', color: '--viz-seq-4', shape: 'line' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: steps of gradient descent"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if V === 'update'}
      <p class="vals num">θ = ({f2(cur.x)}, {f2(cur.y)}) · ∇L = ({f2(cur.tx)}, {f2(cur.ty)}) · θ − η∇L = ({f2(cur.x - lr * cur.tx)}, {f2(cur.y - lr * cur.ty)})
        <span class="base">· η = {f2(lr)} · step {k} · loss {f2(cur.f, 3)}</span></p>
    {:else}
    <p class="vals num">step {k} of {STEPS} · position ({f2(cur.x)}, {f2(cur.y)}) · loss {f2(cur.f, 3)} · |gradient| {f2(Math.hypot(cur.tx, cur.ty))}
      {#if crit && V === 'full'}<span class="base">· on this bowl, η above {f2(crit)} diverges</span>{/if}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if V === 'full'}
      <F1Slider label="η" value={lr} min={0.01} max={S.hmax ? 0.32 : 0.6} step={0.005} digits={3} oninput={(v) => (lr = v)} />
      <label class="chk"><input type="checkbox" bind:checked={sgd} /> Noisy gradient (SGD)</label>
      {/if}
      {#if !init.lock}
        <fieldset class="seg">
          <legend class="sr-only">Loss surface</legend>
          {#each Object.values(SURFACES) as s (s.id)}
            <label><input type="radio" name="surf-{uid}" value={s.id} checked={surfId === s.id} onchange={() => pickSurface(s.id)} /><span>{s.name}</span></label>
          {/each}
        </fieldset>
      {/if}
      <F1Slider label="start x" value={start[0]} min={S.xd[0] + 0.05} max={S.xd[1] - 0.05} step={0.05} oninput={(v) => (start = [v, start[1]])} />
      <F1Slider label="start y" value={start[1]} min={S.yd[0] + 0.05} max={S.yd[1] - 0.05} step={0.05} oninput={(v) => (start = [start[0], v])} />
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.5 var(--font-code); font-variant-numeric: tabular-nums; }
  .base { color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); cursor: pointer; }
  .chk input { accent-color: var(--accent); }
</style>
