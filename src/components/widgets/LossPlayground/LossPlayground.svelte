<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    points, CLASS_PTS, sigmoid, logLoss1, residuals, lossValue, curve, W_RANGE, B_RANGE, duration, markers, frameAt,
    type Task, type Loss, type Frame,
  } from './sim';

  interface Props {
    title?: string;
    /** 'regression': a line through points, squared or absolute error. 'classification': a sigmoid and log loss. */
    task?: Task;
    /** Regression loss shown as squares and on the curve. */
    loss?: 'mse' | 'mae';
  }
  let { title = '', task = 'regression', loss = 'mse' }: Props = $props();
  const init = untrack(() => ({ task, loss, title: title || (task === 'regression' ? 'Drag the line: how wrong is it?' : 'Drag the curve: how surprised is the model?') }));
  const T = duration(init.task), MK = markers(init.task), isReg = init.task === 'regression';

  const uid = $props.id();
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let kind = $state<Loss>(isReg ? init.loss : 'logloss');
  /** Knobs the user set by hand; null while the timeline drives them. */
  let over = $state<{ w: number; b: number } | null>(null);
  let ui = $state<Frame>(frameAt(init.task, 0));
  let outOn = $state(false);
  let t = 0, amtCur = 0, amtTgt = 0, drag = -1, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const phase = $derived(over ? 'manual' : (markerAt(uiT, MK)?.label ?? ''));
  const cur = $derived({ mse: lossValue('mse', ui.w, ui.b, ui.amt), mae: lossValue('mae', ui.w, ui.b, ui.amt), log: lossValue('logloss', ui.w, ui.b, 0) });
  const base = $derived(ui.amt > 0.02 ? { mse: lossValue('mse', ui.w, ui.b, 0), mae: lossValue('mae', ui.w, ui.b, 0) } : null);
  const narration = $derived(
    phase === 'manual' ? 'You are moving the model by hand.' : !isReg ? 'The curve sharpens: confident correct answers cost little, confident mistakes cost a lot.'
      : phase === 'tilt the line' ? 'The line tilts away from the best fit and back: watch the loss curve on the right.'
      : phase === 'add an outlier' ? 'One point slides away from the trend. The line has not moved, but the loss jumps.' : 'The outlier comes back to the trend.',
  );
  $effect(() => { if (playing) over = null; speed; kind; wake(); });

  const frame = (): Frame => (over ? { ...over, amt: amtCur } : frameAt(init.task, t));
  function take() {
    if (over) return;
    const f = frameAt(init.task, t);
    over = { w: f.w, b: f.b }; amtCur = amtTgt = f.amt; playing = false;
  }
  const setW = (v: number) => { take(); over = { ...over!, w: v }; sync(); wake(); };
  const setB = (v: number) => { take(); over = { ...over!, b: v }; sync(); wake(); };
  const toggleOut = (on: boolean) => { take(); amtTgt = on ? 1 : 0; outOn = on; wake(); };
  const sync = () => { ui = frame(); uiT = t; };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const X = isReg ? ([-4.5, 4.5] as const) : ([-4, 4] as const), Y = isReg ? ([-4, 6.5] as const) : ([-0.15, 1.15] as const);
    const HX = isReg ? [-4, 4] : [0];

    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, 0.6);
      const sx = scale(X[0], X[1], main.x + 34, main.x + main.w - 10), sy = scale(Y[0], Y[1], main.y + main.h - 26, main.y + 10);
      return { main, side, stacked, sx, sy };
    };
    const Ymax = { mse: 12, mae: 3.5, logloss: 3 };

    function render() {
      const { ctx, w: W, h: H, c } = stage;
      const { main, side, sx, sy } = lay(), f = frame();
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // ---- main panel ----
      const m = { l: main.x + 34, r: main.x + main.w - 10, t: main.y + 10, b: main.y + main.h - 26 };
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const x of [-4, -2, 0, 2, 4]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(sx(x), m.t); ctx.lineTo(sx(x), m.b); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(x).replace('-', '−'), sx(x), m.b + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const y of isReg ? [-4, -2, 0, 2, 4, 6] : [0, 0.5, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(m.l, sy(y)); ctx.lineTo(m.r, sy(y)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(y).replace('-', '−'), m.l - 6, sy(y)); }
      ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, m.r - m.l, m.b - m.t); ctx.clip();
      const ppu = Math.abs(sy(1) - sy(0));
      if (isReg) {
        const pts = points(f.amt), res = residuals(f.w, f.b, pts);
        ctx.lineCap = 'round';
        pts.forEach((p, i) => {
          const r = res[i], yl = f.w * p.x + f.b, col = r >= 0 ? c.pos : c.neg, out = i === pts.length - 1 && pts.length > 8;
          if (kind === 'mse') {
            const s = Math.abs(r) * ppu;
            ctx.fillStyle = `rgba(${col.rgb.join(',')},0.2)`; ctx.strokeStyle = `rgba(${col.rgb.join(',')},0.7)`; ctx.lineWidth = 1;
            const y0 = Math.min(sy(p.y), sy(yl));
            ctx.beginPath(); ctx.rect(sx(p.x), y0, s, s); ctx.fill(); ctx.stroke();
          }
          ctx.strokeStyle = col.hex; ctx.lineWidth = kind === 'mae' ? 3 : 1.6;
          ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.y)); ctx.lineTo(sx(p.x), sy(yl)); ctx.stroke();
          ctx.fillStyle = out ? c.neg.hex : c.text.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), out ? 6 : 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        });
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(sx(X[0]), sy(f.w * X[0] + f.b)); ctx.lineTo(sx(X[1]), sy(f.w * X[1] + f.b)); ctx.stroke();
      } else {
        // sigmoid curve and per-point loss segments (thicker = costlier)
        for (const p of CLASS_PTS) {
          const pr = sigmoid(f.w * p.x + f.b), l = logLoss1(p.label, pr);
          ctx.strokeStyle = c.neg.hex; ctx.globalAlpha = 0.85; ctx.lineWidth = 1 + 5 * Math.min(1, l / 3);
          ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.label)); ctx.lineTo(sx(p.x), sy(pr)); ctx.stroke(); ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.beginPath();
        for (let i = 0; i <= 80; i++) { const x = X[0] + ((X[1] - X[0]) * i) / 80, y = sigmoid(f.w * x + f.b); i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)); }
        ctx.stroke();
        for (const p of CLASS_PTS) {
          ctx.fillStyle = p.label ? c.accent.hex : c.raised.hex; ctx.strokeStyle = p.label ? c.raised.hex : c.text.hex; ctx.lineWidth = p.label ? 1.5 : 2;
          ctx.beginPath(); ctx.arc(sx(p.x), sy(p.label), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        }
      }
      ctx.restore();
      // handles
      for (const hx of HX) {
        const hy = isReg ? f.w * hx + f.b : 0.5, px = isReg ? sx(hx) : sx(Math.abs(f.w) > 0.05 ? clamp(-f.b / f.w, X[0], X[1]) : 0);
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(px, clamp(sy(hy), m.t, m.b), 8, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(isReg ? 'x' : 'x (feature)', m.r, main.y + main.h - 1);
      if (!isReg) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('thicker red bar = costlier point', m.l + 4, m.t + 2); }

      // ---- side panel: loss against w, b held fixed ----
      const s = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 26, b: side.y + side.h - 30 };
      const [w0, w1] = W_RANGE[isReg ? 'regression' : 'classification'];
      const cv2 = curve(kind, f.b, f.amt, 81), ym = Ymax[kind];
      const cx = scale(w0, w1, s.l, s.r), cy = scale(0, ym, s.b, s.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`${kind === 'mse' ? 'MSE' : kind === 'mae' ? 'MAE' : 'log loss'} as w changes (b fixed)`, s.l - 30, side.y + 2);
      ctx.font = stage.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, ym / 2, ym]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(s.l, cy(v)); ctx.lineTo(s.r, cy(v)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(v % 1 ? v.toFixed(1) : String(v), s.l - 5, cy(v)); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let v = Math.ceil(w0); v <= w1; v += isReg ? 1 : 2) { ctx.fillStyle = c.muted.hex; ctx.fillText(String(v).replace('-', '−'), cx(v), s.b + 5); }
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = c.muted.hex; ctx.fillText('w', s.r, side.y + side.h - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t - 4, s.r - s.l, s.b - s.t + 4); ctx.clip();
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      cv2.w.forEach((w, i) => (i ? ctx.lineTo(cx(w), cy(cv2.v[i])) : ctx.moveTo(cx(w), cy(cv2.v[i]))));
      ctx.stroke(); ctx.restore();
      const vNow = lossValue(kind, f.w, f.b, f.amt), px = cx(clamp(f.w, w0, w1));
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(px, s.b); ctx.lineTo(px, clamp(cy(vNow), s.t, s.b)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, clamp(cy(vNow), s.t, s.b), 6, 0, 6.2832); ctx.fill(); ctx.stroke();
      const bx = cx(cv2.best), bv = clamp(cy(lossValue(kind, cv2.best, f.b, f.amt)), s.t, s.b);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx, bv - 6); ctx.lineTo(bx + 6, bv); ctx.lineTo(bx, bv + 6); ctx.lineTo(bx - 6, bv); ctx.closePath(); ctx.stroke();
      ctx.font = stage.mono(11, 600); ctx.fillStyle = c.fwd.hex; ctx.textAlign = bx > (s.l + s.r) / 2 ? 'right' : 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`best w ${f2(cv2.best)}`, bx + (bx > (s.l + s.r) / 2 ? -9 : 9), Math.min(s.b - 16, bv + 8));
    }

    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      let moving = false;
      if (Math.abs(amtTgt - amtCur) > 0.002) { amtCur += (amtTgt - amtCur) * Math.min(1, dt * 5); moving = true; } else amtCur = amtTgt;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !(playing || moving)) { lastUi = now; sync(); }
      return playing || moving || drag >= 0;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    // ---- dragging the handles ----
    const pos = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const handleAt = (p: { x: number; y: number }) => {
      const { main, sx, sy } = lay(), f = frame();
      let best = -1, bd = 22 * 22;
      HX.forEach((hx, i) => {
        const px = isReg ? sx(hx) : sx(Math.abs(f.w) > 0.05 ? clamp(-f.b / f.w, X[0], X[1]) : 0), py = clamp(sy(isReg ? f.w * hx + f.b : 0.5), main.y + 10, main.y + main.h - 26);
        const d = (px - p.x) ** 2 + (py - p.y) ** 2;
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    };
    const down = (e: PointerEvent) => { const h = handleAt(pos(e)); if (h >= 0) { drag = h; cv.setPointerCapture(e.pointerId); take(); wake(); e.preventDefault(); } };
    const move = (e: PointerEvent) => {
      const p = pos(e);
      if (drag < 0) { cv.style.cursor = handleAt(p) >= 0 ? 'grab' : ''; return; }
      const { sx, sy } = lay(), f = frame();
      const dx = (px: number) => X[0] + ((px - sx(X[0])) / (sx(X[1]) - sx(X[0]))) * (X[1] - X[0]);
      if (isReg) {
        const dy = Y[0] + ((p.y - sy(Y[0])) / (sy(Y[1]) - sy(Y[0]))) * (Y[1] - Y[0]);
        const y = clamp(dy, -6, 8), x1 = HX[0], x2 = HX[1], y1 = f.w * x1 + f.b, y2 = f.w * x2 + f.b;
        const ya = drag === 0 ? y : y1, yb = drag === 1 ? y : y2, w = (yb - ya) / (x2 - x1);
        over = { w: clamp(w, W_RANGE.regression[0], W_RANGE.regression[1]), b: clamp(ya - w * x1, B_RANGE.regression[0], B_RANGE.regression[1]) };
      } else {
        const cx = clamp(dx(p.x), X[0], X[1]);
        over = { w: f.w, b: clamp(-f.w * cx, B_RANGE.classification[0], B_RANGE.classification[1]) };
      }
      sync(); wake();
    };
    const up = () => { drag = -1; cv.style.cursor = ''; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 400" narrowAspect="360 / 580"
  label={isReg ? 'A scatter plot with a draggable line. Vertical segments show each point\'s error. A small chart shows the loss as the slope changes.'
    : 'Points labelled 0 or 1 along one feature with a draggable S-shaped curve. Red bars show each point\'s loss. A small chart shows the loss as the steepness changes.'}
  legend={isReg ? [{ label: 'model', color: '--viz-forward', shape: 'line' }, { label: 'above the line', color: '--viz-pos' }, { label: 'below', color: '--viz-neg' }, { label: 'you are here', color: '--accent' }]
    : [{ label: 'model', color: '--viz-forward', shape: 'line' }, { label: 'loss per point', color: '--viz-neg' }, { label: 'you are here', color: '--accent' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: the model moves automatically"
      onSeek={(to) => { over = null; amtTgt = amtCur = 0; outOn = false; t = to; wake(); }} onStep={(to) => { over = null; outOn = false; t = to >= T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if isReg}
        MSE <b class:on={kind === 'mse'}>{f2(cur.mse, 3)}</b> · MAE <b class:on={kind === 'mae'}>{f2(cur.mae, 3)}</b>
        {#if base}<span class="base">· without the outlier: MSE {f2(base.mse, 3)}, MAE {f2(base.mae, 3)}</span>{/if}
      {:else}
        log loss <b class="on">{f2(cur.log, 3)}</b> <span class="base">· guessing 0.5 everywhere would score {f2(Math.LN2, 3)}</span>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label={isReg ? 'slope w' : 'steepness w'} value={ui.w} min={W_RANGE[init.task][0]} max={W_RANGE[init.task][1]} step={0.01} oninput={setW} />
      <F1Slider label={isReg ? 'intercept b' : 'offset b'} value={ui.b} min={B_RANGE[init.task][0]} max={B_RANGE[init.task][1]} step={0.01} oninput={setB} />
      {#if isReg}
        <fieldset class="seg">
          <legend class="sr-only">Loss</legend>
          <label><input type="radio" name="loss-{uid}" value="mse" bind:group={kind} /><span>Squared (MSE)</span></label>
          <label><input type="radio" name="loss-{uid}" value="mae" bind:group={kind} /><span>Absolute (MAE)</span></label>
        </fieldset>
        <label class="chk"><input type="checkbox" checked={over ? outOn : ui.amt > 0.5} onchange={(e) => toggleOut(e.currentTarget.checked)} /> Outlier</label>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .vals b { font-weight: 500; color: var(--text); } .vals b.on { color: var(--accent); font-weight: 700; }
  .base { font-size: 0.82rem; }
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
