<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels } from '../F1Shared/plot';
  import {
    DOM, GRID, ODD_KS, RHOS, BINS_N, HIST_MAX, DIMS_MAX, makeScene, frameAt, field,
    type Mode, type Frame, type Scene,
  } from './sim';

  interface Props {
    title?: string;
    /** vote: a query drifts and its neighbours vote. metric: Euclidean to Manhattan. k: sweep k. scale: stretch one axis. dims: distance concentration. */
    mode?: Mode;
    /** Neighbours in the vote mode (the other modes fix their own). */
    k?: number;
  }
  let { title = 'k-nearest neighbours: a point is classified by its neighbours', mode = 'vote', k = 5 }: Props = $props();
  const init = untrack(() => ({ mode, k }));

  let kSel = $state(init.k), moved = $state(false);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  let scene: Scene = makeScene(init.mode, init.k);
  const T = scene.T, markers = scene.markers;
  let qo: { x: number; y: number } | null = null;
  let ui = $state<Frame>(frameAt(0, scene));

  const cls = (n: number) => (n === 1 ? 'blue' : 'orange');
  const winner = (f: Frame) => (f.pred === null ? (f.votes[0] + f.votes[1] ? 'a tie' : 'nothing yet') : cls(f.pred));
  const narration = $derived.by(() => {
    const f = ui;
    if (f.mode === 'vote') {
      const [a, b] = f.votes;
      if (f.k === 1) return `The single nearest point is ${cls(a ? 0 : 1)}, and it decides alone.`;
      return a === 0 || b === 0 ? `All ${f.k} nearest points are ${cls(a ? 0 : 1)}, so the vote is unanimous.` : `The ${f.k} nearest points disagree, ${b} blue to ${a} orange. The majority wins.`;
    }
    if (f.mode === 'metric') {
      return ({
        Grow: 'A ball around the query grows until it has swallowed its 5 nearest points. Each one lights up as the edge reaches it.',
        Circle: 'Euclidean distance: the ball is a circle.',
        Morph: 'As p falls from 2 to 1 the circle squares up into a diamond, and the set of nearest points can change.',
        Diamond: f.p < 1.005 ? 'Manhattan distance: the ball is a diamond, and its 5 nearest points can differ from the circle\'s.' : 'Back to a circle, then the ball shrinks and starts again.',
      } as Record<string, string>)[f.phase] ?? '';
    }
    if (f.mode === 'k') {
      return f.k === 1 ? 'k = 1: every point is its own nearest neighbour, so training accuracy is 100% and the border wraps around each noisy point.'
        : f.k <= 5 ? `k = ${f.k}: a few neighbours smooth out the noise and the border follows the shape of the two classes.`
        : f.k <= 15 ? `k = ${f.k}: the border is smoother, and it starts to ignore real detail where the classes overlap.`
        : `k = ${f.k}: each vote covers a large part of the plane, so the border becomes a smooth sweep and both accuracies fall.`;
    }
    if (f.mode === 'scale') {
      return ({
        Equal: 'Both axes count equally. The class depends only on x, so the regions are a clean split left and right.',
        Stretch: 'As y is stretched it counts for more and more in every distance. The ball squeezes into a thin band, the neighbours are whichever points share the query\'s y, and the regions break into horizontal stripes.',
        Standardise: 'Dividing each feature by its own spread brings the axes back onto the same footing: the regions straighten out and accuracy recovers.',
      } as Record<string, string>)[f.phase] ?? '';
    }
    return f.d < 4 ? 'With few dimensions the nearest point is far closer than the farthest, so "nearest" is meaningful.'
      : f.d < 30 ? 'As dimensions are added the spread of distances narrows, and the nearest point is no longer much closer than the rest.'
      : 'In many dimensions almost every point is about the same distance away, so "nearest" barely means anything.';
  });
  const pName = (p: number) => (p > 1.995 ? 'Euclidean' : p < 1.005 ? 'Manhattan' : 'Minkowski');
  const tally = (f: Frame) => `blue ${f.votes[1]}, orange ${f.votes[0]}`;

  $effect(() => { playing; speed; wake(); });

  function setK(v: number) { kSel = v; scene = makeScene(init.mode, v); wake(); }
  function place(x: number, y: number) { qo = { x, y }; moved = true; wake(); }
  function resetQuery() { qo = null; moved = false; t = 0; wake(); }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const N = GRID, grid = new Float32Array(N * N), off = document.createElement('canvas');
    off.width = N; off.height = N;
    const octx = off.getContext('2d')!, img = octx.createImageData(N, N), px = new Uint32Array(img.data.buffer);
    let lastUi = 0;
    const plane = init.mode !== 'dims';
    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, 0.6);
      const s = Math.max(40, Math.min(main.w - 40, main.h - 30));
      const px0 = main.x + 28 + (main.w - 28 - s) / 2, py0 = main.y + 6;
      const sd = { l: side.x + 44, r: side.x + side.w - 14, t: side.y + 34, b: side.y + side.h - 34 };
      return { main, side, stacked, px0, py0, s, sd, gx: scale(-DOM, DOM, px0, px0 + s), gy: scale(-DOM, DOM, py0 + s, py0) };
    };

    const fieldFor = (f: Frame): Float32Array | null => {
      if (scene.mode === 'vote') return scene.field!;
      if (scene.mode === 'scale') { field(scene.train, N, scene.k, 2, f.rho, grid); return grid; }
      if (scene.mode === 'k') {
        const fs = scene.fields!, a = Math.floor(f.ki) % fs.length, b = (a + 1) % fs.length, u = f.ki - Math.floor(f.ki);
        for (let i = 0; i < grid.length; i++) grid[i] = fs[a][i] * (1 - u) + fs[b][i] * u;
        return grid;
      }
      return null;
    };

    // a chart frame in the side panel: box, ticks, title, x label
    const chart = (L: ReturnType<typeof lay>, name: string, xl: string, xt: [number, string][], yt: [number, string][]) => {
      const { ctx, c } = stage, sd = L.sd;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(name, sd.l - 34, L.side.y + 4);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const [y, s] of yt) ctx.fillText(s, sd.l - 6, y);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const [x, s] of xt) ctx.fillText(s, x, sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(xl, sd.r, L.side.y + L.side.h - 1);
    };
    const line = (xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, w: number, dash: number[] = []) => {
      const { ctx } = stage;
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.setLineDash(dash); ctx.beginPath();
      for (let i = 0; i < xs.length; i++) { if (i === 0) ctx.moveTo(xs[i], ys[i]); else ctx.lineTo(xs[i], ys[i]); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    const dot = (x: number, y: number, color: string, r = 5.5) => {
      const { ctx, c } = stage;
      ctx.fillStyle = color; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };

    function renderPlane(f: Frame, L: ReturnType<typeof lay>) {
      const { ctx, c } = stage;
      const g = fieldFor(f);
      if (g) {
        const [pr, pg, pb] = c.pos.rgb, [nr, ng, nb] = c.neg.rgb, [br, bg, bb] = c.raised.rgb;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const p = g[j * N + i], a = Math.min(0.62, Math.abs(2 * p - 1) * 0.7), [r, gg, b] = p > 0.5 ? [pr, pg, pb] : [nr, ng, nb];
          px[(N - 1 - j) * N + i] = 0xff000000 | (Math.round(bb + (b - bb) * a) << 16) | (Math.round(bg + (gg - bg) * a) << 8) | Math.round(br + (r - br) * a);
        }
        octx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, L.px0, L.py0, L.s, L.s);
      }
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(L.px0, L.py0, L.s, L.s);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-1, 0, 1]) ctx.fillText(String(v).replace('-', '−'), L.gx(v), L.py0 + L.s + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [-1, 0, 1]) ctx.fillText(String(v).replace('-', '−'), L.px0 - 5, L.gy(v));

      ctx.save(); ctx.beginPath(); ctx.rect(L.px0, L.py0, L.s, L.s); ctx.clip();
      const q = f.q, lit = new Map(f.nn.map((e) => [e.i, e]));
      if (q) {
        for (const e of f.nn) {
          if (e.a < 0.02) continue;
          const P = scene.train[e.i];
          ctx.strokeStyle = c.accent.hex; ctx.globalAlpha = e.a; ctx.lineWidth = 1 + 1.6 * e.a;
          ctx.beginPath(); ctx.moveTo(L.gx(q.x), L.gy(q.y)); ctx.lineTo(L.gx(P.x), L.gy(P.y)); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        // the ball: |dx|^p + |rho dy|^p = r^p
        if (f.r > 0) {
          const ex = 2 / f.p;
          ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); ctx.beginPath();
          for (let i = 0; i <= 120; i++) {
            const th = (i / 120) * 6.2832, co = Math.cos(th), si = Math.sin(th);
            const X = L.gx(q.x + f.r * Math.sign(co) * Math.abs(co) ** ex), Y = L.gy(q.y + (f.r * Math.sign(si) * Math.abs(si) ** ex) / f.rho);
            if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.stroke(); ctx.setLineDash([]);
        }
      }
      for (let P = 0; P < scene.train.length; P++) {
        const p = scene.train[P], x = L.gx(p.x), y = L.gy(p.y), e = lit.get(P), a = e ? e.a : 0;
        ctx.fillStyle = p.c ? c.pos.hex : c.neg.hex; ctx.strokeStyle = a > 0.5 ? c.accent.hex : c.raised.hex; ctx.lineWidth = a > 0.5 ? 2.4 : 1.5;
        ctx.beginPath();
        if (p.c) ctx.arc(x, y, 4.6, 0, 6.2832); else { ctx.moveTo(x, y - 5.6); ctx.lineTo(x + 5.6, y); ctx.lineTo(x, y + 5.6); ctx.lineTo(x - 5.6, y); ctx.closePath(); }
        ctx.fill(); ctx.stroke();
      }
      if (q) { dot(L.gx(q.x), L.gy(q.y), c.accent.hex, 6.5); ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(L.gx(q.x), L.gy(q.y), 6.5, 0, 6.2832); ctx.stroke(); }
      ctx.restore();
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('x', L.px0 + L.s, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y', L.px0 - 24, L.py0 - 4);

      // side panel
      const sd = L.sd;
      if (scene.mode === 'vote' || scene.mode === 'metric') {
        const m = f.nn.length, rowH = Math.min(28, (sd.b - sd.t + 20) / m), AX = 0.6, bx = scale(0, AX, sd.l + 22, sd.r - 40);
        ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('Nearest points, closest first', sd.l - 34, L.side.y + 4);
        for (let j = 0; j < m; j++) {
          const e = f.nn[j], P = scene.train[e.i], y = sd.t - 6 + j * rowH + rowH / 2, w = Math.min(AX, e.d);
          ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(j + 1), sd.l + 14, y);
          ctx.globalAlpha = 0.25 + 0.75 * e.a; ctx.fillStyle = P.c ? c.pos.hex : c.neg.hex;
          ctx.fillRect(bx(0), y - rowH * 0.32, bx(w) - bx(0), rowH * 0.64);
          ctx.globalAlpha = 1;
          if (e.inside) { ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.strokeRect(bx(0), y - rowH * 0.32, bx(w) - bx(0), rowH * 0.64); }
          ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.fillText(f2(e.d), sd.r, y);
        }
        if (f.r > 0 && f.r < AX) {
          ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(bx(f.r), sd.t - 6); ctx.lineTo(bx(f.r), sd.t - 6 + m * rowH); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText(`${f.p === 2 ? 'Euclidean' : f.p === 1 ? 'Manhattan' : 'Minkowski'} distance to the query`, sd.r, L.side.y + L.side.h - 1);
      } else if (scene.mode === 'k') {
        const lk = (v: number) => Math.log(v), X = scale(lk(1), lk(75), sd.l, sd.r), Y = scale(0.5, 1, sd.b, sd.t);
        chart(L, 'Accuracy as k grows', 'k (log scale)', [1, 5, 15, 75].map((v) => [X(lk(v)), String(v)]), [0.5, 0.75, 1].map((v) => [Y(v), `${Math.round(v * 100)}%`]));
        const xs = ODD_KS.map((v) => X(lk(v)));
        line(xs, scene.accK!.train.map(Y), c.muted.hex, 1.8, [5, 4]);
        line(xs, scene.accK!.val.map(Y), c.accent.hex, 2.2);
        const x = X(lk(f.k));
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, sd.t); ctx.lineTo(x, sd.b); ctx.stroke();
        dot(x, Y(f.trainAcc), c.muted.hex, 5); dot(x, Y(f.valAcc), c.accent.hex, 5.5);
      } else {
        const lr = Math.log(100), X = scale(0, lr, sd.l, sd.r), Y = scale(0.4, 1, sd.b, sd.t);
        chart(L, 'Validation accuracy as y is stretched', 'y stretch (log scale)', [1, 10, 100].map((v) => [X(Math.log(v)), `×${v}`]), [0.5, 0.75, 1].map((v) => [Y(v), `${Math.round(v * 100)}%`]));
        ctx.strokeStyle = c.rule.hex; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(sd.l, Y(0.5)); ctx.lineTo(sd.r, Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
        line(RHOS.map((r) => X(Math.log(r))), scene.accRho!.map(Y), c.accent.hex, 2.2);
        dot(X(Math.log(Math.max(1, f.rho))), Y(f.valAcc), c.accent.hex, 5.5);
      }
    }

    function renderDims(f: Frame, L: ReturnType<typeof lay>) {
      const { ctx, c } = stage, m = L.main, h = f.hist!;
      const bx = { l: m.x + 40, r: m.x + m.w - 14, t: m.y + 48, b: m.y + m.h - 40 };
      const X = scale(0, HIST_MAX, bx.l, bx.r), YM = 0.6, Y = scale(0, YM, bx.b, bx.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`Distances from one query to ${200} random points, d = ${Math.round(f.d)}`, bx.l - 34, m.y + 4);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(bx.l, bx.t, bx.r - bx.l, bx.b - bx.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 1, 2, 3]) ctx.fillText(String(v), X(v), bx.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.2, 0.4, 0.6]) ctx.fillText(`${Math.round(v * 100)}%`, bx.l - 6, Y(v));
      ctx.fillStyle = c.s3.hex;
      const bw = (bx.r - bx.l) / BINS_N;
      for (let b = 0; b < BINS_N; b++) { const y = Y(Math.min(YM, h[b])); ctx.fillRect(bx.l + b * bw + 0.5, y, bw - 1, bx.b - y); }
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('distance, in units of the average distance', bx.r, m.y + m.h - 1);
      // nearest / farthest ticks
      for (const [v, name, dir] of [[f.near, 'nearest', -1], [f.far, 'farthest', 1]] as const) {
        const al = dir * (X(v) - (dir < 0 ? bx.l : bx.r)) > -62 ? -dir : dir; // flip the label inward near an edge
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(v), bx.t - 3); ctx.lineTo(X(v), bx.b); ctx.stroke();
        ctx.font = stage.body(12); ctx.fillStyle = c.text.hex; ctx.textAlign = al > 0 ? 'left' : 'right'; ctx.textBaseline = 'top';
        ctx.fillText(name, X(v) + 4 * al, bx.t - 19);
      }
      // side: nearest / farthest ratio against d
      const sd = L.sd, lr = Math.log(DIMS_MAX), XS = scale(0, lr, sd.l, sd.r), YS = scale(0, 1, sd.b, sd.t);
      chart(L, 'Nearest ÷ farthest distance', 'dimensions d (log scale)', [1, 10, 100].map((v) => [XS(Math.log(v)), String(v)]), [0, 0.5, 1].map((v) => [YS(v), f2(v, 1)]));
      const r = ratioCurve;
      line(r.xs.map((d) => XS(Math.log(d))), r.ys.map(YS), c.accent.hex, 2.2);
      dot(XS(Math.log(Math.max(1, f.d))), YS(f.ratio), c.accent.hex, 5.5);
    }
    // the ratio curve (constant per scene); drawn from the same table the frames interpolate
    const ratioCurve = { xs: [] as number[], ys: [] as number[] };
    if (scene.dims) for (let d = 1; d <= DIMS_MAX; d++) { ratioCurve.xs.push(d); ratioCurve.ys.push(scene.dims.ratio[d - 1]); }

    function render() {
      const { ctx, w: W, h: H } = stage, L = lay(), f = frameAt(t, scene, { q: qo ?? undefined });
      ctx.clearRect(0, 0, W, H);
      if (plane) renderPlane(f, L); else renderDims(f, L);
    }

    const read = () => { ui = frameAt(t, scene, { q: qo ?? undefined }); };
    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const toData = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top;
      const dx = -DOM + ((x - L.px0) / L.s) * 2 * DOM, dy = DOM - ((y - L.py0) / L.s) * 2 * DOM;
      return Math.abs(dx) <= DOM && Math.abs(dy) <= DOM ? { x: dx, y: dy } : null;
    };
    const pick = (e: PointerEvent) => { if (!plane || init.mode === 'k') return; const p = toData(e); if (p) place(p.x, p.y); };
    const down = (e: PointerEvent) => pick(e);
    const move = (e: PointerEvent) => { if (e.buttons & 1 && e.pointerType === 'mouse') pick(e); };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move);
    if (plane && init.mode !== 'k') cv.style.cursor = 'crosshair';
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); };
  });

  const labels: Record<Mode, string> = {
    vote: 'Left: a plane of blue circles and orange diamonds, shaded by what a k-nearest-neighbour vote would say at each spot. A query point drifts across it and lines join it to its nearest neighbours. Right: the nearest points ranked by distance.',
    metric: 'Left: the same points with a query and a ball around it that grows to catch its 5 nearest neighbours, then morphs from a circle to a diamond. Right: the nearest points ranked by distance.',
    k: 'Left: decision regions for increasing k, from ragged at k = 1 to broad blocks at large k. Right: training and validation accuracy against k.',
    scale: 'Left: points whose class depends only on x, with decision regions and a neighbourhood ball that squeezes into a thin band as the y axis is stretched, then recovers after standardising. Right: validation accuracy against the stretch.',
    dims: 'Left: a histogram of distances from one query to 200 random points, narrowing into a spike as the number of dimensions grows. Right: the nearest-to-farthest distance ratio rising towards 1.',
  };
  const legend = {
    vote: [{ label: 'blue circle', color: '--viz-pos' }, { label: 'orange diamond', color: '--viz-neg' }, { label: 'query and its neighbours', color: '--accent' }],
    metric: [{ label: 'blue circle', color: '--viz-pos' }, { label: 'orange diamond', color: '--viz-neg' }, { label: 'query and its neighbours', color: '--accent' }],
    k: [{ label: 'blue circle', color: '--viz-pos' }, { label: 'orange diamond', color: '--viz-neg' }, { label: 'training accuracy', color: '--text-muted', shape: 'line' as const }, { label: 'validation accuracy', color: '--accent', shape: 'line' as const }],
    scale: [{ label: 'blue circle', color: '--viz-pos' }, { label: 'orange diamond', color: '--viz-neg' }, { label: 'query and its neighbours', color: '--accent' }, { label: 'validation accuracy', color: '--accent', shape: 'line' as const }],
    dims: [{ label: 'share of points', color: '--viz-seq-3' }, { label: 'nearest, farthest', color: '--accent', shape: 'line' as const }],
  }[init.mode];
</script>

<FigureCard {title} role="group" aspect="720 / 400" narrowAspect="360 / 640" label={labels[init.mode]} {legend}>
  <canvas bind:this={cv} style="touch-action: pan-y" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Timeline: k-nearest neighbours"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={resetQuery} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if ui.mode === 'vote'}
        query ({f2(ui.q!.x)}, {f2(ui.q!.y)}) · k <b>{ui.k}</b> · votes <b>{tally(ui)}</b> · predicts <b>{winner(ui)}</b>
      {:else if ui.mode === 'metric'}
        <b>{pName(ui.p)}</b>, p = <b>{f2(ui.p)}</b> · ball radius <b>{f2(ui.r)}</b> · inside: <b>{tally(ui)}</b> · predicts <b>{winner(ui)}</b>
      {:else if ui.mode === 'k'}
        k <b>{ui.k}</b> · training accuracy <b>{Math.round(ui.trainAcc * 100)}%</b> · validation accuracy <b>{Math.round(ui.valAcc * 100)}%</b>
      {:else if ui.mode === 'scale'}
        y stretch ×<b>{f2(ui.rho, ui.rho < 10 ? 2 : 0)}</b> · validation accuracy <b>{Math.round(ui.valAcc * 100)}%</b> · query's 5 nearest: <b>{tally(ui)}</b>
      {:else}
        d <b>{Math.round(ui.d)}</b> · nearest <b>{f2(ui.near)}</b> · farthest <b>{f2(ui.far)}</b> (average distance = 1) · nearest ÷ farthest <b>{f2(ui.ratio)}</b>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if init.mode === 'vote' || init.mode === 'metric' || init.mode === 'scale'}
      <div class="ctl">
        {#if init.mode === 'vote'}
          <label>Neighbours k
            <select value={kSel} onchange={(e) => setK(+e.currentTarget.value)}>
              {#each [1, 3, 5, 9] as v (v)}<option value={v}>{v}</option>{/each}
            </select>
          </label>
        {/if}
        <button type="button" class="btn" onclick={resetQuery} disabled={!moved}>{init.mode === 'vote' ? 'Follow the path again' : 'Reset the query'}</button>
        <span class="hint">Click or drag on the plane to put the query anywhere{init.mode === 'vote' ? '.' : ' (the animation restarts from there).'}</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.9rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: 0.5em; font: var(--step--1) var(--font-ui); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
