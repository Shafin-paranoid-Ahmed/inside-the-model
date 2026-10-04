<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels } from '../F1Shared/plot';
  import {
    DOM, K_MAX, T0, DA, T_STRUCT, makeData, withPoint, initCentroids, kmeans, elbow, duration, markers, frameAt, structureFrame,
    sizes, nearest, finalInertia, defaultK, type Mode, type DataName, type KRun, type Frame,
  } from './sim';

  interface Props {
    title?: string;
    /** 'structure': grey points, then the groups appear. 'assign': one assign+update with draggable centroids. 'update': the full loop with trails and the inertia curve. 'init': random vs k-means++ starts. 'elbow': inertia against k. */
    mode?: Mode;
    /** Shade the plane by nearest centroid. */
    regions?: boolean;
    /** Draw each centroid's path. */
    trail?: boolean;
  }
  let { title = '', mode = 'update', regions, trail }: Props = $props();
  const init = untrack(() => {
    const d: DataName = mode === 'init' || mode === 'elbow' ? 'four' : 'three';
    const titles: Record<Mode, string> = {
      structure: 'Unlabelled points: is there structure?', assign: 'The assign step: every point joins its nearest centroid',
      update: 'k-means: assign, update, repeat until nothing changes', init: 'Same data, two different starting points', elbow: 'Choosing k: inertia against the number of clusters',
    };
    return { M: mode, seed: mode === 'init' ? 13 : 10, d, k: defaultK(d), regions: regions ?? (mode === 'assign' || mode === 'update'), trail: trail ?? mode === 'update', title: title || titles[mode] };
  });
  const M = init.M;
  /** Cluster colours, in order (semantic tokens only; the shape also changes every four clusters). */
  const PAL = ['pos', 'neg', 'fwd', 'accent', 'grad', 'text', 's3', 's5'] as const;

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let dur = $state(1), mk = $state<{ at: number; label: string }[]>([]);
  let seed = $state(init.seed), kSel = $state(4);
  let ui = $state({ phase: 'start', iter: 0, inertia: NaN, moved: 0, inA: NaN, inB: NaN, nA: 0, nB: 0, hov: null as null | { i: number; d: number[]; best: number } });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const d0 = makeData(init.d);
  let data = $state.raw(d0), run = $state.raw<KRun>(undefined as unknown as KRun), runB = $state.raw<KRun | null>(null), extra = $state(0);
  const el = M === 'elbow' ? elbow(d0.pts) : null;
  let C0 = initCentroids(d0.pts, init.k, M === 'structure' ? 'plusplus' : 'random', M === 'structure' ? 1 : init.seed);
  let hoverI = -1, dragC = -1;

  function build() {
    if (M === 'init') {
      run = kmeans(data.pts, initCentroids(data.pts, 4, 'random', seed));
      runB = kmeans(data.pts, initCentroids(data.pts, 4, 'plusplus', seed));
      const long = run.n >= runB.n ? run : runB;
      dur = Math.max(duration(run), duration(runB)); mk = markers(long, M);
    } else if (M === 'elbow') {
      run = el!.runs[kSel - 1]; dur = duration(run); mk = markers(run, M);
    } else {
      run = kmeans(data.pts, C0, M === 'assign' ? 1 : 40);
      dur = M === 'structure' ? T_STRUCT : duration(run); mk = markers(run, M);
    }
  }
  build();
  const restart = () => { t = 0; playing = true; wake(); };
  function reset() { data = makeData(init.d); extra = 0; seed = M === 'init' ? 13 : 10; C0 = initCentroids(data.pts, init.k, M === 'structure' ? 'plusplus' : 'random', M === 'structure' ? 1 : seed); build(); restart(); }
  function reseed() { seed++; if (M === 'init') build(); else { C0 = initCentroids(data.pts, init.k, 'random', seed); build(); } restart(); }
  function pickK(k: number) { kSel = k; build(); restart(); }

  const fmt = (v: number) => (Number.isFinite(v) ? f2(v, 1) : '—');
  const gap = $derived(runB ? finalInertia(run) / finalInertia(runB) : 1);
  const verdict = $derived(
    M !== 'init' || !runB ? '' : Math.abs(finalInertia(run) - finalInertia(runB)) < 1e-6 ? 'Both starts reached the same clustering this time. Reseed to try other starts.'
      : finalInertia(run) > finalInertia(runB) ? `The random start settled in a worse local minimum: its inertia is ${f2(gap, 1)} times higher.`
      : `This time k-means++ was the unlucky one: its inertia is ${f2(1 / gap, 1)} times higher.`,
  );
  const narration = $derived.by(() => {
    const p = ui.phase;
    if (M === 'structure') return p === 'unlabelled' ? 'Every point is grey. Nothing in the data says which point belongs to which group.' : 'k-means was told only that there are 3 groups. The colours are what it found; the heavy circles are the group centres.';
    if (M === 'init') return ui.phase === 'converged' ? verdict : 'Both panels see the same points. Only the first centroids differ; each side then runs its own assign and update passes.';
    if (p === 'start') return `Start: ${run.k} centroids (heavy circles) placed ${M === 'elbow' ? 'by k-means++' : 'on random data points'}. The points are still grey.`;
    if (p === 'assign') return 'Assign: each point takes the colour of its nearest centroid. ' + (ui.iter === 0 ? ' Inertia (the total squared distance from every point to its centroid) is measured once every point has a group.' : ' Inertia is measured again on the new groups.');
    if (p === 'update') return 'Update: each centroid glides to the mean of the points that chose it. The inertia falls again.';
    return run.n === 1 ? 'One assign and one update done. Drag a centroid, or press New random start, to see how the regions change.' : `Converged after ${run.n} passes: the last assign step moved no point, so the update cannot move any centroid.`;
  });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const RN = 110;
    const mkRegion = () => { const o = document.createElement('canvas'); o.width = RN; o.height = RN; const x = o.getContext('2d')!, img = x.createImageData(RN, RN); return { o, x, img, px: new Uint32Array(img.data.buffer) }; };
    const regs = [mkRegion(), mkRegion()];
    let lastUi = 0;
    const frameOf = (r: KRun) => (M === 'structure' ? structureFrame(t, r) : frameAt(t, r));
    const sqBox = (b: { x: number; y: number; w: number; h: number }, head: number) => {
      const s = Math.max(40, Math.min(b.w - 8, b.h - head - 6));
      return { x: b.x + (b.w - s) / 2, y: b.y + head + (b.h - head - s) / 2, s };
    };
    const lay = () => {
      const W = stage.w, H = stage.h;
      if (M === 'init') {
        const { main, side } = splitPanels(W, H, 0.5);
        return { a: sqBox(main, 24), b: sqBox(side, 24), pa: main, pb: side, side, chart: null as null | { l: number; r: number; t: number; b: number } };
      }
      const { main, side } = splitPanels(W, H, 0.58);
      const a = sqBox(main, 0);
      return { a, b: a, pa: main, pb: main, side, chart: { l: side.x + 46, r: side.x + side.w - 14, t: side.y + 34, b: side.y + side.h - 36 } };
    };

    function scatter(b: { x: number; y: number; s: number }, r: KRun, f: Frame, ri: number, showTrail: boolean, hi: number) {
      const { ctx, c } = stage, pal = PAL.map((k) => c[k]);
      const gx = scale(-DOM, DOM, b.x, b.x + b.s), gy = scale(-DOM, DOM, b.y + b.s, b.y), k = r.k;
      if (init.regions && f.centA > 0.02) {
        const R = regs[ri], [br, bg, bb] = c.raised.rgb, al = 0.2 * f.centA;
        for (let j = 0; j < RN; j++) {
          const y = -DOM + ((j + 0.5) / RN) * 2 * DOM;
          for (let i = 0; i < RN; i++) {
            const [cr, cg, cb] = pal[nearest(f.cent, k, -DOM + ((i + 0.5) / RN) * 2 * DOM, y)].rgb;
            R.px[(RN - 1 - j) * RN + i] = 0xff000000 | (Math.round(bb + (cb - bb) * al) << 16) | (Math.round(bg + (cg - bg) * al) << 8) | Math.round(br + (cr - br) * al);
          }
        }
        R.x.putImageData(R.img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(R.o, b.x, b.y, b.s, b.s);
      }
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(b.x, b.y, b.s, b.s);
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.s, b.s); ctx.clip();
      if (showTrail && f.centA > 0.5) {
        ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
        for (let q = 0; q < k; q++) {
          ctx.strokeStyle = pal[q].hex; ctx.globalAlpha = 0.6; ctx.beginPath();
          for (let m = 0; m <= f.iter; m++) { const X = gx(r.C[m][2 * q]), Y = gy(r.C[m][2 * q + 1]); if (m) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
          ctx.lineTo(gx(f.cent[2 * q]), gy(f.cent[2 * q + 1])); ctx.stroke();
          ctx.fillStyle = pal[q].hex; ctx.globalAlpha = 0.9;
          for (let m = 0; m <= f.iter; m++) { ctx.beginPath(); ctx.arc(gx(r.C[m][2 * q]), gy(r.C[m][2 * q + 1]), 2.4, 0, 6.2832); ctx.fill(); }
        }
        ctx.globalAlpha = 1;
      }
      const grey = c.muted.rgb.map((v: number, q: number) => Math.round((v + c.raised.rgb[q]) / 2));
      for (let i = 0; i < r.N; i++) {
        const x = gx(r.pts[2 * i]), y = gy(r.pts[2 * i + 1]), m = f.mix[i], fr = f.from[i] < 0 ? grey : pal[f.from[i]].rgb, to = pal[f.to[i]].rgb;
        ctx.fillStyle = `rgb(${Math.round(fr[0] + (to[0] - fr[0]) * m)},${Math.round(fr[1] + (to[1] - fr[1]) * m)},${Math.round(fr[2] + (to[2] - fr[2]) * m)})`;
        ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.2;
        const id = m >= 0.5 ? f.to[i] : f.from[i], rad = 4.4;
        ctx.beginPath();
        if (id < 0 || id % 4 === 0) ctx.arc(x, y, rad, 0, 6.2832);
        else if (id % 4 === 1) { ctx.moveTo(x, y - rad * 1.3); ctx.lineTo(x + rad * 1.3, y); ctx.lineTo(x, y + rad * 1.3); ctx.lineTo(x - rad * 1.3, y); ctx.closePath(); }
        else if (id % 4 === 2) ctx.rect(x - rad * 0.9, y - rad * 0.9, rad * 1.8, rad * 1.8);
        else { ctx.moveTo(x, y - rad * 1.25); ctx.lineTo(x + rad * 1.2, y + rad * 0.9); ctx.lineTo(x - rad * 1.2, y + rad * 0.9); ctx.closePath(); }
        ctx.fill(); ctx.stroke();
      }
      if (hi >= 0 && f.centA > 0.5) {
        const x = gx(r.pts[2 * hi]), y = gy(r.pts[2 * hi + 1]), nb = nearest(f.cent, k, r.pts[2 * hi], r.pts[2 * hi + 1]);
        for (let q = 0; q < k; q++) {
          ctx.strokeStyle = q === nb ? pal[q].hex : c.muted.hex; ctx.lineWidth = q === nb ? 2 : 1; ctx.setLineDash(q === nb ? [] : [3, 4]);
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gx(f.cent[2 * q]), gy(f.cent[2 * q + 1])); ctx.stroke();
        }
        ctx.setLineDash([]); ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.2832); ctx.stroke();
      }
      if (f.centA > 0.01) {
        ctx.globalAlpha = f.centA;
        for (let q = 0; q < k; q++) {
          const x = gx(f.cent[2 * q]), y = gy(f.cent[2 * q + 1]);
          ctx.fillStyle = pal[q].hex; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.4;
          ctx.beginPath(); ctx.arc(x, y, 8.5, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 6.2832); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();
    }

    const axes = (ch: { l: number; r: number; t: number; b: number }, xs: (v: number) => number, ys: (v: number) => number, xt: number[], yt: number[], xl: string, yl: string) => {
      const { ctx, c } = stage;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(ch.l, ch.t, ch.r - ch.l, ch.b - ch.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of xt) ctx.fillText(String(v), xs(v), ch.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of yt) ctx.fillText(String(v), ch.l - 6, ys(v));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(xl, ch.r, ch.b + 32);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = c.text.hex; ctx.font = stage.body(12.5); ctx.fillText(yl, ch.l - 40, ch.t - 26);
    };
    const niceTicks = (mx: number) => { const step = mx > 300 ? 100 : mx > 120 ? 50 : mx > 60 ? 20 : 10; const o: number[] = []; for (let v = 0; v <= mx; v += step) o.push(v); return o; };

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay();
      ctx.clearRect(0, 0, W, H);
      const f = frameOf(run);
      ctx.textBaseline = 'top';
      if (M === 'init') {
        const fb = frameOf(runB!);
        for (const [ri, box, pan, r, fr, name] of [[0, L.a, L.pa, run, f, 'Random start'], [1, L.b, L.pb, runB!, fb, 'k-means++ start']] as const) {
          scatter(box, r, fr, ri, false, -1);
          ctx.font = stage.body(13); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(name, box.x, pan.y + 2);
          ctx.font = stage.mono(12, 600); ctx.textAlign = 'right'; ctx.fillStyle = fr.phase === 'converged' ? c.text.hex : c.muted.hex;
          ctx.fillText(`inertia ${fmt(fr.inertia)}`, box.x + box.s, pan.y + 2);
        }
        return;
      }
      scatter(L.a, run, f, 0, init.trail, hoverI);
      const ch = L.chart!;
      if (M === 'update') {
        const n = run.n, top = run.iA[0] * 1.05, xs = scale(0, Math.max(n, 4), ch.l, ch.r), ys = scale(0, top, ch.b, ch.t);
        axes(ch, xs, ys, Array.from({ length: Math.max(n, 4) + 1 }, (_, i) => i), niceTicks(top), 'passes completed', 'Inertia (total squared distance)');
        if (Number.isFinite(f.inertia)) {
          const conf = (m: number) => (m === 0 ? run.iA[0] : run.iU[m - 1]);
          ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
          for (let m = 0; m <= f.iter; m++) { if (m) ctx.lineTo(xs(m), ys(conf(m))); else ctx.moveTo(xs(0), ys(conf(0))); }
          ctx.lineTo(xs(f.cx), ys(f.inertia)); ctx.stroke();
          ctx.fillStyle = c.grad.hex;
          for (let m = 0; m <= f.iter; m++) { ctx.beginPath(); ctx.arc(xs(m), ys(conf(m)), 3, 0, 6.2832); ctx.fill(); }
          ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(xs(f.cx), ys(f.inertia), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        }
      } else if (M === 'elbow') {
        const v = el!.inertia, top = v[0] * 1.05, xs = scale(1, K_MAX, ch.l + 12, ch.r - 12), ys = scale(0, top, ch.b, ch.t);
        axes(ch, xs, ys, Array.from({ length: K_MAX }, (_, i) => i + 1), niceTicks(top), 'number of clusters k', 'Final inertia');
        ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
        v.forEach((y, i) => (i ? ctx.lineTo(xs(i + 1), ys(y)) : ctx.moveTo(xs(1), ys(y)))); ctx.stroke();
        v.forEach((y, i) => {
          const sel = i + 1 === kSel;
          ctx.fillStyle = sel ? c.accent.hex : c.grad.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(xs(i + 1), ys(y), sel ? 6.5 : 4, 0, 6.2832); ctx.fill(); ctx.stroke();
        });
        ctx.font = stage.mono(12, 600); ctx.fillStyle = c.text.hex; ctx.textAlign = kSel < 5 ? 'left' : 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText(f2(v[kSel - 1], 1), xs(kSel) + (kSel < 5 ? 11 : -11), ys(v[kSel - 1]) - 8);
      } else {
        // cluster sizes, smooth through the colour fade
        const z = sizes(f, run.k), bars = run.k + 1, bw = (ch.r - ch.l) / bars, ys = scale(0, run.N, ch.b, ch.t);
        ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('Points per cluster', ch.l - 40, ch.t - 26);
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ch.l, ch.b); ctx.lineTo(ch.r, ch.b); ctx.stroke();
        for (let q = 0; q < bars; q++) {
          const v = z[q], x = ch.l + q * bw + bw * 0.18, y = ys(v);
          ctx.fillStyle = q === run.k ? c.muted.hex : PAL.map((p) => c[p])[q].hex; ctx.globalAlpha = q === run.k ? 0.5 : 1;
          ctx.fillRect(x, y, bw * 0.64, ch.b - y); ctx.globalAlpha = 1;
          ctx.font = stage.mono(12, 600); ctx.fillStyle = c.text.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(Math.round(v)), x + bw * 0.32, y - 3);
          ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.fillText(q === run.k ? 'unassigned' : `cluster ${q + 1}`, x + bw * 0.32, ch.b + 6);
        }
      }
    }

    const read = () => {
      const f = frameOf(run);
      let hv = null as typeof ui.hov;
      if (hoverI >= 0 && f.centA > 0.5) {
        const x = run.pts[2 * hoverI], y = run.pts[2 * hoverI + 1], d: number[] = [];
        for (let q = 0; q < run.k; q++) d.push((x - f.cent[2 * q]) ** 2 + (y - f.cent[2 * q + 1]) ** 2);
        hv = { i: hoverI, d, best: nearest(f.cent, run.k, x, y) };
      }
      if (M === 'init') { const fb = frameOf(runB!); ui = { ...ui, phase: f.phase === 'converged' && fb.phase === 'converged' ? 'converged' : 'run', inA: f.inertia, inB: fb.inertia, nA: run.n, nB: runB!.n, hov: null }; }
      else ui = { ...ui, phase: f.phase, iter: f.iter, inertia: f.inertia, moved: f.moved, hov: hv };
    };
    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: dur }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const pos = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(), b = L.a; return { px: e.clientX - r.left, py: e.clientY - r.top, b, x: -DOM + ((e.clientX - r.left - b.x) / b.s) * 2 * DOM, y: DOM - ((e.clientY - r.top - b.y) / b.s) * 2 * DOM }; };
    const near = (p: ReturnType<typeof pos>, xs: ArrayLike<number>, n: number, rad: number) => {
      let bi = -1, bd = rad * rad; const k = p.b.s / (2 * DOM);
      for (let i = 0; i < n; i++) { const d = ((xs[2 * i] - p.x) * k) ** 2 + ((xs[2 * i + 1] - p.y) * k) ** 2; if (d < bd) { bd = d; bi = i; } }
      return bi;
    };
    const move = (e: PointerEvent) => {
      if (M === 'init') return;
      const p = pos(e);
      if (dragC >= 0) {
        C0[2 * dragC] = Math.max(-DOM + 0.05, Math.min(DOM - 0.05, p.x)); C0[2 * dragC + 1] = Math.max(-DOM + 0.05, Math.min(DOM - 0.05, p.y));
        build(); t = T0 + DA; wake(); return;
      }
      const inside = Math.abs(p.x) <= DOM && Math.abs(p.y) <= DOM;
      hoverI = M === 'elbow' || !inside ? -1 : near(p, run.pts, run.N, 14);
      const onC = M === 'assign' && inside && near(p, C0, run.k, 14) >= 0;
      cv.style.cursor = onC ? 'grab' : inside && (M === 'update' || M === 'structure') && hoverI < 0 ? 'crosshair' : '';
      wake();
    };
    const leave = () => { hoverI = -1; wake(); };
    const down = (e: PointerEvent) => {
      if (M === 'init') return;
      const p = pos(e), L = lay();
      if (M === 'elbow') {
        const ch = L.chart!, r = cv.getBoundingClientRect();
        if (e.clientX - r.left > L.side.x) { const xs = scale(1, K_MAX, ch.l + 12, ch.r - 12); let bk = 1; for (let k = 2; k <= K_MAX; k++) if (Math.abs(xs(k) - p.px) < Math.abs(xs(bk) - p.px)) bk = k; pickK(bk); }
        return;
      }
      if (!(Math.abs(p.x) <= DOM && Math.abs(p.y) <= DOM)) return;
      if (M === 'assign') {
        const q = near(p, C0, run.k, 14);
        if (q >= 0) { dragC = q; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; playing = false; t = T0 + DA; wake(); }
      } else if ((M === 'update' || M === 'structure') && near(p, run.pts, run.N, 10) < 0) {
        data = withPoint(data, p.x, p.y); extra++; build(); restart();
      }
    };
    const up = () => { if (dragC >= 0) { dragC = -1; cv.style.cursor = ''; t = T0; playing = true; wake(); } };
    cv.addEventListener('pointermove', move); cv.addEventListener('pointerleave', leave); cv.addEventListener('pointerdown', down); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerleave', leave); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
  $effect(() => { playing; speed; wake(); });

  const legend = M === 'elbow'
    ? [{ label: 'final inertia', color: '--viz-gradient', shape: 'line' as const }, { label: 'selected k', color: '--accent' }]
    : M === 'init'
      ? [{ label: 'cluster 1', color: '--viz-pos' }, { label: 'cluster 2', color: '--viz-neg' }, { label: 'cluster 3', color: '--viz-forward' }, { label: 'cluster 4', color: '--accent' }]
      : [{ label: 'cluster 1', color: '--viz-pos' }, { label: 'cluster 2', color: '--viz-neg' }, { label: 'cluster 3', color: '--viz-forward' }, ...(M === 'update' ? [{ label: 'inertia', color: '--viz-gradient', shape: 'line' as const }] : [])];
  const labels: Record<Mode, string> = {
    structure: 'A plane of grey points in three loose groups. Later the points take three colours, one per group that k-means finds. On the right, a bar chart of how many points are in each cluster.',
    assign: 'A plane of points and three centroids drawn as heavy circles. The plane is shaded by nearest centroid and each point takes the colour of its nearest centroid, then the centroids move to the means. On the right, a bar chart of cluster sizes.',
    update: 'A plane of points and three centroids that move with trails while points change colour. On the right, a line chart of inertia falling with each iteration.',
    init: 'Two panels showing the same four groups of points. Left: k-means from a random start. Right: k-means from a k-means++ start. Each panel shows its inertia.',
    elbow: 'Left: a plane of points clustered with the selected number of clusters k. Right: a line chart of final inertia against k from 1 to 8, with the selected k highlighted.',
  };
</script>

<FigureCard title={init.title} role="group" aspect={M === 'init' ? '720 / 380' : '720 / 400'} narrowAspect={M === 'init' ? '360 / 740' : '360 / 660'} label={labels[M]} {legend}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={dur} markers={mk} label="Timeline: the algorithm running"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= dur ? 0 : to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if M === 'init'}
      <p class="vals num">random start: inertia <b>{fmt(ui.inA)}</b> · k-means++ start: inertia <b>{fmt(ui.inB)}</b></p>
      <p class="base">Final: random <b class="n">{f2(finalInertia(run), 1)}</b> after {run.n} passes · k-means++ <b class="n">{f2(finalInertia(runB!), 1)}</b> after {runB!.n} passes (seed {seed})</p>
    {:else if M === 'elbow'}
      <p class="vals num">k = <b>{kSel}</b> · pass <b>{ui.iter + 1}</b> of {run.n} · inertia now <b>{fmt(ui.inertia)}</b> · final <b>{f2(el!.inertia[kSel - 1], 1)}</b></p>
    {:else if M === 'structure'}
      <p class="vals num">{data.n} points · 2 features · labels given: <b>none</b></p>
    {:else}
      <p class="vals num">pass <b>{ui.iter + 1}</b> · inertia <b>{fmt(ui.inertia)}</b>{#if ui.phase !== 'start'}{' '}· points that changed cluster: <b>{ui.moved}</b>{/if}
        {#if ui.hov}<span class="base">· point {ui.hov.i + 1}: squared distance to each centroid {#each ui.hov.d as d, q}{q ? ', ' : ''}<span class:best={q === ui.hov.best}>{q + 1}: {f2(d, 2)}</span>{/each}</span>{/if}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if M === 'elbow'}
        <span>Clusters k</span>
        <span class="ks" role="group" aria-label="Number of clusters">
          {#each Array.from({ length: K_MAX }, (_, i) => i + 1) as k (k)}
            <button type="button" class="btn" aria-pressed={k === kSel} onclick={() => pickK(k)}>{k}</button>
          {/each}
        </span>
        <span class="hint">Or click a point on the chart. Each k is the best of 5 k-means++ starts.</span>
      {:else if M === 'init'}
        <button type="button" class="btn" onclick={reseed}>Reseed both starts</button>
        <span class="hint">Both sides use the same seed. Try several: random starts sometimes get lucky.</span>
      {:else if M === 'assign'}
        <button type="button" class="btn" onclick={reseed}>New random start</button>
        <span class="hint">Drag a heavy circle to move a starting centroid, or hover a point to see its distances.</span>
      {:else if M === 'update'}
        <button type="button" class="btn" onclick={reseed}>New random start</button>
        <button type="button" class="btn" onclick={reset}>Clear my points</button>
        <span class="hint">Click the plane to add a point{extra ? ` (${extra} added)` : ''}; hover a point to see its distances.</span>
      {:else}
        <button type="button" class="btn" onclick={reset}>Clear my points</button>
        <span class="hint">Click the plane to add a point{extra ? ` (${extra} added)` : ''}; k-means runs again.</span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font-size: 0.8rem; color: var(--text-muted); font-variant-numeric: tabular-nums; }
  .base .best { color: var(--text); font-weight: 650; }
  .base b.n { color: var(--text); font-family: var(--font-code); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); font: var(--step--1) var(--font-ui); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .ks { display: inline-flex; flex-wrap: wrap; gap: var(--s-1); }
  .ks .btn { min-width: 2.2em; padding: 0.4em 0.6em; }
  .ks .btn[aria-pressed='true'] { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
