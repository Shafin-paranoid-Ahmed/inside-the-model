<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { FlowRun } from './sim';
  let runP: Promise<FlowRun> | null = null;
  let progress = 0;
  /** Train the velocity network once per page in a worker; every figure on the page shares the result. */
  function load(): Promise<FlowRun> {
    return (runP ??= new Promise((res, rej) => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.kind === 'progress') progress = e.data.frac;
        else { progress = 1; res(e.data.run); w.terminate(); }
      };
      w.onerror = () => rej(new Error('worker failed'));
      w.postMessage(0);
    }));
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, reducedMotion, advance } from '../_shared';
  import {
    MOONS, quality, startNoise, moonData, N_DATA, TRAIN_STEPS, STEP_COUNTS, ANGLE_TS, T_MAX, T_SPOT, SPOT,
    exactVelFn, netVelFn, eulerBuilder, pathAtU, pathStats, endPoints, lossAt, lossFloor, spotVelocities, modeTime, pathTime,
    type Mode, type FlowRun, type Path,
  } from './sim';
  import { reverseBuilder, exactScoreFn, groupOf } from '../DiffusionDenoiser/sim';

  interface Props {
    /** 'flow' = learned field moves the cloud; 'pairs' = straight paths between noise and data points; 'marginal' = the exact field and one spot; 'train' = the network over training; 'sample' = Euler steps; 'compare' = diffusion against flow matching. */
    mode?: Mode;
    /** sample / compare: Euler steps (the starting budget when a selector is shown). */
    steps?: number;
    /** Show the selectors and the small chart. */
    showControls?: boolean;
    title?: string;
  }
  let { mode = 'flow', steps = 64, showControls = true, title = '' }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ mode, steps }));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let uiRead = $state(''), narration = $state('');
  let stepSel = $state(untrack(() => steps)), tSel = $state(0);
  let failed = false;

  const half = 2.2, M = init.mode;
  const needRun = M === 'flow' || M === 'train' || M === 'sample';
  const MT = modeTime(M);
  const zAll = startNoise(), dAll = moonData();
  const zc = zAll.slice(0, 400);
  const spot = spotVelocities(8);
  const PAIRS = 40;

  let run: FlowRun | null = null;
  const built = new Map<string, Path>(), endGrp = new Map<string, Uint8Array>(), building = new Set<string>();
  let qByS: Record<number, ReturnType<typeof quality>> = {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, wake = () => {};

  /** Advance a run a few steps per timer tick so the main thread never stalls. */
  function build(key: string, b: { done: boolean; step(): void; path: Path }) {
    if (built.has(key) || building.has(key)) return;
    building.add(key);
    const go = () => {
      const t0 = performance.now();
      while (!b.done && performance.now() - t0 < 10) b.step();
      if (b.done) { store(key, b.path); building.delete(key); wake(); } else setTimeout(go, 0);
    };
    go();
  }
  function store(key: string, p: Path) {
    built.set(key, p);
    const g = new Uint8Array(p.n); groupOf(MOONS, endPoints(p), g); endGrp.set(key, g);
  }
  $effect(() => {
    if (M === 'marginal') build('marg', eulerBuilder(exactVelFn, zAll, 100));
    if (M === 'compare') for (const S of [4, 64]) { build(`diff${S}`, reverseBuilder(exactScoreFn(MOONS), zc, 'ode', S, 5)); build(`flow${S}`, eulerBuilder(exactVelFn, zc, S)); }
    wake();
  });
  $effect(() => { playing; speed; stepSel; tSel; wake(); });

  let t = 0;
  uiT = t;
  const NAMES = {
    grad: '--viz-gradient', fwd: '--viz-forward', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent',
    s4: '--viz-seq-4', s5: '--viz-seq-5', c1: '--viz-cat-1', c2: '--viz-cat-2',
  };
  const pct = (v: number) => `${Math.round(100 * v)}%`;
  const qtext = (q: { onData: number; upper: number; lower: number }) => `${pct(q.onData)} on the moons (upper ${pct(q.upper)} / lower ${pct(q.lower)})`;

  onMount(() => {
    const ctx = cv.getContext('2d')!;
    let c = readColors(NAMES), font = readVar('--font-code'), ui = readVar('--font-ui'), lastUi = 0, lastKey = '';
    let W = 720, H = 420, S = 1, ox = 0, oy = 0;
    const X = (x: number) => ox + x * S, Y = (y: number) => oy - y * S;
    const pts = new Float32Array(2 * N_DATA), pts2 = new Float32Array(2 * N_DATA);

    const st = 0.4, L = 1.8, grid: number[] = [];
    for (let gx = -L; gx <= L + 1e-9; gx += st) for (let gy = -L; gy <= L + 1e-9; gy += st) grid.push(gx, gy);
    const G = Float64Array.from(grid), ng = G.length / 2, gv = new Float64Array(2 * ng), gv2 = new Float64Array(2 * ng);
    const arrowCache = new Map<string, Float64Array>();
    const gPairs = dAll.moon;

    const txt = (s: string, x: number, y: number, col: string, size = 11, align: CanvasTextAlign = 'left', weight = 600, mono = false) => {
      ctx.font = `${weight} ${size}px ${mono ? font : ui}`; ctx.fillStyle = col; ctx.textAlign = align; ctx.fillText(s, x, y);
    };
    const grpCol = (g: number) => (g === 0 ? c.c1 : c.c2).hex;
    const dots = (p: ArrayLike<number>, grp: Uint8Array | null, count: number, size: number, alpha = 1, col?: string) => {
      ctx.globalAlpha = alpha;
      for (let g = 0; g < (grp ? 2 : 1); g++) {
        ctx.fillStyle = col ?? grpCol(g);
        for (let i = 0; i < count; i++) if (!grp || grp[i] === g) ctx.fillRect(X(p[2 * i]) - size / 2, Y(p[2 * i + 1]) - size / 2, size, size);
      }
      ctx.globalAlpha = 1;
    };
    const arrow = (x1: number, y1: number, dx: number, dy: number, col: string, lw: number, head = 5) => {
      const x2 = x1 + dx, y2 = y1 + dy, m = Math.hypot(dx, dy); if (m < 1) return;
      const ux = dx / m, uy = dy / m;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - ux * head - uy * head * 0.6, y2 - uy * head + ux * head * 0.6); ctx.lineTo(x2 - ux * head + uy * head * 0.6, y2 - uy * head - ux * head * 0.6); ctx.fill();
    };
    const field = (v: Float64Array, col: string, lw: number, alpha: number) => {
      ctx.globalAlpha = alpha;
      for (let i = 0; i < ng; i++) {
        const m = Math.hypot(v[2 * i], v[2 * i + 1]); if (m < 1e-9) continue;
        const len = st * 0.78 * Math.tanh(m / 1.5) * S;
        arrow(X(G[2 * i]), Y(G[2 * i + 1]), (v[2 * i] / m) * len, (-v[2 * i + 1] / m) * len, col, lw, 4.5);
      }
      ctx.globalAlpha = 1;
    };
    const setPlane = (x: number, y: number, side: number) => { S = side / (2 * half + 0.2); ox = x + side / 2; oy = y + side / 2; };
    const frame = () => { ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(ox - half * S, oy - half * S, 2 * half * S, 2 * half * S); };
    const clip = () => { ctx.save(); ctx.beginPath(); ctx.rect(ox - half * S, oy - half * S, 2 * half * S, 2 * half * S); ctx.clip(); };
    const lines = (p: Path, count: number, step: number) => {
      ctx.lineWidth = 1; ctx.globalAlpha = 0.45;
      for (let i = 0; i < count; i += step) {
        ctx.beginPath();
        for (let s = 0; s <= p.S; s++) { const x = X(p.xs[s * 2 * p.n + 2 * i]), y = Y(p.xs[s * 2 * p.n + 2 * i + 1]); s ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };

    // ------------------------------------------------ training curves
    let lossPts: number[] = [], floor = 0;
    const prepLoss = () => {
      if (!run || lossPts.length) return;
      for (let s = 0; s <= TRAIN_STEPS; s += 20) lossPts.push(lossAt(run, Math.min(s, TRAIN_STEPS - 1)));
      floor = lossFloor();
    };
    const lossPlot = (px: number, py: number, pw: number, ph: number, snap: number) => {
      txt('training loss (log scale)', px, py - 12, c.muted.hex);
      const x0 = px + 34, w = pw - 34, lo = Math.log(Math.min(floor, ...lossPts) * 0.97), hi = Math.log(Math.max(...lossPts) * 1.03);
      const yy = (v: number) => py + ph - ((Math.log(v) - lo) / (hi - lo)) * ph, xx = (s: number) => x0 + (s / TRAIN_STEPS) * w;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x0, py, w, ph);
      for (const v of [1.2, 1.3, 1.4, 1.5]) if (Math.log(v) > lo && Math.log(v) < hi) { txt(String(v), x0 - 4, yy(v) + 4, c.muted.hex, 11, 'right', 500, true); ctx.beginPath(); ctx.moveTo(x0, yy(v)); ctx.lineTo(x0 + 4, yy(v)); ctx.stroke(); }
      ctx.strokeStyle = c.muted.hex; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x0, yy(floor)); ctx.lineTo(x0 + w, yy(floor)); ctx.stroke(); ctx.setLineDash([]);
      txt('best possible', x0 + 8, yy(floor) - 5, c.muted.hex, 11, 'left', 500);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2; ctx.beginPath();
      lossPts.forEach((v, k) => (k ? ctx.lineTo(xx(k * 20), yy(v)) : ctx.moveTo(xx(0), yy(v))));
      ctx.stroke();
      const step = run!.steps[snap];
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xx(step), py); ctx.lineTo(xx(step), py + ph); ctx.stroke();
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(xx(step), yy(lossAt(run!, Math.min(step, TRAIN_STEPS - 1))), 3.5, 0, 6.2832); ctx.fill();
      txt('updates →', x0 + w, py + ph + 14, c.muted.hex, 11, 'right', 500);
    };
    /** Small chart of the share on the moons for each step count. */
    const stepChart = (px: number, py: number, pw: number, ph: number) => {
      txt('on the moons, by number of steps', px, py - 12, c.muted.hex);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px, py, pw, ph);
      const bw = pw / STEP_COUNTS.length;
      STEP_COUNTS.forEach((s, k) => {
        const q = qByS[s]; if (!q) return;
        const h = q.onData * ph, on = s === stepSel;
        ctx.fillStyle = on ? c.s5.hex : c.s4.hex; ctx.globalAlpha = on ? 1 : 0.55; ctx.fillRect(px + k * bw + 6, py + ph - h, bw - 12, Math.max(1, h)); ctx.globalAlpha = 1;
        txt(pct(q.onData), px + k * bw + bw / 2, py + ph - h - 4, on ? c.text.hex : c.muted.hex, 11, 'center', on ? 700 : 500, true);
        txt(String(s), px + k * bw + bw / 2, py + ph + 14, on ? c.text.hex : c.muted.hex, 11, 'center', on ? 700 : 500, true);
      });
      txt('steps →', px + pw, py + ph + 28, c.muted.hex, 11, 'right', 500);
    };

    // ------------------------------------------------ drawing
    const frameFn = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), r = cv.getBoundingClientRect();
      W = r.width; H = r.height;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const wide = W >= 540, u = pathTime(M, t);
      const panel = M === 'train' || (M === 'sample' && showControls);
      const sw = wide && panel ? W * 0.62 : W, sh = wide || !panel ? H : H * 0.74, side = Math.min(sw, sh);
      const px = wide ? sw + 24 : 30, py = wide ? 60 : sh + 26, pw = wide ? W - sw - 44 : W - 60, ph = wide ? 150 : H - sh - 44;
      let key = M as string, read = '', q = { onData: 0, upper: 0, lower: 0 };

      if (M === 'compare') {
        const S0 = stepSel, kd = `diff${S0}`, kf = `flow${S0}`, pd = built.get(kd), pf = built.get(kf), n = zc.length / 2;
        const side2 = wide ? Math.min(W / 2 - 14, H - 34) : Math.min(W - 20, (H - 58) / 2);
        const pos = wide ? [[(W / 2 - side2) / 2, 28], [W / 2 + (W / 2 - side2) / 2, 28]] : [[(W - side2) / 2, 22], [(W - side2) / 2, 22 + side2 + 36]];
        const parts: string[] = [];
        [['diffusion panel: ideal denoiser', pd, kd], ['flow panel: ideal field', pf, kf]].forEach(([name, p, k], i) => {
          setPlane(pos[i][0], pos[i][1], side2); frame();
          txt(name as string, ox - half * S, oy - half * S - 8, c.text.hex, 12);
          clip();
          if (p) {
            const pp = p as Path, g = endGrp.get(k as string)!;
            ctx.strokeStyle = c.muted.hex; lines(pp, 40, 5);
            pathAtU(pp, u, pts); dots(pts, g, n, 3);
            const qq = quality(pts.subarray(0, 2 * n), MOONS); if (i === 0) q = qq;
            parts.push(`${i ? 'flow' : 'diffusion'}: ${pct(qq.onData)} on the moons, paths ${pathStats(pp).ratio.toFixed(1)}× the straight distance`);
          }
          ctx.restore();
        });
        read = `${S0} steps · ${parts.join(' · ')}`;
        key = 'compare';
        tl?.paint(t);
        return { key, read, q };
      }

      setPlane((sw - side) / 2, wide || !panel ? (H - side) / 2 : 0, side);
      frame(); clip();
      if (M === 'pairs') {
        const cloud = pts;
        for (let i = 0; i < N_DATA; i++) { cloud[2 * i] = u * dAll.x[2 * i] + (1 - u) * zAll[2 * i]; cloud[2 * i + 1] = u * dAll.x[2 * i + 1] + (1 - u) * zAll[2 * i + 1]; }
        dots(cloud, null, N_DATA, 2.5, 0.35, c.muted.hex);
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
        for (let i = 0; i < PAIRS; i++) { ctx.beginPath(); ctx.moveTo(X(zAll[2 * i]), Y(zAll[2 * i + 1])); ctx.lineTo(X(dAll.x[2 * i]), Y(dAll.x[2 * i + 1])); ctx.stroke(); }
        ctx.globalAlpha = 1;
        dots(cloud, gPairs, PAIRS, 5);
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(zAll[0]), Y(zAll[1])); ctx.lineTo(X(dAll.x[0]), Y(dAll.x[1])); ctx.stroke();
        ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.arc(X(cloud[0]), Y(cloud[1]), 5.5, 0, 6.2832); ctx.fill();
        const x0x = zAll[0], x0y = zAll[1], x1x = dAll.x[0], x1y = dAll.x[1];
        read = `t = ${u.toFixed(2)} · pair 1: x_t = ${(1 - u).toFixed(2)}·(${x0x.toFixed(2)}, ${x0y.toFixed(2)}) + ${u.toFixed(2)}·(${x1x.toFixed(2)}, ${x1y.toFixed(2)}) = (${cloud[0].toFixed(2)}, ${cloud[1].toFixed(2)}) · velocity x₁ − x₀ = (${(x1x - x0x).toFixed(2)}, ${(x1y - x0y).toFixed(2)}), the same at every t`;
        key = u < 1 ? 'moving' : 'done';
      } else if (M === 'marginal') {
        const p = built.get('marg');
        exactVelFn(u, G, ng, gv); field(gv, c.grad.hex, 1.3, 0.8);
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.globalAlpha = 0.3;
        for (let i = 0; i < PAIRS; i++) { ctx.beginPath(); ctx.moveTo(X(zAll[2 * i]), Y(zAll[2 * i + 1])); ctx.lineTo(X(dAll.x[2 * i]), Y(dAll.x[2 * i + 1])); ctx.stroke(); }
        ctx.globalAlpha = 1;
        if (p) { pathAtU(p, u, pts); dots(pts, endGrp.get('marg')!, 300, 3); } else dots(zAll, null, 300, 3, 1, c.muted.hex);
        const hold = t >= 3 && t < 4.6;
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(SPOT[0]), Y(SPOT[1]), 4, 0, 6.2832); ctx.stroke();
        if (hold) {
          const k = 0.42 * S;
          for (let i = 0; i < spot.vs.length / 2; i++) arrow(X(SPOT[0]), Y(SPOT[1]), spot.vs[2 * i] * k, -spot.vs[2 * i + 1] * k, c.muted.hex, 1, 4);
          arrow(X(SPOT[0]), Y(SPOT[1]), spot.exact[0] * k, -spot.exact[1] * k, c.text.hex, 3, 7);
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(SPOT[0]), Y(SPOT[1]), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        }
        let sp = 0; const nv = spot.vs.length / 2; for (let i = 0; i < nv; i++) sp += Math.hypot(spot.vs[2 * i], spot.vs[2 * i + 1]);
        read = hold
          ? `at (${SPOT[0].toFixed(1)}, ${SPOT[1].toFixed(1)}), t = ${T_SPOT.toFixed(1)}: ${nv} sampled paths pass here, each at an average speed of ${(sp / nv).toFixed(2)}, in very different directions · the field's arrow (black) is their average: (${spot.exact[0].toFixed(2)}, ${spot.exact[1].toFixed(2)}), speed ${Math.hypot(spot.exact[0], spot.exact[1]).toFixed(2)}`
          : `t = ${u.toFixed(2)}`;
        key = hold ? 'hold' : 'moving';
      } else if (M === 'train') {
        prepLoss();
        if (!run) { ctx.restore(); txt(failed ? 'Training failed in this browser.' : `Training the network in your browser: ${Math.round(100 * progress)}%`, W / 2, H / 2, c.muted.hex, 13, 'center', 500); tl?.paint(t); return { key: 'wait', read: '', q }; }
        const snap = Math.min(run.snaps.length - 1, Math.floor((t / MT.dur) * run.snaps.length)), tt = ANGLE_TS[tSel];
        const ek = `e${tSel}`, lk = `l${tSel}:${snap}`;
        let ex = arrowCache.get(ek); if (!ex) { exactVelFn(tt, G, ng, gv); arrowCache.set(ek, (ex = gv.slice())); }
        let lr = arrowCache.get(lk); if (!lr) { netVelFn(run.snaps[snap])(tt, G, ng, gv2); arrowCache.set(lk, (lr = gv2.slice())); }
        const cloud = new Float64Array(2 * N_DATA);
        for (let i = 0; i < 2 * N_DATA; i++) cloud[i] = tt * dAll.x[i] + (1 - tt) * zAll[i];
        dots(cloud, null, N_DATA, 2.5, 0.4, c.muted.hex);
        field(ex, c.muted.hex, 1, 0.55); field(lr, c.grad.hex, 1.5, 0.95);
        ctx.restore();
        lossPlot(px, py, pw, ph, snap);
        const step = run.steps[snap];
        read = `after ${step} updates: loss ${lossAt(run, Math.min(step, TRAIN_STEPS - 1)).toFixed(2)}, learned arrows differ from the exact ones by ${run.angles[tSel][snap].toFixed(0)}° on average (t = ${tt})`;
        tl?.paint(t);
        return { key: 'train', read, q };
      } else {
        // flow and sample: the learned field and its Euler samples
        const S0 = M === 'flow' ? 64 : stepSel, p = run?.paths?.[S0];
        if (!run || !p) { ctx.restore(); txt(failed ? 'Training failed in this browser.' : `Training the network in your browser: ${Math.round(100 * progress)}%`, W / 2, H / 2, c.muted.hex, 13, 'center', 500); tl?.paint(t); return { key: 'wait', read: '', q }; }
        const g = endGrp.get(`s${S0}`)!;
        if (M === 'flow') {
          netVelFn(run.snaps[run.snaps.length - 1])(Math.min(u, T_MAX), G, ng, gv); field(gv, c.grad.hex, 1.2, 0.6);
          pathAtU(p, Math.max(0, u - 0.08), pts2); pathAtU(p, u, pts);
          ctx.lineWidth = 1; ctx.globalAlpha = 0.4;
          for (let k = 0; k < 2; k++) { ctx.strokeStyle = grpCol(k); ctx.beginPath(); for (let i = 0; i < N_DATA; i++) if (g[i] === k) { ctx.moveTo(X(pts2[2 * i]), Y(pts2[2 * i + 1])); ctx.lineTo(X(pts[2 * i]), Y(pts[2 * i + 1])); } ctx.stroke(); }
          ctx.globalAlpha = 1;
          dots(pts, g, N_DATA, 3);
        } else {
          pathAtU(p, u, pts);
          // one highlighted particle: its trail and the point reached after every step
          ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath();
          for (let s = 0; s <= p.S; s++) { const x = X(p.xs[s * 2 * p.n]), y = Y(p.xs[s * 2 * p.n + 1]); s ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
          ctx.stroke();
          dots(pts, g, N_DATA, 3, 0.9);
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5;
          for (let s = 0; s <= p.S; s++) { ctx.beginPath(); ctx.arc(X(p.xs[s * 2 * p.n]), Y(p.xs[s * 2 * p.n + 1]), p.S > 16 ? 1.6 : 3.5, 0, 6.2832); if (p.S > 16) { ctx.fillStyle = c.text.hex; ctx.fill(); } else { ctx.fill(); ctx.stroke(); } }
        }
        q = quality(pts, MOONS);
        const done = Math.min(S0, Math.floor(u * S0 + 1e-9));
        read = M === 'flow' ? `t = ${u.toFixed(2)} · ${qtext(q)}` : `${S0} Euler step${S0 > 1 ? 's' : ''}, ${done} taken (t = ${(done / S0).toFixed(2)}) · ${qtext(q)}`;
        key = u < 1 ? 'moving' : 'done';
      }
      ctx.restore();
      if (M === 'sample' && showControls) stepChart(px, py, pw, ph);
      tl?.paint(t);
      return { key, read, q };
    };

    const NARR: Record<string, string> = {
      'flow:moving': 'Every point follows the learned velocity field from noise (t = 0) to data (t = 1). The pink arrows show the field at the current time.',
      'pairs:moving': 'Each noise point is joined to a data point by a straight line, and slides along it at constant speed. The coloured dots are 40 of the 600 pairs, one of them highlighted; the grey cloud is all of them.',
      'marginal:moving': 'The arrows are the exact field. Particles follow it from noise. The faint grey lines are the straight pair paths that the field replaces.',
      'marginal:hold': 'Paused at t = 0.5 on one spot: many different straight paths pass through it, each with its own velocity. A field can only give one arrow here, and it is their average.',
      'sample:moving': 'The sampler takes a few large Euler steps. The dark line is one particle: each circle is where it stands after a step, and it goes straight between them.',
      'compare:': 'Same noise, same moons, same number of steps. The diffusion panel runs the probability-flow ODE of diffusion with the ideal denoiser. The flow panel runs flow matching with the ideal velocity field.',
      'train:': 'Each frame is a real snapshot of the network. Thin grey arrows are the exact field; pink arrows are what the network predicts.',
    };
    const say = (f: { key: string; q: { onData: number; upper: number; lower: number } }): string => {
      if (f.key === 'wait') return 'Training the network in your browser…';
      if (f.key === 'done') return M === 'pairs' ? 'Every point has arrived at its data point. Noise became data along the straight lines.' : `Finished: ${qtext(f.q)}.`;
      return NARR[`${M}:${f.key === 'moving' || f.key === 'hold' ? f.key : ''}`] ?? '';
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: MT.dur, loopFrom: 0, loopTo: MT.dur }).t;
      const f = frameFn();
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        const k = f.key + (run ? 'r' : '') + built.size;
        if (k !== lastKey) { lastKey = k; narration = say(f); }
        uiRead = f.read;
      }
      return playing || (needRun && !run && !failed);
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    if (needRun) {
      load().then((r) => {
        run = r;
        if (r.paths) for (const s of STEP_COUNTS) { store(`s${s}`, r.paths[s]); qByS[s] = quality(endPoints(r.paths[s]), MOONS); }
        lastKey = ''; wake();
      }, () => { failed = true; wake(); });
    }
    const ro = new ResizeObserver(() => wake()); ro.observe(cv);
    const off = onThemeChange(() => { c = readColors(NAMES); font = readVar('--font-code'); ui = readVar('--font-ui'); wake(); });
    document.fonts.ready.then(() => wake());
    return () => { loop.destroy(); ro.disconnect(); off(); };
  });

  const TITLES: Record<Mode, string> = {
    flow: 'Noise flows into data', pairs: 'Straight paths from noise to data', marginal: 'One arrow per place and time', train: 'Training the velocity network',
    sample: 'Sampling in Euler steps', compare: 'Diffusion and flow matching, step for step',
  };
  const LABELS: Record<Mode, string> = {
    flow: 'A cloud of noise points that move along a learned velocity field and land on two moons, with arrows showing the field.',
    pairs: 'Forty straight lines each joining one noise point to one data point, with dots sliding along them and the full cloud faint behind.',
    marginal: 'The exact velocity field as arrows, particles following it, faint straight pair paths, and a paused view of many path velocities at one spot with their average.',
    train: 'Arrows of the velocity network at different points of training, drawn over the exact arrows, with the training loss curve.',
    sample: 'Points moving from noise to the moons in a few large Euler steps, one particle highlighted with a circle after each step.',
    compare: 'Two panels with the same starting noise, a diffusion panel and a flow panel, at the same number of steps.',
  };
  const legend = untrack(() => {
    const m = [{ label: 'ends on upper moon', color: '--viz-cat-1' }, { label: 'ends on lower moon', color: '--viz-cat-2' }];
    if (M === 'train') return [{ label: 'network', color: '--viz-gradient', shape: 'line' as const }, { label: 'exact', color: '--text-muted', shape: 'line' as const }, { label: 'loss', color: '--viz-forward', shape: 'line' as const }];
    if (M === 'marginal') return [{ label: 'field', color: '--viz-gradient', shape: 'line' as const }, { label: 'straight pair paths', color: '--text-muted', shape: 'line' as const }, ...m];
    if (M === 'pairs') return [{ label: 'all 600 pairs', color: '--text-muted' }, ...m];
    if (M === 'flow') return [...m, { label: 'learned field', color: '--viz-gradient', shape: 'line' as const }];
    return [...m, { label: M === 'compare' ? 'paths of 40 points' : 'one particle', color: '--text-muted', shape: 'line' as const }];
  });
  const narrow = untrack(() => (M === 'compare' ? '360 / 560' : '360 / 520'));
</script>

<FigureCard title={title || TITLES[M]} aspect="720 / 420" narrowAspect={narrow} label={LABELS[M]} {legend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={MT.dur} markers={MT.markers} label="Flow time"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= MT.dur ? 0 : to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next num">{uiRead}</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls && (M === 'train' || M === 'sample' || M === 'compare')}
      <div class="ctl">
        {#if M === 'train'}
          <fieldset class="seg"><legend class="sr-only">Time of the arrows</legend>
            <label><input type="radio" name="t-{uid}" value={0} bind:group={tSel} /><span>t = 0.5 (halfway)</span></label>
            <label><input type="radio" name="t-{uid}" value={1} bind:group={tSel} /><span>t = 0.9 (nearly data)</span></label>
          </fieldset>
        {:else if M === 'sample'}
          <fieldset class="seg"><legend class="sr-only">Number of Euler steps</legend>
            {#each STEP_COUNTS as s (s)}
              <label><input type="radio" name="s-{uid}" value={s} bind:group={stepSel} onchange={() => { t = 0; wake(); }} /><span>{s} {s === 1 ? 'step' : 'steps'}</span></label>
            {/each}
          </fieldset>
        {:else}
          <fieldset class="seg"><legend class="sr-only">Step budget</legend>
            {#each [4, 64] as s (s)}
              <label><input type="radio" name="b-{uid}" value={s} bind:group={stepSel} onchange={() => { t = 0; wake(); }} /><span>{s} steps</span></label>
            {/each}
          </fieldset>
        {/if}
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  canvas { touch-action: pan-y; }
  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); font-family: var(--font-ui); font-variant-numeric: tabular-nums; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; color: var(--text-muted); }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
