<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { TrainRun } from './sim';
  let runP: Promise<TrainRun> | null = null;
  let progress = 0;
  /** Train the noise-prediction network once per page in a worker; every figure on the page shares the result. */
  function load(): Promise<TrainRun> {
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
  import { createRng } from '../../../lib/math/rng';
  import { normal } from '../../../lib/gen/mlp';
  import {
    mixture, sampleData, forwardAt, pathAt, tauAt, alphaBar, epsHat, modeTime, sweepTau, groupOf, quality, moonStats, lossAt, lossFloor,
    reverseBuilder, exactScoreFn, guidedScoreFn, netScoreFn, imageRun, imageAt, GLYPHS, ANGLE_TAUS, REVERSE_STEPS, GUIDE_STEPS, TAU_MIN, TRAIN_STEPS,
    type Dataset, type Sampler, type Path, type Phase, type Mode, type TrainRun, type ScoreFn,
  } from './sim';

  interface Props {
    /** 'both' loops noise-in then denoise; 'forward' / 'reverse' loop one half; 'closed' = the one-jump formula for one point; 'field' = the denoising direction everywhere; 'train' = the learned network over training; 'image' = 8x8 glyphs; 'guide' = guided sampling. */
    mode?: Mode;
    dataset?: Dataset;
    /** Which denoiser drives the reverse process and the arrows: the closed-form ideal one, or the trained network. */
    denoiser?: 'exact' | 'learned';
    /** Reverse process: 'ode' (deterministic probability flow) or 'sde' (adds fresh noise each step). */
    sampler?: Sampler;
    /** Draw the denoising-direction arrows. */
    showScore?: boolean;
    /** Show the selectors under the figure. */
    showControls?: boolean;
    /** Number of points. */
    n?: number;
    /** closed: index of the data point whose two parts are drawn. */
    focus?: number;
    /** guide: strength w and the label. */
    guidance?: number;
    label?: 'upper' | 'lower';
    /** image: which digit is noised (0 to 9). */
    glyph?: number;
    title?: string;
  }
  let {
    mode = 'both', dataset = 'moons', denoiser = 'exact', sampler = 'ode', showScore = true, showControls = true, n = 600, focus = 0, guidance = 3,
    label = 'upper', glyph = 3, title = '',
  }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ mode, dataset, n, denoiser, focus, glyph }));
  let smp = $state<Sampler>(untrack(() => sampler));
  let arrows = $state(untrack(() => showScore));
  let tauSel = $state(0), lbl = $state<'upper' | 'lower'>(untrack(() => label)), wSel = $state(untrack(() => guidance));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let uiRead = $state(''), narration = $state('');
  let failed = false;

  const PROBE: [number, number] = [1.5, 0];
  const mix = mixture(init.dataset), moons = init.dataset === 'moons', half = moons ? 2.2 : 1.7;
  const data = sampleData(mix, init.n, 7);
  const gData = new Uint8Array(init.n); groupOf(mix, data.x, gData);
  const eps = (() => { const r = createRng(8); return Float32Array.from({ length: 2 * init.n }, () => normal(r)); })();
  const xT = new Float32Array(2 * init.n); forwardAt(1, data.x, eps, xT);
  const needRun = init.denoiser === 'learned' || init.mode === 'train';
  const MT = modeTime(init.mode);
  const imRun = imageRun(init.glyph);

  let run: TrainRun | null = null, learned: ScoreFn | null = null;
  const paths = new Map<string, Path>(), endGroup = new Map<string, Uint8Array>(), building = new Set<string>();
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, wake = () => {};
  /** Advance a reverse run a few steps per timer tick so the main thread never stalls. */
  function ensure(key: string, sf: ScoreFn, kind: Sampler, S: number) {
    if (paths.has(key) || building.has(key)) return;
    building.add(key);
    const b = reverseBuilder(sf, xT, kind, S, 5), go = () => {
      const t0 = performance.now();
      while (!b.done && performance.now() - t0 < 10) b.step();
      if (b.done) { setPath(key, b.path); building.delete(key); wake(); } else setTimeout(go, 0);
    };
    go();
  }
  function setPath(key: string, p: Path) {
    paths.set(key, p);
    const end = new Float32Array(2 * init.n); pathAt(p, TAU_MIN, end);
    const g = new Uint8Array(init.n); groupOf(mix, end, g); endGroup.set(key, g);
  }
  const keyOf = () => (init.mode === 'guide' ? `g:${lbl}:${wSel}` : `${init.denoiser}:${smp}`);
  $effect(() => {
    const k = keyOf();
    if (init.mode === 'guide') ensure(k, guidedScoreFn(mix, lbl, wSel), 'ode', GUIDE_STEPS);
    else if (init.denoiser === 'exact' && (init.mode === 'both' || init.mode === 'reverse')) ensure(k, exactScoreFn(mix), smp, REVERSE_STEPS);
    wake();
  });
  $effect(() => { playing; speed; arrows; tauSel; wake(); });

  let t = 0;
  uiT = t;
  const NAMES = {
    grad: '--viz-gradient', fwd: '--viz-forward', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', pos: '--viz-pos', neg: '--viz-neg',
    s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5', c1: '--viz-cat-1', c2: '--viz-cat-2',
  };

  const TEXT: Record<Phase, string> = {
    data: 'Clean data: 600 points on two moons.',
    forward: 'Forward process: every point shrinks toward the origin while fresh Gaussian noise is mixed in.',
    noise: 'Almost pure noise: the moons are gone. The arrows now point toward the origin.',
    reverse: 'Reverse process: follow the denoising direction back. Points move toward regions of higher density.',
    clean: 'Denoised: the points have settled back on the moons.',
  };
  const NARR: Partial<Record<Mode, string>> = {
    closed: 'One jump to any noise level: the clean point shrinks by √ᾱ, a fixed noise vector grows by √(1−ᾱ), and the two arrows add up to the noisy point.',
    field: 'At every spot the arrow gives the direction that raises the data density, which is the direction the denoiser wants to move a noisy point.',
    guide: 'Guided sampling from the same noise: the label pulls the points onto one moon, and the guidance strength w controls how hard.',
  };

  onMount(() => {
    const ctx = cv.getContext('2d')!;
    let c = readColors(NAMES), font = readVar('--font-code'), ui = readVar('--font-ui'), lastUi = 0, lastKey = '';
    const pts = new Float32Array(2 * init.n), pts2 = new Float32Array(2 * init.n);
    let W = 720, H = 420, S = 1, ox = 0, oy = 0;
    const X = (x: number) => ox + x * S, Y = (y: number) => oy - y * S;
    let cursor: [number, number] | null = null;

    // arrow grid
    const st = moons ? 0.4 : 0.34, L = Math.floor((half - 0.25) / st) * st, grid: number[] = [];
    for (let gx = -L; gx <= L + 1e-9; gx += st) for (let gy = -L; gy <= L + 1e-9; gy += st) grid.push(gx, gy);
    const G = Float64Array.from(grid), ng = G.length / 2, gsc = new Float64Array(2 * ng), gsc2 = new Float64Array(2 * ng);
    const exact = exactScoreFn(mix), one = new Float64Array(2), cur1 = new Float64Array(2);
    const sfNow = (): ScoreFn => (init.denoiser === 'learned' && run ? (learned ??= netScoreFn(run.snaps[run.snaps.length - 1])) : exact);
    const scoreAt = (sf: ScoreFn, tau: number, x: number, y: number) => { cur1[0] = x; cur1[1] = y; sf(tau, cur1, 1, one); return one; };
    const arrowCache = new Map<string, Float64Array>();

    // training curves
    let lossPts: number[] = [], floor = 0;
    const prepLoss = () => {
      if (!run || lossPts.length) return;
      for (let s = 0; s <= TRAIN_STEPS; s += 20) lossPts.push(lossAt(run, Math.min(s, TRAIN_STEPS - 1)));
      floor = lossFloor();
    };

    const txt = (s: string, x: number, y: number, col: string, size = 11, align: CanvasTextAlign = 'left', weight = 600, mono = false) => {
      ctx.font = `${weight} ${size}px ${mono ? font : ui}`; ctx.fillStyle = col; ctx.textAlign = align; ctx.fillText(s, x, y);
    };
    const grpCol = (g: number) => (moons ? (g === 0 ? c.c1 : c.c2).hex : [c.s2, c.s3, c.s4, c.s5][g].hex);
    const dots = (p: ArrayLike<number>, grp: Uint8Array | null, size: number, alpha = 1, col?: string) => {
      ctx.globalAlpha = alpha;
      for (let g = 0; g < (grp ? 4 : 1); g++) {
        ctx.fillStyle = col ?? grpCol(g);
        for (let i = 0; i < init.n; i++) if (!grp || grp[i] === g) ctx.fillRect(X(p[2 * i]) - size / 2, Y(p[2 * i + 1]) - size / 2, size, size);
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
    const field = (sc: Float64Array, col: string, lw: number, alpha: number) => {
      ctx.globalAlpha = alpha;
      for (let i = 0; i < ng; i++) {
        const m = Math.hypot(sc[2 * i], sc[2 * i + 1]); if (m < 1e-9) continue;
        const len = st * 0.78 * Math.tanh(m / 2.5) * S;
        arrow(X(G[2 * i]), Y(G[2 * i + 1]), (sc[2 * i] / m) * len, (-sc[2 * i + 1] / m) * len, col, lw, 4.5);
      }
      ctx.globalAlpha = 1;
    };

    const sharePlot = (px: number, py: number, pw: number, ph: number, tau: number, title: string) => {
      txt(title, px, py - 12, c.muted.hex);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px, py, pw, ph);
      for (const [f, col] of [[(a: number) => Math.sqrt(a), c.s5.hex], [(a: number) => Math.sqrt(1 - a), c.fwd.hex]] as const) {
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        for (let k = 0; k <= 60; k++) { const tt = k / 60, v = f(alphaBar(tt)); k ? ctx.lineTo(px + tt * pw, py + ph - v * ph) : ctx.moveTo(px, py + ph - v * ph); }
        ctx.stroke();
      }
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px + tau * pw, py); ctx.lineTo(px + tau * pw, py + ph); ctx.stroke();
      txt('data share √ᾱ', px + 4, py + 13, c.s5.hex); txt('noise share √(1−ᾱ)', px + pw * 0.34, py + ph - 7, c.fwd.hex);
      txt('noise level τ →', px + pw, py + ph + 14, c.muted.hex, 11, 'right', 500);
    };

    const lossPlot = (px: number, py: number, pw: number, ph: number, snap: number) => {
      txt('training loss (log scale)', px, py - 12, c.muted.hex);
      const x0 = px + 34, w = pw - 34, lo = Math.log(Math.min(floor, ...lossPts) * 0.92), hi = Math.log(Math.max(...lossPts) * 1.05);
      const yy = (v: number) => py + ph - ((Math.log(v) - lo) / (hi - lo)) * ph, xx = (s: number) => x0 + (s / TRAIN_STEPS) * w;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x0, py, w, ph);
      for (const v of [0.4, 0.5, 0.6]) if (Math.log(v) > lo && Math.log(v) < hi) { txt(String(v), x0 - 4, yy(v) + 4, c.muted.hex, 11, 'right', 500, true); ctx.beginPath(); ctx.moveTo(x0, yy(v)); ctx.lineTo(x0 + 4, yy(v)); ctx.stroke(); }
      ctx.strokeStyle = c.muted.hex; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x0, yy(floor)); ctx.lineTo(x0 + w, yy(floor)); ctx.stroke(); ctx.setLineDash([]);
      txt('best possible', x0 + w - 4, yy(floor) - 5, c.muted.hex, 11, 'right', 500);
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2; ctx.beginPath();
      lossPts.forEach((v, k) => (k ? ctx.lineTo(xx(k * 20), yy(v)) : ctx.moveTo(xx(0), yy(v))));
      ctx.stroke();
      const step = run!.steps[snap];
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xx(step), py); ctx.lineTo(xx(step), py + ph); ctx.stroke();
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(xx(step), yy(lossAt(run!, Math.min(step, TRAIN_STEPS - 1))), 3.5, 0, 6.2832); ctx.fill();
      txt('updates →', x0 + w, py + ph + 14, c.muted.hex, 11, 'right', 500);
    };

    // ------------------------------------------------ the 8x8 image mode
    const xtI = new Float64Array(64), gsI = new Float64Array(64), nzI = new Float64Array(64), wI = new Float64Array(10);
    const mixc = (a: [number, number, number], b: [number, number, number], u: number) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(',')})`;
    function drawImage() {
      const tt = t, tau = sweepTau('image', tt), up = tt < 5.6;
      imageAt(imRun, tau, up, xtI, gsI, nzI, wI);
      const gap = 14, BH = 90, pw = Math.max(60, Math.min((W - 4 * gap) / 3, H - (BH + 90))), x0 = (W - (3 * pw + 2 * gap)) / 2, oy0 = Math.max(0, (H - (pw + BH + 90)) / 2), y0 = oy0 + 34, cell = pw / 8;
      const titles = ['noisy image', 'best guess of the clean image', 'noise in the image'];
      for (let p = 0; p < 3; p++) {
        const px = x0 + p * (pw + gap);
        const wrap = (s: string) => (W < 540 && s.length > 14 ? s.replace(/ (of|in) /, '\n$1 ').split('\n') : [s]);
        wrap(titles[p]).forEach((ln, i) => txt(ln, px + pw / 2, oy0 + 16 + i * 12, c.muted.hex, 11, 'center'));
        for (let r = 0; r < 8; r++) for (let q = 0; q < 8; q++) {
          const i = r * 8 + q, v = p === 0 ? xtI[i] : p === 1 ? gsI[i] : nzI[i];
          if (p < 2) ctx.fillStyle = mixc(c.raised.rgb, c.text.rgb, Math.max(0, Math.min(1, (v + 1) / 2)));
          else { ctx.fillStyle = c.raised.hex; ctx.fillRect(px + q * cell, y0 + r * cell, cell, cell); ctx.globalAlpha = Math.min(1, Math.abs(v) / 2.5); ctx.fillStyle = v >= 0 ? c.pos.hex : c.neg.hex; }
          ctx.fillRect(px + q * cell, y0 + r * cell, cell + 0.5, cell + 0.5); ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px, y0, pw, pw);
      }
      // posterior bars
      const by = y0 + pw + 38, bh = BH, bw = (3 * pw + 2 * gap) / 10;
      txt('how much the guess leans on each digit', x0, by - 8, c.muted.hex);
      let best = 0; for (let k = 1; k < 10; k++) if (wI[k] > wI[best]) best = k;
      for (let k = 0; k < 10; k++) {
        const h = wI[k] * bh;
        ctx.fillStyle = c.rule.hex; ctx.fillRect(x0 + k * bw + 4, by, bw - 8, bh);
        ctx.fillStyle = k === best ? c.s5.hex : c.s3.hex; ctx.fillRect(x0 + k * bw + 4, by + bh - h, bw - 8, Math.max(1, h));
        txt(String(k), x0 + k * bw + bw / 2, by + bh + 13, k === best ? c.text.hex : c.muted.hex, 11, 'center', k === best ? 700 : 500, true);
      }
      const phase = tt < 5 ? 'noising' : tt < 5.6 ? 'static' : tt < 10.6 ? 'generating' : 'done';
      return { tau, phase, best, wb: wI[best] };
    }

    // ------------------------------------------------ everything else
    const draw = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), r = cv.getBoundingClientRect();
      W = r.width; H = r.height;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const m = init.mode;
      if (m === 'image') { const f = drawImage(); tl?.paint(t); return { tau: f.tau, phase: f.phase as string, extra: `${f.best}|${Math.round(f.wb * 100)}` }; }
      const wide = W >= 540;
      const plot = m !== 'guide' && m !== 'field', sw = wide && plot ? W * 0.66 : W, sh = wide || !plot ? H : H * 0.76;
      S = Math.min(sw, sh) / (2 * half + 0.2); ox = sw / 2; oy = sh / 2;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(ox - half * S, oy - half * S, 2 * half * S, 2 * half * S);
      ctx.save(); ctx.beginPath(); ctx.rect(ox - half * S, oy - half * S, 2 * half * S, 2 * half * S); ctx.clip(); // stray points never spill onto the side plot
      let tau = 0, phase = '', extra = '', dim = false;

      if (m === 'train') {
        prepLoss();
        if (!run) { ctx.restore(); txt(failed ? 'Training failed in this browser.' : `Training the network in your browser: ${Math.round(100 * progress)}%`, W / 2, H / 2, c.muted.hex, 13, 'center', 500); tl?.paint(t); return { tau: 0, phase: 'wait', extra: '' }; }
        const snap = Math.min(run.snaps.length - 1, Math.floor(t / MT.dur * run.snaps.length)); tau = ANGLE_TAUS[tauSel];
        const ek = `e${tauSel}`, lk = `l${tauSel}:${snap}`;
        let ex = arrowCache.get(ek); if (!ex) { exact(tau, G, ng, gsc); arrowCache.set(ek, (ex = gsc.slice())); }
        let lr = arrowCache.get(lk); if (!lr) { netScoreFn(run.snaps[snap])(tau, G, ng, gsc2); arrowCache.set(lk, (lr = gsc2.slice())); }
        dots(data.x, null, 2.5, 0.5, c.muted.hex);
        field(ex, c.muted.hex, 1, 0.55); field(lr, c.grad.hex, 1.5, 0.95);
        ctx.restore();
        const pw = wide ? W - sw - 44 : W - 60, px = wide ? sw + 24 : 30, py = wide ? 60 : sh + 24, ph = wide ? 150 : H - sh - 40;
        lossPlot(px, py, pw, ph, snap);
        phase = `snap${snap}`; extra = `${run.steps[snap]}|${lossAt(run, Math.min(run.steps[snap], TRAIN_STEPS - 1)).toFixed(3)}|${run.angles[tauSel][snap].toFixed(0)}|${tau}`;
        tl?.paint(t);
        return { tau, phase, extra };
      }

      if (m === 'both' || m === 'forward' || m === 'reverse') {
        const a = tauAt(t + MT.off); tau = a.tau; phase = a.phase;
        const rev = phase === 'reverse' || phase === 'clean', key = keyOf(), p = paths.get(key);
        let grp = gData;
        if (rev && p) { pathAt(p, tau, pts); grp = endGroup.get(key)!; } else if (rev) pts.set(xT); else forwardAt(tau, data.x, eps, pts);
        if (arrows) { const sf = sfNow(); sf(Math.max(tau, TAU_MIN), G, ng, gsc); field(gsc, c.grad.hex, 1.3, 0.7); }
        if (rev && p && phase === 'reverse' && m === 'reverse') {
          pathAt(p, Math.min(1, tau + 0.04), pts2);
          ctx.lineWidth = 1; ctx.globalAlpha = 0.4;
          for (let g = 0; g < 4; g++) { ctx.strokeStyle = grpCol(g); ctx.beginPath(); for (let i = 0; i < init.n; i++) if (grp[i] === g) { ctx.moveTo(X(pts[2 * i]), Y(pts[2 * i + 1])); ctx.lineTo(X(pts2[2 * i]), Y(pts2[2 * i + 1])); } ctx.stroke(); }
          ctx.globalAlpha = 1;
        }
        dots(pts, grp, 3);
        if (rev && p) { const q = quality(pts, mix); extra = `${Math.round(100 * q.onData)}|${Math.round(100 * q.upper)}|${Math.round(100 * q.lower)}`; }
        if (m === 'reverse' && !p && !needRun) dim = true;
      } else if (m === 'closed') {
        tau = sweepTau('closed', t); phase = t < 3.25 ? 'first' : t < 6.5 ? 'second' : 'noise';
        forwardAt(tau, data.x, eps, pts);
        dots(pts, null, 2.5, 0.35, c.muted.hex);
        const k = init.focus, a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau)), g = gData[k];
        const x0x = data.x[2 * k], x0y = data.x[2 * k + 1], e0 = eps[2 * k], e1 = eps[2 * k + 1];
        ctx.strokeStyle = grpCol(g); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(x0x), Y(x0y), 5, 0, 6.2832); ctx.stroke();
        txt('clean point', X(x0x) + 9, Y(x0y) - 6, c.muted.hex, 11, 'left', 500);
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(X(0) - 4, Y(0)); ctx.lineTo(X(0) + 4, Y(0)); ctx.moveTo(X(0), Y(0) - 4); ctx.lineTo(X(0), Y(0) + 4); ctx.stroke();
        arrow(X(0), Y(0), a * x0x * S, -a * x0y * S, c.s5.hex, 3, 7);
        arrow(X(a * x0x), Y(a * x0y), b * e0 * S, -b * e1 * S, c.fwd.hex, 3, 7);
        ctx.fillStyle = grpCol(g); ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(a * x0x + b * e0), Y(a * x0y + b * e1), 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        extra = `${a.toFixed(2)}|${b.toFixed(2)}|${x0x.toFixed(2)}|${x0y.toFixed(2)}|${e0.toFixed(2)}|${e1.toFixed(2)}|${(a * x0x + b * e0).toFixed(2)}|${(a * x0y + b * e1).toFixed(2)}`;
      } else if (m === 'field') {
        tau = sweepTau('field', t); phase = tau > 0.5 ? 'noisy' : tau > 0.15 ? 'mixed' : 'sharp';
        dots(data.x, null, 2.5, 0.55, c.muted.hex);
        const sf = sfNow(); sf(tau, G, ng, gsc); field(gsc, c.grad.hex, 1.4, 0.85);
        const q = cursor ?? PROBE, s = scoreAt(sf, tau, q[0], q[1]), mm = Math.hypot(s[0], s[1]), len = Math.min(70, 0.3 * Math.tanh(mm / 2.5) * S * 2.2);
        ctx.strokeStyle = c.text.hex; ctx.fillStyle = c.raised.hex; ctx.lineWidth = 2.2;
        if (mm > 1e-9) { ctx.beginPath(); ctx.moveTo(X(q[0]), Y(q[1])); ctx.lineTo(X(q[0]) + (s[0] / mm) * len, Y(q[1]) - (s[1] / mm) * len); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        extra = `${q[0].toFixed(2)}|${q[1].toFixed(2)}|${s[0].toFixed(2)}|${s[1].toFixed(2)}|${epsHat(tau, s[0]).toFixed(2)}|${epsHat(tau, s[1]).toFixed(2)}`;
      } else if (m === 'guide') {
        tau = sweepTau('guide', t); phase = t < 6 ? 'run' : 'done';
        const key = keyOf(), p = paths.get(key);
        dots(data.x, null, 2.5, 0.4, c.muted.hex);
        if (p) pathAt(p, tau, pts); else pts.set(xT);
        if (arrows) { const sf = guidedScoreFn(mix, lbl, wSel); sf(Math.max(tau, TAU_MIN), G, ng, gsc); field(gsc, c.grad.hex, 1.3, 0.7); }
        dots(pts, null, 3, 1, c.fwd.hex);
        if (p) { const q = moonStats(mix, pts, lbl); extra = `${Math.round(100 * q.onMoon)}|${Math.round(100 * q.onData)}|${q.spread.toFixed(1)}`; }
        txt(lbl === 'upper' ? 'label: upper moon' : 'label: lower moon', ox - half * S + 8, oy - half * S + 16, c.text.hex, 12);
      }
      if (dim) txt('sampling…', ox, oy, c.muted.hex, 13, 'center', 500);
      ctx.restore();

      const px = wide ? sw + 24 : 30, py = wide ? 60 : sh + 24, pw = wide ? W - sw - 44 : W - 60, ph = wide ? 130 : H - sh - 40;
      if (m !== 'guide' && m !== 'field') sharePlot(px, py, pw, ph, tau, 'signal and noise share');
      tl?.paint(t);
      return { tau, phase, extra };
    };

    const snr = (ab: number) => { const r = ab / Math.max(1e-9, 1 - ab); return r > 1000 ? '> 1000' : r.toFixed(2); };
    const readout = (f: { tau: number; phase: string; extra: string }): string => {
      const m = init.mode, ab = alphaBar(f.tau), e = f.extra.split('|');
      const base = `τ = ${f.tau.toFixed(2)} · ᾱ = ${ab.toFixed(3)} · signal-to-noise ${snr(ab)}`;
      if (m === 'closed') return `x_t = ${e[0]} · (${e[2]}, ${e[3]}) + ${e[1]} · (${e[4]}, ${e[5]}) = (${e[6]}, ${e[7]})   [τ = ${f.tau.toFixed(2)}]`;
      if (m === 'field') return `${base} · at (${e[0]}, ${e[1]}): score (${e[2]}, ${e[3]}), predicted noise ε̂ (${e[4]}, ${e[5]})`;
      if (m === 'train') return run ? `after ${e[0]} updates: loss ${e[1]}, learned arrows differ from the exact ones by ${e[2]}° on average (τ = ${e[3]})` : 'Training the network in your browser…';
      if (m === 'image') { const [d, pc] = e; return `${base} · the guess is the digit ${d} with weight ${pc}%`; }
      if (m === 'guide') return e.length > 2 ? `${base} · w = ${wSel} · on the chosen moon ${e[0]}% · spread along the moon ${e[2]} (out of 26 positions)` : base;
      return e.length > 2 ? `${base} · on the moons ${e[0]}% (upper ${e[1]}% / lower ${e[2]}%)` : base;
    };
    const IMG_TXT: Record<string, string> = {
      noising: 'Noising: each pixel gets random noise added, while the guess panel is the ideal denoiser’s best answer for what the clean image was.',
      static: 'Almost pure static: nothing is left of the digit. Generation starts from new static.',
      generating: 'Generating: the guess leans on every digit at first, then settles on one while the image itself still looks like static.',
      done: 'Done: new static has turned into a clean digit.',
    };
    const say = (f: { phase: string; tau: number; extra: string }): string => {
      const m = init.mode;
      if (m === 'closed') return NARR.closed!;
      if (m === 'field') return NARR.field!;
      if (m === 'guide') return NARR.guide!;
      if (m === 'image') return IMG_TXT[f.phase] ?? '';
      if (m === 'train') return run ? 'Each frame is a real snapshot of the network. Thin grey arrows are the exact denoising directions; pink arrows are what the network predicts.' : 'Training the network in your browser…';
      if (f.phase === 'clean' && init.denoiser === 'learned') { const q = f.extra.split('|')[0]; return `Finished: ${q}% of the points landed on a moon. The rest are strays, because this small network is less precise than the ideal denoiser of the first figure.`; }
      return TEXT[f.phase as Phase];
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: MT.dur, loopFrom: 0, loopTo: MT.dur }).t;
      const f = draw();
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        const k = f.phase + (run ? 'r' : '');
        if (k !== lastKey) { lastKey = k; narration = say(f); }
        uiRead = needRun && !run && init.mode === 'train' ? '' : readout(f);
      }
      return playing || (needRun && !run && !failed);
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    if (needRun) {
      load().then((r) => { run = r; if (r.paths) for (const k of ['ode', 'sde'] as const) if (r.paths[k]) setPath(`learned:${k}`, r.paths[k]!); lastKey = ''; wake(); }, () => { failed = true; wake(); });
    }
    const ro = new ResizeObserver(() => wake()); ro.observe(cv);
    const off = onThemeChange(() => { c = readColors(NAMES); font = readVar('--font-code'); ui = readVar('--font-ui'); wake(); });
    const mv = (e: PointerEvent) => { if (init.mode !== 'field') return; const r = cv.getBoundingClientRect(); cursor = [Math.max(-half, Math.min(half, (((e.clientX - r.left) / r.width) * W - ox) / S)), Math.max(-half, Math.min(half, -((((e.clientY - r.top) / r.height) * H) - oy) / S))]; wake(); };
    const lv = () => { cursor = null; wake(); };
    cv.addEventListener('pointermove', mv); cv.addEventListener('pointerleave', lv);
    document.fonts.ready.then(() => wake());
    return () => { loop.destroy(); ro.disconnect(); off(); cv.removeEventListener('pointermove', mv); cv.removeEventListener('pointerleave', lv); };
  });

  const TITLES: Record<Mode, string> = {
    both: 'Noise in, data out', forward: 'The forward process on two moons', reverse: 'The reverse process on two moons', closed: 'One jump to any noise level',
    field: 'Which way is the data?', train: 'Training a noise-prediction network', image: 'From noise to an image and back', guide: 'Guided sampling',
  };
  const LABELS: Record<Mode, string> = {
    both: 'Two moons of points. As the timeline runs the points dissolve into round Gaussian noise, then the reverse process gathers them back onto the moons.',
    forward: 'Two moons of points that dissolve into round Gaussian noise as the noise level rises.',
    reverse: 'A cloud of noise points that walk back onto two moons, leaving short trails.',
    closed: 'One highlighted data point and the two arrows, scaled clean point and scaled noise, that add up to its noisy version as the noise level rises.',
    field: 'A field of arrows over the plane showing the denoising direction at the current noise level; hover to read the predicted noise at one spot.',
    train: 'Arrows of a noise-prediction network at different points of training, drawn over the exact arrows, with the training loss curve.',
    image: 'Three 8 by 8 pixel images: a noisy digit, the best guess of the clean digit, and the noise; plus bars for how much the guess leans on each digit.',
    guide: 'Points sampled from the same noise, steered onto one chosen moon with a guidance strength.',
  };
  const LEGEND = {
    moons: [{ label: 'ends on upper moon', color: '--viz-cat-1' }, { label: 'ends on lower moon', color: '--viz-cat-2' }],
    arrow: { label: 'denoising direction', color: '--viz-gradient', shape: 'line' as const },
    shares: [{ label: 'data share', color: '--viz-seq-5', shape: 'line' as const }, { label: 'noise share', color: '--viz-forward', shape: 'line' as const }],
  };
  const arrowLegend = untrack(() => (showScore || showControls) && mode !== 'forward');
  const legend = untrack(() => {
    const md = init.mode;
    if (md === 'closed') return [{ label: 'clean part √ᾱ·x₀', color: '--viz-seq-5', shape: 'line' as const }, { label: 'noise part √(1−ᾱ)·ε', color: '--viz-forward', shape: 'line' as const }, { label: 'other points', color: '--text-muted' }];
    if (md === 'train') return [{ label: 'network', color: '--viz-gradient', shape: 'line' as const }, { label: 'exact', color: '--text-muted', shape: 'line' as const }, { label: 'loss', color: '--viz-forward', shape: 'line' as const }];
    if (md === 'image') return [{ label: 'signed noise', color: '--viz-pos' }, { label: 'weight on a digit', color: '--viz-seq-4' }];
    if (md === 'guide') return [{ label: 'samples', color: '--viz-forward' }, { label: 'training data', color: '--text-muted' }, ...(arrowLegend ? [LEGEND.arrow] : [])];
    if (md === 'field') return [{ label: 'training data', color: '--text-muted' }, LEGEND.arrow];
    return [...(init.dataset === 'moons' ? (md === 'forward' ? [{ label: 'from upper moon', color: '--viz-cat-1' }, { label: 'from lower moon', color: '--viz-cat-2' }] : LEGEND.moons) : [{ label: 'position along the curve', color: '--viz-seq-4' }]), ...(arrowLegend ? [LEGEND.arrow] : []), ...LEGEND.shares];
  });
  const showSampler = untrack(() => init.mode === 'both' || init.mode === 'reverse');
  const showArrows = untrack(() => !['train', 'image', 'closed', 'field', 'forward'].includes(init.mode));
</script>

<FigureCard title={title || TITLES[init.mode]} aspect="720 / 420" narrowAspect={init.mode === 'image' ? '360 / 330' : '360 / 520'} label={LABELS[init.mode]} {legend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={MT.dur} markers={MT.markers} label="Diffusion time"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= MT.dur ? 0 : to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next num">{uiRead}{init.mode === 'field' ? ' · hover the plot to move the probe' : ''}</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls && (showSampler || showArrows || init.mode === 'train' || init.mode === 'guide')}
      <div class="ctl">
        {#if showSampler}
          <fieldset class="seg"><legend class="sr-only">Reverse sampler</legend>
            <label><input type="radio" name="s-{uid}" value="ode" bind:group={smp} /><span>Deterministic (ODE)</span></label>
            <label><input type="radio" name="s-{uid}" value="sde" bind:group={smp} /><span>Stochastic (SDE)</span></label>
          </fieldset>
        {/if}
        {#if init.mode === 'train'}
          <fieldset class="seg"><legend class="sr-only">Noise level of the arrows</legend>
            <label><input type="radio" name="n-{uid}" value={0} bind:group={tauSel} /><span>τ = 0.3 (noisy)</span></label>
            <label><input type="radio" name="n-{uid}" value={1} bind:group={tauSel} /><span>τ = 0.1 (nearly clean)</span></label>
          </fieldset>
        {/if}
        {#if init.mode === 'guide'}
          <fieldset class="seg"><legend class="sr-only">Label</legend>
            <label><input type="radio" name="l-{uid}" value="upper" bind:group={lbl} onchange={() => { t = 0; wake(); }} /><span>upper moon</span></label>
            <label><input type="radio" name="l-{uid}" value="lower" bind:group={lbl} onchange={() => { t = 0; wake(); }} /><span>lower moon</span></label>
          </fieldset>
          <fieldset class="seg"><legend class="sr-only">Guidance strength w</legend>
            {#each [0, 1, 3, 7] as w (w)}
              <label><input type="radio" name="w-{uid}" value={w} bind:group={wSel} onchange={() => { t = 0; wake(); }} /><span>w = {w}</span></label>
            {/each}
          </fieldset>
        {/if}
        {#if showArrows}<label><input type="checkbox" bind:checked={arrows} /> denoising arrows</label>{/if}
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
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  input[type='checkbox'] { accent-color: var(--accent); }
</style>
