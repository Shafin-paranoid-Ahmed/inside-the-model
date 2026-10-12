<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { Run, ModelId } from './sim';
  const runs: Partial<Record<ModelId, Promise<Run>>> = {};
  const progress: Partial<Record<ModelId, number>> = {};
  /** Train each model once per page in its own worker; every figure on the page shares the result. */
  function load(id: ModelId): Promise<Run> {
    return (runs[id] ??= new Promise((res, rej) => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.kind === 'progress') progress[id] = e.data.frac;
        else { progress[id] = 1; res(e.data.run); w.terminate(); }
      };
      w.onerror = () => rej(new Error('worker failed'));
      w.postMessage({ id });
    }));
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import {
    DOMAIN, DATA, N, SPECS, FOCUS_DEFAULT, WIDE, HIST_BINS, SAMPLES, VAE_BETAS, BELL_K, BETA_FADE, BETA_SEG, specFor, buildScene, timing, restAt,
    roundTrip, trainFrame, exploreFrame, sampleFrame, denoiseFrame, encodeFrame, ease, reparamFrame, betaFrame, priorPdf, DENOISE_POINTS,
    type Mode, type Kind, type Scene,
  } from './sim';

  interface Props {
    /** Which trained model: a plain autoencoder, a denoising one, or a variational one. */
    model?: Kind;
    /** 'tour' every point through the code and back; 'focus' one point; 'train' the training snapshots; 'explore' sweep the code; 'sample' random codes; 'denoise' noisy points to clean ones; VAE only: 'encode' one point's bell curve, 'reparam' z = mu + sigma eps, 'beta' the four KL weights. */
    mode?: Mode;
    title?: string;
    /** Index (0 to 599) of the training point 'focus' follows. */
    focus?: number;
    /** Numbers in the code: 1 or 2. */
    latentDim?: 1 | 2;
    /** VAE weight on the KL term (0.03, 0.1, 1 or 4). Unused by the autoencoder figures and by mode 'beta' (which shows all four). */
    beta?: number;
    /** 'tour' also sends three points that are not on either moon through the network. */
    probes?: boolean;
    /** 'sample' with an autoencoder: draw the codes from the standard normal and show that curve on the code line. */
    showPrior?: boolean;
  }
  let { model = 'ae', mode = 'tour', title = '', focus = FOCUS_DEFAULT, latentDim = 1, beta = 0.1, probes = false, showPrior = false }: Props = $props();

  const init = untrack(() => {
    const spec = specFor(model, latentDim, beta);
    return {
      mode, spec, focus: Math.min(N - 1, Math.max(0, focus)), probes, showPrior,
      title: title || { tour: 'Every point through the code and back', focus: 'One point, step by step', train: 'The decoder while the network trains', explore: 'Sliding along the code line', sample: 'Decoding random codes', denoise: 'Noisy points and their cleaned-up versions', encode: 'One point becomes a bell curve', reparam: 'Moving the randomness into \u03b5', beta: 'The same network at four KL weights' }[mode],
    };
  });
  const spec = init.spec, L = spec.latent, prior = spec.vae || init.showPrior;
  const ids = init.mode === 'beta' ? VAE_BETAS.map((b) => SPECS[`vae${b}` as keyof typeof SPECS].id) : [spec.id];

  let sc = $state.raw<Scene | null>(null), scs = $state.raw<Scene[]>([]);
  let failed = $state(false);
  const tm = $derived(timing(init.mode, sc ?? undefined));

  let narrow = $state(false);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let narration = $state(''), sliderZ = $state(0);

  // ------------------------------------------------ layout (canvas CSS pixels; wide 720, narrow 400)
  interface R { x: number; y: number; w: number; h: number }
  const R4 = (x: number, y: number, w: number, h: number): R => ({ x, y, w, h });
  type Lay = { W: number; H: number; P: R; S?: R; X?: R; E?: R };
  function layout(m: Mode, nrw: boolean): Lay {
    const sh = m === 'explore' || m === 'encode' ? 190 : m === 'sample' ? 170 : 150;
    if (!nrw) {
      const W = 720, H = 360, P = R4(12, 15, 330, 330);
      if (m === 'denoise') return { W, H, P: R4(195, 15, 330, 330) };
      if (m === 'train') return spec.vae ? { W, H, P, S: R4(372, 15, 336, 110), X: R4(372, 135, 336, 210) } : { W, H, P, X: R4(372, 15, 336, 330) };
      if (L === 2) return { W, H, P, S: R4(378, 15, 330, 330) };
      if (m === 'focus') return { W, H, P, S: R4(372, 25, 336, 120), X: R4(372, 170, 336, 170) };
      if (m === 'reparam') return { W, H, P, E: R4(372, 15, 336, 80), S: R4(372, 105, 336, 130), X: R4(372, 245, 336, 100) };
      if (m === 'beta') return { W, H, P, S: R4(372, 15, 336, 165), X: R4(372, 190, 336, 155) };
      return { W, H, P, S: R4(372, (H - sh) / 2, 336, sh) };
    }
    const W = 400, P = R4(12, 10, 376, 376), y0 = 396;
    if (m === 'denoise') return { W, H: 400, P };
    if (m === 'train') return spec.vae ? { W, H: y0 + 434, P, S: R4(12, y0 + 4, 376, 110), X: R4(12, y0 + 124, 376, 300) } : { W, H: y0 + 304, P, X: R4(12, y0 + 4, 376, 290) };
    if (L === 2) return { W, H: y0 + 386, P, S: R4(12, y0, 376, 376) };
    if (m === 'focus') return { W, H: y0 + 356, P, S: R4(12, y0 + 4, 376, 120), X: R4(12, y0 + 140, 376, 206) };
    if (m === 'reparam') return { W, H: y0 + 354, P, E: R4(12, y0 + 4, 376, 80), S: R4(12, y0 + 94, 376, 130), X: R4(12, y0 + 234, 376, 110) };
    if (m === 'beta') return { W, H: y0 + 344, P, S: R4(12, y0 + 4, 376, 150), X: R4(12, y0 + 164, 376, 170) };
    return { W, H: y0 + sh + 18, P, S: R4(12, y0 + 4, 376, sh) };
  }
  const lw = layout(init.mode, false), ln = layout(init.mode, true);
  const lay = $derived(narrow ? ln : lw);
  const Wd = $derived(lay.W), Hh = $derived(lay.H);

  const legend = (() => {
    const f = { label: 'rebuilt points', color: '--viz-forward' }, d = { label: 'training points', color: '--text-muted' };
    const pl = { label: 'N(0, 1) prior', color: '--text', shape: 'line' as const }, off = { label: '\u00d7 off the data', color: '--accent' };
    switch (init.mode) {
      case 'tour': return [{ label: 'input points', color: '--text-muted' }, f, ...(init.probes ? [{ label: 'probe points', color: '--accent' }] : [])];
      case 'focus': return [{ label: 'the other points', color: '--text-muted' }, { label: 'the followed point', color: '--accent' }, f];
      case 'train': return spec.vae
        ? [d, { label: 'what the decoder draws', color: '--viz-forward', shape: 'line' as const }, { label: 'upper moon codes', color: '--viz-cat-1' }, { label: 'lower moon codes', color: '--viz-cat-2' }, pl]
        : [d, { label: 'what the decoder draws', color: '--viz-forward', shape: 'line' as const }];
      case 'explore': return [{ label: 'upper moon', color: '--viz-cat-1' }, { label: 'lower moon', color: '--viz-cat-2' }, { label: 'decoded point', color: '--viz-forward' }, off, ...(prior ? [pl] : [])];
      case 'sample': return [d, { label: 'decoded, on the data', color: '--viz-forward' }, { label: 'decoded, off the data (\u00d7)', color: '--accent' }, ...(prior ? [pl] : [])];
      case 'encode': return [{ label: 'the other points (mean codes)', color: '--text-muted' }, { label: 'the followed point and its bell curve', color: '--accent' }, { label: 'codes drawn from the bell, decoded', color: '--viz-forward' }];
      case 'reparam': return [{ label: 'the followed point', color: '--accent' }, { label: 'decoded', color: '--viz-forward' }, { label: 'gradient path', color: '--viz-gradient', shape: 'line' as const }];
      case 'beta': return [d, { label: 'decoded prior samples', color: '--viz-forward' }, off, { label: 'upper moon codes', color: '--viz-cat-1' }, { label: 'lower moon codes', color: '--viz-cat-2' }, pl];
      default: return [{ label: 'noisy input', color: '--text-muted' }, { label: 'cleaned up', color: '--viz-forward' }, { label: 'move', color: '--viz-gradient', shape: 'line' as const }];
    }
  })();

  // ------------------------------------------------ small helpers
  const mn = (v: number, d = 2) => v.toFixed(d).replace('-', '−');
  const err = (v: number) => (v < 0.001 ? v.toPrecision(2) : v.toFixed(3));
  const pair = (a: ArrayLike<number>, i: number) => `(${mn(a[2 * i])}, ${mn(a[2 * i + 1])})`;
  /** z from the mean, spread and epsilon as the readout prints them (3, 3 and 2 decimals), so the printed sum adds up. */
  const zShown = (f: { mu: number; sigma: number; eps: number }) => +f.mu.toFixed(3) + +f.sigma.toFixed(3) * +f.eps.toFixed(2);
  let cur: any = null; // the frame being drawn; the narration reads the same one
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, zOver: number | null = null, poke = () => {};
  $effect(() => { playing; speed; poke(); });
  $effect(() => { sc; poke(); });
  $effect(() => { if (playing) zOver = null; });

  function say(): string {
    if (!sc) return failed ? 'Training failed in this browser.' : 'Training the network in your browser…';
    const f = cur;
    if (!f) return '';
    switch (init.mode) {
      case 'tour': {
        if (f.phase === 0) return 'Every training point starts as two numbers: its position in the plane.';
        if (f.phase === 1) return L === 1 ? 'The encoder squeezes each point into one number.' : 'The encoder turns each point into two numbers.';
        if (f.phase === 2) return L === 1 ? `Each point is now a single number on the code line, between ${mn(sc.zlo, 1)} and ${mn(sc.zhi, 1)}.` : 'Each point is now a code of two numbers, as many as it started with.';
        if (f.phase === 3) return 'The decoder rebuilds a point in the plane from each code.';
        const pr = init.probes ? ` The probe at (1.4, 0.8) comes back at ${pair(sc.R, N + 1)}.` : '';
        return `Rebuilt. On average a point lands a squared distance of ${err(sc.mse)} from where it started.${pr}`;
      }
      case 'focus': {
        const i = init.focus;
        if (f.phase === 0) return `The input is x = ${pair(sc.X, i)}.`;
        if (f.phase === 1) return 'The encoder turns x into one number.';
        if (f.phase === 2) return `The code is z = ${mn(sc.Z[i])}.`;
        if (f.phase === 3) return 'The decoder turns z back into two numbers.';
        const e = (sc.R[2 * i] - sc.X[2 * i]) ** 2 + (sc.R[2 * i + 1] - sc.X[2 * i + 1]) ** 2;
        return `The output is x̂ = ${pair(sc.R, i)}. Squared error ${err(e)}.`;
      }
      case 'train':
        if (spec.vae) return `Step ${f.step}: reconstruction ${err(f.rec)}, KL ${f.kl.toFixed(2)}.${f.k === 0 ? ' Nothing is learned yet.' : ''}`;
        return `Step ${f.step}: average squared error ${err(f.loss)}.${f.k === 0 ? ' Nothing is learned yet.' : ''}`;
      case 'explore':
        return `z = ${mn(f.z)} decodes to (${mn(f.x)}, ${mn(f.y)}): ${f.near ? `a point on the ${f.moon ? 'lower' : 'upper'} moon.` : 'between the moons, away from the data.'}`;
      case 'sample':
        if (sc.prior) return `${f.whole} codes drawn from the standard normal: ${f.on} decode onto the data, ${f.off} do not.`;
        return `${f.whole} codes drawn ${f.batch ? `from −${WIDE} to ${WIDE}` : `from the training range`}: ${f.on} decode onto the data, ${f.off} do not.`;
      case 'encode': {
        const b = sc.bell!;
        if (f.phase === 0) return `The encoder turns x = ${pair(sc.X, b.focus)} into a bell curve: mean \u03bc = ${mn(b.mu)}, spread \u03c3 = ${b.sigma.toFixed(2)}.`;
        if (f.phase === 1) return `${f.whole} of ${BELL_K} codes drawn from the bell curve so far, each decoded to a point.`;
        return `All ${BELL_K} draws decode to points ${b.spread.toFixed(2)} away from the rebuilt point on average.`;
      }
      case 'reparam':
        return `\u03b5 = ${mn(f.eps)}: z = \u03bc + \u03c3\u03b5 = ${mn(f.mu, 3)} + ${f.sigma.toFixed(3)} \u00d7 ${mn(f.eps)} = ${mn(zShown(f), 3)}, which decodes to (${mn(f.x)}, ${mn(f.y)}).`;
      case 'beta': {
        const m = scs[f.k];
        return `\u03b2 = ${VAE_BETAS[f.k]}: reconstruction error ${err(m.mse)}, KL ${m.kl.toFixed(2)}, ${m.samples[0].cum[SAMPLES]} of ${SAMPLES} prior samples decode onto the data.`;
      }
      default:
        return `Noise level ${f.sigma.toFixed(2)}: the average distance to the data is ${f.before.toFixed(2)} before cleaning and ${f.after.toFixed(2)} after.`;
    }
  }

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const C = {
      neg: '--viz-neg', fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule',
      raised: '--bg-raised', sunken: '--bg-sunken', accent: '--accent', c1: '--viz-cat-1', c2: '--viz-cat-2',
    };
    let c = readColors(C), code = readVar('--font-code'), ui = readVar('--font-body');
    let lastUi = 0;

    const dot = (x: number, y: number, r: number, col: string, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); };
    const ring = (x: number, y: number, r: number, col: string, w = 1.6, a = 1) => { ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.stroke(); };
    const line = (x1: number, y1: number, x2: number, y2: number, col: string, w = 1, a = 1, dash: number[] = []) => {
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    };
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'center', w = 500, a = 1, ital = false) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      const f = narrow ? 1.12 : 1; ctx.font = ital ? `italic ${px * f}px ${ui}` : `${w} ${px * f}px ${code}`; ctx.fillText(s, x, y);
    };
    const cross = (x: number, y: number, r: number, col: string, w = 2, a = 1) => { line(x - r, y - r, x + r, y + r, col, w, a); line(x - r, y + r, x + r, y - r, col, w, a); };
    const uitxt = (s: string, x: number, y: number, col: string, px = 14, align: CanvasTextAlign = 'left', w = 500, a = 1) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align; ctx.font = `${w} ${px * (narrow ? 1.12 : 1)}px ${ui}`; ctx.fillText(s, x, y);
    };
    const box = (r: R, fill = c.sunken.hex) => { ctx.globalAlpha = 1; ctx.fillStyle = fill; ctx.fillRect(r.x, r.y, r.w, r.h); ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); };

    const clipPlane = () => { const P = lay.P; ctx.save(); ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip(); };
    // data plane [-DOMAIN, DOMAIN]^2
    const px = (x: number) => lay.P.x + ((x + DOMAIN) / (2 * DOMAIN)) * lay.P.w;
    const py = (y: number) => lay.P.y + ((DOMAIN - y) / (2 * DOMAIN)) * lay.P.h;
    function drawPlane(name = 'data') {
      const P = lay.P;
      box(P);
      line(px(0), P.y, px(0), P.y + P.h, c.rule.hex, 1, 0.8); line(P.x, py(0), P.x + P.w, py(0), c.rule.hex, 1, 0.8);
      for (const v of [-2, 2]) txt(mn(v, 0), px(v), P.y + P.h - 5, c.muted.hex, 12, 'center', 400, 0.8);
      for (const v of [-1, 1]) txt(mn(v, 0), P.x + 4, py(v) + 4, c.muted.hex, 12, 'left', 400, 0.8);
      txt(name, P.x + 8, P.y + 15, c.muted.hex, 12, 'left', 400, 1, true);
    }
    // code line (1 number) or code plane (2 numbers)
    const S = () => lay.S!;
    const view = (s: Scene) => (init.mode === 'sample' && !s.prior ? [-(WIDE + 0.3), WIDE + 0.3] : [s.vlo, s.vhi]);
    const sx = (s: Scene, z: number) => { const [a, b] = view(s), r = S(); return r.x + 14 + ((z - a) / (b - a)) * (r.w - 28); };
    const sBase = () => S().y + S().h - 30;
    const sy = (j: number) => { const top = S().y + 24, b = sBase() - 8; return top + ((j + 1) / 2) * (b - top); };
    const lx = (s: Scene, v: number) => S().x + S().w / 2 + (v / (s.zm * 1.15)) * (S().w / 2);
    const ly = (s: Scene, v: number) => S().y + S().h / 2 - (v / (s.zm * 1.15)) * (S().h / 2);
    function drawCodeAxis(s: Scene) {
      const r = S();
      box(r);
      txt(L === 1 ? (init.mode === 'sample' ? (s.prior ? 'code line (bars: where the 600 training codes sit)' : 'code line (shaded: the training codes)') : 'code line (one number per point)') : 'code plane (two numbers per point)', r.x + 8, r.y + 15, c.muted.hex, 12, 'left', 400, 1, true);
      if (L === 2) {
        line(r.x + r.w / 2, r.y, r.x + r.w / 2, r.y + r.h, c.rule.hex, 1, 0.8); line(r.x, r.y + r.h / 2, r.x + r.w, r.y + r.h / 2, c.rule.hex, 1, 0.8);
        return;
      }
      const [a, b] = view(s), base = sBase();
      line(r.x + 8, base, r.x + r.w - 8, base, c.muted.hex, 1.2, 0.8);
      const step = b - a > 6 ? 2 : 1;
      for (let v = Math.ceil(a / step) * step; v <= b; v += step) { line(sx(s, v), base, sx(s, v), base + 4, c.muted.hex, 1, 0.8); txt(mn(v, 0), sx(s, v), base + 17, c.muted.hex, 12, 'center', 400); }
    }
    const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
    /** Code histogram by moon (stacked) on the code line; the prior's curve over it when the figure has one. */
    const drawHist = (s: Scene, alpha = 0.85, h: { up: Float64Array; lo: Float64Array } = s.hist) => {
      const r = S(), base = sBase(), hmax = base - r.y - 30, bw = (s.hist.b - s.hist.a) / HIST_BINS;
      for (let b = 0; b < HIST_BINS; b++) {
        const x0 = sx(s, s.hist.a + b * bw), x1 = sx(s, s.hist.a + (b + 1) * bw), hu = Math.min(1, h.up[b] / s.hist.top) * hmax, hl = Math.min(1 - Math.min(1, h.up[b] / s.hist.top), h.lo[b] / s.hist.top) * hmax;
        ctx.globalAlpha = alpha; ctx.fillStyle = c.c1.hex; ctx.fillRect(x0 + 0.5, base - hu, x1 - x0 - 1, hu);
        ctx.fillStyle = c.c2.hex; ctx.fillRect(x0 + 0.5, base - hu - hl, x1 - x0 - 1, hl);
      }
    };
    const drawPrior = (s: Scene, alpha = 1) => {
      const r = S(), base = sBase(), hmax = base - r.y - 30;
      ctx.globalAlpha = alpha; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let k = 0; k <= 80; k++) { const z = s.vlo + ((s.vhi - s.vlo) * k) / 80, x = sx(s, z), y = base - (priorPdf(z) / s.hist.top) * hmax; (k ? ctx.lineTo : ctx.moveTo).call(ctx, x, y); }
      ctx.stroke();
      txt('N(0, 1)', sx(s, 1.0) + 6, base - (priorPdf(0.9) / s.hist.top) * hmax - 6, c.text.hex, 12, 'left', 600, alpha);
    };
    /** The bell curve of a point with mean mu and spread sigma on the code line. */
    const drawBell = (s: Scene, mu: number, sg: number, alpha: number) => {
      const base = sBase(), peak = 0.72 * (base - S().y - 30);
      ctx.globalAlpha = 0.22 * alpha; ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(sx(s, mu - 3.4 * sg), base);
      for (let k = 0; k <= 60; k++) { const z = mu + sg * (-3.4 + (6.8 * k) / 60); ctx.lineTo(sx(s, z), base - Math.exp(-0.5 * ((z - mu) / sg) ** 2) * peak); }
      ctx.lineTo(sx(s, mu + 3.4 * sg), base); ctx.fill();
      ctx.globalAlpha = alpha; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let k = 0; k <= 60; k++) { const z = mu + sg * (-3.4 + (6.8 * k) / 60); (k ? ctx.lineTo : ctx.moveTo).call(ctx, sx(s, z), base - Math.exp(-0.5 * ((z - mu) / sg) ** 2) * peak); }
      ctx.stroke();
      return { peak, base };
    };
    const PRIOR_V = 3.3;
    const arrow = (x1: number, y1: number, x2: number, y2: number, col: string, w = 1.5, a = 1, dash: number[] = []) => {
      line(x1, y1, x2, y2, col, w, a, dash);
      const th = Math.atan2(y2 - y1, x2 - x1);
      line(x2, y2, x2 - 7 * Math.cos(th - 0.45), y2 - 7 * Math.sin(th - 0.45), col, w, a); line(x2, y2, x2 - 7 * Math.cos(th + 0.45), y2 - 7 * Math.sin(th + 0.45), col, w, a);
    };
    const clampX = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

    function drawLoading() {
      const p = ids.reduce((a, id) => a + (progress[id] ?? 0), 0) / ids.length, w = Math.min(280, Wd - 80), x = (Wd - w) / 2, y = Hh / 2;
      txt(failed ? 'Training failed in this browser.' : `Training the network in your browser: ${Math.round(100 * p)}%`, Wd / 2, y - 14, c.muted.hex, 13, 'center', 500);
      ctx.globalAlpha = 1; ctx.fillStyle = c.sunken.hex; ctx.fillRect(x, y, w, 8);
      ctx.fillStyle = c.fwd.hex; ctx.fillRect(x, y, w * p, 8);
    }

    // ------------------------------------------------ tour and focus: every point (or one) through the network
    function drawTrip(s: Scene, one: boolean) {
      const f = roundTrip(t, init.mode as 'tour' | 'focus'); cur = f;
      drawPlane('input space'); drawCodeAxis(s);
      const m = f.mix, fi = init.focus, r = narrow ? 1.9 : 2.3;
      const at = (i: number): [number, number] => {
        const a: [number, number] = [px(s.X[2 * i]), py(s.X[2 * i + 1])];
        const z: [number, number] = L === 1 ? [sx(s, s.Z[i]), sy(s.jit[i])] : [lx(s, s.Z[2 * i]), ly(s, s.Z[2 * i + 1])];
        const b: [number, number] = [px(s.R[2 * i]), py(s.R[2 * i + 1])];
        return m <= 1 ? [lerp(a[0], z[0], m), lerp(a[1], z[1], m)] : [lerp(z[0], b[0], m - 1), lerp(z[1], b[1], m - 1)];
      };
      for (let i = 0; i < s.n; i++) {
        if (one) {
          // the other points rest in the plane and on the code line, dimmed
          if (i === fi) continue;
          dot(px(s.X[2 * i]), py(s.X[2 * i + 1]), r, c.muted.hex, 0.22);
          dot(L === 1 ? sx(s, s.Z[i]) : lx(s, s.Z[2 * i]), L === 1 ? sy(s.jit[i]) : ly(s, s.Z[2 * i + 1]), r, c.muted.hex, 0.22);
          continue;
        }
        const probe = i >= N;
        if (m > 0 && !probe) dot(px(s.X[2 * i]), py(s.X[2 * i + 1]), r, c.muted.hex, 0.2);
        if (m > 1) dot(L === 1 ? sx(s, s.Z[i]) : lx(s, s.Z[2 * i]), L === 1 ? sy(s.jit[i]) : ly(s, s.Z[2 * i + 1]), r, probe ? c.accent.hex : c.muted.hex, probe ? 0.5 : 0.3);
        const [x, y] = at(i);
        if (probe) { dot(x, y, 4.2, c.accent.hex, 0.95); if (m === 0) txt('probe', x + 7, y - 6, c.accent.hex, 12, 'left', 600); if (m > 0) ring(px(s.X[2 * i]), py(s.X[2 * i + 1]), 4.2, c.accent.hex, 1.2, 0.6); }
        else dot(x, y, r, m > 1 ? c.fwd.hex : c.muted.hex, 0.78);
      }
      if (one) {
        const [x, y] = at(fi), ox = px(s.X[2 * fi]), oy = py(s.X[2 * fi + 1]);
        ring(ox, oy, 6, c.accent.hex, 1.6);
        if (m >= 1) {
          const zx = L === 1 ? sx(s, s.Z[fi]) : lx(s, s.Z[2 * fi]);
          if (L === 1) { line(zx, sBase(), zx, S().y + 42, c.accent.hex, 1.4, 0.9); txt(`z = ${mn(s.Z[fi])}`, Math.min(S().x + S().w - 50, Math.max(S().x + 50, zx)), S().y + 38, c.accent.hex, 12, 'center', 700); }
        }
        if (m >= 2) {
          const rx = px(s.R[2 * fi]), ry = py(s.R[2 * fi + 1]);
          line(ox, oy, rx, ry, c.neg.hex, 1.6, 0.9, [4, 3]); txt('error', (ox + rx) / 2 + 8, (oy + ry) / 2 + 4, c.neg.hex, 12, 'left', 600);
        }
        dot(x, y, 5.5, m > 1 ? c.fwd.hex : c.accent.hex, 1);
        drawDiagram(s, f);
      }
    }

    // the small encoder, code, decoder picture beside the followed point
    function drawDiagram(s: Scene, f: { phase: number; mix: number }) {
      const r = lay.X!, y = r.y + 40, nodes: [string, number, boolean][] = [['x', 44, false], ['encoder', 84, true], ['z', 36, false], ['decoder', 84, true], ['x̂', 44, false]];
      const gap = (r.w - 8 - nodes.reduce((a, n) => a + n[1], 0)) / 4;
      let x = r.x + 4;
      const xs: [number, number][] = [];
      nodes.forEach(([name, w, net], k) => {
        const on = (k === 0 && f.phase === 0) || (k === 1 && f.phase === 1) || (k === 2 && f.phase === 2) || (k === 3 && f.phase === 3) || (k === 4 && f.phase === 4);
        ctx.globalAlpha = 1; ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.roundRect(x, y - 16, w, 32, 5); ctx.fill();
        ctx.strokeStyle = on ? c.accent.hex : c.rule.hex; ctx.lineWidth = on ? 2.2 : 1.2; ctx.stroke();
        if (net) txt(name, x + w / 2, y + 4, on ? c.text.hex : c.muted.hex, 12, 'center', on ? 700 : 500);
        else uitxt(name, x + w / 2, y + 5, on ? c.text.hex : c.muted.hex, 16, 'center', on ? 700 : 500);
        xs.push([x, x + w]); x += w + gap;
      });
      for (let k = 0; k < 4; k++) {
        const a = xs[k][1] + 1, b = xs[k + 1][0] - 1, lit = f.mix > k / 2 && f.mix < (k + 1) / 2 + 0.0001 && f.phase % 2 === 1;
        line(a, y, b, y, lit ? c.accent.hex : c.muted.hex, 1.5, 0.7);
        line(b - 5, y - 3.5, b, y, c.muted.hex, 1.2, 0.7); line(b - 5, y + 3.5, b, y, c.muted.hex, 1.2, 0.7);
        if (f.phase % 2 === 1 && Math.floor(f.mix * 2) === k) dot(lerp(a, b, f.mix * 2 - k), y, 4.5, c.fwd.hex);
      }
      txt(`2 → 32 → 32 → ${L}`, (xs[1][0] + xs[1][1]) / 2, y + 30, c.muted.hex, 12, 'center', 400);
      txt(`${L} → 32 → 32 → 2`, (xs[3][0] + xs[3][1]) / 2, y + 30, c.muted.hex, 12, 'center', 400);
      const rows: [string, string, number][] = [['x', pair(s.X, init.focus), 0], ['z', mn(s.Z[init.focus]), 2], ['x̂', pair(s.R, init.focus), 4]];
      rows.forEach(([name, val, need], k) => { const a = f.phase >= need ? 1 : 0.12; uitxt(name, r.x + 10, y + 70 + k * 24, c.text.hex, 17, 'left', 600, a); txt(`= ${val}`, r.x + 30, y + 70 + k * 24, c.text.hex, 14, 'left', 600, a); });
      const e = (s.R[2 * init.focus] - s.X[2 * init.focus]) ** 2 + (s.R[2 * init.focus + 1] - s.X[2 * init.focus + 1]) ** 2;
      txt(`squared error ${err(e)}`, r.x + 10, y + 142, c.neg.hex, 13, 'left', 600, f.phase >= 4 ? 1 : 0.12);
    }

    // ------------------------------------------------ training snapshots and the loss
    function drawTrain(s: Scene) {
      const f = trainFrame(t, s); cur = f;
      drawPlane('data space');
      for (let i = 0; i < N; i++) dot(px(DATA.x[2 * i]), py(DATA.x[2 * i + 1]), narrow ? 1.8 : 2.2, c.muted.hex, 0.5);
      clipPlane(); const cu = s.curves[f.k]; ctx.globalAlpha = 1; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let i = 0; i < cu.length / 2; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, px(cu[2 * i]), py(cu[2 * i + 1]));
      ctx.stroke(); ctx.restore();
      if (spec.vae) { drawTrainVae(s, f); return; }
      // loss chart, log scale
      const r = lay.X!, ls = s.run.snapLoss, st = s.run.steps, last = st[st.length - 1];
      const l0 = Math.min(...ls) * 0.7, l1 = Math.max(...ls) * 1.4, lg = (v: number) => Math.log10(v);
      const pl = { x: r.x + 56, y: r.y + 30, w: r.w - 72, h: r.h - 78 };
      const X = (k: number) => pl.x + (st[k] / last) * pl.w, Y = (v: number) => pl.y + pl.h * (1 - (lg(v) - lg(l0)) / (lg(l1) - lg(l0)));
      box(r);
      txt('average squared error (log scale)', r.x + 8, r.y + 17, c.muted.hex, 12, 'left', 400, 1, true);
      for (let e = Math.ceil(lg(l0)); e <= Math.floor(lg(l1)); e++) { const v = 10 ** e; line(pl.x, Y(v), pl.x + pl.w, Y(v), c.rule.hex, 1, 0.9); txt(String(+v.toPrecision(1)), pl.x - 6, Y(v) + 4, c.muted.hex, 12, 'right', 400); }
      line(pl.x, pl.y + pl.h, pl.x + pl.w, pl.y + pl.h, c.muted.hex, 1.2, 0.8);
      for (let q = 0; q <= 3; q++) { const k = Math.round((q / 3) * (st.length - 1)); txt(String(st[k]), X(k), pl.y + pl.h + 16, c.muted.hex, 12, q === 0 ? 'left' : q === 3 ? 'right' : 'center', 400); }
      txt('training steps', pl.x + pl.w / 2, pl.y + pl.h + 34, c.muted.hex, 12, 'center', 400, 1, true);
      ctx.globalAlpha = 0.25; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 2; ctx.beginPath();
      ls.forEach((v, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(v))); ctx.stroke();
      ctx.globalAlpha = 1; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.4; ctx.beginPath();
      for (let k = 0; k <= f.k; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(ls[k]));
      ctx.stroke();
      line(X(f.k), pl.y, X(f.k), pl.y + pl.h, c.accent.hex, 1.2, 0.8, [3, 3]);
      dot(X(f.k), Y(f.loss), 5, c.accent.hex);
      txt(err(f.loss), Math.min(X(f.k) + 9, pl.x + pl.w), Y(f.loss) - 10, c.text.hex, 12, f.k > st.length * 0.8 ? 'right' : 'left', 700);
    }

    /** Two stacked charts over the training steps: reconstruction (log scale) and KL (linear), one shared cursor. */
    function drawTrainVae(s: Scene, f: { k: number; rec: number; kl: number }) {
      drawCodeAxis(s);
      drawHist(s, 0.85, s.snapHist[f.k]); drawPrior(s);
      const R0 = lay.X!, st = s.run.steps, last = st[st.length - 1], nk = st.length;
      const half = (R0.h - 8) / 2, charts: [R, Float32Array, string, boolean, number][] = [
        [R4(R0.x, R0.y, R0.w, half), s.run.snapRec, 'reconstruction term (squared distance, log scale)', true, f.rec],
        [R4(R0.x, R0.y + half + 8, R0.w, half), s.run.snapKl, 'KL term (nats)', false, f.kl],
      ];
      charts.forEach(([r, ys, label, log, now], ci) => {
        box(r);
        txt(label, r.x + 8, r.y + 17, c.muted.hex, 12, 'left', 400, 1, true);
        const pl = { x: r.x + 44, y: r.y + 28, w: r.w - 60, h: r.h - (ci ? 58 : 40) };
        const lo = log ? Math.min(...ys) * 0.7 : 0, hi = log ? Math.max(...ys) * 1.4 : Math.ceil(Math.max(...ys) * 1.05);
        const map = (v: number) => (log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo));
        const X = (k: number) => pl.x + (st[k] / last) * pl.w, Y = (v: number) => pl.y + pl.h * (1 - map(v));
        if (log) for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) { const v = 10 ** e; line(pl.x, Y(v), pl.x + pl.w, Y(v), c.rule.hex, 1, 0.9); txt(String(+v.toPrecision(1)), pl.x - 6, Y(v) + 4, c.muted.hex, 12, 'right', 400); }
        else for (let v = 0; v <= hi; v++) { line(pl.x, Y(v), pl.x + pl.w, Y(v), c.rule.hex, 1, 0.9); txt(String(v), pl.x - 6, Y(v) + 4, c.muted.hex, 12, 'right', 400); }
        line(pl.x, pl.y + pl.h, pl.x + pl.w, pl.y + pl.h, c.muted.hex, 1.2, 0.8);
        if (ci) for (let q = 0; q <= 3; q++) { const k = Math.round((q / 3) * (nk - 1)); txt(String(st[k]), X(k), pl.y + pl.h + 16, c.muted.hex, 12, q === 0 ? 'left' : q === 3 ? 'right' : 'center', 400); }
        if (ci) txt('training steps', pl.x + pl.w / 2, pl.y + pl.h + 34, c.muted.hex, 12, 'center', 400, 1, true);
        ctx.globalAlpha = 0.25; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 2; ctx.beginPath();
        ys.forEach((v, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(v))); ctx.stroke();
        ctx.globalAlpha = 1; ctx.strokeStyle = ci ? c.grad.hex : c.fwd.hex; ctx.lineWidth = 2.4; ctx.beginPath();
        for (let k = 0; k <= f.k; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(ys[k]));
        ctx.stroke();
        line(X(f.k), pl.y, X(f.k), pl.y + pl.h, c.accent.hex, 1.2, 0.8, [3, 3]);
        dot(X(f.k), Y(now), 4.5, c.accent.hex);
        txt(log ? err(now) : now.toFixed(2), Math.min(X(f.k) + 9, pl.x + pl.w), clampX(Y(now) - 8, pl.y + 10, pl.y + pl.h - 4), c.text.hex, 12, f.k > nk * 0.75 ? 'right' : 'left', 700);
      });
    }

    // ------------------------------------------------ sweeping the code
    function drawExplore(s: Scene) {
      const f = exploreFrame(t, s, zOver); cur = f;
      drawPlane('data space'); drawCodeAxis(s);
      for (let i = 0; i < N; i++) dot(px(DATA.x[2 * i]), py(DATA.x[2 * i + 1]), narrow ? 1.8 : 2.2, s.moon[i] ? c.c2.hex : c.c1.hex, 0.6);
      clipPlane(); ctx.globalAlpha = 0.55; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let i = 0; i < s.curve.length / 2; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, px(s.curve[2 * i]), py(s.curve[2 * i + 1]));
      ctx.stroke(); ctx.restore();
      const r = S(), base = sBase();
      drawHist(s); if (s.prior) drawPrior(s);
      const mx = sx(s, f.z);
      line(mx, r.y + 22, mx, base, c.accent.hex, 1.6, 0.95);
      ctx.globalAlpha = 1; ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(mx, base + 1); ctx.lineTo(mx - 5, base + 9); ctx.lineTo(mx + 5, base + 9); ctx.fill();
      const dx = px(f.x), dy = py(f.y);
      if (f.near) { dot(dx, dy, 6.5, c.fwd.hex); ring(dx, dy, 9, c.text.hex, 1.2, 0.8); } else { cross(dx, dy, 7, c.accent.hex, 3); ring(dx, dy, 10, c.accent.hex, 1.2, 0.8); }
    }

    // ------------------------------------------------ random codes
    function drawSample(s: Scene) {
      const f = sampleFrame(t, s), sm = s.samples[f.batch]; cur = f;
      drawPlane('data space'); drawCodeAxis(s);
      for (let i = 0; i < N; i++) dot(px(DATA.x[2 * i]), py(DATA.x[2 * i + 1]), narrow ? 1.8 : 2.2, c.muted.hex, 0.28);
      const base = sBase(), x0 = sx(s, s.zlo), x1 = sx(s, s.zhi);
      if (s.prior) { drawHist(s, 0.3); drawPrior(s); }
      else { ctx.globalAlpha = 0.1; ctx.fillStyle = c.muted.hex; ctx.fillRect(x0, S().y + 22, x1 - x0, base - S().y - 22); }
      for (let i = 0; i < Math.min(SAMPLES, Math.ceil(f.n)); i++) {
        const a = Math.min(1, f.n - i), on = sm.on[i] === 1;
        if (on) { dot(px(sm.pt[2 * i]), py(sm.pt[2 * i + 1]), 3, c.fwd.hex, 0.85 * a); dot(sx(s, sm.z[i]), sy(s.jit[i]), 2.4, c.fwd.hex, 0.7 * a); }
        else { cross(px(sm.pt[2 * i]), py(sm.pt[2 * i + 1]), 3.6, c.accent.hex, 1.8, 0.95 * a); cross(sx(s, sm.z[i]), sy(s.jit[i]), 2.8, c.accent.hex, 1.5, 0.8 * a); }
      }
    }

    // ------------------------------------------------ denoising
    function drawDenoise(s: Scene) {
      const f = denoiseFrame(t, s); cur = f;
      drawPlane('data space');
      for (let i = 0; i < N; i++) dot(px(DATA.x[2 * i]), py(DATA.x[2 * i + 1]), narrow ? 1.6 : 2, c.muted.hex, 0.15);
      for (let i = 0; i < DENOISE_POINTS; i++) {
        const ax = px(f.inp[2 * i]), ay = py(f.inp[2 * i + 1]), bx = px(f.out[2 * i]), by = py(f.out[2 * i + 1]), len = Math.hypot(bx - ax, by - ay);
        if (len > 3) {
          line(ax, ay, bx, by, c.grad.hex, 1.1, 0.55);
          if (len > 9) { const u = [(bx - ax) / len, (by - ay) / len]; line(bx, by, bx - 6 * u[0] - 3 * u[1], by - 6 * u[1] + 3 * u[0], c.grad.hex, 1.1, 0.7); line(bx, by, bx - 6 * u[0] + 3 * u[1], by - 6 * u[1] - 3 * u[0], c.grad.hex, 1.1, 0.7); }
        }
        dot(ax, ay, 2.6, c.muted.hex, 0.85); dot(bx, by, 2.6, c.fwd.hex, 0.95);
      }
      txt(`noise ${f.sigma.toFixed(2)}`, lay.P.x + lay.P.w - 8, lay.P.y + 16, c.text.hex, 13, 'right', 700);
    }

    // ------------------------------------------------ one point's bell curve (VAE)
    function drawEncode(s: Scene) {
      const f = encodeFrame(t), b = s.bell!; cur = f;
      drawPlane('data space'); drawCodeAxis(s);
      const r = narrow ? 1.9 : 2.3, fi = b.focus;
      for (let i = 0; i < N; i++) { if (i === fi) continue; dot(px(s.X[2 * i]), py(s.X[2 * i + 1]), r, c.muted.hex, 0.22); dot(sx(s, s.Z[i]), sy(s.jit[i]), r, c.muted.hex, 0.2); }
      const a = ease(t / 1.2), { peak, base } = drawBell(s, b.mu, b.sigma, a);
      const mx = sx(s, b.mu);
      line(mx, base, mx, base - peak - 6, c.accent.hex, 1.4, 0.9 * a, [3, 3]);
      txt(`\u03bc = ${mn(b.mu)}`, clampX(mx, S().x + 44, S().x + S().w - 44), S().y + 38, c.accent.hex, 12, 'center', 700, a);
      const ys = base - peak * 0.55;
      line(sx(s, b.mu - b.sigma), ys, sx(s, b.mu + b.sigma), ys, c.text.hex, 1.6, 0.9 * a); line(sx(s, b.mu - b.sigma), ys - 4, sx(s, b.mu - b.sigma), ys + 4, c.text.hex, 1.6, 0.9 * a); line(sx(s, b.mu + b.sigma), ys - 4, sx(s, b.mu + b.sigma), ys + 4, c.text.hex, 1.6, 0.9 * a);
      txt(`\u03c3 = ${b.sigma.toFixed(2)}`, sx(s, b.mu - b.sigma) - 8, ys + 4, c.text.hex, 12, 'right', 700, a);
      const ox = px(s.X[2 * fi]), oy = py(s.X[2 * fi + 1]);
      ring(ox, oy, 6.5, c.accent.hex, 1.8);
      for (let k = 0; k < Math.min(BELL_K, Math.ceil(f.n)); k++) {
        const al = Math.min(1, f.n - k), yy = base - 8 - (((k * 37) % 11) / 10) * peak * 0.5;
        dot(sx(s, b.z[k]), yy, 2.5, c.accent.hex, 0.9 * al); dot(px(b.pt[2 * k]), py(b.pt[2 * k + 1]), 3, c.fwd.hex, 0.85 * al);
      }
      ring(px(s.R[2 * fi]), py(s.R[2 * fi + 1]), 5, c.fwd.hex, 1.8);
    }

    // ------------------------------------------------ z = mu + sigma eps (VAE)
    function drawReparam(s: Scene) {
      const f = reparamFrame(t, s, init.focus); cur = f;
      const fi = init.focus, E = lay.E!, r = narrow ? 1.9 : 2.3;
      drawPlane('data space'); drawCodeAxis(s);
      for (let i = 0; i < N; i++) if (i !== fi) dot(px(s.X[2 * i]), py(s.X[2 * i + 1]), r, c.muted.hex, 0.22);
      ring(px(s.X[2 * fi]), py(s.X[2 * fi + 1]), 6.5, c.accent.hex, 1.8);
      // the decoder curve near the point, then the decoded point itself
      dot(px(f.x), py(f.y), 5.5, c.fwd.hex);
      // the standard-normal strip
      box(E); txt('\u03b5 drawn from N(0, 1)', E.x + 8, E.y + 15, c.muted.hex, 12, 'left', 400, 1, true);
      const ex = (v: number) => E.x + 14 + ((v + PRIOR_V) / (2 * PRIOR_V)) * (E.w - 28), eb = E.y + E.h - 20, ep = E.h - 44;
      line(E.x + 8, eb, E.x + E.w - 8, eb, c.muted.hex, 1.2, 0.8);
      for (let v = -3; v <= 3; v++) { line(ex(v), eb, ex(v), eb + 4, c.muted.hex, 1, 0.8); txt(mn(v, 0), ex(v), eb + 16, c.muted.hex, 12, 'center', 400); }
      ctx.globalAlpha = 1; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.6; ctx.beginPath();
      for (let k = 0; k <= 60; k++) { const v = -PRIOR_V + (2 * PRIOR_V * k) / 60; (k ? ctx.lineTo : ctx.moveTo).call(ctx, ex(v), eb - (priorPdf(v) / 0.4) * ep); }
      ctx.stroke();
      const ey = eb - (priorPdf(f.eps) / 0.4) * ep;
      line(ex(f.eps), eb, ex(f.eps), ey, c.accent.hex, 1.6); dot(ex(f.eps), ey, 4.5, c.accent.hex);
      // the code line: the point's bell curve, then z
      const { peak, base } = drawBell(s, f.mu, f.sigma, 1);
      const zx = sx(s, f.z);
      line(zx, base, zx, base - peak - 8, c.accent.hex, 1.8); dot(zx, base - peak - 8, 4.5, c.accent.hex);
      line(ex(f.eps), eb + 20, zx, S().y + 2, c.muted.hex, 1.2, 0.7, [4, 3]);
      txt(`z = ${mn(zShown(f), 3)}`, clampX(zx, S().x + 40, S().x + S().w - 40), S().y + 38, c.accent.hex, 12, 'center', 700);
      // what the gradient reaches
      const X0 = lay.X!, nodes: [string, number][] = [['\u03bc', 14], ['\u03c3', 44], ['\u03b5', 74]], zx0 = X0.x + 118, zy = X0.y + 52;
      box(X0);
      nodes.forEach(([name, dy], k) => {
        const nx = X0.x + 14, ny = X0.y + dy;
        ctx.globalAlpha = 1; ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.roundRect(nx, ny, 40, 24, 5); ctx.fill();
        ctx.strokeStyle = k === 2 ? c.muted.hex : c.rule.hex; ctx.lineWidth = 1.2; ctx.setLineDash(k === 2 ? [3, 3] : []); ctx.stroke(); ctx.setLineDash([]);
        uitxt(name, nx + 20, ny + 17, c.text.hex, 15, 'center', 600);
        arrow(nx + 42, ny + 12 + (k === 2 ? 0 : -3), zx0 - 3, zy + (k - 1) * 5, c.muted.hex, 1.3, 0.85, k === 2 ? [3, 3] : []);
        if (k < 2) arrow(zx0 - 3, zy + (k - 1) * 5 + 8, nx + 42, ny + 12 + 6, c.grad.hex, 1.8, 0.95);
      });
      ctx.globalAlpha = 1; ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.roundRect(zx0, zy - 12, 34, 24, 5); ctx.fill(); ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.2; ctx.stroke();
      uitxt('z', zx0 + 17, zy + 5, c.text.hex, 15, 'center', 600);
      const tx = X0.x + 176;
      txt('\u2202z/\u2202\u03bc = 1', tx, X0.y + 24, c.grad.hex, 13, 'left', 700);
      txt(`\u2202z/\u2202\u03c3 = \u03b5 = ${mn(f.eps)}`, tx, X0.y + 48, c.grad.hex, 13, 'left', 700);
      uitxt('\u03b5 is an input:', tx, X0.y + 74, c.muted.hex, 13, 'left', 400);
      uitxt('nothing to update', tx, X0.y + 90, c.muted.hex, 13, 'left', 400);
    }

    // ------------------------------------------------ the same network at four KL weights (VAE)
    function drawBeta() {
      const f = betaFrame(t), m = scs[f.k], pv = f.k > 0 ? scs[f.k - 1] : null; cur = f;
      drawPlane('data space'); drawCodeAxis(m);
      for (let i = 0; i < N; i++) dot(px(DATA.x[2 * i]), py(DATA.x[2 * i + 1]), narrow ? 1.8 : 2.2, c.muted.hex, 0.3);
      const layers: [Scene, number][] = pv ? [[pv, 1 - f.fade], [m, f.fade]] : [[m, 1]];
      clipPlane();
      for (const [q, al] of layers) {
        if (al <= 0.01) continue;
        ctx.globalAlpha = al; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let i = 0; i < q.curve.length / 2; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, px(q.curve[2 * i]), py(q.curve[2 * i + 1]));
        ctx.stroke();
      }
      ctx.restore();
      for (const [q, al] of layers) {
        if (al <= 0.01) continue;
        const sm = q.samples[0];
        for (let i = 0; i < SAMPLES; i++) {
          if (sm.on[i]) dot(px(sm.pt[2 * i]), py(sm.pt[2 * i + 1]), 2.8, c.fwd.hex, 0.7 * al);
          else cross(px(sm.pt[2 * i]), py(sm.pt[2 * i + 1]), 3.4, c.accent.hex, 1.6, 0.9 * al);
        }
        drawHist(q, 0.85 * al);
      }
      drawPrior(m);
      const X0 = lay.X!, rows: [string, string][] = [
        ['reconstruction error', err(m.mse)], ['KL to the prior (nats)', m.kl.toFixed(2)], ['prior samples on the data', `${m.samples[0].cum[SAMPLES]} of ${SAMPLES}`],
      ];
      box(X0);
      uitxt(`\u03b2 = ${VAE_BETAS[f.k]}`, X0.x + 12, X0.y + 28, c.text.hex, 18, 'left', 700);
      rows.forEach(([name, val], k) => { const y = X0.y + 62 + k * 34; uitxt(name, X0.x + 12, y, c.muted.hex, 14, 'left', 500); txt(val, X0.x + X0.w - 12, y, c.text.hex, 15, 'right', 700); });
    }

    function render() {
      const k0 = cv.width / Wd;
      ctx.setTransform(k0, 0, 0, k0, 0, 0);
      ctx.clearRect(0, 0, Wd, Hh);
      if (!sc) drawLoading();
      else if (init.mode === 'encode') drawEncode(sc);
      else if (init.mode === 'reparam') drawReparam(sc);
      else if (init.mode === 'beta') drawBeta();
      else if (init.mode === 'tour') drawTrip(sc, false);
      else if (init.mode === 'focus') drawTrip(sc, true);
      else if (init.mode === 'train') drawTrain(sc);
      else if (init.mode === 'explore') drawExplore(sc);
      else if (init.mode === 'sample') drawSample(sc);
      else drawDenoise(sc);
      ctx.globalAlpha = 1;
    }

    const loop = createLoop(cv, (dt, now) => {
      const T = tm.T;
      if (sc) { if (t > T) t = 0; t = advance({ t, playing, speed }, dt, { duration: T }).t; }
      render();
      tl.paint(t);
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        const s = say(); if (s !== narration) narration = s;
        if (cur && init.mode === 'explore') sliderZ = cur.z;
      }
      return playing || !sc;
    }, { onView: (inV, first) => { if (inV && first && !reduce) playing = true; } });
    poke = loop.wake;

    Promise.all(ids.map(load)).then((runs) => {
      scs = runs.map((run) => buildScene(run, init.mode, init.probes, init.showPrior, init.focus));
      sc = scs[0];
      if (reduce) t = restAt(init.mode, timing(init.mode, sc).T);
      poke();
    }, () => { failed = true; poke(); });

    const fit = () => {
      const d = Math.min(2, Math.max(1, window.devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; fit(); poke(); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(C); code = readVar('--font-code'); ui = readVar('--font-body'); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });

  // dragging on the code line (explore) or the slider sets the code by hand
  function setZ(z: number) { if (!sc) return; zOver = Math.min(sc.zhi, Math.max(sc.zlo, z)); sliderZ = zOver; playing = false; poke(); }
  function onPointer(e: PointerEvent) {
    if (init.mode !== 'explore' || !sc || !(e.buttons & 1)) return;
    const b = cv.getBoundingClientRect(), s = lay.S!, x = ((e.clientX - b.left) / b.width) * Wd, y = ((e.clientY - b.top) / b.height) * Hh;
    if (y < s.y - 8 || y > s.y + s.h + 8) return;
    setZ(sc.vlo + ((x - s.x - 14) / (s.w - 28)) * (sc.vhi - sc.vlo));
  }

  const stageLabel = $derived({
    tour: `Every training point of the two moons is squeezed into a ${L === 1 ? 'single number' : 'code of two numbers'} and rebuilt from it.`,
    focus: 'One training point passes through the encoder, a single code number and the decoder, and comes out slightly moved.',
    train: 'The curve the decoder draws at each stage of training, with a log-scale chart of the reconstruction error.',
    explore: 'A marker slides along the code line and the decoded point moves along the curve the decoder draws.',
    sample: 'Random codes dropped onto the code line and the points the decoder makes from them.',
    encode: 'One training point becomes a bell curve on the code line; codes drawn from it are decoded to points near the original.',
    reparam: 'A random number epsilon from a standard normal is scaled by sigma and shifted by mu to make the code; the decoded point moves with it.',
    beta: 'The decoder curve, the code histogram and 300 samples from the prior for a VAE trained with four different KL weights, one at a time.',
    denoise: 'Noisy points, and the points the denoising autoencoder returns for them, joined by arrows.',
  }[init.mode]);
</script>

<FigureCard title={init.title} {legend} aspect="720 / {lw.H}" narrowAspect="400 / {ln.H}" label={stageLabel} role={init.mode === 'explore' ? 'group' : 'img'}>
  <canvas bind:this={cv} aria-hidden="true" onpointerdown={onPointer} onpointermove={onPointer} style:touch-action={init.mode === 'explore' ? 'pan-y' : 'auto'}></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={tm.T} markers={tm.markers} label="Timeline"
      onSeek={(to) => { t = to; zOver = null; poke(); }} onStep={(to) => { playing = false; t = to; zOver = null; poke(); }} onReset={() => { t = 0; zOver = null; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if init.mode === 'beta' && sc}
      <div class="ctl">
        <span class="lbl">KL weight β</span>
        <span class="seg" role="group" aria-label="KL weight">
          {#each VAE_BETAS as b, k (b)}
            <button type="button" class="btn" aria-pressed={betaFrame(uiT).k === k} onclick={() => { playing = false; t = k * BETA_SEG + BETA_FADE; poke(); }}>{b}</button>
          {/each}
        </span>
      </div>
    {/if}
    {#if init.mode === 'explore' && sc}
      <div class="ctl">
        <label>code z
          <input type="range" min={sc.zlo} max={sc.zhi} step="0.01" value={sliderZ} oninput={(e) => setZ(+e.currentTarget.value)} aria-label="Code z, from {mn(sc.zlo)} to {mn(sc.zhi)}" />
        </label>
        <span class="hint">drag the slider, or drag along the code line</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-body); font-variant-numeric: tabular-nums; min-height: 2.9em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-3); color: var(--text-muted); font-family: var(--font-code); }
  .ctl input[type='range'] { width: min(18rem, 60vw); accent-color: var(--accent); }
  .lbl { color: var(--text-muted); }
  .seg { display: inline-flex; gap: var(--s-2); }
  .seg .btn[aria-pressed='true'] { border-color: var(--accent); font-weight: 650; }
  .hint { font: var(--step--1) var(--font-ui); color: var(--text-muted); }
</style>
