<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { GanRun, RunName } from './sim';
  const runs: Partial<Record<RunName, Promise<GanRun>>> = {};
  const progress: Partial<Record<RunName, number>> = {};
  /** Train each run once per page in its own worker; every figure on the page shares the result. */
  function load(name: RunName): Promise<GanRun> {
    return (runs[name] ??= new Promise((res, rej) => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.kind === 'progress') progress[name] = e.data.frac;
        else { progress[name] = 1; res(e.data.run); w.terminate(); }
      };
      w.onerror = () => rej(new Error('worker failed'));
      w.postMessage({ name });
    }));
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import { DOMAIN, MOONS } from '../../../lib/gen/toy';
  import {
    RUN_CFG, STAT, DATA, DATA_RING, RING, MAP_N, GRID_N, GRID_R, GRAD_SNAP, GRAD_N,
    points, dMap, smoothLoss, ringShares, noiseGrid, generate, gradDemo,
    durationOf, HOLD, runFrame, runMarkers, genFrame, gradFrame, GEN_MARKERS, GRAD_MARKERS,
    type GanRun, type RunName, type Mode, type GradDemo,
  } from './sim';

  interface Props {
    /** 'duel': whole run. 'generator': noise grid through G. 'discriminator': D trains alone. 'objective': loss curves. 'gradients': one update. 'collapse': ring run with mode shares. */
    mode?: Mode;
    /** Which training run to show (default depends on the mode). */
    run?: RunName;
    title?: string;
    /** 'rate': let the reader switch the generator's learning rate. 'ring': switch between the two ring runs. */
    choose?: '' | 'rate' | 'ring';
  }
  let { mode = 'duel', run = 'stable', title = '', choose = '' }: Props = $props();

  const init = untrack(() => {
    const base: RunName = mode === 'generator' || mode === 'gradients' ? 'stable' : mode === 'discriminator' ? 'disc' : run;
    return {
      mode, choose, base,
      title: title || { duel: 'A generator and a discriminator training together', generator: 'The generator turns a noise grid into points', discriminator: 'The discriminator learns to tell real from fake', objective: 'The two losses during training', gradients: 'How the discriminator pushes each generated point', collapse: 'A generator that covers only some of the modes' }[mode],
    };
  });
  const T = durationOf(init.mode);

  let cur = $state<RunName>(init.base);
  let data = $state.raw<GanRun | null>(null);
  let failed = $state(false);
  const markers = $derived(init.mode === 'generator' ? GEN_MARKERS : init.mode === 'gradients' ? GRAD_MARKERS : data ? runMarkers(data) : []);

  let narrow = $state(false);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let narration = $state('');
  const Wd = $derived(narrow ? 360 : 600);
  const Hh = $derived(narrow ? 540 : 330);

  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, poke = () => {};
  $effect(() => { playing; speed; poke(); });
  $effect(() => {
    const name = cur;
    data = null; failed = false;
    load(name).then((r) => { if (cur === name) { data = r; poke(); } }, () => { failed = true; poke(); });
  });

  const legend = [
    { label: 'real data', color: '--text-muted' },
    { label: 'generated', color: '--viz-forward' },
    ...(init.mode === 'generator' ? [] : [{ label: 'D says real', color: '--viz-pos' }, { label: 'D says fake', color: '--viz-neg' }]),
    ...(init.mode === 'gradients' ? [{ label: 'push direction', color: '--viz-gradient' }] : []),
  ];

  // ---------------------------------------------------------------- per-run memo (plain variables, no deep proxies)
  let smD: Float32Array | null = null, smG: Float32Array | null = null, smFor: GanRun | null = null;
  const smoothOf = (r: GanRun) => {
    if (smFor !== r) { smD = smoothLoss(r.losses, 0); smG = smoothLoss(r.losses, 1); smFor = r; }
    return { d: smD!, g: smG! };
  };
  let demo: GradDemo | null = null, demoFor: GanRun | null = null;
  const demoOf = (r: GanRun) => { if (demoFor !== r) { demo = gradDemo(r.snaps[GRAD_SNAP]); demoFor = r; } return demo!; };
  let gridOut: Float64Array | null = null, gridFor: GanRun | null = null;
  const gridOf = (r: GanRun) => {
    if (gridFor !== r) { gridOut = generate(r.snaps[r.snaps.length - 1], noiseGrid()); gridFor = r; }
    return gridOut!;
  };
  const pct = (v: number) => `${Math.round(100 * v)}%`;

  function say(): string {
    if (!data) return failed ? 'Training failed in this browser.' : 'Training the networks in your browser…';
    const m = init.mode;
    if (m === 'generator') {
      const f = genFrame(t);
      return ['The grid of noise points, two numbers each.', 'Each noise point goes through the generator, and its grid lines stretch and fold with it.', 'The generator’s output: the same grid, now drawn over the real data.'][f.phase];
    }
    if (m === 'gradients') {
      const f = gradFrame(t), d = demoOf(data);
      return [
        `The points the generator had made by step ${data.steps[GRAD_SNAP]}, on the discriminator’s background.`,
        'Each arrow is the direction in which the discriminator’s score for that point rises fastest.',
        'Every point moves along its arrow (an exaggerated step, to make it visible).',
        `After the move the discriminator’s score averages ${d.dAfter.toFixed(2)}, up from ${d.dBefore.toFixed(2)}.`,
      ][f.phase];
    }
    const f = runFrame(data, t), seg = f.seg, S = STAT, n = data.snaps.length, st = data.stats, on = st[f.idx * S];
    // Every sentence below is true for every frame that can show it: it is chosen from that frame's own numbers, or states something true of the whole segment.
    if (m === 'duel') {
      if (cur === 'unstable') return 'In this run each update moves the cloud a long way, so the share of points on the data jumps from one checkpoint to the next instead of settling.';
      if (on < 0.3) return 'The cloud is still mostly off the moons. The background is the discriminator\u2019s current opinion of every place on the plane.';
      if (on < 0.65) return 'The cloud is spreading over the moons while the discriminator keeps redrawing its opinion.';
      return 'Most of the cloud now lies on the moons. Where it overlaps the data the background is pale: the discriminator is unsure there.';
    }
    if (m === 'discriminator') {
      const a = st[f.idx * S + 5];
      if (a < 0.6) return 'Only the discriminator learns. It has barely started, so its background is still close to flat.';
      if (a < 0.97) return 'The background is splitting into a real region (blue) and a fake region (orange) around the frozen cloud.';
      return `The edge of the orange patch is where D(x) = 0.5. The discriminator now calls ${pct(a)} of the points correctly.`;
    }
    if (m === 'objective') {
      const { d, g } = smoothOf(data), mean = (a: Float32Array, lo: number) => { let s = 0; for (let i = lo; i < a.length; i++) s += a[i]; return s / (a.length - lo); };
      return [
        'Early on both losses swing: the discriminator takes a lead and the forger catches up.',
        `From step 500 on the losses average ${mean(d, 500).toFixed(2)} for the detective and ${mean(g, 500).toFixed(2)} for the forger, while the cloud keeps changing.`,
        `Neither loss falls the way a normal training loss does, yet ${pct(on)} of the points are now on the data.`,
      ][seg];
    }
    const sh = ringShares(points(data.snaps[f.idx])), covered = sh.modes.filter((v) => v >= 0.05).length;
    if (sh.off > 0.5) return 'Most points are still spread around the ring, between the clusters.';
    return `${covered} of the 8 clusters hold at least 5% of the points, and ${pct(sh.off)} of the points are between clusters.`;
  }

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const C = {
      fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg', muted: '--text-muted', text: '--text',
      rule: '--rule', raised: '--bg-raised', accent: '--accent', cat: '--viz-cat-7',
    };
    let c = readColors(C), ui = readVar('--font-ui');
    let lastUi = 0, k = 1; // k: canvas pixels per CSS pixel of logical size, so text sizes below are CSS px
    const off = document.createElement('canvas'); off.width = off.height = MAP_N;
    const octx = off.getContext('2d')!;
    const img = octx.createImageData(MAP_N, MAP_N);

    const dot = (x: number, y: number, r: number, col: string, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); };
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'left', w = 500, a = 1) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align; ctx.font = `${w} ${px / k}px ${ui}`; ctx.fillText(s, x, y);
    };
    const line = (pts: ArrayLike<number>, col: string, w: number, a = 1, dash: number[] = []) => {
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < pts.length; i += 2) (i ? ctx.lineTo(pts[i], pts[i + 1]) : ctx.moveTo(pts[i], pts[i + 1]));
      ctx.stroke(); ctx.setLineDash([]);
    };
    interface R { x: number; y: number; w: number; h: number }
    const frame = (r: R) => { ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h); };

    // ------------------------------------------------ layout
    function layout(): { plane: R; panel: R; hud: { x: number; y: number } } {
      if (init.mode === 'generator') {
        return narrow
          ? { panel: { x: 105, y: 22, w: 150, h: 150 }, plane: { x: 10, y: 192, w: 340, h: 320 }, hud: { x: 10, y: 18 } }
          : { panel: { x: 8, y: 36, w: 270, h: 270 }, plane: { x: 322, y: 36, w: 270, h: 270 }, hud: { x: 8, y: 20 } };
      }
      return narrow
        ? { plane: { x: 25, y: 28, w: 310, h: 310 }, panel: { x: 10, y: 372, w: 340, h: 158 }, hud: { x: 10, y: 18 } }
        : { plane: { x: 8, y: 28, w: 290, h: 290 }, panel: { x: 322, y: 52, w: 270, h: 242 }, hud: { x: 8, y: 18 } };
    }
    const sx = (p: R, x: number) => p.x + ((x + DOMAIN) / (2 * DOMAIN)) * p.w;
    const sy = (p: R, y: number) => p.y + ((DOMAIN - y) / (2 * DOMAIN)) * p.h;

    function drawMap(p: R, snap: Float64Array) {
      const m = dMap(snap), bg = c.raised.rgb, dark = bg[0] + bg[1] + bg[2] < 384;
      for (let i = 0; i < m.length; i++) {
        const d = m[i], tgt = d >= 0.5 ? c.pos.rgb : c.neg.rgb, a = Math.min(1, Math.abs(d - 0.5) * 2) * (dark ? 0.42 : 0.62);
        img.data[4 * i] = bg[0] + (tgt[0] - bg[0]) * a; img.data[4 * i + 1] = bg[1] + (tgt[1] - bg[1]) * a;
        img.data[4 * i + 2] = bg[2] + (tgt[2] - bg[2]) * a; img.data[4 * i + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      ctx.globalAlpha = 1; ctx.imageSmoothingEnabled = true; ctx.drawImage(off, p.x, p.y, p.w, p.h);
    }
    function drawReal(p: R, rg: boolean) {
      const d = rg ? DATA_RING.x : DATA.x;
      const dark = c.raised.rgb[0] + c.raised.rgb[1] + c.raised.rgb[2] < 384;
      ctx.globalAlpha = dark ? 0.85 : 0.55; ctx.fillStyle = dark ? c.text.hex : c.muted.hex;
      for (let i = 0; i < d.length; i += 2) { ctx.beginPath(); ctx.arc(sx(p, d[i]), sy(p, d[i + 1]), dark ? 1.9 : 1.7, 0, 6.2832); ctx.fill(); }
    }
    function drawFake(p: R, g: ArrayLike<number>, r = 2.6, a = 0.95) {
      ctx.globalAlpha = a; ctx.fillStyle = c.fwd.hex;
      for (let i = 0; i < g.length; i += 2) { const x = sx(p, g[i]), y = sy(p, g[i + 1]); if (x < p.x - 4 || x > p.x + p.w + 4 || y < p.y - 4 || y > p.y + p.h + 4) continue; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); }
    }
    function clipPlane(p: R, fn: () => void) { ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip(); fn(); ctx.restore(); }
    function arrow(x0: number, y0: number, x1: number, y1: number, col: string, a: number, w = 1.6) {
      const dx = x1 - x0, dy = y1 - y0, l = Math.hypot(dx, dy);
      if (l < 1.5) return;
      const ux = dx / l, uy = dy / l, h = Math.min(6, l * 0.5);
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * h * 0.6, y1 - uy * h * 0.6); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - ux * h - uy * h * 0.55, y1 - uy * h + ux * h * 0.55); ctx.lineTo(x1 - ux * h + uy * h * 0.55, y1 - uy * h - ux * h * 0.55); ctx.fill();
    }

    /** Chart frame: x in [0, xmax], y in [y0, y1]; returns coordinate maps and a series helper. */
    function chart(p: R, xmax: number, y0: number, y1: number, ticks: number[], fmt: (v: number) => string, cursor: number, xlabel: string) {
      const l = p.x + 34, r = p.x + p.w - 6, top = p.y + 4, bot = p.y + p.h - 22;
      const X = (v: number) => l + (v / xmax) * (r - l), Y = (v: number) => bot - ((v - y0) / (y1 - y0)) * (bot - top);
      ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(l, top); ctx.lineTo(l, bot); ctx.lineTo(r, bot); ctx.stroke();
      for (const v of ticks) { txt(fmt(v), l - 5, Y(v) + 4, c.muted.hex, 11, 'right'); ctx.globalAlpha = 0.4; ctx.beginPath(); ctx.moveTo(l, Y(v)); ctx.lineTo(r, Y(v)); ctx.stroke(); }
      txt(xlabel, (l + r) / 2, p.y + p.h - 4, c.muted.hex, 11, 'center');
      txt('0', l, bot + 14, c.muted.hex, 11, 'center'); txt(String(xmax), r, bot + 14, c.muted.hex, 11, 'right');
      const series = (xs: ArrayLike<number>, ys: ArrayLike<number>, n: number, col: string, w = 2) => {
        const pts: number[] = [];
        for (let i = 0; i < n; i++) pts.push(X(xs[i]), Y(ys[i]));
        line(pts, col, w, 1);
      };
      return { X, Y, series, l, r, top, bot, cursor: () => { ctx.globalAlpha = 0.7; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(cursor), top); ctx.lineTo(X(cursor), bot); ctx.stroke(); } };
    }
    function bars(p: R, labels: string[], vals: number[], cols: string[], ymax: number) {
      const n = vals.length, l = p.x + 6, w = (p.w - 12) / n, bot = p.y + p.h - 20, top = p.y + 18;
      ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l, bot + 0.5); ctx.lineTo(l + n * w, bot + 0.5); ctx.stroke();
      for (let i = 0; i < n; i++) {
        const h = Math.min(1, vals[i] / ymax) * (bot - top), x = l + i * w + w * 0.15;
        ctx.globalAlpha = 1; ctx.fillStyle = cols[i]; ctx.fillRect(x, bot - h, w * 0.7, h);
        txt(labels[i], l + (i + 0.5) * w, bot + 14, c.muted.hex, 11, 'center');
        txt(`${Math.round(100 * vals[i])}`, l + (i + 0.5) * w, bot - h - 4, c.text.hex, 11, 'center');
      }
    }

    function drawLoading(p: R) {
      frame(p);
      const prog = progress[cur] ?? 0, w = Math.min(220, p.w - 40), x = p.x + (p.w - w) / 2, y = p.y + p.h / 2;
      txt(failed ? 'Training failed in this browser.' : `Training in your browser… ${Math.round(100 * prog)}%`, p.x + p.w / 2, y - 12, c.muted.hex, 12.5, 'center');
      ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.strokeRect(x + 0.5, y + 0.5, w, 7);
      ctx.fillStyle = c.fwd.hex; ctx.fillRect(x + 1, y + 1.5, (w - 1) * prog, 6);
    }

    // ------------------------------------------------ the kinds of frame
    function drawRun(L: ReturnType<typeof layout>, r: GanRun) {
      const f = runFrame(r, t), snap = r.snaps[f.idx], p = L.plane, rg = r.cfg.ring === true, S = STAT;
      drawMap(p, snap); frame(p);
      clipPlane(p, () => { drawReal(p, rg); drawFake(p, points(snap)); });
      if (rg) for (let k = 0; k < 8; k++) txt(String(k + 1), sx(p, RING.mu[2 * k] * 1.25), sy(p, RING.mu[2 * k + 1] * 1.25) + 4.5, c.text.hex, 11.5, 'center', 600);
      const m = init.mode;
      const hud = (s: string) => txt(s, L.hud.x, L.hud.y, c.text.hex, 12.5, 'left', 600);
      const pn = L.panel, on = r.stats[f.idx * S];
      if (m === 'duel') {
        hud(`Step ${f.step} of ${r.cfg.steps}`);
        txt(`on the data: ${pct(on)}`, p.x + p.w, L.hud.y, c.fwd.hex, 12.5, 'right', 600);
        const n = r.snaps.length, xs = r.steps, ys = Float64Array.from({ length: n }, (_, i) => r.stats[i * S]);
        txt('Share of generated points on the data', pn.x, pn.y - 8, c.muted.hex, 11.5, 'left');
        const ch = chart(pn, r.cfg.steps, 0, 1, [0, 0.5, 1], (v) => `${Math.round(100 * v)}%`, f.step, 'training step');
        ch.series(xs, ys, f.idx + 1, c.fwd.hex); ch.cursor();
        dot(ch.X(xs[f.idx]), ch.Y(ys[f.idx]), 3.5, c.fwd.hex);
      } else if (m === 'collapse') {
        hud(`Step ${f.step} of ${r.cfg.steps}`);
        const sh = ringShares(points(snap)), covered = sh.modes.filter((v) => v >= 0.05).length;
        txt(`modes with 5%+: ${covered} of 8`, p.x + p.w, L.hud.y, c.fwd.hex, 12.5, 'right', 600);
        txt('Share of generated points near each mode (%)', pn.x, pn.y - 8, c.muted.hex, 11.5, 'left');
        bars(pn, [...Array.from({ length: 8 }, (_, i) => String(i + 1)), 'off'], [...sh.modes, sh.off], [...Array(8).fill(c.fwd.hex), c.muted.hex], 0.4);
      } else if (m === 'objective') {
        hud(`Step ${f.step} of ${r.cfg.steps}`);
        const { d, g } = smoothOf(r), xs = Float64Array.from({ length: d.length }, (_, i) => i + 1);
        txt('Average loss over the last 25 steps', pn.x, pn.y - 8, c.muted.hex, 11.5, 'left');
        const ch = chart(pn, r.cfg.steps, 0, 1.6, [0, 0.5, 1, 1.5], (v) => v.toFixed(1), f.step, 'training step');
        for (const [v, lab] of [[Math.log(4), 'log 4'], [Math.log(2), 'log 2']] as const) { line([ch.l, ch.Y(v), ch.r, ch.Y(v)], c.muted.hex, 1, 0.9, [4, 3]); txt(`${lab} = ${v.toFixed(3)}`, ch.r - 2, ch.Y(v) - 4, c.muted.hex, 11, 'right'); }
        const k = Math.max(1, f.step);
        ch.series(xs, d, k, c.cat.hex); ch.series(xs, g, k, c.fwd.hex); ch.cursor();
        txt('detective', ch.X(r.cfg.steps * 0.3), ch.Y(1.2) + 6, c.cat.hex, 11.5, 'left', 600);
        txt('forger', ch.X(r.cfg.steps * 0.62), ch.Y(0.84) - 9, c.fwd.hex, 11.5, 'left', 600);
      } else {
        hud(`Step ${f.step} of ${r.cfg.steps}`);
        const acc = Float64Array.from({ length: r.snaps.length }, (_, i) => r.stats[i * S + 5]);
        txt(`D accuracy: ${pct(acc[f.idx])}`, p.x + p.w, L.hud.y, c.pos.hex, 12.5, 'right', 600);
        txt('Discriminator accuracy', pn.x, pn.y - 8, c.muted.hex, 11.5, 'left');
        const ch = chart(pn, r.cfg.steps, 0.4, 1, [0.5, 0.75, 1], (v) => `${Math.round(100 * v)}%`, f.step, 'training step');
        line([ch.l, ch.Y(0.5), ch.r, ch.Y(0.5)], c.muted.hex, 1, 0.9, [4, 3]); txt('chance', ch.r - 2, ch.Y(0.5) - 4, c.muted.hex, 11, 'right');
        ch.series(r.steps, acc, f.idx + 1, c.pos.hex); ch.cursor();
        dot(ch.X(r.steps[f.idx]), ch.Y(acc[f.idx]), 3.5, c.pos.hex);
      }
    }

    const nearData = (x: number, y: number) => {
      let best = Infinity;
      for (let k = 0; k < MOONS.K; k++) { const dx = x - MOONS.mu[2 * k], dy = y - MOONS.mu[2 * k + 1]; const d = dx * dx + dy * dy; if (d < best) best = d; }
      return best <= 9 * MOONS.sigma * MOONS.sigma;
    };
    function drawGenerator(L: ReturnType<typeof layout>, r: GanRun) {
      const pl = L.plane, pn = L.panel, f = genFrame(t), out = gridOf(r), z = noiseGrid(), N = GRID_N;
      frame(pl); frame(pn);
      clipPlane(pl, () => drawReal(pl, false));
      txt(`noise z (−${GRID_R} to ${GRID_R})`, pn.x, pn.y - 7, c.muted.hex, 12, 'left', 600);
      txt('generated points x = G(z)', pl.x, pl.y - 7, c.muted.hex, 12, 'left', 600);
      const zx = (i: number) => pn.x + ((z[2 * i] + 2.4) / 4.8) * pn.w, zy = (i: number) => pn.y + ((2.4 - z[2 * i + 1]) / 4.8) * pn.h;
      if (f.phase > 0) { ctx.globalAlpha = 0.28; ctx.fillStyle = c.fwd.hex; for (let i = 0; i < N * N; i++) { ctx.beginPath(); ctx.arc(zx(i), zy(i), 1.8, 0, 6.2832); ctx.fill(); } }
      const X = new Float64Array(N * N), Y = new Float64Array(N * N);
      for (let i = 0; i < N * N; i++) {
        const a = zx(i), b = sx(pl, out[2 * i]), a2 = zy(i), b2 = sy(pl, out[2 * i + 1]);
        X[i] = a + f.s * (b - a); Y[i] = a2 + f.s * (b2 - a2);
      }
      const L2: number[] = [];
      for (let rr = 0; rr < N; rr++) { L2.length = 0; for (let cc = 0; cc < N; cc++) L2.push(X[rr * N + cc], Y[rr * N + cc]); line(L2, c.fwd.hex, 1, 0.45); }
      for (let cc = 0; cc < N; cc++) { L2.length = 0; for (let rr = 0; rr < N; rr++) L2.push(X[rr * N + cc], Y[rr * N + cc]); line(L2, c.fwd.hex, 1, 0.45); }
      ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex;
      for (let i = 0; i < N * N; i++) { ctx.beginPath(); ctx.arc(X[i], Y[i], 2.4, 0, 6.2832); ctx.fill(); }
      if (f.phase === 2) {
        let on = 0;
        for (let i = 0; i < N * N; i++) on += nearData(out[2 * i], out[2 * i + 1]) ? 1 : 0;
        txt(`${pct(on / (N * N))} of the grid lands on a moon`, pl.x + pl.w, pl.y + pl.h + 14, c.fwd.hex, 12, 'right', 600);
      }
    }

    function drawGradients(L: ReturnType<typeof layout>, r: GanRun) {
      const p = L.plane, pn = L.panel, f = gradFrame(t), d = demoOf(r), snap = r.snaps[GRAD_SNAP];
      drawMap(p, snap); frame(p);
      clipPlane(p, () => {
        drawReal(p, false);
        const pos = new Float64Array(2 * GRAD_N);
        for (let i = 0; i < 2 * GRAD_N; i++) pos[i] = d.from[i] + f.move * d.arrow[i];
        drawFake(p, pos);
        const a = f.phase === 0 ? 0 : 1 - 0.55 * f.move;
        if (a > 0) for (let i = 0; i < GRAD_N; i++) {
          const x0 = sx(p, d.from[2 * i]), y0 = sy(p, d.from[2 * i + 1]);
          arrow(x0, y0, x0 + f.grow * (sx(p, d.to[2 * i]) - x0), y0 + f.grow * (sy(p, d.to[2 * i + 1]) - y0), c.grad.hex, a);
        }
      });
      txt(`Step ${r.steps[GRAD_SNAP]} of ${r.cfg.steps}`, L.hud.x, L.hud.y, c.text.hex, 12.5, 'left', 600);
      const done = f.phase === 3;
      txt('Mean over the 60 points shown', pn.x, pn.y - 8, c.muted.hex, 11.5, 'left');
      const row = (label: string, before: number, after: number, max: number, y: number, fmt: (v: number) => string) => {
        txt(label, pn.x, y, c.text.hex, 12, 'left', 600);
        const w = pn.w - 70;
        for (const [k, v, col, a] of [[0, before, c.muted.hex, 1], [1, after, c.grad.hex, done ? 1 : 0]] as const) {
          const yy = y + 8 + k * 20;
          if (!a) continue;
          ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.fillRect(pn.x, yy, (v / max) * w, 14);
          txt(fmt(v), pn.x + (v / max) * w + 6, yy + 12, c.text.hex, 11.5, 'left', 600);
          txt(k ? 'after' : 'before', pn.x + pn.w, yy + 12, c.muted.hex, 11, 'right');
        }
      };
      row('D(x), the discriminator’s score', d.dBefore, d.dAfter, 0.7, pn.y + 14, (v) => v.toFixed(3));
      row('length of the push arrow', d.lenBefore, d.lenAfter, 1, pn.y + 14 + 78, (v) => v.toFixed(3));
    }

    function render() {
      k = (cv.clientWidth || Wd) / Wd;
      ctx.setTransform(cv.width / Wd, 0, 0, cv.height / Hh, 0, 0);
      ctx.clearRect(0, 0, Wd, Hh);
      const L = layout();
      if (!data) drawLoading(L.plane);
      else if (init.mode === 'generator') drawGenerator(L, data);
      else if (init.mode === 'gradients') drawGradients(L, data);
      else drawRun(L, data);
      ctx.globalAlpha = 1;
    }

    const loop = createLoop(cv, (dt, now) => {
      if (data) {
        if (t > T + HOLD) t = 0;
        t = advance({ t, playing, speed }, dt, { duration: T, loopTo: T + HOLD }).t;
      }
      render();
      tl.paint(Math.min(t, T));
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = Math.min(t, T); const s = say(); if (s !== narration) narration = s; }
      return playing || !data;
    }, { onView: (inV, first) => { if (inV && first && !reduce) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = Math.min(2, Math.max(1, window.devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; fit(); poke(); });
    ro.observe(cv.parentElement!);
    const offTheme = onThemeChange(() => { c = readColors(C); ui = readVar('--font-ui'); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();
    return () => { loop.destroy(); ro.disconnect(); offTheme(); reduced.removeEventListener('change', onReduce); };
  });

  const stageLabel = $derived({
    duel: 'A plane showing real two-moons data, the generated points, and the discriminator’s score as a background, with a chart of how much of the generated cloud lies on the data.',
    generator: 'A grid of noise points on the left is carried through the generator onto the plane of real data, with its grid lines stretching along.',
    discriminator: 'A plane with the real data, a fixed cloud of generated points and the discriminator’s score as a background, with a chart of its accuracy.',
    objective: 'A plane with the data and generated points, and a chart of the discriminator and generator losses against training step.',
    gradients: 'Generated points with an arrow on each showing the direction that raises the discriminator’s score, then the points moving along the arrows.',
    collapse: 'Eight Gaussian clusters on a ring, the generated points, and bars showing how many generated points lie near each cluster.',
  }[init.mode]);

  const rates = {
    rate: [{ v: 'stable', t: 'Generator rate 0.0003 (step 1)' }, { v: 'unstable', t: 'Generator rate 0.004 (13 times larger)' }],
    ring: [{ v: 'collapse', t: 'Detective rate 0.005, generator 0.0003' }, { v: 'ringOk', t: 'Both rates 0.003' }],
  };
</script>

<FigureCard title={init.title} {legend} aspect="600 / 330" narrowAspect="360 / 540" label={stageLabel}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Timeline"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { playing = false; t = to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if init.choose}
      <div class="ctl">
        <label>{init.choose === 'rate' ? 'Learning rate' : 'Training setting'}
          <select bind:value={cur}>
            {#each rates[init.choose] as o (o.v)}<option value={o.v}>{o.t}</option>{/each}
          </select>
        </label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-body); font-variant-numeric: tabular-nums; min-height: 2.9em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; flex-wrap: wrap; align-items: center; gap: var(--s-2); color: var(--text-muted); max-width: 100%; }
  select { max-width: 100%; font: 0.9rem var(--font-ui); padding: 0.3em 0.5em; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
</style>
