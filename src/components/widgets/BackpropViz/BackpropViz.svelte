<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance as clock } from '../_shared';
  import { defaultCfg, layout, endpoints, fmt, PHASES, T, PH, frameAt, makePlan, phaseAt, type Mode, type Phase, type Cfg, type Plan, type Frame, type NodePlan } from './sim';

  interface Props {
    /** 'mlp': the 2-3-1 network. 'expr': L = (a·b + c)². */
    mode?: Mode;
    /** Card title. */
    title?: string;
    /** Where the timeline starts (default: the start of `loopPhase`, else of Forward). */
    initialPhase?: Phase;
    /** Loop only this phase (weights stay fixed). Default: loop whole cycles and keep training. */
    loopPhase?: Phase;
    showPanel?: boolean;
    /** Node or weight id to open in the chain-rule panel (e.g. 'h2', 'w11', 'v3', 'a'). */
    select?: string;
  }
  let { mode = 'mlp', title = '', initialPhase, loopPhase, showPanel = true, select = '' }: Props = $props();

  const MAX_CYCLES = 40, HOLD = 0.7;
  const LEGEND = [
    { label: 'value', color: '--viz-forward' }, { label: 'gradient', color: '--viz-gradient' },
    { label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' },
  ];
  const uid = $props.id();
  const init = untrack(() => ({
    mode, loop: loopPhase, select,
    title: title || (mode === 'mlp' ? 'Training a 2-3-1 network' : 'The graph of L = (a·b + c)²'),
    start: PH[initialPhase ?? loopPhase ?? 'forward'][0],
  }));
  const [a0, b0] = init.loop ? PH[init.loop] : [0, T];
  const hi0 = init.loop ? b0 + HOLD : T;

  let cfg = $state<Cfg>(defaultCfg(init.mode));
  const snap = (): Cfg => $state.snapshot(cfg) as Cfg;
  const valid = () => [...cfg.x, cfg.target, ...cfg.init, cfg.lr].every((n) => Number.isFinite(n));
  let plan = $state.raw<Plan>(makePlan(snap(), cfg.init));
  let playing = $state(false);
  let speed = $state(1);
  let uiT = $state(init.start);
  let sel = $state(init.select);
  let vertical = $state(false);
  // readouts, refreshed at most ~10x/s from the frame loop
  let phase = $state<Phase>(phaseAt(init.start));
  let cycleNo = $state(1);
  let lossText = $state('');
  let activeId = $state('');
  let reached = $state(false);
  let narration = $state('Watch one training cycle: forward, loss, backward, update.');

  // per-frame data lives in plain variables: only the canvas reads it
  let t = init.start;
  let committed: number[] = [];
  let cycleW = [...cfg.init];
  let hoverId = '';
  let hiId = '';
  let route: { nodes: Set<string>; edges: Set<string> } | null = null;
  let seek: { from: number; to: number; t0: number; dur: number } | null = null;
  let veil: { t0: number; dur: number; swap: (() => void) | null } | null = null;
  let poke = () => {};

  let cv: HTMLCanvasElement;
  let spark: HTMLCanvasElement | undefined = $state();
  let tl: ReturnType<typeof Timeline>;
  const markers = PHASES.map((p) => ({ at: PH[p][0], label: p }));

  const L = $derived(layout(init.mode, vertical));
  const edgeGeo = $derived(
    Object.fromEntries(plan.sim.model.edges.map((e) => {
      const p = endpoints(e, L);
      const hidden = Math.max(0, Number(e.to.slice(1)) - 1);
      const f = init.mode !== 'mlp' || e.from[0] !== 'x' ? 0.5
        : L.vertical ? (e.from === 'x1' ? 0.14 + 0.18 * hidden : 0.6 + 0.17 * hidden) : 0.6;
      return [e.id, { ...p, lx: p.x1 + (p.x2 - p.x1) * f, ly: p.y1 + (p.y2 - p.y1) * f }];
    })),
  );
  const panelId = $derived(sel || (playing && phase === 'backward' ? activeId : ''));
  const rules = $derived(panelId ? plan.sim.rules(panelId) : []);
  const pIdx = $derived(panelId ? plan.sim.paramIndex(panelId) : -1);
  const numsOn = $derived(phase === 'backward' || phase === 'update');
  const lrMax = init.mode === 'mlp' ? 1 : 0.1;
  const lrStep = init.mode === 'mlp' ? 0.01 : 0.005;

  const pname = (i: number) => plan.sim.model.labels.get(plan.sim.model.params[i]) ?? '';
  const nodeTitle = (n: NodePlan) => (n.hasPre ? `${n.label} = ${plan.cfg.act}(${n.preLabel})` : (n.formula ?? n.label));
  const what = (id: string) => {
    const n = plan.nodes.find((x) => x.id === id);
    if (n) return n.role === 'input' ? 'input' : n.role === 'param' ? 'parameter' : n.role === 'loss' ? 'loss' : n.role === 'out' ? 'output' : n.role === 'hidden' ? 'hidden unit' : 'intermediate value';
    const e = plan.edges.find((x) => x.id === id);
    return e ? `weight, ${e.from} to ${e.to}` : '';
  };
  const label = (id: string) => plan.nodes.find((x) => x.id === id)?.label ?? plan.edges.find((x) => x.id === id)?.sym ?? '';
  const say = (ph: Phase): string => {
    const p = plan;
    if (ph === 'forward') return init.mode === 'mlp' ? `Forward pass: each hidden unit sums its weighted inputs, applies ${p.cfg.act}, and feeds the output.` : 'Forward pass: m = a·b, then s = m + c.';
    if (ph === 'loss') return init.mode === 'mlp' ? `Loss: L = (ŷ − t)² = (${fmt(p.nodes.find((n) => n.id === 'yhat')!.value)} − ${fmt(p.cfg.target)})² = ${fmt(p.lossBefore)}` : `Loss: L = s² = ${fmt(p.lossBefore)}`;
    if (ph === 'backward') return 'Backward pass: each node multiplies its local derivative by the gradient from above and hands the result down.';
    return `Update: loss ${fmt(p.lossBefore)} to ${fmt(p.lossAfter)} with learning rate ${p.cfg.lr}`;
  };

  // ---- user actions ----
  function onSeek(to: number, smooth: boolean) {
    seek = null;
    if (smooth) glide(to); else { t = to; poke(); }
  }
  function glide(to: number) {
    playing = false;
    seek = { from: Math.min(t, T), to, t0: performance.now(), dur: 550 };
    poke();
  }
  function dip() { veil = { t0: performance.now() - 175, dur: 500, swap: null }; }
  /** Inputs or activation changed: recompute from the first cycle, keep the playhead. */
  function rebuild() {
    if (!valid()) return;
    committed = []; cycleW = [...cfg.init];
    plan = makePlan(snap(), cycleW);
    cycleNo = 1; dip(); poke();
  }
  function replan() { if (valid()) { plan = makePlan(snap(), cycleW); poke(); } }
  function reset() { rebuild(); t = init.start; seek = null; poke(); }
  const pick = (id: string) => { sel = sel === id ? '' : id; if (sel) { playing = false; seek = null; } poke(); };
  const hover = (id: string) => { hoverId = id; poke(); };

  $effect(() => {
    const id = panelId, p = plan;
    route = id ? p.sim.route(id) : null;
    hiId = id;
    poke();
  });
  $effect(() => { if (playing) seek = null; speed; poke(); });

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const sctx = spark ? spark.getContext('2d') : null;
    let lastRead = 0, va = 1, base = 1;
    if (reduce) t = init.loop ? b0 : PH.update[0];

    const NAMES = {
      pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', grad: '--viz-gradient',
      text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised',
      accent: '--accent', soft: '--accent-soft',
    };
    const readFonts = () => ({ code: readVar('--font-code'), body: readVar('--font-body') });
    let c = readColors(NAMES), fonts = readFonts();
    const dpr = () => Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const mono = (px: number, wt = 600) => `${wt} ${px}px ${fonts.code}`;

    // ---- timeline control ----
    function restart() {
      committed = []; cycleW = [...cfg.init];
      plan = makePlan(snap(), cycleW);
      t = 0;
    }
    function nextCycle(now: number) {
      committed.push(plan.lossBefore);
      const done = committed.length >= MAX_CYCLES || !(plan.lossAfter < 1e6) || plan.lossAfter < 1e-5 || Math.abs(plan.lossAfter - plan.lossBefore) < 1e-9;
      if (done && !veil) veil = { t0: now, dur: 1000, swap: restart };
      cycleW = plan.w1;
      plan = makePlan(valid() ? snap() : plan.cfg, cycleW);
    }
    function advance(dt: number, now: number) {
      if (seek) {
        const e = Math.min(1, (now - seek.t0) / seek.dur);
        t = e >= 1 ? seek.to : seek.from + (seek.to - seek.from) * e * e * (3 - 2 * e);
        if (e >= 1) {
          const end = seek.to >= T;
          seek = null;
          if (end) { nextCycle(now); t = 0; }
        }
        return;
      }
      const n = clock({ t, playing, speed }, dt, { duration: T, loopFrom: a0, loopTo: hi0 });
      if (n.t < t && !init.loop) nextCycle(now);
      t = n.t;
    }

    // ---- drawing (logical units of the layout; scaled to the canvas) ----
    const line = (x1: number, y1: number, x2: number, y2: number, col: string, w: number, a: number) => {
      ctx.globalAlpha = va * a; ctx.strokeStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    };
    const dot = (x: number, y: number, r: number, col: string, a: number) => {
      ctx.globalAlpha = va * a; ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };
    const edgeLabel = (s: string, x: number, y: number, col: string, a: number) => {
      if (a <= 0.01 || !s) return;
      ctx.globalAlpha = va * a; ctx.font = mono(12.5); ctx.textAlign = 'center';
      ctx.lineJoin = 'round'; ctx.lineWidth = 4; ctx.strokeStyle = c.raised.hex; ctx.strokeText(s, x, y + 4);
      ctx.fillStyle = col; ctx.fillText(s, x, y + 4);
    };
    const num = (s: string, x: number, y: number, col: string, a: number, align: CanvasTextAlign) => {
      ctx.textAlign = align; ctx.font = mono(14.5);
      if (a < 1) { ctx.globalAlpha = base * (1 - a) * 0.55; ctx.fillStyle = c.muted.hex; ctx.fillText('—', x, y); }
      if (a > 0) { ctx.globalAlpha = base * a; ctx.fillStyle = col; ctx.fillText(s, x, y); }
    };
    const tag = (s: string, x: number, y: number, col: string, align: CanvasTextAlign = 'left') => {
      ctx.globalAlpha = base; ctx.fillStyle = col; ctx.font = `italic 13px ${fonts.body}`; ctx.textAlign = align; ctx.fillText(s, x, y);
    };

    function drawEdges(fr: Frame) {
      ctx.lineCap = 'round';
      fr.edges.forEach((e, i) => {
        const ep = plan.edges[i], g = edgeGeo[e.id];
        const on = route?.edges.has(e.id) ?? false;
        const dim = route && !on ? 0.32 : 1;
        const w = ep.param ? 0.8 + 4.6 * Math.abs(e.w) / plan.maxW : 1.6;
        if (on || hiId === e.id || hoverId === e.id) line(g.x1, g.y1, g.x2, g.y2, c.soft.hex, w + 8, 0.9);
        const col = !ep.param ? c.muted.hex : e.w > 0 ? c.pos.hex : e.w < 0 ? c.neg.hex : c.muted.hex;
        line(g.x1, g.y1, g.x2, g.y2, col, w, (ep.param ? 0.85 : 0.5) * dim);
        if (!reduce && e.fwd >= 0) {
          const a = Math.max(0, e.fwd - 0.22);
          line(g.x1 + (g.x2 - g.x1) * a, g.y1 + (g.y2 - g.y1) * a, g.x1 + (g.x2 - g.x1) * e.fwd, g.y1 + (g.y2 - g.y1) * e.fwd, c.fwd.hex, w + 2, 0.55 * dim);
          dot(g.x1 + (g.x2 - g.x1) * e.fwd, g.y1 + (g.y2 - g.y1) * e.fwd, 5, c.fwd.hex, dim);
        }
        if (e.fill > 0) {
          const mag = Math.min(1, Math.abs(ep.g) / plan.gradScale);
          const hx = g.x2 + (g.x1 - g.x2) * e.fill, hy = g.y2 + (g.y1 - g.y2) * e.fill;
          line(g.x2, g.y2, hx, hy, c.grad.hex, 1.4 + 6.5 * Math.sqrt(mag), 0.9 * e.alpha * dim);
          if (!reduce && e.fill < 1) dot(hx, hy, 5, c.grad.hex, dim);
        }
      });
      fr.edges.forEach((e, i) => {
        const ep = plan.edges[i], g = edgeGeo[e.id];
        if (!ep.param && init.mode !== 'expr') return;
        const gs = e.fill * e.alpha;
        let s: string, a: number, col: string;
        if (gs > 0.5) { s = fmt(e.g); a = 2 * gs - 1; col = c.grad.hex; }
        else if (ep.param) { s = fmt(e.w); a = 1 - 2 * gs; col = c.text.hex; }
        else { s = fmt(e.local); a = (1 - 2 * gs) * e.localA; col = c.muted.hex; }
        if (hoverId === e.id && ep.param) { s = `${gs > 0.5 ? '∂L/∂' : ''}${ep.sym} = ${s}`; a = 1; }
        edgeLabel(s, g.lx, g.ly, col, a * (route && !route.edges.has(e.id) ? 0.32 : 1));
      });
    }

    function drawNodes(fr: Frame) {
      fr.nodes.forEach((n, i) => {
        const np = plan.nodes[i], b = L.box[n.id];
        const source = np.role === 'input' || np.role === 'param';
        const dim = route && !route.nodes.has(n.id) ? 0.32 : 1;
        const shown = source ? 1 : n.valueP * n.alpha;
        const side = L.vertical && init.mode === 'mlp';
        ctx.save();
        ctx.translate(b.x, b.y);
        base = va * dim;
        if (fr.active === n.id && fr.phase !== 'update') {
          ctx.globalAlpha = base; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.roundRect(-b.hw - 5, -b.hh - 5, 2 * b.hw + 10, 2 * b.hh + 10, 9); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.beginPath(); ctx.roundRect(-b.hw, -b.hh, 2 * b.hw, 2 * b.hh, 6);
        ctx.globalAlpha = base; ctx.fillStyle = c.raised.hex; ctx.fill();
        ctx.globalAlpha = base * 0.07 * shown; ctx.fillStyle = c.fwd.hex; ctx.fill();
        ctx.globalAlpha = base; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.globalAlpha = base * 0.65 * shown; ctx.strokeStyle = c.fwd.hex; ctx.stroke();
        if (hiId === n.id) { ctx.globalAlpha = base; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.4; ctx.stroke(); }
        tag(side ? np.label : nodeTitle(np), side ? -b.hw : 0, -b.hh - 8, c.muted.hex, side ? 'left' : 'center');
        ctx.globalAlpha = base; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(-b.hw + 6, np.hasPre ? 4 : 3); ctx.lineTo(b.hw - 6, np.hasPre ? 4 : 3); ctx.stroke();
        const ga = (p: number) => Math.min(1, p * 5) * n.alpha, va2 = (p: number) => (source ? 1 : Math.min(1, p * 5) * n.alpha);
        if (np.hasPre) {
          tag('z', -b.hw + 8, -24, c.muted.hex); tag('h', -b.hw + 8, -7, c.muted.hex);
          tag('∂z', -b.hw + 8, 21, c.grad.hex); tag('∂h', -b.hw + 8, 38, c.grad.hex);
          num(fmt(n.pre), b.hw - 8, -24, c.fwd.hex, va2(n.preP), 'right');
          num(fmt(n.value), b.hw - 8, -7, c.fwd.hex, va2(n.valueP), 'right');
          num(fmt(n.preGrad), b.hw - 8, 21, c.grad.hex, ga(n.preGradP), 'right');
          num(fmt(n.grad), b.hw - 8, 38, c.grad.hex, ga(n.gradP), 'right');
        } else {
          num(fmt(n.value), 0, -6, c.fwd.hex, va2(n.valueP), 'center');
          num(`∂ ${fmt(n.grad)}`, 0, 18, c.grad.hex, ga(n.gradP), 'center');
        }
        if (n.id === 'loss' && init.mode === 'mlp') {
          ctx.globalAlpha = base; ctx.fillStyle = c.muted.hex; ctx.font = mono(12.5, 400); ctx.textAlign = 'center';
          ctx.fillText(`t = ${fmt(plan.cfg.target)}`, 0, b.hh + 18);
        }
        ctx.restore();
      });
    }

    function drawSpark(fr: Frame) {
      if (!spark || !sctx) return;
      const W = spark.width, H = spark.height, d = dpr(), pad = 3 * d;
      sctx.clearRect(0, 0, W, H);
      sctx.strokeStyle = c.rule.hex; sctx.lineWidth = d;
      sctx.beginPath(); sctx.moveTo(0, H - d / 2); sctx.lineTo(W, H - d / 2); sctx.stroke();
      const top = Math.log((committed[0] ?? plan.lossBefore) + 1e-4), lo = Math.log(1e-4);
      const span = Math.max(20, committed.length + 2); // fixed x scale, so the line grows instead of rescaling
      const x = (i: number) => pad + (i / span) * (W - 2 * pad);
      const y = (v: number) => pad + (1 - Math.min(1, Math.max(0, (Math.log(v + 1e-4) - lo) / (top - lo || 1)))) * (H - 2 * pad - d);
      const k = committed.length;
      const pts: [number, number][] = [...committed, plan.lossBefore].map((v, i) => [x(i), y(v)]);
      let tip = pts[pts.length - 1];
      if (fr.upd > 0) { tip = [x(k + fr.upd), y(fr.loss)]; pts.push(tip); }
      sctx.strokeStyle = c.text.hex; sctx.lineWidth = 1.5 * d; sctx.lineJoin = 'round';
      sctx.beginPath(); pts.forEach(([px, py], i) => (i ? sctx.lineTo(px, py) : sctx.moveTo(px, py))); sctx.stroke();
      sctx.fillStyle = c.grad.hex; sctx.beginPath(); sctx.arc(tip[0], tip[1], 2.6 * d, 0, 6.2832); sctx.fill();
    }

    function veilAlpha(now: number) {
      if (!veil) return 1;
      const e = (now - veil.t0) / veil.dur;
      if (veil.swap && e >= 0.5) { veil.swap(); veil.swap = null; }
      if (e >= 1) { veil = null; return 1; }
      return 0.25 + 0.75 * Math.abs(1 - 2 * e);
    }

    function render(now: number) {
      va = veilAlpha(now);
      // Phase-only looping limits autoplay, not the user's scrub range.
      const tt = Math.min(t, playing && init.loop ? b0 : T);
      const fr = frameAt(tt, plan);
      const k = cv.width / L.w;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      drawEdges(fr);
      drawNodes(fr);
      ctx.globalAlpha = 1;
      drawSpark(fr);
      // readouts: cheap direct writes every frame, Svelte state at most ~10x/s
      tl.paint(tt);
      if (fr.phase !== phase) { phase = fr.phase; narration = say(phase); }
      lossText = fmt(fr.loss);
      if (now - lastRead > 100 || !(playing || seek || veil)) {
        lastRead = now;
        activeId = fr.active;
        cycleNo = committed.length + 1;
        uiT = tt;
        const pid = panelId, ni = fr.nodes.findIndex((n) => n.id === pid);
        reached = ni >= 0 ? fr.nodes[ni].gradP >= 1 : (fr.edges.find((e) => e.id === pid)?.fill ?? 0) >= 1;
      }
    }

    function frame(dt: number, now: number) {
      advance(dt, now);
      render(now);
      return playing || !!seek || !!veil;
    }
    const loop = createLoop(cv.closest('[data-figure]') ?? cv, frame, { onView: (inView, first) => { if (inView && first && !reduce) playing = true; } });
    poke = loop.wake;

    // ---- environment ----
    const fit = () => {
      const d = dpr();
      for (const el of [cv, spark]) {
        if (!el) continue;
        const r = el.getBoundingClientRect();
        const w = Math.round(r.width * d), h = Math.round(r.height * d);
        if (w && h && (el.width !== w || el.height !== h)) { el.width = w; el.height = h; }
      }
    };
    const ro = new ResizeObserver(([e]) => {
      vertical = e.contentRect.width < 540;
      fit();
      render(performance.now());
    });
    ro.observe(cv.parentElement!);
    const offTheme = onThemeChange(() => { c = readColors(NAMES); fonts = readFonts(); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    narration = say(phase);
    fit();

    return () => {
      loop.destroy(); ro.disconnect(); offTheme();
      reduced.removeEventListener('change', onReduce);
    };
  });
</script>

{#snippet frac(a: string, b: string)}<span class="frac"><span>∂{a}</span><span>∂{b}</span></span>{/snippet}

<FigureCard
  title={init.title} legend={LEGEND} role="group"
  aspect={init.mode === 'mlp' ? '720 / 400' : '720 / 320'} narrowAspect={init.mode === 'mlp' ? '360 / 620' : '360 / 500'}
  label={init.mode === 'mlp' ? 'Network graph: 2 inputs, 3 hidden units, 1 output, loss. Signals flow left to right, then gradients flow back.' : 'Expression graph for L = (a times b plus c) squared. Values flow left to right, then gradients flow back.'}
  asideLabel="Chain rule for the selected item"
  aside={showPanel ? panel : undefined}
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  <div class="hits">
    {#each plan.nodes as n (n.id)}
      {@const b = L.box[n.id]}
      <button type="button" class="hit" aria-pressed={sel === n.id}
        style:left="{((b.x - b.hw) / L.w) * 100}%" style:top="{((b.y - b.hh) / L.h) * 100}%"
        style:width="{((2 * b.hw) / L.w) * 100}%" style:height="{((2 * b.hh) / L.h) * 100}%"
        aria-label="{n.label}, {what(n.id)}: value {fmt(n.value)}, gradient {fmt(n.grad)}"
        onclick={() => pick(n.id)} onpointerenter={() => hover(n.id)} onpointerleave={() => hover('')}
        onfocus={() => hover(n.id)} onblur={() => hover('')}></button>
    {/each}
    {#each plan.edges.filter((e) => e.param) as e (e.id)}
      {@const g = edgeGeo[e.id]}
      <button type="button" class="hit lab" aria-pressed={sel === e.id}
        style:left="{(g.lx / L.w) * 100}%" style:top="{(g.ly / L.h) * 100}%"
        aria-label="Weight {e.sym} from {e.from} to {e.to}: {fmt(e.w0)}, gradient {fmt(e.g)}"
        onclick={() => pick(e.id)} onpointerenter={() => hover(e.id)} onpointerleave={() => hover('')}
        onfocus={() => hover(e.id)} onblur={() => hover('')}></button>
    {/each}
  </div>

  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Timeline of one training cycle"
      {onSeek} onStep={(to) => glide(to)} onReset={reset} />
  {/snippet}

  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next num">cycle {cycleNo}{init.loop ? '' : ` of ${MAX_CYCLES}`} · loss at displayed weights {lossText}</p>
    {#if phase === 'update'}<p class="next">Cached forward values and gradients fade while the weights move.</p>{/if}
  {/snippet}

  {#snippet footer()}
    <div class="ctl">
    {#if init.mode === 'mlp'}
      <label>x<sub>1</sub> <input type="number" step="0.1" bind:value={cfg.x[0]} oninput={rebuild} /></label>
      <label>x<sub>2</sub> <input type="number" step="0.1" bind:value={cfg.x[1]} oninput={rebuild} /></label>
      <label>target t <input type="number" step="0.1" bind:value={cfg.target} oninput={rebuild} /></label>
      <fieldset class="seg">
        <legend class="sr-only">Hidden activation</legend>
        <label><input type="radio" name="act-{uid}" value="tanh" bind:group={cfg.act} onchange={rebuild} /><span>tanh</span></label>
        <label><input type="radio" name="act-{uid}" value="relu" bind:group={cfg.act} onchange={rebuild} /><span>ReLU</span></label>
      </fieldset>
    {:else}
      <label>a <input type="number" step="0.5" bind:value={cfg.init[0]} oninput={rebuild} /></label>
      <label>b <input type="number" step="0.5" bind:value={cfg.init[1]} oninput={rebuild} /></label>
      <label>c <input type="number" step="0.5" bind:value={cfg.init[2]} oninput={rebuild} /></label>
    {/if}
    <label class="lr">learning rate η
      <input type="range" min={lrStep} max={lrMax} step={lrStep} bind:value={cfg.lr} oninput={replan} />
      <output class="num">{cfg.lr}</output>
    </label>
    {#if !init.loop}
      <div class="hist" aria-label="Loss history, log scale">
        <canvas bind:this={spark} aria-hidden="true"></canvas>
      </div>
    {/if}
    </div>
  {/snippet}
</FigureCard>

{#snippet panel()}
  <div class="panel">
    {#if rules.length}
      <header>
        <h4 class="sel-title">{label(panelId)}</h4>
        <span class="what">{what(panelId)}</span>
        {#if sel}<button type="button" class="x" onclick={() => (sel = '')} aria-label="Clear selection">×</button>{/if}
      </header>
      {#each rules as r (r.sym)}
        <section class="rule-block">
          {#each r.paths as f, pi (pi)}
            <div class="path">
              <div class="row sym">
                {#if pi === 0}{@render frac('L', r.sym)}<span class="eq">=</span>{:else}<span class="eq plus">+</span>{/if}
                {#if f.length}
                  {#each f as x, i (i)}{#if i}<span class="dot">·</span>{/if}{@render frac(x.a, x.b)}{/each}
                {:else}<span class="n">1</span>{/if}
              </div>
              {#if numsOn}
                <div class="row nums"><span class="eq">{pi === 0 ? '=' : '+'}</span>{#each f as x, i (i)}<span class="n">({fmt(x.value)})</span>{:else}<span class="n">1</span>{/each}</div>
              {/if}
            </div>
          {/each}
          {#if numsOn}
            <div class="row res">
              <span class="eq">=</span><span class="n big">{fmt(r.value)}</span>
              <span class="note">{reached ? 'the completed gradient shown on the graph' : 'precomputed final gradient; the wave has not fully revealed it yet'}</span>
            </div>
            {#if r.paths.length === 1 && r.paths[0].length > 1}
              {@const f = r.paths[0]}
              {@const up = f.slice(0, -1).reduce((a, x) => a * x.value, 1)}
              <p class="reuse">
                Reuse: upstream {@render frac('L', f.at(-1)!.a)} = <span class="n">{fmt(up)}</span> (already computed),
                local {@render frac(f.at(-1)!.a, f.at(-1)!.b)} = <span class="n">{fmt(f.at(-1)!.value)}</span>.
              </p>
            {/if}
          {:else}
            <p class="hint">The numbers appear once the backward pass starts.</p>
          {/if}
        </section>
      {/each}
      {#if phase === 'update' && pIdx >= 0}
        <p class="reuse upd">
          Update: {pname(pIdx)} ← {pname(pIdx)} − η · ∂L/∂{pname(pIdx)} =
          <span class="n">{fmt(plan.w0[pIdx])} − {plan.cfg.lr} × ({fmt(plan.sim.upd!.grads[pIdx])}) = {fmt(plan.w1[pIdx])}</span>
        </p>
      {/if}
    {:else}
      <p class="hint first">Click any node or weight. This panel writes out its chain rule with the numbers from this pass, and the path from the loss lights up on the graph. While the gradient wave runs, it follows the wave.</p>
      {#if phase === 'update'}
        <table class="updt">
          <caption>Update, η = {plan.cfg.lr}</caption>
          <tbody>
            {#each plan.w0 as o, i (i)}
              <tr><th scope="row">{pname(i)}</th><td class="num">{fmt(o)}</td><td>→</td><td class="num">{fmt(plan.w1[i])}</td></tr>
            {/each}
          </tbody>
        </table>
      {/if}
    {/if}
  </div>
{/snippet}

<style>
  .hits { position: absolute; inset: 0; }
  .hit { position: absolute; padding: 0; border: 0; background: transparent; cursor: pointer; border-radius: 6px; }
  .hit.lab { width: 3.4rem; height: 1.7rem; transform: translate(-50%, -50%); }
  .hit:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  .panel {
    font-family: var(--font-body); font-size: 1rem; line-height: 1.4; min-width: 0;
  }
  .panel header { display: flex; align-items: baseline; gap: var(--s-2); margin-bottom: var(--s-2); }
  .sel-title { font: 650 1.05rem var(--font-body); font-style: italic; }
  .what { font: var(--step--1) var(--font-ui); color: var(--text-muted); }
  .x { margin-left: auto; background: none; border: 0; cursor: pointer; font-size: 1.2rem; line-height: 1; color: var(--text-muted); padding: 0 0.2em; }
  .hint { color: var(--text-muted); font: var(--step--1) / 1.5 var(--font-ui); margin: 0.4em 0 0; }
  .hint.first { margin-top: 0; }
  .rule-block + .rule-block { margin-top: var(--s-3); padding-top: var(--s-3); border-top: 1px solid var(--rule); }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.15em 0.2em; margin-top: 0.3em; }
  .eq, .dot { color: var(--text-muted); padding: 0 0.15em; }
  .n { font: 0.86rem var(--font-code); font-variant-numeric: tabular-nums; }
  .n.big { font-size: 1.05rem; font-weight: 600; color: var(--viz-gradient); }
  .nums .n { padding-right: 0.1em; }
  .note { font: 0.78rem var(--font-ui); color: var(--text-muted); margin-left: var(--s-2); }
  .reuse { margin: 0.6em 0 0; font-size: 0.92rem; color: var(--text-muted); }
  .reuse .n { color: var(--text); }
  .reuse.upd { border-top: 1px solid var(--rule); padding-top: var(--s-2); margin-top: var(--s-3); }
  .frac { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; line-height: 1.15; font-style: italic; font-size: 0.95em; margin: 0 0.1em; }
  .frac > span:first-child { border-bottom: 1px solid currentColor; padding: 0 0.15em; }
  .updt { font: var(--step--1) var(--font-ui); border-collapse: collapse; margin-top: var(--s-3); }
  .updt caption { text-align: left; color: var(--text-muted); padding-bottom: 0.3em; }
  .updt th, .updt td { padding: 0.1em 0.6em 0.1em 0; text-align: left; font-weight: 400; font-family: var(--font-code); font-size: 0.8rem; }
  .updt th { font-style: normal; }

  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); }

  .ctl {
    display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5);
    padding: var(--s-3) var(--s-4);
  }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='number'] {
    width: 4.6rem; font: 0.85rem var(--font-code); font-variant-numeric: tabular-nums; padding: 0.3em 0.4em;
    color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1);
  }
  .ctl input[type='range'] { width: 7.5rem; accent-color: var(--accent); }
  .lr output { min-width: 2.8rem; color: var(--text); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .hist { margin-left: auto; width: 9rem; height: 2.4rem; }
  .hist canvas { display: block; width: 100%; height: 100%; }
</style>
