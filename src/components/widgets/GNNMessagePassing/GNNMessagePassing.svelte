<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, reducedMotion, advance } from '../_shared';
  import { createRng } from '../../../lib/math/rng';
  import {
    defaultGraph, nodeFeatures, trace, frameAt, duration, markers as mkMarkers, hops, spread, letter, neighbors, F,
    type Graph, type Agg, type Model, type Phase,
  } from './sim';

  interface Props {
    /** 'gcn' (fixed normalized average, or sum/mean/max) or 'gat' (learned attention on edges). */
    mode?: Model;
    /** Aggregator for mode 'gcn'. */
    agg?: Agg;
    /** Number of message-passing layers (1 to 6). */
    layers?: number;
    /** Add the graph-level readout phase. */
    readout?: boolean;
    /** Show the editing tools (drag, add and remove nodes and edges). */
    editable?: boolean;
    /** Show the model / aggregator / layers controls. */
    showControls?: boolean;
    /** Show the per-node calculation panel. */
    showPanel?: boolean;
    /** Node whose receptive field and calculation are shown (0 = A). */
    select?: number;
    /** Seed for features and weights. */
    seed?: number;
    title?: string;
  }
  let { mode = 'gcn', agg = 'gcn', layers = 2, readout = true, editable = true, showControls = true, showPanel = true, select = 0, seed = 3, title = 'Message passing on a graph' }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ mode, agg, layers, readout, select, seed }));
  let model = $state<Model>(init.mode);
  let aggr = $state<Agg>(init.agg);
  let nl = $state(init.layers);
  let graph = $state.raw<Graph>(defaultGraph(init.seed));
  let sel = $state(init.select);
  let tool = $state<'move' | 'edge'>('move');
  let pending = $state(-1);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let uiPhase = $state<Phase>('send'), uiLayer = $state(0), uiK = $state(0);
  let narration = $state('Watch messages travel along the edges, get combined at each node, and update its features.');

  const cfg = $derived({ model, agg: aggr, layers: nl, seed: init.seed });
  const tr = $derived(trace(graph, cfg));
  const dur = $derived(duration(nl, init.readout));
  const mk = $derived(mkMarkers(nl, init.readout));
  const n = $derived(graph.x.length);
  const dist = $derived(hops(graph, Math.min(sel, n - 1)));
  const S = $derived(Math.min(sel, n - 1));
  const nbr = $derived(neighbors(graph));

  let t = 0, cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, wake = () => {};
  let W = $state(720), Hh = $state(400);
  let drag = -1, moved = false, nextId = 100;
  $effect(() => { playing; speed; nl; model; aggr; graph; sel; wake(); });
  $effect(() => { if (t > dur) { t = 0; uiT = 0; } });

  const fmt = (v: number) => (v < 0 ? '−' : ' ') + Math.abs(v).toFixed(2);
  const vec = (v: number[]) => v.map(fmt).join(' ');

  // ---- graph editing ----
  const setG = (g: Graph) => { graph = g; };
  function toggleEdge(a: number, b: number) {
    if (a === b) return;
    const [p, q] = a < b ? [a, b] : [b, a];
    const has = graph.edges.some(([u, v]) => u === p && v === q);
    setG({ ...graph, edges: has ? graph.edges.filter(([u, v]) => !(u === p && v === q)) : [...graph.edges, [p, q]] });
  }
  function addNode() {
    if (n >= 10) return;
    const r = createRng(nextId * 13 + 1);
    let best: [number, number] = [0.5, 0.5], bd = -1;
    for (let k = 0; k < 40; k++) {
      const c: [number, number] = [r.uniform(0.06, 0.94), r.uniform(0.1, 0.9)];
      const d = Math.min(...graph.x.map((x, i) => Math.hypot(x - c[0], graph.y[i] - c[1])));
      if (d > bd) { bd = d; best = c; }
    }
    setG({ x: [...graph.x, best[0]], y: [...graph.y, best[1]], h: [...graph.h, nodeFeatures(init.seed, nextId++)], edges: [...graph.edges, [S, n] as [number, number]] });
    sel = n;
  }
  function removeNode() {
    if (n <= 2) return;
    const drop = S, re = (i: number) => (i > drop ? i - 1 : i);
    setG({
      x: graph.x.filter((_, i) => i !== drop), y: graph.y.filter((_, i) => i !== drop), h: graph.h.filter((_, i) => i !== drop),
      edges: graph.edges.filter(([a, b]) => a !== drop && b !== drop).map(([a, b]) => [re(a), re(b)] as [number, number]),
    });
    sel = Math.min(sel, n - 2); pending = -1;
  }
  function resetGraph() { setG(defaultGraph(init.seed)); sel = init.select; pending = -1; }
  function activate(i: number) {
    if (tool === 'edge' && editable) {
      if (pending < 0) pending = i; else { toggleEdge(pending, i); pending = -1; }
    } else { pending = -1; }
    sel = i;
  }
  function nudge(e: KeyboardEvent, i: number) {
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const v = d[e.key];
    if (!v || !editable) return;
    e.preventDefault(); e.stopPropagation();
    setG({ ...graph, x: graph.x.map((x, k) => (k === i ? Math.min(0.97, Math.max(0.03, x + v[0] * 0.03)) : x)), y: graph.y.map((y, k) => (k === i ? Math.min(0.95, Math.max(0.05, y + v[1] * 0.03)) : y)) });
  }

  // ---- geometry ----
  const R = () => (W < 540 ? 17 : 21);
  function rects() {
    const ro = init.readout;
    if (!ro) return { g: [0, 0, W, Hh], r: null as null | number[] };
    if (W >= 540) return { g: [0, 0, W * 0.78, Hh], r: [W * 0.8, Hh * 0.3, W * 0.19, Hh * 0.4] };
    return { g: [0, 0, W, Hh * 0.78], r: [W * 0.3, Hh * 0.8, W * 0.4, Hh * 0.17] };
  }
  const pos = (i: number, g = rects().g): [number, number] => {
    const pad = R() + 8;
    return [g[0] + pad + graph.x[i] * (g[2] - 2 * pad), g[1] + pad + graph.y[i] * (g[3] - 2 * pad)];
  };
  const hit = (px: number, py: number) => {
    let best = -1, bd = (R() + 8) ** 2;
    for (let i = 0; i < n; i++) { const [x, y] = pos(i); const d = (x - px) ** 2 + (y - py) ** 2; if (d < bd) { bd = d; best = i; } }
    return best;
  };
  const toLocal = (e: PointerEvent): [number, number] => { const r = cv.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * Hh]; };
  function down(e: PointerEvent) {
    const [px, py] = toLocal(e), i = hit(px, py);
    if (i < 0) { pending = -1; return; }
    activate(i);
    if (tool === 'move' && editable) { drag = i; moved = false; cv.setPointerCapture(e.pointerId); }
  }
  function move(e: PointerEvent) {
    if (drag < 0) return;
    const [px, py] = toLocal(e), g = rects().g, pad = R() + 8;
    const gx = Math.min(1, Math.max(0, (px - g[0] - pad) / (g[2] - 2 * pad))), gy = Math.min(1, Math.max(0, (py - g[1] - pad) / (g[3] - 2 * pad)));
    setG({ ...graph, x: graph.x.map((x, k) => (k === drag ? gx : x)), y: graph.y.map((y, k) => (k === drag ? gy : y)) });
  }
  const up = () => { drag = -1; };

  const PHASE_TEXT: Record<Phase, string> = {
    send: 'Send: every node transforms its features with W and sends the result along each edge (and to itself).',
    aggregate: 'Aggregate: each node combines the messages that arrived.',
    update: 'Update: the combined message passes through tanh and becomes the node’s new features.',
    readout: 'Readout: pool all node features into one vector for the whole graph.',
    hold: 'Done. Drag the timeline to replay, or drag a node to change the graph.',
  };

  onMount(() => {
    const ctx = cv.getContext('2d')!;
    const NAMES = { pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft', s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5' };
    let c = readColors(NAMES), font = readVar('--font-code'), lastUi = 0, lastKey = '';
    const seq = (a: number) => [c.s2, c.s3, c.s4, c.s5][Math.min(3, Math.floor(a * 4))].hex;

    const draw = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), r = cv.getBoundingClientRect();
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, Hh);
      const f = frameAt(t, tr, nl, init.readout), rad = R(), rc = rects();
      const P = Array.from({ length: n }, (_, i) => pos(i, rc.g));
      const layer = tr.layers[f.layer], gat = model === 'gat', sIdx = S;
      const reach = f.phase === 'hold' || f.phase === 'readout' ? nl : f.k;

      // receptive field halos
      for (let i = 0; i < n; i++) if (dist[i] <= reach) {
        ctx.globalAlpha = i === sIdx ? 1 : 0.9; ctx.fillStyle = c.soft.hex;
        ctx.beginPath(); ctx.arc(P[i][0], P[i][1], rad + 7, 0, 6.2832); ctx.fill();
      }
      ctx.globalAlpha = 1;
      // edges: one lane per direction
      const lanes = f.phase === 'send' || f.phase === 'aggregate';
      for (const m of layer.msgs) {
        if (m.from === m.to) continue;
        const [x1, y1] = P[m.from], [x2, y2] = P[m.to], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ox = (-dy / L) * 3, oy = (dx / L) * 3;
        const into = m.to === sIdx, a = gat ? Math.min(1, m.coef) : 0;
        ctx.strokeStyle = gat && (lanes || f.phase === 'update') ? seq(a) : c.rule.hex;
        ctx.lineWidth = gat ? 1 + 4 * a : into && lanes ? 2.5 : 1.6;
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.moveTo(x1 + ox, y1 + oy); ctx.lineTo(x2 + ox, y2 + oy); ctx.stroke();
      }
      // coefficient labels on edges into the selected node
      ctx.font = `600 11px ${font}`; ctx.textAlign = 'center';
      if (lanes) for (const m of layer.msgs) {
        if (m.to !== sIdx || m.from === m.to) continue;
        const [x1, y1] = P[m.from], [x2, y2] = P[m.to], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
        const lx = x1 + dx * 0.62 - (dy / L) * 10, ly = y1 + dy * 0.62 + (dx / L) * 10;
        ctx.lineJoin = 'round'; ctx.lineWidth = 4; ctx.strokeStyle = c.raised.hex; ctx.strokeText(m.coef.toFixed(2), lx, ly + 4);
        ctx.fillStyle = c.text.hex; ctx.fillText(m.coef.toFixed(2), lx, ly + 4);
      }
      // message particles
      if (f.phase === 'send') {
        for (const m of layer.msgs) {
          if (m.from === m.to) continue;
          const [x1, y1] = P[m.from], [x2, y2] = P[m.to], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
          const s = f.sendP, ox = (-dy / L) * 3, oy = (dx / L) * 3;
          ctx.globalAlpha = Math.min(1, s * 8, (1 - s) * 8 + 0.15);
          ctx.fillStyle = c.fwd.hex; ctx.beginPath(); ctx.arc(x1 + dx * s + ox, y1 + dy * s + oy, 3 + 3 * Math.min(1, m.coef), 0, 6.2832); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
      // nodes
      for (let i = 0; i < n; i++) {
        const [x, y] = P[i];
        if (f.aggP > 0) {
          ctx.globalAlpha = Math.sin(f.aggP * Math.PI) * 0.8; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.arc(x, y, rad + 2 + 6 * f.aggP, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
        }
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = i === sIdx ? c.accent.hex : i === pending ? c.fwd.hex : c.muted.hex; ctx.lineWidth = i === sIdx || i === pending ? 3 : 1.5;
        ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.2832); ctx.fill(); ctx.stroke();
        const bw = rad * 0.28, gap = bw * 0.55, x0 = x - (F * bw + (F - 1) * gap) / 2;
        for (let k = 0; k < F; k++) {
          const v = f.feat[i][k], h = v * rad * 0.62;
          ctx.fillStyle = v >= 0 ? c.pos.hex : c.neg.hex;
          ctx.fillRect(x0 + k * (bw + gap), h >= 0 ? y - h : y, bw, Math.abs(h));
        }
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.moveTo(x0 - 2, y); ctx.lineTo(x0 + F * bw + (F - 1) * gap + 2, y); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.font = `${i === sIdx ? 700 : 600} 12px ${font}`; ctx.fillStyle = i === sIdx ? c.accent.hex : c.muted.hex; ctx.textAlign = 'center';
        ctx.fillText(letter(i), x, y - rad - 6);
      }
      // readout box
      if (rc.r) {
        const [bx, by, bw2, bh] = rc.r;
        ctx.globalAlpha = f.readP > 0 || f.phase === 'hold' ? 1 : 0.45;
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.5; ctx.fillStyle = c.raised.hex;
        ctx.beginPath(); ctx.roundRect(bx, by, bw2, bh, 6); ctx.fill(); ctx.stroke();
        ctx.font = `600 11px ${font}`; ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center';
        ctx.fillText('graph vector (mean)', bx + bw2 / 2, by - 6);
        const pv = f.pooled, ws = Math.min(16, bw2 / 6), cx = bx + bw2 / 2, cy = by + bh / 2, mh = bh * 0.38;
        const k0 = f.phase === 'hold' ? 1 : f.readP;
        for (let k = 0; k < F; k++) {
          const h = pv[k] * mh * k0;
          ctx.fillStyle = pv[k] >= 0 ? c.pos.hex : c.neg.hex;
          ctx.fillRect(cx + (k - 1) * ws * 1.4 - ws / 2, h >= 0 ? cy - h : cy, ws, Math.abs(h));
        }
        ctx.globalAlpha = 1;
        if (f.phase === 'readout') {
          const s = Math.min(1, f.readP * 1.4);
          for (let i = 0; i < n; i++) {
            const st = Math.max(0, Math.min(1, s * 1.15 - (i / n) * 0.15));
            ctx.globalAlpha = Math.min(1, st * 6, (1 - st) * 6 + 0.1); ctx.fillStyle = c.fwd.hex;
            ctx.beginPath(); ctx.arc(P[i][0] + (bx + bw2 / 2 - P[i][0]) * st, P[i][1] + (by + bh / 2 - P[i][1]) * st, 4, 0, 6.2832); ctx.fill();
          }
          ctx.globalAlpha = 1;
        }
      }
      tl?.paint(t);
      return f;
    };

    const loop = createLoop(cv, (dt, now) => {
      const nx = advance({ t, playing, speed }, dt, { duration: dur });
      t = nx.t;
      const f = draw();
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t; uiPhase = f.phase; uiLayer = f.layer; uiK = f.k;
        const key = f.phase + f.layer;
        if (key !== lastKey) { lastKey = key; narration = (f.phase === 'hold' || f.phase === 'readout' ? '' : `Layer ${f.layer + 1}. `) + PHASE_TEXT[f.phase]; }
      }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const ro = new ResizeObserver(() => { const r = cv.getBoundingClientRect(); W = r.width; Hh = r.height; wake(); });
    ro.observe(cv);
    const off = onThemeChange(() => { c = readColors(NAMES); font = readVar('--font-code'); wake(); });
    document.fonts.ready.then(() => wake());
    return () => { loop.destroy(); ro.disconnect(); off(); };
  });

  // panel numbers for the selected node at the layer being shown
  const view = $derived.by(() => {
    const l = Math.min(uiLayer, nl - 1), L = tr.layers[l];
    const inc = L.msgs.filter((m) => m.to === S);
    return { l, inc, pre: L.pre[S], out: L.H[S], hin: tr.H[l][S] };
  });
  const coefName = $derived(model === 'gat' ? 'α' : aggr === 'gcn' ? 'c' : aggr === 'mean' ? '1/d̃' : aggr === 'sum' ? '1' : '');
  const reachN = $derived(dist.filter((d) => d <= (uiPhase === 'hold' || uiPhase === 'readout' ? nl : uiK)).length);
  const sp = $derived([tr.H[0], tr.H[Math.min(nl, uiK)]].map(spread));
</script>

<FigureCard
  {title} role="group" aspect="720 / 400" narrowAspect="360 / 470"
  label="A small graph. Colored dots travel along its edges, are combined at each node, and the node's three feature bars change. Nodes can be dragged."
  legend={[{ label: 'message', color: '--viz-forward' }, { label: 'positive feature', color: '--viz-pos' }, { label: 'negative feature', color: '--viz-neg' }, { label: 'receptive field', color: '--accent-soft' }, ...(model === 'gat' ? [{ label: 'attention', color: '--viz-seq-4', shape: 'line' as const }] : [])]}
  asideLabel="Calculation for the selected node"
  aside={showPanel ? panel : undefined}
>
  <canvas bind:this={cv} aria-hidden="true" onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up}></canvas>
  <div class="hits">
    {#each { length: n } as _, i (i)}
      <button type="button" class="hit" aria-pressed={sel === i}
        style:left="{(pos(i)[0] / W) * 100}%" style:top="{(pos(i)[1] / Hh) * 100}%"
        aria-label="Node {letter(i)}, {nbr[i].length} neighbors, features {tr.H[0][i].join(', ')}. {editable ? 'Arrow keys move it.' : ''}"
        onclick={(e) => { if (e.detail === 0) activate(i); }} onkeydown={(e) => nudge(e, i)}></button>
    {/each}
  </div>

  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={dur} markers={mk} label="Timeline of message passing"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= dur ? 0 : to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}

  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next num">
      node {letter(S)} sees {reachN} of {n} nodes after {uiPhase === 'hold' || uiPhase === 'readout' ? nl : uiK} layer{(uiPhase === 'hold' || uiPhase === 'readout' ? nl : uiK) === 1 ? '' : 's'}
      · feature spread {sp[0].toFixed(2)} → {sp[1].toFixed(2)}
    </p>
  {/snippet}

  {#snippet footer()}
    {#if showControls || editable}
      <div class="ctl">
        {#if showControls}
          <fieldset class="seg"><legend class="sr-only">Model</legend>
            <label><input type="radio" name="m-{uid}" value="gcn" bind:group={model} /><span>GCN</span></label>
            <label><input type="radio" name="m-{uid}" value="gat" bind:group={model} /><span>GAT</span></label>
          </fieldset>
          <label>aggregate
            <select bind:value={aggr} disabled={model === 'gat'}>
              <option value="gcn">normalized (GCN)</option><option value="sum">sum</option><option value="mean">mean</option><option value="max">max</option>
            </select>
          </label>
          <label>layers <input type="range" min="1" max="6" step="1" bind:value={nl} /><output class="num">{nl}</output></label>
        {/if}
        {#if editable}
          <fieldset class="seg"><legend class="sr-only">Edit tool</legend>
            <label><input type="radio" name="t-{uid}" value="move" bind:group={tool} onchange={() => (pending = -1)} /><span>Drag nodes</span></label>
            <label><input type="radio" name="t-{uid}" value="edge" bind:group={tool} /><span>Toggle edges</span></label>
          </fieldset>
          <button type="button" class="btn" onclick={addNode} disabled={n >= 10}>Add node</button>
          <button type="button" class="btn" onclick={removeNode} disabled={n <= 2}>Remove {letter(S)}</button>
          <button type="button" class="btn" onclick={resetGraph}>Reset graph</button>
          {#if tool === 'edge'}<span class="hint">{pending >= 0 ? `Now pick the node to link ${letter(pending)} with.` : 'Pick two nodes to add or remove the edge between them.'}</span>{/if}
        {/if}
      </div>
    {/if}
  {/snippet}
</FigureCard>

{#snippet panel()}
  <div class="panel">
    <h4>Node {letter(S)} · layer {view.l + 1}</h4>
    <p class="row lab">features in <span class="n">{vec(view.hin)}</span></p>
    <table>
      <thead><tr><th>from</th><th>{coefName ? coefName + ' ×' : ''} message</th></tr></thead>
      <tbody>
        {#each view.inc as m (m.from)}
          <tr><th scope="row">{letter(m.from)}{m.from === S ? ' (self)' : ''}</th><td class="n">{model === 'gat' || aggr === 'gcn' || aggr === 'mean' ? m.coef.toFixed(2) + ' ×' : ''} <span>{vec(m.vec)}</span></td></tr>
        {/each}
      </tbody>
    </table>
    <p class="row lab">{model === 'gat' ? 'sum' : aggr === 'max' ? 'elementwise max' : aggr === 'mean' ? 'mean' : 'sum'} <span class="n">{vec(view.pre)}</span></p>
    <p class="row lab">tanh → new features <span class="n out">{vec(view.out)}</span></p>
    <p class="hint">
      {#if model === 'gat'}Attention α is a softmax over {letter(S)}'s neighborhood, so the coefficients sum to 1.
      {:else if aggr === 'gcn'}c = 1/√(d̃ᵢ d̃ⱼ), d̃ = degree + 1: {letter(S)} has d̃ = {nbr[S].length + 1}.
      {:else if aggr === 'mean'}Divide by the {nbr[S].length + 1} messages (neighbors and itself).
      {:else if aggr === 'max'}Each feature keeps the largest incoming value.
      {:else}Plain sum: hubs with many neighbors get large values.{/if}
    </p>
  </div>
{/snippet}

<style>
  .hits { position: absolute; inset: 0; pointer-events: none; }
  .hit { position: absolute; width: 2.6rem; height: 2.6rem; transform: translate(-50%, -50%); padding: 0; border: 0; background: transparent; border-radius: 50%; pointer-events: none; }
  .hit:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  canvas { touch-action: none; cursor: grab; }

  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); }

  .panel { font: var(--step--1) / 1.4 var(--font-ui); min-width: 0; }
  .panel h4 { margin: 0 0 var(--s-2); font: 650 1rem var(--font-ui); }
  .panel table { border-collapse: collapse; width: 100%; margin: var(--s-2) 0; }
  .panel th, .panel td { text-align: left; font-weight: 400; padding: 0.12em 0.5em 0.12em 0; white-space: nowrap; }
  .panel thead th { color: var(--text-muted); font-size: 0.72rem; border-bottom: 1px solid var(--rule); }
  .panel tbody th { font: 600 0.8rem var(--font-code); }
  .n { font: 0.78rem var(--font-code); font-variant-numeric: tabular-nums; white-space: pre; }
  .n.out { color: var(--viz-forward); font-weight: 600; }
  .row { margin: 0.4em 0 0; }
  .lab { color: var(--text-muted); }
  .hint { margin: 0.6em 0 0; color: var(--text-muted); font-size: 0.78rem; }

  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; }
  .ctl input[type='range'] { width: 6rem; accent-color: var(--accent); }
  .ctl output { min-width: 1ch; color: var(--text); }
  .ctl .btn { padding: 0.35em 0.8em; }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
