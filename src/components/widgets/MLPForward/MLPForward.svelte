<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance as clock } from '../_shared';
  import { sub } from '../BackpropViz/sim';
  import {
    f2, size, columns, frameAt, markersOf, restAt, matmulPlan, forwardPlan, paramsPlan, dwPlan, mmLayout, fwLayout, paLayout, dwLayout,
    DATA, GX, GY, DOMX, DOMY, type Mode, type Plan, type MatmulPlan, type ForwardPlan, type ParamsPlan, type DWPlan,
    type MatmulFrame, type ForwardFrame, type ParamsFrame, type DWFrame, type Pt,
  } from './sim';

  interface Props {
    /** 'matmul' | 'forward' | 'params' | 'depth-width' */
    mode?: Mode;
    title?: string;
    /** params mode: starting number of hidden layers (1-4) and units per layer (1-8). */
    hidden?: number;
    width?: number;
  }
  let { mode = 'forward', title = '', hidden: hidden0 = 2, width: width0 = 4 }: Props = $props();

  const M = untrack(() => mode);
  const TITLES: Record<Mode, string> = {
    matmul: 'One layer, two views: neurons and a matrix',
    forward: 'A forward pass through a 2-2-2-1 network',
    params: 'Counting the parameters of a network',
    'depth-width': 'Wide or deep: the same job, three shapes',
  };
  const LABELS: Record<Mode, string> = {
    matmul: 'A layer of three neurons drawn as a diagram and as the matrix equation z = W x + b, h = tanh(z), computed one row at a time.',
    forward: 'A network with 2 inputs, two hidden layers of 2 units and 1 output. A value pulse moves layer by layer and every pre-activation and activation is shown.',
    params: 'A layered network whose weights and biases are counted layer by layer, with sliders for depth and width.',
    'depth-width': 'Three network shapes with similar parameter counts and the decision boundary each one learns on the same two-moons data.',
  };
  const LEGENDS: Record<Mode, { label: string; color: string }[]> = {
    matmul: [{ label: 'value', color: '--viz-forward' }, { label: 'positive weight', color: '--viz-pos' }, { label: 'negative weight', color: '--viz-neg' }],
    forward: [{ label: 'value', color: '--viz-forward' }, { label: 'positive weight', color: '--viz-pos' }, { label: 'negative weight', color: '--viz-neg' }],
    params: [{ label: 'weights', color: '--viz-forward' }, { label: 'biases', color: '--accent' }],
    'depth-width': [{ label: 'class A', color: '--viz-pos' }, { label: 'class B', color: '--viz-neg' }],
  };

  let hidden = $state(untrack(() => Math.min(4, Math.max(1, Math.round(hidden0)))));
  let width = $state(untrack(() => Math.min(8, Math.max(1, Math.round(width0)))));
  const fixed: Plan | null = M === 'matmul' ? matmulPlan() : M === 'forward' ? forwardPlan() : M === 'depth-width' ? dwPlan() : null;
  const plan = $derived<Plan>(fixed ?? paramsPlan(hidden, width));
  const T = $derived(plan.T);
  const markers = $derived(markersOf(plan));

  let playing = $state(false);
  let speed = $state(1);
  let uiT = $state(0);
  let narration = $state('');
  let t = 0; // truth for the canvas: a plain variable
  let reduce = false;
  let cv: HTMLCanvasElement;
  let tl: ReturnType<typeof Timeline>;
  let poke = () => {};
  $effect(() => { playing; speed; poke(); });

  // ---------- narration: one sentence per phase, built from the same plan numbers the canvas draws ----------
  function say(fr: ReturnType<typeof frameAt>, P: Plan): string {
    if (P.mode === 'matmul') {
      const f = fr as MatmulFrame, j = f.active;
      if (f.all) return `All three rows at once: h = tanh(Wx + b) = (${P.h.map(f2).join(', ')}). One matrix-vector product replaces three separate dot products.`;
      if (f.sub === 'dot') return `Neuron ${j + 1} is row ${j + 1} of W. Its two weighted input terms arrive in sequence; z shows their running sum.`;
      if (f.sub === 'bias') return `Neuron ${j + 1}: add its bias to the weighted sum. The changing z is the pre-activation.`;
      return `Neuron ${j + 1}: apply tanh to the completed sum. Its activation h settles to ${f2(P.h[j])}.`;
    }
    if (P.mode === 'forward') {
      const l = (fr as ForwardFrame).layer, h = (k: number) => P.tr.h[k].map(f2).join(', ');
      if (l === 0) return `The inputs are x = (${h(0)}). Nothing has been computed yet.`;
      if (l === 1 || l === 2) return `Layer ${l}: input signals arrive along the weights, the sums z settle, then tanh produces the activations h. Each card shows the calculation as it completes.`;
      if (l === 3) return `Output layer: combine the second hidden layer's activations with the output weights and bias. This output is linear, so the completed sum z is also ŷ.`;
      return `Result: ŷ = ${h(3)}. Each weight was used once, in layer order; the output of one layer was the input of the next.`;
    }
    if (P.mode === 'params') {
      const a = (fr as ParamsFrame).active;
      if (a < 0) return `Total: ${P.layers.map((c) => c.total).join(' + ')} = ${P.total} parameters for ${P.sizes.join(' → ')}.`;
      const c = P.layers[a];
      return `Counting layer ${a + 1}: ${c.nIn} inputs per unit and ${c.nOut} units. Each lit edge adds one weight; each filled square adds one bias. The table shows the count so far.`;
    }
    const A = P.archs[(fr as DWFrame).idx];
    return `${A.name}, ${A.sizes.join(' → ')}: ${A.params} parameters, ${Math.round(A.acc * DATA.y.length)} of ${DATA.y.length} training points on the correct side.`;
  }
  const keyOf = (fr: ReturnType<typeof frameAt>, P: Plan): string => {
    if (P.mode === 'matmul') { const f = fr as MatmulFrame; return `${f.active}:${f.sub}`; }
    if (P.mode === 'forward') return `${(fr as ForwardFrame).layer}`;
    if (P.mode === 'params') return `${P.sizes.join('-')}:${(fr as ParamsFrame).active}`;
    return `${(fr as DWFrame).idx}`;
  };
  narration = untrack(() => say(frameAt(0, plan), plan));

  onMount(() => {
    const reduced = prefersReducedMotion();
    reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    let narrow = false, lastUi = 0, lastKey = '';
    const NAMES = {
      pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', grad: '--viz-gradient',
      text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft',
    };
    let c = readColors(NAMES), fonts = { code: readVar('--font-code'), body: readVar('--font-body') };
    const dpr = () => Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const mono = (px: number, wt = 500) => `${wt} ${px}px ${fonts.code}`;
    const ital = (px: number) => `italic ${px}px ${fonts.body}`;

    // decision-boundary thumbnails, painted once per theme into small offscreen canvases
    const bufs: HTMLCanvasElement[] = [];
    function paintBufs() {
      if (plan.mode !== 'depth-width') return;
      plan.archs.forEach((A, i) => {
        const b = (bufs[i] ??= Object.assign(document.createElement('canvas'), { width: GX, height: GY }));
        const g = b.getContext('2d')!, im = g.createImageData(GX, GY), bg = c.raised.rgb;
        for (let k = 0; k < GX * GY; k++) {
          const p = A.grid[k], s = 0.16 + 0.5 * Math.abs(2 * p - 1);
          for (let q = 0; q < 3; q++) im.data[k * 4 + q] = bg[q] * (1 - s) + (c.pos.rgb[q] * (1 - p) + c.neg.rgb[q] * p) * s;
          im.data[k * 4 + 3] = 255;
        }
        g.putImageData(im, 0, 0);
      });
    }

    // ---------- drawing primitives (logical units) ----------
    const line = (a: Pt, b: Pt, col: string, w: number, al: number) => {
      ctx.globalAlpha = al; ctx.strokeStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    };
    const disc = (p: Pt, r: number, col: string, al: number) => {
      ctx.globalAlpha = al; ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };
    const tx = (s: string, x: number, y: number, col: string, font: string, align: CanvasTextAlign = 'center', al = 1) => {
      ctx.globalAlpha = al; ctx.font = font; ctx.textAlign = align; ctx.fillStyle = col; ctx.fillText(s, x, y);
    };
    const halo = (s: string, x: number, y: number, col: string, font: string, al = 1) => {
      ctx.globalAlpha = al; ctx.font = font; ctx.textAlign = 'center'; ctx.lineJoin = 'round'; ctx.lineWidth = 4;
      ctx.strokeStyle = c.raised.hex; ctx.strokeText(s, x, y + 4); ctx.fillStyle = col; ctx.fillText(s, x, y + 4);
    };
    const lerp = (a: Pt, b: Pt, u: number): Pt => ({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u });
    const signCol = (w: number) => (w >= 0 ? c.pos.hex : c.neg.hex);
    const pulse = (a: Pt, b: Pt, u: number, w: number) => {
      if (reduce || u < 0 || u >= 1) return;
      line(lerp(a, b, Math.max(0, u - 0.2)), lerp(a, b, u), c.fwd.hex, w + 2, 0.55);
      disc(lerp(a, b, u), 5, c.fwd.hex, 1);
    };
    const bracket = (xl: number, xr: number, y0: number, y1: number) => {
      ctx.globalAlpha = 1; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(xl + 5, y0); ctx.lineTo(xl, y0); ctx.lineTo(xl, y1); ctx.lineTo(xl + 5, y1);
      ctx.moveTo(xr - 5, y0); ctx.lineTo(xr, y0); ctx.lineTo(xr, y1); ctx.lineTo(xr - 5, y1);
      ctx.stroke();
    };
    const rrect = (x: number, y: number, w: number, h: number, col: string, al: number, r = 6) => {
      ctx.globalAlpha = al; ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill();
    };

    // ---------- matmul ----------
    function drawMatmul(fr: MatmulFrame, P: MatmulPlan) {
      const G = mmLayout(narrow), X = G.X, R = 20, fs = narrow ? 12 : 14;
      // edges
      P.W.forEach((row, j) => row.forEach((w, i) => {
        const on = fr.active === j || fr.all, a = G.inP[i], b = G.hidP[j], lw = 1 + 4.2 * Math.abs(w) / 0.9;
        line(a, b, signCol(w), lw, on ? 0.9 : 0.3);
        pulse(a, b, fr.rows[j].pulse[i], lw);
        if (fr.active === j) halo(f2(w), lerp(a, b, i ? 0.7 : 0.3).x, lerp(a, b, i ? 0.7 : 0.3).y, signCol(w), mono(12.5, 600));
      }));
      // neurons
      G.inP.forEach((p, i) => {
        disc(p, R, c.raised.hex, 1);
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 6.2832); ctx.stroke();
        tx(f2(P.x[i]), p.x, p.y + 4.5, c.text.hex, mono(13));
        tx(`x${sub(i + 1)}`, p.x, p.y - R - 7, c.muted.hex, ital(13));
      });
      G.hidP.forEach((p, j) => {
        const r = fr.rows[j];
        disc(p, R, c.raised.hex, 1);
        ctx.globalAlpha = 0.14 * r.hA; ctx.fillStyle = c.fwd.hex; ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 6.2832); ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 6.2832); ctx.stroke();
        ctx.globalAlpha = 0.8 * r.hA; ctx.strokeStyle = c.fwd.hex; ctx.stroke();
        if (fr.active === j) {
          ctx.globalAlpha = 1; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.arc(p.x, p.y, R + 5, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
        }
        tx(r.hA ? f2(r.hShown) : '·', p.x, p.y + 4.5, r.hA ? c.fwd.hex : c.muted.hex, mono(13, 600));
        tx(`h${sub(j + 1)}`, p.x, p.y - R - 7, c.muted.hex, ital(13));
        if (!narrow && (fr.active === j || fr.all)) line({ x: p.x + R + 8, y: p.y }, { x: X.W1 - 30, y: G.rowY[j] }, c.accent.hex, 1.2, 0.55);
      });
      // the matrix, one row per neuron
      const mid = (X.W1 + X.h) / 2;
      tx('h = tanh(W x + b)', mid, G.titleY + (narrow ? 0 : 0), c.text.hex, ital(narrow ? 17 : 18));
      const hd = ital(narrow ? 12.5 : 13.5);
      tx('W', (X.W1 + X.W2) / 2, G.headY, c.muted.hex, hd); tx('x', X.x, G.headY, c.muted.hex, hd); tx('b', X.b, G.headY, c.muted.hex, hd);
      tx('z', X.z, G.headY, c.muted.hex, hd); tx('h', X.h, G.headY, c.muted.hex, hd); tx('tanh', X.arr, G.headY, c.muted.hex, hd);
      G.rowY.forEach((y, j) => {
        const on = fr.active === j ? 1 : fr.all ? 0.7 : 0;
        if (on) rrect(X.W1 - 28, y - 21, X.h + 28 - (X.W1 - 28), 42, c.soft.hex, on, 8);
      });
      const top = G.rowY[0] - 24, bot = G.rowY[2] + 24, xt = G.xY[0] - 20, xb = G.xY[1] + 20, cw = narrow ? 20 : 24;
      bracket(X.W1 - cw - 4, X.W2 + cw + 4, top, bot); bracket(X.x - cw, X.x + cw, xt, xb);
      bracket(X.b - cw, X.b + cw, top, bot); bracket(X.z - cw, X.z + cw, top, bot); bracket(X.h - cw, X.h + cw, top, bot);
      const op = c.muted.hex, of = mono(narrow ? 13 : 15, 400);
      tx('×', X.mul, G.rowY[1] + 5, op, of);
      G.rowY.forEach((y, j) => {
        const r = fr.rows[j];
        tx('+', X.plus, y + 5, op, of); tx('=', X.eq, y + 5, op, of); tx('→', X.arr, y + 5, op, of);
        P.W[j].forEach((w, i) => {
          const xx = i ? X.W2 : X.W1;
          if (r.pulse[i] > 0 && r.pulse[i] < 1) { rrect(xx - 22, y - 14, 44, 28, c.fwd.hex, 0.16, 5); rrect(X.x - 22, G.xY[i] - 14, 44, 28, c.fwd.hex, 0.16, 5); }
          tx(f2(w), xx, y + 5, signCol(w), mono(fs, 600));
        });
        tx(f2(P.b[j]), X.b, y + 5, r.bA || fr.all ? c.text.hex : c.muted.hex, mono(fs, 500), 'center', r.bA || fr.all ? 1 : 0.7);
        tx(r.zA ? f2(r.zShown) : '·', X.z, y + 5, r.zA ? c.fwd.hex : c.muted.hex, mono(fs, 600));
        tx(r.hA ? f2(r.hShown) : '·', X.h, y + 5, r.hA ? c.fwd.hex : c.muted.hex, mono(fs, 700));
      });
      G.xY.forEach((y, i) => tx(f2(P.x[i]), X.x, y + 5, c.text.hex, mono(fs, 500)));
    }

    // ---------- forward ----------
    const NAME = ['x', 'h¹', 'h²', 'ŷ'];
    function drawForward(fr: ForwardFrame, P: ForwardPlan) {
      const G = fwLayout(narrow), L = P.net.W.length;
      const end = (l: number, n: number, out: boolean): Pt => {
        const p = G.pos[l][n], h = G.hw[l];
        return G.vertical ? { x: p.x, y: p.y + (out ? G.hh : -G.hh) } : { x: p.x + (out ? h : -h), y: p.y };
      };
      for (let l = 1; l <= L; l++) {
        const running = fr.layer === l;
        P.net.W[l - 1].forEach((row, j) => row.forEach((w, i) => {
          const a = end(l - 1, i, true), b = end(l, j, false), lw = 0.8 + 4.6 * Math.abs(w) / P.maxW;
          line(a, b, signCol(w), lw, running ? 0.95 : 0.4);
          pulse(a, b, fr.pulse[l - 1][j][i], lw);
          if (fr.labelA[l - 1] > 0.01) { const m = lerp(a, b, i ? 0.7 : 0.3); halo(f2(w), m.x, m.y, signCol(w), mono(12.5, 600), fr.labelA[l - 1]); }
        }));
      }
      G.pos.forEach((col, l) => col.forEach((p, n) => {
        const nd = fr.nodes[l][n], hw = G.hw[l], hh = l === 0 ? 17 : G.hh, active = fr.layer === l || (fr.layer === 4 && l === L);
        const shown = nd.hA;
        rrect(p.x - hw, p.y - hh, 2 * hw, 2 * hh, c.raised.hex, 1);
        rrect(p.x - hw, p.y - hh, 2 * hw, 2 * hh, c.fwd.hex, 0.09 * shown);
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.roundRect(p.x - hw, p.y - hh, 2 * hw, 2 * hh, 6); ctx.stroke();
        ctx.globalAlpha = 0.65 * shown; ctx.strokeStyle = c.fwd.hex; ctx.stroke();
        if (active) {
          ctx.globalAlpha = 1; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.roundRect(p.x - hw - 5, p.y - hh - 5, 2 * hw + 10, 2 * hh + 10, 9); ctx.stroke(); ctx.setLineDash([]);
        }
        const nm = l === 0 ? `x${sub(n + 1)}` : l === L ? 'ŷ' : `${NAME[l]}${sub(n + 1)}`;
        if (l === 0) { tx(nm, p.x - hw + 8, p.y + 4.5, c.muted.hex, ital(13), 'left'); tx(f2(nd.h), p.x + hw - 8, p.y + 4.5, c.text.hex, mono(13.5, 600), 'right'); return; }
        tx(`${nm} · b = ${f2(P.net.b[l - 1][n])}`, p.x, p.y - hh - 8, c.muted.hex, ital(12.5));
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.x - hw + 7, p.y + 1); ctx.lineTo(p.x + hw - 7, p.y + 1); ctx.stroke();
        tx('z', p.x - hw + 8, p.y - 7, c.muted.hex, ital(13), 'left');
        tx(nd.zA ? f2(nd.z) : '—', p.x + hw - 8, p.y - 7, nd.zA ? c.fwd.hex : c.muted.hex, mono(13.5, 600), 'right', nd.zA ? 1 : 0.6);
        tx(l === L ? 'ŷ' : 'h', p.x - hw + 8, p.y + 16, c.muted.hex, ital(13), 'left');
        tx(nd.hA ? f2(nd.h) : '—', p.x + hw - 8, p.y + 16, nd.hA ? c.fwd.hex : c.muted.hex, mono(13.5, 700), 'right', nd.hA ? 1 : 0.6);
      }));
    }

    // ---------- params ----------
    function drawParams(fr: ParamsFrame, P: ParamsPlan) {
      const G = paLayout(narrow), pts = columns(P.sizes, G.net), L = P.layers.length, ls = G.list;
      if (fr.active >= 0) {
        const a = pts[fr.active][0].x - 22, b = pts[fr.active + 1][0].x + 22;
        rrect(a, G.net.y - 14, b - a, G.net.h + 28, c.soft.hex, 0.85, 10);
      }
      P.layers.forEach((cn, l) => {
        const f = fr.layers[l];
        pts[l + 1].forEach((to, j) => pts[l].forEach((from, i) => {
          const lit = j * cn.nIn + i < f.w;
          line(from, to, lit ? c.fwd.hex : c.rule.hex, lit ? 1.7 : 1, lit ? 0.85 : 0.7);
        }));
      });
      pts.forEach((col, l) => col.forEach((p, j) => {
        disc(p, 9, c.raised.hex, 1);
        ctx.globalAlpha = 1; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, 6.2832); ctx.stroke();
        if (l > 0) {
          const lit = j < fr.layers[l - 1].b;
          ctx.globalAlpha = 1; ctx.fillStyle = lit ? c.accent.hex : c.raised.hex; ctx.strokeStyle = lit ? c.accent.hex : c.muted.hex; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.rect(p.x + 7, p.y - 14, 7, 7); ctx.fill(); ctx.stroke();
        }
      }));
      pts.forEach((col, l) => tx(`${P.sizes[l]}`, col[0].x, G.net.y + G.net.h + 22, c.muted.hex, mono(13), 'center'));
      tx(narrow ? 'units per layer' : 'units per layer (squares: biases)', G.net.x + G.net.w / 2, G.net.y + G.net.h + 42, c.muted.hex, ital(12.5));
      // breakdown table
      const hf = ital(12.5), row = (r: number) => ls.y + 30 + r * ls.rowH;
      tx('layer', ls.cLabel, ls.y, c.muted.hex, hf, 'left'); tx(narrow ? 'weights' : 'weights = in × out', ls.cW, ls.y, c.muted.hex, hf, 'right');
      tx('biases', ls.cB, ls.y, c.muted.hex, hf, 'right'); tx('layer total', ls.cS, ls.y, c.muted.hex, hf, 'right');
      P.layers.forEach((cn, l) => {
        const f = fr.layers[l], y = row(l);
        if (fr.active === l) rrect(ls.cLabel - 8, y - 20, ls.cS - ls.cLabel + 16, 30, c.soft.hex, 1, 6);
        tx(`layer ${l + 1}`, ls.cLabel, y, c.text.hex, ital(13.5), 'left');
        tx(`${cn.nIn} × ${cn.nOut} = ${f.wP > 0 ? f.w : '·'}`, ls.cW, y, f.wP > 0 ? c.fwd.hex : c.muted.hex, mono(13.5, 600), 'right');
        tx(`+ ${f.bP > 0 ? f.b : '·'}`, ls.cB, y, f.bP > 0 ? c.accent.hex : c.muted.hex, mono(13.5, 600), 'right');
        tx(f.wP > 0 ? `= ${f.w + f.b}` : '', ls.cS, y, c.text.hex, mono(13.5, 500), 'right');
      });
      const yt = row(L) + 6;
      ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ls.cLabel - 8, yt - 24); ctx.lineTo(ls.cS + 8, yt - 24); ctx.stroke();
      tx('total parameters', ls.cLabel, yt, c.text.hex, ital(14), 'left');
      tx(`${fr.total}`, ls.cS, yt + 2, fr.active < 0 ? c.accent.hex : c.text.hex, mono(20, 700), 'right');
    }

    // ---------- depth-width ----------
    function drawDW(fr: DWFrame, P: DWPlan) {
      const G = dwLayout(narrow), A = P.archs[fr.idx], pts = columns(A.sizes, G.net);
      const mx = Math.max(...A.net.W.flat(2).map(Math.abs));
      tx(A.sizes.join(' → '), G.net.x + G.net.w / 2, G.titleY, c.text.hex, mono(16, 700), 'center', fr.fade);
      A.net.W.forEach((m, l) => m.forEach((row, j) => row.forEach((w, i) => line(pts[l][i], pts[l + 1][j], c.muted.hex, 0.4 + 3.2 * Math.abs(w) / mx, 0.55 * fr.fade))));
      pts.forEach((col) => col.forEach((p) => disc(p, 8, c.raised.hex, fr.fade)));
      ctx.globalAlpha = fr.fade; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2;
      pts.forEach((col) => col.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, 6.2832); ctx.stroke(); }));
      const yb = G.net.y + G.net.h + (narrow ? 26 : 30);
      tx(`${fr.paramsShown} parameters`, G.net.x + G.net.w / 2, yb, c.accent.hex, mono(17, 700), 'center');
      tx('line width = |weight|', G.net.x + G.net.w / 2, yb + 22, c.muted.hex, ital(12.5));
      P.archs.forEach((a, i) => {
        const r = G.thumbs[i], al = fr.thumbA[i], on = i === fr.idx;
        ctx.globalAlpha = al; ctx.imageSmoothingEnabled = true; ctx.drawImage(bufs[i], r.x, r.y, r.w, r.h);
        DATA.x.forEach((p, k) => {
          const px = r.x + (p[0] / DOMX + 1) / 2 * r.w, py = r.y + (1 - p[1] / DOMY) / 2 * r.h;
          disc({ x: px, y: py }, 2.9, DATA.y[k] ? c.neg.hex : c.pos.hex, al);
        });
        ctx.globalAlpha = 1; ctx.strokeStyle = on ? c.accent.hex : c.rule.hex; ctx.lineWidth = on ? 2.4 : 1.2;
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        const x = G.capX[0], cap = on ? 1 : 0.55;
        tx(a.sizes.join(' → '), x, r.y + 22, c.text.hex, mono(14, 700), 'left', cap);
        tx(`${a.params} parameters`, x, r.y + 44, c.text.hex, mono(13, 500), 'left', cap);
        tx(`${Math.round(a.acc * DATA.y.length)} of ${DATA.y.length} points right`, x, r.y + 64, c.muted.hex, mono(12.5, 400), 'left', cap);
        tx(a.name, x, r.y + 84, c.muted.hex, ital(12.5), 'left', cap);
      });
    }

    function render() {
      const tt = Math.min(t, plan.T);
      const fr = frameAt(tt, plan), S = size(narrow), k = cv.width / S.w;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.clearRect(0, 0, S.w, S.h);
      ctx.lineCap = 'round';
      if (plan.mode === 'matmul') drawMatmul(fr as MatmulFrame, plan);
      else if (plan.mode === 'forward') drawForward(fr as ForwardFrame, plan);
      else if (plan.mode === 'params') drawParams(fr as ParamsFrame, plan);
      else drawDW(fr as DWFrame, plan);
      ctx.globalAlpha = 1;
      tl.paint(tt);
      const key = keyOf(fr, plan);
      if (key !== lastKey) {
        lastKey = key;
        narration = say(fr, plan);
        // Synchronize phase markers on the exact boundary, between throttled time updates.
        uiT = tt;
      }
    }

    // ---------- loop & environment ----------
    if (reduce) t = restAt(plan);
    const loop = createLoop(cv, (dt, now) => {
      t = clock({ t, playing, speed }, dt, { duration: plan.T }).t;
      render();
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = Math.min(t, plan.T); }
      return playing;
    }, { onView: (inView, first) => { if (inView && first && !reduce) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = dpr(), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      narrow = r.height > r.width;
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(() => { fit(); render(); });
    ro.observe(cv.parentElement!);
    const offTheme = onThemeChange(() => { c = readColors(NAMES); fonts = { code: readVar('--font-code'), body: readVar('--font-body') }; paintBufs(); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    paintBufs();
    fit();
    return () => { loop.destroy(); ro.disconnect(); offTheme(); reduced.removeEventListener('change', onReduce); };
  });

  /** A slider moved: restart the count from the top so the new totals are visible. */
  function restart() { t = 0; playing = !reduce; poke(); }
</script>

<FigureCard
  title={title || TITLES[M]} legend={LEGENDS[M]} label={LABELS[M]}
  aspect="720 / 400" narrowAspect="360 / 620"
  footer={M === 'params' ? foot : undefined}
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>

  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Timeline"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}

  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
</FigureCard>

{#snippet foot()}
  <div class="ctl">
    <label>hidden layers
      <input type="range" min="1" max="4" step="1" bind:value={hidden} oninput={restart} />
      <output class="num">{hidden}</output>
    </label>
    <label>units per layer
      <input type="range" min="1" max="8" step="1" bind:value={width} oninput={restart} />
      <output class="num">{width}</output>
    </label>
  </div>
{/snippet}

<style>
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.9em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 8rem; accent-color: var(--accent); }
  .ctl output { min-width: 1.2rem; color: var(--text); }
</style>
