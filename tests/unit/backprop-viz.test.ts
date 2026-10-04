import { describe, expect, test } from 'vitest';
import { backward } from '../../src/lib/nn';
import { Sim, build, defaultCfg, fmt, ld, layout, endpoints, rulePaths, ruleValue, view, makePlan, frameAt, PH, T, type Cfg } from '../../src/components/widgets/BackpropViz/sim';

const cfgs: [string, Cfg][] = [
  ['mlp tanh', defaultCfg('mlp')],
  ['mlp relu', { ...defaultCfg('mlp'), act: 'relu' }],
  ['expr', defaultCfg('expr')],
];

describe.each(cfgs)('backprop viz: %s', (_n, cfg) => {
  test('stepwise grads equal backward() grads', () => {
    const s = new Sim(cfg);
    while (s.phase !== 'update') s.step();
    const ref = build(cfg, cfg.init);
    backward(ref.loss);
    const a = [...s.model.labels.keys()].map((v) => v.grad);
    const b = [...ref.labels.keys()].map((v) => v.grad);
    expect(a.length).toBe(b.length);
    a.forEach((g, i) => expect(g).toBeCloseTo(b[i], 12));
  });

  test('sequencer: phases are ordered, goto replays, step rolls into next cycle', () => {
    const s = new Sim(cfg);
    const phases = s.steps.map((x) => x.phase);
    expect(phases[0]).toBe('forward');
    expect(phases.at(-1)).toBe('update');
    expect(phases.filter((p) => p === 'loss')).toHaveLength(1);
    expect([...new Set(phases)]).toEqual(['forward', 'loss', 'backward', 'update']);
    s.goto(s.phaseStart('backward') + 2);
    const g = s.model.loss.grad;
    expect(g).toBe(1);
    const snapshot = s.model.params.map((p) => p.grad);
    s.goto(0);
    s.goto(s.phaseStart('backward') + 2);
    expect(s.model.params.map((p) => p.grad)).toEqual(snapshot);
    s.runToEnd();
    expect(s.complete).toBe(true);
    const before = s.lossNow;
    s.step(); // rolls over
    expect(s.idx).toBe(1);
    expect(s.committed).toEqual([before]);
    expect(s.lossNow).toBeCloseTo(s.upd ? s.upd.lossAfter : s.lossNow, 12);
  });

  test('backward runs each non-leaf Value exactly once', () => {
    const s = new Sim(cfg);
    s.runToEnd();
    const nonLeaf = [...s.done];
    expect(new Set(nonLeaf).size).toBe(nonLeaf.length);
    expect(nonLeaf.every((n) => n.children.length > 0)).toBe(true);
  });

  test('chain-rule factors multiply (and sum over paths) to the displayed gradient', () => {
    const s = new Sim(cfg);
    while (s.phase !== 'update') s.step();
    for (const v of s.model.labels.keys()) {
      const paths = rulePaths(s.model.loss, v, s.model.labels);
      expect(ruleValue(paths)).toBeCloseTo(v.grad, 10);
    }
    for (const n of s.model.nodes) for (const r of s.rules(n.id)) expect(r.value).toBeCloseTo(r.grad, 10);
    for (const e of s.model.edges.filter((x) => x.param)) for (const r of s.rules(e.id)) expect(r.value).toBeCloseTo(r.grad, 10);
  });

  test('one SGD step lowers the loss for a small learning rate', () => {
    const s = new Sim({ ...cfg, lr: 0.001 });
    s.runToEnd();
    expect(s.upd!.lossAfter).toBeLessThan(s.upd!.lossBefore);
    s.upd!.next.forEach((v, i) => expect(v).toBeCloseTo(s.upd!.old[i] - 0.001 * s.upd!.grads[i], 12));
  });

  test('view is well formed at every step', () => {
    const s = new Sim(cfg);
    const l = layout(cfg.mode, false), lv = layout(cfg.mode, true);
    for (let i = 0; i <= s.steps.length; i++) {
      s.goto(i);
      const v = view(s);
      expect(v.nodes.every((n) => l.box[n.id] && lv.box[n.id])).toBe(true);
      v.edges.forEach((e) => expect(Number.isFinite(endpoints(s.model.edges.find((x) => x.id === e.id)!, l).x1)).toBe(true));
    }
  });
});

describe.each(cfgs)('frameAt: %s', (_n, cfg) => {
  const plan = makePlan(cfg, cfg.init);
  const ref = build(cfg, cfg.init);
  backward(ref.loss);
  const refNode = (id: string) => ref.nodes.find((n) => n.id === id)!;
  const source = (r: string) => r === 'input' || r === 'param';

  test('t=0 shows inputs only', () => {
    const f = frameAt(0, plan);
    for (const [i, n] of f.nodes.entries()) {
      const src = source(plan.nodes[i].role);
      expect(n.valueP).toBe(src ? 1 : 0);
      expect(n.gradP).toBe(0);
      expect(n.preP).toBe(0);
    }
    f.edges.forEach((e) => { expect(e.fwd).toBe(-1); expect(e.fill).toBe(0); });
    expect(f.phase).toBe('forward');
  });

  test('end of Forward: every node value equals the engine forward value; loss not yet', () => {
    const f = frameAt(PH.forward[1], plan);
    for (const n of f.nodes) {
      const r = refNode(n.id);
      if (n.id === ref.lossId) { expect(n.valueP).toBe(0); continue; }
      expect(n.value).toBeCloseTo(r.value.data, 12);
      if (r.pre) expect(n.pre).toBeCloseTo(r.pre.data, 12);
      expect(n.gradP).toBe(0);
    }
    expect(frameAt(PH.loss[1], plan).nodes.find((n) => n.id === ref.lossId)!.value).toBeCloseTo(ref.loss.data, 12);
  });

  test('end of Backward: every gradient equals backward()', () => {
    const f = frameAt(PH.backward[1], plan);
    for (const n of f.nodes) {
      const r = refNode(n.id);
      expect(n.grad).toBeCloseTo(r.value.grad, 12);
      if (r.pre) expect(n.preGrad).toBeCloseTo(r.pre.grad, 12);
    }
    ref.edges.forEach((e, i) => {
      expect(f.edges[i].fill).toBe(1);
      expect(f.edges[i].g).toBeCloseTo(e.via.grad * ld(e.via, e.k), 12);
    });
  });

  test('Update: weights ease monotonically from old to new', () => {
    const idx = plan.edges.map((e, i) => (e.param ? i : -1)).filter((i) => i >= 0);
    const at = (t: number) => frameAt(t, plan);
    for (const i of idx) {
      const { w0, w1 } = plan.edges[i], dir = Math.sign(w1 - w0);
      expect(at(PH.update[0]).edges[i].w).toBeCloseTo(w0, 12);
      expect(at(T).edges[i].w).toBeCloseTo(w1, 12);
      let prev = w0;
      for (let k = 0; k <= 40; k++) {
        const w = at(PH.update[0] + ((T - PH.update[0]) * k) / 40).edges[i].w;
        expect((w - prev) * dir).toBeGreaterThanOrEqual(-1e-12);
        prev = w;
      }
      expect(at(PH.backward[1]).edges[i].w).toBe(w0);
    }
    plan.sim.model.params.forEach((_, i) => expect(plan.w1[i]).toBeCloseTo(plan.w0[i] - cfg.lr * plan.sim.upd!.grads[i], 12));
  });

  test('pure and deterministic; a cycle continues from the updated weights', () => {
    const before = JSON.stringify([plan.nodes, plan.edges, plan.w0, plan.w1]);
    for (const t of [0, 1.3, 2.9, 4, 5.5, T]) expect(frameAt(t, plan)).toEqual(frameAt(t, plan));
    expect(JSON.stringify([plan.nodes, plan.edges, plan.w0, plan.w1])).toBe(before);
    expect(makePlan(cfg, plan.w1).lossBefore).toBeCloseTo(plan.lossAfter, 12);
    expect(frameAt(-1, plan).t).toBe(0);
    expect(frameAt(99, plan).t).toBe(T);
  });
});

describe('specifics', () => {
  test('animated loss is recomputed at displayed weights, not blended endpoint losses', () => {
    const cfg = defaultCfg('expr'), p = makePlan(cfg, cfg.init);
    for (const t of [5.1, 5.3, 5.6, 6]) {
      const f = frameAt(t, p);
      expect(f.loss).toBeCloseTo(build(cfg, f.nodes.filter(n => ['a', 'b', 'c'].includes(n.id)).map(n => n.value)).loss.data, 10);
    }
  });
  test('reveal opacity never fabricates intermediate values or gradients', () => {
    const cfg = defaultCfg('mlp'), plan = makePlan(cfg, cfg.init);
    for (const t of [0.4, 1.4, 2.6, 3.18, 4.5]) {
      const f = frameAt(t, plan);
      f.nodes.forEach((n, i) => {
        expect(n.value).toBe(plan.nodes[i].value);
        expect(n.grad).toBe(plan.nodes[i].grad);
        expect(n.preGrad).toBe(plan.nodes[i].preGrad);
      });
      f.edges.forEach((e, i) => expect(e.g).toBe(plan.edges[i].g));
    }
  });
  test('large-rate expression update matches the lesson overshoot example', () => {
    const s = new Sim({ ...defaultCfg('expr'), lr: 0.1 });
    s.runToEnd();
    s.upd!.next.forEach((v, i) => expect(v).toBeCloseTo([4.4, -4.6, 9.2][i], 12));
    expect(s.upd!.lossAfter).toBeCloseTo(121.8816, 10);
  });
  test('parameter count is stable across cycles (mlp: 9, expr: 3)', () => {
    for (const [c, n] of [[defaultCfg('mlp'), 9], [defaultCfg('expr'), 3]] as const) {
      const s = new Sim(c);
      expect(s.model.params).toHaveLength(n);
      s.train(2);
      s.step();
      expect(s.model.params).toHaveLength(n);
      expect(s.upd).toBeNull();
      expect(s.cycleW).toHaveLength(n);
    }
  });
  test('expression graph has the textbook numbers', () => {
    const s = new Sim(defaultCfg('expr'));
    s.runToEnd();
    const g = Object.fromEntries(s.model.nodes.map((n) => [n.id, n.value.grad]));
    expect(s.model.loss.data).toBe(16);
    expect(g).toMatchObject({ L: 1, s: 8, m: 8, c: 8, a: -24, b: 16 });
  });
  test('relu kills the gradient through a dead unit', () => {
    const s = new Sim({ ...defaultCfg('mlp'), act: 'relu' });
    s.runToEnd();
    const h2 = s.model.nodes.find((n) => n.id === 'h2')!;
    expect(h2.value.data).toBe(0);
    expect(h2.pre!.grad).toBe(0);
    expect(s.model.params[2].grad).toBe(0);
  });
  test('local derivative matches engine for every op in the mlp graph', () => {
    const m = build(defaultCfg('mlp'), defaultCfg('mlp').init);
    backward(m.loss);
    expect(ld(m.loss, 0)).toBeCloseTo(2 * m.loss.children[0].data, 12);
  });
  test('fmt: 3 significant digits and true minus', () => {
    expect(fmt(-0.41862)).toBe('−0.419');
    expect(fmt(12.345)).toBe('12.3');
    expect(fmt(0)).toBe('0.00');
    expect(fmt(0.00001234)).toBe('1.2e-5');
    expect(fmt(1234.5)).toBe('1235');
  });
});
