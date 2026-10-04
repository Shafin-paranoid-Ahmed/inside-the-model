import { describe, expect, it } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import { Value, backward } from '../../src/lib/nn/value';
import {
  N, NA, START, GOAL, PIT, WALL, idx, parseCell, makeGrid, transitions, step, rollout, randomPolicy, returns, coords,
  valueIteration, optimal, policyEvaluation, policyValue, qFromV, deterministic, greedy, qLearning, softmaxPolicy, gridReinforce, actorCritic,
  makeBandit, gradientSamples, columnStats, exactGradient, banditPG, compareStrategies,
  paramCount, initNet, forward, regressionGrad, tdLossGrad, tdTarget, ReplayBuffer, trainDQN, type Transition,
  clippedSurrogate, klCategorical, gae, makeBatch, ppoEpochs, surrogateGrad, trainPPO,
} from '../../src/lib/rl';

const g = makeGrid();

describe('gridworld', () => {
  it('lays out the 4x3 world like the book', () => {
    expect([START, WALL, PIT, GOAL]).toEqual([idx(1, 1), idx(2, 2), idx(4, 2), idx(4, 3)]);
    expect(parseCell('3,1', -1)).toBe(2);
    expect(parseCell('2,2', -1)).toBe(WALL); // the wall parses; callers decide whether it is usable
    expect(parseCell('9,9', 4)).toBe(4);
    expect(g.states).toHaveLength(9);
  });

  it('transition probabilities sum to 1 for every state and action', () => {
    for (const slip of [0, 0.2, 0.5]) {
      const gg = makeGrid({ slip });
      for (const s of gg.states) for (let a = 0; a < NA; a++) {
        const p = transitions(gg, s, a).reduce((x, o) => x + o.p, 0);
        expect(p).toBeCloseTo(1, 12);
      }
    }
    // (3,1) up: 0.8 to (3,2), 0.1 left to (2,1), 0.1 right to (4,1)
    const o = transitions(g, idx(3, 1), 0).map((x) => [x.s2, x.p]);
    expect(o).toEqual([[idx(3, 2), 0.8], [idx(2, 1), 0.1], [idx(4, 1), 0.1]]);
    // next to the wall: (1,2) right bumps into the wall and stays
    expect(transitions(g, idx(1, 2), 1)[0]).toMatchObject({ s2: idx(1, 2), moved: false });
    // entering the goal pays stepReward + 1
    expect(transitions(g, idx(3, 3), 1)[0]).toMatchObject({ s2: GOAL, done: true, r: 1 - 0.04 });
  });

  it('step samples the transition probabilities', () => {
    const rng = createRng(3), n = 20000, s = idx(3, 1), count = new Map<number, number>();
    for (let i = 0; i < n; i++) { const r = step(g, s, 0, rng); count.set(r.s2, (count.get(r.s2) ?? 0) + 1); }
    for (const o of transitions(g, s, 0)) expect(Math.abs(count.get(o.s2)! / n - o.p)).toBeLessThan(0.01);
  });

  it('returns follow G_t = r + gamma G_{t+1}', () => {
    const G = returns([-0.04, -0.04, 1], 0.9);
    expect(G[0]).toBeCloseTo(-0.04 + 0.9 * -0.04 + 0.81 * 1, 12);
    expect(G[2]).toBe(1);
  });
});

describe('dynamic programming', () => {
  // Russell and Norvig, AIMA 2nd/3rd ed., Fig. 17.3: gamma = 1, R(s) = -0.04 (three decimals).
  const BOOK: [number, number, number][] = [
    [1, 3, 0.812], [2, 3, 0.868], [3, 3, 0.918],
    [1, 2, 0.762], [3, 2, 0.660],
    [1, 1, 0.705], [2, 1, 0.655], [3, 1, 0.611], [4, 1, 0.388],
  ];
  it('value iteration reproduces the textbook utilities of the 4x3 world', () => {
    const { V } = optimal(g);
    for (const [x, y, u] of BOOK) expect(Math.abs(V[idx(x, y)] - u)).toBeLessThan(5e-4);
  });

  it('the converged values are a Bellman fixed point to 1e-9', () => {
    for (const gamma of [1, 0.9, 0.5]) {
      const { V } = optimal(makeGrid({ gamma }));
      const Q = qFromV(g, V, gamma);
      for (const s of g.states) expect(Math.abs(Math.max(...Q.subarray(s * NA, s * NA + NA)) - V[s])).toBeLessThan(1e-9);
    }
  });

  it('the textbook optimal policy goes up from the start and the long way round from (3,1)', () => {
    const pi = optimal(g).policy;
    expect(pi[START]).toBe(0);
    expect(pi[idx(3, 1)]).toBe(3); // left, away from the pit
    expect(pi[idx(3, 2)]).toBe(0);
    expect(pi[idx(3, 3)]).toBe(1);
  });

  it('value iteration records a contracting max change per sweep', () => {
    const r = valueIteration(makeGrid({ gamma: 0.9 }), 30);
    expect(r.V).toHaveLength(31);
    expect(r.Q).toHaveLength(30);
    for (let k = 1; k < 30; k++) expect(r.delta[k]).toBeLessThanOrEqual(0.9 * r.delta[k - 1] + 1e-12);
  });

  it('policy evaluation converges to the exact policy value', () => {
    const probs = deterministic(optimal(g).policy), ev = policyEvaluation(g, probs, 400), exact = policyValue(g, probs);
    for (const s of g.states) expect(ev.V[400][s]).toBeCloseTo(exact[s], 9);
    const uniform = softmaxPolicy(new Float64Array(N * NA)), u = policyValue(g, uniform);
    const ue = policyEvaluation(g, uniform, 3000).V[3000];
    for (const s of g.states) expect(ue[s]).toBeCloseTo(u[s], 8);
  });
});

describe('Q-learning', () => {
  it('records each TD update exactly', () => {
    const r = qLearning(g, { alpha: 0.1, epsilon: 0.1, episodes: 30, seed: 11 }), t = r.trace!;
    for (let i = 0; i < t.s.length; i++) {
      expect(t.tdError[i]).toBeCloseTo(t.target[i] - t.qBefore[i], 12);
      expect(t.qAfter[i * NA + t.a[i]]).toBeCloseTo(t.qBefore[i] + 0.1 * t.tdError[i], 12);
      if (t.done[i]) expect(t.target[i]).toBe(t.r[i]); // terminal target is the reward alone
    }
    expect(r.snapshots).toHaveLength(31);
  });

  it('is deterministic under a seed and approaches Q*', () => {
    const a = qLearning(g, { seed: 4, episodes: 50 }), b = qLearning(g, { seed: 4, episodes: 50 });
    expect(Array.from(a.Q)).toEqual(Array.from(b.Q));
    const long = qLearning(g, { seed: 4, episodes: 3000, alpha: 0.05, epsilon: 0.3, record: false });
    expect(Math.abs(Math.max(...long.Q.subarray(0, 4)) - optimal(g).V[START])).toBeLessThan(0.08);
  });
});

describe('policy gradients', () => {
  const b = makeBandit(3, 0, { means: [1, 2, 3] });
  const theta = [0.3, -0.2, 0.1];

  it('the REINFORCE bandit gradient is unbiased', () => {
    const n = 200000, s = gradientSamples(theta, b, n, 9), ex = exactGradient(theta, b.means);
    for (let j = 0; j < 3; j++) {
      const { mean, variance } = columnStats(s.plain, 3, j);
      expect(Math.abs(mean - ex[j])).toBeLessThan(4 * Math.sqrt(variance / n));
    }
  });

  it('a baseline keeps the mean and lowers the variance', () => {
    const n = 200000, s = gradientSamples(theta, b, n, 10), ex = exactGradient(theta, b.means);
    for (let j = 0; j < 3; j++) {
      const p = columnStats(s.plain, 3, j), q = columnStats(s.based, 3, j);
      expect(Math.abs(q.mean - ex[j])).toBeLessThan(4 * Math.sqrt(q.variance / n));
      expect(q.variance).toBeLessThan(p.variance);
    }
    // E[b grad log pi] = 0 exactly: sum_a pi_a (onehot(a) - pi) = 0
    const p = s.probs;
    for (let j = 0; j < 3; j++) expect(p.reduce((acc, pa, a) => acc + pa * ((a === j ? 1 : 0) - p[j]), 0)).toBeCloseTo(0, 14);
  });

  it('bandit PG, the testbed, grid REINFORCE and actor-critic are deterministic under a seed', () => {
    expect(Array.from(banditPG(b, { seed: 5 }).theta)).toEqual(Array.from(banditPG(b, { seed: 5 }).theta));
    const c1 = compareStrategies({ strategies: [{ kind: 'greedy' }], runs: 5, steps: 50 }), c2 = compareStrategies({ strategies: [{ kind: 'greedy' }], runs: 5, steps: 50 });
    expect(Array.from(c1[0].avgReward)).toEqual(Array.from(c2[0].avgReward));
    expect(Array.from(gridReinforce(g, { episodes: 20, seed: 2 }).theta)).toEqual(Array.from(gridReinforce(g, { episodes: 20, seed: 2 }).theta));
    expect(Array.from(actorCritic(g, { episodes: 20, seed: 2 }).theta)).toEqual(Array.from(actorCritic(g, { episodes: 20, seed: 2 }).theta));
  });

  it('a grid REINFORCE episode applies theta += lr * sum_t adv_t (onehot - pi)', () => {
    const r = gridReinforce(g, { episodes: 3, seed: 13, lr: 0.1, baseline: 'state-value', detail: [2] }), d = r.details[0];
    for (let i = 0; i < N * NA; i++) expect(d.thetaAfter[i]).toBeCloseTo(d.thetaBefore[i] + 0.1 * d.grad[i], 12);
    for (const st of d.steps) expect(st.adv).toBeCloseTo(st.G - st.b, 12);
  });
});

describe('DQN network and loss', () => {
  const H = 16, w = initNet(H, 17);
  const items = [{ s: START, a: 0, y: 0.5 }, { s: idx(3, 1), a: 3, y: -0.2 }, { s: idx(3, 3), a: 1, y: 0.9 }];

  it('has 7H + 4 parameters', () => { expect(paramCount(16)).toBe(116); expect(w.length).toBe(116); });

  it('analytic gradients match finite differences and src/lib/nn autograd', () => {
    const grad = new Float64Array(w.length), { loss } = regressionGrad(w, H, items, grad);
    const L = (ww: Float64Array) => regressionGrad(ww, H, items, new Float64Array(ww.length)).loss, h = 1e-6;
    for (let i = 0; i < w.length; i++) {
      const p = w.slice(), m = w.slice(); p[i] += h; m[i] -= h;
      expect(Math.abs((L(p) - L(m)) / (2 * h) - grad[i])).toBeLessThan(1e-7);
    }
    // oracle: the same loss built from Value nodes
    const P = Array.from(w, (x) => new Value(x));
    let total = new Value(0);
    for (const { s, a, y } of items) {
      const [x0, x1] = coords(s), hs: Value[] = [];
      for (let j = 0; j < H; j++) hs.push(P[2 * j].mul(x0).add(P[2 * j + 1].mul(x1)).add(P[2 * H + j]).relu());
      let q = P[7 * H + a];
      for (let j = 0; j < H; j++) q = q.add(P[3 * H + a * H + j].mul(hs[j]));
      total = total.add(q.sub(y).pow(2).mul(0.5 / items.length));
    }
    backward(total);
    expect(total.data).toBeCloseTo(loss, 12);
    P.forEach((p, i) => expect(p.grad).toBeCloseTo(grad[i], 10));
    // forward agrees with qOf-style evaluation
    const [x0, x1] = coords(START);
    expect(forward(w, H, x0, x1).length).toBe(4);
  });

  it('the TD loss is semi-gradient: the target is detached even when it uses the online weights', () => {
    const batch: Transition[] = [{ s: START, a: 0, r: -0.04, s2: idx(1, 2), done: false }, { s: idx(3, 3), a: 1, r: 0.96, s2: GOAL, done: true }];
    const grad = new Float64Array(w.length), res = tdLossGrad(w, w, H, batch, 0.9, grad);
    // equal to the regression gradient with the targets frozen at their current values
    const ref = new Float64Array(w.length);
    regressionGrad(w, H, batch.map((t, i) => ({ s: t.s, a: t.a, y: res.ys[i] })), ref);
    for (let i = 0; i < w.length; i++) expect(grad[i]).toBeCloseTo(ref[i], 14);
    // and NOT the full derivative of the loss with the target moving with w
    const full = (ww: Float64Array) => tdLossGrad(ww, ww, H, batch, 0.9, new Float64Array(ww.length)).loss, h = 1e-6;
    let differs = false;
    for (let i = 0; i < w.length; i++) {
      const p = w.slice(), m = w.slice(); p[i] += h; m[i] -= h;
      if (Math.abs((full(p) - full(m)) / (2 * h) - grad[i]) > 1e-5) differs = true;
    }
    expect(differs).toBe(true);
  });

  it('terminal targets equal the transition reward, whatever the target network says', () => {
    const t: Transition = { s: idx(3, 3), a: 1, r: 0.96, s2: GOAL, done: true };
    expect(tdTarget(w, H, t, 0.9)).toBe(0.96);
    expect(tdTarget(initNet(H, 99), H, { ...t, s: idx(4, 1), a: 0, s2: PIT, r: -1.04 }, 0.9)).toBe(-1.04);
    const live: Transition = { s: START, a: 0, r: -0.04, s2: idx(1, 2), done: false };
    const q = forward(w, H, ...coords(idx(1, 2)));
    expect(tdTarget(w, H, live, 0.9)).toBeCloseTo(-0.04 + 0.9 * Math.max(...q), 14);
  });
});

describe('replay buffer', () => {
  const tr = (i: number): Transition => ({ s: i % 9, a: i % 4, r: i, s2: (i + 1) % 9, done: false });
  it('keeps at most `capacity` transitions and overwrites the oldest', () => {
    const b = new ReplayBuffer(4);
    for (let i = 0; i < 6; i++) b.push(tr(i), i);
    expect(b.size).toBe(4);
    expect(Array.from(b.born).sort()).toEqual([2, 3, 4, 5]);
    expect(b.get(0).r).toBe(4); // slot 0 was overwritten by transition 4
  });
  it('samples distinct slots inside the filled part, deterministically under a seed', () => {
    const b = new ReplayBuffer(256);
    for (let i = 0; i < 10; i++) b.push(tr(i), i);
    const s = b.sample(16, createRng(1));
    expect(s.length).toBe(10); // fewer stored than requested
    expect(new Set(s).size).toBe(10);
    for (let i = 10; i < 300; i++) b.push(tr(i), i);
    const x = b.sample(16, createRng(7)), y = b.sample(16, createRng(7));
    expect(Array.from(x)).toEqual(Array.from(y));
    expect(new Set(x).size).toBe(16);
    for (const i of x) { expect(i).toBeGreaterThanOrEqual(0); expect(i).toBeLessThan(256); }
  });
});

describe('DQN training', () => {
  it('target weights stay frozen between copies and equal the online weights on a copy', () => {
    const r = trainDQN(g, { replay: true, target: true, copyEvery: 5, batch: 4, episodes: 30, keepSteps: 120, seed: 3 });
    let last: Float64Array | null = null, lastUpdate = -1;
    for (const st of r.steps) {
      if (st.update === lastUpdate) continue;
      if (st.update % 5 === 0 && st.update > 0) expect(Array.from(st.tw)).toEqual(Array.from(st.w));
      else if (last && Math.floor(st.update / 5) === Math.floor(lastUpdate / 5)) expect(Array.from(st.tw)).toEqual(Array.from(last));
      last = st.tw; lastUpdate = st.update;
    }
    expect(r.updates).toBeGreaterThan(20);
  });

  it('without a target network the target uses the online weights; runs are deterministic', () => {
    const a = trainDQN(g, { replay: false, target: false, episodes: 10, keepSteps: 30, seed: 4 });
    for (const st of a.steps) expect(Array.from(st.tw)).toEqual(Array.from(st.w));
    const b = trainDQN(g, { replay: false, target: false, episodes: 10, seed: 4 });
    expect(Array.from(a.w)).toEqual(Array.from(b.w));
  });
});

describe('PPO', () => {
  it('the clipped surrogate and its derivative for both advantage signs, inside and outside the band', () => {
    const eps = 0.2, h = 1e-7;
    for (const A of [1, -1, 0.5, -2]) for (const rho of [0.5, 0.79, 0.9, 1, 1.1, 1.21, 1.6]) {
      const c = clippedSurrogate(rho, A, eps), num = (clippedSurrogate(rho + h, A, eps).value - clippedSurrogate(rho - h, A, eps).value) / (2 * h);
      expect(c.dRho).toBeCloseTo(num, 5);
      expect(c.value).toBeCloseTo(Math.min(rho * A, Math.min(Math.max(rho, 1 - eps), 1 + eps) * A), 14);
    }
    // favourable side flat, the other side still active
    expect(clippedSurrogate(1.5, 1, 0.2)).toMatchObject({ dRho: 0, clipped: true });
    expect(clippedSurrogate(0.5, 1, 0.2).dRho).toBe(1);
    expect(clippedSurrogate(0.5, -1, 0.2)).toMatchObject({ dRho: 0, clipped: true });
    expect(clippedSurrogate(1.5, -1, 0.2).dRho).toBe(-1);
    expect(clippedSurrogate(1.2, 1, 0.2).value).toBeCloseTo(1.2, 14);
    expect(clippedSurrogate(1.5, 1, 0.2).value).toBeCloseTo(1.2, 14);
  });

  it('the logit gradient of the surrogate matches finite differences', () => {
    const rng = createRng(2), theta = Float64Array.from({ length: N * NA }, () => rng.normal(0, 0.5));
    const b = makeBatch(g, theta, new Float64Array(N), 64, createRng(5), 0.9, 0.95);
    const moved = theta.map((x) => x + rng.normal(0, 0.3)); // some ratios leave the band
    const all = Int32Array.from({ length: 64 }, (_, i) => i), grad = new Float64Array(N * NA);
    surrogateGrad(b, moved, new Float64Array(N), all, 0.2, grad);
    const obj = (th: Float64Array) => {
      let s = 0;
      for (let i = 0; i < 64; i++) { const p = softmaxPolicy(th); s += clippedSurrogate(p[b.s[i] * NA + b.a[i]] / Math.exp(b.logpOld[i]), b.adv[i], 0.2).value; }
      return -s / 64;
    };
    const h = 1e-6;
    for (let i = 0; i < N * NA; i++) {
      const p = moved.slice(), m = moved.slice(); p[i] += h; m[i] -= h;
      expect(Math.abs((obj(p) - obj(m)) / (2 * h) - grad[i])).toBeLessThan(1e-6);
    }
  });

  it('keeps old log-probabilities and advantages frozen across epochs; ratios start at 1', () => {
    const b = makeBatch(g, new Float64Array(N * NA), new Float64Array(N), 64, createRng(29), 0.9, 0.95);
    const snap = JSON.stringify([Array.from(b.logpOld), Array.from(b.adv), Array.from(b.ret), Array.from(b.probsOld)]);
    const theta = new Float64Array(N * NA), V = new Float64Array(N);
    const it = ppoEpochs(b, theta, V, { rng: createRng(1), lr: 0.03, epochs: 4, minibatch: 16 });
    expect(JSON.stringify([Array.from(b.logpOld), Array.from(b.adv), Array.from(b.ret), Array.from(b.probsOld)])).toBe(snap);
    expect(it.updates).toHaveLength(16);
    expect(it.epochs).toHaveLength(4);
    expect(Array.from(it.thetaOld).every((x) => x === 0)).toBe(true);
    for (let i = 0; i < 64; i++) expect(Math.exp(b.logpOld[i])).toBeCloseTo(0.25, 14); // uniform collection policy
  });

  it('GAE stops at terminations and truncations, bootstrapping only truncations', () => {
    const gamma = 0.9, lambda = 0.5;
    const r = [1, 2, 3, 4], v = [0.1, 0.2, 0.3, 0.4], v2 = [0.2, 9, 0.5, 0.6], term = [0, 1, 0, 0], trunc = [0, 0, 1, 1];
    const { delta, adv } = gae(r, v, v2, term, trunc, gamma, lambda);
    expect(delta[1]).toBeCloseTo(2 - 0.2, 14); // terminal: no V(s')
    expect(delta[2]).toBeCloseTo(3 + 0.9 * 0.5 - 0.3, 14); // truncated: bootstrap
    expect(adv[3]).toBeCloseTo(delta[3], 14);
    expect(adv[2]).toBeCloseTo(delta[2], 14); // truncation cuts the sum
    expect(adv[1]).toBeCloseTo(delta[1], 14); // termination cuts the sum
    expect(adv[0]).toBeCloseTo(delta[0] + gamma * lambda * delta[1], 14);
  });

  it('KL is 0 for identical policies and positive otherwise', () => {
    expect(klCategorical([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])).toBe(0);
    expect(klCategorical([0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25])).toBeGreaterThan(0);
  });

  it('training is deterministic and improves the exact start value', () => {
    const a = trainPPO(g, { iterations: 30, seed: 37 }), b = trainPPO(g, { iterations: 30, seed: 37 });
    expect(Array.from(a.theta)).toEqual(Array.from(b.theta));
    expect(a.evalReturn[30]).toBeGreaterThan(a.evalReturn[0]);
  });
});

describe('rollouts', () => {
  it('rollout is reproducible and ends in a terminal or at maxSteps', () => {
    const a = rollout(g, randomPolicy, createRng(1), 100), b = rollout(g, randomPolicy, createRng(1), 100);
    expect(a).toEqual(b);
    expect(a.terminated || a.actions.length === 100).toBe(true);
    expect(greedy(g, qFromV(g, optimal(g).V))[START]).toBe(0);
  });
});
