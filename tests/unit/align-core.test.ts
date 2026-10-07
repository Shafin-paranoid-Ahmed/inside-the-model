import { describe, expect, it } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import { Value, backward } from '../../src/lib/nn/value';
import {
  C, D, K, P, PHI, TRAIN, HELDOUT, THETA_BASE, W_GOLD, Y_CONTINUE, Y_RAMBLING, Y_HEDGED, Y_REFUSAL, TEXT, phi, wordCount,
  probsInto, kl, expected, expectedFeatures, meanProb, GOLD, scores, rewardGrad,
  nllGrad, trainSFT, makeDemos, demoFeatures, sftTheta, type Demo,
  samplePairs, fitRM, pairAccuracy, sigmoid, type Pair,
  trainAlignPPO, alignSurrogate, alignSurrogateGrad,
  dpoLossGrad, trainDPO, dpoMargin, optimalTheta, optimalRow, exactAscent, klObjective,
  violations, revise, critiqueRevise, slCAI, DEFAULT_PRINCIPLES,
  trainBigram, lossGrad, TRAIN_TRANSCRIPTS, V, counted,
  solve, respond, parseStrict, parseLenient, verify, groupAdvantages, trainGRPO, sampleGroups, startLogits, rowStats, OPEN, NACT,
} from '../../src/lib/align';

const sft = sftTheta();
const rnd = (seed: number, s = 1) => { const r = createRng(seed); return Float64Array.from({ length: D }, () => s * r.normal()); };
/** log pi(y|x) as a Value graph of theta (Values), for gradient checks. */
function logpV(th: Value[], x: number, y: number): Value {
  const z = Array.from({ length: K }, (_, c) => th.reduce((s: Value, t, j) => s.add(t.mul(PHI[(x * K + c) * D + j])), new Value(C[x * K + c])));
  const m = Math.max(...z.map((v) => v.data));
  const lse = z.reduce((s: Value, v) => s.add(v.sub(m).exp()), new Value(0)).log().add(m);
  return z[y].sub(lse);
}
const vals = (a: ArrayLike<number>) => Array.from(a, (v) => new Value(v));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));
const maxProbDiff = (a: ArrayLike<number>, b: ArrayLike<number>, prompts = TRAIN) => {
  let m = 0;
  for (const x of prompts) { const p = probsInto(a, x), q = probsInto(b, x); for (let y = 0; y < K; y++) m = Math.max(m, Math.abs(p[y] - q[y])); }
  return m;
};

describe('bank and policy', () => {
  it('reads length off the generated text and keeps the designed rambling answer rare', () => {
    expect(PHI.length).toBe(P * K * D);
    for (let x = 0; x < P; x++) for (let y = 0; y < K; y++) expect(phi(x, y)[2]).toBeCloseTo(wordCount(TEXT[x][y]) / 10, 12);
    expect(wordCount(TEXT[0][Y_RAMBLING])).toBe(48);
    expect(meanProb(sft, Y_RAMBLING)).toBeLessThan(1e-4);
  });

  it('the base model mostly continues the question list', () => {
    expect(meanProb(THETA_BASE, Y_CONTINUE)).toBeGreaterThan(0.5);
    for (const x of TRAIN) { const p = Array.from(probsInto(THETA_BASE, x)); expect(p.indexOf(Math.max(...p))).toBe(Y_CONTINUE); }
  });

  it('KL is non-negative, and zero at theta = theta_ref', () => {
    for (let s = 0; s < 20; s++) {
      const a = rnd(s), b = rnd(100 + s);
      expect(kl(a, b)).toBeGreaterThanOrEqual(0);
      expect(kl(a, a)).toBe(0);
    }
  });

  it('the exact reward gradient matches autograd of E_pi[r]', () => {
    const th = rnd(3), r = GOLD, tv = vals(th);
    let J = new Value(0);
    for (const x of TRAIN) for (let y = 0; y < K; y++) J = J.add(logpV(tv, x, y).exp().mul(r[x * K + y] / TRAIN.length));
    backward(J);
    expect(maxDiff(rewardGrad(th, r), tv.map((v) => v.grad))).toBeLessThan(1e-9);
  });
});

describe('SFT', () => {
  it('the analytic NLL gradient matches autograd to 1e-9', () => {
    const demos = makeDemos({ n: 24, accuracy: 0.8, seed: 2 }), th = rnd(5), tv = vals(th);
    let L = new Value(0);
    for (const d of demos) L = L.sub(logpV(tv, d.x, d.y).mul(1 / demos.length));
    backward(L);
    const g = new Float64Array(D), loss = nllGrad(th, demos, g);
    expect(loss).toBeCloseTo(L.data, 12);
    expect(maxDiff(g, tv.map((v) => v.grad))).toBeLessThan(1e-9);
  });

  it('matches feature moments at the optimum (E_pi[phi] = E_demo[phi])', () => {
    // every candidate appears once per train prompt, so all moments are interior and the optimum exists
    const demos: Demo[] = TRAIN.flatMap((x) => [...Array.from({ length: K }, (_, y) => ({ x, y })), { x, y: 2 }, { x, y: 1 }]);
    const th = trainSFT(demos, { steps: 4000, lr: 0.05 }).thetas[4000];
    expect(maxDiff(expectedFeatures(th), demoFeatures(demos))).toBeLessThan(1e-4);
  });

  it('the cap: SFT correctness ends at the demonstrators\' rate', () => {
    for (const acc of [0.6, 0.8, 0.95]) {
      const d = makeDemos({ n: 40, accuracy: acc, seed: 2 }), th = trainSFT(d, { steps: 300, lr: 0.05 }).thetas[300];
      expect(demoFeatures(d)[1]).toBeCloseTo(acc, 12);
      expect(Math.abs(expectedFeatures(th)[1] - acc)).toBeLessThan(0.01);
    }
  });

  it('the day\'s SFT model answers, and held-out prompts improve too', () => {
    expect(meanProb(sft, Y_CONTINUE)).toBeLessThan(0.1);
    expect(expectedFeatures(sft)[1]).toBeCloseTo(0.8, 1);
    for (const x of HELDOUT) expect(expectedFeatures(sft, [x])[1]).toBeGreaterThan(expectedFeatures(THETA_BASE, [x])[1] + 0.4);
  });

  it('the bigram mask: gradients are exact and only response tokens count', () => {
    const W = Float64Array.from({ length: V * V }, (_, i) => Math.sin(i)), g = new Float64Array(V * V), h = 1e-6;
    lossGrad(W, TRAIN_TRANSCRIPTS, 'response', g);
    for (const i of [3, 77, 400, 1201]) {
      const a = W.slice(), b = W.slice(); a[i] += h; b[i] -= h;
      expect(Math.abs((lossGrad(a, TRAIN_TRANSCRIPTS, 'response') - lossGrad(b, TRAIN_TRANSCRIPTS, 'response')) / (2 * h) - g[i])).toBeLessThan(1e-6);
    }
    const t = TRAIN_TRANSCRIPTS[0];
    expect(counted(t, 'response')[0]).toBe(t.respStart);
    expect(counted(t, 'all')).toHaveLength(t.ids.length - 1);
    const all = trainBigram('all', { steps: 50 }), resp = trainBigram('response', { steps: 50 });
    expect(all.heldResponse[50]).toBeLessThan(all.heldResponse[0]);
    expect(resp.heldResponse[50]).toBeLessThan(resp.heldResponse[0]);
  });
});

describe('reward model, PPO, DPO and the optimum', () => {
  const pairs = samplePairs(sft, 200, 4, 'human');
  const rm = fitRM(pairs);

  it('pairs are reproducible and the same seed gives the same pairs to both labelers', () => {
    const ai = samplePairs(sft, 200, 4, 'ai');
    expect(samplePairs(sft, 200, 4, 'human')).toEqual(pairs);
    expect(ai.map((q) => [q.x, q.a, q.b])).toEqual(pairs.map((q) => [q.x, q.a, q.b]));
    for (const q of pairs) expect(q.a).not.toBe(q.b);
  });

  it('the reward model fits held-out labels better than chance and extrapolates length to the rambling answer', () => {
    expect(pairAccuracy(rm.w, samplePairs(sft, 1000, 99, 'human'))).toBeGreaterThan(0.65);
    const r = scores(rm.w);
    let top = 0;
    for (let i = 0; i < P * K; i++) if (r[i] > r[top]) top = i;
    expect(top % K).toBe(Y_RAMBLING);
    expect(GOLD[top]).toBeLessThan(0);
    for (const q of pairs) { expect(q.a).not.toBe(Y_RAMBLING); expect(q.b).not.toBe(Y_RAMBLING); }
  });

  it('the PPO surrogate gradient matches autograd inside and outside the clip band, for both advantage signs', () => {
    const rng = createRng(8), old = rnd(11, 0.5), th = rnd(12, 0.8), n = 24;
    const b = { x: new Int32Array(n), y: new Int32Array(n), logpOld: new Float64Array(n), adv: new Float64Array(n) };
    for (let i = 0; i < n; i++) { b.x[i] = TRAIN[i % 8]; b.y[i] = Math.floor(rng.next() * K); b.logpOld[i] = Math.log(probsInto(old, b.x[i])[b.y[i]]); b.adv[i] = i % 2 ? 1.3 : -0.7; }
    const tv = vals(th), eps = 0.2;
    let S = new Value(0), clipped = 0;
    for (let i = 0; i < n; i++) {
      const rho = logpV(tv, b.x[i], b.y[i]).sub(b.logpOld[i]).exp(), A = b.adv[i], c = Math.min(Math.max(rho.data, 1 - eps), 1 + eps);
      const flat = c * A < rho.data * A; // the minimum picks the constant clipped branch
      if (flat) clipped++;
      S = S.add((flat ? new Value(c * A) : rho.mul(A)).mul(1 / n));
    }
    backward(S);
    expect(clipped).toBeGreaterThan(2);
    expect(clipped).toBeLessThan(n - 2);
    expect(alignSurrogate(th, b, eps)).toBeCloseTo(S.data, 12);
    expect(maxDiff(alignSurrogateGrad(th, b, eps), tv.map((v) => v.grad))).toBeLessThan(1e-9);
  });

  it('the DPO gradient matches autograd', () => {
    const th = rnd(21, 0.7), tv = vals(th), beta = 0.3, ps = pairs.slice(0, 30);
    let L = new Value(0);
    for (const q of ps) {
      const m = logpV(tv, q.x, q.w).sub(Math.log(probsInto(sft, q.x)[q.w])).sub(logpV(tv, q.x, q.l)).add(Math.log(probsInto(sft, q.x)[q.l])).mul(beta);
      L = L.sub(m.sigmoid().log().mul(1 / ps.length));
    }
    backward(L);
    const g = new Float64Array(D);
    expect(dpoLossGrad(th, sft, ps, beta, g)).toBeCloseTo(L.data, 12);
    expect(maxDiff(g, tv.map((v) => v.grad))).toBeLessThan(1e-9);
  });

  it('theta* = theta_ref + w / beta equals long exact ascent on E r - beta KL, and pi_ref exp(r / beta) / Z', () => {
    for (const beta of [1, 0.5]) {
      const ts = optimalTheta(sft, rm.w, beta), asc = exactAscent(sft, rm.w, beta, { steps: 60000, lr: 3 });
      // ascent creeps toward pi* where pi* has almost no mass, so compare probabilities to 1e-4 and the objective exactly
      expect(maxProbDiff(asc, ts)).toBeLessThan(1e-4);
      expect(klObjective(asc, sft, rm.w, beta)).toBeLessThanOrEqual(klObjective(ts, sft, rm.w, beta) + 1e-12);
      expect(klObjective(ts, sft, rm.w, beta) - klObjective(asc, sft, rm.w, beta)).toBeLessThan(1e-5);
      for (const x of [0, 5]) expect(maxDiff(optimalRow(sft, rm.w, beta, x).p, probsInto(ts, x))).toBeLessThan(1e-12);
    }
  }, 30000);

  it('DPO on many Bradley-Terry pairs recovers the direction w / beta', () => {
    const w = Float64Array.from([0.5, 1.5, -0.4, -0.8, -1.2, 0.3]), r = scores(w), rng = createRng(31), beta = 0.5, ps: Pair[] = [];
    const ref = new Float64Array(D);
    for (let i = 0; i < 1500; i++) {
      const x = TRAIN[i % 8], a = Math.floor(rng.next() * K);
      let b = Math.floor(rng.next() * K); if (b === a) b = (a + 1) % K;
      const aw = rng.next() < sigmoid(r[x * K + a] - r[x * K + b]);
      ps.push({ x, a, b, aWins: aw, w: aw ? a : b, l: aw ? b : a });
    }
    const th = trainDPO(ps, ref, { beta, steps: 400 }).thetas[400], v = Array.from(th, (t, j) => beta * (t - ref[j]));
    const cos = v.reduce((s, x, j) => s + x * w[j], 0) / Math.hypot(...v) / Math.hypot(...w);
    expect(cos).toBeGreaterThan(0.98);
  }, 20000);

  it('DPO from theta_ref follows the reward-model fit step for step, so it lands on pi*', () => {
    const d = trainDPO(pairs, sft, { beta: 0.1, steps: 200 });
    const target = optimalTheta(sft, rm.w, 0.1);
    // equal up to Adam's epsilon, which acts on gradients scaled by beta
    expect(maxProbDiff(d.thetas[200], target)).toBeLessThan(1e-3);
    expect(d.loss[200]).toBeCloseTo(rm.loss[200], 5);
    expect(dpoMargin(d.thetas[200], sft, pairs[0], 0.1)).toBeCloseTo(scores(rm.w)[pairs[0].x * K + pairs[0].w] - scores(rm.w)[pairs[0].x * K + pairs[0].l], 4);
  });

  // ---- the tuned headline effects (plans/day-11.md §7): a later edit must not silently remove them ----
  const g0 = expected(sft, GOLD);
  const ppo = (beta: number, iters = 200) => trainAlignPPO({ rm: rm.w, beta, iters, seed: 6 });

  it('beta 0: proxy rises while gold peaks early and then collapses onto the rambling answer', () => {
    const r = ppo(0);
    let pk = 0;
    for (let i = 0; i <= 200; i++) if (r.gold[i] > r.gold[pk]) pk = i;
    expect(pk).toBeGreaterThanOrEqual(10);
    expect(pk).toBeLessThanOrEqual(30);
    expect(r.gold[pk]).toBeGreaterThan(g0 + 0.4);
    expect(r.gold[200]).toBeLessThan(0);
    expect(r.proxy[200]).toBeGreaterThan(r.proxy[pk] + 0.5);
    expect(meanProb(r.thetas[200], Y_RAMBLING)).toBeGreaterThan(0.99);
    expect(r.kl[200]).toBeGreaterThan(10);
  });

  it('the beta sweep: 0.1 ends best, 0.5 above SFT, 0.02 and 0 hacked; PPO 0.1 ends near pi*', () => {
    const [a, b, c] = [ppo(0.5), ppo(0.1), ppo(0.02)];
    expect(b.gold[200]).toBeGreaterThan(a.gold[200]);
    expect(a.gold[200]).toBeGreaterThan(g0 + 0.3);
    expect(c.gold[200]).toBeLessThan(0);
    expect(maxProbDiff(b.thetas[200], optimalTheta(sft, rm.w, 0.1))).toBeLessThan(0.2);
    expect(trainAlignPPO({ rm: rm.w, beta: 0.1, seed: 6, iters: 30 }).gold).toEqual(trainAlignPPO({ rm: rm.w, beta: 0.1, seed: 6, iters: 30 }).gold);
  });

  it('AI labels (the constitution) avoid the length exploit even at beta 0; the cautious principle makes the policy hedge', () => {
    const w = fitRM(samplePairs(sft, 200, 4, 'ai')).w;
    const r0 = trainAlignPPO({ rm: w, beta: 0, seed: 6 });
    expect(meanProb(r0.thetas[200], Y_RAMBLING)).toBeLessThan(0.01);
    expect(r0.gold[200]).toBeGreaterThan(2);
    const wc = fitRM(samplePairs(sft, 200, 4, 'ai', { active: [...DEFAULT_PRINCIPLES, 'cautious'] })).w;
    const rc = trainAlignPPO({ rm: wc, beta: 0.1, seed: 6 }), rd = trainAlignPPO({ rm: w, beta: 0.1, seed: 6 });
    expect(meanProb(rc.thetas[200], Y_HEDGED) + meanProb(rc.thetas[200], Y_REFUSAL)).toBeGreaterThan(meanProb(rd.thetas[200], Y_HEDGED) + 0.3);
  });
});

describe('constitution', () => {
  it('checks principles and revises with the fewest other changes', () => {
    expect(Array.from({ length: K }, (_, y) => violations(0, y).length)).toEqual([1, 0, 0, 1, 1, 1, 2, 0]);
    expect(revise(0, 3, 'no-flattery')).toBe(2); // padded correct -> clear correct
    expect(revise(0, 5, 'accurate')).toBe(1); // terse wrong -> terse correct
    expect(revise(0, 2, 'accurate')).toBe(2); // nothing to fix
    const rs = critiqueRevise(0, Y_CONTINUE, 6, createRng(1));
    rs.forEach((r, i) => { if (i) expect(r.from).toBe(rs[i - 1].to); expect(r.violated).toBe(r.from !== r.to); });
  });

  it('SL-CAI lowers the expected violations of the policy', () => {
    const ev = (t: ArrayLike<number>) => TRAIN.reduce((s, x) => { const p = probsInto(t, x); return s + p.reduce((a, q, y) => a + q * violations(x, y).length, 0) / TRAIN.length; }, 0);
    const r = slCAI(sft, { n: 64, rounds: 4, seed: 3 });
    expect(ev(r.run.thetas[150])).toBeLessThan(ev(sft));
  });
});

describe('verifier and GRPO', () => {
  it('the solver slips only on carries it did not work out', () => {
    expect(solve({ a: 7, b: 5, digits: 1 }, 0).answer).toBe(2);
    expect(solve({ a: 7, b: 5, digits: 1 }, 1).answer).toBe(12);
    expect(solve({ a: 4789, b: 5634, digits: 4 }, 4).answer).toBe(10423);
    expect(solve({ a: 4789, b: 5634, digits: 4 }, 2).answer).toBe(9323);
    expect(solve({ a: 12, b: 34, digits: 2 }, 0).answer).toBe(46);
  });

  it('parses strictly and leniently', () => {
    expect(parseStrict('7 + 5 = 12, write 2, carry 1\nAnswer: 12')).toBe(12);
    expect(parseStrict('Answer: 12 ')).toBeNull();
    expect(parseStrict('answer: 12')).toBeNull();
    expect(parseStrict('Answer: 12\n')).toBeNull();
    expect(parseStrict('The answer is 12')).toBeNull();
    expect(parseStrict('Answer: 2 or 12 or 102 or 112')).toBeNull();
    expect(parseLenient('Answer: 2 or 12')).toEqual([2, 12]);
    expect(verify('Answer: 112', 12, 'lenient')).toBe(false);
    const p = { a: 7, b: 5, digits: 1 };
    expect(verify(respond(p, 0), 12, 'strict')).toBe(false);
    expect(verify(respond(p, 5), 12, 'lenient')).toBe(true); // the guess list covers the dropped carry
    expect(verify(respond(p, 5), 12, 'strict')).toBe(false);
    expect(verify(respond(p, 1), 12, 'strict')).toBe(true);
  });

  it('GRPO advantages sum to 0 per group and vanish when every attempt scores the same', () => {
    const a = groupAdvantages([1, 0, 0, 0]);
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12);
    expect(a[0]).toBeCloseTo(Math.sqrt(3), 12);
    expect(Array.from(groupAdvantages([1, 1, 1, 1]))).toEqual([0, 0, 0, 0]);
    const g = sampleGroups(startLogits(), { problems: 12, G: 8, rng: createRng(2) });
    for (const row of g.rows) expect(Math.abs(row.adv.reduce((s, v) => s + v, 0))).toBeLessThan(1e-12);
  });

  it('strict training raises accuracy and length (more for more digits); the lenient verifier is gamed with guess lists', () => {
    const s = trainGRPO({ verifier: 'strict', iters: 150, seed: 5 }), l = trainGRPO({ verifier: 'lenient', iters: 150, seed: 5 });
    const at = (r: typeof s, f: 'strict' | 'words' | 'list' | 'lenient', it: number, d: number) => r[f][it * 4 + d - 1];
    for (let d = 1; d <= 4; d++) expect(at(s, 'strict', 150, d)).toBeGreaterThan(0.95);
    expect(at(s, 'words', 150, 4) - at(s, 'words', 0, 4)).toBeGreaterThan(at(s, 'words', 150, 1) - at(s, 'words', 0, 1) + 20);
    for (const d of [1, 2]) {
      expect(at(l, 'lenient', 150, d)).toBeGreaterThan(0.95);
      expect(at(l, 'list', 150, d)).toBeGreaterThan(0.5);
      expect(at(l, 'strict', 150, d)).toBeLessThan(at(l, 'strict', 0, d));
    }
    expect(trainGRPO({ iters: 5, seed: 5 }).logits).toEqual(trainGRPO({ iters: 5, seed: 5 }).logits);
  });

  it('unverifiable prompts never change, and pass@8 is at least pass@1', () => {
    const r = trainGRPO({ domains: true, iters: 40, seed: 5 });
    expect(Array.from(r.logits[40].subarray(OPEN * NACT, OPEN * NACT + NACT))).toEqual(Array.from(r.logits[0].subarray(OPEN * NACT, OPEN * NACT + NACT)));
    for (let d = 1; d <= 4; d++) { const st = rowStats(startLogits(), d); expect(st.pass8).toBeGreaterThanOrEqual(st.pass1); }
  });
});

it('W_GOLD is the documented gold utility', () => expect(Array.from(W_GOLD)).toEqual([1, 2, -0.3, -1, -1, -0.5]));
