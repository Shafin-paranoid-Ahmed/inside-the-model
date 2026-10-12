import { describe, it, expect, beforeAll } from 'vitest';
import {
  train, buildScene, specFor, SPECS, STEPS, SNAPS, N, DATA, MOONS, quality, finalParams, encodeAll, decodeAll, sqErr,
  roundTrip, timing, trainFrame, exploreFrame, sampleFrame, denoiseFrame, SAMPLES, makeStepper, encodeDist, priorMass, encodeFrame, reparamFrame, betaFrame, BELL_K, BETA_SEG, VAE_BETAS, HIST_BINS, WIDE, FOCUS_DEFAULT, DENOISE_SIGMA, type Run, type Scene,
} from '../../src/components/widgets/LatentSpace/sim';

const runs: Record<string, Run> = {};
beforeAll(() => { for (const id of ['ae1', 'ae2', 'dae', 'vae0.03', 'vae0.1', 'vae1', 'vae4'] as const) runs[id] = train(id); }, 120000);
const f2 = (v: number) => v.toFixed(2);

describe('LatentSpace trainer', () => {
  it('picks the model a figure needs', () => {
    expect(specFor('ae').id).toBe('ae1');
    expect(specFor('ae', 2).id).toBe('ae2');
    expect(specFor('dae').noise).toBe(0.25);
    expect(specFor('vae', 1, 4).id).toBe('vae4');
    expect(specFor('vae', 1, 0.1).beta).toBe(0.1);
  });

  it('is deterministic: two runs give identical snapshots', () => {
    const a = train('ae1', 60), b = train('ae1', 60);
    expect(a.snaps.length).toBe(SNAPS + 1);
    a.snaps.forEach((s, k) => expect(Array.from(s)).toEqual(Array.from(b.snaps[k])));
    expect(a.steps[SNAPS]).toBe(60);
  });

  it('a VAE trains: its loss falls', () => {
    const r = train('vae1', 300);
    const mean = (a: Float32Array, i: number, j: number) => a.slice(i, j).reduce((x, y) => x + y, 0) / (j - i);
    expect(mean(r.losses, 270, 300)).toBeLessThan(0.95 * mean(r.losses, 0, 30));
    expect(encodeAll(finalParams(r), SPECS.vae1, DATA.x, 5).length).toBe(5);
  });

  it('1-number autoencoder: error falls by over 30x and rebuilds nearly every point onto a moon', () => {
    const r = runs.ae1, sc = buildScene(r, 'tour');
    expect(r.snapLoss[0]).toBeGreaterThan(1);
    expect(r.snapLoss[SNAPS]).toBeLessThan(r.snapLoss[0] / 30);
    expect(sc.mse).toBeCloseTo(r.snapLoss[SNAPS], 5);
    expect(sc.mse).toBeGreaterThan(0.03); expect(sc.mse).toBeLessThan(0.09);
    expect(quality(sc.R.slice(0, 2 * N), MOONS).onData).toBeGreaterThan(0.97);
  });

  it('2-number autoencoder copies its input: probes far from the moons come back where they started', () => {
    const sc = buildScene(runs.ae2, 'tour', true);
    expect(sc.mse).toBeLessThan(0.002);
    for (let i = N; i < sc.n; i++) expect(Math.hypot(sc.R[2 * i] - sc.X[2 * i], sc.R[2 * i + 1] - sc.X[2 * i + 1])).toBeLessThan(0.2);
    const one = buildScene(runs.ae1, 'tour', true); // the 1-number code pulls the same probes onto the data
    for (let i = N; i < one.n; i++) expect(Math.hypot(one.R[2 * i] - one.X[2 * i], one.R[2 * i + 1] - one.X[2 * i + 1])).toBeGreaterThan(0.3);
  });

  it('the codes of the two moons run along one line and meet near zero', () => {
    const sc = buildScene(runs.ae1, 'explore');
    const up: number[] = [], lo: number[] = [];
    for (let i = 0; i < N; i++) (sc.moon[i] ? lo : up).push(sc.Z[i]);
    expect(Math.max(...up)).toBeGreaterThan(Math.min(...lo)); // the clusters touch, no gap
    expect(Math.min(...up)).toBeLessThan(-2); expect(Math.max(...lo)).toBeGreaterThan(2);
  });
});

describe('LatentSpace frames', () => {
  it('roundTrip: mix is 0 at the start, 1 at the code, 2 when rebuilt, and never jumps', () => {
    for (const m of ['tour', 'focus'] as const) {
      const { T, markers } = timing(m);
      expect(roundTrip(0, m)).toMatchObject({ mix: 0, phase: 0 });
      expect(roundTrip(T - 0.01, m)).toMatchObject({ mix: 2, phase: 4 });
      expect(roundTrip(markers[2].at + 0.1, m).mix).toBe(1);
      let prev = 0;
      for (let t = 0; t < T; t += 0.02) { const { mix } = roundTrip(t, m); expect(mix - prev).toBeGreaterThanOrEqual(0); expect(mix - prev).toBeLessThan(0.1); prev = mix; }
    }
  });

  it('train: the snapshot index follows the timeline and the loss is that snapshot\'s real error', () => {
    const sc = buildScene(runs.ae1, 'train');
    expect(trainFrame(0, sc)).toMatchObject({ k: 0, step: 0 });
    const end = trainFrame(timing('train', sc).T - 0.01, sc);
    expect(end.k).toBe(SNAPS); expect(end.step).toBe(STEPS);
    const k = 15, p = sc.run.snaps[k];
    expect(trainFrame(k * 0.35 + 0.01, sc).loss).toBeCloseTo(sqErr(decodeAll(p, sc.spec, encodeAll(p, sc.spec, DATA.x, N), N), DATA.x, N), 5);
    expect(sc.curves.length).toBe(SNAPS + 1);
  });

  it('explore: the sweep covers the whole code range and comes back; only a short stretch is off the data', () => {
    const sc = buildScene(runs.ae1, 'explore'), T = timing('explore').T;
    expect(exploreFrame(0, sc).z).toBeCloseTo(sc.zlo, 6);
    expect(exploreFrame(T / 2, sc).z).toBeCloseTo(sc.zhi, 6);
    let off = 0; const n = 400;
    for (let i = 0; i < n; i++) if (!exploreFrame((i / n) * (T / 2), sc).near) off++;
    expect(off / n).toBeLessThan(0.08); expect(off).toBeGreaterThan(0);
    expect(exploreFrame(1, sc, 0.5).z).toBe(0.5);
  });

  it('sample: counts match the quality() criterion; a wider code range lands fewer on the data', () => {
    const sc = buildScene(runs.ae1, 'sample');
    const [a, b] = sc.samples;
    expect(a.cum[SAMPLES] / SAMPLES).toBeCloseTo(a.q.onData, 10);
    expect(b.cum[SAMPLES] / SAMPLES).toBeCloseTo(b.q.onData, 10);
    expect(a.q.onData).toBeGreaterThan(0.95); expect(b.q.onData).toBeLessThan(a.q.onData - 0.05);
    expect(Math.min(...b.z)).toBeGreaterThan(-WIDE); expect(Math.max(...a.z)).toBeLessThanOrEqual(sc.zhi);
    const f = sampleFrame(5, sc); expect(f).toMatchObject({ batch: 0, whole: SAMPLES });
    expect(f.on + f.off).toBe(f.whole);
    expect(sampleFrame(6.4, sc).whole).toBe(SAMPLES); expect(sampleFrame(6.6, sc)).toMatchObject({ batch: 1 });
  });

  it('denoise: noise ramps 0 to the full level; cleaning moves points closer to the data', () => {
    const sc: Scene = buildScene(runs.dae, 'denoise');
    expect(denoiseFrame(0, sc).sigma).toBe(0);
    const full = denoiseFrame(5, sc);
    expect(full.sigma).toBeCloseTo(DENOISE_SIGMA, 10);
    expect(full.after).toBeLessThan(0.75 * full.before);
    expect(denoiseFrame(9.99, sc).sigma).toBeLessThan(0.01);
    const a = denoiseFrame(3, sc), b = denoiseFrame(3, sc);
    expect(Array.from(a.out)).toEqual(Array.from(b.out));
  });

  it('the followed point is on the upper moon near its left tip', () => {
    expect(DATA.moon[FOCUS_DEFAULT]).toBe(0);
    expect(DATA.x[2 * FOCUS_DEFAULT]).toBeLessThan(-1.7);
  });

  it('the numbers quoted in autoencoders.mdx', () => {
    const r = runs.ae1, sc = buildScene(r, 'sample'), f = buildScene(r, 'focus'), i = FOCUS_DEFAULT;
    expect([r.snapLoss[0], r.snapLoss[15], r.snapLoss[30]].map((v) => v.toFixed(3))).toEqual(['1.765', '0.083', '0.053']);
    expect(r.steps[15]).toBe(750);
    const e = (f.R[2 * i] - f.X[2 * i]) ** 2 + (f.R[2 * i + 1] - f.X[2 * i + 1]) ** 2;
    expect([f2(f.X[2 * i]), f2(f.X[2 * i + 1]), f2(f.Z[i]), f2(f.R[2 * i]), f2(f.R[2 * i + 1]), e.toFixed(3)]).toEqual(['-1.90', '-0.32', '-2.32', '-1.90', '-0.12', '0.041']);
    expect([sc.samples[0].cum[SAMPLES], sc.samples[1].cum[SAMPLES]]).toEqual([295, 253]);
    expect([f2(sc.zlo), f2(sc.zhi)]).toEqual(['-2.58', '2.54']);
    const up: number[] = [], lo: number[] = [];
    for (let k = 0; k < N; k++) (sc.moon[k] ? lo : up).push(sc.Z[k]);
    expect([f2(Math.min(...up)), f2(Math.max(...up)), f2(Math.min(...lo)), f2(Math.max(...lo))]).toEqual(['-2.58', '0.22', '-0.13', '2.54']);
    // the stretch of code line that decodes off the data
    let first = Infinity, last = -Infinity;
    for (let k = 0; k <= 2000; k++) if (!exploreFrame(0, sc, sc.zlo + ((sc.zhi - sc.zlo) * k) / 2000).near) { const z = sc.zlo + ((sc.zhi - sc.zlo) * k) / 2000; first = Math.min(first, z); last = Math.max(last, z); }
    expect([f2(first), f2(last)]).toEqual(['-0.04', '0.06']);
    const two = buildScene(runs.ae2, 'tour', true);
    expect(two.mse.toPrecision(2)).toBe('0.00034');
    expect([f2(two.R[2 * (N + 1)]), f2(two.R[2 * (N + 1) + 1])]).toEqual(['1.44', '0.67']);
    const d = denoiseFrame(5, buildScene(runs.dae, 'denoise')), d0 = denoiseFrame(0, buildScene(runs.dae, 'denoise'));
    expect([d.before, d.after, d0.before, d0.after].map(f2)).toEqual(['0.20', '0.13', '0.07', '0.13']);
  });
});

describe('VAE loss', () => {
  it('analytic gradient (reparameterisation and KL) matches central finite differences', () => {
    for (const id of ['vae1', 'vae0.1', 'ae1'] as const) {
      const spec = SPECS[id], B = 8, p = finalParams(train(id, 60)), step = makeStepper(spec, B);
      const X = DATA.x.slice(0, 2 * B), eps = Float64Array.from({ length: B }, (_, i) => Math.sin(1 + 2.3 * i) * 1.5);
      const g = new Float64Array(p.length); step(p, X, X, eps, g);
      const loss = (q: Float64Array) => { const { rec, kl } = step(q, X, X, eps, null); return rec + spec.beta * kl; };
      let worst = 0, checked = 0;
      for (let k = 0; k < p.length; k += 7) {
        const h = 1e-6, a = p.slice(), b = p.slice(); a[k] += h; b[k] -= h;
        const num = (loss(a) - loss(b)) / (2 * h), err = Math.abs(num - g[k]) / (1e-4 + Math.abs(num) + Math.abs(g[k]));
        worst = Math.max(worst, err); checked++;
      }
      expect(checked).toBeGreaterThan(100);
      expect(worst).toBeLessThan(1e-4);
    }
  });

  it('KL is zero for the prior and positive otherwise (closed form)', () => {
    const run = runs['vae4']; // collapsed: every bell curve is the prior
    const { mu, sigma } = encodeDist(finalParams(run), SPECS.vae4, DATA.x, N);
    expect(Math.max(...Array.from(mu).map(Math.abs))).toBeLessThan(0.1);
    expect(Math.abs(Math.max(...sigma) - 1)).toBeLessThan(0.05);
    expect(run.snapKl[SNAPS]).toBeLessThan(0.01);
    expect(runs['vae0.1'].snapKl[SNAPS]).toBeGreaterThan(1);
    expect(priorMass(-1.96, 1.96)).toBeCloseTo(0.95, 3);
  });

  it('numbers quoted in vaes.mdx', () => {
    const ae = buildScene(runs.ae1, 'sample', false, true), v = buildScene(runs['vae0.1'], 'sample');
    expect(ae.samples[0].cum[SAMPLES]).toBe(297);
    expect(Array.from(ae.samples[0].z)).toEqual(Array.from(v.samples[0].z)); // the same 300 prior draws
    const zs = Array.from(ae.Z.slice(0, N)), m = zs.reduce((a, b) => a + b, 0) / N, sd = Math.sqrt(zs.reduce((a, b) => a + (b - m) ** 2, 0) / N);
    expect([f2(m), f2(sd)]).toEqual(['-0.03', '1.07']);
    expect(v.samples[0].cum[SAMPLES]).toBe(260);
    expect([v.samples[0].q.onData.toFixed(2)]).toEqual(['0.87']);
    expect([f2(v.samples[0].q.upper), f2(v.samples[0].q.lower)]).toEqual(['0.48', '0.52']);
    // the followed point
    const e = buildScene(runs['vae0.1'], 'encode'), b = e.bell!;
    expect([f2(b.mu), f2(b.sigma), f2(b.spread)]).toEqual(['2.15', '0.31', '0.12']);
    expect(b.z.length).toBe(BELL_K);
    // reparameterisation at eps = 1
    const mu = e.Z[FOCUS_DEFAULT], sg = e.sigma[FOCUS_DEFAULT];
    expect([mu.toFixed(3), sg.toFixed(3), (mu + sg).toFixed(3), (mu + sg * 2).toFixed(3)]).toEqual(['2.147', '0.306', '2.453', '2.758']);
    // training
    const t = runs['vae0.1'], r = [0, 1, 30];
    expect(r.map((k) => f2(t.snapRec[k]))).toEqual(['1.51', '0.49', '0.13']);
    expect(r.map((k) => f2(t.snapKl[k]))).toEqual(['0.01', '1.18', '2.07']);
    // the four KL weights: reconstruction error with the mean code, KL, prior samples on the data, typical sigma
    const rows = VAE_BETAS.map((bt) => {
      const sc = buildScene(runs[`vae${bt}`], 'beta'), sgs = Array.from(sc.sigma.slice(0, N)).sort((a, c) => a - c);
      return [bt, sc.mse.toFixed(3), sc.kl.toFixed(2), sc.samples[0].cum[SAMPLES], sgs[N / 2].toFixed(2)];
    });
    expect(rows).toEqual([[0.03, '0.041', '2.95', 294, '0.05'], [0.1, '0.076', '2.07', 260, '0.12'], [1, '0.375', '0.50', 114, '0.61'], [4, '1.469', '0.00', 0, '1.00']]);
    // the stretch of the code line that decodes off the data, and its probability under the prior
    let first = Infinity, last = -Infinity;
    for (let k = 0; k <= 2000; k++) { const z = -2.5 + (5 * k) / 2000; if (!exploreFrame(0, v, z).near) { first = Math.min(first, z); last = Math.max(last, z); } }
    expect([f2(first), f2(last), f2(priorMass(first, last))]).toEqual(['-0.15', '0.30', '0.18']);
    const up: number[] = [], lo: number[] = [];
    for (let i = 0; i < N; i++) (v.moon[i] ? lo : up).push(v.Z[i]);
    expect([f2(Math.min(...up)), f2(Math.max(...up)), f2(Math.min(...lo)), f2(Math.max(...lo))]).toEqual(['-0.38', '2.30', '-2.45', '0.64']);
  });

  it('VAE frames: encode draws, epsilon sweep, beta segments, prior histogram', () => {
    const e = buildScene(runs['vae0.1'], 'encode');
    expect(encodeFrame(0)).toMatchObject({ phase: 0, whole: 0 });
    expect(encodeFrame(9.9)).toMatchObject({ phase: 2, whole: BELL_K });
    expect(reparamFrame(0, e).eps).toBeCloseTo(-2, 10);
    expect(reparamFrame(6, e).eps).toBeCloseTo(2, 10);
    expect(reparamFrame(3, e).z).toBeCloseTo(e.Z[FOCUS_DEFAULT] + e.sigma[FOCUS_DEFAULT] * reparamFrame(3, e).eps, 10);
    expect(betaFrame(0)).toEqual({ k: 0, fade: 1 });
    expect(betaFrame(BETA_SEG * 2 + 0.01).k).toBe(2);
    expect(betaFrame(BETA_SEG * 4 - 0.01).fade).toBe(1);
    expect(timing('beta').markers.length).toBe(VAE_BETAS.length);
    // the code histogram of a VAE is a density: bars times width add up to 1
    const bw = (e.hist.b - e.hist.a) / HIST_BINS;
    expect(e.hist.up.reduce((a, b) => a + b, 0) * bw + e.hist.lo.reduce((a, b) => a + b, 0) * bw).toBeCloseTo(1, 6);
    const tr = buildScene(runs['vae0.1'], 'train');
    expect(tr.snapHist.length).toBe(SNAPS + 1);
    expect(trainFrame(0, tr).kl).toBeLessThan(0.05);
  });
});
