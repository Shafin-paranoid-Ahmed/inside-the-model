/** Offline trainer for the depth-width figure's hardcoded weights. Imported by the test only, never by the widget. */
import { buildNet, type Net, type NetParams } from './sim';

/** Full-batch Adam on logistic loss, tanh hidden layers, one linear logit output. Deterministic. */
export function trainNet(sizes: number[], data: { x: number[][]; y: number[] }, seed: number, epochs: number, lr = 0.05): Net {
  const { net } = buildNet(sizes, seed, 'tanh');
  const L = net.W.length, n = data.y.length;
  const zeros = (): NetParams => ({ W: net.W.map((m) => m.map((r) => r.map(() => 0))), b: net.b.map((v) => v.map(() => 0)) });
  const m = zeros(), v = zeros();
  const [b1, b2, eps] = [0.9, 0.999, 1e-8];
  for (let ep = 1; ep <= epochs; ep++) {
    const g = zeros();
    for (let s = 0; s < n; s++) {
      const hs: number[][] = [data.x[s]];
      for (let l = 0; l < L; l++) hs.push(net.W[l].map((row, j) => { const z = row.reduce((a, w, i) => a + w * hs[l][i], net.b[l][j]); return l === L - 1 ? z : Math.tanh(z); }));
      let d = [(1 / (1 + Math.exp(-hs[L][0])) - data.y[s]) / n];
      for (let l = L - 1; l >= 0; l--) {
        net.W[l].forEach((row, j) => { g.b[l][j] += d[j]; row.forEach((_, i) => { g.W[l][j][i] += d[j] * hs[l][i]; }); });
        if (l > 0) d = hs[l].map((h, i) => (1 - h * h) * net.W[l].reduce((a, row, j) => a + row[i] * d[j], 0));
      }
    }
    const c1 = 1 - b1 ** ep, c2 = 1 - b2 ** ep;
    const step = (p: number[], gr: number[], mm: number[], vv: number[]) => p.forEach((_, k) => {
      mm[k] = b1 * mm[k] + (1 - b1) * gr[k]; vv[k] = b2 * vv[k] + (1 - b2) * gr[k] ** 2;
      p[k] -= (lr * mm[k] / c1) / (Math.sqrt(vv[k] / c2) + eps);
    });
    for (let l = 0; l < L; l++) {
      step(net.b[l], g.b[l], m.b[l], v.b[l]);
      net.W[l].forEach((row, j) => step(row, g.W[l][j], m.W[l][j], v.W[l][j]));
    }
  }
  return net;
}

export const round4 = (net: Net): NetParams => {
  const r = (x: number) => Math.round(x * 1e4) / 1e4;
  return { W: net.W.map((m) => m.map((row) => row.map(r))), b: net.b.map((v) => v.map(r)) };
};
