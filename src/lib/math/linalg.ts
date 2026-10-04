/** Small dense linear algebra shared by PCA widgets (PCAProjector, EmbeddingSpace). Pure, no DOM. */

/** Covariance matrix of rows (n × d), dividing by n. Columns are centred first. */
export function covariance(rows: number[][]): number[][] {
  const n = rows.length, d = rows[0].length, mu = new Array<number>(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mu[j] += r[j] / n;
  const C = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const r of rows) for (let i = 0; i < d; i++) for (let j = i; j < d; j++) C[i][j] += ((r[i] - mu[i]) * (r[j] - mu[j])) / n;
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
  return C;
}

/**
 * Eigen-decomposition of a symmetric matrix by Jacobi rotations.
 * Returns eigenvalues in descending order and matching unit eigenvectors (vectors[i] belongs to values[i]).
 * Each eigenvector's sign is fixed so its largest component is positive.
 */
export function eigSym(M: number[][]): { values: number[]; vectors: number[][] } {
  const n = M.length, A = M.map((r) => [...r]), V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const x = A[k][p], y = A[k][q]; A[k][p] = c * x - s * y; A[k][q] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = A[p][k], y = A[q][k]; A[p][k] = c * x - s * y; A[q][k] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
  const vectors = order.map((i) => {
    const v = V.map((row) => row[i]);
    let big = 0;
    for (let k = 1; k < n; k++) if (Math.abs(v[k]) > Math.abs(v[big])) big = k;
    return v[big] < 0 ? v.map((x) => -x) : v;
  });
  return { values: order.map((i) => A[i][i]), vectors };
}
