/**
 * The PPO-clip term for one sample: min(rho A, clip(rho, 1 - eps, 1 + eps) A) and its derivative with respect to rho.
 * Kept independent of the grid trainer so response-policy figures can reuse it without loading gridworld training.
 * The derivative is zero on the favourable side outside the band (A > 0 and rho > 1 + eps, or A < 0 and rho < 1 - eps).
 */
export function clippedSurrogate(rho: number, A: number, eps: number): { value: number; dRho: number; clipped: boolean } {
  const lo = 1 - eps, hi = 1 + eps, c = rho < lo ? lo : rho > hi ? hi : rho, un = rho * A, cl = c * A;
  const clipped = cl < un;
  return { value: clipped ? cl : un, dRho: clipped ? 0 : A, clipped };
}
