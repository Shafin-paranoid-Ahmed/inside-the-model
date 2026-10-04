/** Seeded, deterministic PRNG (mulberry32). */
export interface Rng {
  next(): number;
  uniform(a: number, b: number): number;
  normal(mean?: number, std?: number): number;
}

export function createRng(seed: number): Rng {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    uniform: (a, b) => a + (b - a) * next(),
    // Box-Muller
    normal: (mean = 0, std = 1) =>
      mean + std * Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next()),
  };
}
