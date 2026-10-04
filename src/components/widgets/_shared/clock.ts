/** Pure timeline clock: no DOM, unit-tested. The widget's `t` (seconds) drives `frameAt(t)` in sim.ts. */
export interface ClockState { t: number; playing: boolean; speed: number }
export interface Marker { at: number; label: string }
export interface AdvanceOptions {
  /** Total length of the timeline in seconds. */
  duration: number;
  /** Loop range start (default 0). */
  loopFrom?: number;
  /** Loop range end (default `duration`). May exceed `duration` to hold on the last frame. */
  loopTo?: number;
}

/**
 * Step the clock by `dt` seconds (already clamped by the loop). Paused → unchanged.
 * On reaching `loopTo` it wraps to `loopFrom` (carrying the overshoot), so `next.t < s.t` means "a cycle just ended".
 */
export function advance(s: ClockState, dt: number, o: AdvanceOptions): ClockState {
  if (!s.playing) return s;
  const from = o.loopFrom ?? 0, to = o.loopTo ?? o.duration, span = to - from;
  let t = s.t + dt * s.speed;
  if (t >= to) t = span > 0 ? from + ((t - to) % span) : from;
  return { ...s, t };
}

/** The first marker time strictly after `t` (+eps), or undefined. Used by the Step button. */
export function nextMarker(t: number, markers: readonly { at: number }[], eps = 0.02): number | undefined {
  let best: number | undefined;
  for (const m of markers) if (m.at > t + eps && (best === undefined || m.at < best)) best = m.at;
  return best;
}

/** The phase at `t`: the last marker with `at <= t`. */
export function markerAt<M extends { at: number }>(t: number, markers: readonly M[]): M | undefined {
  let best: M | undefined;
  for (const m of markers) if (m.at <= t + 1e-9 && (!best || m.at >= best.at)) best = m;
  return best;
}
