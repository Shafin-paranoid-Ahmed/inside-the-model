/** `prefers-reduced-motion` query (a function so importing this module during SSR is safe). */
export const prefersReducedMotion = (): MediaQueryList => matchMedia('(prefers-reduced-motion: reduce)');
export const reducedMotion = (): boolean => prefersReducedMotion().matches;

export interface Loop {
  /** Restart the loop after `frame` returned false (call after any user input or state change). */
  wake(): void;
  destroy(): void;
}
export interface LoopOptions {
  /** Largest dt (seconds) passed to `frame`. Default 0.05. */
  maxDt?: number;
  /** Called when `el` enters/leaves the viewport. `first` is true on the first entry (use it to autoplay). */
  onView?: (inView: boolean, first: boolean) => void;
}

/**
 * One rAF loop that runs only while `el` is in view and the tab is visible.
 * `frame(dt, now)`: dt in seconds (0 on the first frame after idle), now = rAF timestamp in ms.
 * Return `false` to go idle; `wake()` restarts. Runs one frame on entering view so a paused widget still paints.
 */
export function createLoop(el: Element, frame: (dt: number, now: number) => boolean | void, opts: LoopOptions = {}): Loop {
  const maxDt = opts.maxDt ?? 0.05;
  let inView = false, seen = false, dead = false, raf = 0, last = 0;
  const alive = () => inView && !document.hidden && !dead;
  const tick = (now: number) => {
    raf = 0;
    if (!alive()) return;
    const dt = last ? Math.min(maxDt, (now - last) / 1000) : 0;
    last = now;
    if (frame(dt, now) === false) last = 0;
    else raf = requestAnimationFrame(tick);
  };
  const wake = () => { if (alive() && !raf) raf = requestAnimationFrame(tick); };
  const sync = () => {
    if (alive()) { last = 0; wake(); } else { cancelAnimationFrame(raf); raf = 0; last = 0; }
  };
  const io = new IntersectionObserver(([e]) => {
    inView = e.isIntersecting;
    opts.onView?.(inView, inView && !seen);
    if (inView) seen = true;
    sync();
  });
  io.observe(el);
  document.addEventListener('visibilitychange', sync);
  return {
    wake,
    destroy() {
      dead = true; cancelAnimationFrame(raf); raf = 0;
      io.disconnect();
      document.removeEventListener('visibilitychange', sync);
    },
  };
}
