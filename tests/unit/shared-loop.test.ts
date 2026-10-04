import { afterEach, describe, expect, it, vi } from 'vitest';
import { createLoop } from '../../src/components/widgets/_shared/loop';

// Browser scheduling is fake; assertions cover our visibility/idle contract,
// not headless-browser frame timing or a production performance guarantee.
function harness() {
  let observeChange: (entries: { isIntersecting: boolean }[]) => void = () => {};
  const frames = new Map<number, FrameRequestCallback>();
  const listeners = new Map<string, () => void>();
  let nextId = 0;
  const document = { hidden: false,
    addEventListener: vi.fn((name: string, fn: () => void) => listeners.set(name, fn)),
    removeEventListener: vi.fn((name: string) => listeners.delete(name)),
  };
  const observe = vi.fn(), disconnect = vi.fn();
  vi.stubGlobal('document', document);
  vi.stubGlobal('requestAnimationFrame', (fn: FrameRequestCallback) => {
    frames.set(++nextId, fn); return nextId;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id));
  vi.stubGlobal('IntersectionObserver', class {
    constructor(fn: typeof observeChange) { observeChange = fn; }
    observe = observe;
    disconnect = disconnect;
  });
  return { document, frames, observe, disconnect,
    visible: (value: boolean) => observeChange([{ isIntersecting: value }]),
    hidden: (value: boolean) => { document.hidden = value; listeners.get('visibilitychange')?.(); },
    tick: (now: number) => {
      const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn(now));
    },
  };
}
afterEach(() => vi.unstubAllGlobals());
describe('shared widget loop', () => {
  it('waits for visibility, autoplays only on first entry and caps elapsed time', () => {
    const h = harness(), frame = vi.fn(() => true), onView = vi.fn();
    const loop = createLoop({} as Element, frame, { maxDt: 0.05, onView });
    loop.wake(); expect(h.frames.size).toBe(0);
    h.visible(true); h.tick(1000); h.tick(1800);
    expect(frame.mock.calls).toEqual([[0, 1000], [0.05, 1800]]);
    expect(onView).toHaveBeenCalledWith(true, true);
    h.visible(false); expect(h.frames.size).toBe(0);
    h.visible(true); h.tick(5000);
    expect(onView).toHaveBeenLastCalledWith(true, false);
    expect(frame).toHaveBeenLastCalledWith(0, 5000);
    loop.destroy();
  });
  it('hidden tabs cancel scheduling and resume with zero elapsed time', () => {
    const h = harness(), frame = vi.fn(() => true);
    const loop = createLoop({} as Element, frame);
    h.visible(true); h.tick(1000);
    h.hidden(true); loop.wake(); h.tick(9000);
    expect(frame).toHaveBeenCalledTimes(1); expect(h.frames.size).toBe(0);
    h.hidden(false); h.tick(12000);
    expect(frame).toHaveBeenLastCalledWith(0, 12000);
    loop.destroy();
  });
  it('paused frames go idle, user input wakes once, destruction removes observers', () => {
    const h = harness(), frame = vi.fn(() => false);
    const loop = createLoop({} as Element, frame);
    h.visible(true); h.tick(1000); expect(h.frames.size).toBe(0);
    loop.wake(); loop.wake(); expect(h.frames.size).toBe(1);
    h.tick(2000); expect(frame).toHaveBeenLastCalledWith(0, 2000);
    loop.destroy(); expect(h.disconnect).toHaveBeenCalledOnce();
    expect(h.document.removeEventListener).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    loop.wake(); expect(h.frames.size).toBe(0);
  });
});
