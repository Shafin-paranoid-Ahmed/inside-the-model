/** Read semantic colors from CSS tokens (never hard-code colors in a widget). */
export interface Color { hex: string; rgb: [number, number, number] }

let probe: HTMLElement | null = null;

/** Resolve any CSS color (hex, rgb(), var(...) alias) to [r, g, b] via the browser. */
function resolve(raw: string): [number, number, number] {
  if (!probe) {
    probe = document.createElement('span');
    probe.style.display = 'none';
    document.body.appendChild(probe);
  }
  probe.style.color = '';
  probe.style.color = raw;
  const m = getComputedStyle(probe).color.match(/[\d.]+/g) ?? ['0', '0', '0'];
  return [Math.round(+m[0]), Math.round(+m[1]), Math.round(+m[2])];
}

/** Raw computed value of a CSS custom property on <html> (use for fonts: `readVar('--font-code')`). */
export const readVar = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * `readColors({ fwd: '--viz-forward', grad: '--viz-gradient' })` → `{ fwd: { hex, rgb }, grad: { hex, rgb } }`.
 * Call again inside `onThemeChange` to pick up light/dark.
 */
export function readColors<K extends string>(names: Record<K, string>): Record<K, Color> {
  const out = {} as Record<K, Color>;
  for (const k in names) {
    const rgb = resolve(readVar(names[k]));
    out[k] = { rgb, hex: '#' + rgb.map((n) => n.toString(16).padStart(2, '0')).join('') };
  }
  return out;
}

/** Fires when `data-theme` on <html> changes or the OS color scheme flips. Returns an unsubscribe fn. */
export function onThemeChange(cb: () => void): () => void {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const mq = matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener('change', cb);
  return () => { mo.disconnect(); mq.removeEventListener('change', cb); };
}
