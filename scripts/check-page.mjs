// Visual + runtime check for one page, in an isolated headless Chrome (safe to run in parallel).
//
//   node scripts/check-page.mjs learn/sequences/rnn-cell/ [--base http://localhost:4321] [--steps] [--mobile]
//
// For light and dark: loads the page with motion enabled, scrolls to every lesson step when --steps,
// records console errors, measures frame rate while the first widget is on screen, records each step's longest
// main-thread stall (stepMaxGapMs; > 100 ms counts as an error, since the step-1 fps reading misses later islands), and saves screenshots
// of each step's first figure to .playwright-mcp/check/<slug>/. Prints a JSON summary; exit code 1 on errors.
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';

const args = process.argv.slice(2);
const base = args.includes('--base') ? args[args.indexOf('--base') + 1] : 'http://localhost:4321';
// Pass the path WITHOUT a leading slash: Git Bash rewrites '/x' into a Windows path.
const path = '/' + (args.find((a) => !a.startsWith('--') && a !== base) ?? '').replace(/^\/+/, '');
const walk = args.includes('--steps');
const mobile = args.includes('--mobile');
const slug = path.replace(/^\/|\/$/g, '').replace(/\//g, '_') || 'home';
const out = `.playwright-mcp/check/${slug}`;
mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const summary = { path, errors: [], themes: {} };

for (const scheme of ['light', 'dark']) {
  const ctx = await browser.newContext({
    viewport: mobile ? { width: 390, height: 844 } : { width: 1280, height: 800 },
    deviceScaleFactor: 1, colorScheme: scheme, reducedMotion: 'no-preference',
  });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error') summary.errors.push(`[${scheme}] ${m.text()}`); });
  page.on('pageerror', (e) => summary.errors.push(`[${scheme}] ${e.message}`));
  const url = new URL(path, base).href;
  await page.goto(url, { waitUntil: 'networkidle' });
  const steps = walk ? await page.locator('[data-step]').count() : 0;
  const res = { steps, fps: null, maxFrameGapMs: null, overflowX: false, shots: [] };

  for (let s = walk && steps ? 1 : 0; s <= (walk ? steps : 0); s++) {
    // Prefer step s's figure; fall back to the first visible figure on the page.
    let fig = page.locator(`[data-step]:nth-of-type(${s || 1}) [data-figure]`).first();
    if (!(await fig.count())) fig = page.locator('main [data-figure]:visible, main figure:visible').first();
    if (await fig.count()) await fig.scrollIntoViewIfNeeded();
    // Let it hydrate + autoplay (~1.8 s), measuring the longest main-thread stall meanwhile (budget: ~100 ms).
    const gap = await page.evaluate(() => new Promise((done) => {
      let max = 0, prev = performance.now();
      const t0 = prev;
      const f = (t) => { max = Math.max(max, t - prev); prev = t; t - t0 < 1800 ? requestAnimationFrame(f) : done(max); };
      requestAnimationFrame(f);
    }));
    if (s >= 1) {
      res.stepMaxGapMs = [...(res.stepMaxGapMs ?? []), Math.round(gap)];
      if (gap > 100) summary.errors.push(`[${scheme}] step ${s}: main thread blocked ${Math.round(gap)} ms`);
    }
    if (s <= 1 && res.fps === null) {
      Object.assign(res, await page.evaluate(() => new Promise((done) => {
        let n = 0, max = 0, prev = performance.now();
        const t0 = prev;
        const f = (t) => { n++; max = Math.max(max, t - prev); prev = t; t - t0 < 2000 ? requestAnimationFrame(f) : done({ fps: +(n / 2).toFixed(1), maxFrameGapMs: +max.toFixed(1) }); };
        requestAnimationFrame(f);
      })));
      res.overflowX = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
    }
    const file = `${out}/${scheme}${s ? `-step${s}` : ''}${mobile ? '-mobile' : ''}.png`;
    if (await fig.count()) await fig.screenshot({ path: file }); else await page.screenshot({ path: file });
    res.shots.push(file);
  }
  summary.themes[scheme] = res;
  await ctx.close();
}
await browser.close();
console.log(JSON.stringify(summary, null, 2));
process.exit(summary.errors.length ? 1 : 0);
