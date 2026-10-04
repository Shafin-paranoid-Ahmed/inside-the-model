// Additional production layout and exact endpoint regression checks.
import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
const base = process.argv[2] || 'http://127.0.0.1:4322';
const root = '.playwright-mcp/check/day6-acceptance';
mkdirSync(root, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const errors = [], results = [];
let passed = false;
const slugs = ['optimizers', 'learning-rate-schedules', 'weight-initialization', 'normalization', 'dropout-regularizers', 'loss-landscapes'];
const seek = (range, value) => range.evaluate((el, v) => {
  el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true }));
}, value);
try {
  for (const width of [320, 768, 1024, 1440]) for (const theme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width, height: 844 }, reducedMotion: 'reduce', colorScheme: theme });
    page.on('pageerror', e => errors.push(e.message));
    for (const slug of slugs) {
      await page.goto(`${base}/learn/neural-networks/${slug}/`, { waitUntil: 'networkidle' });
      const figures = page.locator('main [data-figure]');
      for (const figure of await figures.all()) {
        await figure.scrollIntoViewIfNeeded();
        const range = figure.locator('input.scrub');
        await range.waitFor();
        await seek(range, Number(await range.getAttribute('max')));
        await page.waitForTimeout(80);
      }
      await page.locator('details').evaluateAll(ds => ds.forEach(d => { d.open = true; }));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${slug}/${width}/${theme}`);
      results.push({ slug, width, theme, overflowX: false });
      if (width === 320 && theme === 'light') {
        if (slug === 'normalization') {
          const figure = figures.first();
          for (const input of await figure.locator('input[type=number]').all()) {
            for (const value of ['', '999', '-999', await input.getAttribute('min'), await input.getAttribute('max')]) {
              await input.fill(value); await input.blur();
              const actual = Number(await input.inputValue());
              assert(Number.isFinite(actual));
              assert(actual >= Number(await input.getAttribute('min')) && actual <= Number(await input.getAttribute('max')));
              assert(!/NaN|Infinity/.test(await figure.innerText()));
            }
          }
        }
        if (slug === 'dropout-regularizers') {
          const figure = figures.nth(4), range = figure.locator('input.scrub');
          const max = Number(await range.getAttribute('max'));
          await seek(range, max * 0.7); await page.waitForTimeout(100);
          assert.match(await figure.innerText(), /Epoch 6/);
          await figure.screenshot({ path: `${root}/early-stop-320-before-terminal.png` });
          await seek(range, max); await page.waitForTimeout(100);
          assert.match(await figure.innerText(), /Epoch 7/);
          await figure.screenshot({ path: `${root}/early-stop-320-terminal.png` });
        }
      }
    }
    await page.close();
  }
  assert.deepEqual(errors, []);
  passed = true;
  console.log(JSON.stringify({ passed: true, layouts: results.length, errors }));
} finally {
  await browser.close();
  writeFileSync(`${root}/layout-results.json`, JSON.stringify({ passed, results, errors }, null, 2));
}
