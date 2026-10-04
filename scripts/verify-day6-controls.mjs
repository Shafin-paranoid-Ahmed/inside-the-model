import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
const base = process.argv[2] || 'http://127.0.0.1:4322';
const root = '.playwright-mcp/check/day6-teaching'; mkdirSync(root, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const errors = [], results = [];
const seek = async (figure, fraction = 1) => {
  const input = figure.locator('input.scrub'); await input.waitFor();
  await input.evaluate((el, f) => { el.value = String(Number(el.max) * f); el.dispatchEvent(new Event('input', { bubbles: true })); }, fraction);
  await figure.page().waitForTimeout(100);
};
let passed = false;
try {
  for (const width of [320, 768, 1024, 1440]) for (const theme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width, height: 844 }, reducedMotion: 'reduce', colorScheme: theme });
    page.on('pageerror', e => errors.push(e.message));
    for (const slug of ['optimizers', 'weight-initialization', 'normalization', 'learning-rate-schedules']) {
      await page.goto(`${base}/learn/neural-networks/${slug}/`, { waitUntil: 'networkidle' });
      const figures = page.locator('main [data-figure]');
      for (const figure of await figures.all()) {
        await figure.scrollIntoViewIfNeeded();
        for (const fraction of slug === 'normalization' ? [0, 0.5, 1] : [1]) {
          await seek(figure, fraction);
          const stage = await figure.locator('.fig').evaluate(el => ({ height: el.clientHeight, content: el.scrollHeight, width: el.clientWidth }));
          if (stage.content > stage.height + 2) {
            console.log({ slug, width, theme, fraction, title: await figure.locator('.ttl').innerText(), stage });
            await figure.screenshot({ path: `${root}/failure-${slug}-${width}-${theme}-${fraction}.png` });
          }
          assert.equal(stage.content > stage.height + 2, false, `${slug}/${width}/${theme} stage height at ${fraction}`);
        }
        if (slug === 'normalization' && (width === 320 || width === 1440)) {
          const name = (await figure.locator('.ttl').innerText()).replace(/[^a-z0-9]+/gi, '-');
          await figure.screenshot({ path: `${root}/normalization-final-${width}-${theme}-${name}.png` });
        }
      }
      if (slug === 'optimizers') {
        const first = figures.first(); await first.scrollIntoViewIfNeeded();
        await first.locator('input[id$="-lr"]').fill('0.16'); await seek(first);
        assert.match(await first.innerText(), /rate 0\.160/);
        if (width === 320) await first.screenshot({ path: `${root}/sgd-double-rate-${theme}.png` });
        const race = figures.last(); await race.scrollIntoViewIfNeeded();
        await race.locator('input[id$="-lr"]').fill('1.2'); await seek(race);
        assert.match(await race.innerText(), /outside plot/);
        if (width === 320) await race.screenshot({ path: `${root}/race-extreme-${theme}.png` });
      }
      if (slug === 'weight-initialization') {
        const he = figures.nth(3); await he.scrollIntoViewIfNeeded(); await seek(he);
        assert(Math.abs(Number(await he.getByRole('slider', { name: 'Weight scale multiplier' }).inputValue()) - Math.SQRT2) < 1e-12, 'He preset remains sqrt(2)');
        assert.equal(await he.locator('.scale output').innerText(), '1.41');
        if (width === 320) await he.screenshot({ path: `${root}/he-exact-${theme}.png` });
      }
      if (slug === 'learning-rate-schedules') {
        const first = figures.first(); await first.scrollIntoViewIfNeeded();
        await first.getByLabel('Include the too-large rate (0.7)').check();
        await first.locator('select[id$="-focus"]').selectOption('2');
        await first.getByLabel('Include the too-large rate (0.7)').uncheck();
        assert(await first.locator('select[id$="-focus"]').inputValue(), 'a focused run remains selected');
      }
      await page.locator('details').evaluateAll(ds => ds.forEach(d => { d.open = true; }));
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) {
        console.log(await page.evaluate(() => [...document.querySelectorAll('main *')].map(el => ({ tag: el.tagName, cls: el.className, right: el.getBoundingClientRect().right, width: el.getBoundingClientRect().width, text: el.textContent.slice(0, 80) })).filter(el => el.right > innerWidth + 1).slice(0, 12)));
        await page.screenshot({ path: `${root}/overflow-${slug}-${width}-${theme}.png`, fullPage: true });
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${slug}/${width}/${theme} overflow`);
      for (const figure of await figures.all()) assert.equal(await figure.locator('.fig').evaluate(el => el.scrollHeight > el.clientHeight + 2), false, `${slug}/${width}/${theme} stage height`);
      results.push({ slug, width, theme, overflowX: false });
    }
    await page.close();
  }
  assert.deepEqual(errors, []); passed = true;
} finally { await browser.close(); writeFileSync(`${root}/control-results.json`, JSON.stringify({ passed, results, errors }, null, 2)); }
console.log(JSON.stringify({ passed, layouts: results.length, errors }));
