import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
const base = process.argv[2] || 'http://127.0.0.1:4321';
const root = '.playwright-mcp/check/day6-teaching';
mkdirSync(root, { recursive: true });
const slugs = ['optimizers', 'learning-rate-schedules', 'weight-initialization', 'normalization', 'dropout-regularizers', 'loss-landscapes'];
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const results = [], errors = [];
const seek = async (figure, fraction) => {
  const range = figure.locator('input.scrub');
  await range.waitFor();
  await range.evaluate((input, fraction) => { input.value = String(Number(input.max) * fraction); input.dispatchEvent(new Event('input', { bubbles: true })); }, fraction);
  await figure.page().waitForTimeout(80);
};
let passed = false;
try {
  for (const width of [390, 1280, 320]) for (const theme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport: { width, height: 844 }, reducedMotion: 'reduce', colorScheme: theme });
    const page = await context.newPage();
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    for (const slug of slugs) {
      const response = await page.goto(`${base}/learn/neural-networks/${slug}/`, { waitUntil: 'networkidle' });
      assert.equal(response.status(), 200, `${slug} HTTP`);
      const figures = page.locator('main [data-figure]');
      const count = await figures.count();
      assert(count > 0, `${slug} figures`);
      for (let i = 0; i < count; i++) {
        const figure = figures.nth(i);
        await figure.scrollIntoViewIfNeeded();
        const collisions = await figure.locator('.tick').evaluateAll(ticks => {
          const boxes = ticks.map(el => ({ text: el.textContent, box: el.getBoundingClientRect() })).sort((a, b) => a.box.left - b.box.left);
          return boxes.slice(1).filter((item, i) => item.box.left < boxes[i].box.right - 1).map(item => item.text);
        });
        assert.deepEqual(collisions, [], `${slug}/${width}/${theme}/${i} timeline labels overlap`);
        for (const fraction of [0, 0.5, 1]) {
          await seek(figure, fraction);
          assert(!/NaN|Infinity/.test(await figure.innerText()), `${slug}/${i} finite values`);
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${slug}/${width}/${theme}/${i} overflow`);
          const stage = figure.locator('.fig');
          assert.equal(await stage.evaluate(el => el.scrollHeight > el.clientHeight + 2), false, `${slug}/${width}/${theme}/${i} stage height`);
          if (width !== 320) await figure.screenshot({ path: `${root}/${slug}-${width}-${theme}-figure${i + 1}-${fraction}.png` });
        }
        assert.match(await figure.innerText(), /Try this\./, `${slug}/${i} guided interaction`);
      }
      if (slug === 'optimizers') {
        const momentum = figures.nth(2); await momentum.scrollIntoViewIfNeeded();
        const update = momentum.locator('input[id$="-update"]');
        await update.fill('1'); await seek(momentum, 1); assert.match(await momentum.innerText(), /· update 1(?!\d)/);
        await update.fill('8'); await seek(momentum, 1); assert.match(await momentum.innerText(), /· update 8(?!\d)/);
        const noise = figures.nth(1); await noise.scrollIntoViewIfNeeded();
        await noise.locator('select[id$="-batch"]').selectOption('16'); await seek(noise, 0.5);
      }
      if (slug === 'learning-rate-schedules') {
        const first = figures.first(); await first.scrollIntoViewIfNeeded();
        await first.getByLabel('Include the too-large rate (0.7)').check(); await seek(first, 1);
        assert.match(await first.innerText(), /Too large/);
        if (width === 390) await first.screenshot({ path: `${root}/unstable-${theme}.png` });
      }
      if (slug === 'normalization') {
        const first = figures.first(); await first.scrollIntoViewIfNeeded();
        await first.locator('details.advanced summary').click();
        for (const input of await first.locator('input[type=number]').all()) {
          for (const value of ['', '999', '-999']) {
            await input.fill(value); await input.blur();
            const actual = Number(await input.inputValue());
            assert(Number.isFinite(actual));
            assert(actual >= Number(await input.getAttribute('min')) && actual <= Number(await input.getAttribute('max')));
          }
        }
        const batch = figures.nth(1); await batch.scrollIntoViewIfNeeded(); await seek(batch, 1);
        const before = await batch.locator('.worked').innerText();
        await batch.getByLabel('Companion example offset').fill('3');
        await page.waitForTimeout(100); assert.notEqual(await batch.locator('.worked').innerText(), before);
      }
      if (slug === 'loss-landscapes') {
        const first = figures.first(); await first.scrollIntoViewIfNeeded();
        await first.getByLabel('Starting side').selectOption('left'); await seek(first, 1);
        assert.match(await first.innerText(), /\(−?-/);
        await first.locator('select[id$="-view"]').selectOption('map');
        if (width === 390) await first.screenshot({ path: `${root}/landscape-map-${theme}.png` });
      }
      await page.locator('main details').evaluateAll(ds => ds.forEach(d => { d.open = true; }));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${slug}/${width}/${theme} expanded details`);
      results.push({ slug, width, theme, figures: count, overflowX: false });
      console.log(`${slug} ${width} ${theme}: ${count} figures checked`);
    }
    await context.close();
  }
  // Motion and keyboard controls still work with the teaching views.
  const page = await browser.newPage({ reducedMotion: 'no-preference' });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${base}/learn/neural-networks/optimizers/`, { waitUntil: 'networkidle' });
  const figure = page.locator('main [data-figure]').first(); await figure.scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);
  assert(await figure.getByRole('button', { name: 'Pause', exact: true }).count(), 'autoplay');
  await figure.getByRole('button', { name: 'Pause', exact: true }).click();
  const range = figure.locator('input.scrub'); await range.focus();
  await range.press('End'); await page.waitForTimeout(100);
  assert.equal(Number(await range.inputValue()), Number(await range.getAttribute('max')), 'keyboard seek');
  await figure.getByRole('button', { name: 'Play', exact: true }).click();
  const timing = await page.evaluate(() => new Promise(resolve => {
    let count = 0, maxGap = 0, previous = performance.now(); const start = previous;
    const frame = now => { count++; maxGap = Math.max(maxGap, now - previous); previous = now; if (now - start < 1000) requestAnimationFrame(frame); else resolve({ fps: count, maxGap }); }; requestAnimationFrame(frame);
  }));
  console.log(JSON.stringify({ timing }));
  await page.close();
  assert.deepEqual(errors, []); passed = true;
} finally {
  await browser.close();
  writeFileSync(`${root}/results.json`, JSON.stringify({ passed, results, errors }, null, 2));
}
console.log(JSON.stringify({ passed, layouts: results.length, errors }));
