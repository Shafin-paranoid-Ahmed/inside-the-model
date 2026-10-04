// Detailed curriculum acceptance: fixed timeline frames, reduced motion, keyboard,
// theme changes, control changes and normal autoplay in isolated Chrome.
// node scripts/verify-day4.mjs [--day 4|5|6] [--base URL] [--lesson slug] [--width 320]
import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const base = option('--base', 'http://localhost:4321');
const only = option('--lesson', null);
const requestedWidth = option('--width', null);
const widths = requestedWidth ? [Number(requestedWidth)] : [1280, 390];
const day = Number(option('--day', '4'));
const curriculum = {
  4: ['neuron', 'activation-functions', 'mlp', 'hidden-layers', 'universal-approximation'],
  5: ['computational-graphs', 'chain-rule-visually', 'forward-backward-pass', 'backprop-mlp', 'autodiff', 'vanishing-exploding-gradients'],
  6: ['optimizers', 'learning-rate-schedules', 'weight-initialization', 'normalization', 'dropout-regularizers', 'loss-landscapes'],
};
const lessons = (curriculum[day] || []).filter(s => !only || s === only);
assert(lessons.length, 'Unknown day or lesson');
const root = `.playwright-mcp/check/day${day}-acceptance`;
mkdirSync(root, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const results = [];
const errors = [];
let passed = false;
const seek = async (range, value) => {
  await range.evaluate((el, v) => { el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true })); }, value);
};
const overflow = page => page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
try {
  for (const slug of lessons) {
    const shots = `${root}/${slug}`;
    mkdirSync(shots, { recursive: true });
    for (const width of widths) for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({ viewport: { width, height: 844 }, colorScheme: theme, reducedMotion: 'reduce' });
      page.on('pageerror', e => errors.push(`${slug}/${width}/${theme}: ${e.message}`));
      page.on('console', m => { if (m.type() === 'error') errors.push(`${slug}/${width}/${theme}: ${m.text()}`); });
      const response = await page.goto(`${base}/learn/neural-networks/${slug}/`, { waitUntil: 'networkidle' });
      assert.equal(response.status(), 200);
      assert.equal(await page.locator('main h1').count(), 1);
      assert.equal(await page.locator('.banner').count(), 0, 'Lesson is still a stub');
      const figures = page.locator('main [data-figure]');
      const count = await figures.count();
      assert(count > 0);
      for (let i = 0; i < count; i++) {
        const figure = figures.nth(i);
        await figure.scrollIntoViewIfNeeded();
        const play = figure.locator('button.play');
        await play.waitFor();
        // Worker-driven figures may finish after hydration.
        // Figures may use accessible DOM arithmetic instead of a canvas/SVG.
        await figure.locator('.fig').waitFor();
        if (slug === 'hidden-layers') await page.waitForFunction(index => document.querySelectorAll('main [data-figure]')[index]?.querySelector('[aria-live]')?.textContent.trim().startsWith('Phase:'), i);
        await page.waitForTimeout(slug === 'hidden-layers' ? 1200 : 180);
        assert.equal(await play.innerText(), 'Play', 'Reduced motion must start paused');
        const range = figure.locator('input.scrub');
        const max = Number(await range.getAttribute('max'));
        for (const fraction of [0.18, 0.53, 0.94]) {
          await seek(range, max * fraction);
          await page.waitForTimeout(120);
          assert.equal(await play.innerText(), 'Play');
          assert.equal(await overflow(page), false, `${slug} overflows at ${width}`);
          await figure.screenshot({ path: `${shots}/${theme}-${width}-figure${i + 1}-t${fraction}.png` });
        }
        await range.focus();
        const before = Number(await range.inputValue());
        await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(700);
        assert(Number(await range.inputValue()) > before, 'Keyboard scrub did not advance');
        await figure.getByRole('button', { name: 'Reset', exact: true }).click();
        await page.waitForTimeout(120);
        assert.equal(await play.innerText(), 'Play', 'Reduced-motion Reset should remain paused');
        // Controls remain useful without autoplay. Switch each available radio.
        for (const radio of await figure.locator('input[type=radio]').all()) {
          await radio.check();
          await page.waitForTimeout(80);
          assert.equal(await overflow(page), false);
        }
        if (day === 6) {
          for (const select of await figure.locator('select:not(.tl select)').all()) {
            const values = await select.locator('option').evaluateAll(options => options.map(o => o.value));
            for (const value of values) {
              await select.selectOption(value);
              await page.waitForTimeout(80);
              assert.equal(await overflow(page), false, `${slug}: select ${value} overflows`);
            }
          }
          for (const parameter of await figure.locator('input[type=range]:not(.scrub)').all()) {
            for (const bound of ['min', 'max']) {
              await parameter.evaluate((el, name) => {
                el.value = el.getAttribute(name);
                el.dispatchEvent(new Event('input', { bubbles: true }));
              }, bound);
              await seek(range, max * 0.94);
              await page.waitForTimeout(100);
              assert.equal(await overflow(page), false, `${slug}: ${bound} parameter overflows`);
            }
          }
          for (const checkbox of await figure.locator('input[type=checkbox]').all()) {
            await checkbox.check(); await page.waitForTimeout(100);
            assert.equal(await overflow(page), false);
            await checkbox.uncheck();
          }
        }
      }
      // A live theme change must retain a usable figure and scrubber.
      await page.getByRole('button', { name: 'Toggle light and dark theme' }).click();
      await page.waitForTimeout(100);
      assert.equal(await overflow(page), false);
      await page.locator('details').evaluateAll(details => details.forEach(el => { el.open = true; }));
      assert.equal(await overflow(page), false, 'Expanded mathematics overflows');
      results.push({ slug, width, theme, figures: count, reducedMotion: 'paused and keyboard scrubbable', overflowX: false });
      console.log(JSON.stringify(results.at(-1)));
      await page.close();
    }
    // Normal motion: every island must actually advance after entering view.
    const page = await browser.newPage({ viewport: { width: 1280, height: 844 }, reducedMotion: 'no-preference' });
    page.on('pageerror', e => errors.push(`${slug}/autoplay: ${e.message}`));
    await page.goto(`${base}/learn/neural-networks/${slug}/`, { waitUntil: 'networkidle' });
    const figures = page.locator('main [data-figure]');
    for (let i = 0; i < await figures.count(); i++) {
      const fig = figures.nth(i);
      await fig.scrollIntoViewIfNeeded();
      const play = fig.locator('button.play');
      await play.waitFor();
      if (slug === 'hidden-layers') await page.waitForFunction(index => document.querySelectorAll('main [data-figure]')[index]?.querySelector('[aria-live]')?.textContent.trim().startsWith('Phase:'), i);
      await page.waitForTimeout(slug === 'hidden-layers' ? 1800 : 300);
      assert.equal(await play.innerText(), 'Pause', 'Normal motion must autoplay');
      const range = fig.locator('input.scrub');
      const before = Number(await range.inputValue());
      // Phase-only demonstrations deliberately hold their last frame for 0.7 s.
      await page.waitForTimeout(1000);
      assert.notEqual(Number(await range.inputValue()), before, 'Autoplay timeline is frozen');
      await play.click();
      await page.waitForTimeout(100);
      const stopped = Number(await range.inputValue());
      await page.waitForTimeout(140);
      assert.equal(Number(await range.inputValue()), stopped, 'Paused timeline advanced');
    }
    await page.close();
  }
  assert.deepEqual(errors, [], 'Browser errors');
  passed = true;
} catch (error) {
  errors.push(error.message);
  throw error;
} finally {
  await browser.close();
  writeFileSync(`${root}/results${only ? `-${only}` : ''}${requestedWidth ? `-${requestedWidth}` : ''}.json`, JSON.stringify({ day, passed, results, errors }, null, 2));
}
