// Exercise every step of selected lessons, including lazy hydration and timeline controls.
// node scripts/check-curriculum.mjs --base http://localhost:4321 --mobile --reduced math/vectors
import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';

const args = process.argv.slice(2);
const baseIndex = args.indexOf('--base');
const base = baseIndex < 0 ? 'http://localhost:4321' : args[baseIndex + 1];
const lessons = args.filter((arg, i) => !arg.startsWith('--') && (baseIndex < 0 || i !== baseIndex + 1));
if (!lessons.length) throw new Error('Pass at least one lesson id, e.g. math/vectors');
const mobile = args.includes('--mobile'), reduced = args.includes('--reduced');
const mode = `${mobile ? 'mobile' : 'desktop'}-${reduced ? 'reduced' : 'motion'}`;
const directory = `.playwright-mcp/curriculum/${mode}`;
mkdirSync(directory, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const reports = [];

try {
  for (const lesson of lessons) {
    for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({
        viewport: mobile ? { width: 390, height: 844 } : { width: 1280, height: 900 },
        deviceScaleFactor: 1, colorScheme: theme,
        reducedMotion: reduced ? 'reduce' : 'no-preference',
      });
      const page = await context.newPage();
      const report = { lesson, theme, mode, steps: 0, figures: 0, errors: [], screenshots: [] };
      page.on('pageerror', error => report.errors.push(error.message));
      page.on('console', message => {
        if (message.type() === 'error') report.errors.push(message.text());
      });
      try {
        const response = await page.goto(new URL(`/learn/${lesson}/`, base).href, { waitUntil: 'networkidle' });
        if (response.status() !== 200) throw new Error(`HTTP ${response.status()}`);
        if (await page.locator('main h1').count() !== 1) throw new Error('Expected one main h1');
        await page.evaluate(() => document.fonts.ready);
        report.steps = await page.locator('[data-step]').count();
        if (report.steps < 6) throw new Error('Expected a full lesson with at least six steps');

        for (let step = 1; step <= report.steps; step++) {
          const active = page.locator('[data-step].is-active');
          const figures = active.locator('[data-figure]');
          const count = await figures.count();
          report.figures += count;
          for (let index = 0; index < count; index++) {
            const figure = figures.nth(index);
            // Keep the drawing in view; tall mobile figures need not fit entirely in the viewport.
            await figure.locator('.fig').scrollIntoViewIfNeeded();
            await figure.evaluate(async el => {
              const island = el.closest('astro-island');
              if (island?.hasAttribute('ssr')) {
                await Promise.race([
                  new Promise(resolve => island.addEventListener('astro:hydrate', resolve, { once: true })),
                  new Promise((_, reject) => setTimeout(() => reject(new Error('Island did not hydrate')), 10000)),
                ]);
              }
            });
            await page.waitForTimeout(250);
            const play = figure.locator('.tl .play');
            const scrub = figure.locator('.tl input[type=range]');
            if (await play.count() !== 1 || await scrub.count() !== 1) {
              report.errors.push(`Step ${step}: missing timeline controls`);
              continue;
            }
            if (reduced && (await play.textContent()).trim() !== 'Play') {
              report.errors.push(`Step ${step}: reduced motion autoplayed`);
            }
            // Seek using the native input event and confirm the public paused state.
            await scrub.evaluate(el => {
              el.value = String(Number(el.max) * 0.62);
              el.dispatchEvent(new Event('input', { bubbles: true }));
            });
            await page.waitForTimeout(100);
            if ((await play.textContent()).trim() !== 'Play') report.errors.push(`Step ${step}: seeking did not pause`);
            const seekValue = Number(await scrub.inputValue());
            const seekExpected = Number(await scrub.getAttribute('max')) * 0.62;
            if (Math.abs(seekValue - seekExpected) > 0.05) report.errors.push(`Step ${step}: timeline seek failed`);

            if (step === 1 && index === 0) {
              const shot = `${directory}/${lesson.replaceAll('/', '_')}-${theme}.png`;
              await figure.screenshot({ path: shot });
              report.screenshots.push(shot);
            }
            // Exercise resume and a real parameter control, including keyboard input.
            await play.click();
            await figure.locator('.fig').scrollIntoViewIfNeeded();
            await page.waitForTimeout(200);
            if (Number(await scrub.inputValue()) <= seekValue) report.errors.push(`Step ${step}: resumed timeline did not advance`);
            const parameters = figure.locator('input[type=range]:not(.scrub)');
            if (await parameters.count()) {
              const parameter = parameters.first();
              await parameter.focus();
              await parameter.press('ArrowRight');
              if ((await active.count()) !== 1 || (await page.locator('[data-count]').textContent()).trim() !== `${step} / ${report.steps}`) {
                report.errors.push(`Step ${step}: parameter arrow key navigated the lesson`);
              }
            }
          }
          const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
          if (overflow) report.errors.push(`Step ${step}: horizontal overflow`);
          if (step < report.steps) await page.locator('[data-next]').click();
        }
        if (!report.figures) report.errors.push('No interactive figures');
        // Deep links must show the requested step on a fresh load.
        await page.goto(new URL(`/learn/${lesson}/?step=2`, base).href, { waitUntil: 'networkidle' });
        if ((await page.locator('[data-count]').textContent()).trim() !== `2 / ${report.steps}`) report.errors.push('Step deep link failed');
      } catch (error) {
        report.errors.push(error.message);
      } finally {
        reports.push(report);
        console.log(JSON.stringify(report));
        await context.close();
      }
    }
  }
} finally {
  await browser.close();
  writeFileSync(`${directory}/report-${lessons.join('_').replaceAll('/', '_')}.json`, JSON.stringify(reports, null, 2));
}
process.exitCode = reports.some(report => report.errors.length) ? 1 : 0;
