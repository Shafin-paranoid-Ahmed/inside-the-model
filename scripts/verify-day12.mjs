// Day 12 review evidence: fixed frames, mobile layouts, controls, keyboard seeking and motion.
// node scripts/verify-day12.mjs --base http://localhost:4322 [--lesson sft] [--width 390]
import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const base = option('--base', 'http://localhost:4322');
const only = option('--lesson', null), width = option('--width', null);
const widths = width ? [Number(width)] : [1280, 390];
const lessons = ['sft', 'rlhf', 'dpo', 'constitutional-ai', 'rlvr', 'evaluations'].filter(s => !only || s === only);
assert(lessons.length, 'Unknown Day 12 lesson');
const root = '.playwright-mcp/check/day12-acceptance';
mkdirSync(root, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const results = [], errors = [];
const seek = async (range, value) => {
  await range.evaluate((el, v) => { el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true })); }, value);
};
const ready = async figure => {
  await figure.locator('input.scrub').waitFor();
  await figure.page().waitForFunction(el => !el.querySelector('.loading') && el.querySelector('.status')?.textContent.trim(), await figure.elementHandle());
};
async function inspect(figure) {
  return figure.evaluate(el => {
    const svg = el.querySelector('.stagebox svg'), box = svg.getBoundingClientRect();
    const outside = [], small = [];
    for (const t of svg.querySelectorAll('text')) {
      if (Number(getComputedStyle(t).opacity) < 0.05 || !t.textContent.trim()) continue;
      const r = t.getBoundingClientRect(), size = parseFloat(getComputedStyle(t).fontSize);
      if (r.left < box.left - 2 || r.right > box.right + 2 || r.top < box.top - 2 || r.bottom > box.bottom + 2) outside.push(t.textContent);
      if (size < 11) small.push(t.textContent);
    }
    return {
      status: el.querySelector('.status')?.textContent,
      readouts: el.querySelector('.ro')?.textContent,
      active: el.querySelector('.tick.on')?.textContent,
      outside, small,
      overflow: document.documentElement.scrollWidth > innerWidth + 1,
      captionOutside: [...el.querySelectorAll('.status, .ro, .detail, .ctls')].some(e => e.getBoundingClientRect().bottom > el.getBoundingClientRect().bottom + 2),
    };
  });
}
try {
  for (const slug of lessons) for (const w of widths) for (const theme of ['light', 'dark']) {
    const name = `${slug}/${w}/${theme}`, shots = `${root}/${slug}`;
    mkdirSync(shots, { recursive: true });
    const ctx = await browser.newContext({ viewport: { width: w, height: 900 }, deviceScaleFactor: 1, colorScheme: theme, reducedMotion: 'reduce' });
    const page = await ctx.newPage(), row = { slug, width: w, theme, figures: [] };
    page.on('pageerror', e => errors.push(`${name}: ${e.message}`));
    page.on('console', m => { if (m.type() === 'error') errors.push(`${name}: ${m.text()}`); });
    const response = await page.goto(`${base}/learn/alignment/${slug}/`, { waitUntil: 'networkidle' });
    assert.equal(response.status(), 200);
    assert.equal(await page.locator('main h1').count(), 1);
    const figures = page.locator('main [data-figure]'), count = await figures.count();
    assert(count > 0);
    for (let i = 0; i < count; i++) {
      const figure = figures.nth(i), record = { index: i + 1, frames: [], controls: 0 };
      await figure.scrollIntoViewIfNeeded();
      await ready(figure);
      const play = figure.locator('button.play'), range = figure.locator('input.scrub');
      assert.equal(await play.innerText(), 'Play', `${name}/${i + 1}: reduced motion starts paused`);
      const max = Number(await range.getAttribute('max'));
      for (const fraction of [0, 0.2, 0.55, 1]) {
        await seek(range, max * fraction);
        await page.waitForTimeout(100);
        const frame = await inspect(figure);
        frame.fraction = fraction;
        frame.shot = `${shots}/${theme}-${w}-figure${i + 1}-t${fraction}.png`;
        await figure.screenshot({ path: frame.shot });
        if (frame.overflow || frame.captionOutside || frame.outside.length || frame.small.length) errors.push(`${name}/${i + 1}@${fraction}: ${JSON.stringify({ overflow: frame.overflow, captionOutside: frame.captionOutside, outside: frame.outside, small: frame.small })}`);
        record.frames.push(frame);
      }
      await seek(range, max / 2);
      await range.focus();
      const before = Number(await range.inputValue());
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(100);
      assert(Number(await range.inputValue()) > before, `${name}/${i + 1}: keyboard seek`);
      await figure.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.equal(await play.innerText(), 'Play');
      // Exercise mode controls once per width. The other theme still covers every default frame.
      if (theme === 'light') {
        for (const control of await figure.locator('.ctls input, .ctls select').all()) {
          const kind = await control.evaluate(el => el.tagName === 'SELECT' ? 'select' : el.type);
          if (kind === 'checkbox') {
            const original = await control.isChecked();
            await control.setChecked(!original); await page.waitForTimeout(600);
            await seek(range, Number(await range.getAttribute('max'))); await page.waitForTimeout(100);
            const state = await inspect(figure);
            if (state.overflow || state.outside.length) errors.push(`${name}/${i + 1}/toggle: ${JSON.stringify(state)}`);
            await figure.screenshot({ path: `${shots}/${theme}-${w}-figure${i + 1}-control${record.controls}.png` });
            await control.setChecked(original);
          } else if (kind === 'select') {
            const original = await control.inputValue(), values = await control.locator('option').evaluateAll(opts => opts.map(o => o.value));
            for (const value of values) { await control.selectOption(value); await page.waitForTimeout(300); }
            await control.selectOption(original);
          } else if (kind === 'range') {
            const original = await control.inputValue();
            for (const edge of ['min', 'max']) {
              await seek(control, Number(await control.getAttribute(edge))); await page.waitForTimeout(600);
              await seek(range, Number(await range.getAttribute('max'))); await page.waitForTimeout(100);
              const state = await inspect(figure);
              if (state.outside.length || state.overflow) errors.push(`${name}/${i + 1}/${edge}: ${JSON.stringify(state)}`);
              await figure.screenshot({ path: `${shots}/${theme}-${w}-figure${i + 1}-control${record.controls}-${edge}.png` });
            }
            await seek(control, original);
          }
          record.controls++;
          await page.waitForTimeout(500);
          await ready(figure);
        }
        // Click two samples in pickable plots; these are part of the lesson's teaching interactions.
        if (await figure.locator('svg.pickable').count()) {
          await seek(range, Number(await range.getAttribute('max')));
          await page.waitForTimeout(150);
          const choices = figure.locator('svg.pickable [data-i]');
          const count = await choices.count();
          assert(count > 0, `${name}/${i + 1}: selectable samples exist`);
          for (const index of [0, Math.floor(count / 2)]) {
            await choices.nth(index).dispatchEvent('pointerdown');
            await page.waitForTimeout(150);
            const state = await inspect(figure);
            assert(state.status?.length, `${name}/${i + 1}: sample selection preserves narration`);
            if (state.outside.length || state.overflow) errors.push(`${name}/${i + 1}/pick: ${JSON.stringify(state)}`);
          }
          await figure.screenshot({ path: `${shots}/${theme}-${w}-figure${i + 1}-picked.png` });
        }
        // Check autoplay and pause after explicitly enabling normal motion.
        await page.emulateMedia({ reducedMotion: 'no-preference' });
        await seek(range, 0);
        await play.click(); await page.waitForTimeout(300);
        assert(Number(await range.inputValue()) > 0, `${name}/${i + 1}: play`);
        await play.click();
        const paused = await range.inputValue(); await page.waitForTimeout(160);
        assert.equal(await range.inputValue(), paused, `${name}/${i + 1}: pause`);
        await page.emulateMedia({ reducedMotion: 'reduce' });
      }
      row.figures.push(record);
    }
    for (const details of await page.locator('main details').all()) await details.evaluate(el => { el.open = true; });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${name}: expanded details overflow`);
    results.push(row);
    console.log(`${name}: ${count} figures checked`);
    await ctx.close();
  }
} catch (error) {
  errors.push(String(error.stack ?? error));
} finally {
  await browser.close();
  const file = `${root}/results${only ? `-${only}` : ''}${width ? `-${width}` : ''}.json`;
  writeFileSync(file, JSON.stringify({ base, results, errors }, null, 2));
  console.log(JSON.stringify({ file, scenarios: results.length, errors }, null, 2));
  process.exitCode = errors.length ? 1 : 0;
}
