// Production island gzip sizes, including their emitted local JS dependencies.
// For widgets that lazily import one chunk per mode, closureGzipKB is the worst single island (static closure + its largest mode).
// node scripts/check-widget-budgets.mjs WidgetName ... (run after coordinator build)
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, basename } from 'node:path';
import { gzipSync } from 'node:zlib';
import assert from 'node:assert/strict';

const dir = 'dist/_astro';
const files = readdirSync(dir);
const names = process.argv.slice(2);
assert(names.length, 'Provide widget component names');
const size = file => gzipSync(readFileSync(join(dir, file))).length;
function closure(entry, seen = new Set()) {
  if (seen.has(entry)) return seen;
  seen.add(entry);
  const code = readFileSync(join(dir, entry), 'utf8');
  // Emitted local chunk references only; excludes absolute worker asset URLs.
  for (const match of code.matchAll(/["']\.\/([^"']+\.js)["']/g)) {
    const dependency = basename(match[1]);
    if (existsSync(join(dir, dependency))) closure(dependency, seen);
  }
  return seen;
}
let failed = false;
for (const name of names) {
  const entry = files.find(file => file.startsWith(`${name}.`) && file.endsWith('.js'));
  assert(entry, `No production chunk for ${name}`);
  const base = closure(entry);
  const kb = (set) => [...set].reduce((sum, file) => sum + size(file), 0);
  // Lazily imported modes (Vite emits import(`./m-x.js`)): an island loads the static closure plus ONE of them.
  const lazy = [...readFileSync(join(dir, entry), 'utf8').matchAll(/import\(`\.\/([^`]+\.js)`\)/g)].map((m) => basename(m[1]));
  const worst = lazy.reduce((max, chunk) => Math.max(max, kb(closure(chunk, new Set(base)))), kb(base));
  const result = { widget: name, ownGzipKB: +(size(entry) / 1024).toFixed(2),
    closureGzipKB: +(worst / 1024).toFixed(2), chunks: base.size, lazyModes: lazy.length, under40KB: worst < 40 * 1024 };
  console.log(JSON.stringify(result));
  if (!result.under40KB) failed = true;
}
process.exitCode = failed ? 1 : 0;
