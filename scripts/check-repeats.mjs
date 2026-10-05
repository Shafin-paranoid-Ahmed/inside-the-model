// Fails when a lesson shows the same widget with the same props twice (title aside).
//   node scripts/check-repeats.mjs
// Rule: plans/audit-2026-10-05.md "no two figures in a lesson may have identical props apart from the title".
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = 'src/content/lessons';
// Repeats the owner accepted: the figure is there so the reader can try something, and the step says so.
const allowed = new Set([]);
let bad = 0;
for (const track of readdirSync(root).filter((d) => !d.startsWith('.'))) {
  for (const file of readdirSync(join(root, track)).filter((f) => f.endsWith('.mdx'))) {
    const seen = new Map();
    const src = readFileSync(join(root, track, file), 'utf8');
    for (const m of src.matchAll(/<([A-Z]\w+)\s+client:visible([^>]*?)\/>/gs)) {
      const props = m[2].replace(/\btitle=("[^"]*"|\{[^}]*\})/, '').replace(/\s+/g, ' ').trim().split(' ').sort().join(' ');
      const key = `${m[1]} ${props}`;
      const line = src.slice(0, m.index).split('\n').length;
      if (seen.has(key) && !allowed.has(`${track}/${file}:${line}`)) {
        console.log(`${track}/${file}:${line} repeats line ${seen.get(key)}: <${key}>`);
        bad++;
      } else seen.set(key, line);
    }
  }
}
console.log(bad ? `${bad} repeated figure(s)` : 'no repeated figures');
process.exit(bad ? 1 : 0);
