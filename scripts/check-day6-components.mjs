import fs from 'node:fs';
import { compile } from 'svelte/compiler';
const names = ['OptimizerLab', 'ScheduleLab', 'InitializationLab', 'NormalizationLab', 'DropoutLab', 'LandscapeLab'];
let failed = false;
for (const name of names) {
  const filename = `src/components/widgets/${name}/${name}.svelte`;
  const source = fs.readFileSync(filename, 'utf8');
  try { const result = compile(source, { filename }); console.log(name, result.warnings.map(w => `${w.code}: ${w.message}`).join('\n') || '0 warnings'); if (result.warnings.length) failed = true; }
  catch (e) { console.error(name, e.message); failed = true; }
}
process.exitCode = failed ? 1 : 0;
