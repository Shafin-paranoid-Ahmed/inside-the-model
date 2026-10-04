/// <reference types="vitest/config" />
import { getViteConfig } from 'astro/config';

export default getViteConfig({
  // Bound CPU contention on Windows; keep the normal per-test timeout intact.
  test: { include: ['tests/unit/**/*.test.ts'], maxWorkers: 2 },
});
