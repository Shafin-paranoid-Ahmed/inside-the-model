import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'A leaked test: public score against fresh score',
  legend: [
    { label: 'Answered right', color: '--viz-pos' },
    { label: 'Answered wrong', color: '--viz-neg' },
    { label: 'Leaked question (ringed, dotted)', color: '--text' },
    { label: 'Public test score', color: '--viz-cat-3', shape: 'line' },
    { label: 'Fresh test score', color: '--viz-cat-7', shape: 'line' },
  ],
  controls: [{ key: 'leakPct', label: 'Share of the public test that leaked', kind: 'range', min: 0, max: 50, step: 5, digits: 0, unit: '%' }],
  aspect: '720 / 400',
  narrowAspect: '360 / 600',
  lanes: 2,
} satisfies ModeMeta;
