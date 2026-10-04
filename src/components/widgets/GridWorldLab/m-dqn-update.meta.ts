import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One transition, one update',
  legend: [
    { label: 'Positive Q', color: '--viz-pos' },
    { label: 'Negative Q', color: '--viz-neg' },
    { label: 'Agent', color: '--accent' },
    { label: 'Error', color: '--viz-gradient', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 640',
  lanes: 2,
  controls: [{ key: 'lr', label: 'Learning rate', kind: 'range', min: 0.01, max: 0.2, step: 0.01, digits: 2 }],
} satisfies ModeMeta;
