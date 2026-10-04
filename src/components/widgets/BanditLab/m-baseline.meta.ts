import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The same gradient estimated with and without a baseline',
  legend: [
    { label: 'No baseline', color: '--viz-cat-1' },
    { label: 'With baseline', color: '--viz-cat-2' },
    { label: 'Exact gradient', color: '--text', shape: 'line' },
    { label: 'Average so far', color: '--accent', shape: 'line' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 600',
  lanes: 2,
  controls: [{ key: 'arm', label: 'Logit of arm', kind: 'select', options: [{ value: 1, label: 'Arm 1 (mean 1)' }, { value: 2, label: 'Arm 2 (mean 2)' }, { value: 3, label: 'Arm 3 (mean 3)' }] }],
} satisfies ModeMeta;
