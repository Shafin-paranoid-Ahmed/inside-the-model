import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Four ways to train the same network',
  legend: [
    { label: 'Neither', color: '--viz-cat-1', shape: 'line' },
    { label: 'Replay only', color: '--viz-cat-2', shape: 'line' },
    { label: 'Target network only', color: '--viz-cat-3', shape: 'line' },
    { label: 'Replay + target', color: '--viz-cat-4', shape: 'line' },
    { label: 'Best possible', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 600',
  lanes: 2,
  loading: 'Training 4 variants × 10 runs in your browser (about 10 seconds)…',
  controls: [
    { key: 'lr', label: 'Adam learning rate', kind: 'select', options: [{ value: 0.001, label: '0.001' }, { value: 0.01, label: '0.01' }] },
  ],
} satisfies ModeMeta;
