import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Chasing a target that moves with the weights',
  legend: [
    { label: 'Learning from its own bootstrapped target', color: '--viz-cat-2', shape: 'line' },
    { label: 'Fitting fixed answers (value iteration)', color: '--viz-cat-1', shape: 'line' },
    { label: 'Largest true Q', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 640',
  lanes: 2,
  loading: 'Training two networks for 1,000 updates in your browser…',
  controls: [{ key: 'lr', label: 'Adam learning rate', kind: 'select', options: [{ value: 0.1, label: '0.1' }, { value: 0.01, label: '0.01' }] }],
} satisfies ModeMeta;
