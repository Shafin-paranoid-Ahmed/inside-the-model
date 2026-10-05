import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One episode, and the return from every move',
  legend: [
    { label: 'Policy arrow', color: '--text-muted', shape: 'line' },
    { label: 'Path walked', color: '--viz-forward', shape: 'line' },
    { label: 'Return (found backward)', color: '--viz-gradient' },
  ],
  aspect: '720 / 380',
  lanes: 1,
  narrowAspect: '360 / 490',
  controls: [
    { key: 'gamma', label: 'Discount γ', kind: 'select', options: [{ value: 0.9, label: '0.9' }, { value: 1, label: '1 (none)' }, { value: 0.5, label: '0.5' }], then: 'end' },
  ],
} satisfies ModeMeta;
