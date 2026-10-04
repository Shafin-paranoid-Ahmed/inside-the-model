import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Where one move can end up',
  legend: [
    { label: 'Chance of each outcome', color: '--viz-forward', shape: 'line' },
    { label: 'One sampled try', color: '--viz-forward' },
    { label: 'Cell being asked', color: '--accent' },
  ],
  aspect: '720 / 380',
  lanes: 1,
  narrowAspect: '360 / 560',
  controls: [
    { key: 'focus', label: 'Cell', kind: 'select', options: [{ value: '3,1', label: '(3,1)' }, { value: '2,1', label: '(2,1)' }, { value: '3,2', label: '(3,2)' }, { value: '4,1', label: '(4,1)' }] },
  ],
} satisfies ModeMeta;
