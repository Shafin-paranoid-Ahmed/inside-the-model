import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'A softmax policy learns which arm pays most',
  legend: [
    { label: 'Arm 1', color: '--viz-cat-1', shape: 'line' },
    { label: 'Arm 2', color: '--viz-cat-2', shape: 'line' },
    { label: 'Arm 3', color: '--viz-cat-3', shape: 'line' },
    { label: 'Push from this pull', color: '--viz-gradient' },
    { label: 'Average push (exact)', color: '--text-muted' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 640',
  lanes: 2,
  controls: [{ key: 'lr', label: 'Learning rate', kind: 'range', min: 0.02, max: 0.5, step: 0.01, digits: 2 }],
} satisfies ModeMeta;
