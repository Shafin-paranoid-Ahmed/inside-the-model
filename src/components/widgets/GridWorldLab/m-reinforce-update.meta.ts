import type { ModeMeta } from '../_rl/kit';

const cells = [[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [1, 3], [2, 3], [3, 3]];

export default {
  title: 'One REINFORCE update, from episode to new policy',
  legend: [
    { label: 'Path of the episode', color: '--viz-forward', shape: 'line' },
    { label: 'Better than usual (positive advantage)', color: '--viz-pos' },
    { label: 'Worse than usual (negative advantage)', color: '--viz-neg' },
    { label: 'Push on the logit', color: '--viz-gradient' },
    { label: 'Chance of the action', color: '--viz-forward' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 700',
  lanes: 2,
  controls: [{ key: 'focus', label: 'Policy shown for cell', kind: 'select', options: cells.map(([x, y]) => ({ value: `${x},${y}`, label: `(${x},${y})` })) }],
} satisfies ModeMeta;
