import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Q-learning from sampled moves',
  legend: [
    { label: 'Agent', color: '--accent' },
    { label: 'Updated entry', color: '--text', shape: 'line' },
    { label: 'Best action so far (outlined)', color: '--accent', shape: 'line' },
    { label: 'Start-cell estimate', color: '--viz-forward', shape: 'line' },
    { label: 'Q* from value iteration', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 450',
  lanes: 2,
  narrowAspect: '360 / 680',
} satisfies ModeMeta;
