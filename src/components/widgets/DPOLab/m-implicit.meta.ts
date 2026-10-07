import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The reward hidden inside the DPO policy',
  legend: [
    { label: 'One candidate answer', color: '--viz-forward', shape: 'dot' },
    { label: 'Rambling wrong answer', color: '--accent', shape: 'dot' },
    { label: 'Implicit weight (bar)', color: '--viz-pos' },
    { label: 'Reward model weight (tick)', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 700',
  lanes: 2,
  loading: 'Training DPO on 200 pairs…',
} satisfies ModeMeta;
