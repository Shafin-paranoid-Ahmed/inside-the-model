import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'DPO against PPO: the same reward, two ways to chase it',
  legend: [
    { label: 'PPO (solid line)', color: '--text', shape: 'line' },
    { label: 'DPO (dashed line)', color: '--text', shape: 'line' },
    { label: 'Chosen responses', color: '--viz-pos', shape: 'line' },
    { label: 'Rejected responses', color: '--viz-neg', shape: 'line' },
  ],
  aspect: '720 / 380',
  narrowAspect: '360 / 760',
  lanes: 2,
  loading: 'Training PPO and DPO at two penalty strengths…',
} satisfies ModeMeta;
