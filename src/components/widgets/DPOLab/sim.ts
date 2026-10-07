/**
 * DPO lesson modes use the same response bank, preference pairs and training core as RLHF.
 * The separate shell keeps each island below 40 KB gzip without duplicating the policy or fitting helpers.
 * Mode contract: ../GridWorldLab/sim.ts. Model and worker helpers: ../PreferenceLab/sim.ts.
 */
export { prefData, ppoRun, type PrefData, type Opts } from '../PreferenceLab/sim';
