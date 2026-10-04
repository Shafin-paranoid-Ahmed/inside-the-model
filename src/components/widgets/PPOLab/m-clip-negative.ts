/** clip-negative (ppo step 4): the same sweep as clip-positive with a negative advantage (see clipMode there). */
import { clipMode } from './m-clip-positive';

export default clipMode(-1, '0.6,1.4');
