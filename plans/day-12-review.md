# Day 12 completion and review

Completed 2026-10-07. **All six lessons are coordinator-accepted and wave-checked.** They remain `status: draft` for owner clarity review and the separate release gates. No Day 13 work or core agent was started. No commits or publication were performed.

## Owner gate

Review Day 12 with the owner before advancing. **Explicit permission is required before starting Day 13 or its core agent.** The owner gave standing authorization for local dev/preview servers on 2026-10-07; do not request that permission again. The preview is available at `http://localhost:4322/`.

## Completed lessons

| Lesson | Steps | Figures | Preview |
| --- | ---: | ---: | --- |
| SFT | 9 | 6 | [Open](http://localhost:4322/learn/alignment/sft/) |
| RLHF | 11 | 8 | [Open](http://localhost:4322/learn/alignment/rlhf/) |
| DPO | 8 | 5 | [Open](http://localhost:4322/learn/alignment/dpo/) |
| Constitutional AI | 9 | 6 | [Open](http://localhost:4322/learn/alignment/constitutional-ai/) |
| RLVR | 10 | 7 | [Open](http://localhost:4322/learn/alignment/rlvr/) |
| Evaluations | 9 | 6 | [Open](http://localhost:4322/learn/alignment/evaluations/) |

Total: 56 steps, 38 figures. No brief comments, repeated figures, or broken internal lesson links remain.

## Continuation changes

Claude had completed the shared alignment core and accepted Evaluations; the other five agents stopped during verification. The continuation preserved that work and completed the coordinator review.

- Fixed prose/simulation disagreements: PPO's first epoch has five ratios below its clip band, not zero; SFT claims now describe the demonstrated model and data; DPO comparisons distinguish the two penalty settings. Corrected RLHF's displayed colour names and its code example's gradient reset.
- Clarified RLVR's independently checkable strict verdict, its separate probabilities for open prompts, and the distinction between R1-Zero's rule-based training and R1's broader reward signals.
- Corrected the Evaluations lesson and saturation widget: overlapping individual confidence intervals do not establish a tie. The plot now describes interval overlap without claiming a hypothesis-test result.
- Fixed mobile and desktop labels, chart margins, timeline collisions, the SFT tally height, RLVR style names/update labels, and the initial-versus-current bar legend. Screenshots were inspected at intermediate and final states in both themes and screen sizes.
- Used the planned DPO shell split after PreferenceLab initially measured 43.44 KB gzip. DPO's five modes now live in `DPOLab/`, retaining the shared policy, preference data, and training mathematics.
- Extracted the pure PPO clipping helper into `src/lib/rl/surrogate.ts`, preserving the export from `src/lib/rl/ppo.ts`. This avoids pulling grid-training code into alignment display modes.
- Added reusable `scripts/verify-day12.mjs`; removed the three obsolete `scripts/zz-*-shots.mjs` scratch files. Preserved unrelated owner edits in Base.astro and linear-regression.mdx.

## Validation

- Latest full-suite checkpoint: **982 tests in 83 files passed** (2026-10-06). After subsequent review edits, affected suites passed: 85 tests in the SFT/RLHF/DPO/Evaluations run, 80 in the later RLHF/DPO/RLVR/Evaluations run, and all 11 RLVR tests after the last spacing adjustment. These are overlapping subsets, not additional tests.
- `npm run check`: **0 errors, 0 warnings, 0 hints** across 409 files. The final display-only RLVR adjustment also passed its unit suite and production build.
- Final `npm run build`: **104 pages built**.
- Production widget dependency closures, KB gzip: **SFTLab 35.80; PreferenceLab 39.59; DPOLab 39.01; CritiqueRevise 37.59; VerifierRL 35.89; EvalLab 31.66.** Shared-helper regression budgets: PPOLab 36.09, GridWorldLab 39.86. All pass the existing 40 KB budget checker.
- `verify-day12.mjs`: **24 desktop/mobile/theme scenarios, 608 fixed-frame checks, zero errors**. Also exercised 14 parameter-control instances, clickable sample selection, keyboard seeking, reset, play/pause, reduced motion and expanded details. RLVR was rechecked in all four scenarios after its final adjustment: another 112 frame checks, zero errors.
- `check-page.mjs --steps`, with and without `--mobile`, on every lesson: **12 clean reports covering 24 theme scenarios**, no console errors or horizontal overflow, about **75 fps**, worst measured step stall **67 ms** (below the 100 ms gate). Runs were sequential on the production preview.
- `check-repeats.mjs`, internal content/link checks, and `git diff --check` passed.

## Evidence and reproduction

Evidence is under `.playwright-mcp/check/day12-acceptance/` (local verification artifacts, ignored by Git):

- `results-1280.json`, `results-390.json`, and the final `results-rlvr.json` are the passing interaction/frame reports. The earlier `results.json` intentionally records the first failing pass, superseded by those reports.
- Per-lesson folders hold the screenshots. Names encode theme, viewport, figure, and timeline fraction; control and picked-sample captures are alongside them. Contact sheets summarize reviewed layouts.
- `perf-<lesson>-desktop.json`, `perf-<lesson>-mobile.json`, and `performance-summary.json` contain the passing performance results.
- `build.log`, `check.log`, `review-tests.log`, `final-tests.log`, `rlvr-tests.log`, and `content-check.json` preserve verification output.
- `probes-initial.json` records the continuation's simulation probes: SFT loss 2.97336 to 1.43149, early weight changes around 0.49 in magnitude, and the first PPO epoch's ratio range 0.67717 to 1.10253 with five ratios outside the band. Existing unit tests assert the other lessons' tuned headline values.

With the preview running:

```sh
node scripts/verify-day12.mjs --base http://127.0.0.1:4322
node scripts/check-page.mjs learn/alignment/sft/ --base http://127.0.0.1:4322 --steps
node scripts/check-page.mjs learn/alignment/sft/ --base http://127.0.0.1:4322 --steps --mobile
```

Repeat the last two commands for the other five slugs. Run performance checks sequentially. The measured browser checks do not replace Lighthouse/LCP/INP, a full accessibility audit, or owner review; no release or deployment is implied by coordinator acceptance.
