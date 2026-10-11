# Day 1 and Day 2 mathematical clarity audit — 2026-10-11

This audit records the owner's feedback, the reasons for the subsequent changes, how they were implemented, and the evidence collected during the October 7–8 revision sessions. It was written retrospectively on October 11, 2026, from the conversation, current source, Git history, and saved verification artifacts.

**Scope:** `foundations/linear-regression`, `foundations/gradient-descent`, and `foundations/logistic-regression`. This is a focused follow-up to the [October 5 lesson and figure audit](audit-2026-10-05.md), not a new audit of every Day 1 or Day 2 lesson.

**Status:** all three requested revisions were implemented and technically checked. Owner clarity acceptance of these revisions is not recorded here. The lessons remain drafts; passing checks does not establish that a beginner finds them clear.

## 1. What prompted the changes

The owner identified three connected problems:

1. **Linear regression skipped mathematical steps.** The explanation jumped between formulas and results, leaving the reader unable to reconstruct the calculation.
2. **The transition to gradient descent introduced too much at once.** It did not clearly distinguish the model from the training procedure, and switching to an unrelated loss made gradient descent feel like a separate subject. The owner specifically requested continuity through MSE.
3. **The sigmoid explanation skipped the conversion to probability.** It displayed parameters, a score, and a probability without showing the intervening arithmetic. The owner's example was `w = 1.26`, `b = 0.44`, `x = -4`, score about `-4.61`, and probability about `0.01`.

These were comprehension problems, even where the underlying mathematics was correct. A formula or animated result alone did not explain how the reader could reproduce it.

## 2. Findings and teaching decisions

| Finding | Why it confused the reader | Decision |
|---|---|---|
| A prediction became a loss without enough intermediate work | The learner could not connect a point's error to the average loss | Show prediction, signed error, squared error, sum, and division by the example count |
| Gradient descent moved from regression to a separate quadratic bowl | The learner had to learn new notation and a new objective before understanding the update | Keep the regression line, twelve examples, and MSE throughout the gradient-descent lesson |
| Parameters changed without an expanded gradient/update calculation | The signs, learning rate, and order of operations were unexplained | Calculate both gradients at the old parameters, substitute numbers into both updates, then recompute MSE |
| Sigmoid output appeared without exponentiation and division | The learner could not reproduce the probability | Show multiplication, addition, negation, exponentiation, denominator, reciprocal, and percentage |
| Rounded parameters did not exactly reproduce the displayed score | The displayed arithmetic appeared inconsistent | Distinguish rounded hand calculations from full-precision simulation values explicitly |

The central distinction is now stated directly: **linear regression is the model, MSE is the loss, and gradient descent is the procedure used to fit the parameters.** The first two lessons share the same numerical example. Classification then introduces its probability model and explains the later change to log-loss explicitly.

## 3. Day 1: linear regression

### What changed and why

The lesson now builds the calculation in learner order:

1. Compute one prediction with `prediction = w × x + b`.
2. Subtract the observed answer to obtain a signed error.
3. Square that error and repeat for all twelve examples.
4. Average the squared errors to obtain MSE.
5. Interpret the parameter map as a map of that same MSE.
6. Calculate the two loss gradients, make one update, and repeat.

The least-squares formula comes after the iterative explanation as a shortcut to the same answer. Its detailed derivation remains available in a collapsible section. Essential substituted arithmetic is visible in the main lesson and widget readouts.

### How it was implemented

- `LineFit/sim.ts` gained a pure `calculation()` helper exposing predictions, signed errors, squared errors, their sums, MSE gradients, and the next parameter pair.
- New `prediction`, `errors`, and `update` modes give the introductory calculations distinct figures. The errors view displays twelve squared-error bars and their MSE average, with an expandable table of all examples.
- The single-update view separates the destination calculation from the currently moving parameters. Current loss is recomputed from the displayed parameters.
- Readouts follow the same animation state as the drawing. The highlighted prediction remains visible at extreme slider settings, timeline labels were shortened for mobile, and Reset respects reduced motion.

### Worked values retained in the lesson

At `w = -1.4`, `b = -2.2`, example 8 has approximately:

```text
x = 0.931516, observed y = 1.720559
prediction = -3.504122
error = prediction - observed y = -5.224681
squared error = 27.297293

Sum of all twelve squared errors = 343.972793
MSE = 343.972793 / 12 = 28.664399
```

This makes the distinction between one example's squared error and the dataset's mean squared error visible.

## 4. Day 1: gradient descent

### What changed and why

The previous core explanation used `L = (theta_1² + 8 theta_2²) / 2`. Although useful as a generic optimization example, it broke continuity with the regression lesson. The revised lesson uses the actual regression MSE instead. Its map axes are `w` and `b`, and each map position represents a complete regression line.

The lesson explains what a partial derivative measures, derives the MSE gradient from squared errors, explains why subtracting a negative gradient increases a parameter, and recalculates the loss after the update. Repeated updates, learning-rate choices, and SGD all continue to use the same MSE objective.

### How it was implemented

- `GradientDescent2D/sim.ts` gained a `regression` surface that imports the existing `LineFit` data, loss, gradient, and starting parameters. The earlier generic surfaces remain available in the widget implementation; the revised lesson selects the regression surface.
- A `single` view holds the starting calculation, animates one update slowly, and holds the result. A `rates` view compares learning-rate behavior with a logarithmic loss chart.
- During movement, the current MSE is calculated at the current parameters rather than interpolated between endpoint losses. Update arithmetic is explicitly labeled as using the update's starting parameters.
- For regression, SGD now samples an actual training example and uses its squared-error gradient. The readout compares that sample gradient with the full-data gradient. Full-data MSE still measures the result of the update.
- A floating-point boundary fix prevents the counter from saying 23 completed updates at the exact 24-update position. Timeline labels describe phases (`early`, `middle`, `late`, `done`), and small-screen plot labels have room to remain readable.

### The numerical bridge between the two lessons

```text
sum(error × x) = -105.016349
sum(error)     = -33.549238

gradient_w = (2/12) × sum(error × x) = -17.502725
gradient_b = (2/12) × sum(error)     = -5.591540
learning rate = 0.08

new w = -1.4 - 0.08 × (-17.502725) ≈  0.000218
new b = -2.2 - 0.08 × (-5.591540)  ≈ -1.752677

MSE: 28.664399 before → 9.073422 after one update
MSE after ten updates: 0.430168
MSE after forty updates: approximately 0.200
```

Both gradients use the old parameters. Both parameters are then updated, and predictions and MSE are calculated again.

The learning-rate examples report MSE after 48 updates: approximately **1.296** at `0.01`, **0.200** at `0.08`, **0.202** at `0.26`, and **3855.114** at `0.28`. The stability boundary for this dataset is approximately `0.2724`; it is not presented as a universal rate.

The first sampled SGD update uses example 9. Its gradient is approximately `(-12.9624, -10.2874)`, producing full-data MSE **10.565** after the update, compared with **9.073** for the full-batch update.

## 5. Day 2: logistic regression and sigmoid

### What changed and why

The lesson now separates the linear score from its conversion to probability. It defines the sigmoid symbol, the constant `e`, exponentiation, the probability notation, and conversion from a fraction to a percentage. The worked example is expanded through every operation.

The original short parameter display concealed a real rounding distinction:

```text
Using rounded parameters:
1.26 × (-4) + 0.44 = -4.60

Using the fitted parameters at full precision:
w ≈ 1.262401319, b ≈ 0.440404123
z = w × (-4) + b ≈ -4.609201153
```

The revised explanation acknowledges this difference instead of suggesting the rounded inputs produce exactly `-4.61`.

### How the probability is now explained

```text
z                  ≈ -4.609201
-z                 ≈  4.609201
exp(-z)            ≈ 100.403910
1 + exp(-z)        ≈ 101.403910
p = 1 / denominator ≈ 0.009862
100 × p            ≈ 0.986% probability of class 1
```

The lesson also works through `x = 0` and `x = 4`, yielding probabilities approximately **0.608355** and **0.995889**. It explains why a negative score creates a large denominator and therefore a small probability.

### How it was implemented

- `LogisticFit/sim.ts` gained `probabilityCalculation()`, exposing the product, score, negated score, exponential, denominator, probability, and percentage.
- The `sweep` figure focuses on the probability curve and selected input. It removes the loss map and training-error stems from this step so the reader can follow one calculation.
- Live readouts expand the arithmetic, with an input slider and editable parameters. Tiny probabilities use scientific notation rather than rounding the probability itself to zero. Drawing and readouts use the same frame.
- The plotted domain leaves room for the probe at both endpoints. Timeline labels describe segments, and Reset respects reduced motion.
- The decision-boundary explanation now derives the `0.5` threshold and `x = -b/w`, including the effect of a negative weight.
- The log-loss section explains why classification introduces a different loss and substitutes labels into its two cases. The training section expands both parameter updates and states that the gradient-descent procedure is unchanged. It no longer claims forty updates reach the optimum exactly.

## 6. Implementation files and history

| Lesson | Lesson source | Component and simulation | Tests |
|---|---|---|---|
| Linear regression | [linear-regression.mdx](../src/content/lessons/foundations/linear-regression.mdx) | [LineFit.svelte](../src/components/widgets/LineFit/LineFit.svelte), [sim.ts](../src/components/widgets/LineFit/sim.ts) | [f1-day1.test.ts](../tests/unit/f1-day1.test.ts) |
| Gradient descent | [gradient-descent.mdx](../src/content/lessons/foundations/gradient-descent.mdx) | [GradientDescent2D.svelte](../src/components/widgets/GradientDescent2D/GradientDescent2D.svelte), [sim.ts](../src/components/widgets/GradientDescent2D/sim.ts) | [f1-day1.test.ts](../tests/unit/f1-day1.test.ts) |
| Logistic regression | [logistic-regression.mdx](../src/content/lessons/foundations/logistic-regression.mdx) | [LogisticFit.svelte](../src/components/widgets/LogisticFit/LogisticFit.svelte), [sim.ts](../src/components/widgets/LogisticFit/sim.ts) | [logistic-fit.test.ts](../tests/unit/logistic-fit.test.ts) |

Relevant commits, both dated October 8, 2026:

- `565afd0` — `fixes to day1 lessons`: includes the linear-regression revision. This commit also contains unrelated work; its entire diff is not the scope of this audit.
- `badffcd` — `fixes to day1/2`, with body `gradient descent and sigmoid function update`: records the subsequent two-lesson revision.

The implementation sessions did not commit or publish the changes themselves. The commits above were confirmed in repository history when preparing this audit. Production deployment status was not checked.

## 7. Verification evidence and its limits

**These are results recorded during implementation, not newly rerun checks on October 11.** Today's work checked source, history, saved report contents, and this document's links. It did not change lesson code or rerun tests, browser checks, or a build.

| Check | Recorded result |
|---|---|
| Initial linear-regression tests | 40 tests passed in `f1-day1.test.ts` |
| Combined follow-up tests | 54 tests passed across `f1-day1.test.ts` and `logistic-fit.test.ts`; this includes the earlier tests, so the counts must not be added |
| Astro/TypeScript check | 406 files; zero errors, warnings, or hints |
| Production build | 104 pages built successfully; existing `use astro:head-inject` bundler warnings remained |
| Repeated-figure check | `no repeated figures` |
| Svelte autofixer | No issues reported; advisory suggestions about the existing effect/control architecture remained |
| Widget dependency closures, gzip | LineFit: 29.51 KB; GradientDescent2D: 30.94 KB; LogisticFit: 29.12 KB; all below 40 KB |
| Browser layout/performance | Desktop/mobile, light/dark passed; no console errors or horizontal overflow in final checks; measured frame rate about 75.5 fps, with final Day 1/2 page-check stalls at most 67 ms |

### Saved evidence

The following are local artifacts under `.playwright-mcp/`, which is Git-ignored. They were present when this audit was written but will not accompany a fresh clone unless copied separately. The summary above is retained in this tracked audit so it does not depend solely on those artifacts.

- Linear regression: [108 frame checks](../.playwright-mcp/check/linear-regression-review/results.json), [validated control checks](../.playwright-mcp/check/linear-regression-review/final-controls-validated.json), and [worked-update screenshot](../.playwright-mcp/check/linear-regression-review/390-light-figure4-t6.png).
- Gradient descent and sigmoid: [198 frame checks](../.playwright-mcp/check/day1-day2-math/frames.json), [12 grouped control checks](../.playwright-mcp/check/day1-day2-math/final-controls.json), [30 final label/layout frames](../.playwright-mcp/check/day1-day2-math/final-labels.json), and [build log](../.playwright-mcp/check/day1-day2-math/build.log).
- Review screenshots: [single MSE update](../.playwright-mcp/check/day1-day2-math/gradient-descent-reviewed.png) and [expanded sigmoid arithmetic](../.playwright-mcp/check/day1-day2-math/sigmoid-reviewed.png).

The 198-frame pass preceded the final counter/label/layout refinements; the control and label reports cover the subsequent checks. They are successive verification passes, not a claim that every earlier screenshot shows the final presentation.

The earlier linear-regression `final-controls.json` contains false mismatches caused by Unicode-minus handling in the verification script. Use `final-controls-validated.json`, whose errors list is empty, rather than treating the earlier file as a successful report. The other linked JSON reports also have empty errors lists.

The tests cover hand-checkable arithmetic, gradient correctness, continuity with the preceding regression example, actual sampled-example gradients, learning-rate behavior, update boundaries, and sigmoid precision. The browser checks cover fixed frames, control behavior, reduced-motion reset, extreme inputs, and small-screen layouts. They do not constitute a fresh full-suite run, a complete accessibility audit, or owner approval of teaching clarity.

## 8. Review and future authoring checks

The requested implementation work is complete. For clarity review, a reader should now be able to:

1. Reproduce one prediction and explain how twelve squared errors become MSE.
2. Distinguish model, loss, and optimizer without changing examples.
3. Calculate both parameter updates and explain why the MSE is measured again afterward.
4. Reproduce the sigmoid probability with a calculator and account for rounding differences.
5. Explain what changes when moving to classification: the prediction and loss change, while gradient descent remains the fitting procedure.

Future revisions should preserve this continuity. Introduce each symbol before using it, show the intermediate operation that produces a quoted number, keep essential arithmetic visible, and label a future update's destination separately from the state currently being drawn. These requirements address the owner's original confusion; automated checks support that review but cannot replace it.
