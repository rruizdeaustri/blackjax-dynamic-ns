# Stage 4T: fixed-frequency reciprocal-exchange design audit

## Frozen design, recorded before selector statistics

The documented interval is fixed at `[0.0018407250, 0.0018412366] Hz`. Its midpoint is **f_star = 0.0018409808 Hz** and half-width is **Delta_f = 2.558e-7 Hz**. The primary selector uses **sigma_f = Delta_f**, without a width scan or optional sensitivity run. No interval refitting is allowed. Frozen contour: `ell_9 = -110252.99476697217`.

Design artifact: `/tmp/lisa_dns_stage4t_frequency_design/design.json`. This section was written before evaluating selection-success statistics.

## Results

**The fixed soft-frequency selector passes this design audit and deserves ONE sequential benchmark next.** It substantially increases relevant-label selection probability, and the exact selection correction preserves most of the saved dominant-swap feasibility. This is a proposal-design conclusion only: no kernel, chain, new model evaluation or overall-acceptance estimate is produced.

## Inputs and mathematical selector

Only saved Stage-4R base states, dominant-swap positions and prior/likelihood outcomes are used for the replay; Stage-4S hashes verify their retained contents. Every one of the 128 prescribed bases and 448 unordered walker-pair/index cases is retained. No likelihood- or source_gain-based resampling occurs.

For each labelled source, `log w_k = -0.5 * ((f0_k - f_star)/Delta_f)^2`, `log Z = logsumexp(log w)`, and `log p_k = log w_k - log Z`. Thus `p_k = w_k/Z` with the single frozen width. No clipping, probability floor, uniform mixture or width tuning is introduced. The source_gain label is used only to assess the physical selector and identify the already saved dominant proposals.

The Gaussian has strictly positive mathematical support, but some weights in these catalogues are below float64’s smallest subnormal. MH algebra therefore uses finite log probabilities. Explicit positivity checks use NumPy extended precision; the JSON stores full probability vectors as decimal strings, alongside ordinary finite log probabilities, to prevent serialization from silently turning tiny probabilities into zero. This is a numerical representation of the specified Gaussian, not a modified selector. A future implementation must preserve log-space selection/support; naively exponentiating all weights in float64 would violate the requested nonzero support.

## Agreement with diagnostic source_gain

| state group | states | maximum-p = dominant | top-2 contains dominant | top-3 contains dominant | mean p(dominant) |
| --- | --- | --- | --- | --- | --- |
| persistent_six | 96 | 80/96 (83.33%) | 95/96 (98.96%) | 96/96 (100.00%) | 0.813646 |
| replaced_1_6 | 32 | 23/32 (71.88%) | 32/32 (100.00%) | 32/32 (100.00%) | 0.769198 |
| walker_1 | 16 | 10/16 (62.50%) | 16/16 (100.00%) | 16/16 (100.00%) | 0.781575 |
| walker_6 | 16 | 13/16 (81.25%) | 16/16 (100.00%) | 16/16 (100.00%) | 0.756821 |
| all | 128 | 103/128 (80.47%) | 127/128 (99.22%) | 128/128 (100.00%) | 0.802534 |

Probability mass assigned to the source_gain-dominant label, reported as min / q25 / median / q75 / max:

| state group | p(dominant) distribution |
| --- | --- |
| persistent_six | 0.325663 / 0.601116 / 0.956172 / 1.000000 / 1.000000 |
| replaced_1_6 | 0.399433 / 0.479839 / 0.942813 / 1.000000 / 1.000000 |
| walker_1 | 0.405453 / 0.443263 / 0.971664 / 1.000000 / 1.000000 |
| walker_6 | 0.399433 / 0.495098 / 0.805541 / 0.988506 / 1.000000 |
| all | 0.325663 / 0.538597 / 0.953133 / 1.000000 / 1.000000 |

The primary maximum-p selector and the diagnostic nearest-frequency label are identical in all 128 states, with the lowest label breaking ties. Consequently hard-nearest agreement is also **103/128 = 80.47%** overall, **80/96 = 83.33%** for persistent walkers, and **23/32 = 71.88%** for walkers 1 and 6 (10/16 and 13/16 respectively). Frequency alone is informative but is not a perfect identifier; another source closer to the fixed center sometimes receives the most weight. Soft selection still gives the source_gain-dominant component at least 0.325663 probability in every saved state. This does not resolve unique astrophysical-source identity.

The hard nearest-frequency rule is a diagnostic only. After exchange it can select different labels and give the required reverse label pair zero probability. It is not proposed as the sampler rule. All 128 individual maximum-p labels, source_gain-dominant labels, top-2/top-3 memberships, dominant masses and full probability/log-probability vectors are retained in `analysis.json`; compact identities are also in `state_selection.csv`.

## Probability of attempting a relevant reciprocal move

For each original pair/index case, `P_both = p_A(k_dom_A | x) * p_B(k_dom_B | y)`. This is a label-selection probability, separate from contour survival, MH acceptance and displacement.

| selection | min | q25 | median | q75 | max |
| --- | --- | --- | --- | --- | --- |
| uniform | 0.012346 | 0.012346 | 0.012346 | 0.012346 | 0.012346 |
| fixed soft frequency | 0.132041 | 0.426021 | 0.583160 | 0.943267 | 1.000000 |

The soft selector’s median is **58.32%**, compared with **1/81 = 1.23457%** for uniform labels. Even its minimum, **13.20%**, exceeds the uniform rate; the median and minimum increases are 47.24× and 10.70× respectively. Each pair’s two marginal probabilities and their product are saved. These ratios compare selection probabilities only; no efficiency score or overall swap-acceptance comparison is made.

## Exact state-dependent proposal algebra

Condition on a state-independent, symmetric choice of unordered walker pair. Choose source label `i ~ p(x)` and `j ~ p(y)`, independently, then exchange the complete six-coordinate blocks. There is no within-catalogue relabelling or sorting: the incoming block in x remains in slot **i**, and the incoming block in y remains in slot **j**. Therefore **i′ = i and j′ = j**, and applying the same labelled exchange again exactly restores both original states. Its absolute Jacobian is one.

The forward and reverse label probabilities are

```text
q_forward = p_i(x) p_j(y)
q_reverse = p_i(x') p_j(y')
log r_selection = log p_i(x') + log p_j(y')
                - log p_i(x)  - log p_j(y).
```

Reverse probabilities must be evaluated at the same slot labels on the proposed catalogues, not at newly computed dominant or nearest labels.

There is a useful independent algebraic check. Write `a = w_i(x)`, `b = w_j(y)`, and let `Z_x`, `Z_y` be the two normalizers. The same fixed physical weight function applies in both walkers, so swapping the blocks exchanges their unnormalized weights:

```text
Z_x' = Z_x - a + b
Z_y' = Z_y - b + a
w_i(x') = b,  w_j(y') = a
q_reverse/q_forward = (Z_x Z_y)/(Z_x' Z_y').
```

Thus `log r_selection = log Z_x + log Z_y - log Z_x′ - log Z_y′`. This cancellation concerns only selected unnormalized weights; the normalizers must be retained. The replay computes both forms and checks agreement, and verifies that reversing the exchange negates the log ratio.

For the joint level-9 target proportional to `pi_impl(x) pi_impl(y) 1[logL(x)>ell9] 1[logL(y)>ell9]`, the general MH probability is

```text
alpha = 0                                      if either proposed logL <= ell9
alpha = exp(min(0, log r_selection + Delta_pi)) otherwise
Delta_pi = log pi_impl(x') + log pi_impl(y')
         - log pi_impl(x)  - log pi_impl(y).
```

The shared labelled-block prior gives algebraic `Delta_pi = 0`, as numerically audited in Stage 4R. This yields the requested `min(1, exp(log r_selection))` inside both contours. For fidelity to finite-precision pi_impl, the analysis also retains the already saved joint-prior residual and reports its negligible effect. No prior evaluation or modification is needed.

## Saved dominant-to-dominant replay

The following is an **unweighted replay of the same 448 deliberately chosen dominant-label proposals**. Forward probabilities come from the original states; reverse probabilities are computed from the exact saved post-swap positions using only the saved physical bounds. Saved likelihoods supply contour membership. No labels are newly sampled, no acceptance uniforms are drawn and no state is updated.

| quantity | value |
| --- | --- |
| raw joint survival | 265/448 = 59.15178571% |
| mean alpha with selection correction, all 448 | 0.576890916847 (57.6891%) |
| mean alpha also retaining saved prior residual | 0.576890916847 |
| mean alpha among 265 joint survivors | 0.975272191500 |
| survivor alpha: min / q25 / median / q75 / max | 0.617910 / 0.983445 / 1.000000 / 1.000000 / 1.000000 |
| survivor log selection ratio: min / q25 / median / q75 / max | -0.481412 / -0.016694 / 0.000000 / 0.039010 / 0.285189 |
| maximum per-case alpha effect of saved prior residual | 1.13576e-13 |
| maximum error against normalizer-ratio identity | 2.22045e-16 |

Selection correction reduces the replay mean from 59.15% to **57.69%**, a 1.46-percentage-point change. Among joint survivors the mean correction is **97.53%**, with minimum **61.79%** and median **100%**. The correction therefore does not destroy the saved feasibility.

This is **not** overall expected acceptance of a frequency-guided sampler: the 448 dominant moves were prescribed diagnostically, not drawn according to the soft probabilities, and outcomes for all other soft-selected label combinations are not available. No weighted product of selection probability, acceptance and displacement is presented. Sequential behavior, evolving population states and actual mixing remain untested.

## Integrity and tests

All base and proposed selection vectors are finite, strictly positive in the checked extended representation, and normalized to maximum absolute sum error **1.65883e-16**. The minimum explicit probability is **5.759075753258299696e-400**, with minimum log probability **-919.283260**. There are **492** base/post probability entries below the smallest positive float64 value; decimal-string probabilities and log-space arithmetic preserve these instead of flooring or deleting them. Values printed as 1 in summary tables are rounded, not a hard-selector rule.

All **448** forward/reverse labelled exchanges were checked exactly against saved positions, including both donor copies, unchanged coordinates and involution. Reverse labels remain i,j in every case. Recorded Stage-4R provenance hashes (23) and Stage-4S retained-input hashes (10) were verified; hashes of all consumed inputs, the frozen Stage-4T design and analyzer remained unchanged. Structural failures: **0**. New likelihood/prior/source_gain calls: **0/0/0**. MCMC steps: **0**.

Seven new cheap tests cover the exact interval construction, Gaussian normalization, strictly positive extreme weights, forward/reverse and normalizer-ratio algebra, unchanged label slots/involution, contour-gated MH alpha with prior correction, and a synthetic saved-NPZ execution path with model/kernel imports forbidden. Existing tests are preserved. The full DNS/GB suite passed **191 tests**. No real LISA likelihood is invoked in pytest. Log: `/tmp/lisa_dns_stage4t_tests.log`.

## Decision and stopping point

1. **Physical identification:** the fixed frequency selector usually selects the same component as source_gain (80.47% maximum-label agreement), and almost always includes it among its two highest-probability labels (99.22%). It gives that component substantial soft mass even when the nearest label differs, including walkers 1 and 6. It is not a perfect or unique physical-identity classifier.
2. **Relevant selection:** the median probability of attempting both dominant labels rises from uniform’s 1.23% to 58.32%, with no fitted interval or width.
3. **Exact correction:** the saved dominant replay retains mean alpha 57.69%, versus 59.15% raw contour survival. The reverse-selection correction is modest on this fixed diagnostic set.

These observations support only the next-step conclusion: **an exact frequency-guided reciprocal population swap deserves ONE sequential benchmark next.** This stage does not build or run that benchmark, claim validated population mixing, or recommend production integration. The candidate remains the predeclared Gaussian with sigma_f = Delta_f; no sensitivity result was used to select a different rule.

No new likelihood evaluations, population sampler, sequential chain, level-10 retry, level 11, sigma scan, source_gain/prior/slice change, production run or evidence implementation occurred. Level 9 remains frozen, Stage-4P level 10 remains rejected, and this design audit stops here.

## Code and artifacts

Code: [saved-array design analyzer](../../examples/lisa_dns_stage4/analyze_frequency_exchange_design.py), [cheap tests](../../tests/experimental/test_dns_gb_frequency_exchange_design.py). Artifacts in `/tmp/lisa_dns_stage4t_frequency_design/`: `design.json` (written before statistics), `analysis.json` (input hashes, 128 state identities/full probabilities, 448 forward/reverse ratios and alphas), `state_selection.csv`, and `dominant_replay.csv`. Logs: `/tmp/lisa_dns_stage4t_analysis.log` and `/tmp/lisa_dns_stage4t_tests.log`. The `/tmp` artifacts are local, not versioned.

The analyzer requires a frozen `design.json` in its output directory and refuses to overwrite `analysis.json`. To reproduce, copy the existing frozen design into a new output directory, then run `PYTHONPATH=. python -m examples.lisa_dns_stage4.analyze_frequency_exchange_design --output <new-directory>` using the existing scientific Python environment. No model import is involved.
