# Stage 4S: relevance of uniform source-block exchange

**Relevant-touch survival collapses:** uniform swaps touching at least one pre-swap dominant source survive jointly in **6/76 = 7.89%**, versus **325/372 = 87.37%** when neither dominant source is touched. Exactly-one-touch swaps survive **0/68**; both-touch swaps survive **6/8 = 75%**. Thus **325/331 = 98.19%** of all surviving uniform swaps leave both pre-swap dominant blocks unchanged. The 73.88% global survival is mostly exchange irrelevant to the specified critical-component bottleneck, not evidence of better critical-source mixing.

This is read-only analysis of the completed Stage-4R arrays. The frozen contour is `ell_9 = -110252.99476697217`. No new likelihood, prior, or source_gain evaluation occurred; no kernel or chain was built.

## Saved inputs, identity and integrity

Verified all **23** Stage-4R recorded provenance hashes before analysis. Saved states agree byte-for-byte with Stage 4Q; all 128 saved priors/likelihoods are finite and the likelihoods strictly exceed ell9. Physical arrays agree exactly with the deterministic box transform of saved coordinates using the already audited bounds. Saved proposal arrays agree with both the raw evaluation NPZ files and report JSON. All **2272** proposed catalogues across uniform, dominant and oracle controls were reconstructed exactly from their original bases: donor copies and untouched coordinates agree byte-for-byte, with no sequential updates. All saved proposal priors and likelihoods are finite. Structural failures: **0**.

The Stage-4R output NPZ files did not have a previously recorded output-hash manifest. This analysis therefore distinguishes verification of the existing provenance hashes from new SHA-256 hashes of the consumed outputs. The latter were recorded before analysis and checked unchanged afterward; redundant saved-array/report comparisons provide the output-integrity check. Stage-4R files were not modified. No fresh model evaluation was used to revalidate cached likelihoods.

All 448 uniform cases retain the original `i < j` walker ordering (A = i, B = j), 16 prescribed trace indices per pair, and the PCG64 seed `20260923`. Labels reproduce the frozen draw sequence and saved design exactly. Ranks are reconstructed from saved source_gain using descending stable sorting, with lowest label breaking ties, and agree exactly with saved Stage-4R ranks/dominant labels. Top-2 and top-3 lists are saved for every base state; they are sensitivity descriptions only. No importance definition was chosen using survival. Source_gain remains a nonadditive conditional/profiled diagnostic, not an integrated-likelihood decomposition.

## Primary stratification

N touches neither pre-swap dominant block; A touches only A’s dominant block; B touches only B’s; AB touches both. These categories are mutually exclusive and exhaustive.

| category | proposals | joint survival | A-side survival | B-side survival |
| --- | --- | --- | --- | --- |
| N | 372 | 325/372 (87.37%) | 363/372 (97.58%) | 333/372 (89.52%) |
| A | 41 | 0/41 (0.00%) | 0/41 (0.00%) | 41/41 (100.00%) |
| B | 27 | 0/27 (0.00%) | 27/27 (100.00%) | 0/27 (0.00%) |
| AB | 8 | 6/8 (75.00%) | 6/8 (75.00%) | 8/8 (100.00%) |

| dominant sources touched | proposals | joint survival |
| --- | --- | --- |
| 0 | 372 | 325/372 (87.37%) |
| 1 | 68 | 0/68 (0.00%) |
| 2 | 8 | 6/8 (75.00%) |
| >=1 | 76 | 6/76 (7.89%) |

Delta logL is relative to the original catalogue. Values below are median [25th percentile, 75th percentile]. The per-proposal minimum-side margin is `min(logL_A_new, logL_B_new) - ell9`; its minimum is also reported explicitly. Summaries include failures.

| category | A delta logL | B delta logL | minimum-side margin | worst margin |
| --- | --- | --- | --- | --- |
| N | 11.16 [-54.70, 88.19] | -9.08 [-85.97, 53.83] | 265.05 [85.84, 553.86] | -461.87 |
| A | -4257.67 [-4635.80, -3874.13] | 1508.69 [1000.32, 2081.89] | -3630.78 [-3743.04, -3339.10] | -4019.89 |
| B | 1762.14 [1012.27, 2200.94] | -3916.33 [-4511.74, -3508.89] | -3443.18 [-3534.79, -3335.73] | -3765.74 |
| AB | -556.66 [-1281.01, 85.26] | 562.53 [-158.88, 1368.41] | 540.07 [80.03, 647.58] | -504.09 |
| 1 | -3723.39 [-4310.10, 1237.89] | 972.02 [-3635.02, 1597.11] | -3503.70 [-3695.72, -3329.40] | -4019.89 |
| >=1 | -3351.50 [-4292.28, 1049.60] | 870.81 [-3537.14, 1597.11] | -3491.63 [-3690.17, -3173.45] | -4019.89 |

The asymmetry is mechanistic in these saved cases: whenever exactly one dominant block is selected, the walker losing that block fails (68/68), while the walker receiving it survives (68/68). The losing side is at least 2752.54 logL units below ell9. These are not near-contour rounding failures. A/B differences also reflect fixed walker ordering, not randomized side identities.

## Physical movement, separate from relevance and survival

Every proposal has side-specific geometry saved in `analysis.json`. Labelled block movement is recorded both as latent Euclidean distance and physical six-coordinate RMS distance normalized by the existing prior widths (psi period pi and lam period 2pi). Labelled full-catalogue RMS is also saved; for one changed block it is block RMS/3. Permutation-invariant distances use the existing Hungarian assignment: frequency-set RMS in Fourier bins (`1 bin = 1/31536000 Hz`) and normalized six-coordinate catalogue RMS. Matching is post-hoc and changes no proposal. All-proposal and joint-survivor distributions are retained separately; no large-displacement cutoff or scalar efficiency score is introduced.

| strategy | proposals | joint survival | survivor block RMS | survivor |f0 shift| (bins) |
| --- | --- | --- | --- | --- |
| Targeted dominant | 448 | 265/448 (59.15%) | 0.20825 [0.16791, 0.24227] | 6.427 [1.556, 9.450] |
| Historical ORACLE DIAGNOSTIC | 240 | 184/240 (76.67%) | 0.20821 [0.17365, 0.24176] | 6.421 [1.569, 9.340] |
| Uniform: 0 dominant | 372 | 325/372 (87.37%) | 0.35539 [0.30041, 0.42280] | 184.033 [90.151, 314.747] |
| Uniform: exactly 1 | 68 | 0/68 (0.00%) | — (no survivors) | — (no survivors) |
| Uniform: 2 dominant | 8 | 6/8 (75.00%) | 0.17282 [0.16592, 0.25596] | 4.098 [1.362, 7.324] |
| Uniform: >=1 dominant | 76 | 6/76 (7.89%) | 0.17282 [0.16592, 0.25596] | 4.098 [1.362, 7.324] |

The block RMS and absolute f0 shift are equal for the two exchange directions. Their signed f0 shifts have opposite signs. Side-specific surviving catalogue distances and physical f0 shifts are:

| strategy / side | frequency-set RMS (bins) | six-coordinate set RMS | signed f0 shift (Hz) |
| --- | --- | --- | --- |
| Uniform: 0 dominant / A | 38.773 [24.775, 58.929] | 0.11611 [0.09789, 0.13538] | 1.61140e-07 [-5.77389e-06, 5.90478e-06] |
| Uniform: 0 dominant / B | 40.420 [24.352, 58.943] | 0.11596 [0.09740, 0.13460] | -1.61140e-07 [-5.90478e-06, 5.77389e-06] |
| Uniform: >=1 dominant / A | 1.366 [0.454, 2.441] | 0.05761 [0.05531, 0.08532] | 4.67005e-11 [-9.48130e-08, 1.08748e-07] |
| Uniform: >=1 dominant / B | 1.366 [0.454, 2.441] | 0.05761 [0.05531, 0.08532] | -4.67005e-11 [-1.08748e-07, 9.48130e-08] |
| Targeted dominant / A | 2.067 [0.519, 3.071] | 0.06939 [0.05597, 0.08076] | -3.05257e-08 [-2.50955e-07, 9.18015e-08] |
| Targeted dominant / B | 2.032 [0.519, 3.066] | 0.06939 [0.05597, 0.08076] | 3.05257e-08 [-9.18015e-08, 2.50955e-07] |
| Historical ORACLE DIAGNOSTIC / A | 2.090 [0.523, 3.038] | 0.06938 [0.05788, 0.08059] | -8.44473e-09 [-2.12712e-07, 1.92742e-07] |
| Historical ORACLE DIAGNOSTIC / B | 2.048 [0.523, 3.007] | 0.06938 [0.05788, 0.08059] | 8.44473e-09 [-1.92742e-07, 2.12712e-07] |

Non-dominant surviving swaps actually have **larger geometric displacement** by these measures (median swapped-source f0 shift 184.03 bins) than relevant surviving swaps (4.10 bins). Calling the former low-impact refers to the dominant-source bottleneck and their comparatively modest likelihood change, not to small coordinate movement. Wide movement of low-gain sources is not evidence of critical-component communication. The six relevant survivors have movement comparable in scale to the targeted survivors, but are too few to establish a general mixing benefit.

## Physical replacement in relevant surviving swaps

All **6/6 (100%)** relevant surviving uniform proposals are AB cases. In each of their **12/12** catalogue outcomes, the original dominant six-coordinate block is absent, the new block is an exact copy of the donor’s pre-swap dominant block, and the realization differs from the original. Frequency-only and six-coordinate Hungarian matching both associate the original dominant block with the incoming replacement in **12/12** cases. These are actual replacements rather than relabellings of an unchanged source set. The six-coordinate permutation-invariant distance is nonzero in all twelve outcomes.

No post-swap source_gain was saved in Stage 4R, and none was recomputed. “Replacement” here is the geometric replacement of the pre-swap critical-frequency realization; it does not assert post-swap gain dominance or unique astrophysical identity. The exact donor copies and post-hoc geometry support that narrower conclusion without a tuned matching-distance threshold.

| walkers A,B | retained index | labels A,B | A f0 before (Hz) | A f0 after (Hz) | A shift (bins) | B shift (bins) | six-coordinate set RMS A/B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 153 | 7,6 | 0.001840999485 | 0.001841138700 | +4.3903 | -4.3903 | 0.05528 / 0.05528 |
| 0,1 | 170 | 7,6 | 0.001840889975 | 0.001841153209 | +8.3013 | -8.3013 | 0.05984 / 0.05984 |
| 1,2 | 34 | 6,3 | 0.001841323554 | 0.001841202888 | -3.8053 | +3.8053 | 0.02194 / 0.02194 |
| 1,5 | 119 | 6,2 | 0.001841195121 | 0.001840927429 | -8.4419 | +8.4419 | 0.05538 / 0.05538 |
| 1,7 | 119 | 6,4 | 0.001841195121 | 0.001841177867 | -0.5441 | +0.5441 | 0.09604 / 0.09604 |
| 2,3 | 221 | 3,0 | 0.001841211387 | 0.001841228735 | +0.5471 | -0.5471 | 0.09381 / 0.09381 |

B’s before/after frequencies are A’s after/before frequencies in these AB swaps. Absolute frequency changes span 0.544–8.442 bins; block RMS spans 0.06583–0.28812. These raw scales are reported without categorizing a tuned “large” threshold. Most survivors involve walker 1, further limiting generalization from this small subset.

## Top-2 / top-3 sensitivity

Only category counts and survival are repeated here. A/B side-survival counts are included for completeness; the primary definition remains top 1.

| rank set | category | proposals | joint survival | A-side survival | B-side survival |
| --- | --- | --- | --- | --- | --- |
| 2 | N | 296 | 271/296 (91.55%) | 292/296 (98.65%) | 275/296 (92.91%) |
| 2 | A | 68 | 29/68 (42.65%) | 29/68 (42.65%) | 68/68 (100.00%) |
| 2 | B | 62 | 18/62 (29.03%) | 61/62 (98.39%) | 18/62 (29.03%) |
| 2 | AB | 22 | 13/22 (59.09%) | 14/22 (63.64%) | 21/22 (95.45%) |
| 2 | 0 | 296 | 271/296 (91.55%) | 292/296 (98.65%) | 275/296 (92.91%) |
| 2 | 1 | 130 | 47/130 (36.15%) | 90/130 (69.23%) | 86/130 (66.15%) |
| 2 | 2 | 22 | 13/22 (59.09%) | 14/22 (63.64%) | 21/22 (95.45%) |
| 2 | >=1 | 152 | 60/152 (39.47%) | 104/152 (68.42%) | 107/152 (70.39%) |
| 3 | N | 214 | 199/214 (92.99%) | 212/214 (99.07%) | 201/214 (93.93%) |
| 3 | A | 98 | 60/98 (61.22%) | 60/98 (61.22%) | 98/98 (100.00%) |
| 3 | B | 92 | 49/92 (53.26%) | 91/92 (98.91%) | 49/92 (53.26%) |
| 3 | AB | 44 | 23/44 (52.27%) | 33/44 (75.00%) | 34/44 (77.27%) |
| 3 | 0 | 214 | 199/214 (92.99%) | 212/214 (99.07%) | 201/214 (93.93%) |
| 3 | 1 | 190 | 109/190 (57.37%) | 151/190 (79.47%) | 147/190 (77.37%) |
| 3 | 2 | 44 | 23/44 (52.27%) | 33/44 (75.00%) | 34/44 (77.27%) |
| 3 | >=1 | 234 | 132/234 (56.41%) | 184/234 (78.63%) | 181/234 (77.35%) |

Broader sets include lower-gain blocks: >=1 top-2 survival is 60/152 = 39.47%; >=1 top-3 survival is 132/234 = 56.41%. These do not rescue the primary dominant-touch result and do not justify selecting a rank cutoff.

## Fixed selection mechanism and exactness

For one pre-swap dominant label out of nine in each walker, the predeclared probabilities are `(64/81, 16/81, 1/81)` for zero, one and two dominant labels touched. Expected counts in 448 cases are `(353.9753, 88.4938, 5.5309)`; observed counts are `(372, 68, 8)`. Expected A-only and B-only counts are each 44.2469, versus observed 41 and 27. The probability of any dominant touch is **17/81 = 20.99%**, but **16/17** of those relevant selections touch only one dominant source. Both-dominant selection occurs with probability only **1/81 = 1.23%**.

An exact multinomial goodness-of-fit calculation enumerating all three-category count vectors and ordering them by Pearson discrepancy gives statistic **6.76618**, tail probability **0.03358**. Thus this is a mildly unusual draw, and it would not pass a conventional 5% goodness-of-fit check; it should not be described as unremarkably consistent with expected counts. Nevertheless, exact replay of the frozen independent PCG64 labels verifies the actual selection construction. An unusual finite realization does not demonstrate a different generator, and no draws were changed or resampled. This selection-only calculation treats pre-swap dominant labels as fixed; it does not attach independent-binomial uncertainty claims to survival on correlated trace states.

Uniform state-independent labels give a symmetric fixed-block involution. With the constrained joint target and implemented-prior MH correction, this is an exact proposal construction; Stage 4R measured at most 1.14e-13 joint log-prior cancellation discrepancy. Here we only reuse contour-survival outcomes—no MH draws or chain were executed. Dynamic dominant selection and the historical oracle remain diagnostic controls, not exact population kernels or available selection rules.

The fixed mechanism makes relevant communication possible but sparse: nearly all relevant selections under the mechanism touch just one dominant source, and those all fail in the saved experiment. The rare both-dominant cases can replace viable realizations, but eight selected cases and six survivors do not establish adequate sequential communication. Global survival alone provides no justification for integration. Acceptance, relevance and displacement are reported separately, without combining them into an efficiency score.

## Decision and scope

**Generic uniform population block exchange should also be deprioritized for this critical-source bottleneck.** Relevant-touch survival is only 7.89%, and the apparently high global survival is overwhelmingly from proposals leaving both dominant blocks untouched. Rare both-dominant exchanges are physically viable and do move the critical-frequency realization, so the result is not a claim that such exchange is impossible. It does not meet the requested condition of substantial survival across relevant-touch proposals for recommending a sequential uniform-swap benchmark next.

No kernel or sequential population chain, new likelihood evaluations, source-selection weight tuning, source_gain/prior modifications, level-10 retry, level 11, production or evidence implementation occurred. Level 9 remains frozen and Stage-4P level 10 rejected. This analysis stops here.

## Tests, code and artifacts

Five new cheap tests cover category assignment, exhaustive/disjoint partitions, top-k membership and ties, combinatorial probabilities, and a synthetic NPZ-only analysis path with model/kernel imports forbidden. All existing tests are preserved. The complete DNS/GB suite passed **184 tests**; no LISA likelihood is called in pytest. Test log: `/tmp/lisa_dns_stage4s_tests.log`.

Code: [saved-array analyzer](../../examples/lisa_dns_stage4/analyze_uniform_source_relevance.py), [tests](../../tests/experimental/test_dns_gb_uniform_source_relevance.py). Output: `/tmp/lisa_dns_stage4s_uniform_relevance/analysis.json`, containing verified input hashes, all source ranks/top-k identities, every proposal’s categories/outcomes/geometry, all side-specific distributions, replacement details and selection-probability audit. These local `/tmp` artifacts are not versioned. The analyzer refuses to overwrite an existing output directory.

Reproduce with `PYTHONPATH=. python -m examples.lisa_dns_stage4.analyze_uniform_source_relevance --output /tmp/lisa_dns_stage4s_uniform_relevance_recheck` in the existing scientific Python environment. This command loads saved arrays and hashes source files; it never imports or invokes the LISA model.
