# Stage 4R: dominant-source exchange diagnostic

**Mixed result:** source_gain finds the dominant source near the critical frequency under new labels, but all-walker dominant joint survival is **59.15%**, below the uniform-label control’s **73.88%**. The evidence does not satisfy the requested favorable-outcome condition for recommending an exact population exchange kernel next.

Resumed the existing partial runner and tests without recreating Stage 4R. This is a fixed-base physical-feasibility diagnostic, with no MH step, chain update, level construction, or production run.

Frozen target: `ell_9 = -110252.99476697217`, `log X_9 = -7.67544001580533`. Stage-4P level 10 remains rejected. The old isotropic threshold `-109401.11425363769` and Stage-4P threshold `-109301.45870884096` are diagnostic comparisons only.

## Design and source identity

The eight Stage-4P calibration walkers contribute retained indices `[0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]`: 16 states each, 128 total, identical byte-for-byte to Stage-4Q `bases.npz`. No state was selected using likelihood or source_gain. The original arrays are read-only. Historical labels are `[7, 3, 3, 0, 8, 2, 7, 4]`; persistent walkers are 0, 2, 3, 4, 5, 7.

All nine gains and ranks, dominant labels and physical f0, historical labels and their current rank/f0 are in `report.json` and `bases.npz`. The unchanged BayesLISAx `_marg_recon(...)[6]` diagnostic agrees with Stage 4Q. Source_gain is a conditional regularized profiled quadratic gain with other amplitudes reprofiled; it excludes the integrated log-determinant and is **not an additive decomposition of integrated likelihood**. Ranking is descending gain with the lowest label breaking ties. Dynamic selection is diagnostic only.

| walker | dominant label counts | historical rank 1 | historical top 2 | historical top 3 | dominant f0 range (Hz) |
| --- | --- | --- | --- | --- | --- |
| 0 | 7: 16/16 | 100.00% | 100.00% | 100.00% | [0.001840875085, 0.001841177126] |
| 1 | 6: 16/16 | 0.00% | 6.25% | 18.75% | [0.001841120925, 0.001841337770] |
| 2 | 3: 16/16 | 100.00% | 100.00% | 100.00% | [0.001841149515, 0.001841234976] |
| 3 | 0: 16/16 | 100.00% | 100.00% | 100.00% | [0.001841179744, 0.001841236632] |
| 4 | 8: 16/16 | 100.00% | 100.00% | 100.00% | [0.001840725011, 0.001840957664] |
| 5 | 2: 16/16 | 100.00% | 100.00% | 100.00% | [0.001840835422, 0.001841163334] |
| 6 | 2: 15/16, 8: 1/16 | 0.00% | 18.75% | 37.50% | [0.001840829944, 0.001840905037] |
| 7 | 4: 16/16 | 100.00% | 100.00% | 100.00% | [0.001841084088, 0.001841203079] |

Hungarian matching below is post-hoc only and never changes labels, assignments, or selection. Frequency differences are in Fourier bins, `DF = 1/31536000 Hz`. Full matching uses the existing prior-width normalized six-coordinate metric and periodic-coordinate handling.

Walker 1: across 96 comparisons with persistent walkers, frequency-only Hungarian matching pairs the two dominant sources in 13/96 cases; full six-coordinate matching does so in 35/96. Absolute dominant f0 separation is median 4.391 bins, range [0.018, 15.636].

Walker 6: across 96 comparisons with persistent walkers, frequency-only Hungarian matching pairs the two dominant sources in 16/96 cases; full six-coordinate matching does so in 31/96. Absolute dominant f0 separation is median 8.051 bins, range [0.033, 11.350].

The persistent critical-frequency region spans **0.001840725011–0.001841236632 Hz**. Walker 1 moved from historical label 3 to dominant label 6 in all 16 states: 9 frequencies are inside that empirical span and the other 7 extend at most 3.19 Fourier bins above it. Walker 6 moved from historical label 7 to dominant label 2 (15 states) or 8 (one state), with all 16 frequencies inside the region. The frequency-localized high-gain component has therefore migrated to other arbitrary slots in the operational diagnostic sense. Whole-catalogue Hungarian matches do **not** reliably identify it across catalogues, as the low matching counts above show; this is not proof of a unique astrophysical-source identity or of a tracked migration trajectory. Median dominant gains are 9266.3 in walker 1 and 6218.6 in walker 6, versus 7318.2–10717.3 in the persistent walkers.

## Compatibility

| selection | joint ell9 survival | fraction | one-way ell9 survival |
| --- | --- | --- | --- |
| Dominant source_gain | 265/448 | 59.15% | 713/896 (79.58%) |
| Uniform PCG64 labels | 331/448 | 73.88% | 778/896 (86.83%) |
| ORACLE DIAGNOSTIC (six persistent walkers) | 184/240 | 76.67% | 424/480 (88.33%) |

All 448 unordered pair/index cases are used for dominant and uniform swaps. Uniform labels use NumPy Generator PCG64, seed `20260923`, one `(448, 2)` integer draw from `[0,9)`, ordered by walker A, walker B, retained index. There are no redraws. Oracle uses 240 corresponding cases among the six persistent walkers and is **not a usable source-selection rule**. Both results of each dominant swap are exactly the 896 directed transplants, so the same evaluations serve both diagnostics.

| selection on same six-walker cases | joint survival |
| --- | --- |
| dominant | 184/240 (76.67%) |
| uniform | 187/240 (77.92%) |
| oracle | 184/240 (76.67%) |

Dominant transplantation survival is 79.58%, compared with Stage-4Q persistent historical-slot single-source survival of **424/480 = 88.33%**. The matched persistent subset uses identical labels and states; the all-walker comparison adds walkers 1 and 6. No success threshold was chosen after observing results.

Dominant transplants `passes_old_ell10`: 397/896 (44.31%); diagnostic only.

Dominant transplants `passes_stage4p_ell10`: 367/896 (40.96%); diagnostic only.

The uniform control usually exchanges two non-dominant blocks. As a post-hoc description of the fixed draws (without redrawing or changing the rule): neither selected block is dominant in 372 cases, with 325 joint survivors (87.37%); exactly one is dominant in 68 cases, with zero survivors; both are dominant in 8 cases, with 6 survivors (75%). Thus its greater aggregate survival mostly concerns exchanges that leave the dominant sources untouched. This does not reverse the requested aggregate comparison or establish a tuned selection rule.

## Compatibility matrices

Rows are recipients and columns are donors for transplantation; every off-diagonal cell has 16 tests. Diagonals were not evaluated (—). All zero cells are retained. Joint matrices are symmetric by construction.

Directed dominant-source transplant ell9 survival fraction:

| recipient / donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 0.8750 | 0.9375 | 0.8125 | 0.9375 | 1.0000 | 0.0625 | 0.6250 |
| 1 | 1.0000 | — | 1.0000 | 0.8750 | 1.0000 | 0.9375 | 0.0000 | 0.5000 |
| 2 | 1.0000 | 0.8125 | — | 0.7500 | 0.8750 | 0.8750 | 0.0000 | 0.4375 |
| 3 | 1.0000 | 0.9375 | 1.0000 | — | 1.0000 | 1.0000 | 0.3125 | 0.6875 |
| 4 | 1.0000 | 0.9375 | 1.0000 | 0.8750 | — | 1.0000 | 0.0000 | 0.6875 |
| 5 | 1.0000 | 0.9375 | 1.0000 | 0.7500 | 0.9375 | — | 0.0000 | 0.5625 |
| 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | 1.0000 |
| 7 | 1.0000 | 0.8750 | 0.9375 | 0.9375 | 0.8750 | 1.0000 | 0.0000 | — |

Directed dominant-source transplant median delta logL:

| recipient / donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | -329.1 | -511.8 | -953.7 | 589.0 | -336.6 | -1796.9 | -938.2 |
| 1 | 359.9 | — | -240.3 | -663.8 | 549.8 | 120.5 | -1579.6 | -975.7 |
| 2 | 509.4 | 175.8 | — | -421.1 | 1047.4 | 400.7 | -1206.7 | -677.5 |
| 3 | 923.2 | 698.7 | 405.4 | — | 1305.2 | 999.3 | -779.6 | -180.2 |
| 4 | -601.9 | -534.0 | -1140.9 | -1450.8 | — | -545.9 | -2350.0 | -1634.0 |
| 5 | 556.7 | -249.7 | -616.1 | -939.7 | 278.6 | — | -1851.7 | -1230.2 |
| 6 | 1742.0 | 1408.4 | 1216.7 | 689.6 | 2143.0 | 1526.8 | — | 619.9 |
| 7 | 924.3 | 955.7 | 609.9 | 173.7 | 1879.9 | 1209.0 | -644.1 | — |

Symmetric dominant joint-swap ell9 survival fraction:

| recipient / donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 0.8750 | 0.9375 | 0.8125 | 0.9375 | 1.0000 | 0.0625 | 0.6250 |
| 1 | 0.8750 | — | 0.8125 | 0.8125 | 0.9375 | 0.8750 | 0.0000 | 0.3750 |
| 2 | 0.9375 | 0.8125 | — | 0.7500 | 0.8750 | 0.8750 | 0.0000 | 0.3750 |
| 3 | 0.8125 | 0.8125 | 0.7500 | — | 0.8750 | 0.7500 | 0.3125 | 0.6250 |
| 4 | 0.9375 | 0.9375 | 0.8750 | 0.8750 | — | 0.9375 | 0.0000 | 0.5625 |
| 5 | 1.0000 | 0.8750 | 0.8750 | 0.7500 | 0.9375 | — | 0.0000 | 0.5625 |
| 6 | 0.0625 | 0.0000 | 0.0000 | 0.3125 | 0.0000 | 0.0000 | — | 0.0000 |
| 7 | 0.6250 | 0.3750 | 0.3750 | 0.6250 | 0.5625 | 0.5625 | 0.0000 | — |

Directional asymmetry, explicitly reporting both directions:

| i,j | j → i | i → j | difference |
| --- | --- | --- | --- |
| 0,1 | 0.8750 | 1.0000 | -0.1250 |
| 0,2 | 0.9375 | 1.0000 | -0.0625 |
| 0,3 | 0.8125 | 1.0000 | -0.1875 |
| 0,4 | 0.9375 | 1.0000 | -0.0625 |
| 0,5 | 1.0000 | 1.0000 | +0.0000 |
| 0,6 | 0.0625 | 1.0000 | -0.9375 |
| 0,7 | 0.6250 | 1.0000 | -0.3750 |
| 1,2 | 1.0000 | 0.8125 | +0.1875 |
| 1,3 | 0.8750 | 0.9375 | -0.0625 |
| 1,4 | 1.0000 | 0.9375 | +0.0625 |
| 1,5 | 0.9375 | 0.9375 | +0.0000 |
| 1,6 | 0.0000 | 1.0000 | -1.0000 |
| 1,7 | 0.5000 | 0.8750 | -0.3750 |
| 2,3 | 0.7500 | 1.0000 | -0.2500 |
| 2,4 | 0.8750 | 1.0000 | -0.1250 |
| 2,5 | 0.8750 | 1.0000 | -0.1250 |
| 2,6 | 0.0000 | 1.0000 | -1.0000 |
| 2,7 | 0.4375 | 0.9375 | -0.5000 |
| 3,4 | 1.0000 | 0.8750 | +0.1250 |
| 3,5 | 1.0000 | 0.7500 | +0.2500 |
| 3,6 | 0.3125 | 1.0000 | -0.6875 |
| 3,7 | 0.6875 | 0.9375 | -0.2500 |
| 4,5 | 1.0000 | 0.9375 | +0.0625 |
| 4,6 | 0.0000 | 1.0000 | -1.0000 |
| 4,7 | 0.6875 | 0.8750 | -0.1875 |
| 5,6 | 0.0000 | 1.0000 | -1.0000 |
| 5,7 | 0.5625 | 1.0000 | -0.4375 |
| 6,7 | 1.0000 | 0.0000 | +1.0000 |

## Physical catalogue displacement

For dominant transplants, the signed physical f0 shift and permutation-invariant catalogue displacements are summarized below. Joint swaps have the same two directional catalogue displacements; every row is retained in the artifacts. Hungarian matching is exclusively a post-hoc physical diagnostic.

| ell9 outcome | metric | median | 5–95% |
| --- | --- | --- | --- |
| False | f0_displacement_hz | -8.43727e-08 | [-3.59596e-07, 2.89419e-07] |
| False | frequency_set_displacement_bins | 1.86578 | [0.040234, 3.78008] |
| False | six_coordinate_set_displacement | 0.0651463 | [0.039761, 0.101665] |
| True | f0_displacement_hz | 1.4747e-08 | [-3.41996e-07, 3.55401e-07] |
| True | frequency_set_displacement_bins | 2.00753 | [0.0752261, 3.86734] |
| True | six_coordinate_set_displacement | 0.068432 | [0.038793, 0.0997886] |

## Descriptive source_gain associations

Spearman correlations below treat each direction separately; joint survival is repeated for its two directions. These are dependent, correlated trace observations, not independent samples; no p-values, classifier, tuned rule, or causal claims are supplied. The donor/recipient ratio is the unmodified gain ratio (undefined denominators are excluded). Dominant and oracle selected rank is constant at 1, so its correlation is undefined.

| selection | feature | transplant survival rho | joint survival rho | delta logL rho |
| --- | --- | --- | --- | --- |
| dominant | donor_gain | 0.6509 | 0.4154 | 0.6809 |
| dominant | donor_rank | undefined | undefined | undefined |
| dominant | gain_ratio | 0.5290 | 0.0000 | 0.9741 |
| uniform | donor_gain | 0.1012 | -0.3125 | 0.6137 |
| uniform | donor_rank | -0.0996 | 0.2950 | -0.5600 |
| uniform | gain_ratio | 0.4971 | 0.0000 | 0.9245 |
| oracle | donor_gain | 0.4906 | 0.3057 | 0.6785 |
| oracle | donor_rank | undefined | undefined | undefined |
| oracle | gain_ratio | 0.3447 | 0.0000 | 0.9749 |

The donor/recipient ratio has a near-zero joint-survival correlation by construction when both directions are included: the reciprocal ratios share the same outcome. It should not be interpreted as evidence that gain mismatch is irrelevant. A supplementary symmetric description, `min(gain_A, gain_B) / max(gain_A, gain_B)`, has joint-survival rho **0.3026** for dominant, **0.5064** for uniform, and **0.1066** for oracle cases (one observation per unordered pair/index). This post-hoc summary changes no selection rule. The strong directional gain-ratio/delta-logL correlations are descriptive and do not make source_gain an additive integrated-likelihood decomposition.

## Fixed-label algebra and integrity

For fixed labels B,C, exchanging complete blocks is an involution and has unit absolute Jacobian. Since the actual K=9 implemented prior applies identical coordinate-wise terms to every labelled block, the product pi_impl(x) pi_impl(y) is preserved algebraically. Floating-point summation can change the evaluated joint log prior slightly. Applying every tested exchange twice returned both states byte-for-byte.

Maximum absolute joint implemented-prior discrepancy across 18 deterministic probes and all saved-state swaps: **1.1368683772161603e-13**. Deterministic probes include a +36 latent coordinate and use the actual prior, not an idealized Logistic replacement. The real configured-prior audit is separate from the cheap extracted-prior test.

For a symmetric fixed-label selection rule, the required constrained joint MH acceptance is `1[both logL > ell9] * min(1, exp(Delta_prior))`, where `Delta_prior = log pi_impl(x_new) + log pi_impl(y_new) - log pi_impl(x) - log pi_impl(y)`. Retain the computed residual if implementing this later. Dynamic argmax label selection is state dependent: the reverse move need not select the same labels. Its measured survival therefore establishes **physical feasibility only**, not exactness or an acceptance probability. No such kernel is implemented here.

All 128 originals have finite prior/likelihood and strict ell9 membership. Maximum cache discrepancies: prior 8.52651e-14, likelihood 2.93076e-08. Independent reconstruction verified 2272 proposed catalogues, including exact donor copies, untouched coordinates, both exchange directions, frozen original bases and uniform-label ordering. Structural failures: 0; nonfinite proposed priors: 0; nonfinite proposed likelihoods: 0. Protected input/model/configuration hashes remained unchanged.

Total cost including the reporting interruption: **3,296** scalar likelihood calls, **3,368** prior calls, and **128** reconstruction diagnostic calls. The completed resumed evaluation accounts for 2,272 likelihood/prior calls; base checks and deterministic prior probes were reused from the interrupted attempt. No sequential update, level construction, production, or evidence computation.

Regression: **179 passed** (all 166 existing Stage-4Q-era DNS/GB tests plus 13 Stage-4R tests). Tests include stable ranking/ties, exact block copies, involution, prior cancellation, untouched coordinates, fixed PCG64 labels, symmetric zero-cell bookkeeping, and no sequential updates. Pytest invokes no real LISA likelihood. Log: `/tmp/lisa_dns_stage4r_tests.log`.

A reporting-only interruption occurred after the first 896 dominant evaluations: NumPy rejected Boolean subtraction in the correlation helper. The original catch-all incorrectly recorded any exception as a structural failure. The preserved `interrupted_report.json` and `interrupted_runner.py` document that error; no structural assertion failed. The Boolean helper was fixed and regression-tested, verified base/gain/prior-probe data were reused, and the 896 unsaved evaluations were repeated. The resumed runner saves raw evaluations before aggregation. The independent analyzer also required an explicit numeric dtype for matrices whose diagonal is serialized as the string `nan`; it needed no likelihood reevaluation. The final structural count is zero.

## Interpretation and stopping point

1. **Dynamic recovery:** yes for the high-gain component in the critical frequency region, including new slots in walkers 1 and 6, with the physical-identity limitations of the post-hoc matching stated above.
2. **One-way compatibility:** the six persistent walkers exactly retain Stage-4Q’s 88.33% survival. Across all eight walkers it falls to 79.58%. Walker 6 is particularly asymmetric: its dominant block survives in other recipients only 6/112 times, while receiving other dominant blocks survives 112/112 times. Dynamic identification alone does not ensure transferable realizations.
3. **Simultaneous feasibility:** both walkers survive in 265/448 cases (59.15%), including 184/240 (76.67%) among the persistent six. Pairs involving walker 6 survive only 6/112 (5.36%). Several pair cells are exactly zero, so feasibility is strongly heterogeneous.
4. **Dominant versus uniform:** dominant survival is **14.73 percentage points lower** overall (59.15% versus 73.88%). On the same persistent-six cases it is also slightly lower (76.67% versus 77.92%). The uniform control mostly moves non-dominant sources, but the requested claim of materially greater dominant-exchange viability is not supported.

These results establish some physical feasibility and fixed-label prior cancellation, but are not uniformly favorable. **No recommendation to build an exact population source-exchange kernel is made from this diagnostic.** No kernel, state-dependent MCMC selection rule, tuned probabilities, new levels, prior/source_gain/slice changes, sequential population chain, production, or evidence implementation was made. Stage 4R stops here with level 9 frozen and Stage-4P level 10 rejected.

## Artifacts and reproduction

Code: [runner](../../examples/lisa_dns_stage4/dominant_source_exchange.py), [independent verification](../../examples/lisa_dns_stage4/analyze_source_exchange.py), [cheap tests](../../tests/experimental/test_dns_gb_dominant_source_exchange.py). Runtime artifacts: `/tmp/lisa_dns_stage4r_source_exchange/{design.json,bases.npz,dominant_swaps.npz,uniform_swaps.npz,oracle_swaps.npz,report.json,validation.json}`. The report retains every proposed pi_impl/logL, delta logL, contour outcome, source identity, correlation, and geometric displacement. These `/tmp` artifacts are local, not versioned; archive them for long-term retention. A fresh run refuses to overwrite existing output; the narrow `--resume-reporting-failure` option supports only the preserved Boolean-reporting interruption and refuses once dominant swap results exist. Additional artifacts include `{dominant,uniform,oracle}_evaluations.npz`, the preserved interruption files, and logs `/tmp/lisa_dns_stage4r.log` and `/tmp/lisa_dns_stage4r_interrupted.log`.
