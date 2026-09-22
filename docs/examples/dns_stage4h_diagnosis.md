# Stage-4H: diagnosis of rejected level 10

Final classification: **C — still ambiguous**, with persistent poor communication on the observed timescale. Neither coherent convergence of all walkers nor stable mutually inconsistent behavior across all measured quantities was observed. Level 10 remains rejected. The frozen ladder through level 9 is unchanged.

The resumed worktree was clean and had no partial Stage-4H files. Existing Stage-4G artifacts were inspected first, with no new MCMC steps. Trace filenames name the proposed level: `level9_*` sampled at ell8, while `level10_*` sampled at ell9 and is the relevant failed-calibration history. Both selection/calibration traces for levels 9 and 10 were inspected, together with all accepted level-5–9 traces. Exact saved thresholds were used: ell9 = -110252.99476697217 and diagnostic ell10 = -109401.11425363769. No threshold was recomputed.

## Existing failed-attempt traces

The table contains strict ell10 exceedances in consecutive 32-step blocks for each calibration walker. Blocks 1–4 are the originally declared burn-in; blocks 5–12 are retained. All blocks were inspected.

| Walker | Burn-in blocks 1–4 | Retained blocks 5–8 | Retained blocks 9–12 |
|---|---|---|---|
| 0 | 1.00000, 0.68750, 0.59375, 0.53125 | 0.21875, 0.50000, 0.34375, 0.87500 | 0.71875, 0.53125, 0.34375, 0.28125 |
| 1 | 0.84375, 0.87500, 0.96875, 0.46875 | 0.12500, 0.15625, 0.31250, 0.59375 | 0.18750, 0.37500, 0.06250, 0.00000 |
| 2 | 0.25000, 0.18750, 0.62500, 0.68750 | 0.87500, 0.59375, 0.96875, 0.93750 | 0.87500, 0.71875, 0.78125, 0.78125 |
| 3 | 0.28125, 0.00000, 0.06250, 0.03125 | 0.12500, 0.00000, 0.00000, 0.00000 | 0.00000, 0.00000, 0.00000, 0.09375 |
| 4 | 0.81250, 1.00000, 1.00000, 0.93750 | 0.59375, 0.90625, 0.65625, 0.68750 | 0.96875, 0.62500, 0.03125, 0.12500 |
| 5 | 0.62500, 0.87500, 0.96875, 0.78125 | 0.65625, 0.75000, 0.40625, 0.81250 | 0.75000, 0.68750, 0.93750, 0.96875 |
| 6 | 0.12500, 0.03125, 0.00000, 0.03125 | 0.00000, 0.00000, 0.00000, 0.00000 | 0.00000, 0.00000, 0.00000, 0.09375 |
| 7 | 0.00000, 0.00000, 0.00000, 0.03125 | 0.00000, 0.00000, 0.00000, 0.00000 | 0.00000, 0.09375, 0.18750, 0.00000 |

| Walker | Early / late retained fraction | Early / late mean logL | Change in mean logL | Early/late logL KS | Norm of change in nine f0_u means |
|---|---:|---:|---:|---:|---:|
| 0 | 0.48438 / 0.46875 | -109189.042 / -108883.111 | +305.931 | 0.25000 | 6.127 |
| 1 | 0.29688 / 0.15625 | -109589.474 / -109800.795 | -211.321 | 0.25781 | 2.423 |
| 2 | 0.84375 / 0.78906 | -108645.128 / -108388.587 | +256.541 | 0.21094 | 3.169 |
| 3 | 0.03125 / 0.02344 | -110010.866 / -109979.344 | +31.523 | 0.15625 | 2.670 |
| 4 | 0.71094 / 0.43750 | -108911.702 / -108855.135 | +56.567 | 0.31250 | 4.325 |
| 5 | 0.65625 / 0.83594 | -108819.390 / -108735.113 | +84.276 | 0.25781 | 4.143 |
| 6 | 0.00000 / 0.02344 | -110047.167 / -110025.722 | +21.444 | 0.10938 | 3.950 |
| 7 | 0.00000 / 0.07031 | -110008.489 / -109910.420 | +98.069 | 0.20312 | 2.205 |

Walkers 3, 6, and 7 stay almost entirely below ell10; walkers 2 and 5 remain high. Walker 4’s exceedance drops substantially, while walker 5 increases. There is no common direction of relaxation. Burn-in includes strong high-logL transients in walkers 1 and 4, but this does not explain away the retained separation. Early-to-late between-walker exceedance SD is 0.34415 → 0.33375; logL-mean SD is 584.07 → 668.59. Frequency-coordinate mean shifts have norms 2.20–6.13, so the walkers are not simply frozen at their starts.

End-of-existing-trace comparison uses the last 128 retained steps, not single endpoints. The nine-dimensional mean-f0_u distances are descriptive latent-coordinate distances, without sorting labels or clustering.

| Walker | Final-window source-label mean f0_u, labels 0–8 |
|---|---|
| 0 | -0.214, 3.030, 1.002, -1.039, 0.734, 0.027, 0.410, 0.172, -1.683 |
| 1 | -2.583, -0.185, -0.838, 0.184, -0.834, 0.644, -0.813, -0.729, -0.041 |
| 2 | -0.021, -0.314, -1.346, 0.162, -0.278, 0.909, 2.403, 0.492, -0.466 |
| 3 | 0.244, -0.860, -1.176, 0.702, -1.077, 1.388, -1.090, -2.291, -0.222 |
| 4 | 1.322, -3.850, -0.164, -2.345, 0.115, -0.514, -0.257, 0.146, 0.175 |
| 5 | -0.362, 0.371, 0.200, 1.570, -5.428, -0.025, -0.926, 1.303, -2.670 |
| 6 | 0.883, -1.000, -2.975, 0.102, -0.167, 2.962, -1.788, 0.351, -0.500 |
| 7 | 0.598, -0.937, 0.314, 1.460, 0.234, -0.103, -0.504, 0.207, 0.782 |

| Walker | Final-window source-label physical mean f0 (mHz), labels 0–8 |
|---|---|
| 0 | 1.839057, 1.848599, 1.841373, 1.835542, 1.842828, 1.839912, 1.840885, 1.840865, 1.834974 |
| 1 | 1.831823, 1.838406, 1.836401, 1.840925, 1.836180, 1.843097, 1.836324, 1.836945, 1.839825 |
| 2 | 1.839914, 1.838476, 1.834323, 1.840815, 1.838955, 1.844026, 1.848141, 1.842021, 1.838109 |
| 3 | 1.841225, 1.836644, 1.834805, 1.843155, 1.835553, 1.845753, 1.835495, 1.833144, 1.838959 |
| 4 | 1.845732, 1.830622, 1.839302, 1.833180, 1.840600, 1.838099, 1.839134, 1.840627, 1.840883 |
| 5 | 1.838623, 1.841633, 1.841007, 1.845758, 1.830098, 1.839564, 1.836106, 1.845636, 1.831612 |
| 6 | 1.844086, 1.835899, 1.831089, 1.840518, 1.839202, 1.847696, 1.833286, 1.841646, 1.837778 |
| 7 | 1.842807, 1.836034, 1.841070, 1.846008, 1.841175, 1.839413, 1.837592, 1.840931, 1.842866 |

| Walker | Distances to walkers 0–7 |
|---|---|
| 0 | 0.00, 5.35, 5.05, 6.22, 7.57, 7.52, 7.09, 5.50 |
| 1 | 5.35, 0.00, 4.39, 3.46, 6.24, 6.39, 5.04, 4.12 |
| 2 | 5.05, 4.39, 0.00, 4.64, 5.64, 7.00, 5.07, 4.07 |
| 3 | 6.22, 3.46, 4.64, 0.00, 5.68, 6.66, 3.86, 3.80 |
| 4 | 7.57, 6.24, 5.64, 5.68, 0.00, 8.77, 6.10, 4.93 |
| 5 | 7.52, 6.39, 7.00, 6.66, 8.77, 0.00, 7.66, 6.93 |
| 6 | 7.09, 5.04, 5.07, 3.86, 6.10, 7.66, 0.00, 5.06 |
| 7 | 5.50, 4.12, 4.07, 3.80, 4.93, 6.93, 5.06, 0.00 |

The final-window fractions span 0.02344–0.83594 and walker mean logL spans −110025.72 to −108388.59. The largest mean-logL separation is about 1637.14. Full signed pairwise mean-logL, exceedance, source-wise latent/physical frequency differences, and descriptive logL KS matrices are saved in the JSON. These are different labelled regions/behaviors; no physical catalogue families are inferred.

## Selection/calibration differences across accepted levels

| Level | Selection ratio | Calibration ratio | Difference | Selection SE | Calibration SE | Difference / combined SE | Calibration interval | Pooled logL KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.367676 | 0.430664 | 0.062988 | 0.08905 | 0.06637 | 0.567 | [0.27372, 0.58761] | 0.06787 |
| 6 | 0.367676 | 0.437988 | 0.070312 | 0.09875 | 0.09564 | 0.511 | [0.21184, 0.66414] | 0.07568 |
| 7 | 0.367676 | 0.486328 | 0.118652 | 0.10143 | 0.07611 | 0.936 | [0.30635, 0.66631] | 0.12354 |
| 8 | 0.367676 | 0.551270 | 0.183594 | 0.10022 | 0.07560 | 1.462 | [0.37250, 0.73004] | 0.19434 |
| 9 | 0.367676 | 0.595215 | 0.227539 | 0.10425 | 0.09001 | 1.652 | [0.38238, 0.80805] | 0.23926 |

| Level | Selection / calibration mean logL | Selection / calibration median logL | Calibration between-walker SE |
|---|---:|---:|---:|
| 5 | -112362.561 / -112197.849 | -112542.417 / -112472.918 | 0.06637 |
| 6 | -111748.166 / -111701.129 | -112071.850 / -111998.193 | 0.09564 |
| 7 | -111256.584 / -111073.917 | -111532.599 / -111381.253 | 0.07611 |
| 8 | -110745.831 / -110452.954 | -110990.313 / -110717.385 | 0.07560 |
| 9 | -110074.067 / -109674.100 | -110449.284 / -110057.963 | 0.09001 |

A. The differences are not compelling evidence of bias given the reported uncertainties. Selection and calibration intervals overlap at every accepted level. The differences are 0.51–1.65 combined-SE units, where combined SE is sqrt(SE_selection² + SE_calibration²). This is only a descriptive compatibility scale: the selection threshold is estimated from its own bank, so its fraction is not an independent mass estimate and this calculation is not a formal z-test or a propagated threshold-confidence interval. Conditional calibration intervals cover the target at levels 5–7; they narrowly exclude it at 8 and 9. A definitive compatibility test would require stronger assumptions than these short, adaptively related histories support.

B. The saved calibration banks have consistently higher pooled logL means and medians than the selection banks. KS distances rise from 0.068 to 0.239. This is systematic descriptive bank divergence, not proof of sampler bias or statistically independent evidence across levels. No IID KS p-values are used for correlated traces.

C. Between-walker calibration SE varies 0.0664, 0.0956, 0.0761, 0.0756, 0.0900 over levels 5–9, then 0.1174 at failed level 10. The trend is not monotonic, but disagreement is persistently large and strongest at the failure. Threshold adaptation and inherited walker states correlate consecutive-level comparisons, so no pooled significance claim is made.

## Conditional fixed-contour continuation

The existing traces were classified C before further sampling: persistent separation coexisted with substantial transient changes in logL/frequency statistics. This met the user’s explicit condition for one continuation. The predeclared decision is saved in `decision_before_continuation.json`. Exactly eight existing `level10_calibration_trace.npz` end states were continued at unchanged ell9 for 512 steps each, with the unchanged isotropic scales and 20/40/40 proposal. Fresh streams were 810100–810107. No samples were discarded, and the two 256-step halves were analyzed as declared. No calibration acceptance or level construction call was made.

| Walker | Continuation blocks 1–8 | Continuation blocks 9–16 |
|---|---|---|
| 0 | 0.56250, 0.50000, 0.56250, 0.71875, 0.59375, 0.00000, 0.00000, 0.59375 | 0.75000, 0.75000, 0.03125, 0.00000, 0.00000, 0.00000, 0.00000, 0.43750 |
| 1 | 0.28125, 0.93750, 0.37500, 0.37500, 0.87500, 0.71875, 0.28125, 0.59375 | 0.68750, 0.40625, 0.46875, 0.53125, 0.87500, 0.75000, 0.68750, 0.25000 |
| 2 | 0.56250, 0.59375, 0.84375, 0.68750, 0.21875, 0.59375, 0.75000, 0.96875 | 0.68750, 1.00000, 0.81250, 0.96875, 0.40625, 0.84375, 0.62500, 0.31250 |
| 3 | 0.00000, 0.00000, 0.00000, 0.00000, 0.09375, 0.03125, 0.00000, 0.06250 | 0.00000, 0.12500, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| 4 | 0.00000, 0.03125, 0.00000, 0.00000, 0.00000, 0.03125, 0.37500, 0.75000 | 0.81250, 0.81250, 0.21875, 0.15625, 0.00000, 0.46875, 0.71875, 0.40625 |
| 5 | 1.00000, 0.96875, 0.65625, 0.93750, 0.59375, 0.40625, 0.62500, 0.34375 | 0.31250, 0.56250, 0.53125, 0.81250, 0.43750, 0.50000, 0.75000, 0.78125 |
| 6 | 0.31250, 0.62500, 0.18750, 0.53125, 0.90625, 0.12500, 0.28125, 0.03125 | 0.18750, 0.06250, 0.03125, 0.00000, 0.09375, 0.03125, 0.21875, 0.00000 |
| 7 | 0.00000, 0.03125, 0.09375, 0.21875, 0.03125, 0.03125, 0.00000, 0.03125 | 0.00000, 0.31250, 0.12500, 0.15625, 0.09375, 0.00000, 0.00000, 0.00000 |

| Walker | First / second 256 fraction | First / second mean logL | Mean logL change | Half-to-half logL KS | Norm of change in f0_u means |
|---|---:|---:|---:|---:|---:|
| 0 | 0.44141 / 0.24609 | -109388.161 / -109688.027 | -299.866 | 0.33203 | 4.470 |
| 1 | 0.55469 / 0.58203 | -109332.351 / -108401.521 | +930.830 | 0.45312 | 3.018 |
| 2 | 0.65234 / 0.70703 | -109074.875 / -108972.706 | +102.169 | 0.16797 | 4.052 |
| 3 | 0.02344 / 0.01562 | -109897.264 / -109986.700 | -89.436 | 0.23047 | 4.954 |
| 4 | 0.14844 / 0.44922 | -109794.438 / -109161.683 | +632.755 | 0.47656 | 3.524 |
| 5 | 0.69141 / 0.58594 | -108632.630 / -108980.737 | -348.107 | 0.23828 | 5.706 |
| 6 | 0.37500 / 0.07812 | -109527.296 / -109788.525 | -261.228 | 0.32812 | 3.221 |
| 7 | 0.05469 / 0.08594 | -109895.590 / -109864.805 | +30.785 | 0.11328 | 6.482 |

The final-half exceedance range is 0.015625–0.70703125. Between-walker exceedance SD is 0.26480 → 0.27065 across the continuation halves, and logL-mean SD is 439.04 → 559.40. There is no continuing contraction toward a common distribution. Walker 3 remains low throughout; walker 7 mostly remains low; walkers 2 and 5 remain relatively high. Walker 6 explores higher likelihoods then returns low. Walker 4 rises from 0.1484 to 0.4492 exceedance. Walker 1’s exceedance changes little but its mean logL increases by 930.83, emphasizing that exceedance alone can hide distribution shifts.

Latent frequency means continue moving: half-to-half vector norms are 3.02–6.48. Final-half pairwise mean-frequency distances remain 2.91–7.70. These finite-window mean differences, together with ongoing within-walker movement, do not identify fixed physical modes or establish that different walkers should converge to one identical frequency vector. The relevant test is agreement of distributions, which remains unestablished.

Conclusion: **C — mixed/ambiguous**, with strong evidence of slow constrained communication on this horizon. The continuation does not support a simple claim that 128 burn-in steps were merely too short and the walkers now equilibrate. It also does not meet a clean “stable in time but mutually different” criterion, because several walkers keep changing their logL and frequency distributions. Persistent constrained mixing is a concern, but separating it from longer transient relaxation remains unresolved. No further sampling or tuning follows.

Continuation integrity: contour 0, cache 0, nonfinite 0, parameter-contract 0, slice 0. All 4096 steps and 8 resumed states were checked. Cost: 27,342 slice likelihood proxy evaluations, 4,104 direct cache-check likelihood evaluations (each also checks prior), 0 extra initialization likelihood evaluations because cached states were reused.

## Artifacts and validation

`/tmp/lisa_dns_stage4h_diagnosis/existing_trace_analysis.json` contains complete per-walker/per-block mean, median, minimum and maximum logL, exceedance fractions, all nine f0_u means/variances, physical-frequency means/variances, early/late changes, and pairwise matrices for both banks at attempts 9 and 10. Corresponding `level*_blocks.csv` files provide one row per walker and block with all source-label columns. The attempt-9 blocks use the diagnostic ell10 for the requested comparison, but their generating contour is explicitly ell8.

Physical f0 is decoded cheaply as f_min + sigmoid(f0_u)*(f_max−f_min), matching the inspected unordered transform and previously audited real bounds [0.001830003805175038, 0.0018500126839167937]. Means are taken after transforming individual samples. No waveform, likelihood, sorting, or catalogue reconstruction is involved in these diagnostics.

`/tmp/lisa_dns_stage4h_continuation/report.json`, `trace.npz`, `initial_states.npz`, and `blocks.csv` contain the complete fixed-contour experiment and identical diagnostics. All source input hashes and random streams are recorded. The ladder and accepted checkpoint files are not modified.

Evidence reconstruction remains unvalidated and out of scope. No level-10 acceptance/rebuild, level 11, production, proposal/scaling change, or posterior reconstruction occurred.

Validation: all 101 existing tests and three new cheap block/early-late/classification tests passed (104 total). The expensive continuation is outside pytest. All hashed Stage-4G inputs were verified unchanged after analysis and continuation; continuation initial states match the original eight calibration end states exactly.
