# Stage 4U: one fixed-contour frequency-guided population benchmark

## Frozen design and invariance

Target: `ell_9 = -110252.99476697217`, `log X_9 = -7.67544001580533`. Every walker targets `pi_impl(u) I(logL(u)>ell_9)`; the ensemble target is their product. Stage-4P level 10 remains rejected. No level is constructed in this benchmark.

Exactly eight walkers undergo 512 population sweeps, with no burn-in removal. The initial positions, cached priors and cached likelihoods are byte-identical to the Stage-4H starts, the original Stage-4G calibration endpoints and the later Stage-4N/4O controlled starts. One sweep applies eight independent validated isotropic slice transitions, then four reciprocal exchanges on disjoint walker pairs.

The slice uses unchanged `proposals.build_parameter_step`, all scales `pi/sqrt(3)`, mixture 20% global / 40% uniformly selected labelled source / 40% uniformly selected unordered source pair, `max_steps=10`, `max_shrinkage=100`. Neither `proposals.py`, generic DNS code nor the prior is modified. The JAX `lax.map` batch simply applies the same scalar transition independently to eight states. Per-walker slice streams reuse Stage-4H seeds 810100–810107, splitting once per transition; trajectories diverge as soon as exchanges change states.

The frozen physical interval is `[0.0018407250, 0.0018412366] Hz`, `f_star = 0.0018409808 Hz`, `sigma_f = 2.558e-7 Hz`. Labels are selected independently from `log p_k = -0.5*((f0_k-f_star)/sigma_f)^2 - logsumexp(log w)`, on current **post-slice** states. No source_gain is evaluated or used in any transition. NumPy PCG64 seed 2026092401 preallocates independent Gumbels (512×4×2×9), and labels use Gumbel-max directly on normalized log weights. Seed 2026092402 independently preallocates exactly one MH uniform per pair (512×4). No probabilities are clipped, no draws are retried and no streams are adapted. Explicit probabilities are checked in extended precision; finite log probabilities are stored and used for MH. As with any floating-point PRNG, realizable random draws have finite precision; tiny mathematical probabilities are not advertised as empirically resolvable.

The predetermined seven-round circle schedule covers every unordered pair exactly once per cycle, with every walker appearing once in each round. The 512-sweep run has 73 complete cycles plus the first round: four pairs receive 74 opportunities and the other 24 receive 73. The schedule depends on neither states nor likelihoods.

For selected slots i,j, exchange all six latent coordinates. Incoming blocks remain in slots i,j. With `q_fwd = p_i(x)p_j(y)` and `q_rev = p_i(x')p_j(y')`, compute the full evaluated joint prior change plus `log q_rev - log q_fwd`. If either proposed logL is at or below ell9, reject jointly; otherwise accept both or neither with one uniform and `log U < min(0, log r)`. The reverse probabilities are recomputed on proposed catalogues at the same slot labels. Numerical prior residuals are retained, not set to zero.

Each unchanged single-walker slice kernel preserves its constrained target. Independent application to all eight walkers preserves the product target. Each fixed pair exchange is an involutive coordinate permutation with unit absolute Jacobian and uses the exact state-dependent MH ratio, so preserves that two-walker product. Four disjoint pair kernels preserve the eight-walker product. Deterministic composition of these invariant steps, including the predetermined round-robin cycle, preserves the same joint target. The schedule requires no extra MH factor. **Invariance, not reversibility of the entire composed sweep, is asserted.**

The frozen design and random draws were saved before model evaluation in `/tmp/lisa_dns_stage4u_population/design.json` and `random_schedule.npz`. All 202 regression tests passed before starting the single run. No classification threshold or selector parameter is adjusted from outcomes.

## Outcome

**Classification: B — modest improvement overall, with a strong improvement in likelihood agreement.** Cross-walker logL means, diagnostic exceedance fractions and pairwise logL KS improve substantially in both halves and in the mechanically cost-matched prefix. Exact provenance replay confirms physical source-realization transfers. However, whole-catalogue permutation-invariant distances and labelled-coordinate exploration do not show comparably strong or consistent improvement. This limitation prevents a strong, broad physical-mixing classification based on the exchange acceptance or likelihood agreement alone.

The answer to the narrow sequential benchmark question is **yes, for cross-walker communication of the likelihood-relevant component, with incomplete whole-catalogue mixing**. The agreement improvement is convincing enough that **one controlled level-10 reconstruction with this population kernel may be justified next**. None is performed here. This is not a claim of converged physical catalogues or reliable evidence.

Exactly 512 population sweeps completed: **4,096 slice transitions and 2,048 joint exchange proposals**. All eight walkers remain strictly above ell9. The benchmark was run once and was not extended. No baseline was rerun.

## Primary mixing comparison

SD uses sample standard deviation across the eight walker summaries. MAD is the unscaled median absolute deviation of the eight **walker logL means**, and IQR is their 75th–25th percentile difference. KS is the median of 28 pairwise empirical logL KS distances; no IID p-values are inferred from these correlated trajectories. H is primary; O Hybrid P is descriptive.

| run / window | logL mean SD | mean MAD | mean IQR | median logL KS | old ell10 exceedance SD | Stage-P ell10 exceedance SD |
| --- | --- | --- | --- | --- | --- | --- |
| H isotropic / first256 | 439.037 | 359.782 | 551.744 | 0.402344 | 0.264798 | 0.242644 |
| H isotropic / second256 | 559.401 | 442.034 | 828.866 | 0.410156 | 0.270649 | 0.257252 |
| H isotropic / all512 | 446.210 | 403.013 | 728.905 | 0.332031 | 0.252070 | 0.236604 |
| O Hybrid P / first256 | 490.850 | 334.761 | 628.838 | 0.402344 | 0.273746 | 0.265467 |
| O Hybrid P / second256 | 598.350 | 251.241 | 443.729 | 0.294922 | 0.202413 | 0.198121 |
| O Hybrid P / all512 | 471.850 | 291.811 | 488.809 | 0.289062 | 0.222468 | 0.213312 |
| U population / first256 | 45.053 | 37.275 | 56.328 | 0.082031 | 0.029264 | 0.030397 |
| U population / second256 | 92.451 | 47.984 | 174.580 | 0.087891 | 0.038440 | 0.028226 |
| U population / all512 | 54.365 | 50.321 | 94.178 | 0.062500 | 0.026136 | 0.020470 |

Over all 512 sweeps, H→U logL-mean SD is **446.21→54.36**, MAD **403.01→50.32**, IQR **728.91→94.18**, old-threshold exceedance SD **0.25207→0.02614**, and median pairwise KS **0.33203→0.06250**. This agreement gain is present in both halves, not just in the pooled trace. U has somewhat greater between-walker mean spread in its second half than its first, so the trajectory should not be declared stationary from these summaries.

Per-walker mean logL (columns are original walker indices, not physical catalogue-family identities):

| run / window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H isotropic / first256 | -109388.16 | -109332.35 | -109074.87 | -109897.26 | -109794.44 | -108632.63 | -109527.30 | -109895.59 |
| H isotropic / second256 | -109688.03 | -108401.52 | -108972.71 | -109986.70 | -109161.68 | -108980.74 | -109788.52 | -109864.80 |
| H isotropic / all512 | -109538.09 | -108866.94 | -109023.79 | -109941.98 | -109478.06 | -108806.68 | -109657.91 | -109880.20 |
| O Hybrid P / first256 | -109344.83 | -108980.19 | -108521.81 | -109893.12 | -109840.08 | -109295.16 | -109686.50 | -109860.78 |
| O Hybrid P / second256 | -109303.32 | -108049.25 | -109405.91 | -109985.17 | -109177.50 | -109569.73 | -109679.98 | -109822.44 |
| O Hybrid P / all512 | -109324.08 | -108514.72 | -108963.86 | -109939.15 | -109508.79 | -109432.44 | -109683.24 | -109841.61 |
| U population / first256 | -109422.19 | -109505.96 | -109374.15 | -109446.12 | -109401.15 | -109439.83 | -109386.32 | -109476.71 |
| U population / second256 | -109561.32 | -109559.25 | -109546.60 | -109353.75 | -109500.15 | -109358.81 | -109380.64 | -109524.45 |
| U population / all512 | -109491.75 | -109532.60 | -109460.38 | -109399.94 | -109450.65 | -109399.32 | -109383.48 | -109500.58 |

Per-walker exceedance fractions for the **failed old isotropic ell10 = -109401.11425363769**; diagnostic only:

| run / window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H isotropic / first256 | 0.4414 | 0.5547 | 0.6523 | 0.0234 | 0.1484 | 0.6914 | 0.3750 | 0.0547 |
| H isotropic / second256 | 0.2461 | 0.5820 | 0.7070 | 0.0156 | 0.4492 | 0.5859 | 0.0781 | 0.0859 |
| H isotropic / all512 | 0.3438 | 0.5684 | 0.6797 | 0.0195 | 0.2988 | 0.6387 | 0.2266 | 0.0703 |
| O Hybrid P / first256 | 0.4805 | 0.5312 | 0.8477 | 0.0508 | 0.1211 | 0.4453 | 0.2227 | 0.1055 |
| O Hybrid P / second256 | 0.3203 | 0.6680 | 0.4648 | 0.0195 | 0.3281 | 0.3047 | 0.2188 | 0.1055 |
| O Hybrid P / all512 | 0.4004 | 0.5996 | 0.6562 | 0.0352 | 0.2246 | 0.3750 | 0.2207 | 0.1055 |
| U population / first256 | 0.3633 | 0.3789 | 0.4219 | 0.3398 | 0.4023 | 0.3750 | 0.4180 | 0.3594 |
| U population / second256 | 0.2852 | 0.3047 | 0.3086 | 0.3203 | 0.3438 | 0.3867 | 0.3828 | 0.3672 |
| U population / all512 | 0.3242 | 0.3418 | 0.3652 | 0.3301 | 0.3730 | 0.3809 | 0.4004 | 0.3633 |

Per-walker exceedance fractions for the **failed Stage-4P ell10 = -109301.45870884096**; diagnostic only:

| run / window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H isotropic / first256 | 0.3945 | 0.4961 | 0.5977 | 0.0039 | 0.1367 | 0.6016 | 0.3125 | 0.0273 |
| H isotropic / second256 | 0.2305 | 0.5625 | 0.6445 | 0.0156 | 0.4219 | 0.5469 | 0.0586 | 0.0586 |
| H isotropic / all512 | 0.3125 | 0.5293 | 0.6211 | 0.0098 | 0.2793 | 0.5742 | 0.1855 | 0.0430 |
| O Hybrid P / first256 | 0.4141 | 0.4805 | 0.8047 | 0.0312 | 0.0781 | 0.3867 | 0.1641 | 0.0859 |
| O Hybrid P / second256 | 0.2852 | 0.6289 | 0.4062 | 0.0195 | 0.3164 | 0.2383 | 0.1328 | 0.0625 |
| O Hybrid P / all512 | 0.3496 | 0.5547 | 0.6055 | 0.0254 | 0.1973 | 0.3125 | 0.1484 | 0.0742 |
| U population / first256 | 0.3281 | 0.3047 | 0.3672 | 0.3047 | 0.3477 | 0.3398 | 0.3789 | 0.2969 |
| U population / second256 | 0.2539 | 0.2852 | 0.2695 | 0.3008 | 0.3086 | 0.3320 | 0.3203 | 0.3281 |
| U population / all512 | 0.2910 | 0.2949 | 0.3184 | 0.3027 | 0.3281 | 0.3359 | 0.3496 | 0.3125 |

No quantile was promoted into a new level; neither failed threshold was retried.

## Autocorrelation and unique states

| run / window | median labelled f0_u ESS | minimum labelled f0_u ESS | median logL ESS |
| --- | --- | --- | --- |
| H isotropic / first256 | 5.386 | 2.806 | 33.528 |
| H isotropic / second256 | 5.248 | 2.810 | 29.681 |
| H isotropic / all512 | 6.923 | 2.895 | 33.529 |
| O Hybrid P / first256 | 29.944 | 2.868 | 46.260 |
| O Hybrid P / second256 | 30.704 | 3.607 | 23.645 |
| O Hybrid P / all512 | 55.277 | 3.852 | 25.548 |
| U population / first256 | 6.792 | 3.170 | 108.393 |
| U population / second256 | 6.301 | 2.889 | 97.314 |
| U population / all512 | 7.873 | 3.085 | 211.937 |

Every walker has unique-state fraction **1.000** in each of the three windows, for H, U and O Hybrid P. Uniqueness is therefore not discriminating. ESS uses the established initial-positive, monotone paired-autocorrelation estimator, with constant-coordinate ESS zero and no antithetic bonus. U’s full-run median logL ESS rises from 33.53 to 211.94, but its labelled f0_u median/minimum ESS remain only 7.87/3.08. Label exchange can improve labelled summaries without establishing physical convergence; cross-walker interaction also means eight traces are not eight independent convergence replications.

## Permutation-invariant physical movement

The unchanged Stage-4I metrics use Hungarian matching, with frequency-set RMS in Fourier bins (`DF=1/31536000 Hz`) and six-coordinate RMS normalized by prior widths, with psi period pi and lam period 2pi. Frequency distributions use all cross-time catalogue pairs; full-coordinate distances use every 32nd window endpoint and all cross-pairs of those endpoints. These are post-hoc measurements only and do not influence selection or acceptance.

| run / window | median frequency-set RMS | median sorted-frequency mean distance | median six-coordinate set RMS |
| --- | --- | --- | --- |
| H isotropic / first256 | 94.397 | 44.330 | 0.249317 |
| H isotropic / second256 | 103.429 | 45.808 | 0.259253 |
| H isotropic / all512 | 108.162 | 37.090 | 0.256308 |
| H isotropic / last32 | 113.740 | 104.554 | 0.257559 |
| O Hybrid P / first256 | 105.076 | 17.672 | 0.260624 |
| O Hybrid P / second256 | 107.378 | 18.769 | 0.256447 |
| O Hybrid P / all512 | 106.419 | 12.497 | 0.258209 |
| O Hybrid P / last32 | 98.986 | 36.844 | 0.260470 |
| U population / first256 | 94.154 | 46.382 | 0.256022 |
| U population / second256 | 101.484 | 64.002 | 0.260729 |
| U population / all512 | 101.303 | 42.073 | 0.257219 |
| U population / last32 | 93.976 | 79.861 | 0.261755 |

Whole-catalogue physical improvement is **mixed/absent**: the second-half median sorted-frequency mean distance is **45.81 bins for H versus 64.00 for U**, and six-coordinate set RMS is **0.25925 versus 0.26073**. Over all 512 sweeps, those values are **37.09 versus 42.07 bins**, and **0.25631 versus 0.25722**. The final 32-sweep frequency mean distance improves (104.55→79.86 bins), but that shorter descriptive window does not override the predeclared 256-sweep result. O Hybrid P moves the whole frequency set more effectively by these measures. Cross-time RMS includes within-walker spread and is not itself a convergence test; the mean-distance result likewise supplies no broad convergence claim for U.

Per-walker movement from the common starting catalogue. All distances below are permutation invariant; max is over all 512 states. Smaller endpoint distance is not automatically worse or better mixing.

| run | walker | start→end frequency bins | max frequency bins from start | start→end full RMS | max full RMS from start |
| --- | --- | --- | --- | --- | --- |
| H isotropic | 0 | 85.808 | 140.261 | 0.242891 | 0.307194 |
| H isotropic | 1 | 106.705 | 177.727 | 0.256508 | 0.289365 |
| H isotropic | 2 | 123.167 | 155.652 | 0.230920 | 0.278276 |
| H isotropic | 3 | 123.261 | 180.507 | 0.269175 | 0.282622 |
| H isotropic | 4 | 172.797 | 194.037 | 0.252231 | 0.275614 |
| H isotropic | 5 | 100.958 | 175.719 | 0.225005 | 0.301610 |
| H isotropic | 6 | 105.564 | 118.338 | 0.263951 | 0.265753 |
| H isotropic | 7 | 162.591 | 273.189 | 0.252632 | 0.311885 |
| O Hybrid P | 0 | 84.590 | 171.828 | 0.243798 | 0.310752 |
| O Hybrid P | 1 | 30.762 | 209.140 | 0.247076 | 0.321422 |
| O Hybrid P | 2 | 67.348 | 160.786 | 0.258852 | 0.281066 |
| O Hybrid P | 3 | 105.083 | 196.898 | 0.237100 | 0.290958 |
| O Hybrid P | 4 | 56.760 | 265.736 | 0.223232 | 0.309959 |
| O Hybrid P | 5 | 85.697 | 222.720 | 0.238226 | 0.318197 |
| O Hybrid P | 6 | 84.458 | 224.211 | 0.263721 | 0.295153 |
| O Hybrid P | 7 | 119.673 | 225.484 | 0.260072 | 0.287159 |
| U population | 0 | 60.543 | 117.670 | 0.237522 | 0.294390 |
| U population | 1 | 75.580 | 171.366 | 0.240420 | 0.297093 |
| U population | 2 | 114.052 | 158.606 | 0.248740 | 0.275566 |
| U population | 3 | 57.362 | 141.944 | 0.262963 | 0.277031 |
| U population | 4 | 141.495 | 189.066 | 0.245218 | 0.296444 |
| U population | 5 | 124.826 | 155.066 | 0.242055 | 0.321368 |
| U population | 6 | 76.851 | 162.367 | 0.250208 | 0.302900 |
| U population | 7 | 174.760 | 225.442 | 0.264458 | 0.295441 |

All 28 final-window (second256) pair distances, including zero cells if any:

| pair | H freq RMS | U freq RMS | H freq mean distance | U freq mean distance | H full RMS | U full RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 90.772 | 101.434 | 38.861 | 68.236 | 0.261235 | 0.265230 |
| 0,2 | 85.587 | 124.322 | 31.570 | 92.447 | 0.262944 | 0.277049 |
| 0,3 | 101.416 | 82.016 | 16.945 | 32.090 | 0.274059 | 0.260631 |
| 0,4 | 87.099 | 115.240 | 34.545 | 97.815 | 0.260272 | 0.260826 |
| 0,5 | 132.379 | 68.171 | 96.875 | 10.872 | 0.260074 | 0.248848 |
| 0,6 | 81.463 | 122.952 | 26.680 | 102.608 | 0.252171 | 0.266878 |
| 0,7 | 153.204 | 87.234 | 127.459 | 31.637 | 0.269781 | 0.261797 |
| 1,2 | 95.583 | 107.730 | 48.595 | 56.184 | 0.247919 | 0.264685 |
| 1,3 | 111.102 | 95.773 | 43.581 | 44.000 | 0.254455 | 0.262456 |
| 1,4 | 95.637 | 96.348 | 48.034 | 63.489 | 0.260819 | 0.255522 |
| 1,5 | 126.236 | 100.749 | 85.731 | 63.888 | 0.248069 | 0.255615 |
| 1,6 | 87.909 | 96.656 | 36.841 | 56.692 | 0.255279 | 0.261567 |
| 1,7 | 144.184 | 112.616 | 114.530 | 67.335 | 0.259783 | 0.262905 |
| 2,3 | 101.817 | 112.357 | 17.977 | 64.116 | 0.267681 | 0.275103 |
| 2,4 | 84.644 | 81.696 | 26.949 | 12.058 | 0.256983 | 0.254571 |
| 2,5 | 151.384 | 122.559 | 121.376 | 87.608 | 0.253230 | 0.256139 |
| 2,6 | 83.488 | 89.788 | 31.630 | 25.626 | 0.256945 | 0.254651 |
| 2,7 | 174.955 | 143.842 | 152.769 | 106.138 | 0.273368 | 0.283459 |
| 3,4 | 104.066 | 100.600 | 26.867 | 69.313 | 0.261738 | 0.257803 |
| 3,5 | 154.095 | 83.584 | 109.045 | 29.213 | 0.258723 | 0.251473 |
| 3,6 | 101.591 | 106.808 | 26.103 | 72.218 | 0.257148 | 0.272354 |
| 3,7 | 175.623 | 101.534 | 141.077 | 45.788 | 0.276895 | 0.256197 |
| 4,5 | 141.023 | 113.147 | 107.880 | 93.027 | 0.250171 | 0.242158 |
| 4,6 | 80.065 | 69.142 | 19.395 | 23.929 | 0.250252 | 0.254124 |
| 4,7 | 164.348 | 137.042 | 140.267 | 112.038 | 0.276010 | 0.265921 |
| 5,6 | 128.656 | 120.520 | 93.656 | 97.470 | 0.257176 | 0.251328 |
| 5,7 | 102.793 | 89.430 | 38.528 | 30.871 | 0.251597 | 0.255562 |
| 6,7 | 151.030 | 139.264 | 126.266 | 110.861 | 0.270566 | 0.273322 |

## Physical communication and provenance

Accepted exchanges produce **2004 directional transfers**. Of these, **1916** place a realization in a walker other than the walker where that realization was born; **1841** are its first visit to the receiving walker. **1077 distinct exact-realization tokens** visit another walker, and **7 transfers** involve exact tokens present in the initial ensemble. **51/72 initial source-block ancestries** visit another walker through accepted exchanges.

These counts are obtained by replaying operations, not by nearest-source matching or assigning physical family names. Initial ancestry tags follow descendants through slice updates. Separate exact-realization tokens are conservatively replaced whenever the slice direction targets a block, even if a hypothetical roundoff-identical update occurred; parent token, birth walker and sweep are recorded. Unselected tokens persist. Acceptance swaps both tokens/ancestries; rejection changes neither. Thus ancestry transfer is not conflated with transfer of an unmodified initial realization. Value checks separately prove the copied six-coordinate realizations are exact.

This establishes actual source-realization communication, rather than merely fast label changes. Together with sharply improved likelihood agreement it supports a targeted mixing benefit. It does not imply full-catalogue source-set convergence: the global physical distances above remain limiting evidence.

## Exchange diagnostics

Of **2048** joint proposals, **1002** are accepted (**48.9258%**). Both catalogues survive ell9 in **1034** cases (**50.4883%**). The exact joint MH correction rejects **32/1034 = 3.0948%** of joint contour survivors. This is measured sequential soft-selector acceptance, not Stage-4T’s prescribed dominant-swap replay statistic.

| quantity | min / q25 / median / q75 / max |
| --- | --- |
| log selection correction | -0.887722 / -0.0410798 / 3.03816e-10 / 0.0527341 / 4.393 |
| joint log-prior change | -1.7053e-13 / -2.84217e-14 / 0 / 2.84217e-14 / 1.42109e-13 |
| selected-label probability A | 0.00123148 / 0.562733 / 0.965678 / 0.999999 / 1 |
| selected-label probability B | 0.0416255 / 0.530232 / 0.934348 / 0.999998 / 1 |
| selected-label frequency rank A | 1 / 1 / 1 / 1 / 4 |
| selected-label frequency rank B | 1 / 1 / 1 / 1 / 5 |

Maximum absolute joint log-prior change is **1.7053e-13**. Every evaluated residual was retained. Nonfinite proposed priors: **0**; nonfinite proposed likelihoods: **0**. All current states and normalized selector logs were finite. Every proposal’s selected probabilities, ranks, frequencies, full forward/reverse log-probability vectors, MH uniform and outcome are saved; no source_gain classification is used.

| pair | opportunities | joint accepted | acceptance | joint contour survivors |
| --- | --- | --- | --- | --- |
| 0,1 | 73 | 45 | 61.6438% | 45 |
| 0,2 | 73 | 29 | 39.7260% | 32 |
| 0,3 | 73 | 33 | 45.2055% | 33 |
| 0,4 | 73 | 35 | 47.9452% | 36 |
| 0,5 | 73 | 44 | 60.2740% | 45 |
| 0,6 | 73 | 35 | 47.9452% | 36 |
| 0,7 | 74 | 30 | 40.5405% | 31 |
| 1,2 | 73 | 25 | 34.2466% | 25 |
| 1,3 | 73 | 36 | 49.3151% | 37 |
| 1,4 | 73 | 40 | 54.7945% | 40 |
| 1,5 | 73 | 34 | 46.5753% | 36 |
| 1,6 | 74 | 43 | 58.1081% | 45 |
| 1,7 | 73 | 30 | 41.0959% | 30 |
| 2,3 | 73 | 34 | 46.5753% | 36 |
| 2,4 | 73 | 37 | 50.6849% | 38 |
| 2,5 | 74 | 35 | 47.2973% | 36 |
| 2,6 | 73 | 33 | 45.2055% | 35 |
| 2,7 | 73 | 33 | 45.2055% | 33 |
| 3,4 | 74 | 32 | 43.2432% | 32 |
| 3,5 | 73 | 34 | 46.5753% | 36 |
| 3,6 | 73 | 39 | 53.4247% | 39 |
| 3,7 | 73 | 36 | 49.3151% | 37 |
| 4,5 | 73 | 34 | 46.5753% | 37 |
| 4,6 | 73 | 35 | 47.9452% | 37 |
| 4,7 | 73 | 41 | 56.1644% | 42 |
| 5,6 | 73 | 34 | 46.5753% | 38 |
| 5,7 | 73 | 44 | 60.2740% | 44 |
| 6,7 | 73 | 42 | 57.5342% | 43 |

Frequency-weight-rank strata. “At least one” includes “both”; these displayed rows are not all mutually exclusive. Counts and outcomes are descriptive; no rank cutoff is selected.

| rank set | touch category | proposals | accepted | acceptance |
| --- | --- | --- | --- | --- |
| 1 | both | 1373 | 803 | 58.4851% |
| 1 | at_least_one | 1977 | 971 | 49.1148% |
| 1 | neither | 71 | 31 | 43.6620% |
| 1 | exactly_one | 604 | 168 | 27.8146% |
| 2 | both | 1973 | 985 | 49.9240% |
| 2 | at_least_one | 2047 | 1001 | 48.9008% |
| 2 | neither | 1 | 1 | 100.0000% |
| 2 | exactly_one | 74 | 16 | 21.6216% |

Inside the frozen physical critical interval before exchange (inclusive endpoints):

| selected sources inside | proposals | accepted | acceptance |
| --- | --- | --- | --- |
| 0 | 213 | 75 | 35.2113% |
| 1 | 940 | 403 | 42.8723% |
| 2 | 895 | 524 | 58.5475% |

Of accepted proposals, 75 have neither selected frequency inside the interval, 403 have exactly one inside and 524 have both inside. These are physical-frequency facts only, not claims that post-swap source_gain dominance is known.

## Cost and mechanical cost matching

Cost accounting uses the baseline-compatible slice proxy `num_steps + num_shrink`. Exchange proposal evaluations and direct cache checks are counted calls, not proxies. Every direct cache check evaluates both prior and likelihood. Eight initial checks are included; each sweep checks eight post-slice and eight post-exchange states. The independent updates and disjoint pair results are validated in batches before any subsequent interaction, so every intermediate/current state receives its check.

| cost component | H isotropic | U population | O Hybrid P (saved) |
| --- | --- | --- | --- |
| slice likelihood proxy | 27,342 | 27,152 | 27,222 |
| exchange/refresh proposal likelihood evaluations | 0 | 4,096 | 4,096 |
| direct likelihood cache checks, including initial | 4,104 | 8,200 | 8,200 |
| total likelihood-cost proxy | 31,446 | 39,448 | 39,518 |
| slice prior proxy | 27,342 | 27,152 | 27,222 |
| exchange/refresh prior evaluations | 0 | 4,096 | 4,096 |
| direct prior cache evaluations | 4,104 | 8,200 | 8,200 |
| total prior-cost proxy | 31,446 | 39,448 | 39,518 |

For U, 12,296 prior evaluations are counted outside slice internals (4,096 exchange + 8,200 cache), plus the 27,152 slice-prior proxy. The full U run costs 25.45% more than H including the required audits. Synchronized U loop wall time is **20.97 s**, plus **7.61 s** compilation; the loop includes per-step host checks and intermediate checkpoint saves but excludes final analysis. H has no comparable reliable wall time, so no wall-speedup claim is made. Saved O Hybrid P loop timing is 69.82 s, but implementation/batching differences prevent treating that as a controlled timing result.

Primary cost prefix: largest complete U population-sweep prefix whose cumulative slice proxy + exchange evaluations + direct checks, including eight initial checks, is at most H’s **31,446** total. It is **409 sweeps**, cost **31,412**; sweep 410 would cost **31,500** and is excluded. This choice uses only cumulative cost, never diagnostic outcomes. Supplemental proposal-only matching gives 449 sweeps at 27,333 against H’s 27,342; the next prefix would cost 27,390.

| comparison | logL SD | MAD | IQR | old ell10 fraction SD | median KS | freq mean distance | full set RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H all512 | 446.210 | 403.013 | 728.905 | 0.252070 | 0.332031 | 37.090 | 0.256308 |
| U all512 | 54.365 | 50.321 | 94.178 | 0.026136 | 0.062500 | 42.073 | 0.257219 |
| U cost-matched409 | 46.049 | 39.168 | 72.246 | 0.036894 | 0.072127 | 37.023 | 0.255611 |
| U proposal-only449 | 45.774 | 39.986 | 73.631 | 0.035733 | 0.069042 | 36.104 | 0.256728 |

The likelihood-agreement improvement survives equal-cost comparison: cost-matched U logL-mean SD is **46.05**, exceedance SD **0.03689**, and KS **0.07213**, versus H’s **446.21**, **0.25207**, **0.33203**. At equal cost, global frequency mean distance and full-set RMS are nearly unchanged (37.02 versus 37.09 bins; 0.25561 versus 0.25631). This corroborates the targeted benefit while retaining the whole-catalogue limitation. No scalar efficiency score is formed.

## Integrity, validation and stopping point

Direct checks after all 4,096 slice transitions and all 4,096 pair-member exchange outcomes verified finite current priors/likelihoods, strict ell9 membership and cache consistency, plus the eight starts. Maximum cached-prior and cached-likelihood errors are both **0**. Every exchange proposal and committed outcome was independently reconstructed from saved post-slice arrays, including selected donor copies, untouched coordinates, both directions, joint acceptance and exact two-state rejection restoration. Saved RNG draws, forward/reverse selector vectors, ranks, interval classifications, prior/selection ratios and MH decisions were replayed without model evaluations. Provenance and lineage permutations were independently replayed across all 512 sweeps. Structural failures: **0**.

All selector log probabilities were finite; the smallest was **-914.72905**, so ordinary float64 probabilities can underflow for distant labels. The runner checked positivity and normalization in extended precision at every forward/reverse evaluation while sampling and computing MH exclusively in log space. No epsilon clipping, replacement of finite log weights, or width tuning occurred.

**202 tests passed**: all 191 existing tests preserved, plus 11 cheap Stage-4U tests covering round-robin disjointness/completeness, stable log probabilities, Gumbel label draws, fixed selector constants/streams, block exchange, reverse probabilities, full joint MH ratio, joint commit/rejection, provenance, and deterministic cost prefixes. Pytest contains no real LISA trajectories. Protected model, prior/configuration, proposals, generic DNS/slice code, ladder, starts and baseline files retained their hashes.

Classification prioritizes the concordant, cost-robust improvements in logL agreement, diagnostic exceedance agreement and KS together with demonstrated physical transfers; it is capped at **B (modest overall)** because whole-catalogue geometry is not consistently improved. A single finite ensemble can agree through shared realizations without fully exploring the constrained target, and the eight interacting histories are dependent. The result justifies at most the stated **one controlled level-10 reconstruction next**, not a claim that its construction gate will pass.

This stage stops after the one fixed-contour benchmark and its saved-trace analysis. No level-10 reconstruction/retry, level 11, selector-width or pairing tuning, frequency changes, source_gain calls in the chain, prior/slice changes, production or evidence computation occurred.

## Code and artifacts

Code: [benchmark driver](../../examples/lisa_dns_stage4/frequency_population_benchmark.py), [saved-trace analyzer](../../examples/lisa_dns_stage4/analyze_frequency_population.py), [cheap tests](../../tests/experimental/test_dns_gb_frequency_population.py). Artifacts in `/tmp/lisa_dns_stage4u_population/`: `design.json`, `initial_states.npz`, `random_schedule.npz`, `trace.npz`, `exchanges.npz`, `provenance.json`, `report.json`, `comparison.json`, `exchange_rows.csv`, and `provenance_transfers.csv`. Logs: `/tmp/lisa_dns_stage4u.log`, `/tmp/lisa_dns_stage4u_analysis.log`, `/tmp/lisa_dns_stage4u_tests.log`. The local `/tmp` artifacts are not versioned. The driver refuses to overwrite/repeat the experiment; analysis can be rerun from saved files.
