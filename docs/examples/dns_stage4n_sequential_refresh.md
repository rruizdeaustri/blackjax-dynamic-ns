# Stage-4N: sequential standalone Logistic refresh benchmarks

**Neither standalone refresh kernel improves the primary cross-walker communication diagnostics in this 512-transition benchmark. Single refresh: D (worse). Pair refresh: D (worse).** Both improve median labelled frequency-coordinate ESS and make large physical moves cheaply, but walkers remain more separated in likelihood than under the saved isotropic continuation. These are finite-budget comparisons from fixed starts, not an asymptotic ranking of kernels.

## Frozen protocol and exact transition

The unchanged target is `pi_impl(u) * I(logL(u) > -110252.99476697217)`. The rejected level-10 threshold `-109401.11425363769` is diagnostic only. No levels were constructed or retried.

Two independent standalone benchmarks each used exactly eight walkers and 512 transitions per walker, with no discarded burn-in. Both copied the exact Stage-4H initial positions, cached priors and cached likelihoods. Bytes matched the saved starts and the preceding calibration trace endpoints. No favorable starts were selected. Stage-4H was read from disk, not rerun.

Single-source streams use seeds 860100–860107; pair-source streams use 860200–860207. Each walker key splits into independent proposal and MH-uniform keys; the proposal key splits again into uniform-category and direct JAX Logistic keys. A fixed bijection maps categories to nine labels or 36 unordered pairs. This schedule has no state input.

Every proposal refreshes the selected six or twelve coordinates of the **current** state. The actual scalar prior is evaluated separately from the stable analytic proposal density. The transition uses:

```text
log q(v) = sum[-softplus(v_k) - softplus(-v_k)]
log r = lp_impl(new) - lp_impl(current) + log q(current_B) - log q(new_B)
alpha = 0 on contour failure or negative-infinite proposed prior
alpha = exp(min(0, log r)) otherwise
accept iff the independent MH uniform < alpha
```

An acceptance advances position and both caches; rejection returns the entire current state unchanged. Nonfinite Logistic draws are recorded and rejected without resampling or model calls. Neither kernel includes slice moves or a mixture with the other refresh type.

## Cross-walker communication

All SDs use ddof=1 across eight walkers. KS values are descriptive distances across 28 walker pairs, without IID significance tests. Mean logL, median, MAD and IQR for each walker and every pairwise KS value are saved in comparison.json.

| Kernel | Window | Exceedance SD | LogL-mean SD | Pairwise KS median | Pairwise KS max |
|---|---|---:|---:|---:|---:|
| isotropic | first256 | 0.264798 | 439.037 | 0.410156 | 0.695312 |
| isotropic | second256 | 0.270649 | 559.401 | 0.433594 | 0.726562 |
| isotropic | all512 | 0.252070 | 446.210 | 0.337891 | 0.677734 |
| single | first256 | 0.344681 | 1141.235 | 0.703125 | 1.000000 |
| single | second256 | 0.338710 | 1133.546 | 0.609375 | 1.000000 |
| single | all512 | 0.339276 | 1136.648 | 0.652344 | 1.000000 |
| pair | first256 | 0.330508 | 1127.876 | 0.597656 | 1.000000 |
| pair | second256 | 0.339834 | 1150.622 | 0.531250 | 1.000000 |
| pair | all512 | 0.333667 | 1138.871 | 0.544922 | 1.000000 |

The worsening is present in both halves, not just a poor best/worst-walker comparison. Walker 5 stays above ell10 at every retained state in both refresh benchmarks, whereas several other walkers almost never exceed it. Its all-512 mean logL is about −106694 (single) or −106681 (pair); most other refresh walkers remain near −109600 to −110050. This is persistent cross-walker separation, even though within-walker summaries can be relatively stable. No named catalogue families are inferred.

### Per-walker first256

| Walker | Isotropic exceedance | Isotropic mean logL | Single exceedance | Single mean logL | Pair exceedance | Pair mean logL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.441406 | -109388.161 | 0.140625 | -109623.848 | 0.164062 | -109619.913 |
| 1 | 0.554688 | -109332.351 | 0.000000 | -110032.540 | 0.015625 | -109995.300 |
| 2 | 0.652344 | -109074.875 | 0.148438 | -109643.633 | 0.140625 | -109635.520 |
| 3 | 0.023438 | -109897.264 | 0.000000 | -109890.926 | 0.066406 | -109856.490 |
| 4 | 0.148438 | -109794.438 | 0.000000 | -109996.565 | 0.101562 | -109911.153 |
| 5 | 0.691406 | -108632.630 | 1.000000 | -106713.270 | 1.000000 | -106699.650 |
| 6 | 0.375000 | -109527.296 | 0.007812 | -110073.592 | 0.015625 | -110018.812 |
| 7 | 0.054688 | -109895.590 | 0.000000 | -110050.599 | 0.042969 | -109979.052 |
### Per-walker second256

| Walker | Isotropic exceedance | Isotropic mean logL | Single exceedance | Single mean logL | Pair exceedance | Pair mean logL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.246094 | -109688.027 | 0.363281 | -109491.301 | 0.312500 | -109513.237 |
| 1 | 0.582031 | -108401.521 | 0.105469 | -109906.441 | 0.011719 | -109992.412 |
| 2 | 0.707031 | -108972.706 | 0.296875 | -109537.765 | 0.093750 | -109730.025 |
| 3 | 0.015625 | -109986.700 | 0.003906 | -109950.661 | 0.062500 | -109904.518 |
| 4 | 0.449219 | -109161.683 | 0.031250 | -110047.268 | 0.035156 | -109932.573 |
| 5 | 0.585938 | -108980.737 | 1.000000 | -106674.341 | 1.000000 | -106661.842 |
| 6 | 0.078125 | -109788.525 | 0.054688 | -109938.948 | 0.003906 | -110082.734 |
| 7 | 0.085938 | -109864.805 | 0.003906 | -109927.465 | 0.042969 | -109981.212 |
### Per-walker all512

| Walker | Isotropic exceedance | Isotropic mean logL | Single exceedance | Single mean logL | Pair exceedance | Pair mean logL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.343750 | -109538.094 | 0.251953 | -109557.575 | 0.238281 | -109566.575 |
| 1 | 0.568359 | -108866.936 | 0.052734 | -109969.491 | 0.013672 | -109993.856 |
| 2 | 0.679688 | -109023.790 | 0.222656 | -109590.699 | 0.117188 | -109682.773 |
| 3 | 0.019531 | -109941.982 | 0.001953 | -109920.793 | 0.064453 | -109880.504 |
| 4 | 0.298828 | -109478.060 | 0.015625 | -110021.917 | 0.068359 | -109921.863 |
| 5 | 0.638672 | -108806.683 | 1.000000 | -106693.806 | 1.000000 | -106680.746 |
| 6 | 0.226562 | -109657.911 | 0.031250 | -110006.270 | 0.009766 | -110050.773 |
| 7 | 0.070312 | -109880.197 | 0.001953 | -109989.032 | 0.042969 | -109980.132 |

First-half versus second-half logL KS distances by walker:

| Kernel | Walkers 0–7 |
|---|---|
| isotropic | 0.332, 0.453, 0.168, 0.230, 0.477, 0.238, 0.328, 0.113 |
| single | 0.305, 0.168, 0.164, 0.215, 0.223, 0.133, 0.211, 0.289 |
| pair | 0.246, 0.062, 0.258, 0.137, 0.137, 0.160, 0.176, 0.062 |

## Autocorrelation and persistent blocks

The same short-chain initial-positive monotone autocorrelation estimator is used for all traces. Labelled f0_u ESS summarizes 72 walker/coordinate series. Constants receive ESS zero. LogL ESS summarizes eight series. These estimates do not establish stationarity or physical convergence.

| Kernel | Window | f0_u ESS median | f0_u ESS minimum | LogL ESS median | LogL ESS minimum |
|---|---|---:|---:|---:|---:|
| isotropic | first256 | 5.386 | 2.806 | 33.528 | 7.869 |
| isotropic | second256 | 5.248 | 2.810 | 29.681 | 8.556 |
| isotropic | all512 | 6.923 | 2.895 | 33.529 | 10.723 |
| single | first256 | 16.063 | 0.000 | 22.538 | 12.310 |
| single | second256 | 16.503 | 0.000 | 18.029 | 15.276 |
| single | all512 | 28.039 | 0.000 | 32.636 | 26.545 |
| pair | first256 | 27.521 | 0.000 | 32.937 | 19.314 |
| pair | second256 | 29.533 | 0.000 | 38.854 | 21.678 |
| pair | all512 | 57.150 | 0.000 | 58.311 | 42.464 |

**Each walker retained one entire starting source block for all 512 transitions, under both kernels.** Every one of these labels was proposed, but none of those proposals was accepted. All other blocks were refreshed repeatedly. This is direct evidence that frequent nonlocal changes to the other sources did not replace every constrained part of the configuration. It does not identify which sources are physically important without additional model diagnostics.

| Walker | Never-refreshed label (both kernels) | Single attempts / accepted | Pair proposals including label / accepted |
|---|---:|---:|---:|
| 0 | 7 | 53 / 0 | 121 / 0 |
| 1 | 3 | 49 / 0 | 112 / 0 |
| 2 | 3 | 64 / 0 | 109 / 0 |
| 3 | 0 | 58 / 0 | 101 / 0 |
| 4 | 8 | 64 / 0 | 113 / 0 |
| 5 | 2 | 58 / 0 | 123 / 0 |
| 6 | 7 | 59 / 0 | 117 / 0 |
| 7 | 4 | 41 / 0 | 107 / 0 |

Unique-state fractions count distinct retained positions divided by window length, excluding the initial state. Rejections explain the repeated states. Acceptance lag-1 autocorrelation and ESS are descriptive binary-series diagnostics, saved for all windows.

| Walker | Single all-512 unique fraction | Pair all-512 unique fraction | Single acceptance lag-1 | Pair acceptance lag-1 |
|---|---:|---:|---:|---:|
| 0 | 0.898438 | 0.765625 | 0.0128 | 0.0713 |
| 1 | 0.802734 | 0.667969 | 0.1608 | 0.1742 |
| 2 | 0.875000 | 0.787109 | -0.0539 | 0.0554 |
| 3 | 0.867188 | 0.771484 | 0.0184 | 0.0245 |
| 4 | 0.767578 | 0.671875 | 0.1918 | 0.0865 |
| 5 | 0.886719 | 0.761719 | 0.0298 | 0.0068 |
| 6 | 0.660156 | 0.480469 | 0.1488 | 0.0903 |
| 7 | 0.826172 | 0.671875 | 0.0067 | 0.0363 |

All isotropic unique-state fractions are 1. First/second-half unique fractions, all 72 coordinate ESS values, and per-walker logL/acceptance autocorrelation estimates are retained in comparison.json.

## Permutation-invariant physical exploration

Frequency-set distances use sorted catalogues, equivalent to Hungarian matching in one-dimensional squared frequency distance, with a Fourier bin of 1/31536000 Hz. Full-coordinate matching reuses Stage-4I equal normalized coordinate weights and circular psi/lambda differences. Full window metrics use every 32nd endpoint (eight endpoints per half); frequency window RMS uses all cross-time pairs exactly.

| Kernel | Window | Median cross-walker frequency RMS (bins) | Median sorted-frequency mean distance (bins) | Median full-coordinate assignment RMS |
|---|---|---:|---:|---:|
| isotropic | first256 | 94.928 | 45.055 | 0.249369 |
| isotropic | second256 | 104.066 | 48.034 | 0.259783 |
| isotropic | all512 | 109.341 | 38.564 | 0.256396 |
| single | first256 | 104.265 | 25.813 | 0.259344 |
| single | second256 | 102.234 | 31.000 | 0.258018 |
| single | all512 | 104.261 | 21.434 | 0.258437 |
| pair | first256 | 109.066 | 16.987 | 0.258878 |
| pair | second256 | 106.946 | 15.224 | 0.256388 |
| pair | all512 | 106.487 | 12.333 | 0.257445 |

The pairwise tables for all 28 walker pairs in both halves, including the final 256-state window, are in comparison.json. The changes relative to baseline are:

| Kernel | Window | Change in frequency RMS median | Change in sorted-mean distance median | Change in full-coordinate RMS median |
|---|---|---:|---:|---:|
| single | first256 | +9.337 | -19.242 | +0.009975 |
| single | second256 | -1.832 | -17.034 | -0.001765 |
| single | all512 | -5.081 | -17.130 | +0.002041 |
| pair | first256 | +14.138 | -28.068 | +0.009509 |
| pair | second256 | +2.880 | -32.810 | -0.003395 |
| pair | all512 | -2.854 | -26.231 | +0.001050 |

Sorted-frequency mean agreement improves, especially for pairs, but the broader cross-time and full-coordinate distances remain similar. Cross-time RMS includes within-walker dispersion, so a larger value alone does not imply worse mixing. Neither frequency-mean agreement nor label ESS resolves the strong likelihood disagreement.

| Kernel | Walker | Within first half RMS (bins) | Within second half RMS (bins) | Between-half sorted mean shift (bins) | Start-to-end set distance (bins) | Maximum set distance from start (bins) |
|---|---:|---:|---:|---:|---:|---:|
| isotropic | 0 | 82.166 | 79.266 | 28.995 | 85.808 | 140.261 |
| isotropic | 1 | 81.958 | 84.709 | 56.860 | 106.705 | 177.727 |
| isotropic | 2 | 78.603 | 79.837 | 16.107 | 123.167 | 155.652 |
| isotropic | 3 | 82.568 | 117.103 | 31.521 | 123.261 | 180.507 |
| isotropic | 4 | 95.262 | 80.640 | 68.203 | 172.797 | 194.037 |
| isotropic | 5 | 97.691 | 99.980 | 121.095 | 100.958 | 175.719 |
| isotropic | 6 | 66.546 | 74.605 | 25.776 | 105.564 | 118.338 |
| isotropic | 7 | 78.967 | 90.377 | 120.182 | 162.591 | 273.189 |
| single | 0 | 91.688 | 100.496 | 36.505 | 49.826 | 172.623 |
| single | 1 | 95.198 | 89.537 | 22.071 | 50.653 | 178.095 |
| single | 2 | 100.466 | 92.855 | 18.631 | 140.768 | 191.717 |
| single | 3 | 103.047 | 98.778 | 46.960 | 79.491 | 203.085 |
| single | 4 | 107.208 | 99.089 | 14.883 | 27.069 | 210.434 |
| single | 5 | 120.534 | 101.481 | 36.793 | 78.277 | 266.582 |
| single | 6 | 101.884 | 99.480 | 21.891 | 114.293 | 159.173 |
| single | 7 | 98.219 | 92.880 | 28.961 | 99.425 | 255.181 |
| pair | 0 | 111.314 | 112.615 | 14.337 | 72.112 | 209.415 |
| pair | 1 | 99.215 | 114.570 | 14.537 | 104.705 | 199.839 |
| pair | 2 | 109.134 | 105.346 | 12.102 | 90.847 | 192.298 |
| pair | 3 | 109.467 | 98.418 | 12.035 | 87.248 | 207.774 |
| pair | 4 | 109.017 | 101.268 | 16.448 | 97.542 | 222.424 |
| pair | 5 | 104.523 | 91.256 | 14.154 | 60.347 | 202.372 |
| pair | 6 | 96.017 | 98.958 | 16.845 | 62.093 | 187.891 |
| pair | 7 | 111.009 | 108.758 | 11.775 | 114.237 | 247.426 |

Both refresh chains visit substantially different source sets over time, with maximum distances from their starts of roughly 159–267 bins (single) and 188–247 bins (pair). This is real physical-set exploration, alongside the persistent labelled blocks and likelihood separation.

## Acceptance and correction diagnostics

| Kernel | Accepted / 4096 | Acceptance | Mean alpha | Contour-failure fraction | MH rejections among survivors |
|---|---:|---:|---:|---:|---:|
| single | 3369 | 0.822509766 | 0.822509765624987 | 0.177490234 | 0 / 3369 |
| pair | 2854 | 0.696777344 | 0.696777343749988 | 0.303222656 | 0 / 2854 |

No binomial uncertainty interval is attached to these sequential-chain fractions. Every contour survivor happened to pass its independent MH uniform; the exact correction was still calculated. There were zero proposed negative-infinite priors, zero nonfinite draws or draw coordinates, zero nonfinite proposed likelihoods, and zero correction magnitudes greater than the predeclared materiality threshold of 1e-8. This does not erase the Stage-4L tail mismatch.

| Kernel | log r min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| single | -3.3377745e-12 | -5.6843419e-14 | 0 | 5.5067062e-14 | 3.3715253e-12 |
| pair | -5.8975047e-13 | -6.3948846e-14 | 0 | 6.750156e-14 | 2.4549252e-12 |

| Walker | Single acceptance | Pair acceptance |
|---|---:|---:|
| 0 | 0.896484 | 0.763672 |
| 1 | 0.802734 | 0.667969 |
| 2 | 0.875000 | 0.787109 |
| 3 | 0.867188 | 0.771484 |
| 4 | 0.767578 | 0.671875 |
| 5 | 0.886719 | 0.759766 |
| 6 | 0.660156 | 0.480469 |
| 7 | 0.824219 | 0.671875 |

All single categories (none suppressed):

| Label(s) | Proposals | Accepted | Acceptance |
|---|---:|---:|---:|
| [0] | 442 | 360 | 0.814480 |
| [1] | 447 | 408 | 0.912752 |
| [2] | 448 | 349 | 0.779018 |
| [3] | 428 | 295 | 0.689252 |
| [4] | 470 | 394 | 0.838298 |
| [5] | 468 | 446 | 0.952991 |
| [6] | 469 | 440 | 0.938166 |
| [7] | 454 | 320 | 0.704846 |
| [8] | 470 | 357 | 0.759574 |

All pair categories (none suppressed):

| Label(s) | Proposals | Accepted | Acceptance |
|---|---:|---:|---:|
| [0, 1] | 112 | 92 | 0.821429 |
| [0, 2] | 100 | 61 | 0.610000 |
| [0, 3] | 114 | 63 | 0.552632 |
| [0, 4] | 128 | 83 | 0.648438 |
| [0, 5] | 122 | 98 | 0.803279 |
| [0, 6] | 115 | 89 | 0.773913 |
| [0, 7] | 127 | 72 | 0.566929 |
| [0, 8] | 135 | 83 | 0.614815 |
| [1, 2] | 110 | 91 | 0.827273 |
| [1, 3] | 98 | 59 | 0.602041 |
| [1, 4] | 109 | 76 | 0.697248 |
| [1, 5] | 108 | 96 | 0.888889 |
| [1, 6] | 113 | 103 | 0.911504 |
| [1, 7] | 117 | 82 | 0.700855 |
| [1, 8] | 105 | 82 | 0.780952 |
| [2, 3] | 112 | 61 | 0.544643 |
| [2, 4] | 126 | 86 | 0.682540 |
| [2, 5] | 128 | 104 | 0.812500 |
| [2, 6] | 107 | 76 | 0.710280 |
| [2, 7] | 100 | 58 | 0.580000 |
| [2, 8] | 108 | 77 | 0.712963 |
| [3, 4] | 117 | 74 | 0.632479 |
| [3, 5] | 99 | 69 | 0.696970 |
| [3, 6] | 113 | 73 | 0.646018 |
| [3, 7] | 124 | 55 | 0.443548 |
| [3, 8] | 112 | 64 | 0.571429 |
| [4, 5] | 108 | 82 | 0.759259 |
| [4, 6] | 111 | 83 | 0.747748 |
| [4, 7] | 115 | 75 | 0.652174 |
| [4, 8] | 100 | 62 | 0.620000 |
| [5, 6] | 107 | 98 | 0.915888 |
| [5, 7] | 117 | 76 | 0.649573 |
| [5, 8] | 126 | 94 | 0.746032 |
| [6, 7] | 128 | 95 | 0.742188 |
| [6, 8] | 118 | 95 | 0.805085 |
| [7, 8] | 107 | 67 | 0.626168 |

## Accepted-move geometry

Physical f0 displacement is RMS over the selected/changed labelled sources. The permutation-invariant frequency RMS is over all nine sources after matching. All quantities below are conditional on accepted moves; the zero-motion rejections remain in the chain diagnostics.

| Kernel / geometry | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| isotropic / latent_l2 | 6.15501e-06 | 0.00273492 | 0.402941 | 3.95945 | 9.77257 |
| isotropic / physical_f0_rms_bins | 0.000250639 | 0.0609322 | 6.58854 | 149.429 | 562.562 |
| isotropic / source_set_frequency_rms_bins | 0.000118152 | 0.0364804 | 2.72815 | 41.8761 | 112.23 |
| isotropic / source_set_full_rms | 1.4698e-07 | 8.3517e-05 | 0.0112075 | 0.0959815 | 0.172562 |
| single / latent_l2 | 0.975851 | 3.11801 | 5.81176 | 9.43676 | 15.7099 |
| single / physical_f0_rms_bins | 0.049284 | 16.2317 | 177.995 | 480.971 | 625.095 |
| single / source_set_frequency_rms_bins | 0.016428 | 5.14918 | 39.7529 | 79.3948 | 118.709 |
| single / source_set_full_rms | 0.0322145 | 0.0709078 | 0.115467 | 0.152632 | 0.176566 |
| pair / latent_l2 | 2.33144 | 5.59776 | 8.52921 | 12.1246 | 20.5621 |
| pair / physical_f0_rms_bins | 7.2056 | 64.6159 | 231.541 | 423.48 | 590.692 |
| pair / source_set_frequency_rms_bins | 0.414184 | 18.7936 | 56.9433 | 108.5 | 170.783 |
| pair / source_set_full_rms | 0.0675964 | 0.112954 | 0.153863 | 0.189109 | 0.237482 |

The corresponding isotropic single/pair component median frequency-set steps are 5.624 and 5.198 bins (the pooled baseline, including global moves, is 2.728 bins). Sequential refresh medians of 39.75 and 56.94 bins remain strongly nonlocal. The improvement in step size does not translate into the required cross-walker likelihood communication.

## Cost-normalized comparison

| Kernel | Proposal likelihood calls or proxy | Direct initial + per-step cache likelihood checks | Total actual likelihood calls | Total actual prior calls | Measured loop seconds |
|---|---:|---:|---:|---:|---:|
| Isotropic | 27342 (proxy) | 4104 | Not recorded | Not recorded | Not comparable / not saved |
| single | 4096 actual | 4104 | 8200 | 8200 | 16.832 |
| pair | 4096 actual | 4104 | 8200 | 8200 | 16.988 |

Each refresh uses one likelihood and one prior evaluation per transition: 4096 proposal evaluations, plus eight initial checks and 4096 per-transition direct cache checks, for 8200 calls of each scalar function. The baseline proxy is `num_steps + num_shrink`; its uninstrumented actual proposal prior/likelihood totals cannot be reconstructed exactly. Proposal-cost comparison is approximately 6.68 times fewer evaluations for refresh. Including audit checks gives 31446 baseline proxy-plus-checks versus 8200 actual refresh likelihood calls (about 3.83 times). These are evaluation comparisons, not measured speedups.

New runs used CPU float64. Reported loop timings exclude initial compilation/start validation, schedule generation and NPZ writes, and include per-step integrity checks; Stage-4H used GPU. No wall-clock speedup is claimed.

Per-walker median ESS divided by average per-walker proposal likelihood cost, expressed per 1000 evaluations (baseline denominator is a proxy):

| Kernel | Window | Median f0_u ESS / 1000 | Median logL ESS / 1000 |
|---|---|---:|---:|
| isotropic | first256 | 3.204 | 19.944 |
| isotropic | second256 | 3.022 | 17.091 |
| isotropic | all512 | 2.026 | 9.810 |
| single | first256 | 62.748 | 88.040 |
| single | second256 | 64.463 | 70.426 |
| single | all512 | 54.764 | 63.742 |
| pair | first256 | 107.505 | 128.661 |
| pair | second256 | 115.363 | 151.773 |
| pair | all512 | 111.620 | 113.889 |

As a saved-data cost comparison, the largest common isotropic prefix within 4096 proposal-cost units is 78 steps per walker, costing 4031 proxy evaluations. Its exceedance SD is 0.330273, logL-mean SD 627.811, and median pairwise KS 0.551282. At 4096 actual proposal evaluations, single/pair have exceedance SDs 0.339276/0.333667 and logL-mean SDs 1136.648/1138.871. Thus lower cost buys better within-coordinate sampling, but no demonstrated cross-walker likelihood agreement advantage. This prefix comparison excludes audit overhead and uses short, unequal trajectory lengths; it is descriptive, not a convergence test or a single efficiency score.

## Integrity, tests and decision

Every retained current state, including after rejection, was checked directly for finite coordinates, finite prior and likelihood, strict ell9 membership, and cache consistency. Maximum cache differences were 4.27e-14 in prior and 3.48e-8 in likelihood, within the inherited 1e-9/1e-7 tolerances. Saved starts matched exactly before recomputation; CPU/GPU differences were not used to replace their caches. Untouched coordinates matched byte for byte and rejected states retained their exact caches. **Zero structural failures occurred.**

A separate saved-array audit reconstructs each transition from its preceding accepted/rejected state, checks selected draws and untouched coordinates, recomputes q, log r and alpha, checks each MH uniform decision, and verifies that every recorded position/cache follows that decision. Model, configuration, accepted checkpoint, Stage-4H traces, isotropic kernel and DNS code hashes remained unchanged.

**All 133 existing tests plus eight new tests passed (141 total).** New cheap tests cover single/pair ratio antisymmetry with a non-Logistic coupled target, accepted caches, contour and MH rejection preservation, sequential advancement rather than fixed-base reuse, untouched coordinates, independent selection/uniform streams, and nonfinite/zero-density rejection. No real LISA likelihood probes are in pytest.

**Single refresh: D — worse. Pair refresh: D — worse.** This classification prioritizes cross-walker agreement: logL spread, exceedance spread and KS disagreement are worse in both halves and overall. The pair kernel has better labelled-coordinate ESS and logL ESS, both kernels are cheap, and both explore different physical sets. Those benefits do not overcome persistent likelihood separation and the never-refreshed blocks in the requested standalone communication test. Neither baseline nor refresh chains establish global equilibrium.

Artifacts: `/tmp/lisa_dns_stage4n_sequential/report.json`, `comparison.json`, exact `initial_states.npz`, and `single/trace.npz` / `pair/trace.npz`. Traces include every retained position/cache, proposal position/cache, label(s), Logistic draw, MH uniform, proposal log densities, log ratio, alpha, and decision. Runner: `examples/lisa_dns_stage4/sequential_logistic_refresh.py`; read-only comparison: `examples/lisa_dns_stage4/compare_sequential_refresh.py`.

No chains were extended, no combined kernel was built, and no DNS integration, prior/kernel modification, tuning, production, or evidence work occurred. **Level 10 remains rejected.**
