# Stage-4P: one controlled Hybrid-P level-10 construction

**Level 10 remains rejected.** The attempt failed the independent calibration ESS gate. The ladder remains frozen through level 9. The candidate was `-109301.45870884096`, with selection ESS 25.327148 and independent calibration ESS 18.274643. No retry or budget change occurred.

## Preserved partial work and exact historical initialization

The resumed session first ran `git status --short --untracked-files=all` and `git diff --check`, then inspected both existing Stage-4P files. The construction runner and eight cheap tests were already complete and were preserved without modification. There was no Stage-4P documentation or sampling-output directory. The read-only comparison and this report were added; no completed construction was restarted.

Before sampling, all 148 existing tests plus eight Stage-4P tests passed (156 total). A separate read-only recovery check passed, followed by the runner’s own preflight. The immutable Stage-4G checkpoint digest and historical runner source hash verified. Checkpoint promotion indices reproduced both starting banks from each bank’s own saved level-9 retained traces. Positions, log priors and log likelihoods matched byte for byte, including an independent replay of the historical promotion RNG. The same eight selection and eight calibration starts were used as in the failed isotropic attempt.

Frozen input: `ell9 = -110252.99476697217`, `log X9 = -7.67544001580533`. Levels 0–9 were loaded from `/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json`, copied byte for byte to the Stage-4P output, and checked numerically/bytewise throughout. No lower level was rebuilt or recalibrated.

| Bank | Historical promotion indices within the 256 retained level-9 states | Fixed Hybrid-P seeds |
|---|---|---|
| selection | [133, 209, 107, 226, 248, 45, 58, 181] | [880100, 880101, 880102, 880103, 880104, 880105, 880106, 880107] |
| calibration | [168, 137, 95, 193, 219, 173, 144, 142] | [880200, 880201, 880202, 880203, 880204, 880205, 880206, 880207] |

The banks have distinct starts, separate genealogies and disjoint random streams. No current/retained states were shared between the new banks. Independence refers to bank construction and streams; finite MCMC burn-in does not guarantee equilibrium.

## Unchanged construction protocol

Each bank has eight walkers, each with 128 burn-in sweeps and 256 retained sweeps: 3072 sweeps per bank, 6144 total. A sweep is exactly the validated Stage-4O isotropic slice transition followed by the unchanged Stage-4N exact pair refresh. Scales are pi/sqrt(3); the slice mixture is 20% global, 40% labelled single, 40% unordered labelled pair; max_steps=10 and max_shrinkage=100. Pair refresh selects uniformly among 36 unordered pairs, uses direct Logistic draws and the full MH correction against the unchanged implemented prior. Both constituents remain constrained strictly above ell9.

Only the selection bank chooses the new threshold. With N=2048 retained selection observations and rho=exp(-1), the existing rule selects sorted observation floor(N*(1-rho))=1294 (zero-based) and excludes ties with strict `>`. The selection candidate was written to `selection_candidate.json` before any calibration walker was run. The old isotropic threshold was never imposed on Hybrid P.

The unchanged `assess`, `build_next_level`, `tail_diagnostics`, and uncertainty machinery determine the result. Block size is 32; min_ess is 20. Conservative SE is max(IID SE, block-means SE, between-walker SE), and ESS is the Bernoulli-variance equivalent capped at N. The separately reported within-walker block SE is not substituted for the conservative error. The existing selection-status gate, nondegenerate calibration fraction and integrity requirements remain in force. No probability, seed or budget was adjusted after observing results.

## Direct comparison with the failed isotropic construction

| Attempt | Independently selected candidate threshold | Selection compression | Selection ESS | Calibration compression | Calibration ESS | Frozen? |
|---|---:|---:|---:|---:|---:|---|
| Historical isotropic | -109401.11425363769 | 0.367675781 | 24.272357 | 0.364257812 | 16.798215 | no |
| Hybrid P | -109301.45870884096 | 0.367675781 | 25.327148 | 0.352539062 | 18.274643 | no |

Both attempts use the same starts, contour and budget. They select different thresholds, so the following exceedance diagnostics each refer to that attempt’s own selection-only candidate. Original isotropic results were recomputed from saved traces and matched the historical report; no old trajectory was rerun.

| Attempt / bank | IID SE | Within-walker block SE | Block-means SE | Between-walker SE | Conservative SE | Approximate correlated 95% interval |
|---|---:|---:|---:|---:|---:|---|
| Isotropic / selection | 0.010654607 | 0.024402201 | 0.039919555 | 0.097869292 | 0.097869292 | [0.136252, 0.599100] |
| Isotropic / calibration | 0.010633592 | 0.022343870 | 0.044446751 | 0.117412243 | 0.117412243 | [0.086622, 0.641894] |
| Hybrid P / selection | 0.010654607 | 0.022910599 | 0.038555377 | 0.095809655 | 0.095809655 | [0.141122, 0.594230] |
| Hybrid P / calibration | 0.010557119 | 0.019580017 | 0.041576287 | 0.111759888 | 0.111759888 | [0.088269, 0.616809] |

Intervals and ESS are the existing approximate correlation-aware diagnostics, not proof of complete target coverage. Selection diagnostics are not an independent mass calibration.

| Attempt / bank | Walker exceedance range | Exceedance SD | Walker logL-mean range | LogL-mean SD | Median f0_u ESS | Minimum f0_u ESS |
|---|---|---:|---|---:|---:|---:|
| Isotropic / selection | [0.000000, 0.742188] | 0.276816 | [-110138.387, -108330.425] | 687.897 | 5.571 | 2.744 |
| Isotropic / calibration | [0.011719, 0.816406] | 0.332092 | [-110036.445, -108516.858] | 622.792 | 5.847 | 2.854 |
| Hybrid P / selection | [0.011719, 0.722656] | 0.270991 | [-110029.946, -107786.637] | 754.496 | 28.709 | 2.935 |
| Hybrid P / calibration | [0.019531, 0.773438] | 0.316105 | [-110081.410, -108405.030] | 641.988 | 31.804 | 3.813 |

Walker SDs use ddof=1. Frequency-coordinate ESS uses the unchanged short-chain autocorrelation estimator across 72 walker/coordinate series; coordinate ESS alone is not physical convergence.

### Per-walker selection diagnostics

| Walker | Isotropic exceedance | Hybrid-P exceedance | Isotropic mean logL | Hybrid-P mean logL |
|---|---:|---:|---:|---:|
| 0 | 0.246094 | 0.011719 | -109659.363 | -110029.946 |
| 1 | 0.742188 | 0.601562 | -108483.863 | -108692.179 |
| 2 | 0.000000 | 0.183594 | -110138.387 | -109730.217 |
| 3 | 0.656250 | 0.722656 | -108330.425 | -107786.637 |
| 4 | 0.210938 | 0.371094 | -109683.220 | -109521.133 |
| 5 | 0.062500 | 0.589844 | -109993.324 | -109048.437 |
| 6 | 0.468750 | 0.023438 | -109308.583 | -110010.623 |
| 7 | 0.554688 | 0.437500 | -108803.414 | -109298.706 |
### Per-walker calibration diagnostics

| Walker | Isotropic exceedance | Hybrid-P exceedance | Isotropic mean logL | Hybrid-P mean logL |
|---|---:|---:|---:|---:|
| 0 | 0.476562 | 0.773438 | -109036.076 | -108405.030 |
| 1 | 0.226562 | 0.406250 | -109695.134 | -109370.738 |
| 2 | 0.816406 | 0.207031 | -108516.858 | -109553.511 |
| 3 | 0.027344 | 0.058594 | -109995.105 | -109851.469 |
| 4 | 0.574219 | 0.687500 | -108883.418 | -108512.930 |
| 5 | 0.746094 | 0.640625 | -108777.251 | -109083.771 |
| 6 | 0.011719 | 0.019531 | -110036.445 | -110081.410 |
| 7 | 0.035156 | 0.027344 | -109959.455 | -109976.408 |

The between-walker calibration SE changed from 0.117412243 to 0.111759888; the corresponding exceedance SD changed from 0.332092 to 0.316105. Independent calibration ESS changed from 16.798215 to 18.274643. These measures, rather than coordinate ESS or refresh acceptance, determine whether the original calibration disagreement was reduced enough to pass the fixed gate.

## Source-block diagnostics (burn-in plus retained sweeps)

Calibration label mapping uses exact labelled genealogy: the failed isotropic level-10 endpoints are the saved Stage-4H starts, and no intervening algorithm relabelled sources. This identifies the historical labelled slot, not necessarily a physically unchanged source. Labels are [7,3,3,0,8,2,7,4]. Selection is an independent bank with no established Stage-4N/O sticky-source mapping; no sticky identity is assigned there. All nine selection labels are reported. These diagnostics are computed after sampling and never affect pair selection.

| Calibration walker | Historical label | Slice changes | Pair attempts | Pair accepts | Slice latent path L2 | Refresh latent path L2 | Slice f0 path (bins) | Net f0 change (bins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 146 | 91 | 0 | 4.91327 | 0.00000 | 138.612 | -2.610 |
| 1 | 3 | 121 | 79 | 53 | 33.14522 | 325.50696 | 1085.103 | -315.070 |
| 2 | 3 | 126 | 97 | 0 | 1.75892 | 0.00000 | 36.237 | 0.779 |
| 3 | 0 | 126 | 90 | 0 | 1.01504 | 0.00000 | 26.779 | 0.323 |
| 4 | 8 | 114 | 93 | 0 | 2.33072 | 0.00000 | 71.168 | -7.776 |
| 5 | 2 | 139 | 93 | 0 | 4.42391 | 0.00000 | 89.125 | 0.892 |
| 6 | 7 | 118 | 89 | 39 | 34.81677 | 206.52149 | 1056.540 | 43.677 |
| 7 | 4 | 123 | 102 | 0 | 1.58886 | 0.00000 | 32.000 | 0.630 |

The historical calibration sticky slot was successfully refreshed in 2 of eight walkers: [1, 6]. Stage-4O had one such walker. This is a descriptive comparison: the construction starts earlier in the same calibration genealogies and uses 384 rather than 512 sweeps, with independently fixed streams.

### All selection-bank labels (no sticky mapping)

| Walker | Label | Slice changes | Pair attempts | Pair accepts | Slice latent path L2 | Refresh latent path L2 | Slice f0 path (bins) | Net f0 change (bins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 105 | 90 | 55 | 38.1457 | 309.8833 | 1139.058 | 161.844 |
| 0 | 1 | 120 | 76 | 39 | 47.3435 | 228.7143 | 1773.963 | -440.391 |
| 0 | 2 | 112 | 70 | 41 | 50.9965 | 257.3755 | 1650.564 | -171.289 |
| 0 | 3 | 115 | 107 | 64 | 41.7055 | 379.9055 | 1655.689 | -242.952 |
| 0 | 4 | 110 | 87 | 61 | 61.9807 | 360.9104 | 2525.420 | 406.597 |
| 0 | 5 | 116 | 87 | 52 | 51.0386 | 313.6318 | 1783.211 | 78.053 |
| 0 | 6 | 113 | 77 | 0 | 0.7272 | 0.0000 | 20.915 | -1.052 |
| 0 | 7 | 103 | 84 | 46 | 41.6727 | 294.1997 | 998.581 | -256.687 |
| 0 | 8 | 112 | 90 | 56 | 50.9721 | 369.0377 | 2046.021 | 201.754 |
| 1 | 0 | 113 | 78 | 64 | 67.3907 | 355.6723 | 2639.892 | -35.116 |
| 1 | 1 | 132 | 93 | 74 | 57.6134 | 442.8495 | 1753.443 | 294.327 |
| 1 | 2 | 124 | 91 | 0 | 3.7580 | 0.0000 | 104.211 | 15.626 |
| 1 | 3 | 122 | 89 | 74 | 65.5714 | 427.1690 | 2807.739 | 150.439 |
| 1 | 4 | 125 | 70 | 56 | 67.2141 | 323.9362 | 1982.553 | -394.657 |
| 1 | 5 | 139 | 78 | 60 | 76.0097 | 339.3745 | 2318.052 | -103.900 |
| 1 | 6 | 117 | 93 | 81 | 58.7965 | 483.8480 | 2151.879 | -186.728 |
| 1 | 7 | 136 | 90 | 77 | 63.9837 | 457.0458 | 2472.396 | -37.922 |
| 1 | 8 | 125 | 86 | 70 | 51.4799 | 438.7347 | 2482.818 | 3.876 |
| 2 | 0 | 134 | 83 | 54 | 38.7880 | 345.9871 | 1179.088 | -46.843 |
| 2 | 1 | 134 | 85 | 43 | 48.3574 | 223.3789 | 1666.059 | -16.523 |
| 2 | 2 | 133 | 88 | 12 | 12.8957 | 72.3624 | 283.464 | 161.228 |
| 2 | 3 | 137 | 79 | 38 | 33.2348 | 217.2999 | 1647.750 | 16.638 |
| 2 | 4 | 144 | 70 | 22 | 26.0432 | 133.2448 | 1089.390 | -299.575 |
| 2 | 5 | 134 | 94 | 57 | 48.4526 | 307.7568 | 1494.809 | 282.160 |
| 2 | 6 | 133 | 90 | 20 | 17.0243 | 119.5247 | 424.999 | -338.245 |
| 2 | 7 | 145 | 88 | 43 | 66.3507 | 271.7083 | 2703.868 | -276.600 |
| 2 | 8 | 128 | 91 | 47 | 35.3196 | 268.5697 | 1390.415 | -202.405 |
| 3 | 0 | 134 | 82 | 70 | 66.5915 | 410.9341 | 2285.757 | -16.311 |
| 3 | 1 | 120 | 82 | 73 | 46.3888 | 407.2980 | 1448.927 | 429.285 |
| 3 | 2 | 128 | 77 | 67 | 63.9019 | 345.4303 | 2330.086 | 468.408 |
| 3 | 3 | 120 | 93 | 78 | 57.6946 | 458.7952 | 2255.798 | 385.778 |
| 3 | 4 | 125 | 86 | 75 | 66.2852 | 476.1916 | 1919.558 | 58.443 |
| 3 | 5 | 116 | 73 | 0 | 2.5370 | 0.0000 | 88.427 | -3.664 |
| 3 | 6 | 125 | 95 | 85 | 57.6464 | 529.1198 | 2325.768 | -160.649 |
| 3 | 7 | 127 | 77 | 66 | 61.4726 | 389.9100 | 2201.844 | -63.531 |
| 3 | 8 | 125 | 103 | 92 | 60.7582 | 560.6486 | 2758.529 | -504.717 |
| 4 | 0 | 142 | 78 | 70 | 54.5042 | 408.4431 | 1818.916 | 528.309 |
| 4 | 1 | 131 | 102 | 41 | 27.7876 | 226.5510 | 1081.996 | 106.317 |
| 4 | 2 | 143 | 84 | 71 | 60.6803 | 411.9957 | 2745.346 | 268.812 |
| 4 | 3 | 134 | 91 | 77 | 58.0061 | 445.0612 | 2027.051 | -27.743 |
| 4 | 4 | 144 | 80 | 68 | 63.1483 | 411.4044 | 2384.959 | -122.487 |
| 4 | 5 | 138 | 80 | 73 | 56.5534 | 437.7286 | 2302.678 | 148.646 |
| 4 | 6 | 138 | 72 | 34 | 42.2508 | 206.0585 | 1277.411 | -76.683 |
| 4 | 7 | 141 | 84 | 71 | 47.7787 | 410.7696 | 1837.473 | 234.267 |
| 4 | 8 | 144 | 97 | 79 | 50.5237 | 479.8601 | 1402.359 | 169.802 |
| 5 | 0 | 128 | 66 | 60 | 48.2950 | 377.2295 | 1976.957 | -136.627 |
| 5 | 1 | 130 | 77 | 59 | 66.8005 | 355.7970 | 2195.614 | 339.637 |
| 5 | 2 | 123 | 101 | 69 | 45.8987 | 425.3498 | 2315.385 | 41.944 |
| 5 | 3 | 124 | 82 | 59 | 59.3146 | 355.6179 | 2199.796 | -95.802 |
| 5 | 4 | 138 | 92 | 74 | 69.9839 | 439.0199 | 3188.108 | -175.667 |
| 5 | 5 | 134 | 94 | 59 | 36.2724 | 355.5966 | 1273.747 | 8.876 |
| 5 | 6 | 120 | 96 | 73 | 49.7237 | 461.6868 | 1298.854 | 304.934 |
| 5 | 7 | 125 | 82 | 19 | 21.4382 | 120.9459 | 705.234 | 85.826 |
| 5 | 8 | 116 | 78 | 62 | 46.5709 | 384.9579 | 1743.239 | -143.851 |
| 6 | 0 | 115 | 78 | 35 | 46.7087 | 231.9122 | 2029.562 | -214.933 |
| 6 | 1 | 131 | 73 | 30 | 34.3504 | 185.1291 | 1252.431 | 240.648 |
| 6 | 2 | 125 | 82 | 33 | 43.0287 | 201.4481 | 1327.869 | 295.713 |
| 6 | 3 | 128 | 85 | 34 | 38.2981 | 215.0981 | 1587.491 | -529.522 |
| 6 | 4 | 134 | 91 | 10 | 5.3082 | 53.6091 | 256.461 | 220.588 |
| 6 | 5 | 124 | 90 | 16 | 15.3342 | 92.3206 | 380.873 | -469.154 |
| 6 | 6 | 121 | 97 | 46 | 41.2118 | 281.1500 | 1555.580 | -11.409 |
| 6 | 7 | 126 | 87 | 0 | 1.3919 | 0.0000 | 37.029 | 5.437 |
| 6 | 8 | 122 | 85 | 26 | 22.0884 | 151.4762 | 983.969 | 233.379 |
| 7 | 0 | 123 | 83 | 63 | 56.1307 | 395.3838 | 2274.067 | 68.891 |
| 7 | 1 | 119 | 80 | 55 | 61.4123 | 305.3717 | 2091.977 | -146.148 |
| 7 | 2 | 125 | 110 | 83 | 44.0758 | 514.1375 | 1336.188 | -86.831 |
| 7 | 3 | 130 | 82 | 40 | 53.0693 | 233.0544 | 1137.389 | 58.502 |
| 7 | 4 | 130 | 82 | 56 | 57.8828 | 336.3120 | 1692.995 | 320.850 |
| 7 | 5 | 128 | 87 | 61 | 65.6100 | 338.7034 | 2678.454 | -272.285 |
| 7 | 6 | 127 | 73 | 0 | 4.4160 | 0.0000 | 101.194 | 28.896 |
| 7 | 7 | 127 | 79 | 51 | 50.5203 | 312.6771 | 2046.943 | 386.026 |
| 7 | 8 | 127 | 92 | 63 | 60.4139 | 360.2724 | 2368.909 | 135.648 |

One Fourier bin is 1/31536000 Hz. Path lengths sum constituent steps, while net displacement compares initial and final states. JSON also records all calibration labels, initial/final physical f0, net latent displacement, refresh f0 paths and first accepted-refresh sweep. Zero categories remain visible.

## Pair-refresh diagnostics

| Bank / window | Proposals | Accepted | Acceptance | Contour rejection | MH rejections among survivors |
|---|---:|---:|---:|---:|---|
| selection / all384 | 3072 | 1866 | 0.607421875 | 0.392578125 | 0 / 1866 |
| selection / burn128 | 1024 | 560 | 0.546875000 | 0.453125000 | 0 / 560 |
| selection / retained256 | 2048 | 1306 | 0.637695312 | 0.362304688 | 0 / 1306 |
| calibration / all384 | 3072 | 2090 | 0.680338542 | 0.319661458 | 0 / 2090 |
| calibration / burn128 | 1024 | 729 | 0.711914062 | 0.288085938 | 0 / 729 |
| calibration / retained256 | 2048 | 1361 | 0.664550781 | 0.335449219 | 0 / 1361 |

| Bank | Proposed prior = -inf | Nonfinite draws | Nonfinite coordinates | Material corrections (>1e-8) |
|---|---:|---:|---:|---:|
| selection | 0 | 0 | 0 | 0 |
| calibration | 0 | 0 | 0 | 0 |

The implemented prior was not replaced by analytic Logistic. The analytic density is used only for q. The full pre-truncation correction/log MH ratio distribution is:

| Bank | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| selection | -7.10542736e-13 | -6.03961325e-14 | 0 | 6.03961325e-14 | 2.67874611e-12 |
| calibration | -4.21707114e-12 | -6.03961325e-14 | 0 | 6.39488462e-14 | 4.206413e-12 |

Nonfinite draws are rejected without redraw; proposed negative-infinite prior means alpha zero. No clipping or prior repair is introduced. Per-walker refresh summaries and survivor-alpha quantiles are saved in comparison.json.

## Cost and integrity

| Attempt / bank | Slice likelihood/prior proxy | Refresh likelihood calls | Refresh prior calls | Direct checks per scalar, including starts | Total per-scalar proxy + counted calls |
|---|---:|---:|---:|---:|---:|
| Isotropic / selection | 21008 | 0 | 0 | 3080 | 24088 |
| Hybrid P / selection | 21104 | 3072 | 3072 | 6152 | 30328 |
| Isotropic / calibration | 20837 | 0 | 0 | 3080 | 23917 |
| Hybrid P / calibration | 20797 | 3072 | 3072 | 6152 | 30021 |

Slice cost retains the historical num_steps+num_shrink convention; both scalar functions are called in each slice evaluation. Direct initial/constituent cache checks and refresh calls are counted separately. No cost optimization or cost-matched truncation was used in this construction. New CPU loop timings (including per-step checks, excluding compilation/start checks/I/O) were selection: 54.23 s, calibration: 53.87 s. The historical GPU wall time is not a comparable speed benchmark.

Every current state was checked after both constituents for finite coordinates, finite implemented prior and likelihood, strict ell9 membership and direct cache consistency. The prior tolerance remained 1e-9 and likelihood tolerance 1e-7. Rejected refreshes return the exact post-slice state and caches; accepted caches correspond to the proposed position; unselected coordinates are unchanged. No sampling structural failures occurred.

The saved-data audit independently reconstructed all refresh q terms, MH ratios, alphas and uniform decisions; verified sequential post-slice state/cache contracts and slice block masks; reconciled costs; recomputed selection/calibration diagnostics with the original machinery; checked bank separation, initial states and all protected file hashes. The lower checkpoint is unchanged byte for byte.

**156 tests passed before the real construction: all 148 existing tests plus eight Stage-4P tests.** They cover exact prefix preservation, historical start/cache recovery, separate banks/streams, 128/256 bookkeeping, the ESS=20 boundary, failure leaving no new checkpoint, success freezing exactly one level, and rejection of any level-11 call. No real LISA trajectory is in pytest.

Artifacts: `/tmp/lisa_dns_stage4p_hybrid_level10/preflight.json`, `selection_candidate.json`, `report.json`, `comparison.json`, both initial-bank NPZs, both 384-sweep-per-walker trace NPZs, and `levels.npz`. The copied `checkpoint_level9.json` is the exact historical lower checkpoint.

No `checkpoint_level10.json` was created. The failed candidate and its diagnostic mass estimate are not frozen levels.

Exactly one attempt was made. No retry, level 11, prior or generic-DNS modification, probability tuning, production, or evidence implementation occurred. Evidence reconstruction remains unvalidated and out of scope.
