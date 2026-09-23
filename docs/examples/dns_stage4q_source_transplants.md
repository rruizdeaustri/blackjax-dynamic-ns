# Stage 4Q: fixed-base source transplantation
Classification: **MIXED overall, predominantly A (single-source transferable) for the six still-critical historical source blocks.** Cross-walker single transplants among those six walkers survive 424/480 (88.33%), compared with 89/96 (92.71%) temporal controls. Including the nearest-frequency companion gives 426/480 (88.75%). This does not support an obligatory local pair or generally incompatible catalogues. Some context sensitivity remains.

The full historical-slot experiment is heterogeneous: Stage 4P had already replaced the mapped sticky slots in walkers 1 and 6. Those slots now contain low-contribution sources. Transplanting them into the six still-critical slots fails 192/192 single tests; the reverse direction survives 192/192. We retained the prescribed historical labels and did not retarget donors to a newly identified source. This asymmetry must not be interpreted as evidence that the six remaining critical sources are mutually incompatible.

Frozen target: `ell_9 = -110252.99476697217`, `log X_9 = -7.67544001580533`. The failed isotropic (`-109401.11425363769`) and Stage-4P (`-109301.45870884096`) candidates are diagnostic only. No levels or Markov-chain transitions were constructed.

## Fixed design and provenance

Only the eight Stage-4P calibration traces were used. The mapping `[7, 3, 3, 0, 8, 2, 7, 4]` was recovered and verified against Stage-4P preflight/report, Stage-4O bottleneck audit, and both Stage-4N saved refresh traces. No selection-bank identities were assigned. Retained indices are `0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255`; temporal donors use `(index + 128) % 256`.

Each experiment contains 896 ordered cross-walker tests and 128 temporal controls. Every test starts from its original recipient. Single tests copy the historical sticky block exactly. Pair tests use the nearest other source in physical frequency (smallest label breaks a tie), then choose the donor-to-recipient assignment minimizing the sum of squared frequency differences; ties preserve sticky-to-sticky assignment. All assignments were written to `design.json` before likelihood evaluation. No likelihood-based assignment, source sorting, MH decision or sequential update occurs.

## Aggregate compatibility

| unit | group | ell9 survivors | fraction | median delta logL | delta logL IQR | old candidate fraction | Stage4P candidate fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | cross | 643/896 | 0.717634 | -11.62 | [-2295.97, 1270.06] | 0.466518 | 0.447545 |
| single | temporal_control | 115/128 | 0.898438 | -14.06 | [-237.10, 304.97] | 0.468750 | 0.421875 |
| pair | cross | 647/896 | 0.722098 | -12.72 | [-1674.45, 1204.19] | 0.467634 | 0.449777 |
| pair | temporal_control | 110/128 | 0.859375 | 27.47 | [-344.19, 381.86] | 0.476562 | 0.445312 |

These are compatibility fractions, not acceptance rates. The deterministic, correlated trace sample does not justify independent-binomial uncertainty claims.

Historical groups below were defined from saved Stage-4P refresh history, already known before evaluation: persistent slots in walkers 0, 2, 3, 4, 5, 7; replaced slots in walkers 1, 6. These summaries reuse all prescribed tests without selecting favorable states or adding evaluations.

| group (donor → recipient) | single survives | pair survives |
| --- | --- | --- |
| persistent_to_persistent | 424/480 | 426/480 |
| persistent_temporal | 89/96 | 89/96 |
| replaced_to_persistent | 0/192 | 7/192 |
| persistent_to_replaced | 192/192 | 192/192 |
| replaced_to_replaced | 27/32 | 22/32 |
| replaced_temporal | 26/32 | 21/32 |

Of the 896 paired cross-walker tests, 630 survive both operations, 17 are rescued by the pair, 13 lose survival with the pair, and 236 fail both. Temporal controls have 108 joint successes, 2 rescues, 7 losses and 11 joint failures. The companion therefore gives little net compatibility benefit.

## Complete recipient/donor matrices

Rows are recipients, columns are donors, and diagonals are temporal controls. Each cell has 16 tests. No zero cells are omitted. Delta-logL values are rounded to one decimal here; machine-readable JSON and CSV retain full precision.

### Single-block transplantation

ell9 survival fraction

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 0.0000 | 0.9375 | 0.8125 | 0.9375 | 1.0000 | 0.0000 | 0.6250 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0000 | 0.8750 | 0.7500 | 0.8750 | 0.8750 | 0.0000 | 0.4375 |
| 3 | 1.0000 | 0.0000 | 1.0000 | 0.8125 | 1.0000 | 1.0000 | 0.0000 | 0.6875 |
| 4 | 1.0000 | 0.0000 | 1.0000 | 0.8750 | 1.0000 | 1.0000 | 0.0000 | 0.6875 |
| 5 | 1.0000 | 0.0000 | 1.0000 | 0.7500 | 0.9375 | 0.9375 | 0.0000 | 0.5625 |
| 6 | 1.0000 | 0.6875 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6250 | 1.0000 |
| 7 | 1.0000 | 0.0000 | 0.9375 | 0.9375 | 0.8750 | 1.0000 | 0.0000 | 0.9375 |

Median delta logL

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 589.3 | -4750.1 | -511.8 | -953.7 | 589.0 | -336.6 | -4579.6 | -938.2 |
| 1 | 1799.9 | 12.4 | 1239.5 | 1036.3 | 1629.0 | 1615.5 | 128.9 | 1552.3 |
| 2 | 509.4 | -4244.4 | -3.1 | -421.1 | 1047.4 | 400.7 | -4113.9 | -677.5 |
| 3 | 923.2 | -3826.0 | 405.4 | -137.5 | 1305.2 | 999.3 | -3626.2 | -180.2 |
| 4 | -601.9 | -5332.7 | -1140.9 | -1450.8 | -424.7 | -545.9 | -5088.9 | -1634.0 |
| 5 | 556.7 | -4876.7 | -616.1 | -939.7 | 278.6 | 199.6 | -4803.6 | -1230.2 |
| 6 | 2543.3 | -87.6 | 1686.5 | 1732.1 | 2664.0 | 2870.1 | -33.2 | 2029.4 |
| 7 | 924.3 | -3600.9 | 609.9 | 173.7 | 1879.9 | 1209.0 | -3552.9 | 46.3 |

Delta logL interquartile interval [25%, 75%]

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [-374.1, 1593.2] | [-5171.0, -4318.9] | [-869.8, -193.2] | [-1335.4, -827.2] | [-1241.6, 1469.0] | [-1044.7, 409.8] | [-5062.4, -4248.9] | [-1953.4, -807.5] |
| 1 | [1311.7, 2168.3] | [-41.9, 65.9] | [611.5, 1592.3] | [216.4, 1698.7] | [1235.9, 2137.4] | [1054.8, 2431.3] | [66.1, 241.0] | [1433.1, 1658.7] |
| 2 | [364.1, 981.1] | [-4373.0, -4096.3] | [-186.6, 389.1] | [-534.5, -293.6] | [415.1, 1781.7] | [208.7, 1074.3] | [-4289.3, -3920.2] | [-933.3, -349.7] |
| 3 | [684.4, 1323.6] | [-3955.4, -3673.7] | [248.9, 582.9] | [-241.2, 159.7] | [688.3, 2408.7] | [487.3, 1319.9] | [-3781.2, -3506.3] | [-458.6, -6.3] |
| 4 | [-1393.6, 1244.5] | [-6048.7, -4536.3] | [-1718.2, -410.4] | [-2308.2, -675.3] | [-1118.4, 651.3] | [-1767.5, 635.3] | [-5954.0, -4308.4] | [-2265.7, -927.4] |
| 5 | [-373.7, 1037.8] | [-5203.8, -4403.6] | [-1040.7, -255.0] | [-1362.9, -364.3] | [-681.7, 1659.5] | [-653.0, 638.6] | [-5088.4, -4148.8] | [-1602.1, -556.4] |
| 6 | [1952.1, 3899.7] | [-268.6, -40.1] | [1514.3, 1851.4] | [1503.7, 1872.0] | [2274.4, 3409.6] | [2557.7, 3891.5] | [-164.1, 93.9] | [1867.5, 2173.1] |
| 7 | [816.0, 1845.8] | [-3854.4, -3423.5] | [394.8, 843.3] | [1.9, 440.4] | [991.0, 2414.0] | [649.9, 1546.4] | [-3662.8, -3344.1] | [-42.7, 274.1] |

### Pair-block transplantation

ell9 survival fraction

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 0.0625 | 0.9375 | 0.9375 | 0.9375 | 1.0000 | 0.0000 | 0.6250 |
| 1 | 1.0000 | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9375 | 1.0000 |
| 2 | 1.0000 | 0.0625 | 0.8125 | 0.8125 | 0.8750 | 0.9375 | 0.0000 | 0.3750 |
| 3 | 1.0000 | 0.0625 | 1.0000 | 0.8750 | 1.0000 | 1.0000 | 0.0625 | 0.6250 |
| 4 | 1.0000 | 0.0625 | 1.0000 | 0.9375 | 1.0000 | 0.9375 | 0.0000 | 0.7500 |
| 5 | 1.0000 | 0.0625 | 1.0000 | 0.7500 | 0.9375 | 1.0000 | 0.0000 | 0.5625 |
| 6 | 1.0000 | 0.4375 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4375 | 1.0000 |
| 7 | 1.0000 | 0.0625 | 0.9375 | 0.8750 | 0.9375 | 0.9375 | 0.0000 | 0.8750 |

Median delta logL

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 686.8 | -4733.7 | -536.3 | -867.3 | 535.3 | -281.9 | -4216.4 | -1019.4 |
| 1 | 1687.3 | 42.2 | 1230.7 | 993.7 | 1503.1 | 1658.3 | 318.6 | 1536.8 |
| 2 | 575.4 | -4155.5 | 69.5 | -343.7 | 1125.9 | 580.7 | -3842.1 | -571.3 |
| 3 | 872.1 | -3801.0 | 328.1 | -99.4 | 1245.6 | 1051.1 | -3589.8 | -158.1 |
| 4 | -557.9 | -4990.5 | -1172.4 | -1230.9 | -429.7 | -296.8 | -4446.4 | -1657.4 |
| 5 | 427.6 | -4712.3 | -600.5 | -1028.4 | 283.0 | 219.4 | -4479.8 | -1223.9 |
| 6 | 2126.9 | -344.2 | 1325.9 | 1673.0 | 2541.9 | 2573.6 | -271.3 | 1904.2 |
| 7 | 1011.9 | -3603.7 | 566.9 | 172.2 | 1829.3 | 1260.0 | -3362.5 | 6.5 |

Delta logL interquartile interval [25%, 75%]

| recipient \ donor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [-442.5, 1423.1] | [-5092.8, -4164.9] | [-869.1, -235.0] | [-1295.3, -711.8] | [-1188.2, 1474.8] | [-1084.7, 448.1] | [-4979.4, -3751.4] | [-1782.2, -787.1] |
| 1 | [1368.6, 2057.8] | [-125.7, 199.8] | [592.8, 1596.9] | [275.2, 1786.4] | [1028.6, 2112.5] | [1061.5, 2366.9] | [69.5, 494.5] | [1318.8, 1651.7] |
| 2 | [331.7, 1003.0] | [-4374.1, -3887.4] | [-114.1, 286.1] | [-521.0, -205.9] | [376.4, 1776.6] | [311.5, 1056.0] | [-4019.4, -3294.2] | [-776.0, -331.3] |
| 3 | [585.8, 1412.1] | [-3932.7, -3465.0] | [179.0, 523.6] | [-144.6, 200.8] | [613.5, 2372.1] | [667.3, 1304.4] | [-3845.0, -2993.1] | [-467.6, 15.0] |
| 4 | [-1352.1, 1208.9] | [-6083.0, -4319.9] | [-1778.9, -391.8] | [-2308.8, -559.2] | [-1176.1, 795.0] | [-1673.7, 521.8] | [-5719.2, -3828.9] | [-2244.6, -827.8] |
| 5 | [-429.9, 1082.7] | [-5031.9, -4310.4] | [-1066.9, -319.8] | [-1298.5, -665.1] | [-574.6, 1612.0] | [-703.9, 603.9] | [-4951.8, -3840.4] | [-1553.1, -772.0] |
| 6 | [1679.5, 3660.0] | [-613.7, -42.8] | [1121.1, 1637.1] | [1253.2, 1779.8] | [2142.6, 3417.8] | [2038.1, 3003.2] | [-441.9, 231.8] | [1336.1, 1956.9] |
| 7 | [794.8, 1741.1] | [-3740.2, -3431.4] | [314.6, 779.6] | [32.6, 465.2] | [889.9, 2326.1] | [672.0, 1585.1] | [-3620.9, -2767.0] | [-92.7, 196.4] |

## Directional asymmetry

The median absolute difference between opposite-direction survival fractions is 0.40625 for single blocks and 0.4375 for pairs; the maximum is 1 for both. Of 28 unordered walker pairs, 27 single and 26 pair comparisons are asymmetric.

| i,j | single j→i | single i→j | difference | pair j→i | pair i→j | difference |
| --- | --- | --- | --- | --- | --- | --- |
| 0, 1 | 0.0000 | 1.0000 | -1.0000 | 0.0625 | 1.0000 | -0.9375 |
| 0, 2 | 0.9375 | 1.0000 | -0.0625 | 0.9375 | 1.0000 | -0.0625 |
| 0, 3 | 0.8125 | 1.0000 | -0.1875 | 0.9375 | 1.0000 | -0.0625 |
| 0, 4 | 0.9375 | 1.0000 | -0.0625 | 0.9375 | 1.0000 | -0.0625 |
| 0, 5 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| 0, 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 1.0000 | -1.0000 |
| 0, 7 | 0.6250 | 1.0000 | -0.3750 | 0.6250 | 1.0000 | -0.3750 |
| 1, 2 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0625 | +0.9375 |
| 1, 3 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0625 | +0.9375 |
| 1, 4 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0625 | +0.9375 |
| 1, 5 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0625 | +0.9375 |
| 1, 6 | 1.0000 | 0.6875 | +0.3125 | 0.9375 | 0.4375 | +0.5000 |
| 1, 7 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0625 | +0.9375 |
| 2, 3 | 0.7500 | 1.0000 | -0.2500 | 0.8125 | 1.0000 | -0.1875 |
| 2, 4 | 0.8750 | 1.0000 | -0.1250 | 0.8750 | 1.0000 | -0.1250 |
| 2, 5 | 0.8750 | 1.0000 | -0.1250 | 0.9375 | 1.0000 | -0.0625 |
| 2, 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 1.0000 | -1.0000 |
| 2, 7 | 0.4375 | 0.9375 | -0.5000 | 0.3750 | 0.9375 | -0.5625 |
| 3, 4 | 1.0000 | 0.8750 | +0.1250 | 1.0000 | 0.9375 | +0.0625 |
| 3, 5 | 1.0000 | 0.7500 | +0.2500 | 1.0000 | 0.7500 | +0.2500 |
| 3, 6 | 0.0000 | 1.0000 | -1.0000 | 0.0625 | 1.0000 | -0.9375 |
| 3, 7 | 0.6875 | 0.9375 | -0.2500 | 0.6250 | 0.8750 | -0.2500 |
| 4, 5 | 1.0000 | 0.9375 | +0.0625 | 0.9375 | 0.9375 | +0.0000 |
| 4, 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 1.0000 | -1.0000 |
| 4, 7 | 0.6875 | 0.8750 | -0.1875 | 0.7500 | 0.9375 | -0.1875 |
| 5, 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | 1.0000 | -1.0000 |
| 5, 7 | 0.5625 | 1.0000 | -0.4375 | 0.5625 | 0.9375 | -0.3750 |
| 6, 7 | 1.0000 | 0.0000 | +1.0000 | 1.0000 | 0.0000 | +1.0000 |

## Physical displacement (post-hoc only)

Stage-4I permutation-invariant machinery measures frequency-set displacement in Fourier bins (1 bin = 1/31536000 Hz), and Hungarian-matched six-coordinate source-set distance normalized by prior widths with existing periodic-coordinate handling. Matching never affects transplantation. Values below are median [5%, 95%]; full distributions are saved.

| unit | group | ell9 outcome | n | frequency-set bins | six-coordinate distance |
| --- | --- | --- | --- | --- | --- |
| single | cross | survives | 643 | 3.04438 [0.13718, 53.21834] | 0.07666 [0.04289, 0.12978] |
| single | cross | fails | 253 | 28.98642 [0.40802, 64.33177] | 0.09991 [0.04628, 0.14367] |
| single | temporal_control | survives | 115 | 0.57476 [0.03919, 70.30769] | 0.00809 [0.00264, 0.14253] |
| single | temporal_control | fails | 13 | 1.11496 [0.12931, 49.47208] | 0.01074 [0.00164, 0.14963] |
| pair | cross | survives | 647 | 16.92168 [2.56867, 91.38476] | 0.12941 [0.08861, 0.17305] |
| pair | cross | fails | 249 | 47.77978 [4.71921, 101.71876] | 0.14102 [0.10050, 0.18524] |
| pair | temporal_control | survives | 110 | 17.76315 [1.66786, 121.02044] | 0.11449 [0.07612, 0.18250] |
| pair | temporal_control | fails | 18 | 52.63828 [6.80987, 88.93631] | 0.15166 [0.08518, 0.19266] |

## Existing source-contribution diagnostic

The unchanged model’s `_marg_recon` exposes `source_gain = |x_k|² / (M_jitter⁻¹)_kk`, a conditional regularized **profiled quadratic gain**, with other amplitudes reprofiled. Its mask and jitter match the configured integrated model. This is nonadditive, excludes the integrated log-determinant, and is **not an additive per-source integrated likelihood**. We use it only to characterize source importance.

| walker | historical label | sticky median gain | companion median gain | largest median gain | sticky largest | sticky frequency range Hz | median nearest gap bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7 | 9584.5 | 114.3 | 9584.5 | 16/16 | [0.001840875085, 0.001841177126] | 43.30 |
| 1 | 3 | 115.8 | 162.2 | 9266.3 | 0/16 | [0.001830752715, 0.001849324095] | 23.05 |
| 2 | 3 | 8631.5 | 89.6 | 8631.5 | 16/16 | [0.001841149515, 0.001841234976] | 17.83 |
| 3 | 0 | 7768.3 | 227.7 | 7768.3 | 16/16 | [0.001841179744, 0.001841236632] | 14.93 |
| 4 | 8 | 10717.3 | 67.6 | 10717.3 | 16/16 | [0.001840725011, 0.001840957664] | 22.28 |
| 5 | 2 | 10012.9 | 112.5 | 10012.9 | 16/16 | [0.001840835422, 0.001841163334] | 17.67 |
| 6 | 7 | 326.1 | 386.4 | 6218.6 | 0/16 | [0.001830138494, 0.001849380895] | 29.15 |
| 7 | 4 | 7318.2 | 150.5 | 7318.2 | 16/16 | [0.001841084088, 0.001841203079] | 26.06 |

The historical sticky block is the largest-gain source at every selected state of all six persistent walkers. Their frequencies span approximately 0.001840725–0.001841237 Hz (about 16.14 Fourier bins). The nearest companion has much smaller median gain. In walker 1 the largest gain is instead at label 6 in all 16 states; in walker 6 it is at label 2 in 15 states and label 8 in one state. These are measurements only: no source was relabelled or transplanted using this information.

The evidence favors difficulty exploring viable configurations of a dominant source as a contributor to the persistent disagreement. It does not establish that this source alone explains all disagreement. Learned critical-source blocks are usually transferable across the six persistent catalogues, and pairing them with a nearest-frequency neighbour provides negligible additional survival. The temporal controls do not support an intrinsically nontransferable block. Residual failures and directional effects leave some context dependence unresolved; no numerical classification cutoffs were fitted.

## Integrity, tests and artifacts

All 128 base states have finite implemented prior and likelihood and strict ell9 membership. Direct cached-prior discrepancy is zero; maximum cached-likelihood discrepancy is 1.46e-11. Every saved transplant was independently reconstructed from the original trace and checked byte-for-byte, including donor copies and untouched coordinates. Both experiments have zero nonfinite proposed priors, zero nonfinite likelihoods and zero structural failures. Protected model, prior configuration, frozen checkpoint, input traces, slice settings and generic DNS file hashes remain unchanged.

Cost: 2,176 scalar likelihood evaluations and 2,176 implemented-prior evaluations (128 base checks + 2,048 transplants), plus 128 existing reconstruction diagnostic calls. No sampling, retries, new levels or evidence computation occurred.

Regression: **166 passed** (156 existing + 10 cheap Stage-4Q tests), before real evaluations. Tests cover deterministic indices, temporal control mapping, single/pair copying, nearest-companion ties, frequency-only assignment, untouched coordinates, no sequential update and complete 8×8 bookkeeping. No real LISA likelihood is invoked by these tests.

Code: [diagnostic runner](../../examples/lisa_dns_stage4/source_transplant_diagnostic.py), [read-only analysis](../../examples/lisa_dns_stage4/analyze_source_transplants.py), [cheap tests](../../tests/experimental/test_dns_gb_source_transplant_diagnostic.py).

Artifacts in `/tmp/lisa_dns_stage4q_transplantations/`: `design.json` (pre-evaluation assignments), `bases.npz`, `single_transplants.npz`, `pair_transplants.npz`, `source_contributions.npz`, `report.json` (every evaluation and complete distributions), `analysis.json` (independent reconstruction and historical-group interpretation), `single_matrix_cells.csv`, `pair_matrix_cells.csv`, and `validation.json`. Test log: `/tmp/lisa_dns_stage4q_tests.log`. The ladder remains frozen through level 9; both previous level-10 candidates remain rejected.
