# Stage 4X: one independent population level-11 construction attempt

**Yes.** The single authorized 128+256 attempt passed the unchanged existing gate. Independent calibration ESS = **83.04742208142542**, required ≥20. Candidate `ell_11 = -108868.60077896297`.

Exactly one additional level is frozen: `ell_11 = -108868.60077896297`, `log X_11 = -9.643899891984567`. The mass increment is the log of the independent calibration compression, not the selection compression.

Calibration between-walker SE remained controlled under the unchanged gate and fell from Stage-4V 0.06913128436451788 to 0.0496193565736982 (28.2% lower). Residual heterogeneity remains: calibration exceedance fractions span 0.0703125–0.43359375. This is operationally acceptable independent construction, not proof of whole-catalogue physical convergence.

## Frozen setup and independent starts

The exact Stage-4V checkpoint was loaded, including its complete level-0–10 prefix, and levels 0–9 were verified against Stage 4G. The lower checkpoint was copied byte for byte; no lower threshold or log mass was rebuilt or recalibrated. Source checkpoint SHA256: `5860e81f4f5c54237559248ea404eaa12a685d2ddd2c2c4b48ecdf1d3b9a864a`.

Each bank uses only its own Stage-4V retained history. For each walker the starting state is the **last** retained state with logL > ell10; maxima were never selected. Every position and both caches are exact copies from the indicated trace index. Banks have distinct states, separate array/cache storage, distinct random streams and recorded source-file hashes. Every starting state passed direct prior/likelihood cache checks before its first sweep.

**selection** source: `/tmp/lisa_dns_stage4v_population_level10/selection/trace.npz`, SHA256 `114da5f7a30530a6db319a55182ce284beef41a0ac2fc334669c22d115682074`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 381 | 253 | -109767.51730725613 |
| 1 | 382 | 254 | -109765.03767304153 |
| 2 | 383 | 255 | -108900.02506924949 |
| 3 | 383 | 255 | -109423.70695176245 |
| 4 | 382 | 254 | -109686.13800569519 |
| 5 | 355 | 227 | -109335.12647204267 |
| 6 | 380 | 252 | -109493.97881553495 |
| 7 | 381 | 253 | -109639.21059164709 |

**calibration** source: `/tmp/lisa_dns_stage4v_population_level10/calibration/trace.npz`, SHA256 `06b0ce9536ce03169b34a3a0a77e679a36f8620a714a57fc87e7302925d9b55b`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 383 | 255 | -107861.51300179477 |
| 1 | 383 | 255 | -109604.57077308747 |
| 2 | 383 | 255 | -109550.11122971782 |
| 3 | 383 | 255 | -108988.73167866454 |
| 4 | 382 | 254 | -109013.57187485501 |
| 5 | 377 | 249 | -109261.12086012302 |
| 6 | 376 | 248 | -109713.97700038827 |
| 7 | 382 | 254 | -109133.6817948025 |

Both banks targeted the fixed ell10 contour throughout all 384 sweeps, including calibration after the candidate was selected. Each has exactly 8 walkers, 128 discarded burn-in population sweeps and 256 retained population sweeps. Block size 32, target compression exp(-1), min_ess 20. Stage-4W states were not used for initialization, and its pseudo-threshold was not used for candidate selection.

The population transition is unchanged: eight validated isotropic slice updates followed by four disjoint reciprocal exchanges on the frozen seven-round schedule. Slice scales pi/sqrt(3), mixture .2/.4/.4, max_steps 10, max_shrinkage 100. Frequency selector f_star = 0.0018409808 Hz, sigma_f = 2.558e-7 Hz, stable logsumexp, no clipping. Each exchange swaps complete six-coordinate blocks, evaluates the full implemented joint-prior ratio, recomputes reverse label probabilities, and requires both proposals strictly above ell10. One joint MH decision advances both or neither. No source_gain enters the kernel.

Frozen slice seeds: `{'selection': [910100, 910101, 910102, 910103, 910104, 910105, 910106, 910107], 'calibration': [910200, 910201, 910202, 910203, 910204, 910205, 910206, 910207]}`; independent PCG64 label/MH seeds: `{'selection': [2026092807, 2026092808], 'calibration': [2026092809, 2026092810]}`. Both banks’ draws, designs and starts were saved before sampling. A source-equivalence test verifies that the Stage-4W run body differs only in output location, bank slice streams and sweep count.

## Construction statistics and Stage-4V comparison

The candidate was selected from the 2048 retained selection states by the existing strict empirical rule, without interpolation or jitter, and saved **before calibration sampling**. Independent calibration evaluates strict exceedance of this fixed candidate while continuing to target ell10. The historical selection-status and structural gates are retained in addition to calibration ESS ≥20.

| attempt | candidate threshold | selection compression | selection ESS | calibration compression | calibration ESS |
| --- | --- | --- | --- | --- | --- |
| Stage 4V: level 10 | -109780.28875081123 | 0.3676757812 | 32.20894459 | 0.4873046875 | 52.27700334 |
| Stage 4X: level 11 | -108868.60077896297 | 0.3676757812 | 54.10245944 | 0.2866210938 | 83.04742208 |

| diagnostic | V selection | V calibration | X selection | X calibration |
| --- | --- | --- | --- | --- |
| IID/Bernoulli SE | 0.01065460721 | 0.01104498147 | 0.01065460721 | 0.009991926486 |
| block-means SE | 0.03537767973 | 0.02871102186 | 0.02861827499 | 0.02680751378 |
| between-walker SE | 0.08495993527 | 0.06913128436 | 0.0655532029 | 0.04961935657 |
| walker exceedance range | 0.05078125 – 0.66796875 | 0.01953125 – 0.6171875 | 0.10546875 – 0.5625 | 0.0703125 – 0.43359375 |
| walker exceedance SD | 0.2403029854 | 0.1955327999 | 0.1854124572 | 0.140344734 |
| walker mean-logL SD | 275.8816872 | 311.4313296 | 330.5827719 | 270.3919525 |
| walker mean-logL MAD | 218.5796885 | 71.97596207 | 252.0894754 | 192.8220507 |
| walker mean-logL IQR | 486.8528254 | 126.0180293 | 510.277951 | 397.8920646 |

Stage-4X **selection** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.546875 | -108540.8719 |
| 1 | 0.46875 | -108700.0978 |
| 2 | 0.5625 | -108553.2644 |
| 3 | 0.4765625 | -108704.4525 |
| 4 | 0.41015625 | -108893.8627 |
| 5 | 0.26171875 | -109136.3375 |
| 6 | 0.109375 | -109382.6726 |
| 7 | 0.10546875 | -109285.6573 |

Stage-4X **calibration** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.1953125 | -109221.2187 |
| 1 | 0.38671875 | -108797.1918 |
| 2 | 0.0703125 | -109503.0577 |
| 3 | 0.43359375 | -108772.3332 |
| 4 | 0.41015625 | -108877.9105 |
| 5 | 0.3671875 | -108949.0269 |
| 6 | 0.3125 | -109006.1422 |
| 7 | 0.1171875 | -109358.8354 |

Uncertainty and conservative Bernoulli-equivalent ESS use the unchanged existing machinery: maximum of IID, block-means and between-walker SE, block size 32. SD uses ddof=1; mean-logL MAD is the unscaled median absolute deviation of the eight walker means; IQR is their 75th–25th percentile difference. These interacting walkers are not independent convergence replications. The construction gate is the specified operational criterion, not proof of physical equilibrium.

The historical failed level-10 attempts remain read-only context:

| attempt | candidate threshold | selection ESS | calibration ESS | calibration between-walker SE |
| --- | --- | --- | --- | --- |
| original isotropic | -109401.11425363769 | 24.27235681 | 16.79821519 | 0.1174122426 |
| Hybrid P | -109301.45870884096 | 25.32714806 | 18.27464305 | 0.1117598878 |

## Historical walker 6

Walker 6 is an ensemble-lineage label, not a permanently pathological physical mode. It receives exactly the same transition and pairing opportunities as every other walker.

| bank | starting logL | retained mean logL | exceedance fraction | other seven mean-logL range | other seven fraction range | accepted exchanges / transfers received | distinct donors | donor IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | -109493.9788 | -109382.6726 | 0.109375 | -109285.6573 – -108540.8719 | 0.10546875 – 0.5625 | 17 | 7 | [0, 1, 2, 3, 4, 5, 7] |
| calibration | -109713.977 | -109006.1422 | 0.3125 | -109503.0577 – -108772.3332 | 0.0703125 – 0.43359375 | 108 | 7 | [0, 1, 2, 3, 4, 5, 7] |

## Population exchanges and exact realization communication

| attempt / bank | proposals | joint survivors | survival fraction | accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 1536 | 278 | 0.1809895833 | 255 | 0.166015625 | 23 | 0.08273381295 |
| V / calibration | 1536 | 635 | 0.4134114583 | 600 | 0.390625 | 35 | 0.05511811024 |
| X / selection | 1536 | 448 | 0.2916666667 | 431 | 0.2805989583 | 17 | 0.03794642857 |
| X / calibration | 1536 | 484 | 0.3151041667 | 469 | 0.3053385417 | 15 | 0.03099173554 |

| bank | distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selection | selection_log_ratio | -3.980997252 | -0.02753743559 | 1.650049773e-07 | 0.03670017302 | 3.971915581 | 0 |
| selection | joint_prior_log_ratio | -1.705302566e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.98951966e-13 | 0 |
| calibration | selection_log_ratio | -1.160511201 | -0.03544816751 | 4.297603939e-13 | 0.05669368511 | 3.960985842 | 0 |
| calibration | joint_prior_log_ratio | -1.705302566e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.421085472e-13 | 0 |

| bank | nonfinite proposed priors | nonfinite proposed likelihoods | nonfinite MH ratios |
| --- | --- | --- | --- |
| selection | 0 | 0 | 0 |
| calibration | 0 | 0 | 0 |

| pair | selection accepted / proposed | selection survival fraction | selection acceptance fraction | calibration accepted / proposed | calibration survival fraction | calibration acceptance fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 35 / 54 | 0.6666666667 | 0.6481481481 | 15 / 54 | 0.2962962963 | 0.2777777778 |
| 0,2 | 47 / 55 | 0.8545454545 | 0.8545454545 | 13 / 55 | 0.2363636364 | 0.2363636364 |
| 0,3 | 36 / 55 | 0.6545454545 | 0.6545454545 | 15 / 55 | 0.2727272727 | 0.2727272727 |
| 0,4 | 26 / 55 | 0.5090909091 | 0.4727272727 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 0,5 | 10 / 55 | 0.1818181818 | 0.1818181818 | 16 / 55 | 0.3090909091 | 0.2909090909 |
| 0,6 | 2 / 55 | 0.03636363636 | 0.03636363636 | 18 / 55 | 0.3272727273 | 0.3272727273 |
| 0,7 | 5 / 55 | 0.09090909091 | 0.09090909091 | 10 / 55 | 0.2 | 0.1818181818 |
| 1,2 | 31 / 55 | 0.5636363636 | 0.5636363636 | 10 / 55 | 0.1818181818 | 0.1818181818 |
| 1,3 | 28 / 55 | 0.5090909091 | 0.5090909091 | 32 / 55 | 0.5818181818 | 0.5818181818 |
| 1,4 | 20 / 55 | 0.3636363636 | 0.3636363636 | 31 / 55 | 0.5636363636 | 0.5636363636 |
| 1,5 | 12 / 55 | 0.2181818182 | 0.2181818182 | 23 / 55 | 0.4363636364 | 0.4181818182 |
| 1,6 | 5 / 55 | 0.1636363636 | 0.09090909091 | 27 / 55 | 0.4909090909 | 0.4909090909 |
| 1,7 | 9 / 55 | 0.1636363636 | 0.1636363636 | 15 / 55 | 0.2727272727 | 0.2727272727 |
| 2,3 | 37 / 55 | 0.6727272727 | 0.6727272727 | 9 / 55 | 0.1636363636 | 0.1636363636 |
| 2,4 | 26 / 55 | 0.4909090909 | 0.4727272727 | 14 / 55 | 0.2545454545 | 0.2545454545 |
| 2,5 | 15 / 55 | 0.3272727273 | 0.2727272727 | 12 / 55 | 0.2181818182 | 0.2181818182 |
| 2,6 | 1 / 55 | 0.03636363636 | 0.01818181818 | 10 / 55 | 0.1818181818 | 0.1818181818 |
| 2,7 | 5 / 54 | 0.09259259259 | 0.09259259259 | 15 / 54 | 0.2962962963 | 0.2777777778 |
| 3,4 | 22 / 55 | 0.4 | 0.4 | 27 / 55 | 0.4909090909 | 0.4909090909 |
| 3,5 | 13 / 55 | 0.2363636364 | 0.2363636364 | 20 / 55 | 0.3818181818 | 0.3636363636 |
| 3,6 | 1 / 54 | 0.01851851852 | 0.01851851852 | 16 / 54 | 0.2962962963 | 0.2962962963 |
| 3,7 | 3 / 55 | 0.05454545455 | 0.05454545455 | 17 / 55 | 0.3272727273 | 0.3090909091 |
| 4,5 | 9 / 54 | 0.1851851852 | 0.1666666667 | 20 / 54 | 0.3888888889 | 0.3703703704 |
| 4,6 | 6 / 55 | 0.1090909091 | 0.1090909091 | 19 / 55 | 0.4 | 0.3454545455 |
| 4,7 | 8 / 55 | 0.1636363636 | 0.1454545455 | 12 / 55 | 0.2363636364 | 0.2181818182 |
| 5,6 | 1 / 55 | 0.05454545455 | 0.01818181818 | 15 / 55 | 0.3090909091 | 0.2727272727 |
| 5,7 | 17 / 55 | 0.3272727273 | 0.3090909091 | 15 / 55 | 0.2727272727 | 0.2727272727 |
| 6,7 | 1 / 55 | 0.01818181818 | 0.01818181818 | 3 / 55 | 0.07272727273 | 0.05454545455 |

Every bank executes 54 complete seven-round cycles and the first six rounds. Each pair receives 54 or 55 fixed opportunities; every walker receives exactly 384. Pairing is never adapted from acceptance or provenance.

**selection:** 862 directional inter-walker transfers (510 in Stage 4V); 536 distinct exact realizations migrated (390 in Stage 4V); 817 arrivals to a walker other than the realization’s birth walker; 793 first recipient visits. 1 transfers involved exact initial tokens; 35/72 initial ancestries visited another walker.

| recipient | X accepted transfers received | X distinct donor walkers | X donor IDs | V distinct donor walkers |
| --- | --- | --- | --- | --- |
| 0 | 161 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 |
| 1 | 140 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 |
| 2 | 162 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 |
| 3 | 140 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 |
| 4 | 117 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 |
| 5 | 77 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 |
| 6 | 17 | 7 | [0, 1, 2, 3, 4, 5, 7] | 7 |
| 7 | 48 | 7 | [0, 1, 2, 3, 4, 5, 6] | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 35 | 47 | 36 | 26 | 10 | 2 | 5 |
| 1 | 35 | 0 | 31 | 28 | 20 | 12 | 5 | 9 |
| 2 | 47 | 31 | 0 | 37 | 26 | 15 | 1 | 5 |
| 3 | 36 | 28 | 37 | 0 | 22 | 13 | 1 | 3 |
| 4 | 26 | 20 | 26 | 22 | 0 | 9 | 6 | 8 |
| 5 | 10 | 12 | 15 | 13 | 9 | 0 | 1 | 17 |
| 6 | 2 | 5 | 1 | 1 | 6 | 1 | 0 | 1 |
| 7 | 5 | 9 | 5 | 3 | 8 | 17 | 1 | 0 |

**calibration:** 938 directional inter-walker transfers (1200 in Stage 4V); 587 distinct exact realizations migrated (663 in Stage 4V); 914 arrivals to a walker other than the realization’s birth walker; 890 first recipient visits. 5 transfers involved exact initial tokens; 42/72 initial ancestries visited another walker.

| recipient | X accepted transfers received | X distinct donor walkers | X donor IDs | V distinct donor walkers |
| --- | --- | --- | --- | --- |
| 0 | 107 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 |
| 1 | 153 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 |
| 2 | 83 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 |
| 3 | 136 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 |
| 4 | 143 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 |
| 5 | 121 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 |
| 6 | 108 | 7 | [0, 1, 2, 3, 4, 5, 7] | 6 |
| 7 | 87 | 7 | [0, 1, 2, 3, 4, 5, 6] | 6 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 15 | 13 | 15 | 20 | 16 | 18 | 10 |
| 1 | 15 | 0 | 10 | 32 | 31 | 23 | 27 | 15 |
| 2 | 13 | 10 | 0 | 9 | 14 | 12 | 10 | 15 |
| 3 | 15 | 32 | 9 | 0 | 27 | 20 | 16 | 17 |
| 4 | 20 | 31 | 14 | 27 | 0 | 20 | 19 | 12 |
| 5 | 16 | 23 | 12 | 20 | 20 | 0 | 15 | 15 |
| 6 | 18 | 27 | 10 | 16 | 19 | 15 | 0 | 3 |
| 7 | 10 | 15 | 15 | 17 | 12 | 15 | 3 | 0 |

Provenance is exactly the Stage-4U/W scheme: initial ancestry IDs follow slice descendants; exact-realization tokens are newly minted for each slice-targeted block, including conservative invalidation of roundoff-identical updates; untouched tokens persist. Tokens and ancestry swap only on joint acceptance. Independent saved-operation replay verifies copied six-coordinate blocks, forward/reverse probabilities, joint decisions, cache outcomes and token histories **before any freeze**. Full transfers and token migration counts are in the bank provenance files and comparison.json. Communication does not by itself establish whole-catalogue convergence.

## Permutation-invariant physical geometry

The final window is exactly each bank’s retained 256 sweeps (absolute sweeps 129–384, 1-based). These are unchanged Stage-4I/U/W Hungarian-matched distances: frequency RMS in Fourier bins (DF=1/31536000 Hz), six-coordinate RMS normalized by prior widths, psi period pi and lam period 2pi. Frequency distances use all cross-time pairs; full-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. Geometry is diagnostic only and does not add a gate.

| bank | median frequency-set RMS (bins) | median sorted-frequency mean distance (bins) | median six-coordinate RMS |
| --- | --- | --- | --- |
| selection | 104.1483974 | 69.54760641 | 0.2612805554 |
| calibration | 115.5827323 | 87.36333498 | 0.2601356146 |

| pair | selection frequency-set RMS | selection sorted-frequency mean distance | selection six-coordinate RMS | calibration frequency-set RMS | calibration sorted-frequency mean distance | calibration six-coordinate RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 103.0802349 | 66.43940911 | 0.2504497204 | 125.4568254 | 92.20650289 | 0.2660177643 |
| 0,2 | 171.6301706 | 157.1554572 | 0.2729530466 | 131.1279596 | 102.9386374 | 0.2582827221 |
| 0,3 | 113.4664838 | 87.20781142 | 0.2531625802 | 76.57278833 | 22.39067032 | 0.2529370882 |
| 0,4 | 97.00206661 | 61.34698647 | 0.2528599679 | 138.5838363 | 115.5459581 | 0.2663759167 |
| 0,5 | 138.2036149 | 120.325901 | 0.2644053556 | 85.22086587 | 43.97017616 | 0.2685457668 |
| 0,6 | 216.9115473 | 204.6483022 | 0.2819440842 | 145.2028675 | 117.4086266 | 0.2610431858 |
| 0,7 | 126.9507577 | 100.4165106 | 0.2668673615 | 112.965879 | 82.52016707 | 0.2564800512 |
| 1,2 | 125.1266793 | 95.70105045 | 0.2587729216 | 89.44705601 | 21.3438222 | 0.2629104489 |
| 1,3 | 90.97030559 | 35.59829809 | 0.2480268894 | 124.8829201 | 96.36729517 | 0.2718112944 |
| 1,4 | 93.73338929 | 37.43104306 | 0.24754344 | 88.61645671 | 32.42262745 | 0.2503456523 |
| 1,5 | 98.86597855 | 58.42643139 | 0.2689340061 | 123.7086719 | 95.01356507 | 0.271742281 |
| 1,6 | 169.1528131 | 147.3229966 | 0.266280281 | 109.5015437 | 61.19201411 | 0.2592280434 |
| 1,7 | 96.95132926 | 40.3438205 | 0.2688326004 | 93.29930073 | 42.49716806 | 0.2477667007 |
| 2,3 | 104.0911209 | 72.65580371 | 0.2585624383 | 130.3193147 | 106.3636817 | 0.2658063764 |
| 2,4 | 129.6548535 | 104.2965856 | 0.2607224279 | 82.14563756 | 24.1961423 | 0.2493175113 |
| 2,5 | 89.60837022 | 55.86670036 | 0.2646134431 | 129.6327688 | 105.6768374 | 0.2801839554 |
| 2,6 | 94.04559521 | 58.20848004 | 0.2491101075 | 103.8063178 | 56.29230232 | 0.2475824324 |
| 2,7 | 102.7527781 | 65.10344876 | 0.2618386829 | 92.94783256 | 48.78264565 | 0.2524816447 |
| 3,4 | 89.44511962 | 39.46470966 | 0.2479500008 | 139.5670116 | 120.6342703 | 0.2578643453 |
| 3,5 | 91.45344964 | 54.26600033 | 0.2535327762 | 71.36294049 | 26.30090101 | 0.2570022959 |
| 3,6 | 144.6159758 | 122.2582136 | 0.2652685145 | 150.7868477 | 127.9291759 | 0.2675843612 |
| 3,7 | 88.41876612 | 31.42993763 | 0.2633893349 | 118.1995855 | 94.5923774 | 0.2626181012 |
| 4,5 | 112.2138889 | 82.44097645 | 0.2589897338 | 139.1818015 | 120.3248956 | 0.2707660902 |
| 4,6 | 167.7220893 | 147.605833 | 0.2630114289 | 93.80140513 | 43.99253054 | 0.2500772453 |
| 4,7 | 104.2056739 | 60.43407672 | 0.2545492174 | 90.33078847 | 51.41659011 | 0.2503478632 |
| 5,6 | 130.9120507 | 108.7142087 | 0.2585196532 | 157.0788401 | 135.409722 | 0.2803088579 |
| 5,7 | 86.16753778 | 35.24671882 | 0.2631885092 | 125.4188543 | 103.6305702 | 0.2671984417 |
| 6,7 | 144.5415088 | 119.0040308 | 0.2638630977 | 94.92230858 | 45.27453159 | 0.2503997727 |

## Cost and integrity

| attempt / bank | slice_likelihood_proxy | exchange_likelihood | slice_prior_proxy | exchange_prior | direct_cache_likelihood | direct_cache_prior | total_likelihood_proxy | total_prior_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 21468 | 3072 | 21468 | 3072 | 6152 | 6152 | 30692 | 30692 |
| V / calibration | 20951 | 3072 | 20951 | 3072 | 6152 | 6152 | 30175 | 30175 |
| X / selection | 20855 | 3072 | 20855 | 3072 | 6152 | 6152 | 30079 | 30079 |
| X / calibration | 21308 | 3072 | 21308 | 3072 | 6152 | 6152 | 30532 | 30532 |

Slice cost retains the historical num_steps+num_shrink proxy, with prior and likelihood evaluated together. Exchange evaluations and direct cache checks, including starts, are counted separately. No cost-based truncation, optimization, added sweeps or extra walkers were used.

selection: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -918.4168830377727. CPU loop time 51.42444437 s (not a cross-hardware speed comparison).

calibration: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -901.6519130696037. CPU loop time 51.9310324 s (not a cross-hardware speed comparison).

Every start and every constituent outcome passed finite prior/likelihood, finite coordinates, strict ell10 membership and direct cache checks (prior atol 1e-9, likelihood atol 1e-7, rtol 0). Selected blocks swapped exactly, untouched coordinates were unchanged, rejection restored both post-slice states, and accepted exchanges advanced both jointly. Both banks’ exact saved-state and provenance replays passed. The complete lower prefix, historical files, prior/model/configuration, selector, fixed schedule and slice source retained their protected hashes.

**235 pre-run tests passed:** all 223 existing tests plus 12 Stage-4X tests. Coverage includes exact lower prefix, deterministic own-bank survivor recovery and caches, missing-survivor stop, independent storage/streams, strict ell10 transition and source equivalence, selection-only candidate, unchanged ESS=20 boundary, no freeze on failure, exactly level 11 on success, impossible level 12, calibration-only mass, and reuse of the exact ell10 provenance replay. No real LISA trajectory runs in pytest.

Artifacts: `/tmp/lisa_dns_stage4x_population_level11/` contains preflight.json, the copied checkpoint_level10.json, each bank’s initial_states.npz, random_schedule.npz, design.json, all 384-sweep trace.npz, exchanges.npz, provenance.json and provenance_transfers.json, selection_candidate.json, report.json, comparison.json, levels.npz and validation.json, plus checkpoint_level11.json.

Exactly one attempt was made. No retry, level 12, lower-level recalibration, kernel tuning, special walker-6 probabilities, production or evidence implementation occurred. Evidence remains unvalidated and out of scope. Stop here.
