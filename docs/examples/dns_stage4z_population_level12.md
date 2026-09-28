# Stage 4Z: one independent population level-12 construction attempt

**Yes.** The single authorized 128+256 attempt passed the unchanged existing gate. Independent calibration ESS = **634.1474269819192**, required ≥20. Candidate `ell_12 = -108187.99712765554`.

Exactly one additional level is frozen: `ell_12 = -108187.99712765554`, `log X_12 = -10.418392711998465`. The mass increment is the log of the independent calibration compression, not the selection compression.

## Frozen setup and independent starts

The exact Stage-4X checkpoint was loaded, including its complete level-0–11 prefix, and levels 0–9 were verified against Stage 4G. The lower checkpoint was copied byte for byte; no lower threshold or log mass was rebuilt or recalibrated. Source checkpoint SHA256: `95025df6db0eb236038bdeec5e69a68089c06fcac73cc64151565557084de331`.

Each bank uses only its own Stage-4X retained history. For each walker the starting state is the **last** retained state with logL > ell11; maxima were never selected. Every position and both caches are exact copies from the indicated trace index. Banks have distinct states, separate array/cache storage, distinct random streams and recorded source-file hashes. Every starting state passed direct prior/likelihood cache checks before its first sweep.

**selection** source: `/tmp/lisa_dns_stage4x_population_level11/selection/trace.npz`, SHA256 `1d9edd611b723311d626cf14a235072a1783b221478b9166a5ac7dbc3ffc64cb`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 381 | 253 | -107834.23522757222 |
| 1 | 383 | 255 | -108762.93816231715 |
| 2 | 376 | 248 | -105835.85982515386 |
| 3 | 362 | 234 | -108740.19922178706 |
| 4 | 377 | 249 | -108855.48523506355 |
| 5 | 383 | 255 | -107508.06331426531 |
| 6 | 369 | 241 | -108523.47477027444 |
| 7 | 380 | 252 | -108597.18504896767 |

**calibration** source: `/tmp/lisa_dns_stage4x_population_level11/calibration/trace.npz`, SHA256 `9427128d9711592778d3d98f9964b431f79f33a3a3c91fa8d340ea2bc2ec3576`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 383 | 255 | -108615.50843659024 |
| 1 | 383 | 255 | -105888.61895776488 |
| 2 | 222 | 94 | -108218.73287424145 |
| 3 | 379 | 251 | -108842.17649887555 |
| 4 | 382 | 254 | -108368.31002360515 |
| 5 | 372 | 244 | -108106.80476661834 |
| 6 | 381 | 253 | -108228.74174357545 |
| 7 | 383 | 255 | -108375.78618604878 |

Both banks targeted the fixed ell11 contour throughout all 384 sweeps, including calibration after the candidate was selected. Each has exactly 8 walkers, 128 discarded burn-in population sweeps and 256 retained population sweeps. Block size 32, target compression exp(-1), min_ess 20. Stage-4Y states were not used for initialization, and its pseudo-threshold was not used for candidate selection.

The population transition is unchanged: eight validated isotropic slice updates followed by four disjoint reciprocal exchanges on the frozen seven-round schedule. Slice scales pi/sqrt(3), mixture .2/.4/.4, max_steps 10, max_shrinkage 100. Frequency selector f_star = 0.0018409808 Hz, sigma_f = 2.558e-7 Hz, stable logsumexp, no clipping. Each exchange swaps complete six-coordinate blocks, evaluates the full implemented joint-prior ratio, recomputes reverse label probabilities, and requires both proposals strictly above ell11. One joint MH decision advances both or neither. No source_gain enters the kernel.

Frozen slice seeds: `{'selection': [930100, 930101, 930102, 930103, 930104, 930105, 930106, 930107], 'calibration': [930200, 930201, 930202, 930203, 930204, 930205, 930206, 930207]}`; independent PCG64 label/MH seeds: `{'selection': [2026092813, 2026092814], 'calibration': [2026092815, 2026092816]}`. Both banks’ draws, designs and starts were saved before sampling. A source-equivalence test verifies that the Stage-4Y run body differs only in output location, bank slice streams and sweep count.

## Successive independent construction statistics

The candidate was selected from the 2048 retained selection states by the existing strict empirical rule, without interpolation or jitter, and saved **before calibration sampling**. Independent calibration evaluates strict exceedance of this fixed candidate while continuing to target ell11. The historical selection-status and structural gates are retained in addition to calibration ESS ≥20.

| attempt | candidate threshold | selection compression | selection ESS | calibration compression | calibration ESS |
| --- | --- | --- | --- | --- | --- |
| Stage 4V: level 10 | -109780.28875081123 | 0.3676757812 | 32.20894459 | 0.4873046875 | 52.27700334 |
| Stage 4X: level 11 | -108868.60077896297 | 0.3676757812 | 54.10245944 | 0.2866210938 | 83.04742208 |
| Stage 4Z: level 12 | -108187.99712765554 | 0.3676757812 | 52.67298655 | 0.4609375 | 634.147427 |

| diagnostic | V selection | V calibration | X selection | X calibration | Z selection | Z calibration |
| --- | --- | --- | --- | --- | --- | --- |
| IID/Bernoulli SE | 0.01065460721 | 0.01104498147 | 0.01065460721 | 0.009991926486 | 0.01065460721 | 0.01101477437 |
| block-means SE | 0.03537767973 | 0.02871102186 | 0.02861827499 | 0.02680751378 | 0.02843999187 | 0.01868841158 |
| between-walker SE | 0.08495993527 | 0.06913128436 | 0.0655532029 | 0.04961935657 | 0.06643676057 | 0.0197945423 |
| walker exceedance range | 0.05078125 – 0.66796875 | 0.01953125 – 0.6171875 | 0.10546875 – 0.5625 | 0.0703125 – 0.43359375 | 0.0546875 – 0.51953125 | 0.390625 – 0.53125 |
| walker exceedance SD | 0.2403029854 | 0.1955327999 | 0.1854124572 | 0.140344734 | 0.1879115357 | 0.05598742036 |
| walker mean-logL SD | 275.8816872 | 311.4313296 | 330.5827719 | 270.3919525 | 282.4914539 | 78.07450742 |
| walker mean-logL MAD | 218.5796885 | 71.97596207 | 252.0894754 | 192.8220507 | 44.37154202 | 63.78298004 |
| walker mean-logL IQR | 486.8528254 | 126.0180293 | 510.277951 | 397.8920646 | 204.4774119 | 108.1417429 |

Stage-4Z **selection** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.4765625 | -108021.1816 |
| 1 | 0.43359375 | -107971.3803 |
| 2 | 0.4375 | -107995.5111 |
| 3 | 0.0546875 | -108606.8786 |
| 4 | 0.51953125 | -107929.2894 |
| 5 | 0.5078125 | -108000.5523 |
| 6 | 0.08203125 | -108595.4531 |
| 7 | 0.4296875 | -108060.1234 |

Stage-4Z **calibration** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.3984375 | -108220.6803 |
| 1 | 0.48828125 | -108102.4654 |
| 2 | 0.51953125 | -108107.3427 |
| 3 | 0.4921875 | -108051.2185 |
| 4 | 0.40625 | -108193.5095 |
| 5 | 0.4609375 | -108179.2811 |
| 6 | 0.390625 | -108265.1288 |
| 7 | 0.53125 | -108061.2456 |

Uncertainty and conservative Bernoulli-equivalent ESS use the unchanged existing machinery: maximum of IID, block-means and between-walker SE, block size 32. SD uses ddof=1; mean-logL MAD is the unscaled median absolute deviation of the eight walker means; IQR is their 75th–25th percentile difference. These interacting walkers are not independent convergence replications. The construction gate is the specified operational criterion, not proof of physical equilibrium.

Selection order-statistic bookkeeping:

| candidate ell_12 | zero-based order index | sample count | strict exceedances |
| --- | --- | --- | --- |
| -108187.99712765554 | 1294 | 2048 | 753 |

## Symmetric lineage and half-to-half diagnostics

Every walker has identical transition settings. Each half below contains 128 retained sweeps: absolute sweeps 129–256 and 257–384 (1-based). These are descriptive diagnostics, not an added gate.

**selection**:

| walker | first-half exceedance | second-half exceedance | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- |
| 0 | 0.578125 | 0.375 | -107851.1175 | -108191.2457 |
| 1 | 0.5 | 0.3671875 | -107878.547 | -108064.2136 |
| 2 | 0.4296875 | 0.4453125 | -107938.7912 | -108052.231 |
| 3 | 0.078125 | 0.03125 | -108595.8262 | -108617.9311 |
| 4 | 0.515625 | 0.5234375 | -107842.4877 | -108016.0911 |
| 5 | 0.625 | 0.390625 | -107885.7728 | -108115.3319 |
| 6 | 0.03125 | 0.1328125 | -108620.3689 | -108570.5373 |
| 7 | 0.5234375 | 0.3359375 | -107925.6747 | -108194.572 |

**calibration**:

| walker | first-half exceedance | second-half exceedance | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- |
| 0 | 0.34375 | 0.453125 | -108271.027 | -108170.3336 |
| 1 | 0.4296875 | 0.546875 | -108176.9033 | -108028.0275 |
| 2 | 0.453125 | 0.5859375 | -108199.2107 | -108015.4747 |
| 3 | 0.453125 | 0.53125 | -108151.9473 | -107950.4896 |
| 4 | 0.3046875 | 0.5078125 | -108287.7687 | -108099.2503 |
| 5 | 0.4296875 | 0.4921875 | -108209.3189 | -108149.2434 |
| 6 | 0.3359375 | 0.4453125 | -108354.1434 | -108176.1143 |
| 7 | 0.546875 | 0.515625 | -108042.0661 | -108080.425 |

After the run, lowest calibration exceedance was 0.390625 for walker(s) [6]; highest was 0.53125 for walker(s) [7]. These labels received no privileged probabilities or initialization rule.

## Population exchanges and exact realization communication

| attempt / bank | proposals | joint survivors | survival fraction | accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 1536 | 278 | 0.1809895833 | 255 | 0.166015625 | 23 | 0.08273381295 |
| V / calibration | 1536 | 635 | 0.4134114583 | 600 | 0.390625 | 35 | 0.05511811024 |
| X / selection | 1536 | 448 | 0.2916666667 | 431 | 0.2805989583 | 17 | 0.03794642857 |
| X / calibration | 1536 | 484 | 0.3151041667 | 469 | 0.3053385417 | 15 | 0.03099173554 |
| Z / selection | 1536 | 491 | 0.3196614583 | 480 | 0.3125 | 11 | 0.02240325866 |
| Z / calibration | 1536 | 730 | 0.4752604167 | 718 | 0.4674479167 | 12 | 0.01643835616 |

| bank | distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selection | selection_log_ratio | -3.188990666 | -0.01292940071 | 0.0001056127694 | 0.06050899823 | 5.274048485 | 0 |
| selection | joint_prior_log_ratio | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.705302566e-13 | 0 |
| calibration | selection_log_ratio | -0.3951989328 | -0.00629918844 | 3.333528021e-07 | 0.02762480545 | 3.309832587 | 0 |
| calibration | joint_prior_log_ratio | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.421085472e-13 | 0 |

| bank | nonfinite proposed priors | nonfinite proposed likelihoods | nonfinite MH ratios |
| --- | --- | --- | --- |
| selection | 0 | 0 | 0 |
| calibration | 0 | 0 | 0 |

| pair | selection accepted / proposed | selection survival fraction | selection acceptance fraction | calibration accepted / proposed | calibration survival fraction | calibration acceptance fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 32 / 54 | 0.5925925926 | 0.5925925926 | 27 / 54 | 0.5 | 0.5 |
| 0,2 | 35 / 55 | 0.6363636364 | 0.6363636364 | 26 / 55 | 0.4727272727 | 0.4727272727 |
| 0,3 | 1 / 55 | 0.01818181818 | 0.01818181818 | 10 / 55 | 0.2 | 0.1818181818 |
| 0,4 | 33 / 55 | 0.6363636364 | 0.6 | 23 / 55 | 0.4363636364 | 0.4181818182 |
| 0,5 | 34 / 55 | 0.6181818182 | 0.6181818182 | 31 / 55 | 0.5818181818 | 0.5636363636 |
| 0,6 | 0 / 55 | 0 | 0 | 31 / 55 | 0.5636363636 | 0.5636363636 |
| 0,7 | 39 / 55 | 0.7272727273 | 0.7090909091 | 21 / 55 | 0.3818181818 | 0.3818181818 |
| 1,2 | 24 / 55 | 0.4545454545 | 0.4363636364 | 36 / 55 | 0.6545454545 | 0.6545454545 |
| 1,3 | 2 / 55 | 0.03636363636 | 0.03636363636 | 17 / 55 | 0.3090909091 | 0.3090909091 |
| 1,4 | 29 / 55 | 0.5272727273 | 0.5272727273 | 32 / 55 | 0.6 | 0.5818181818 |
| 1,5 | 28 / 55 | 0.5090909091 | 0.5090909091 | 31 / 55 | 0.5818181818 | 0.5636363636 |
| 1,6 | 3 / 55 | 0.05454545455 | 0.05454545455 | 29 / 55 | 0.5272727273 | 0.5272727273 |
| 1,7 | 32 / 55 | 0.6 | 0.5818181818 | 26 / 55 | 0.4909090909 | 0.4727272727 |
| 2,3 | 0 / 55 | 0 | 0 | 18 / 55 | 0.3272727273 | 0.3272727273 |
| 2,4 | 28 / 55 | 0.5272727273 | 0.5090909091 | 25 / 55 | 0.4545454545 | 0.4545454545 |
| 2,5 | 33 / 55 | 0.6363636364 | 0.6 | 32 / 55 | 0.5818181818 | 0.5818181818 |
| 2,6 | 0 / 55 | 0 | 0 | 28 / 55 | 0.5272727273 | 0.5090909091 |
| 2,7 | 33 / 54 | 0.6111111111 | 0.6111111111 | 32 / 54 | 0.5925925926 | 0.5925925926 |
| 3,4 | 0 / 55 | 0 | 0 | 20 / 55 | 0.3818181818 | 0.3636363636 |
| 3,5 | 1 / 55 | 0.03636363636 | 0.01818181818 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 3,6 | 2 / 54 | 0.03703703704 | 0.03703703704 | 17 / 54 | 0.3148148148 | 0.3148148148 |
| 3,7 | 3 / 55 | 0.05454545455 | 0.05454545455 | 18 / 55 | 0.3454545455 | 0.3272727273 |
| 4,5 | 27 / 54 | 0.5 | 0.5 | 29 / 54 | 0.5555555556 | 0.537037037 |
| 4,6 | 1 / 55 | 0.03636363636 | 0.01818181818 | 32 / 55 | 0.5818181818 | 0.5818181818 |
| 4,7 | 24 / 55 | 0.4363636364 | 0.4363636364 | 23 / 55 | 0.4363636364 | 0.4181818182 |
| 5,6 | 0 / 55 | 0 | 0 | 31 / 55 | 0.5636363636 | 0.5636363636 |
| 5,7 | 36 / 55 | 0.6545454545 | 0.6545454545 | 29 / 55 | 0.5454545455 | 0.5272727273 |
| 6,7 | 0 / 55 | 0.01818181818 | 0 | 24 / 55 | 0.4363636364 | 0.4363636364 |

Every bank executes 54 complete seven-round cycles and the first six rounds. Each pair receives 54 or 55 fixed opportunities; every walker receives exactly 384. Pairing is never adapted from acceptance or provenance.

| attempt / bank | directional transfers | distinct migrated exact realizations | foreign-birth arrivals | first recipient visits | initial ancestries reaching another walker |
| --- | --- | --- | --- | --- | --- |
| V / selection | 510 | 390 | 500 | 497 | 48 |
| V / calibration | 1200 | 663 | 1151 | 1104 | 48 |
| X / selection | 862 | 536 | 817 | 793 | 35 |
| X / calibration | 938 | 587 | 914 | 890 | 42 |
| Z / selection | 960 | 537 | 910 | 871 | 29 |
| Z / calibration | 1436 | 749 | 1376 | 1302 | 42 |

Stage-4Z **selection** accepted realization communication:

| recipient | accepted transfers received | distinct donor walkers | donor IDs | V distinct donors | X distinct donors |
| --- | --- | --- | --- | --- | --- |
| 0 | 174 | 6 | [1, 2, 3, 4, 5, 7] | 7 | 7 |
| 1 | 150 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 2 | 153 | 5 | [0, 1, 4, 5, 7] | 7 | 7 |
| 3 | 9 | 5 | [0, 1, 5, 6, 7] | 7 | 7 |
| 4 | 142 | 6 | [0, 1, 2, 5, 6, 7] | 7 | 7 |
| 5 | 159 | 6 | [0, 1, 2, 3, 4, 7] | 7 | 7 |
| 6 | 6 | 3 | [1, 3, 4] | 7 | 7 |
| 7 | 167 | 6 | [0, 1, 2, 3, 4, 5] | 7 | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 32 | 35 | 1 | 33 | 34 | 0 | 39 |
| 1 | 32 | 0 | 24 | 2 | 29 | 28 | 3 | 32 |
| 2 | 35 | 24 | 0 | 0 | 28 | 33 | 0 | 33 |
| 3 | 1 | 2 | 0 | 0 | 0 | 1 | 2 | 3 |
| 4 | 33 | 29 | 28 | 0 | 0 | 27 | 1 | 24 |
| 5 | 34 | 28 | 33 | 1 | 27 | 0 | 0 | 36 |
| 6 | 0 | 3 | 0 | 2 | 1 | 0 | 0 | 0 |
| 7 | 39 | 32 | 33 | 3 | 24 | 36 | 0 | 0 |

Stage-4Z **calibration** accepted realization communication:

| recipient | accepted transfers received | distinct donor walkers | donor IDs | V distinct donors | X distinct donors |
| --- | --- | --- | --- | --- | --- |
| 0 | 169 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 1 | 198 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 2 | 197 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 | 7 |
| 3 | 120 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 | 7 |
| 4 | 184 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 | 7 |
| 5 | 203 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 | 7 |
| 6 | 192 | 7 | [0, 1, 2, 3, 4, 5, 7] | 6 | 7 |
| 7 | 173 | 7 | [0, 1, 2, 3, 4, 5, 6] | 6 | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 27 | 26 | 10 | 23 | 31 | 31 | 21 |
| 1 | 27 | 0 | 36 | 17 | 32 | 31 | 29 | 26 |
| 2 | 26 | 36 | 0 | 18 | 25 | 32 | 28 | 32 |
| 3 | 10 | 17 | 18 | 0 | 20 | 20 | 17 | 18 |
| 4 | 23 | 32 | 25 | 20 | 0 | 29 | 32 | 23 |
| 5 | 31 | 31 | 32 | 20 | 29 | 0 | 31 | 29 |
| 6 | 31 | 29 | 28 | 17 | 32 | 31 | 0 | 24 |
| 7 | 21 | 26 | 32 | 18 | 23 | 29 | 24 | 0 |

Provenance is exactly the Stage-4U/W scheme: initial ancestry IDs follow slice descendants; exact-realization tokens are newly minted for each slice-targeted block, including conservative invalidation of roundoff-identical updates; untouched tokens persist. Tokens and ancestry swap only on joint acceptance. Independent saved-operation replay verifies copied six-coordinate blocks, forward/reverse probabilities, joint decisions, cache outcomes and token histories **before any freeze**. Full transfers and token migration counts are in the bank provenance files and comparison.json. Communication does not by itself establish whole-catalogue convergence.

## Permutation-invariant physical geometry

The final window is exactly each bank’s retained 256 sweeps (absolute sweeps 129–384, 1-based). These are unchanged Stage-4I/U/W Hungarian-matched distances: frequency RMS in Fourier bins (DF=1/31536000 Hz), six-coordinate RMS normalized by prior widths, psi period pi and lam period 2pi. Frequency distances use all cross-time pairs; full-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. Geometry is diagnostic only and does not add a gate.

| bank | median frequency-set RMS (bins) | median sorted-frequency mean distance (bins) | median six-coordinate RMS |
| --- | --- | --- | --- |
| selection | 91.31681988 | 33.82121518 | 0.2548692085 |
| calibration | 104.9209144 | 60.90711108 | 0.2530557187 |

| pair | selection frequency-set RMS | selection sorted-frequency mean distance | selection six-coordinate RMS | calibration frequency-set RMS | calibration sorted-frequency mean distance | calibration six-coordinate RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 88.18552799 | 33.98618918 | 0.2605017501 | 95.32639879 | 28.50190531 | 0.251332743 |
| 0,2 | 83.72686633 | 18.33235058 | 0.2437505514 | 160.0182996 | 126.2827788 | 0.2556709122 |
| 0,3 | 149.5508602 | 121.4567117 | 0.2733459995 | 105.4637926 | 24.81210057 | 0.2529539441 |
| 0,4 | 91.48443814 | 32.39609059 | 0.2521426434 | 109.3374909 | 44.68429659 | 0.2531574932 |
| 0,5 | 94.20335923 | 33.65624118 | 0.2555068584 | 94.15791196 | 34.07603468 | 0.252744917 |
| 0,6 | 102.7029628 | 55.75397666 | 0.251459986 | 161.170904 | 133.5436148 | 0.2658279647 |
| 0,7 | 91.14920162 | 16.14005898 | 0.2527199245 | 122.7177644 | 81.41511105 | 0.2462423532 |
| 1,2 | 73.22575055 | 32.6415887 | 0.264095686 | 130.2088868 | 101.4861069 | 0.2602592493 |
| 1,3 | 155.3266704 | 137.4417091 | 0.2642667354 | 91.27949961 | 28.78484911 | 0.2592359874 |
| 1,4 | 75.01625815 | 26.17477957 | 0.2442390279 | 88.95068724 | 30.94860098 | 0.2511150436 |
| 1,5 | 75.51369677 | 18.39467085 | 0.2428421664 | 72.43585339 | 23.36682088 | 0.247497272 |
| 1,6 | 87.69432425 | 51.26868138 | 0.2540315319 | 128.3952118 | 106.5211664 | 0.2586243332 |
| 1,7 | 80.34060454 | 28.00916139 | 0.23863227 | 91.44037315 | 54.16367926 | 0.2443307136 |
| 2,3 | 139.5767138 | 119.1344666 | 0.2643876201 | 145.326015 | 110.600718 | 0.2576891648 |
| 2,4 | 76.25965097 | 28.64834533 | 0.2542315587 | 128.0103542 | 89.71337485 | 0.2612467092 |
| 2,5 | 77.97632536 | 25.76559834 | 0.2623143022 | 123.2920532 | 95.47896909 | 0.2585433141 |
| 2,6 | 95.39015299 | 63.12667974 | 0.2489155742 | 91.13975598 | 42.2389783 | 0.2544285534 |
| 2,7 | 78.95995587 | 22.62936673 | 0.2518280491 | 103.7985718 | 62.95040518 | 0.2456929394 |
| 3,4 | 138.7632439 | 115.415845 | 0.2642366897 | 99.52252492 | 26.79769752 | 0.2571473645 |
| 3,5 | 152.4345209 | 129.9239316 | 0.2765984835 | 85.66042676 | 20.11359384 | 0.2454110219 |
| 3,6 | 184.1015786 | 166.8516519 | 0.2755072449 | 149.3159631 | 122.1657799 | 0.2732308269 |
| 3,7 | 149.2771332 | 124.985848 | 0.2741283983 | 115.6048616 | 75.52905938 | 0.2498678165 |
| 4,5 | 80.23871507 | 19.39142768 | 0.2461818377 | 81.66976367 | 16.90470039 | 0.250614373 |
| 4,6 | 100.9107365 | 66.50481541 | 0.2580188061 | 131.9887218 | 102.9524856 | 0.255235223 |
| 4,7 | 83.48590613 | 24.52605677 | 0.2499360212 | 104.3780362 | 61.5134122 | 0.2482282714 |
| 5,6 | 100.0705118 | 61.90627181 | 0.2624257902 | 126.7873101 | 107.2691845 | 0.2569578725 |
| 5,7 | 86.9241654 | 27.67321605 | 0.2457999698 | 92.16071787 | 60.30080996 | 0.2447079727 |
| 6,7 | 96.47395553 | 53.09762819 | 0.2562791837 | 92.84115942 | 57.65798869 | 0.2499916839 |

## Cost and integrity

| attempt / bank | slice_likelihood_proxy | exchange_likelihood | slice_prior_proxy | exchange_prior | direct_cache_likelihood | direct_cache_prior | total_likelihood_proxy | total_prior_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 21468 | 3072 | 21468 | 3072 | 6152 | 6152 | 30692 | 30692 |
| V / calibration | 20951 | 3072 | 20951 | 3072 | 6152 | 6152 | 30175 | 30175 |
| X / selection | 20855 | 3072 | 20855 | 3072 | 6152 | 6152 | 30079 | 30079 |
| X / calibration | 21308 | 3072 | 21308 | 3072 | 6152 | 6152 | 30532 | 30532 |
| Z / selection | 21194 | 3072 | 21194 | 3072 | 6152 | 6152 | 30418 | 30418 |
| Z / calibration | 21051 | 3072 | 21051 | 3072 | 6152 | 6152 | 30275 | 30275 |

Slice cost retains the historical num_steps+num_shrink proxy, with prior and likelihood evaluated together. Exchange evaluations and direct cache checks, including starts, are counted separately. No cost-based truncation, optimization, added sweeps or extra walkers were used.

selection: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -912.1993017107317. CPU loop time 51.93475268 s (not a cross-hardware speed comparison).

calibration: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -917.7316874489941. CPU loop time 51.54393508 s (not a cross-hardware speed comparison).

Every start and every constituent outcome passed finite prior/likelihood, finite coordinates, strict ell11 membership and direct cache checks (prior atol 1e-9, likelihood atol 1e-7, rtol 0). Selected blocks swapped exactly, untouched coordinates were unchanged, rejection restored both post-slice states, and accepted exchanges advanced both jointly. Both banks’ exact saved-state and provenance replays passed. The complete lower prefix, historical files, prior/model/configuration, selector, fixed schedule and slice source retained their protected hashes.

**261 pre-run tests passed:** all 248 existing tests plus 13 Stage-4Z tests. Coverage includes exact lower prefix, deterministic own-bank survivor recovery and caches, missing-survivor stop, independent storage/streams, strict ell11 transition and source equivalence, selection-only candidate saved before calibration, unchanged ESS=20 boundary, no freeze on failure, exactly level 12 on success, impossible level 13, calibration-only mass, and reuse of the exact ell11 provenance replay. No real LISA trajectory runs in pytest.

Artifacts: `/tmp/lisa_dns_stage4z_population_level12/` contains preflight.json, the copied checkpoint_level11.json, each bank’s initial_states.npz, random_schedule.npz, design.json, all 384-sweep trace.npz, exchanges.npz, provenance.json and provenance_transfers.json, selection_candidate.json, report.json, comparison.json, levels.npz and validation.json, plus checkpoint_level12.json.

Exactly one attempt was made. No retry, level 13, lower-level recalibration, kernel tuning, walker-specific probabilities, production or evidence implementation occurred. Evidence remains unvalidated and out of scope. Stop here.
