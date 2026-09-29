# Stage 5B: one independent population level-13 construction attempt

**No — the authorized attempt failed the existing survivor/promotion contract.** Independent calibration passed the numerical ESS gate (417.19413243922884 ≥20; compression 0.1298828125), but selection walker 6 has zero retained states strictly above the fixed candidate. The existing promotion routine stops rather than clone another walker. No level 13 was frozen and no retry was made.

Candidate only: `ell_13_candidate = -107123.66036615883`. The numerical selection/calibration gate passed, but the full construction decision is invalidated by the terminal promotion failure. The estimated log mass in the raw decision record is a provisional calculation, not a frozen level.

The ladder remains frozen through level 12: `ell_12 = -108187.99712765554`, `log X_12 = -10.418392711998465`. No level-13 checkpoint or replacement ladder file was created.

The existing requirement comes from `run_ladder_extension.promote`: “A walker has no survivor; stop without cloning other walkers”. This is the same per-walker promotion contract used by previous stages, not a new statistical gate. It fails before save_checkpoint is called.

## Frozen setup and independent starts

The exact Stage-4Z checkpoint was loaded, including its complete level-0–12 prefix, and levels 0–9 were verified against Stage 4G. The lower checkpoint was copied byte for byte; no lower threshold or log mass was rebuilt or recalibrated. Source checkpoint SHA256: `0f0ac7c6294b4cd5e06de1fe28eb975f6fb590b9daf833f9ea084c6ed73ab0c4`.

Each bank uses only its own Stage-4Z retained history. For each walker the starting state is the **last** retained state with logL > ell12; maxima were never selected. Every position and both caches are exact copies from the indicated trace index. Banks have distinct states, separate array/cache storage, distinct random streams and recorded source-file hashes. Every starting state passed direct prior/likelihood cache checks before its first sweep.

**selection** source: `/tmp/lisa_dns_stage4z_population_level12/selection/trace.npz`, SHA256 `0c626a84d5b0ebdde836a05a1e22ffa7108ae7a18e9fc1c9207b881ce190161c`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 379 | 251 | -107505.05442259027 |
| 1 | 382 | 254 | -108162.2845493009 |
| 2 | 382 | 254 | -108137.13802681696 |
| 3 | 318 | 190 | -108138.5725454652 |
| 4 | 383 | 255 | -107950.79382602416 |
| 5 | 381 | 253 | -107593.41655313903 |
| 6 | 360 | 232 | -108051.78966563809 |
| 7 | 383 | 255 | -108016.52364961838 |

**calibration** source: `/tmp/lisa_dns_stage4z_population_level12/calibration/trace.npz`, SHA256 `de6a48ac5f546d5c4589d68baa2a162e564fec1926fc69d60e958578a6c173c0`.

| walker | absolute trace index (0-based) | retained index (0-based) | starting logL |
| --- | --- | --- | --- |
| 0 | 383 | 255 | -107961.74030917077 |
| 1 | 382 | 254 | -108095.61134281836 |
| 2 | 382 | 254 | -107615.99568137276 |
| 3 | 383 | 255 | -107854.32441057786 |
| 4 | 381 | 253 | -107938.52776163933 |
| 5 | 374 | 246 | -107896.709403001 |
| 6 | 382 | 254 | -107714.82856569697 |
| 7 | 383 | 255 | -107942.4107417032 |

Both banks targeted the fixed ell12 contour throughout all 384 sweeps, including calibration after the candidate was selected. Each has exactly 8 walkers, 128 discarded burn-in population sweeps and 256 retained population sweeps. Block size 32, target compression exp(-1), min_ess 20. Stage-5A states were not used for initialization, and its pseudo-threshold was not used for candidate selection.

The population transition is unchanged: eight validated isotropic slice updates followed by four disjoint reciprocal exchanges on the frozen seven-round schedule. Slice scales pi/sqrt(3), mixture .2/.4/.4, max_steps 10, max_shrinkage 100. Frequency selector f_star = 0.0018409808 Hz, sigma_f = 2.558e-7 Hz, stable logsumexp, no clipping. Each exchange swaps complete six-coordinate blocks, evaluates the full implemented joint-prior ratio, recomputes reverse label probabilities, and requires both proposals strictly above ell12. One joint MH decision advances both or neither. No source_gain enters the kernel.

Frozen slice seeds: `{'selection': [950100, 950101, 950102, 950103, 950104, 950105, 950106, 950107], 'calibration': [950200, 950201, 950202, 950203, 950204, 950205, 950206, 950207]}`; independent PCG64 label/MH seeds: `{'selection': [2026092919, 2026092920], 'calibration': [2026092921, 2026092922]}`. Both banks’ draws, designs and starts were saved before sampling. A source-equivalence test verifies that the Stage-5A run body differs only in output location, bank slice streams and sweep count.

## Successive independent construction statistics

The candidate was selected from the 2048 retained selection states by the existing strict empirical rule, without interpolation or jitter, and saved **before calibration sampling**. Independent calibration evaluates strict exceedance of this fixed candidate while continuing to target ell12. The historical selection-status and structural gates are retained in addition to calibration ESS ≥20.

| attempt | candidate threshold | selection compression | selection ESS | calibration compression | calibration ESS |
| --- | --- | --- | --- | --- | --- |
| Stage 4V: level 10 | -109780.28875081123 | 0.3676757812 | 32.20894459 | 0.4873046875 | 52.27700334 |
| Stage 4X: level 11 | -108868.60077896297 | 0.3676757812 | 54.10245944 | 0.2866210938 | 83.04742208 |
| Stage 4Z: level 12 | -108187.99712765554 | 0.3676757812 | 52.67298655 | 0.4609375 | 634.147427 |
| Stage 5B: level 13 | -107123.66036615883 | 0.3676757812 | 35.58905417 | 0.1298828125 | 417.1941324 |

| diagnostic | V selection | V calibration | X selection | X calibration | Z selection | Z calibration | 5B selection | 5B calibration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID/Bernoulli SE | 0.01065460721 | 0.01104498147 | 0.01065460721 | 0.009991926486 | 0.01065460721 | 0.01101477437 | 0.01065460721 | 0.007428476259 |
| block-means SE | 0.03537767973 | 0.02871102186 | 0.02861827499 | 0.02680751378 | 0.02843999187 | 0.01868841158 | 0.03104582191 | 0.01645870344 |
| between-walker SE | 0.08495993527 | 0.06913128436 | 0.0655532029 | 0.04961935657 | 0.06643676057 | 0.0197945423 | 0.0808247178 | 0.01332365048 |
| walker exceedance range | 0.05078125 – 0.66796875 | 0.01953125 – 0.6171875 | 0.10546875 – 0.5625 | 0.0703125 – 0.43359375 | 0.0546875 – 0.51953125 | 0.390625 – 0.53125 | 0 – 0.5390625 | 0.07421875 – 0.1796875 |
| walker exceedance SD | 0.2403029854 | 0.1955327999 | 0.1854124572 | 0.140344734 | 0.1879115357 | 0.05598742036 | 0.2286068242 | 0.03768497441 |
| walker mean-logL SD | 275.8816872 | 311.4313296 | 330.5827719 | 270.3919525 | 282.4914539 | 78.07450742 | 432.0439562 | 54.4834341 |
| walker mean-logL MAD | 218.5796885 | 71.97596207 | 252.0894754 | 192.8220507 | 44.37154202 | 63.78298004 | 105.6981434 | 46.03588611 |
| walker mean-logL IQR | 486.8528254 | 126.0180293 | 510.277951 | 397.8920646 | 204.4774119 | 108.1417429 | 339.1289785 | 71.8451209 |

Stage-5B **selection** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.5234375 | -107053.8601 |
| 1 | 0.5390625 | -106961.965 |
| 2 | 0.51171875 | -106954.8934 |
| 3 | 0.00390625 | -107877.1464 |
| 4 | 0.4765625 | -107009.8285 |
| 5 | 0.42578125 | -107129.3297 |
| 6 | 0 | -108042.6198 |
| 7 | 0.4609375 | -107156.94 |

Stage-5B **calibration** retained per-walker summaries:

| walker | candidate exceedance fraction | mean logL |
| --- | --- | --- |
| 0 | 0.14453125 | -107663.9574 |
| 1 | 0.1796875 | -107636.1657 |
| 2 | 0.1328125 | -107663.694 |
| 3 | 0.11328125 | -107729.9157 |
| 4 | 0.12109375 | -107667.8265 |
| 5 | 0.07421875 | -107749.7429 |
| 6 | 0.09375 | -107728.2375 |
| 7 | 0.1796875 | -107587.6736 |

Uncertainty and conservative Bernoulli-equivalent ESS use the unchanged existing machinery: maximum of IID, block-means and between-walker SE, block size 32. SD uses ddof=1; mean-logL MAD is the unscaled median absolute deviation of the eight walker means; IQR is their 75th–25th percentile difference. These interacting walkers are not independent convergence replications. The construction gate is the specified operational criterion, not proof of physical equilibrium.

Candidate SHA256 saved before calibration and verified afterward: `c5fce9bfaef12306b0d215f157dc667661a4cd7e782c0ecf22fa2d840c2fbd60`.

Selection order-statistic bookkeeping:

| candidate ell_13 | zero-based order index | sample count | strict exceedances |
| --- | --- | --- | --- |
| -107123.66036615883 | 1294 | 2048 | 753 |

| bank | walker 0 candidate survivors | walker 1 candidate survivors | walker 2 candidate survivors | walker 3 candidate survivors | walker 4 candidate survivors | walker 5 candidate survivors | walker 6 candidate survivors | walker 7 candidate survivors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 134 | 138 | 131 | 1 | 122 | 109 | 0 | 118 |
| calibration | 37 | 46 | 34 | 29 | 31 | 19 | 24 | 46 |

## Symmetric lineage and half-to-half diagnostics

Every walker has identical transition settings. Each half below contains 128 retained sweeps: absolute sweeps 129–256 and 257–384 (1-based). These are descriptive diagnostics, not an added gate.

**selection**:

| walker | first-half exceedance | second-half exceedance | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- |
| 0 | 0.4296875 | 0.6171875 | -107159.3771 | -106948.3431 |
| 1 | 0.421875 | 0.65625 | -107137.8314 | -106786.0987 |
| 2 | 0.453125 | 0.5703125 | -107067.1484 | -106842.6385 |
| 3 | 0.0078125 | 0 | -107908.6325 | -107845.6603 |
| 4 | 0.3671875 | 0.5859375 | -107213.5408 | -106806.1161 |
| 5 | 0.3359375 | 0.515625 | -107344.0036 | -106914.6558 |
| 6 | 0 | 0 | -108016.0083 | -108069.2314 |
| 7 | 0.3984375 | 0.5234375 | -107246.8332 | -107067.0467 |

**calibration**:

| walker | first-half exceedance | second-half exceedance | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- |
| 0 | 0.0546875 | 0.234375 | -107804.9721 | -107522.9427 |
| 1 | 0.0703125 | 0.2890625 | -107841.1011 | -107431.2303 |
| 2 | 0.078125 | 0.1875 | -107782.7637 | -107544.6243 |
| 3 | 0.0625 | 0.1640625 | -107774.6905 | -107685.1408 |
| 4 | 0.0859375 | 0.15625 | -107751.8433 | -107583.8097 |
| 5 | 0.03125 | 0.1171875 | -107814.2302 | -107685.2556 |
| 6 | 0.015625 | 0.171875 | -107826.7602 | -107629.7147 |
| 7 | 0.1484375 | 0.2109375 | -107750.2025 | -107425.1448 |

After the run, lowest selection exceedance was 0 for walker(s) [6]; highest was 0.5390625 for walker(s) [1]. These labels received no privileged probabilities or initialization rule.

After the run, lowest calibration exceedance was 0.07421875 for walker(s) [5]; highest was 0.1796875 for walker(s) [1, 7]. These labels received no privileged probabilities or initialization rule.

Selection exhibits persistent low-tail behavior in lineages 3 and 6, identified only after sampling. Walker 3 has fractions 0.0078125 then 0, and walker 6 has 0 in both halves. The other six walkers increase from 0.3359375–0.453125 to 0.515625–0.65625. Those two lineages receive only 21 and 18 accepted transfers over the full bank, respectively, so provenance does not establish their recovery. Calibration is more homogeneous across walkers (full fractions 0.07421875–0.1796875), with all walkers receiving transfers from all seven donors, but every walker’s exceedance fraction rises in the second half. The pooled calibration fractions rise from 0.068359375 to 0.19140625. Thus controlled nominal between-walker uncertainty in calibration does not erase the selection separation or demonstrate temporal equilibrium.

## Population exchanges and exact realization communication

| attempt / bank | proposals | joint survivors | survival fraction | accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 1536 | 278 | 0.1809895833 | 255 | 0.166015625 | 23 | 0.08273381295 |
| V / calibration | 1536 | 635 | 0.4134114583 | 600 | 0.390625 | 35 | 0.05511811024 |
| X / selection | 1536 | 448 | 0.2916666667 | 431 | 0.2805989583 | 17 | 0.03794642857 |
| X / calibration | 1536 | 484 | 0.3151041667 | 469 | 0.3053385417 | 15 | 0.03099173554 |
| Z / selection | 1536 | 491 | 0.3196614583 | 480 | 0.3125 | 11 | 0.02240325866 |
| Z / calibration | 1536 | 730 | 0.4752604167 | 718 | 0.4674479167 | 12 | 0.01643835616 |
| 5B / selection | 1536 | 438 | 0.28515625 | 433 | 0.2819010417 | 5 | 0.01141552511 |
| 5B / calibration | 1536 | 648 | 0.421875 | 648 | 0.421875 | 0 | 0 |

| bank | distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selection | selection_log_ratio | -0.4069328139 | -0.01575208891 | 1.03850226e-06 | 0.02788514901 | 4.005347564 | 0 |
| selection | joint_prior_log_ratio | -1.705302566e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.136868377e-13 | 0 |
| calibration | selection_log_ratio | -0.3138960762 | -0.000349522047 | 4.605417438e-08 | 0.005553886472 | 4.439056514 | 0 |
| calibration | joint_prior_log_ratio | -1.705302566e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.421085472e-13 | 0 |

| bank | nonfinite proposed priors | nonfinite proposed likelihoods | nonfinite MH ratios |
| --- | --- | --- | --- |
| selection | 0 | 0 | 0 |
| calibration | 0 | 0 | 0 |

| pair | selection accepted / proposed | selection survival fraction | selection acceptance fraction | calibration accepted / proposed | calibration survival fraction | calibration acceptance fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 27 / 54 | 0.5 | 0.5 | 28 / 54 | 0.5185185185 | 0.5185185185 |
| 0,2 | 26 / 55 | 0.4727272727 | 0.4727272727 | 23 / 55 | 0.4181818182 | 0.4181818182 |
| 0,3 | 2 / 55 | 0.03636363636 | 0.03636363636 | 26 / 55 | 0.4727272727 | 0.4727272727 |
| 0,4 | 31 / 55 | 0.5636363636 | 0.5636363636 | 18 / 55 | 0.3272727273 | 0.3272727273 |
| 0,5 | 23 / 55 | 0.4363636364 | 0.4181818182 | 24 / 55 | 0.4363636364 | 0.4363636364 |
| 0,6 | 0 / 55 | 0 | 0 | 29 / 55 | 0.5272727273 | 0.5272727273 |
| 0,7 | 27 / 55 | 0.4909090909 | 0.4909090909 | 29 / 55 | 0.5272727273 | 0.5272727273 |
| 1,2 | 33 / 55 | 0.6 | 0.6 | 19 / 55 | 0.3454545455 | 0.3454545455 |
| 1,3 | 1 / 55 | 0.01818181818 | 0.01818181818 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 1,4 | 33 / 55 | 0.6 | 0.6 | 22 / 55 | 0.4 | 0.4 |
| 1,5 | 26 / 55 | 0.4727272727 | 0.4727272727 | 22 / 55 | 0.4 | 0.4 |
| 1,6 | 1 / 55 | 0.01818181818 | 0.01818181818 | 27 / 55 | 0.4909090909 | 0.4909090909 |
| 1,7 | 18 / 55 | 0.3454545455 | 0.3272727273 | 27 / 55 | 0.4909090909 | 0.4909090909 |
| 2,3 | 1 / 55 | 0.01818181818 | 0.01818181818 | 24 / 55 | 0.4363636364 | 0.4363636364 |
| 2,4 | 29 / 55 | 0.5272727273 | 0.5272727273 | 26 / 55 | 0.4727272727 | 0.4727272727 |
| 2,5 | 26 / 55 | 0.4727272727 | 0.4727272727 | 19 / 55 | 0.3454545455 | 0.3454545455 |
| 2,6 | 0 / 55 | 0 | 0 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 2,7 | 32 / 54 | 0.5925925926 | 0.5925925926 | 21 / 54 | 0.3888888889 | 0.3888888889 |
| 3,4 | 0 / 55 | 0 | 0 | 23 / 55 | 0.4181818182 | 0.4181818182 |
| 3,5 | 1 / 55 | 0.01818181818 | 0.01818181818 | 19 / 55 | 0.3454545455 | 0.3454545455 |
| 3,6 | 15 / 54 | 0.3148148148 | 0.2777777778 | 22 / 54 | 0.4074074074 | 0.4074074074 |
| 3,7 | 1 / 55 | 0.01818181818 | 0.01818181818 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 4,5 | 29 / 54 | 0.537037037 | 0.537037037 | 25 / 54 | 0.462962963 | 0.462962963 |
| 4,6 | 0 / 55 | 0 | 0 | 27 / 55 | 0.4909090909 | 0.4909090909 |
| 4,7 | 23 / 55 | 0.4363636364 | 0.4181818182 | 25 / 55 | 0.4545454545 | 0.4545454545 |
| 5,6 | 1 / 55 | 0.01818181818 | 0.01818181818 | 20 / 55 | 0.3636363636 | 0.3636363636 |
| 5,7 | 26 / 55 | 0.4727272727 | 0.4727272727 | 19 / 55 | 0.3454545455 | 0.3454545455 |
| 6,7 | 1 / 55 | 0.01818181818 | 0.01818181818 | 24 / 55 | 0.4363636364 | 0.4363636364 |

Every bank executes 54 complete seven-round cycles and the first six rounds. Each pair receives 54 or 55 fixed opportunities; every walker receives exactly 384. Pairing is never adapted from acceptance or provenance.

| attempt / bank | directional transfers | distinct migrated exact realizations | foreign-birth arrivals | first recipient visits | initial ancestries reaching another walker |
| --- | --- | --- | --- | --- | --- |
| V / selection | 510 | 390 | 500 | 497 | 48 |
| V / calibration | 1200 | 663 | 1151 | 1104 | 48 |
| X / selection | 862 | 536 | 817 | 793 | 35 |
| X / calibration | 938 | 587 | 914 | 890 | 42 |
| Z / selection | 960 | 537 | 910 | 871 | 29 |
| Z / calibration | 1436 | 749 | 1376 | 1302 | 42 |
| 5B / selection | 866 | 534 | 834 | 816 | 40 |
| 5B / calibration | 1296 | 672 | 1226 | 1162 | 29 |

Stage-5B **selection** accepted realization communication:

| recipient | accepted transfers received | distinct donor walkers | donor IDs | V distinct donors | X distinct donors |
| --- | --- | --- | --- | --- | --- |
| 0 | 136 | 6 | [1, 2, 3, 4, 5, 7] | 7 | 7 |
| 1 | 139 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 2 | 147 | 6 | [0, 1, 3, 4, 5, 7] | 7 | 7 |
| 3 | 21 | 6 | [0, 1, 2, 5, 6, 7] | 7 | 7 |
| 4 | 145 | 5 | [0, 1, 2, 5, 7] | 7 | 7 |
| 5 | 132 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 | 7 |
| 6 | 18 | 4 | [1, 3, 5, 7] | 7 | 7 |
| 7 | 128 | 7 | [0, 1, 2, 3, 4, 5, 6] | 7 | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 27 | 26 | 2 | 31 | 23 | 0 | 27 |
| 1 | 27 | 0 | 33 | 1 | 33 | 26 | 1 | 18 |
| 2 | 26 | 33 | 0 | 1 | 29 | 26 | 0 | 32 |
| 3 | 2 | 1 | 1 | 0 | 0 | 1 | 15 | 1 |
| 4 | 31 | 33 | 29 | 0 | 0 | 29 | 0 | 23 |
| 5 | 23 | 26 | 26 | 1 | 29 | 0 | 1 | 26 |
| 6 | 0 | 1 | 0 | 15 | 0 | 1 | 0 | 1 |
| 7 | 27 | 18 | 32 | 1 | 23 | 26 | 1 | 0 |

Stage-5B **calibration** accepted realization communication:

| recipient | accepted transfers received | distinct donor walkers | donor IDs | V distinct donors | X distinct donors |
| --- | --- | --- | --- | --- | --- |
| 0 | 177 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 1 | 165 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 | 7 |
| 2 | 152 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 | 7 |
| 3 | 154 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 | 7 |
| 4 | 166 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 | 7 |
| 5 | 148 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 | 7 |
| 6 | 169 | 7 | [0, 1, 2, 3, 4, 5, 7] | 6 | 7 |
| 7 | 165 | 7 | [0, 1, 2, 3, 4, 5, 6] | 6 | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 28 | 23 | 26 | 18 | 24 | 29 | 29 |
| 1 | 28 | 0 | 19 | 20 | 22 | 22 | 27 | 27 |
| 2 | 23 | 19 | 0 | 24 | 26 | 19 | 20 | 21 |
| 3 | 26 | 20 | 24 | 0 | 23 | 19 | 22 | 20 |
| 4 | 18 | 22 | 26 | 23 | 0 | 25 | 27 | 25 |
| 5 | 24 | 22 | 19 | 19 | 25 | 0 | 20 | 19 |
| 6 | 29 | 27 | 20 | 22 | 27 | 20 | 0 | 24 |
| 7 | 29 | 27 | 21 | 20 | 25 | 19 | 24 | 0 |

Provenance is exactly the Stage-4U/W scheme: initial ancestry IDs follow slice descendants; exact-realization tokens are newly minted for each slice-targeted block, including conservative invalidation of roundoff-identical updates; untouched tokens persist. Tokens and ancestry swap only on joint acceptance. Independent saved-operation replay verifies copied six-coordinate blocks, forward/reverse probabilities, joint decisions, cache outcomes and token histories **before any freeze**. Full transfers and token migration counts are in the bank provenance files and comparison.json. Communication does not by itself establish whole-catalogue convergence.

## Permutation-invariant physical geometry

The final window is exactly each bank’s retained 256 sweeps (absolute sweeps 129–384, 1-based). These are unchanged Stage-4I/U/W Hungarian-matched distances: frequency RMS in Fourier bins (DF=1/31536000 Hz), six-coordinate RMS normalized by prior widths, psi period pi and lam period 2pi. Frequency distances use all cross-time pairs; full-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. Geometry is diagnostic only and does not add a gate.

| bank | median frequency-set RMS (bins) | median sorted-frequency mean distance (bins) | median six-coordinate RMS |
| --- | --- | --- | --- |
| selection | 94.26622186 | 33.11483555 | 0.2515979041 |
| calibration | 105.3386093 | 50.1829384 | 0.2564770172 |

| pair | selection frequency-set RMS | selection sorted-frequency mean distance | selection six-coordinate RMS | calibration frequency-set RMS | calibration sorted-frequency mean distance | calibration six-coordinate RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 90.71222632 | 43.35426303 | 0.2460552711 | 100.6998044 | 32.67932445 | 0.264043406 |
| 0,2 | 84.77152009 | 33.26333026 | 0.2515267262 | 105.1282699 | 17.95788191 | 0.2529134241 |
| 0,3 | 107.7140978 | 47.92699904 | 0.2650392314 | 106.7442187 | 53.5094928 | 0.2477048985 |
| 0,4 | 82.21262691 | 28.90320665 | 0.2483554274 | 175.495971 | 133.0011535 | 0.2605454827 |
| 0,5 | 74.65396376 | 16.53730856 | 0.2399571715 | 114.3868314 | 58.87793061 | 0.2473350582 |
| 0,6 | 91.81002622 | 9.060670311 | 0.2469239355 | 104.4231415 | 24.0489375 | 0.2542591283 |
| 0,7 | 90.83181998 | 17.0695938 | 0.2510423343 | 108.4258472 | 39.4084086 | 0.2620906446 |
| 1,2 | 90.17766318 | 36.03903903 | 0.2655808956 | 94.28937167 | 27.7223847 | 0.2613827636 |
| 1,3 | 103.4902722 | 25.52390424 | 0.2821405653 | 102.6528176 | 67.92680748 | 0.268835608 |
| 1,4 | 84.97491416 | 23.31905323 | 0.2589147538 | 161.3903542 | 124.6789078 | 0.2727580225 |
| 1,5 | 91.22720948 | 47.636487 | 0.255906459 | 101.9723962 | 58.20778054 | 0.2417986617 |
| 1,6 | 103.2936495 | 39.61257031 | 0.2642709262 | 92.48851418 | 28.91015019 | 0.2637089899 |
| 1,7 | 107.7703269 | 53.8698223 | 0.2515028406 | 106.8240258 | 61.77170426 | 0.2579900155 |
| 2,3 | 112.6042527 | 53.76156042 | 0.251669082 | 100.7182335 | 50.63257369 | 0.247499644 |
| 2,4 | 83.96698717 | 25.35451726 | 0.2487454287 | 174.9432329 | 135.8166525 | 0.2614840129 |
| 2,5 | 81.14193736 | 28.28779853 | 0.2475748665 | 105.5489487 | 49.7333031 | 0.2533943512 |
| 2,6 | 99.07108235 | 31.36895709 | 0.2589239666 | 97.82267085 | 13.92807268 | 0.2537531878 |
| 2,7 | 102.8696321 | 46.24964864 | 0.2566336484 | 106.3312066 | 45.31822154 | 0.2540551661 |
| 3,4 | 103.537197 | 32.96634085 | 0.2519536567 | 206.2257716 | 180.4674554 | 0.2745133006 |
| 3,5 | 110.3000626 | 56.18470613 | 0.2510337906 | 92.44972489 | 45.5870345 | 0.2440584027 |
| 3,6 | 118.5596022 | 44.71848885 | 0.2510547186 | 96.70959373 | 46.64618958 | 0.254964019 |
| 3,7 | 121.0397089 | 54.60688388 | 0.2572786185 | 90.04687358 | 32.46937555 | 0.2532761718 |
| 4,5 | 82.8350703 | 35.12940691 | 0.2426481283 | 192.3316425 | 161.0720962 | 0.2615236888 |
| 4,6 | 96.42192893 | 24.94717015 | 0.2578433259 | 178.179282 | 141.3948777 | 0.2662163705 |
| 4,7 | 98.85954236 | 38.55280424 | 0.2591593667 | 197.511239 | 165.4631193 | 0.2743227834 |
| 5,6 | 92.11051479 | 20.81607054 | 0.2416300661 | 99.05907212 | 39.35920693 | 0.2503046364 |
| 5,7 | 91.89061499 | 27.94250633 | 0.2500549402 | 108.3717038 | 60.02976766 | 0.2391153511 |
| 6,7 | 105.2166208 | 18.97847209 | 0.2609034599 | 104.0435198 | 44.45330391 | 0.2614201152 |

## Cost and integrity

| attempt / bank | slice_likelihood_proxy | exchange_likelihood | slice_prior_proxy | exchange_prior | direct_cache_likelihood | direct_cache_prior | total_likelihood_proxy | total_prior_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V / selection | 21468 | 3072 | 21468 | 3072 | 6152 | 6152 | 30692 | 30692 |
| V / calibration | 20951 | 3072 | 20951 | 3072 | 6152 | 6152 | 30175 | 30175 |
| X / selection | 20855 | 3072 | 20855 | 3072 | 6152 | 6152 | 30079 | 30079 |
| X / calibration | 21308 | 3072 | 21308 | 3072 | 6152 | 6152 | 30532 | 30532 |
| Z / selection | 21194 | 3072 | 21194 | 3072 | 6152 | 6152 | 30418 | 30418 |
| Z / calibration | 21051 | 3072 | 21051 | 3072 | 6152 | 6152 | 30275 | 30275 |
| 5B / selection | 21956 | 3072 | 21956 | 3072 | 6152 | 6152 | 31180 | 31180 |
| 5B / calibration | 21686 | 3072 | 21686 | 3072 | 6152 | 6152 | 30910 | 30910 |

Slice cost retains the historical num_steps+num_shrink proxy, with prior and likelihood evaluated together. Exchange evaluations and direct cache checks, including starts, are counted separately. No cost-based truncation, optimization, added sweeps or extra walkers were used.

selection: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -907.9054852443512. CPU loop time 53.19022106 s (not a cross-hardware speed comparison).

calibration: 384 sweeps, 0 structural failures; maximum prior-cache error 0.0, maximum logL-cache error 0.0. Minimum selector log probability -916.0063043354958. CPU loop time 52.19812154 s (not a cross-hardware speed comparison).

Every start and every constituent outcome passed finite prior/likelihood, finite coordinates, strict ell12 membership and direct cache checks (prior atol 1e-9, likelihood atol 1e-7, rtol 0). Selected blocks swapped exactly, untouched coordinates were unchanged, rejection restored both post-slice states, and accepted exchanges advanced both jointly. Both banks’ exact saved-state and provenance replays passed before the attempted promotion; maximum cache discrepancies were also required to be exactly zero at that audit. The complete lower prefix, historical files, prior/model/configuration, selector, fixed schedule and slice source retained their protected hashes.

**294 pre-run tests passed:** all 280 existing tests plus 14 Stage-5B tests. Coverage includes exact lower prefix, deterministic own-bank survivor recovery and caches, missing-survivor stop, independent storage/streams, strict ell12 transition and source equivalence, selection-only candidate saved before calibration, candidate-hash tamper rejection, unchanged ESS=20 boundary, no freeze on failure, exactly level 13 on success, impossible level 14, calibration-only mass, and reuse of the exact ell12 provenance replay. No real LISA trajectory runs in pytest.

One terminal structural/promotion failure was recorded. Both sampling banks completed all 384 sweeps with zero transition structural failures and exact cache equality. Read-only failure diagnostics preserve the failed-run report and all protected source hashes; no sampler or promotion path was rerun.

Artifacts: `/tmp/lisa_dns_stage5b_population_level13/` contains preflight.json, the copied checkpoint_level12.json, each bank’s initial_states.npz, random_schedule.npz, design.json, all 384-sweep trace.npz, exchanges.npz, provenance.json and provenance_transfers.json, selection_candidate.json, report.json, comparison.json, interpretation.json and validation.json. No checkpoint_level13.json exists.

No: this single attempt does not support an admissible level-13 construction under all unchanged rules. Calibration ESS 417.19413243922884 passes, and its between-walker SE 0.013323650477238604 is smaller than its block-means SE 0.016458703439884353. However, selection between-walker SE is 0.08082471780084231, its retained mean-logL SD is 432.04395620494734, and walker 6 has no candidate survivor. The existing promotion contract therefore prevents any freeze. The ladder stays at ell_12 = -108187.99712765554 and log X_12 = -10.418392711998465. No retry, alternate promotion, budget increase or kernel change is authorized or performed. Whole-catalogue physical convergence is not established; level-13 validation is not run because level 13 does not exist. Stop ladder growth here.

Exactly one attempt was made. No retry, level 14, lower-level recalibration, kernel tuning, walker-specific probabilities, production or evidence implementation occurred. Evidence remains unvalidated and out of scope. Stop here.
