# Stage 6B: final pooled-promotion LISA validation

**PASS.** Both banks completed all 384 sweeps and passed structural, numerical, independence, and exact replay checks. The severe Stage-5B selection separation did not persist: selection full-window median/max KS were 0.09180/0.13672 (half-window maxima 0.25000 and 0.30469), and calibration median/max were 0.09180/0.16797 (half-window maxima 0.31250 and 0.18750). Every retained state was unique within its walker. Cross-bank pooled KS was 0.10498 overall (0.09375 and 0.13867 by half), with comparable six-coordinate geometry: within-bank medians 0.25696 and 0.25911, cross-bank 0.26016. Calibration mean logL increased by about 249.25 between retained halves, versus 18.88 for selection; the second-half cross-bank mean gap was about 283.94. This residual drift is recorded, but the broad distributions overlap and neither bank shows severe persistent cross-walker separation. The finite-run evidence supports qualitatively compatible, usable populations under the stated validation criterion, without establishing full physical convergence.

Exactly one authorized Stage-6B attempt was made. No retry, second seed, burn-in extension, threshold adaptation, tuning, production, or evidence implementation occurred. Stage 5B remains officially failed.

The ladder remains frozen at levels 0–12: `ell_12 = -108187.99712765554`, `log X_12 = -10.418392711998465`. The Stage-5B candidate `ell_test = -107123.66036615883` was used only as the fixed validation contour, never as level 13. No level-13 checkpoint or mass was created.

## CPU environment and numerical contract

The [Stage-6A contract](dns_lisa_numerical_validation_contract.md) was unchanged: `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, `float64`; `rtol=0`, prior `atol=1e-9`, likelihood `atol=1e-7`. Copied coordinates and scalar caches must be byte-exact from their source. Recomputed scalars use numerical tolerance; finite coordinates/scalars and strict cached **and** recomputed `logL > ell_test` are required.

| Environment field | Value |
| --- | --- |
| backend | cpu |
| devices | [CpuDevice(id=0)] |
| x64 | True |
| dtype | float64 |
| jax_version | 0.10.0 |
| jaxlib_version | 0.10.0 |
| numpy_version | 2.4.4 |
| executable | /r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python |
| config_path | /r5/home/rruiz/projects/sbi/BayesLISAx/automation/snakemake_lisa_gb/results/model_order/gb_0004/configs/k9.json |
| config_sha256 | bfafd9e8784bca263bd7298e10491d0010d27483222358b84bd1b4a877536670 |
| compilation_seconds | 2.429563936 |
| environment | {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"} |

## Pools, seeds, and promotion

Only Stage-5B retained histories were used: eight walkers × 256 states per bank, absolute trace indices 128–383. No Stage-5A states, cross-bank rescue, or newly sampled ell_12 states were used. Pool order is Stage-3 time-major: `flat = retained_index * 8 + donor`. The generic `dns_levels._survivors` samples eight starts uniformly with replacement in each bank independently. Seeds and protected hashes were persisted before drawing. No donor-diversity optimization or uniqueness constraint was applied.

| Bank | Retained | Eligible | Fraction | Donors | Counts per donor 0–7 |
| --- | --- | --- | --- | --- | --- |
| selection | 2048 | 753 | 0.3676757812 | [0, 1, 2, 3, 4, 5, 7] | [134, 138, 131, 1, 122, 109, 0, 118] |
| calibration | 2048 | 266 | 0.1298828125 | [0, 1, 2, 3, 4, 5, 6, 7] | [37, 46, 34, 29, 31, 19, 24, 46] |

Selection donor 6 has zero eligible retained states; this did not block pooled promotion.

| Bank | Promotion seed | Slice seeds | Label seed / MH seed |
| --- | --- | --- | --- |
| selection | 2026093011 | [970100, 970101, 970102, 970103, 970104, 970105, 970106, 970107] | [2026093013, 2026093014] |
| calibration | 2026093012 | [970200, 970201, 970202, 970203, 970204, 970205, 970206, 970207] | [2026093015, 2026093016] |

### Selection promotion

Donor multiplicities: `[2, 4, 0, 0, 1, 1, 0, 0]`. Unique promoted positions: **8**; duplicates: **0**. Duplicate source hashes: `{}`.

| Recipient | Eligible-pool index | Flat index | Retained index | Absolute index | Donor | Starting logL |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 312 | 1032 | 129 | 257 | 0 | -106605.5583 |
| 1 | 744 | 2032 | 254 | 382 | 0 | -107065.0972 |
| 2 | 205 | 693 | 86 | 214 | 5 | -106979.4858 |
| 3 | 586 | 1689 | 211 | 339 | 1 | -106269.3469 |
| 4 | 61 | 257 | 32 | 160 | 1 | -105282.7552 |
| 5 | 220 | 729 | 91 | 219 | 1 | -105756.204 |
| 6 | 660 | 1836 | 229 | 357 | 4 | -105411.4812 |
| 7 | 413 | 1297 | 162 | 290 | 1 | -106535.6228 |

Source state SHA-256 values (position bytes, prior scalar bytes, likelihood scalar bytes):

| Recipient | SHA-256 |
| --- | --- |
| 0 | a7313db0a006e2f3ea6dc67b9723444394f1c518fabf3138f1b1337126042d67 |
| 1 | 0614dfb3ea6da511e7d4d4ff0ffb36ec3b7a8907e6ae56fa188579ddc24787fd |
| 2 | 2e17797d66c537b35bd3fcec0aab63ba0dc7d5bb117b2f6f2ac39a65307ce121 |
| 3 | ab6a56418d6ec05cdb1e5a0d69611c4061ff349b8d62a214f17dd2b1d1725495 |
| 4 | c8bad2957ced5cb75d53e023f3418f3c59237e088380699056a227489fa6e0bf |
| 5 | e8b519eff5eff1248f71b51e158bc609a14133cbf895d6c66faa14e5ede5c377 |
| 6 | 586392bf2a02da17e8ace246cc9bbfb5eee2f34cf9aa868b30f336918748cd6f |
| 7 | 2a7ed5ca194715a7c87bc70503f5df4cd0793eeaf82be696d24eb7ef0efa5772 |

### Calibration promotion

Donor multiplicities: `[2, 0, 2, 0, 2, 0, 1, 1]`. Unique promoted positions: **8**; duplicates: **0**. Duplicate source hashes: `{}`.

| Recipient | Eligible-pool index | Flat index | Retained index | Absolute index | Donor | Starting logL |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 120 | 15 | 143 | 0 | -106986.3795 |
| 1 | 164 | 1526 | 190 | 318 | 6 | -107101.9511 |
| 2 | 186 | 1656 | 207 | 335 | 0 | -106474.7064 |
| 3 | 82 | 1068 | 133 | 261 | 4 | -106265.0059 |
| 4 | 122 | 1274 | 159 | 287 | 2 | -105545.564 |
| 5 | 21 | 298 | 37 | 165 | 2 | -106529.544 |
| 6 | 242 | 1935 | 241 | 369 | 7 | -104813.8333 |
| 7 | 209 | 1772 | 221 | 349 | 4 | -105997.9053 |

Source state SHA-256 values (position bytes, prior scalar bytes, likelihood scalar bytes):

| Recipient | SHA-256 |
| --- | --- |
| 0 | 3ed56392d7ca94d447beb4c5c541639a1876926278e0b838a24546d4b9898e13 |
| 1 | a51c3b231853260236a58f7d279d98b4608ed0569272b80f4e2ddbdbea21efb4 |
| 2 | e4f83500d2fc704ed3572fe399ce3967285e2fb6f36d4c08aff99e73ac033027 |
| 3 | ceb3b01c79868fab948800f0f26afbafeb314a7a6b52cd443483b868f97ef08d |
| 4 | 4a7d6477417cefab91e9050b673464d9996cdf66035cf5b86ae90078af46b43b |
| 5 | c2f858ae9c9b14d040fb7054d8d737179edf59a29267aa1634889673eb4f40ed |
| 6 | cfc2bd67cee40cf6b15dbfe5d860f61b33414ad0ee3d90b7630b4a1be9c81c43 |
| 7 | e0cc9b47995a2b1494bd7947a432f7b89690236b3ac22f76c7158e24daa95b53 |

## Integrity, independence, and frozen transition

All 16 starts passed the pre-sampling gate before either bank began sweep 1: correct source bank and eligible-pool membership, byte-exact copied coordinates/caches, finite values, strict cached/recomputed support, and frozen numerical tolerances. Separate array storage and 22 distinct seeds were checked. Initial slice-key hashes and independent PCG64 label/MH schedule hashes were saved in `integrity_gate.json` before sweep 1.

| Bank | Initial max prior error | Initial max likelihood error |
| --- | --- | --- |
| selection | 0 | 0 |
| calibration | 0 | 0 |

Both banks target exactly `pi_impl(u) * I(logL(u) > ell_test)`. Each sweep uses eight validated constrained-slice updates, followed by four disjoint reciprocal frequency-guided source exchanges. The frozen kernel uses scales `pi/sqrt(3)`, mixture 20% global / 40% labelled single source / 40% labelled unordered source pair, `max_steps=10`, `max_shrinkage=100`, `f_star=0.0018409808`, `sigma_f=2.558e-7`, the seven-round pairing schedule, and exact state-dependent MH selector correction. No source_gain enters the kernel.

Each individual post-slice and post-exchange outcome is checked, using batches of eight disjoint outcomes as in the historical runner. Finite state/cache, strict support, and direct cache recomputation are mandatory. Exchange checks enforce exact six-coordinate copying, unchanged untouched coordinates, joint acceptance, and exact restoration of both rejected states. Recomputed scalars are saved with the trace for post-run numerical-contract checks without additional model calls.

| Bank | Status | Completed sweeps | Max prior-cache error | Max likelihood-cache error | Structural failures |
| --- | --- | --- | --- | --- | --- |
| selection | completed_one_fixed_contour_benchmark | 384 | 0 | 0 | 0 |
| calibration | completed_one_fixed_contour_benchmark | 384 | 0 | 0 | 0 |

Both banks completed exactly eight walkers × (128 burn-in + 256 retained population sweeps). No extension occurred.

## Retained mixing diagnostics

First128 is trace sweeps 129–256; second128 is sweeps 257–384; full256 combines both after burn-in. SD uses `ddof=1`; MAD is the median absolute deviation of walker means; IQR is their 75th–25th percentile difference. ESS reuses the existing autocorrelation estimator.

### Selection

| Window | Mean-logL SD | MAD | IQR | Median KS | Max KS | Median ESS |
| --- | --- | --- | --- | --- | --- | --- |
| first128 | 75.92836838 | 26.87751598 | 55.04608713 | 0.1328125 | 0.25 | 48.48387287 |
| second128 | 104.9155759 | 55.29338937 | 84.86573251 | 0.15234375 | 0.3046875 | 59.0874076 |
| full256 | 54.47095968 | 27.66365373 | 55.61310901 | 0.091796875 | 0.13671875 | 104.941465 |

**first128**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -106183.9667 | 42.343773 | 1 |
| 1 | -106317.486 | 20.28135455 | 1 |
| 2 | -106251.0688 | 56.39092315 | 1 |
| 3 | -106168.2215 | 30.71566298 | 1 |
| 4 | -106214.0803 | 37.13344091 | 1 |
| 5 | -106221.9765 | 69.99329533 | 1 |
| 6 | -106053.5323 | 57.49920799 | 1 |
| 7 | -106176.1975 | 54.62397275 | 1 |

**second128**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -106017.7462 | 43.25144385 | 1 |
| 1 | -106207.3625 | 128 | 1 |
| 2 | -106180.1983 | 114.1804943 | 1 |
| 3 | -106203.2625 | 31.71514831 | 1 |
| 4 | -106032.8148 | 40.29214819 | 1 |
| 5 | -106200.5355 | 53.96697633 | 1 |
| 6 | -106302.7956 | 64.20783886 | 1 |
| 7 | -106290.7851 | 97.83272788 | 1 |

**full256**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -106100.8564 | 83.00859773 | 1 |
| 1 | -106262.4243 | 89.58893755 | 1 |
| 2 | -106215.6335 | 142.8721372 | 1 |
| 3 | -106185.742 | 66.07461056 | 1 |
| 4 | -106123.4475 | 64.86313963 | 1 |
| 5 | -106211.256 | 124.4162365 | 1 |
| 6 | -106178.164 | 120.2939925 | 1 |
| 7 | -106233.4913 | 133.9314051 | 1 |

### Calibration

| Window | Mean-logL SD | MAD | IQR | Median KS | Max KS | Median ESS |
| --- | --- | --- | --- | --- | --- | --- |
| first128 | 152.0306302 | 70.21280912 | 131.8260039 | 0.16015625 | 0.3125 | 55.98748084 |
| second128 | 87.02154425 | 51.94666778 | 77.43581765 | 0.10546875 | 0.1875 | 107.3148685 |
| full256 | 83.83929634 | 52.65018959 | 109.2519003 | 0.091796875 | 0.16796875 | 120.0659063 |

**first128**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -106389.7226 | 82.61733453 | 1 |
| 1 | -106079.4428 | 25.35641252 | 1 |
| 2 | -105863.9937 | 41.30574254 | 1 |
| 3 | -106220.4628 | 73.41198162 | 1 |
| 4 | -106204.3084 | 59.68342048 | 1 |
| 5 | -106080.0372 | 49.29250325 | 1 |
| 6 | -106111.2403 | 52.2915412 | 1 |
| 7 | -106208.7985 | 70.89186639 | 1 |

**second128**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -105918.1788 | 128 | 1 |
| 1 | -105856.3516 | 111.2371028 | 1 |
| 2 | -105966.7999 | 128 | 1 |
| 3 | -105901.4695 | 55.40185502 | 1 |
| 4 | -105842.5385 | 103.3926342 | 1 |
| 5 | -105742.2003 | 112.8935761 | 1 |
| 6 | -106035.1548 | 18.91363596 | 1 |
| 7 | -105901.3139 | 79.35220415 | 1 |

**full256**

| Walker | Mean logL | logL ESS | Unique-state fraction |
| --- | --- | --- | --- |
| 0 | -106153.9507 | 134.4171985 | 1 |
| 1 | -105967.8972 | 86.71868675 | 1 |
| 2 | -105915.3968 | 89.55716153 | 1 |
| 3 | -106060.9661 | 114.7390154 | 1 |
| 4 | -106023.4234 | 170.9577306 | 1 |
| 5 | -105911.1187 | 126.1762531 | 1 |
| 6 | -106073.1976 | 42.80852522 | 1 |
| 7 | -106055.0562 | 125.3927972 | 1 |

## Promotion relaxation and physical geometry

Only the existing Stage-4I permutation-invariant metrics are used. Frequency-set RMS and sorted-frequency mean distance are in Fourier-bin units; full source-set RMS uses the historical normalized six-coordinate assignment with angular wrapping. For retained windows, pair geometry uses the existing cross-time frequency comparison and every 32nd endpoint for full source-set matching. Values below are medians across the 28 within-bank walker pairs.

| Bank | Stage | Frequency-set RMS | Sorted-frequency mean distance | Six-coordinate source-set RMS |
| --- | --- | --- | --- | --- |
| selection | promoted | 80.05199016 | 80.05199016 | 0.2410989358 |
| selection | postburn | 81.26054033 | 81.26054033 | 0.2584647007 |
| selection | first128 | 86.67030031 | 44.06747414 | 0.2565006754 |
| selection | second128 | 93.95540881 | 55.25705122 | 0.2569686592 |
| selection | full256 | 92.19196162 | 41.63891095 | 0.2569638043 |
| calibration | promoted | 106.8132211 | 106.8132211 | 0.2505691062 |
| calibration | postburn | 88.43935976 | 88.43935976 | 0.2627945784 |
| calibration | first128 | 97.64191718 | 57.68035563 | 0.2554496214 |
| calibration | second128 | 103.781032 | 70.21339823 | 0.2594554453 |
| calibration | full256 | 104.7807066 | 45.41823909 | 0.2591101471 |

Distance min / q25 / median / q75 / max summaries and individual pair values are in the accompanying JSON.

Selection duplicate relaxation: no duplicate promoted positions occurred; duplicate decorrelation is not applicable.

Calibration duplicate relaxation: no duplicate promoted positions occurred; duplicate decorrelation is not applicable.

## Cross-bank agreement

| Window | Pooled logL KS | Selection mean | Calibration mean |
| --- | --- | --- | --- |
| first128 | 0.09375 | -106198.3162 | -106144.7508 |
| second128 | 0.138671875 | -106179.4376 | -105895.5009 |
| full256 | 0.1049804688 | -106188.8769 | -106020.1259 |

**first128 pooled logL quantiles**

| Bank | q10 | q25 | q50 | q75 | q90 |
| --- | --- | --- | --- | --- | --- |
| selection | -107031.7885 | -106821.2861 | -106469.948 | -105781.81 | -104864.5872 |
| calibration | -106976.8012 | -106761.6101 | -106323.8004 | -105710.2103 | -105004.961 |

**second128 pooled logL quantiles**

| Bank | q10 | q25 | q50 | q75 | q90 |
| --- | --- | --- | --- | --- | --- |
| selection | -106965.491 | -106767.7765 | -106380.56 | -105675.2425 | -105077.9091 |
| calibration | -106921.4963 | -106683.5376 | -106139.4309 | -105142.8982 | -104548.9684 |

**full256 pooled logL quantiles**

| Bank | q10 | q25 | q50 | q75 | q90 |
| --- | --- | --- | --- | --- | --- |
| selection | -107005.0898 | -106788.4401 | -106430.6939 | -105718.7975 | -104978.5423 |
| calibration | -106952.1586 | -106717.5488 | -106223.1405 | -105473.818 | -104663.1588 |

| Window | Cross-bank frequency RMS | Sorted-frequency mean distance | Six-coordinate RMS |
| --- | --- | --- | --- |
| first128 | 96.62124963 | 58.57229383 | 0.2606678785 |
| second128 | 96.19851623 | 59.41175296 | 0.2594867546 |
| full256 | 99.41140601 | 44.28790825 | 0.2601602155 |

## Exchange diagnostics and provenance

Promotion donors identify the original Stage-5B walker. Subsequent exchange provenance starts with 72 recipient/source-slot ancestries per bank; those are distinct from promotion donors. Exact-realization tokens are conservatively renewed for slice-targeted source blocks, as in the frozen provenance implementation. Counts below cover the full 384 sweeps, including burn-in.

### Selection

| Quantity | Value |
| --- | --- |
| n | 1536 |
| contour_survivors | 874 |
| contour_fraction | 0.5690104167 |
| accepted | 861 |
| fraction | 0.560546875 |
| mh_rejected_survivors | 13 |
| mh_rejection_fraction | 0.01487414188 |

| Log ratio | min | q25 | median | q75 | max |
| --- | --- | --- | --- | --- | --- |
| Selector | -0.2284215648 | -0.005781220111 | 2.207819497e-07 | 0.02480137132 | 3.114720217 |
| Joint prior | -1.705302566e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.705302566e-13 |

Nonfinite counts: zero in every numeric exchange field (all field counts are recorded in JSON).

| Provenance quantity | Count |
| --- | --- |
| accepted_directional_transfers | 1722 |
| distinct_migrated_realizations | 846 |
| foreign_birth_arrivals | 1647 |
| initial_lineages_visiting_other_walker | 40 |
| exact_initial_token_transfers | 9 |

| Walker | Accepted transfers received | Distinct donor walkers | Donor IDs |
| --- | --- | --- | --- |
| 0 | 205 | 7 | [1, 2, 3, 4, 5, 6, 7] |
| 1 | 229 | 7 | [0, 2, 3, 4, 5, 6, 7] |
| 2 | 211 | 7 | [0, 1, 3, 4, 5, 6, 7] |
| 3 | 216 | 7 | [0, 1, 2, 4, 5, 6, 7] |
| 4 | 220 | 7 | [0, 1, 2, 3, 5, 6, 7] |
| 5 | 222 | 7 | [0, 1, 2, 3, 4, 6, 7] |
| 6 | 205 | 7 | [0, 1, 2, 3, 4, 5, 7] |
| 7 | 214 | 7 | [0, 1, 2, 3, 4, 5, 6] |

| Pair | Proposals | Contour survivors | Accepted | Acceptance | MH rejection among survivors |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 54 | 29 | 28 | 0.5185185185 | 0.03448275862 |
| 0,2 | 55 | 33 | 31 | 0.5636363636 | 0.06060606061 |
| 0,3 | 55 | 30 | 29 | 0.5272727273 | 0.03333333333 |
| 0,4 | 55 | 33 | 33 | 0.6 | 0 |
| 0,5 | 55 | 28 | 28 | 0.5090909091 | 0 |
| 0,6 | 55 | 28 | 28 | 0.5090909091 | 0 |
| 0,7 | 55 | 28 | 28 | 0.5090909091 | 0 |
| 1,2 | 55 | 39 | 39 | 0.7090909091 | 0 |
| 1,3 | 55 | 35 | 35 | 0.6363636364 | 0 |
| 1,4 | 55 | 32 | 31 | 0.5636363636 | 0.03125 |
| 1,5 | 55 | 31 | 31 | 0.5636363636 | 0 |
| 1,6 | 55 | 26 | 25 | 0.4545454545 | 0.03846153846 |
| 1,7 | 55 | 40 | 40 | 0.7272727273 | 0 |
| 2,3 | 55 | 28 | 26 | 0.4727272727 | 0.07142857143 |
| 2,4 | 55 | 31 | 31 | 0.5636363636 | 0 |
| 2,5 | 55 | 29 | 29 | 0.5272727273 | 0 |
| 2,6 | 55 | 28 | 28 | 0.5090909091 | 0 |
| 2,7 | 54 | 27 | 27 | 0.5 | 0 |
| 3,4 | 55 | 35 | 35 | 0.6363636364 | 0 |
| 3,5 | 55 | 34 | 34 | 0.6181818182 | 0 |
| 3,6 | 54 | 32 | 31 | 0.5740740741 | 0.03125 |
| 3,7 | 55 | 26 | 26 | 0.4727272727 | 0 |
| 4,5 | 54 | 33 | 32 | 0.5925925926 | 0.0303030303 |
| 4,6 | 55 | 30 | 30 | 0.5454545455 | 0 |
| 4,7 | 55 | 30 | 28 | 0.5090909091 | 0.06666666667 |
| 5,6 | 55 | 34 | 33 | 0.6 | 0.02941176471 |
| 5,7 | 55 | 35 | 35 | 0.6363636364 | 0 |
| 6,7 | 55 | 30 | 30 | 0.5454545455 | 0 |

### Calibration

| Quantity | Value |
| --- | --- |
| n | 1536 |
| contour_survivors | 1011 |
| contour_fraction | 0.658203125 |
| accepted | 1010 |
| fraction | 0.6575520833 |
| mh_rejected_survivors | 1 |
| mh_rejection_fraction | 0.0009891196835 |

| Log ratio | min | q25 | median | q75 | max |
| --- | --- | --- | --- | --- | --- |
| Selector | -0.1928856511 | -2.865307592e-05 | 1.224793685e-07 | 0.006254641962 | 3.976437008 |
| Joint prior | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.705302566e-13 |

Nonfinite counts: zero in every numeric exchange field (all field counts are recorded in JSON).

| Provenance quantity | Count |
| --- | --- |
| accepted_directional_transfers | 2020 |
| distinct_migrated_realizations | 842 |
| foreign_birth_arrivals | 1909 |
| initial_lineages_visiting_other_walker | 26 |
| exact_initial_token_transfers | 18 |

| Walker | Accepted transfers received | Distinct donor walkers | Donor IDs |
| --- | --- | --- | --- |
| 0 | 267 | 7 | [1, 2, 3, 4, 5, 6, 7] |
| 1 | 226 | 7 | [0, 2, 3, 4, 5, 6, 7] |
| 2 | 266 | 7 | [0, 1, 3, 4, 5, 6, 7] |
| 3 | 253 | 7 | [0, 1, 2, 4, 5, 6, 7] |
| 4 | 255 | 7 | [0, 1, 2, 3, 5, 6, 7] |
| 5 | 253 | 7 | [0, 1, 2, 3, 4, 6, 7] |
| 6 | 263 | 7 | [0, 1, 2, 3, 4, 5, 7] |
| 7 | 237 | 7 | [0, 1, 2, 3, 4, 5, 6] |

| Pair | Proposals | Contour survivors | Accepted | Acceptance | MH rejection among survivors |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 54 | 31 | 31 | 0.5740740741 | 0 |
| 0,2 | 55 | 38 | 38 | 0.6909090909 | 0 |
| 0,3 | 55 | 43 | 43 | 0.7818181818 | 0 |
| 0,4 | 55 | 39 | 39 | 0.7090909091 | 0 |
| 0,5 | 55 | 42 | 42 | 0.7636363636 | 0 |
| 0,6 | 55 | 41 | 41 | 0.7454545455 | 0 |
| 0,7 | 55 | 33 | 33 | 0.6 | 0 |
| 1,2 | 55 | 33 | 33 | 0.6 | 0 |
| 1,3 | 55 | 30 | 30 | 0.5454545455 | 0 |
| 1,4 | 55 | 35 | 35 | 0.6363636364 | 0 |
| 1,5 | 55 | 31 | 31 | 0.5636363636 | 0 |
| 1,6 | 55 | 35 | 35 | 0.6363636364 | 0 |
| 1,7 | 55 | 31 | 31 | 0.5636363636 | 0 |
| 2,3 | 55 | 39 | 39 | 0.7090909091 | 0 |
| 2,4 | 55 | 36 | 36 | 0.6545454545 | 0 |
| 2,5 | 55 | 40 | 40 | 0.7272727273 | 0 |
| 2,6 | 55 | 42 | 42 | 0.7636363636 | 0 |
| 2,7 | 54 | 38 | 38 | 0.7037037037 | 0 |
| 3,4 | 55 | 38 | 38 | 0.6909090909 | 0 |
| 3,5 | 55 | 34 | 34 | 0.6181818182 | 0 |
| 3,6 | 54 | 37 | 37 | 0.6851851852 | 0 |
| 3,7 | 55 | 33 | 32 | 0.5818181818 | 0.0303030303 |
| 4,5 | 54 | 39 | 39 | 0.7222222222 | 0 |
| 4,6 | 55 | 32 | 32 | 0.5818181818 | 0 |
| 4,7 | 55 | 36 | 36 | 0.6545454545 | 0 |
| 5,6 | 55 | 38 | 38 | 0.6909090909 | 0 |
| 5,7 | 55 | 29 | 29 | 0.5272727273 | 0 |
| 6,7 | 55 | 38 | 38 | 0.6909090909 | 0 |

## Historical comparison and decision

| Run / bank | Median retained pairwise KS | Maximum |
| --- | --- | --- |
| Stage 5A fixed ell_12 | 0.0586 | 0.0977 |
| Stage 5B selection | 0.1445 | 0.7852 |
| Stage 5B calibration | 0.1016 | 0.1523 |
| Stage 6B selection | 0.091796875 | 0.13671875 |
| Stage 6B calibration | 0.091796875 | 0.16796875 |

Stage 6B uses the tighter fixed ell_test contour, so reproducing Stage-5A numbers is not a requirement. The decision assesses persistent separation in both retained halves, full-window distributions, cross-bank compatibility, and the existing physical geometry. These finite diagnostics do not establish whole-catalogue convergence.

Both banks completed all 384 sweeps and passed structural, numerical, independence, and exact replay checks. The severe Stage-5B selection separation did not persist: selection full-window median/max KS were 0.09180/0.13672 (half-window maxima 0.25000 and 0.30469), and calibration median/max were 0.09180/0.16797 (half-window maxima 0.31250 and 0.18750). Every retained state was unique within its walker. Cross-bank pooled KS was 0.10498 overall (0.09375 and 0.13867 by half), with comparable six-coordinate geometry: within-bank medians 0.25696 and 0.25911, cross-bank 0.26016. Calibration mean logL increased by about 249.25 between retained halves, versus 18.88 for selection; the second-half cross-bank mean gap was about 283.94. This residual drift is recorded, but the broad distributions overlap and neither bank shows severe persistent cross-walker separation. The finite-run evidence supports qualitatively compatible, usable populations under the stated validation criterion, without establishing full physical convergence.

## Replay, hashes, costs, and tests

Exact saved-operation replay, exact exchange replay, and exact provenance replay passed in both banks. Replay reconstructs post-slice/exchange states and caches from saved operations; it does not resample or reevaluate the model. Saved direct recomputations passed the frozen Stage-6A tolerances for every post-slice and post-exchange state.

| Bank | Slice likelihood proxy | Exchange likelihood calls | Direct cache likelihood calls | Total likelihood proxy |
| --- | --- | --- | --- | --- |
| selection | 21522 | 3072 | 6152 | 30746 |
| calibration | 21694 | 3072 | 6152 | 30918 |

The two-bank pre-sampling gate additionally used 16 prior and 16 likelihood evaluations. Prior counters match the corresponding likelihood counters. Slice proxy is the historical `num_steps + num_shrink`; it is not a claim of exact internal model-call instrumentation.

All **140 protected hashes** match, including the level-12 checkpoint, all Stage-5B artifacts and candidate, frozen kernel, selector constants, and Stage-6A evidence. The level-12 checkpoint SHA-256 remains `0f0ac7c6294b4cd5e06de1fe28eb975f6fb590b9daf833f9ea084c6ed73ab0c4`. No output checkpoint exists.

Cheap-test result:

```text
........................................................................ [ 19%]
........................................................................ [ 39%]
........................................................................ [ 58%]
........................................................................ [ 78%]
........................................................................ [ 97%]
.........                                                                [100%]
369 passed in 91.95s (0:01:31)
```

Existing tests were preserved. New synthetic tests cover CPU/x64 enforcement and contract use, both eligible pools, replacement and duplicates, zero-survivor donors and empty pools, byte-exact source copying, tolerances and strict support, bank isolation, immutable levels 0–12 and absent checkpoint-writing paths, unchanged transition source, exact provenance replay/tamper rejection, both-bank gating, and the persistent no-retry guard. No real LISA trajectory ran in pytest.

## Terminal consequence

**PASS.** Freeze pooled-with-replacement promotion and the population kernel; manual DNS methodology development is complete. The ladder remains frozen through level 12. No further methodology change or validation retry is authorized in this campaign. No level 13, production, or evidence work was performed.

Machine-readable record: [validation JSON](dns_stage6b_pooled_promotion_lisa_validation.json). Full run artifacts: `/tmp/lisa_dns_stage6b_pooled_promotion_validation/`; execution log: `/tmp/lisa_dns_stage6b_run.log`. Stream hashes, all source hashes, pairwise diagnostics, provenance, and artifact hashes are included in the JSON.

Implementation: [one-shot CPU runner](../../examples/lisa_dns_stage4/stage6b_pooled_promotion_validation.py), [saved-operation analysis](../../examples/lisa_dns_stage4/analyze_stage6b_pooled_promotion.py), [cheap tests](../../tests/experimental/test_dns_gb_stage6b_pooled_promotion.py).

STOP.
