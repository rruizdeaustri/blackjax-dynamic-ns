# Stage 3: automatic DNS level construction and calibration

Stage 3 passes its controlled toy validation. Separate threshold-selection and mass-calibration histories produce five-level ladders whose frozen production agrees with the supplied joint target and repeatedly changes high modes. This supports proceeding to a carefully monitored LISA exploration experiment, **not** a general claim that multimodal mass calibration is solved. No LISA integration, evidence, posterior reconstruction, or adaptive production is included.

The starting checkout was clean at `4314a18193b93492fd8701c1ac5710bfe11756a7` (`dns-stage2-frozen-multimodal`). All pre-existing files, including `dns.py`, `dns_kernels.py`, `dns_diagnostics.py`, `nss.py`, and prior tests, remain unchanged. BayesLISAx was not modified.

## Algorithm and statistical contract

`blackjax/ns/dns_levels.py` exposes `ConstructionConfig`, `build_next_level`, `construct_levels`, `calibrate_level_masses`, and `tail_diagnostics`. Construction is a bounded host loop around JIT-compiled, vmapped fixed-shape scans of the Stage-2 constrained-prior slice kernel. Production uses the unchanged Stage-1 frozen kernel.

1. Initialize two separate walker banks from the prior with independent random streams. Set ell[0] = -infinity and log Xhat[0] = 0.
2. At current level j, burn in and collect thinned samples separately in each bank, always targeting pi(theta) 1[logL > ell[j]]. Never pool samples generated under different contours.
3. Choose the next threshold using only the selection bank: sort N values ascending, set zero-based k = floor(N(1-rho)), and use ell[j+1] = sorted[k], rho = exp(-1). Membership is strictly greater than ell. There is no interpolation or tie jitter. For distinct observations the selection tail is (N-1-k)/N; all ties at the threshold are excluded.
4. Estimate r[j] using only the separate calibration bank sampled at ell[j], then set log Xhat[j+1] = log Xhat[j] + log(r[j]). The nearly forced selection fraction is never used as the mass estimate.
5. Promote each bank independently by resampling its own eligible history, then burn in at the new contour. Stop at the configured level count, budget, terminal threshold, or first insufficient/non-increasing/constant proposal. Failed proposals are recorded; there is no unbounded retry loop. Callers must check status before production.
6. Freeze thresholds, masses, uniform weights, and slice parameters. No construction state enters production.

`calibrate_level_masses` independently samples each contour of an existing frozen ladder with new random keys and its own prior-initialized bank, reconstructing cumulative log masses without moving thresholds. Insufficient calibration raises an error.

Separate banks avoid validating a quantile with its own order statistic. They do not imply exactly independent equilibrium samples: shared thresholds, survivor genealogy, and finite burn-in remain relevant. The caller must supply valid initialization and a constrained-prior-invariant parameter kernel.

## Correlation, uncertainty, and stopping

For each exceedance indicator, report raw N, tail fraction, naive IID standard error and Wilson interval, within-walker batch-mean standard error, and between-walker variation. The reported standard error is the maximum of IID, batch-mean, and between-walker estimates. Bernoulli-equivalent ESS is min(N, p(1-p)/SE²). The reported approximate 95% interval uses a Student multiplier, with walker degrees of freedom when between-walker variation dominates. These are diagnostics, not exact finite-MCMC coverage guarantees. Batch means require suitable mixing and block lengths; see [Flegal and Jones](https://arxiv.org/abs/0811.1729).

The builder requires at least four complete blocks per walker, positive configuration values, sufficient exceedance ESS in both banks, finite likelihood histories, increasing thresholds, and nondegenerate tails. It stops on the first failed proposal. The maximum parameter-step budget is checked before collecting a full level block. This implementation does not support construction histories containing negative-infinite likelihoods. Plateaus can cause early stopping; no plateau tie-breaking is implemented.

Tests include synthetic long correlations, opposed stuck walkers, and a real deliberately slow slice kernel that fails the minimum-ESS gate. Fixed thinning and blocks cannot diagnose every longer correlation or missing mode. Multiple constructions and sample/walker sensitivity supplement these checks.

## Reproduction and settings

CPU validation used JAX 0.10.0 and Python from `/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python`. Run from the repository root:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 PYTHONPATH=. taskset -c 0-3 /r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python examples/dns_level_construction_validation.py --output /tmp/stage3-float32.json
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=. taskset -c 0-3 /r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python examples/dns_level_construction_validation.py --output /tmp/stage3-float64.json
```

Main construction: seeds 101, 202, 303, 404; 8 walkers per bank; 5 levels including prior; 256 burn-in steps per contour; thinning 2; 1024 selection and 2048 calibration observations per walker; blocks of 64; minimum ESS 100; cap 10,000,000 individual walker parameter steps. Each finite level uses Nselection=8192 and Ncalibration=16384. Kernel parameters stay fixed: symmetric directions ±0.5, max_steps=10, max_shrinkage=100. Every main construction used 212,992 parameter transitions. Initial bank evaluation adds 16 likelihood calls.

Production: independent seed = construction seed + 1000; 8 walkers, 8000 sweeps, burn-in 1024, no adaptation. Seeds 101/303 start all walkers in A; 202/404 start all in B for the double well. Statistics use 6976 retained sweeps per walker; first-passage costs include the start and pre-burn trajectory. Production uncertainty uses time blocks of 256 and between-walker variation. Tests use six estimated standard-error regression envelopes with small numerical/Monte Carlo floors; these are not exact simultaneous confidence regions.

Full precision metadata, initial positions, all per-level moments/CDF points, edge counts, uncertainty intervals, refinement histories, and mechanism witnesses are archived in [float32 results](dns_stage3_results_float32.json) and [float64 results](dns_stage3_results_float64.json). Archived JSON represents nonfinite prior thresholds as strings (`-inf`); the example's Python JSON writer emits `-Infinity`.

## Test results

| Suite | float32 | float64 |
| --- | --- | --- |
| Stage 1 frozen core | 34 passed | 34 passed |
| Stage 2 constrained kernel / multimodal | 13 passed | 13 passed |
| Stage 3 construction / adaptive multimodal | 18 passed | 18 passed |
| All DNS | 65 passed | 65 passed |

Legacy NS: **54 passed**, each test in a fresh Python process. No assertion failures or resource failures occurred in this Stage-3 regression run. The final Stage-3 rerun took 55.14 s (float32) and 57.74 s (float64). Existing tests were not weakened. New tests cover quantile ties, finite sampling against a Beta order-statistic oracle, reproducibility, stopping, invalid configurations, correlation rejection, independent calibration, aggregate refinement improvement, frozen conditional distributions, level transitions, and assigned-high mode-switch witnesses.

## Targets and oracles

Gaussian: theta ~ Uniform[-1,1], logL = -theta²/2. For finite ell < 0, X(ell)=min(1,sqrt(-2ell)); ideal thresholds are -exp(-2j)/2.

Double well: theta ~ Uniform[-6,6], logL = -0.5((abs(theta)-1)/0.3)². Modes A/B are theta < 0 / theta > 0, centered at -1/+1. Set d=0.3 sqrt(-2ell). Then X(ell)=[min(6,1+d)-max(0,1-d)]/6. The ideal exp(-j) ladder uses d=6X-1 for X>=1/3, otherwise d=3X. All weights are w[j]=1. These formulas are used only for validation, never for construction or calibration. Level 1 connects the modal regions; higher levels separate them.

## Per-seed construction results

In all following tables level 0 is exactly ell=-infinity, X=Xhat=1, Delta logX=0 and requires no compression estimate. SE and ESS refer to the independent calibration exceedance indicator. Each table row uses 8192 selection and 16384 calibration samples. `ell ref` is the ideal exp(-j) threshold, not the true threshold for the empirical ladder.

### gaussian, float32
| seed | j | ell | ell ref | true X | Xhat | Delta logX | true r | calibrated r ± SE | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | -0.0711504 | -0.0676676 | 0.377228 | 0.374939 | -0.00608629 | 0.377228 | 0.374939 ± 0.00378209 | 16384 |
| 101 | 2 | -0.00966202 | -0.00915782 | 0.139011 | 0.136895 | -0.0153399 | 0.368507 | 0.365112 ± 0.00402762 | 14290 |
| 101 | 3 | -0.00127331 | -0.00123938 | 0.050464 | 0.0494222 | -0.0208602 | 0.363021 | 0.361023 ± 0.00380643 | 15921 |
| 101 | 4 | -0.000170714 | -0.000167731 | 0.0184778 | 0.0177973 | -0.0375221 | 0.366158 | 0.360107 ± 0.00398822 | 14487 |
| 202 | 1 | -0.0663972 | -0.0676676 | 0.36441 | 0.367737 | 0.0090884 | 0.36441 | 0.367737 ± 0.0037671 | 16384 |
| 202 | 2 | -0.00877257 | -0.00915782 | 0.132458 | 0.133727 | 0.00953102 | 0.363487 | 0.363647 ± 0.00375819 | 16384 |
| 202 | 3 | -0.0012431 | -0.00123938 | 0.0498617 | 0.0502862 | 0.00847745 | 0.376434 | 0.376038 ± 0.00435799 | 12354 |
| 202 | 4 | -0.000166557 | -0.000167731 | 0.0182514 | 0.0184675 | 0.0117736 | 0.36604 | 0.367249 ± 0.00376605 | 16384 |
| 303 | 1 | -0.0679995 | -0.0676676 | 0.36878 | 0.377319 | 0.0228905 | 0.36878 | 0.377319 ± 0.00380948 | 16190 |
| 303 | 2 | -0.00974685 | -0.00915782 | 0.13962 | 0.144005 | 0.0309248 | 0.378599 | 0.381653 ± 0.00379525 | 16384 |
| 303 | 3 | -0.0013099 | -0.00123938 | 0.0511839 | 0.0524549 | 0.0245292 | 0.366595 | 0.364258 ± 0.00394483 | 14881 |
| 303 | 4 | -0.00017957 | -0.000167731 | 0.018951 | 0.0190559 | 0.00552154 | 0.370252 | 0.363281 ± 0.00389858 | 15219 |
| 404 | 1 | -0.066822 | -0.0676676 | 0.365573 | 0.369385 | 0.0103717 | 0.365573 | 0.369385 ± 0.00377061 | 16384 |
| 404 | 2 | -0.00932447 | -0.00915782 | 0.136561 | 0.137167 | 0.00442374 | 0.373553 | 0.371338 ± 0.00431387 | 12544 |
| 404 | 3 | -0.00122646 | -0.00123938 | 0.049527 | 0.0496207 | 0.00189185 | 0.362672 | 0.361755 ± 0.00377135 | 16233 |
| 404 | 4 | -0.000165884 | -0.000167731 | 0.0182145 | 0.0182322 | 0.000973701 | 0.367769 | 0.367432 ± 0.00392627 | 15077 |

### double_well, float32
| seed | j | ell | ell ref | true X | Xhat | Delta logX | true r | calibrated r ± SE | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | -7.93863 | -8.09732 | 0.365898 | 0.362427 | -0.00953233 | 0.365898 | 0.362427 ± 0.00527359 | 8309 |
| 101 | 2 | -0.910569 | -0.915782 | 0.13495 | 0.132658 | -0.0171242 | 0.368817 | 0.366028 ± 0.00400673 | 14455 |
| 101 | 3 | -0.125572 | -0.123938 | 0.0501143 | 0.0495363 | -0.0116003 | 0.371356 | 0.373413 ± 0.00458304 | 11139 |
| 101 | 4 | -0.0166083 | -0.0167731 | 0.0182254 | 0.0178203 | -0.0224819 | 0.363677 | 0.359741 ± 0.00374941 | 16384 |
| 202 | 1 | -8.00672 | -8.09732 | 0.366751 | 0.369995 | 0.00880748 | 0.366751 | 0.369995 ± 0.00699888 | 4759 |
| 202 | 2 | -0.910684 | -0.915782 | 0.134958 | 0.136942 | 0.0145901 | 0.367983 | 0.370117 ± 0.00544509 | 7863 |
| 202 | 3 | -0.126262 | -0.123938 | 0.0502519 | 0.0506677 | 0.00824094 | 0.372352 | 0.369995 ± 0.00390037 | 15322 |
| 202 | 4 | -0.0172215 | -0.0167731 | 0.0185588 | 0.0182334 | -0.0176873 | 0.369316 | 0.359863 ± 0.00755879 | 4032 |
| 303 | 1 | -8.36054 | -8.09732 | 0.371124 | 0.37439 | 0.00876147 | 0.371124 | 0.37439 ± 0.00521919 | 8598 |
| 303 | 2 | -0.904684 | -0.915782 | 0.134513 | 0.137266 | 0.0202584 | 0.362447 | 0.366638 ± 0.00388498 | 15386 |
| 303 | 3 | -0.122058 | -0.123938 | 0.0494082 | 0.0500838 | 0.0135825 | 0.367312 | 0.364868 ± 0.00377794 | 16236 |
| 303 | 4 | -0.0162454 | -0.0167731 | 0.0180252 | 0.0182343 | 0.0115333 | 0.364822 | 0.364075 ± 0.00381217 | 15931 |
| 404 | 1 | -8.63428 | -8.09732 | 0.374444 | 0.376038 | 0.00424695 | 0.374444 | 0.376038 ± 0.00473381 | 10470 |
| 404 | 2 | -0.955246 | -0.915782 | 0.138221 | 0.139545 | 0.00953817 | 0.369135 | 0.371094 ± 0.00400633 | 14540 |
| 404 | 3 | -0.134341 | -0.123938 | 0.0518346 | 0.0520825 | 0.0047698 | 0.375014 | 0.37323 ± 0.00377861 | 16384 |
| 404 | 4 | -0.0175813 | -0.0167731 | 0.0187517 | 0.0187076 | -0.00235486 | 0.36176 | 0.359192 ± 0.00560836 | 7318 |

### gaussian, float64
| seed | j | ell | ell ref | true X | Xhat | Delta logX | true r | calibrated r ± SE | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | -0.0683708 | -0.0676676 | 0.369786 | 0.371582 | 0.0048453 | 0.369786 | 0.371582 ± 0.00397259 | 14796 |
| 101 | 2 | -0.00925483 | -0.00915782 | 0.13605 | 0.135125 | -0.00682485 | 0.367916 | 0.363647 ± 0.00375819 | 16384 |
| 101 | 3 | -0.00123748 | -0.00123938 | 0.049749 | 0.049344 | -0.00817398 | 0.365666 | 0.365173 ± 0.00504613 | 9104 |
| 101 | 4 | -0.000162486 | -0.000167731 | 0.018027 | 0.0176878 | -0.0189919 | 0.362358 | 0.358459 ± 0.00374647 | 16384 |
| 202 | 1 | -0.0664293 | -0.0676676 | 0.364498 | 0.363281 | -0.0033428 | 0.364498 | 0.363281 ± 0.00494864 | 9445 |
| 202 | 2 | -0.00895096 | -0.00915782 | 0.133798 | 0.135122 | 0.00984531 | 0.367075 | 0.371948 ± 0.00392832 | 15138 |
| 202 | 3 | -0.001136 | -0.00123938 | 0.0476655 | 0.0484275 | 0.0158579 | 0.35625 | 0.358398 ± 0.00397218 | 14574 |
| 202 | 4 | -0.000155179 | -0.000167731 | 0.017617 | 0.017912 | 0.0166085 | 0.369596 | 0.369873 ± 0.00403085 | 14345 |
| 303 | 1 | -0.0662285 | -0.0676676 | 0.363946 | 0.358032 | -0.0163837 | 0.363946 | 0.358032 ± 0.0038123 | 15815 |
| 303 | 2 | -0.00886472 | -0.00915782 | 0.133152 | 0.132929 | -0.00167534 | 0.365856 | 0.371277 ± 0.00377458 | 16384 |
| 303 | 3 | -0.00112286 | -0.00123938 | 0.047389 | 0.0474063 | 0.000365718 | 0.355901 | 0.356628 ± 0.00433988 | 12182 |
| 303 | 4 | -0.00015208 | -0.000167731 | 0.0174402 | 0.0173838 | -0.00323592 | 0.368022 | 0.366699 ± 0.00418709 | 13246 |
| 404 | 1 | -0.0668305 | -0.0676676 | 0.365597 | 0.364868 | -0.0019946 | 0.365597 | 0.364868 ± 0.00376088 | 16384 |
| 404 | 2 | -0.00880473 | -0.00915782 | 0.132701 | 0.133062 | 0.00271906 | 0.36297 | 0.364685 ± 0.00406457 | 14024 |
| 404 | 3 | -0.00113748 | -0.00123938 | 0.0476966 | 0.0482333 | 0.0111904 | 0.35943 | 0.362488 ± 0.00375561 | 16384 |
| 404 | 4 | -0.000156814 | -0.000167731 | 0.0177096 | 0.0178844 | 0.00982291 | 0.371296 | 0.370789 ± 0.00404644 | 14249 |

### double_well, float64
| seed | j | ell | ell ref | true X | Xhat | Delta logX | true r | calibrated r ± SE | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | -7.73348 | -8.09732 | 0.363307 | 0.366943 | 0.00995959 | 0.363307 | 0.366943 ± 0.00501869 | 9223 |
| 101 | 2 | -0.934863 | -0.915782 | 0.136738 | 0.137268 | 0.00386764 | 0.37637 | 0.374084 ± 0.00422304 | 13129 |
| 101 | 3 | -0.12827 | -0.123938 | 0.0506498 | 0.0508806 | 0.00454606 | 0.370415 | 0.370667 ± 0.00383343 | 15874 |
| 101 | 4 | -0.0171793 | -0.0167731 | 0.0185361 | 0.0187013 | 0.00887571 | 0.365966 | 0.367554 ± 0.00436122 | 12222 |
| 202 | 1 | -7.34794 | -8.09732 | 0.358343 | 0.361389 | 0.00846557 | 0.358343 | 0.361389 ± 0.0048195 | 9936 |
| 202 | 2 | -0.887568 | -0.915782 | 0.133234 | 0.134881 | 0.012286 | 0.371807 | 0.37323 ± 0.00546923 | 7820 |
| 202 | 3 | -0.118904 | -0.123938 | 0.0487655 | 0.0499301 | 0.0236019 | 0.366013 | 0.370178 ± 0.00500121 | 9321 |
| 202 | 4 | -0.0154364 | -0.0167731 | 0.0175707 | 0.018157 | 0.0328234 | 0.36031 | 0.363647 ± 0.00378313 | 16169 |
| 303 | 1 | -8.14265 | -8.09732 | 0.368442 | 0.368835 | 0.0010674 | 0.368442 | 0.368835 ± 0.00443689 | 11825 |
| 303 | 2 | -0.871221 | -0.915782 | 0.132002 | 0.132978 | 0.00736956 | 0.35827 | 0.360535 ± 0.0037825 | 16114 |
| 303 | 3 | -0.119837 | -0.123938 | 0.0489565 | 0.0502644 | 0.026366 | 0.370878 | 0.377991 ± 0.00398976 | 14770 |
| 303 | 4 | -0.0162027 | -0.0167731 | 0.0180015 | 0.0185915 | 0.0322484 | 0.367704 | 0.369873 ± 0.00377164 | 16384 |
| 404 | 1 | -7.13114 | -8.09732 | 0.355494 | 0.352905 | -0.00730811 | 0.355494 | 0.352905 ± 0.0049358 | 9374 |
| 404 | 2 | -0.860317 | -0.915782 | 0.131173 | 0.131241 | 0.000518662 | 0.368988 | 0.371887 ± 0.00377585 | 16384 |
| 404 | 3 | -0.107903 | -0.123938 | 0.0464549 | 0.0465319 | 0.00165566 | 0.35415 | 0.354553 ± 0.00389504 | 15084 |
| 404 | 4 | -0.0151335 | -0.0167731 | 0.0173974 | 0.0174495 | 0.00298629 | 0.374501 | 0.375 ± 0.00378221 | 16384 |

## Across-seed error and sensitivity

Scatter is sample standard deviation across four independent constructions; RMS includes bias. Threshold bias is mean(ell - ell_ref). No malformed/non-increasing level occurred in the main, sensitivity, or refinement construction studies. Deliberately malformed and insufficient-information inputs stop in unit tests.

| bits | target | j | threshold bias | mean Delta logX | scatter | RMS Delta logX |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | gaussian | 1 | -0.000424647 | 0.00906609 | 0.0118661 | 0.0137039 |
| 32 | gaussian | 2 | -0.000218658 | 0.00738493 | 0.019008 | 0.0180421 |
| 32 | gaussian | 3 | -2.38133e-05 | 0.00350958 | 0.0188238 | 0.0166754 |
| 32 | gaussian | 4 | -2.94969e-06 | -0.00481331 | 0.0222507 | 0.0198618 |
| 32 | double_well | 1 | -0.137728 | 0.00307089 | 0.00867017 | 0.00811229 |
| 32 | double_well | 2 | -0.00451355 | 0.00681564 | 0.0165497 | 0.0158705 |
| 32 | double_well | 3 | -0.00312098 | 0.00374824 | 0.0108554 | 0.0101207 |
| 32 | double_well | 4 | -0.000140988 | -0.00774771 | 0.0154567 | 0.0154664 |
| 64 | gaussian | 1 | 0.000702876 | -0.00421895 | 0.00886675 | 0.00876151 |
| 64 | gaussian | 2 | 0.000189009 | 0.00101604 | 0.00706115 | 0.00619897 |
| 64 | gaussian | 3 | 8.09206e-05 | 0.00481003 | 0.0108182 | 0.0105315 |
| 64 | gaussian | 4 | 1.10916e-05 | 0.00105089 | 0.0156958 | 0.0136336 |
| 64 | double_well | 1 | 0.508514 | 0.00304611 | 0.00792237 | 0.00750678 |
| 64 | double_well | 2 | 0.0272897 | 0.00601047 | 0.0050326 | 0.00742436 |
| 64 | double_well | 3 | 0.00520923 | 0.0140424 | 0.0127393 | 0.0178579 |
| 64 | double_well | 4 | 0.000785147 | 0.0192334 | 0.0155491 | 0.0234789 |

Sensitivity uses Gaussian seeds 501–504, blocks 32 and minimum ESS 30. Calibration draws are twice selection draws. The equal-total-sample comparison is 8×256 versus 2×1024. Four seeds describe variability; they do not establish a universal walker-count advantage.

| bits | walkers | selection / walker | calibration N | terminal bias | terminal RMS |
| --- | --- | --- | --- | --- | --- |
| 32 | 2 | 256 | 1024 | -0.0345709 | 0.0528711 |
| 32 | 8 | 256 | 4096 | -0.00393546 | 0.0446135 |
| 32 | 2 | 1024 | 4096 | -0.0193231 | 0.0282033 |
| 32 | 8 | 1024 | 16384 | -0.000873983 | 0.0119704 |
| 64 | 2 | 256 | 1024 | -0.0858427 | 0.142192 |
| 64 | 8 | 256 | 4096 | -0.0308775 | 0.0448101 |
| 64 | 2 | 1024 | 4096 | -0.0123968 | 0.026332 |
| 64 | 8 | 1024 | 16384 | -0.0115457 | 0.0208863 |

Independent refinement increases calibration draws per walker from 256 to 2048 at unchanged thresholds. RMS below is over levels 1–4. It improves aggregate error, not every realization; finite Monte Carlo recalibration can make an individual ladder worse.

| bits | seed | before RMS | after RMS |
| --- | --- | --- | --- |
| 32 | 71 | 0.00490553 | 0.00629213 |
| 32 | 72 | 0.028312 | 0.0173255 |
| 32 | 73 | 0.0224452 | 0.00470453 |
| 32 | 74 | 0.0403633 | 0.0232586 |
| 64 | 71 | 0.0264292 | 0.00659014 |
| 64 | 72 | 0.049228 | 0.00944231 |
| 64 | 73 | 0.0157502 | 0.0210553 |
| 64 | 74 | 0.0307288 | 0.0142805 |

## Frozen production

The expected assigned-level occupancy is normalized w[j] Xtrue[j]/Xhat[j], not uniform by assumption. Conditional checks use the exact interval/union CDF at probabilities .1,.25,.5,.75,.9, its first two moments, and physical position moments. The following table reports observed / predicted occupancy and maximum standardized discrepancies across levels/CDF points. Full uncertainties and moments are in the archived results.

| bits | target | seed | occupancy observed / predicted, j=0..4 | max occupancy z | max CDF z | low-high-low / high-low-high |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | gaussian | 101 | 0.196/0.197, 0.199/0.198, 0.202/0.200, 0.205/0.201, 0.198/0.204 | 0.837645 | 2.2783 | [573, 577] |
| 32 | gaussian | 202 | 0.208/0.202, 0.203/0.200, 0.194/0.200, 0.201/0.200, 0.194/0.199 | 1.55369 | 1.86084 | [570, 574] |
| 32 | gaussian | 303 | 0.204/0.203, 0.195/0.199, 0.196/0.197, 0.201/0.198, 0.205/0.202 | 0.953465 | 2.17974 | [576, 580] |
| 32 | gaussian | 404 | 0.198/0.201, 0.195/0.199, 0.196/0.200, 0.205/0.200, 0.207/0.201 | 1.11588 | 1.81054 | [582, 588] |
| 32 | double_well | 101 | 0.200/0.198, 0.201/0.199, 0.198/0.201, 0.198/0.200, 0.204/0.202 | 0.897549 | 1.04276 | [583, 588] |
| 32 | double_well | 202 | 0.211/0.201, 0.207/0.199, 0.195/0.198, 0.190/0.199, 0.197/0.204 | 1.84272 | 1.14825 | [554, 557] |
| 32 | double_well | 303 | 0.204/0.202, 0.197/0.200, 0.192/0.198, 0.201/0.199, 0.206/0.200 | 1.59351 | 2.11199 | [554, 558] |
| 32 | double_well | 404 | 0.191/0.201, 0.196/0.200, 0.201/0.199, 0.204/0.200, 0.208/0.201 | 1.38444 | 3.23502 | [601, 605] |
| 64 | gaussian | 101 | 0.215/0.199, 0.205/0.198, 0.204/0.200, 0.192/0.200, 0.185/0.203 | 2.8089 | 2.08338 | [555, 557] |
| 64 | gaussian | 202 | 0.214/0.202, 0.198/0.202, 0.201/0.200, 0.198/0.198, 0.188/0.198 | 1.89289 | 2.43085 | [611, 614] |
| 64 | gaussian | 303 | 0.200/0.199, 0.204/0.202, 0.190/0.199, 0.198/0.199, 0.208/0.200 | 2.72633 | 2.25323 | [575, 581] |
| 64 | gaussian | 404 | 0.195/0.201, 0.194/0.201, 0.202/0.200, 0.204/0.199, 0.205/0.199 | 1.55219 | 1.42265 | [574, 580] |
| 64 | double_well | 101 | 0.213/0.201, 0.201/0.199, 0.204/0.200, 0.192/0.200, 0.190/0.199 | 1.84455 | 2.05051 | [589, 593] |
| 64 | double_well | 202 | 0.210/0.203, 0.202/0.201, 0.194/0.201, 0.197/0.198, 0.196/0.197 | 1.22369 | 1.70584 | [600, 604] |
| 64 | double_well | 303 | 0.205/0.203, 0.203/0.202, 0.199/0.201, 0.197/0.197, 0.196/0.196 | 0.633663 | 2.31802 | [588, 592] |
| 64 | double_well | 404 | 0.201/0.200, 0.197/0.201, 0.206/0.200, 0.204/0.200, 0.192/0.199 | 1.38554 | 2.24424 | [551, 557] |

Uniform weights and decreasing estimated masses imply eligible upward acceptance 1; downward acceptance equals the estimated compression, and upward eligibility equals the true compression. Per-edge observed / expected values follow. Exact exp(-1) limits and half-proposal crossing probabilities remain covered by unchanged Stage-1/2 tests. With imperfect masses, successful upward crossing per visit is true_r/2 and downward is estimated_r/2; unequal occupancies restore balance. All observed eligible upward acceptances were 1.

| bits | target | seed | edge j→j+1 | up eligibility ± SE / theory | down acceptance ± SE / theory |
| --- | --- | --- | --- | --- | --- |
| 32 | gaussian | 101 | 0 | 0.3799 ± 0.0069 / 0.3772 | 0.3679 ± 0.0061 / 0.3749 |
| 32 | gaussian | 101 | 1 | 0.3707 ± 0.0068 / 0.3685 | 0.3666 ± 0.0066 / 0.3651 |
| 32 | gaussian | 101 | 2 | 0.3618 ± 0.0074 / 0.3630 | 0.3685 ± 0.0085 / 0.3610 |
| 32 | gaussian | 101 | 3 | 0.3494 ± 0.0070 / 0.3662 | 0.3646 ± 0.0071 / 0.3601 |
| 32 | gaussian | 202 | 0 | 0.3563 ± 0.0122 / 0.3644 | 0.3632 ± 0.0065 / 0.3677 |
| 32 | gaussian | 202 | 1 | 0.3602 ± 0.0066 / 0.3635 | 0.3755 ± 0.0066 / 0.3636 |
| 32 | gaussian | 202 | 2 | 0.3778 ± 0.0065 / 0.3764 | 0.3730 ± 0.0060 / 0.3760 |
| 32 | gaussian | 202 | 3 | 0.3587 ± 0.0081 / 0.3660 | 0.3744 ± 0.0069 / 0.3672 |
| 32 | gaussian | 303 | 0 | 0.3680 ± 0.0063 / 0.3688 | 0.3800 ± 0.0065 / 0.3773 |
| 32 | gaussian | 303 | 1 | 0.3741 ± 0.0063 / 0.3786 | 0.3766 ± 0.0070 / 0.3817 |
| 32 | gaussian | 303 | 2 | 0.3815 ± 0.0063 / 0.3666 | 0.3777 ± 0.0068 / 0.3643 |
| 32 | gaussian | 303 | 3 | 0.3742 ± 0.0061 / 0.3703 | 0.3632 ± 0.0063 / 0.3633 |
| 32 | gaussian | 404 | 0 | 0.3673 ± 0.0072 / 0.3656 | 0.3757 ± 0.0071 / 0.3694 |
| 32 | gaussian | 404 | 1 | 0.3710 ± 0.0072 / 0.3736 | 0.3748 ± 0.0063 / 0.3713 |
| 32 | gaussian | 404 | 2 | 0.3644 ± 0.0062 / 0.3627 | 0.3589 ± 0.0062 / 0.3618 |
| 32 | gaussian | 404 | 3 | 0.3665 ± 0.0069 / 0.3678 | 0.3690 ± 0.0069 / 0.3674 |
| 32 | double_well | 101 | 0 | 0.3636 ± 0.0112 / 0.3659 | 0.3632 ± 0.0071 / 0.3624 |
| 32 | double_well | 101 | 1 | 0.3656 ± 0.0072 / 0.3688 | 0.3707 ± 0.0058 / 0.3660 |
| 32 | double_well | 101 | 2 | 0.3677 ± 0.0064 / 0.3714 | 0.3706 ± 0.0067 / 0.3734 |
| 32 | double_well | 101 | 3 | 0.3699 ± 0.0066 / 0.3637 | 0.3645 ± 0.0068 / 0.3597 |
| 32 | double_well | 202 | 0 | 0.3659 ± 0.0144 / 0.3668 | 0.3667 ± 0.0069 / 0.3700 |
| 32 | double_well | 202 | 1 | 0.3571 ± 0.0075 / 0.3680 | 0.3772 ± 0.0070 / 0.3701 |
| 32 | double_well | 202 | 2 | 0.3598 ± 0.0066 / 0.3724 | 0.3774 ± 0.0069 / 0.3700 |
| 32 | double_well | 202 | 3 | 0.3674 ± 0.0069 / 0.3693 | 0.3642 ± 0.0069 / 0.3599 |
| 32 | double_well | 303 | 0 | 0.3690 ± 0.0085 / 0.3711 | 0.3745 ± 0.0070 / 0.3744 |
| 32 | double_well | 303 | 1 | 0.3721 ± 0.0071 / 0.3624 | 0.3783 ± 0.0068 / 0.3666 |
| 32 | double_well | 303 | 2 | 0.3833 ± 0.0068 / 0.3673 | 0.3694 ± 0.0096 / 0.3649 |
| 32 | double_well | 303 | 3 | 0.3652 ± 0.0080 / 0.3648 | 0.3582 ± 0.0065 / 0.3641 |
| 32 | double_well | 404 | 0 | 0.3808 ± 0.0086 / 0.3744 | 0.3761 ± 0.0070 / 0.3760 |
| 32 | double_well | 404 | 1 | 0.3642 ± 0.0064 / 0.3691 | 0.3617 ± 0.0064 / 0.3711 |
| 32 | double_well | 404 | 2 | 0.3730 ± 0.0067 / 0.3750 | 0.3752 ± 0.0068 / 0.3732 |
| 32 | double_well | 404 | 3 | 0.3614 ± 0.0065 / 0.3618 | 0.3613 ± 0.0060 / 0.3592 |
| 64 | gaussian | 101 | 0 | 0.3600 ± 0.0059 / 0.3698 | 0.3711 ± 0.0065 / 0.3716 |
| 64 | gaussian | 101 | 1 | 0.3728 ± 0.0080 / 0.3679 | 0.3633 ± 0.0063 / 0.3636 |
| 64 | gaussian | 101 | 2 | 0.3535 ± 0.0061 / 0.3657 | 0.3677 ± 0.0081 / 0.3652 |
| 64 | gaussian | 101 | 3 | 0.3552 ± 0.0070 / 0.3624 | 0.3664 ± 0.0070 / 0.3585 |
| 64 | gaussian | 202 | 0 | 0.3583 ± 0.0067 / 0.3645 | 0.3837 ± 0.0061 / 0.3633 |
| 64 | gaussian | 202 | 1 | 0.3726 ± 0.0077 / 0.3671 | 0.3630 ± 0.0064 / 0.3719 |
| 64 | gaussian | 202 | 2 | 0.3608 ± 0.0059 / 0.3562 | 0.3642 ± 0.0066 / 0.3584 |
| 64 | gaussian | 202 | 3 | 0.3556 ± 0.0068 / 0.3696 | 0.3768 ± 0.0099 / 0.3699 |
| 64 | gaussian | 303 | 0 | 0.3590 ± 0.0062 / 0.3639 | 0.3465 ± 0.0059 / 0.3580 |
| 64 | gaussian | 303 | 1 | 0.3588 ± 0.0069 / 0.3659 | 0.3871 ± 0.0077 / 0.3713 |
| 64 | gaussian | 303 | 2 | 0.3650 ± 0.0072 / 0.3559 | 0.3585 ± 0.0061 / 0.3566 |
| 64 | gaussian | 303 | 3 | 0.3779 ± 0.0075 / 0.3680 | 0.3662 ± 0.0062 / 0.3667 |
| 64 | gaussian | 404 | 0 | 0.3656 ± 0.0069 / 0.3656 | 0.3679 ± 0.0078 / 0.3649 |
| 64 | gaussian | 404 | 1 | 0.3704 ± 0.0065 / 0.3630 | 0.3608 ± 0.0063 / 0.3647 |
| 64 | gaussian | 404 | 2 | 0.3619 ± 0.0068 / 0.3594 | 0.3634 ± 0.0066 / 0.3625 |
| 64 | gaussian | 404 | 3 | 0.3790 ± 0.0063 / 0.3713 | 0.3754 ± 0.0064 / 0.3708 |
| 64 | double_well | 101 | 0 | 0.3602 ± 0.0130 / 0.3633 | 0.3733 ± 0.0070 / 0.3669 |
| 64 | double_well | 101 | 1 | 0.3824 ± 0.0069 / 0.3764 | 0.3682 ± 0.0063 / 0.3741 |
| 64 | double_well | 101 | 2 | 0.3646 ± 0.0064 / 0.3704 | 0.3791 ± 0.0073 / 0.3707 |
| 64 | double_well | 101 | 3 | 0.3655 ± 0.0066 / 0.3660 | 0.3690 ± 0.0066 / 0.3676 |
| 64 | double_well | 202 | 0 | 0.3574 ± 0.0137 / 0.3583 | 0.3700 ± 0.0066 / 0.3614 |
| 64 | double_well | 202 | 1 | 0.3704 ± 0.0069 / 0.3718 | 0.3826 ± 0.0090 / 0.3732 |
| 64 | double_well | 202 | 2 | 0.3730 ± 0.0060 / 0.3660 | 0.3653 ± 0.0069 / 0.3702 |
| 64 | double_well | 202 | 3 | 0.3660 ± 0.0062 / 0.3603 | 0.3697 ± 0.0070 / 0.3636 |
| 64 | double_well | 303 | 0 | 0.3676 ± 0.0083 / 0.3684 | 0.3714 ± 0.0064 / 0.3688 |
| 64 | double_well | 303 | 1 | 0.3582 ± 0.0064 / 0.3583 | 0.3705 ± 0.0081 / 0.3605 |
| 64 | double_well | 303 | 2 | 0.3659 ± 0.0062 / 0.3709 | 0.3733 ± 0.0064 / 0.3780 |
| 64 | double_well | 303 | 3 | 0.3641 ± 0.0068 / 0.3677 | 0.3721 ± 0.0065 / 0.3699 |
| 64 | double_well | 404 | 0 | 0.3431 ± 0.0090 / 0.3555 | 0.3502 ± 0.0059 / 0.3529 |
| 64 | double_well | 404 | 1 | 0.3791 ± 0.0110 / 0.3690 | 0.3614 ± 0.0058 / 0.3719 |
| 64 | double_well | 404 | 2 | 0.3485 ± 0.0088 / 0.3542 | 0.3482 ± 0.0082 / 0.3546 |
| 64 | double_well | 404 | 3 | 0.3631 ± 0.0079 / 0.3745 | 0.3904 ± 0.0073 / 0.3750 |

## Mode switching and matched control

Switch counts below require both endpoints at **assigned level >=3**, with opposite geometric mode labels. This is stronger than merely being eligible for a high contour or completing a level-index round trip. Every successful assigned-high switch visited level 0 or 1; both directional witnesses are retained per run. The fixed-high control uses the identical slice kernel at constructed level 3. All its matched-cost prefixes had zero mode switches.

| bits | seed | start | A→B | B→A | round trips LHL / HLH | switch depth 0 / 1 | post-burn B fraction ± SE | control switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 101 | A | 278 | 275 | [583, 588] | 384 / 169 | 0.4905 ± 0.0127 | 0 |
| 32 | 202 | B | 263 | 266 | [554, 557] | 371 / 158 | 0.5068 ± 0.0148 | 0 |
| 32 | 303 | A | 269 | 266 | [554, 558] | 353 / 182 | 0.4898 ± 0.0172 | 0 |
| 32 | 404 | B | 275 | 278 | [601, 605] | 390 / 163 | 0.4996 ± 0.0177 | 0 |
| 64 | 101 | A | 283 | 279 | [589, 593] | 387 / 175 | 0.5290 ± 0.0160 | 0 |
| 64 | 202 | B | 296 | 300 | [600, 604] | 410 / 186 | 0.5067 ± 0.0110 | 0 |
| 64 | 303 | A | 281 | 278 | [588, 592] | 386 / 173 | 0.5134 ± 0.0121 | 0 |
| 64 | 404 | B | 276 | 281 | [551, 557] | 380 / 177 | 0.4831 ± 0.0125 | 0 |

First-passage costs below are per walker, from the known initial high mode to the opposite high-eligible region (eligible level >=3), including burn-in. Assigned-high return switches are counted separately above. Costs sum algorithmic scalar likelihood evaluations (`num_steps + num_shrink`), not wall time, compilation, or physical batched GPU work. The control precomputes a bounded trajectory and analyzes each walker’s prefix reaching the DNS evaluation budget; unused suffix work is excluded. Construction cost is separate; add 16 initial cache evaluations. Production budgets include burn-in.

| bits | target | seed | construction cost | DNS cost | control cost | control / DNS | first-passage costs (8 walkers) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | gaussian | 101 | 1142234 | 361453 | — | — | — |
| 32 | gaussian | 202 | 1142984 | 362968 | — | — | — |
| 32 | gaussian | 303 | 1140106 | 361004 | — | — | — |
| 32 | gaussian | 404 | 1142578 | 362871 | — | — | — |
| 32 | double_well | 101 | 1556525 | 442326 | 442339 | 1.00003 | [308, 355, 753, 273, 108, 222, 308, 167] |
| 32 | double_well | 202 | 1560508 | 448539 | 448551 | 1.00003 | [262, 312, 521, 90, 251, 177, 475, 294] |
| 32 | double_well | 303 | 1561375 | 445782 | 445793 | 1.00002 | [389, 109, 324, 196, 798, 107, 104, 137] |
| 32 | double_well | 404 | 1564011 | 441620 | 441632 | 1.00003 | [114, 831, 383, 134, 232, 752, 257, 652] |
| 64 | gaussian | 101 | 1142211 | 361255 | — | — | — |
| 64 | gaussian | 202 | 1144767 | 362631 | — | — | — |
| 64 | gaussian | 303 | 1145796 | 364123 | — | — | — |
| 64 | gaussian | 404 | 1144464 | 364126 | — | — | — |
| 64 | double_well | 101 | 1558924 | 447190 | 447203 | 1.00003 | [229, 175, 420, 428, 59, 1477, 291, 173] |
| 64 | double_well | 202 | 1557715 | 446568 | 446582 | 1.00003 | [242, 65, 79, 404, 428, 341, 491, 347] |
| 64 | double_well | 303 | 1561076 | 443553 | 443578 | 1.00006 | [352, 174, 142, 609, 396, 371, 664, 83] |
| 64 | double_well | 404 | 1555362 | 443033 | 443050 | 1.00004 | [366, 594, 127, 364, 162, 417, 103, 465] |

Actual float64 seed-101, walker-0 witnesses (times count production transitions):

- A_to_B: times 28 → 35 → 43; positions [-1.0768, -0.2819, 1.621, 0.93]; assigned levels [3, 1, 1, 3]; crossing parameter update generated under level 1.

- B_to_A: times 103 → 106 → 111; positions [0.9465, 0.6487, -0.1468, -0.9308]; assigned levels [3, 1, 0, 3]; crossing parameter update generated under level 1.

## Limitations and decision

No particle-cache/constraint failures or slice failures occurred in the reported production runs. The two precisions need not yield identical random trajectories; both independently satisfy the statistical checks.

The modest four-edge Gaussian study shows controlled accumulated mass error with no obvious systematic bias at its sample size. It does not prove unbiasedness: log of an empirical compression has finite-sample Jensen bias. The independent refinement and larger-sample studies show the expected aggregate improvement, with individual exceptions explicitly reported.

The double-well symmetry is a substantial limitation: both disconnected modes have the same likelihood-tail law. A locally trapped calibration bank can therefore estimate compression correctly while having incorrect modal weights. The frozen production tests independently validate equal conditional modal probabilities and backtracking, but these results do not certify level construction for unequal, missing catalogue families. ESS of one exceedance indicator cannot certify global exploration. Longer correlations, survivor ancestry, deeper ladders, unequal mode volumes and narrow rare modes require additional scrutiny in any subsequent benchmark.

The host builder is intentionally experimental. It is bounded and reproducible; the frozen transition remains JIT/vmap/scan compatible. CPU tests do not establish GPU performance. No production tuning is adapted. No evidence or posterior accuracy is claimed.

**Conclusion:** automatic construction and independent mass calibration are reliable enough on these controlled targets to justify a subsequent, explicitly exploratory LISA GB benchmark with mode-coverage diagnostics. They are not sufficient to assume accurate LISA compression estimates without further validation. Work stops at Stage 3.

## Files added

- `blackjax/ns/dns_levels.py`
- `tests/ns/test_dns_levels.py`
- `tests/ns/test_dns_adaptive_multimodal.py`
- `examples/dns_level_construction_validation.py`
- `docs/examples/dns_stage3.md`
- `docs/examples/dns_stage3_results_float32.json`
- `docs/examples/dns_stage3_results_float64.json`

No existing files modified.
