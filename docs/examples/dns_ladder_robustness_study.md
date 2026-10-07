# Stage 9A automatic-ladder mixing robustness development study

**This is DEVELOPMENT, not validation. Termination remains UNVALIDATED. Stage 8 and Stage 8H remain FAIL; Stage 8R and all historical reports are unchanged.**

Each of the three configurations was run once on a new development model. The Stage-8H trimodal trajectory was not rerun. No LISA, production, reconstruction stream or termination test was run. Scientific transition, promotion, gates and estimator code were unchanged.

## Frozen design and single control variable

Design SHA256: `c6a3219e025a2d2e971c35d0037e07388619b3b7e4aa9ec7482db94283f92db2`. The complete preregistration is in `dns_ladder_robustness_design.json`, with a separate SHA256 file. Both were saved before the sampling-start marker. The model is explicitly a development model and cannot supply held-out validation evidence.

| Item | Frozen value |
|---|---|
| Prior | u~N(0,I_54); theta=3*u[0] |
| Likelihood | sum_k c_k NormalPDF(theta;mu_k,width_k) |
| Component centers | [-2.4, 2.5] |
| Widths | [0.45, 0.55] |
| Weights | [0.45, 0.55] |
| Construction seed | 91401 |
| Reserved, unused reconstruction seed | 91402 |
| Walkers | 8 independently initialized in each bank |
| Initialization | 16 independent prior states per walker per bank, initial_state SeedSequence(seed).spawn(2); pooled promotion to 8 walkers |
| Compression | exp(-1), strict-tail empirical order statistic |
| ESS reliability gate | 20, selection and calibration independently |
| Retained events | 1024 per walker per bank, 8192 total; block size 64 |
| Maximum | 6 accepted levels beyond level zero |
| Restart boundary | Completed level 1; highest factor completing all requested levels only |
| Numerical contract | CPU/x64/float64; rtol=0; prior atol=1e-9; likelihood atol=1e-7 |
| Environment | JAX 0.10.0, jaxlib 0.10.0, NumPy 2.4.4 |

The recovered active analytic construction baseline is burn-in 256 plus retained 1024 sweeps per bank, pinned to the saved Stage-8H design. The generic `LadderConfig()` API defaults are burn-in 128, retained 256 and block size 32; those smaller defaults were not the budget of the audited collapse and are not used as the experimental baseline.

| Budget | Burn-in physical sweeps | Physical sweeps between retained samples | Retained samples/walker | Total physical sweeps/bank |
|---|---:|---:|---:|---:|
| 1x | 256 | 1 | 1024 | 1280 |
| 2x | 512 | 2 | 1024 | 2560 |
| 4x | 1024 | 4 | 1024 | 5120 |

The external development adapter invokes the unchanged frozen population runner for the expanded physical schedule, then thins its output to exactly 1024 retained samples per walker. The automatic ladder, candidate/calibration procedure and retained-event block size receive the same configuration for all factors. Increasing transitions does not increase the observations used for compression ESS. Each physical sweep retains the original eight slice updates and four reciprocal exchange proposals, proposal mixture [0.2,0.4,0.4], scales pi/sqrt(3), max steps 10 and max shrinkage 100. Instrumentation observes results without modifying scientific transitions.

## Matched RNG and reproducibility semantics

same construction seed and initial states; same iteration/bank/purpose seeds; per-walker repeated threefry split; PCG64 label and MH arrays have identical common sweep prefixes. Factor f retains indices f*256+f-1, then every f sweeps. Round-robin sweep count extends unchanged. Different retained pools/contours cause matched random inputs to act on different states after level 1; promotion draw seeds remain matched but eligible-pool indexing can diverge.

An identical common RNG prefix is not an identical retained trajectory. The first-level starts are identical. Larger budgets extend the same physical transition schedule but retain different sweep indices. At subsequent levels the changed candidate contours and source pools legitimately make trajectories and selected donors differ. Selection/calibration remain independently namespaced; no budget factor is inserted into their seeds.

## Predeclared interpretation

- **CASE A:** clear monotonic improvement in ESS / walker balance / mode switching and at least one larger budget completes all requested levels: current algorithm viable but baseline mixing too small
- **CASE B:** all 1x/2x/4x show severe walker/modal persistence and fail reliability: more transitions insufficient; kernel robustness unresolved
- **CASE C:** nonmonotonic or ambiguous: no production change justified

The numerical classification rules were frozen with the design, not chosen after outcomes:

- **A:** baseline fails, a larger factor completes six; at every commonly attempted level and bank ESS nondecreasing, survivor CV nonincreasing, total retained switches nondecreasing across factors; at least one strict ESS and balance improvement and a strict switching improvement at a multi-component generating contour
- **B:** all three fail an ESS gate; each failing bank has survivor CV>=0.5 and at least four walkers with zero retained component switches while both components are observed in that bank
- **C:** all other outcomes, including all budgets succeeding or a structural failure

Severe persistence means survivor CV >=0.5 and at least four zero-switch walkers while both components are observed in the failing bank. Monotonic comparisons use exact observed inequalities with no new tolerance. The strict CASE A rule can therefore reject apparently favorable but nonmonotonic finite-sample results.

## Level-by-level results

| Budget | Attempted level | Accepted | Selection ESS | Calibration ESS | Selection SE between walkers | Calibration SE between walkers |
|---|---:|---|---:|---:|---:|---:|
| 1x | 1 | True | 268.86007 | 333.574461 | 0.0294041803 | 0.0268110428 |
| 1x | 2 | True | 855.458392 | 1082.95976 | 0.0156967695 | 0.0119752514 |
| 1x | 3 | True | 1143.81217 | 1359.56139 | 0.0140514239 | 0.0120455276 |
| 1x | 4 | True | 1330.50872 | 1543.01775 | 0.0121503112 | 0.0103348631 |
| 1x | 5 | True | 1575.36893 | 1531.08179 | 0.00880055962 | 0.00947879822 |
| 1x | 6 | True | 1061.90806 | 1514.05525 | 0.0147964987 | 0.0116589245 |
| 2x | 1 | True | 579.535593 | 594.504652 | 0.0200305337 | 0.020289026 |
| 2x | 2 | True | 1678.83541 | 1349.27054 | 0.0117662466 | 0.0132447294 |
| 2x | 3 | True | 1703.34319 | 2663.41738 | 0.0116837296 | 0.00917571193 |
| 2x | 4 | True | 2641.23747 | 2099.05266 | 0.00933943314 | 0.00828730668 |
| 2x | 5 | True | 2793.75024 | 2977.49743 | 0.00799763528 | 0.00891220773 |
| 2x | 6 | True | 2986.23484 | 2641.81154 | 0.00713100813 | 0.00720333062 |
| 4x | 1 | True | 1207.06366 | 1589.50524 | 0.0138783339 | 0.00673918471 |
| 4x | 2 | True | 3734.47843 | 2549.12693 | 0.00507751293 | 0.00938139198 |
| 4x | 3 | True | 2896.03097 | 4947.03563 | 0.00895985165 | 0.00476275391 |
| 4x | 4 | True | 4339.48031 | 4726.03511 | 0.00568512257 | 0.00590281055 |
| 4x | 5 | True | 1705.20734 | 4786.7545 | 0.0116765306 | 0.00560916549 |
| 4x | 6 | True | 6404.59038 | 5477.50208 | 0.00602541408 | 0.00655552812 |

Each bank below is evaluated against the single candidate selected from that attempt’s selection bank. Calibration is absent when selection fails; its promoted starts do not constitute a sampled calibration trajectory. Mode labels are the largest Gaussian likelihood contribution at saved coordinates, not latent-component IDs or posterior probabilities. Occupancy is conditional on the generating contour and is not expected to equal posterior mode weights.

### 1x budget

Accepted levels: **6/6**. Status: `ready`. Completed all requested levels: **True**.

**Attempted level 1.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -2.563687566153867 | 0.367553710938 | 3011 | 0.226273273 | 8 | 0.156094321 | 2938/5120 |
| calibration | -2.563687566153867 | 0.39892578125 | 3268 | 0.190093206 | 8 | 0.14504284 | 3007/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 234 | 0.228515625 | [599, 425] | 9 | 14 |
| selection | 1 | 376 | 0.3671875 | [544, 480] | 41 | 43 |
| selection | 2 | 391 | 0.381835938 | [536, 488] | 8 | 14 |
| selection | 3 | 454 | 0.443359375 | [387, 637] | 47 | 57 |
| selection | 4 | 294 | 0.287109375 | [607, 417] | 32 | 44 |
| selection | 5 | 459 | 0.448242188 | [684, 340] | 29 | 33 |
| selection | 6 | 333 | 0.325195312 | [450, 574] | 30 | 34 |
| selection | 7 | 470 | 0.458984375 | [379, 645] | 29 | 31 |
| calibration | 0 | 419 | 0.409179688 | [603, 421] | 20 | 26 |
| calibration | 1 | 232 | 0.2265625 | [359, 665] | 28 | 31 |
| calibration | 2 | 474 | 0.462890625 | [570, 454] | 31 | 48 |
| calibration | 3 | 457 | 0.446289062 | [491, 533] | 18 | 29 |
| calibration | 4 | 471 | 0.459960938 | [549, 475] | 14 | 25 |
| calibration | 5 | 409 | 0.399414062 | [563, 461] | 6 | 11 |
| calibration | 6 | 416 | 0.40625 | [359, 665] | 26 | 33 |
| calibration | 7 | 390 | 0.380859375 | [547, 477] | 25 | 38 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 2, 6, 7, 5, 0, 4, 5]`; retained indices `[8, 7, 4, 1, 9, 5, 5, 9]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 1, 1, 0, 1, 1]`; final labels `[1, 0, 0, 0, 1, 1, 1, 0]`. Retained switch total 225; physical-sweep switch total 270. Accepted exchange recipients touching the likelihood-bearing block: 684. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[1, 0, 6, 3, 2, 5, 6, 5]`; retained indices `[6, 7, 9, 7, 11, 14, 0, 1]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 0, 1, 1, 0, 0]`; final labels `[0, 1, 0, 1, 0, 0, 1, 0]`. Retained switch total 168; physical-sweep switch total 241. Accepted exchange recipients touching the likelihood-bearing block: 656. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 2.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -1.14498332571999 | 0.36767578125 | 3012 | 0.120750865 | 8 | 0.14309429 | 2895/5120 |
| calibration | -1.14498332571999 | 0.359252929688 | 2943 | 0.0942821148 | 8 | 0.136595311 | 2865/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 431 | 0.420898438 | [0, 1024] | 0 | 0 |
| selection | 1 | 342 | 0.333984375 | [1024, 0] | 0 | 0 |
| selection | 2 | 412 | 0.40234375 | [0, 1024] | 0 | 0 |
| selection | 3 | 379 | 0.370117188 | [0, 1024] | 0 | 0 |
| selection | 4 | 324 | 0.31640625 | [1024, 0] | 0 | 0 |
| selection | 5 | 406 | 0.396484375 | [0, 1024] | 0 | 0 |
| selection | 6 | 310 | 0.302734375 | [1024, 0] | 0 | 0 |
| selection | 7 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| calibration | 0 | 402 | 0.392578125 | [0, 1024] | 0 | 0 |
| calibration | 1 | 340 | 0.33203125 | [688, 336] | 1 | 1 |
| calibration | 2 | 387 | 0.377929688 | [0, 1024] | 0 | 0 |
| calibration | 3 | 306 | 0.298828125 | [0, 1024] | 0 | 0 |
| calibration | 4 | 341 | 0.333007812 | [0, 1024] | 0 | 0 |
| calibration | 5 | 401 | 0.391601562 | [0, 1024] | 0 | 0 |
| calibration | 6 | 385 | 0.375976562 | [1024, 0] | 0 | 0 |
| calibration | 7 | 381 | 0.372070312 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 5, 7, 3, 4, 3, 2, 6]`; retained indices `[121, 520, 557, 908, 797, 49, 781, 97]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 1, 1, 0, 1, 0, 1]`; final labels `[1, 0, 1, 1, 0, 1, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 10. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[2, 3, 6, 0, 4, 7, 4, 2]`; retained indices `[338, 854, 836, 911, 238, 64, 421, 342]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 1, 1, 1, 1, 0, 1]`; final labels `[1, 1, 1, 1, 1, 1, 0, 1]`. Retained switch total 1; physical-sweep switch total 1. Accepted exchange recipients touching the likelihood-bearing block: 6. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 3.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9498960190586865 | 0.367797851562 | 3013 | 0.108057805 | 8 | 0.139727846 | 2932/5120 |
| calibration | -0.9498960190586865 | 0.38037109375 | 3116 | 0.0895701529 | 8 | 0.144094994 | 2892/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 421 | 0.411132812 | [1024, 0] | 0 | 0 |
| selection | 1 | 403 | 0.393554688 | [1024, 0] | 0 | 0 |
| selection | 2 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| selection | 3 | 371 | 0.362304688 | [1024, 0] | 0 | 0 |
| selection | 4 | 301 | 0.293945312 | [1024, 0] | 0 | 0 |
| selection | 5 | 402 | 0.392578125 | [0, 1024] | 0 | 0 |
| selection | 6 | 338 | 0.330078125 | [1024, 0] | 0 | 0 |
| selection | 7 | 369 | 0.360351562 | [0, 1024] | 0 | 0 |
| calibration | 0 | 404 | 0.39453125 | [0, 1024] | 0 | 0 |
| calibration | 1 | 418 | 0.408203125 | [0, 1024] | 0 | 0 |
| calibration | 2 | 350 | 0.341796875 | [1024, 0] | 0 | 0 |
| calibration | 3 | 449 | 0.438476562 | [0, 1024] | 0 | 0 |
| calibration | 4 | 390 | 0.380859375 | [1024, 0] | 0 | 0 |
| calibration | 5 | 384 | 0.375 | [0, 1024] | 0 | 0 |
| calibration | 6 | 342 | 0.333984375 | [1024, 0] | 0 | 0 |
| calibration | 7 | 379 | 0.370117188 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[6, 4, 7, 6, 1, 2, 4, 7]`; retained indices `[779, 478, 224, 483, 235, 455, 80, 120]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 1, 0, 0, 1, 0, 1]`; final labels `[0, 0, 1, 0, 0, 1, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[7, 5, 6, 3, 1, 3, 1, 1]`; retained indices `[371, 157, 685, 899, 91, 488, 351, 545]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 1, 0, 1, 0, 0]`; final labels `[1, 1, 0, 1, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 4.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.923294941439622 | 0.367309570312 | 3009 | 0.0935621406 | 8 | 0.141242938 | 2840/5120 |
| calibration | -0.923294941439622 | 0.386108398438 | 3163 | 0.0757077732 | 8 | 0.140689219 | 3035/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 354 | 0.345703125 | [1024, 0] | 0 | 0 |
| selection | 1 | 421 | 0.411132812 | [1024, 0] | 0 | 0 |
| selection | 2 | 374 | 0.365234375 | [1024, 0] | 0 | 0 |
| selection | 3 | 373 | 0.364257812 | [0, 1024] | 0 | 0 |
| selection | 4 | 376 | 0.3671875 | [0, 1024] | 0 | 0 |
| selection | 5 | 371 | 0.362304688 | [0, 1024] | 0 | 0 |
| selection | 6 | 315 | 0.307617188 | [1024, 0] | 0 | 0 |
| selection | 7 | 425 | 0.415039062 | [0, 1024] | 0 | 0 |
| calibration | 0 | 407 | 0.397460938 | [0, 1024] | 0 | 0 |
| calibration | 1 | 363 | 0.354492188 | [1024, 0] | 0 | 0 |
| calibration | 2 | 445 | 0.434570312 | [1024, 0] | 0 | 0 |
| calibration | 3 | 426 | 0.416015625 | [1024, 0] | 0 | 0 |
| calibration | 4 | 381 | 0.372070312 | [0, 1024] | 0 | 0 |
| calibration | 5 | 379 | 0.370117188 | [0, 1024] | 0 | 0 |
| calibration | 6 | 361 | 0.352539062 | [1024, 0] | 0 | 0 |
| calibration | 7 | 401 | 0.391601562 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[3, 0, 1, 2, 7, 5, 6, 7]`; retained indices `[217, 195, 321, 778, 739, 813, 376, 292]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 1, 1, 1, 0, 1]`; final labels `[0, 0, 0, 1, 1, 1, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[0, 6, 2, 7, 5, 3, 4, 0]`; retained indices `[573, 640, 740, 759, 521, 973, 838, 37]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 0, 0, 1, 1, 0, 1]`; final labels `[1, 0, 0, 0, 1, 1, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 5.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9195957976658315 | 0.367797851562 | 3013 | 0.0676777786 | 8 | 0.135413209 | 2888/5120 |
| calibration | -0.9195957976658315 | 0.405883789062 | 3325 | 0.0660536112 | 8 | 0.136541353 | 3019/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| selection | 1 | 348 | 0.33984375 | [1024, 0] | 0 | 0 |
| selection | 2 | 346 | 0.337890625 | [1024, 0] | 0 | 0 |
| selection | 3 | 381 | 0.372070312 | [1024, 0] | 0 | 0 |
| selection | 4 | 402 | 0.392578125 | [1024, 0] | 0 | 0 |
| selection | 5 | 381 | 0.372070312 | [1024, 0] | 0 | 0 |
| selection | 6 | 350 | 0.341796875 | [0, 1024] | 0 | 0 |
| selection | 7 | 397 | 0.387695312 | [1024, 0] | 0 | 0 |
| calibration | 0 | 405 | 0.395507812 | [0, 1024] | 0 | 0 |
| calibration | 1 | 453 | 0.442382812 | [0, 1024] | 0 | 0 |
| calibration | 2 | 373 | 0.364257812 | [0, 1024] | 0 | 0 |
| calibration | 3 | 454 | 0.443359375 | [1024, 0] | 0 | 0 |
| calibration | 4 | 411 | 0.401367188 | [0, 1024] | 0 | 0 |
| calibration | 5 | 398 | 0.388671875 | [1024, 0] | 0 | 0 |
| calibration | 6 | 407 | 0.397460938 | [1024, 0] | 0 | 0 |
| calibration | 7 | 424 | 0.4140625 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 1, 0, 0, 1, 1, 4, 1]`; retained indices `[1008, 788, 16, 602, 991, 208, 460, 341]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 0, 0, 0, 0, 1, 0]`; final labels `[1, 0, 0, 0, 0, 0, 1, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[7, 4, 0, 1, 0, 6, 3, 4]`; retained indices `[932, 32, 642, 457, 559, 382, 13, 555]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 1, 0, 1, 0, 0, 1]`; final labels `[1, 1, 1, 0, 1, 0, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 6.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9190235736728288 | 0.36767578125 | 3012 | 0.113825333 | 8 | 0.157038513 | 3127/5120 |
| calibration | -0.9190235736728288 | 0.379272460938 | 3107 | 0.0869465141 | 8 | 0.139040875 | 2937/5120 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 473 | 0.461914062 | [1024, 0] | 0 | 0 |
| selection | 1 | 346 | 0.337890625 | [1024, 0] | 0 | 0 |
| selection | 2 | 336 | 0.328125 | [0, 1024] | 0 | 0 |
| selection | 3 | 356 | 0.34765625 | [1024, 0] | 0 | 0 |
| selection | 4 | 374 | 0.365234375 | [1024, 0] | 0 | 0 |
| selection | 5 | 394 | 0.384765625 | [1024, 0] | 0 | 0 |
| selection | 6 | 371 | 0.362304688 | [1024, 0] | 0 | 0 |
| selection | 7 | 362 | 0.353515625 | [1024, 0] | 0 | 0 |
| calibration | 0 | 373 | 0.364257812 | [0, 1024] | 0 | 0 |
| calibration | 1 | 354 | 0.345703125 | [0, 1024] | 0 | 0 |
| calibration | 2 | 347 | 0.338867188 | [0, 1024] | 0 | 0 |
| calibration | 3 | 392 | 0.3828125 | [1024, 0] | 0 | 0 |
| calibration | 4 | 425 | 0.415039062 | [1024, 0] | 0 | 0 |
| calibration | 5 | 420 | 0.41015625 | [0, 1024] | 0 | 0 |
| calibration | 6 | 364 | 0.35546875 | [1024, 0] | 0 | 0 |
| calibration | 7 | 432 | 0.421875 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[1, 1, 0, 3, 2, 5, 7, 2]`; retained indices `[821, 304, 660, 668, 537, 323, 282, 308]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 1, 0, 0, 0, 0, 0]`; final labels `[0, 0, 1, 0, 0, 0, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[1, 4, 7, 6, 5, 4, 6, 3]`; retained indices `[911, 151, 563, 47, 737, 882, 894, 793]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 1, 0, 0, 1, 0, 0]`; final labels `[1, 1, 1, 0, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

### 2x budget

Accepted levels: **6/6**. Status: `ready`. Completed all requested levels: **True**.

**Attempted level 1.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -2.974534275401428 | 0.367797851562 | 3013 | 0.154038161 | 8 | 0.146033853 | 5831/10240 |
| calibration | -2.974534275401428 | 0.427368164062 | 3501 | 0.134277741 | 8 | 0.151385318 | 5997/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 270 | 0.263671875 | [495, 529] | 30 | 46 |
| selection | 1 | 440 | 0.4296875 | [476, 548] | 52 | 72 |
| selection | 2 | 395 | 0.385742188 | [506, 518] | 48 | 65 |
| selection | 3 | 318 | 0.310546875 | [523, 501] | 67 | 90 |
| selection | 4 | 359 | 0.350585938 | [426, 598] | 55 | 80 |
| selection | 5 | 390 | 0.380859375 | [649, 375] | 33 | 49 |
| selection | 6 | 415 | 0.405273438 | [557, 467] | 60 | 70 |
| selection | 7 | 426 | 0.416015625 | [361, 663] | 44 | 58 |
| calibration | 0 | 496 | 0.484375 | [509, 515] | 54 | 66 |
| calibration | 1 | 386 | 0.376953125 | [508, 516] | 28 | 48 |
| calibration | 2 | 421 | 0.411132812 | [493, 531] | 56 | 93 |
| calibration | 3 | 490 | 0.478515625 | [538, 486] | 43 | 65 |
| calibration | 4 | 530 | 0.517578125 | [324, 700] | 32 | 48 |
| calibration | 5 | 388 | 0.37890625 | [641, 383] | 32 | 43 |
| calibration | 6 | 410 | 0.400390625 | [405, 619] | 56 | 80 |
| calibration | 7 | 380 | 0.37109375 | [506, 518] | 51 | 75 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 2, 6, 7, 5, 0, 4, 5]`; retained indices `[8, 7, 4, 1, 9, 5, 5, 9]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 1, 1, 0, 1, 1]`; final labels `[1, 1, 1, 1, 1, 1, 1, 1]`. Retained switch total 389; physical-sweep switch total 530. Accepted exchange recipients touching the likelihood-bearing block: 1282. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[1, 0, 6, 3, 2, 5, 6, 5]`; retained indices `[6, 7, 9, 7, 11, 14, 0, 1]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 0, 1, 1, 0, 0]`; final labels `[0, 0, 1, 1, 1, 0, 0, 1]`. Retained switch total 352; physical-sweep switch total 518. Accepted exchange recipients touching the likelihood-bearing block: 1366. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 2.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -1.22830257834649 | 0.367431640625 | 3010 | 0.0905745926 | 8 | 0.143521595 | 5986/10240 |
| calibration | -1.22830257834649 | 0.384643554688 | 3151 | 0.0973934213 | 8 | 0.13995557 | 5870/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 368 | 0.359375 | [65, 959] | 1 | 1 |
| selection | 1 | 365 | 0.356445312 | [103, 921] | 1 | 1 |
| selection | 2 | 392 | 0.3828125 | [734, 290] | 1 | 2 |
| selection | 3 | 405 | 0.395507812 | [1024, 0] | 0 | 1 |
| selection | 4 | 376 | 0.3671875 | [0, 1024] | 0 | 0 |
| selection | 5 | 320 | 0.3125 | [527, 497] | 2 | 2 |
| selection | 6 | 432 | 0.421875 | [1024, 0] | 0 | 0 |
| selection | 7 | 352 | 0.34375 | [924, 100] | 1 | 1 |
| calibration | 0 | 423 | 0.413085938 | [455, 569] | 3 | 3 |
| calibration | 1 | 395 | 0.385742188 | [0, 1024] | 0 | 0 |
| calibration | 2 | 410 | 0.400390625 | [1024, 0] | 0 | 0 |
| calibration | 3 | 339 | 0.331054688 | [953, 71] | 1 | 1 |
| calibration | 4 | 441 | 0.430664062 | [997, 27] | 2 | 2 |
| calibration | 5 | 335 | 0.327148438 | [0, 1024] | 0 | 0 |
| calibration | 6 | 416 | 0.40625 | [1024, 0] | 0 | 1 |
| calibration | 7 | 392 | 0.3828125 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[2, 4, 7, 0, 0, 2, 3, 2]`; retained indices `[443, 382, 410, 736, 962, 258, 825, 257]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 1, 1, 0, 0, 0]`; final labels `[0, 0, 0, 0, 1, 0, 0, 1]`. Retained switch total 6; physical-sweep switch total 8. Accepted exchange recipients touching the likelihood-bearing block: 15. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[0, 6, 4, 1, 5, 5, 6, 3]`; retained indices `[145, 245, 107, 903, 648, 220, 822, 479]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 0, 1, 0, 1, 1, 0]`; final labels `[1, 1, 0, 0, 0, 1, 0, 0]`. Retained switch total 6; physical-sweep switch total 7. Accepted exchange recipients touching the likelihood-bearing block: 7. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 3.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9600964718572264 | 0.367797851562 | 3013 | 0.089849839 | 8 | 0.146033853 | 5918/10240 |
| calibration | -0.9600964718572264 | 0.358032226562 | 2933 | 0.0724874204 | 8 | 0.138765769 | 5906/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 350 | 0.341796875 | [730, 294] | 1 | 1 |
| selection | 1 | 440 | 0.4296875 | [0, 1024] | 0 | 0 |
| selection | 2 | 366 | 0.357421875 | [0, 1024] | 0 | 0 |
| selection | 3 | 381 | 0.372070312 | [0, 1024] | 0 | 0 |
| selection | 4 | 344 | 0.3359375 | [0, 1024] | 0 | 0 |
| selection | 5 | 403 | 0.393554688 | [1024, 0] | 0 | 0 |
| selection | 6 | 341 | 0.333007812 | [1024, 0] | 0 | 0 |
| selection | 7 | 388 | 0.37890625 | [1024, 0] | 0 | 0 |
| calibration | 0 | 396 | 0.38671875 | [1024, 0] | 0 | 0 |
| calibration | 1 | 375 | 0.366210938 | [0, 1024] | 0 | 0 |
| calibration | 2 | 337 | 0.329101562 | [1024, 0] | 0 | 0 |
| calibration | 3 | 407 | 0.397460938 | [1024, 0] | 0 | 0 |
| calibration | 4 | 359 | 0.350585938 | [0, 1024] | 0 | 0 |
| calibration | 5 | 375 | 0.366210938 | [1024, 0] | 0 | 0 |
| calibration | 6 | 333 | 0.325195312 | [0, 1024] | 0 | 0 |
| calibration | 7 | 351 | 0.342773438 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[7, 0, 4, 4, 5, 2, 2, 7]`; retained indices `[86, 842, 994, 614, 319, 908, 929, 40]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 1, 1, 1, 0, 0, 0]`; final labels `[1, 1, 1, 1, 1, 0, 0, 0]`. Retained switch total 1; physical-sweep switch total 1. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[2, 5, 0, 4, 0, 4, 1, 1]`; retained indices `[426, 316, 201, 447, 82, 428, 827, 900]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 0, 0, 1, 0, 1, 1]`; final labels `[0, 1, 0, 0, 1, 0, 1, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 1. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 4.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.924274838557277 | 0.367797851562 | 3013 | 0.0718218062 | 8 | 0.135413209 | 5895/10240 |
| calibration | -0.924274838557277 | 0.361206054688 | 2959 | 0.0648938264 | 8 | 0.139236228 | 6110/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 335 | 0.327148438 | [0, 1024] | 0 | 0 |
| selection | 1 | 355 | 0.346679688 | [1024, 0] | 0 | 0 |
| selection | 2 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| selection | 3 | 394 | 0.384765625 | [1024, 0] | 0 | 0 |
| selection | 4 | 353 | 0.344726562 | [1024, 0] | 0 | 0 |
| selection | 5 | 395 | 0.385742188 | [0, 1024] | 0 | 0 |
| selection | 6 | 403 | 0.393554688 | [1024, 0] | 0 | 0 |
| selection | 7 | 370 | 0.361328125 | [1024, 0] | 0 | 0 |
| calibration | 0 | 337 | 0.329101562 | [0, 1024] | 0 | 0 |
| calibration | 1 | 357 | 0.348632812 | [1024, 0] | 0 | 0 |
| calibration | 2 | 412 | 0.40234375 | [1024, 0] | 0 | 0 |
| calibration | 3 | 355 | 0.346679688 | [0, 1024] | 0 | 0 |
| calibration | 4 | 380 | 0.37109375 | [1024, 0] | 0 | 0 |
| calibration | 5 | 389 | 0.379882812 | [1024, 0] | 0 | 0 |
| calibration | 6 | 353 | 0.344726562 | [1024, 0] | 0 | 0 |
| calibration | 7 | 376 | 0.3671875 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[3, 0, 1, 7, 6, 0, 7, 5]`; retained indices `[227, 209, 343, 783, 745, 827, 391, 317]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 1, 0, 0, 1, 0, 0]`; final labels `[1, 0, 1, 0, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[1, 2, 0, 1, 3, 2, 5, 1]`; retained indices `[490, 543, 448, 575, 91, 536, 412, 669]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 0, 1, 0, 0, 0, 1]`; final labels `[1, 0, 0, 1, 0, 0, 0, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 5.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9197517213368271 | 0.367797851562 | 3013 | 0.0615031558 | 8 | 0.135413209 | 5676/10240 |
| calibration | -0.9197517213368271 | 0.3837890625 | 3144 | 0.0656806891 | 8 | 0.138040712 | 5916/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 333 | 0.325195312 | [1024, 0] | 0 | 0 |
| selection | 1 | 396 | 0.38671875 | [1024, 0] | 0 | 0 |
| selection | 2 | 391 | 0.381835938 | [1024, 0] | 0 | 0 |
| selection | 3 | 380 | 0.37109375 | [1024, 0] | 0 | 0 |
| selection | 4 | 366 | 0.357421875 | [1024, 0] | 0 | 0 |
| selection | 5 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| selection | 6 | 364 | 0.35546875 | [1024, 0] | 0 | 0 |
| selection | 7 | 375 | 0.366210938 | [1024, 0] | 0 | 0 |
| calibration | 0 | 405 | 0.395507812 | [0, 1024] | 0 | 0 |
| calibration | 1 | 434 | 0.423828125 | [1024, 0] | 0 | 0 |
| calibration | 2 | 370 | 0.361328125 | [1024, 0] | 0 | 0 |
| calibration | 3 | 425 | 0.415039062 | [0, 1024] | 0 | 0 |
| calibration | 4 | 374 | 0.365234375 | [1024, 0] | 0 | 0 |
| calibration | 5 | 372 | 0.36328125 | [0, 1024] | 0 | 0 |
| calibration | 6 | 371 | 0.362304688 | [1024, 0] | 0 | 0 |
| calibration | 7 | 393 | 0.383789062 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[1, 6, 4, 6, 3, 2, 7, 4]`; retained indices `[118, 897, 94, 983, 397, 1016, 381, 507]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 0, 0, 1, 0, 0]`; final labels `[0, 0, 0, 0, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[0, 1, 4, 7, 1, 3, 4, 2]`; retained indices `[1015, 109, 27, 508, 739, 620, 747, 8]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 0, 1, 0, 1, 0, 0]`; final labels `[1, 0, 0, 1, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 6.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9190588860656218 | 0.36767578125 | 3012 | 0.0548568545 | 8 | 0.135126162 | 5949/10240 |
| calibration | -0.9190588860656218 | 0.383911132812 | 3145 | 0.0530698226 | 8 | 0.133227345 | 5872/10240 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 372 | 0.36328125 | [0, 1024] | 0 | 0 |
| selection | 1 | 348 | 0.33984375 | [1024, 0] | 0 | 0 |
| selection | 2 | 407 | 0.397460938 | [0, 1024] | 0 | 0 |
| selection | 3 | 388 | 0.37890625 | [0, 1024] | 0 | 0 |
| selection | 4 | 387 | 0.377929688 | [0, 1024] | 0 | 0 |
| selection | 5 | 376 | 0.3671875 | [1024, 0] | 0 | 0 |
| selection | 6 | 347 | 0.338867188 | [1024, 0] | 0 | 0 |
| selection | 7 | 387 | 0.377929688 | [1024, 0] | 0 | 0 |
| calibration | 0 | 400 | 0.390625 | [1024, 0] | 0 | 0 |
| calibration | 1 | 419 | 0.409179688 | [0, 1024] | 0 | 0 |
| calibration | 2 | 372 | 0.36328125 | [1024, 0] | 0 | 0 |
| calibration | 3 | 415 | 0.405273438 | [0, 1024] | 0 | 0 |
| calibration | 4 | 401 | 0.391601562 | [1024, 0] | 0 | 0 |
| calibration | 5 | 402 | 0.392578125 | [0, 1024] | 0 | 0 |
| calibration | 6 | 369 | 0.360351562 | [1024, 0] | 0 | 0 |
| calibration | 7 | 367 | 0.358398438 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 6, 5, 5, 5, 6, 7, 6]`; retained indices `[823, 299, 667, 681, 521, 323, 283, 305]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 1, 1, 1, 0, 0, 0]`; final labels `[1, 0, 1, 1, 1, 0, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[6, 5, 6, 3, 4, 5, 6, 1]`; retained indices `[281, 166, 30, 3, 624, 684, 483, 108]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 0, 1, 0, 1, 0, 0]`; final labels `[0, 1, 0, 1, 0, 1, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

### 4x budget

Accepted levels: **6/6**. Status: `ready`. Completed all requested levels: **True**.

**Attempted level 1.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -2.94861907096732 | 0.36767578125 | 3012 | 0.106762148 | 8 | 0.146082337 | 11757/20480 |
| calibration | -2.94861907096732 | 0.383666992188 | 3143 | 0.0496818679 | 8 | 0.135539294 | 11966/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 390 | 0.380859375 | [446, 578] | 78 | 109 |
| selection | 1 | 440 | 0.4296875 | [618, 406] | 80 | 127 |
| selection | 2 | 366 | 0.357421875 | [547, 477] | 93 | 125 |
| selection | 3 | 297 | 0.290039062 | [509, 515] | 113 | 172 |
| selection | 4 | 359 | 0.350585938 | [396, 628] | 101 | 164 |
| selection | 5 | 389 | 0.379882812 | [572, 452] | 69 | 111 |
| selection | 6 | 381 | 0.372070312 | [535, 489] | 87 | 133 |
| selection | 7 | 390 | 0.380859375 | [371, 653] | 85 | 132 |
| calibration | 0 | 426 | 0.416015625 | [499, 525] | 107 | 143 |
| calibration | 1 | 378 | 0.369140625 | [416, 608] | 82 | 119 |
| calibration | 2 | 368 | 0.359375 | [482, 542] | 102 | 170 |
| calibration | 3 | 410 | 0.400390625 | [528, 496] | 82 | 124 |
| calibration | 4 | 406 | 0.396484375 | [403, 621] | 79 | 116 |
| calibration | 5 | 383 | 0.374023438 | [499, 525] | 88 | 111 |
| calibration | 6 | 392 | 0.3828125 | [513, 511] | 98 | 149 |
| calibration | 7 | 380 | 0.37109375 | [535, 489] | 88 | 135 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 2, 6, 7, 5, 0, 4, 5]`; retained indices `[8, 7, 4, 1, 9, 5, 5, 9]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 1, 1, 0, 1, 1]`; final labels `[0, 0, 1, 1, 1, 1, 0, 1]`. Retained switch total 706; physical-sweep switch total 1073. Accepted exchange recipients touching the likelihood-bearing block: 2588. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[1, 0, 6, 3, 2, 5, 6, 5]`; retained indices `[6, 7, 9, 7, 11, 14, 0, 1]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 0, 1, 1, 0, 0]`; final labels `[1, 1, 0, 0, 1, 0, 1, 1]`. Retained switch total 726; physical-sweep switch total 1067. Accepted exchange recipients touching the likelihood-bearing block: 2777. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 2.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -1.15747289831027 | 0.367797851562 | 3013 | 0.0390469255 | 8 | 0.130766678 | 11675/20480 |
| calibration | -1.15747289831027 | 0.33984375 | 2784 | 0.0780787746 | 8 | 0.141163793 | 11628/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 367 | 0.358398438 | [205, 819] | 1 | 2 |
| selection | 1 | 391 | 0.381835938 | [999, 25] | 1 | 2 |
| selection | 2 | 352 | 0.34375 | [1024, 0] | 0 | 0 |
| selection | 3 | 385 | 0.375976562 | [486, 538] | 1 | 1 |
| selection | 4 | 386 | 0.376953125 | [996, 28] | 1 | 1 |
| selection | 5 | 394 | 0.384765625 | [356, 668] | 1 | 1 |
| selection | 6 | 365 | 0.356445312 | [709, 315] | 2 | 3 |
| selection | 7 | 373 | 0.364257812 | [418, 606] | 1 | 1 |
| calibration | 0 | 329 | 0.321289062 | [416, 608] | 2 | 3 |
| calibration | 1 | 361 | 0.352539062 | [397, 627] | 2 | 2 |
| calibration | 2 | 348 | 0.33984375 | [487, 537] | 2 | 2 |
| calibration | 3 | 348 | 0.33984375 | [136, 888] | 4 | 4 |
| calibration | 4 | 393 | 0.383789062 | [649, 375] | 5 | 5 |
| calibration | 5 | 362 | 0.353515625 | [228, 796] | 1 | 2 |
| calibration | 6 | 299 | 0.291992188 | [1024, 0] | 0 | 1 |
| calibration | 7 | 344 | 0.3359375 | [345, 679] | 1 | 1 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 0, 2, 7, 4, 6, 0, 6]`; retained indices `[823, 482, 27, 206, 646, 131, 904, 910]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 0, 1, 0, 0, 1]`; final labels `[1, 1, 0, 1, 0, 1, 1, 0]`. Retained switch total 8; physical-sweep switch total 11. Accepted exchange recipients touching the likelihood-bearing block: 27. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[5, 2, 1, 2, 6, 1, 4, 0]`; retained indices `[256, 288, 295, 295, 315, 362, 31, 983]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 0, 1, 1, 0, 1, 0]`; final labels `[1, 1, 0, 1, 0, 0, 0, 1]`. Retained switch total 17; physical-sweep switch total 20. Accepted exchange recipients touching the likelihood-bearing block: 17. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 3.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9503274479309677 | 0.36767578125 | 3012 | 0.068925637 | 8 | 0.137118194 | 11685/20480 |
| calibration | -0.9503274479309677 | 0.3564453125 | 2920 | 0.0377929008 | 8 | 0.129452055 | 11737/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 396 | 0.38671875 | [0, 1024] | 0 | 0 |
| selection | 1 | 377 | 0.368164062 | [0, 1024] | 0 | 0 |
| selection | 2 | 354 | 0.345703125 | [0, 1024] | 0 | 0 |
| selection | 3 | 363 | 0.354492188 | [420, 604] | 1 | 1 |
| selection | 4 | 413 | 0.403320312 | [1024, 0] | 0 | 0 |
| selection | 5 | 334 | 0.326171875 | [1024, 0] | 0 | 0 |
| selection | 6 | 399 | 0.389648438 | [0, 1024] | 0 | 0 |
| selection | 7 | 376 | 0.3671875 | [1024, 0] | 0 | 0 |
| calibration | 0 | 377 | 0.368164062 | [724, 300] | 1 | 1 |
| calibration | 1 | 355 | 0.346679688 | [1024, 0] | 0 | 0 |
| calibration | 2 | 375 | 0.366210938 | [1024, 0] | 0 | 0 |
| calibration | 3 | 370 | 0.361328125 | [1024, 0] | 0 | 0 |
| calibration | 4 | 337 | 0.329101562 | [0, 1024] | 0 | 0 |
| calibration | 5 | 366 | 0.357421875 | [0, 1024] | 0 | 0 |
| calibration | 6 | 378 | 0.369140625 | [0, 1024] | 0 | 0 |
| calibration | 7 | 362 | 0.353515625 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 0, 5, 5, 1, 1, 0, 7]`; retained indices `[367, 823, 526, 332, 454, 683, 231, 970]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 1, 0, 0, 0, 1, 0]`; final labels `[1, 1, 1, 1, 0, 0, 1, 0]`. Retained switch total 1; physical-sweep switch total 1. Accepted exchange recipients touching the likelihood-bearing block: 2. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[2, 6, 1, 1, 1, 5, 1, 0]`; retained indices `[263, 252, 592, 765, 249, 98, 333, 366]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 0, 0, 0, 1, 1, 1, 0]`; final labels `[1, 0, 0, 0, 1, 1, 1, 0]`. Retained switch total 1; physical-sweep switch total 1. Accepted exchange recipients touching the likelihood-bearing block: 1. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 4.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9232886767705049 | 0.367797851562 | 3013 | 0.0437195454 | 8 | 0.133089944 | 11866/20480 |
| calibration | -0.9232886767705049 | 0.36669921875 | 3004 | 0.0455296019 | 8 | 0.132822903 | 11845/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 357 | 0.348632812 | [1024, 0] | 0 | 0 |
| selection | 1 | 387 | 0.377929688 | [0, 1024] | 0 | 0 |
| selection | 2 | 364 | 0.35546875 | [1024, 0] | 0 | 0 |
| selection | 3 | 365 | 0.356445312 | [0, 1024] | 0 | 0 |
| selection | 4 | 394 | 0.384765625 | [0, 1024] | 0 | 0 |
| selection | 5 | 382 | 0.373046875 | [0, 1024] | 0 | 0 |
| selection | 6 | 363 | 0.354492188 | [0, 1024] | 0 | 0 |
| selection | 7 | 401 | 0.391601562 | [1024, 0] | 0 | 0 |
| calibration | 0 | 399 | 0.389648438 | [1024, 0] | 0 | 0 |
| calibration | 1 | 349 | 0.340820312 | [0, 1024] | 0 | 0 |
| calibration | 2 | 381 | 0.372070312 | [0, 1024] | 0 | 0 |
| calibration | 3 | 364 | 0.35546875 | [1024, 0] | 0 | 0 |
| calibration | 4 | 377 | 0.368164062 | [0, 1024] | 0 | 0 |
| calibration | 5 | 390 | 0.380859375 | [1024, 0] | 0 | 0 |
| calibration | 6 | 358 | 0.349609375 | [1024, 0] | 0 | 0 |
| calibration | 7 | 386 | 0.376953125 | [1024, 0] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 0, 7, 6, 3, 3, 6, 4]`; retained indices `[848, 839, 717, 348, 627, 976, 446, 888]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 0, 1, 1, 1, 1, 0]`; final labels `[0, 1, 0, 1, 1, 1, 1, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 1. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[3, 6, 6, 7, 5, 3, 2, 0]`; retained indices `[473, 664, 742, 336, 854, 548, 38, 611]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 1, 0, 1, 0, 0, 0]`; final labels `[0, 1, 1, 0, 1, 0, 0, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 1. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 5.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9195265684766354 | 0.36767578125 | 3012 | 0.0898242898 | 8 | 0.144754316 | 11670/20480 |
| calibration | -0.9195265684766354 | 0.360595703125 | 2954 | 0.043996963 | 8 | 0.134055518 | 11569/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 327 | 0.319335938 | [0, 1024] | 0 | 0 |
| selection | 1 | 368 | 0.359375 | [0, 1024] | 0 | 0 |
| selection | 2 | 396 | 0.38671875 | [0, 1024] | 0 | 0 |
| selection | 3 | 374 | 0.365234375 | [1024, 0] | 0 | 0 |
| selection | 4 | 401 | 0.391601562 | [0, 1024] | 0 | 0 |
| selection | 5 | 436 | 0.42578125 | [1024, 0] | 0 | 0 |
| selection | 6 | 359 | 0.350585938 | [0, 1024] | 0 | 0 |
| selection | 7 | 351 | 0.342773438 | [1024, 0] | 0 | 0 |
| calibration | 0 | 383 | 0.374023438 | [0, 1024] | 0 | 0 |
| calibration | 1 | 363 | 0.354492188 | [1024, 0] | 0 | 0 |
| calibration | 2 | 379 | 0.370117188 | [1024, 0] | 0 | 0 |
| calibration | 3 | 373 | 0.364257812 | [0, 1024] | 0 | 0 |
| calibration | 4 | 350 | 0.341796875 | [1024, 0] | 0 | 0 |
| calibration | 5 | 396 | 0.38671875 | [0, 1024] | 0 | 0 |
| calibration | 6 | 352 | 0.34375 | [0, 1024] | 0 | 0 |
| calibration | 7 | 358 | 0.349609375 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[6, 5, 1, 2, 6, 0, 4, 0]`; retained indices `[143, 911, 121, 995, 425, 1018, 410, 515]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 1, 0, 1, 0, 1, 0]`; final labels `[1, 1, 1, 0, 1, 0, 1, 0]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[2, 0, 0, 4, 5, 4, 2, 1]`; retained indices `[524, 431, 467, 443, 321, 891, 157, 718]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 0, 0, 1, 0, 1, 1, 1]`; final labels `[1, 0, 0, 1, 0, 1, 1, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 0. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

**Attempted level 6.**

| Bank | Candidate log threshold | Compression | Survivors | Survivor CV | Nonzero walkers | Largest survivor share | Accepted exchanges / proposals |
|---|---:|---:|---:|---:|---:|---:|---|
| selection | -0.9190227334320126 | 0.367797851562 | 3013 | 0.0463364442 | 8 | 0.13740458 | 11613/20480 |
| calibration | -0.9190227334320126 | 0.379150390625 | 3106 | 0.0489036382 | 8 | 0.132002576 | 11845/20480 |

| Bank | Walker | Survivors | Survival fraction | M1/M2 retained counts | Retained mode switches | Physical-sweep mode switches |
|---|---:|---:|---:|---|---:|---:|
| selection | 0 | 385 | 0.375976562 | [1024, 0] | 0 | 0 |
| selection | 1 | 383 | 0.374023438 | [0, 1024] | 0 | 0 |
| selection | 2 | 414 | 0.404296875 | [0, 1024] | 0 | 0 |
| selection | 3 | 368 | 0.359375 | [1024, 0] | 0 | 0 |
| selection | 4 | 364 | 0.35546875 | [0, 1024] | 0 | 0 |
| selection | 5 | 360 | 0.3515625 | [0, 1024] | 0 | 0 |
| selection | 6 | 367 | 0.358398438 | [0, 1024] | 0 | 0 |
| selection | 7 | 372 | 0.36328125 | [0, 1024] | 0 | 0 |
| calibration | 0 | 397 | 0.387695312 | [0, 1024] | 0 | 0 |
| calibration | 1 | 363 | 0.354492188 | [0, 1024] | 0 | 0 |
| calibration | 2 | 368 | 0.359375 | [1024, 0] | 0 | 0 |
| calibration | 3 | 401 | 0.391601562 | [1024, 0] | 0 | 0 |
| calibration | 4 | 369 | 0.360351562 | [1024, 0] | 0 | 0 |
| calibration | 5 | 410 | 0.400390625 | [0, 1024] | 0 | 0 |
| calibration | 6 | 408 | 0.3984375 | [0, 1024] | 0 | 0 |
| calibration | 7 | 390 | 0.380859375 | [0, 1024] | 0 | 0 |

Immediate pooled-promotion donor provenance (walker and retained index, zero-based):

- selection: donors `[5, 1, 4, 7, 6, 4, 0, 4]`; retained indices `[389, 626, 545, 89, 674, 834, 210, 354]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[0, 1, 1, 0, 1, 1, 1, 1]`; final labels `[0, 1, 1, 0, 1, 1, 1, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 6. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.
- calibration: donors `[3, 6, 2, 2, 2, 3, 7, 0]`; retained indices `[793, 354, 948, 786, 514, 133, 384, 576]`. Prospective starts are recorded even if this bank was not run.
  Initial component labels `[1, 1, 0, 0, 0, 1, 1, 1]`; final labels `[1, 1, 0, 0, 0, 1, 1, 1]`. Retained switch total 0; physical-sweep switch total 0. Accepted exchange recipients touching the likelihood-bearing block: 4. Surviving exact initial-block ancestry at endpoint: 0 of 72 blocks; likelihood-bearing blocks: 0 of 8.

## Cross-budget interpretation

**CASE C: nonmonotonic or ambiguous: no production change justified.**

| Common level | Bank | ESS [1x,2x,4x] | Survivor CV [1x,2x,4x] | Retained switches [1x,2x,4x] | All monotonic conditions |
|---|---|---|---|---|---|
| 1 | selection | [268.86007, 579.535593, 1207.06366] | [0.226273, 0.154038, 0.106762] | [225, 389, 706] | True |
| 1 | calibration | [333.574461, 594.504652, 1589.505236] | [0.190093, 0.134278, 0.049682] | [168, 352, 726] | True |
| 2 | selection | [855.458392, 1678.835414, 3734.478434] | [0.120751, 0.090575, 0.039047] | [0, 6, 8] | True |
| 2 | calibration | [1082.959759, 1349.270535, 2549.126935] | [0.094282, 0.097393, 0.078079] | [1, 6, 17] | False |
| 3 | selection | [1143.812175, 1703.343194, 2896.030972] | [0.108058, 0.08985, 0.068926] | [0, 1, 1] | True |
| 3 | calibration | [1359.561385, 2663.417384, 4947.035628] | [0.08957, 0.072487, 0.037793] | [0, 0, 1] | True |
| 4 | selection | [1330.508719, 2641.237472, 4339.480311] | [0.093562, 0.071822, 0.04372] | [0, 0, 0] | True |
| 4 | calibration | [1543.017751, 2099.052657, 4726.03511] | [0.075708, 0.064894, 0.04553] | [0, 0, 0] | True |
| 5 | selection | [1575.368934, 2793.750244, 1705.207344] | [0.067678, 0.061503, 0.089824] | [0, 0, 0] | False |
| 5 | calibration | [1531.081789, 2977.497427, 4786.754499] | [0.066054, 0.065681, 0.043997] | [0, 0, 0] | True |
| 6 | selection | [1061.908058, 2986.234839, 6404.590384] | [0.113825, 0.054857, 0.046336] | [0, 0, 0] | True |
| 6 | calibration | [1514.05525, 2641.81154, 5477.50208] | [0.086947, 0.05307, 0.048904] | [0, 0, 0] | True |

The full JSON includes every gate/ESS record, mode occupancy and switching count, initial-block survival curve, source hashes and promotion multiplicities. The runtime directory additionally preserves raw retained populations, physical-sweep mode labels and individual exchange decisions for every attempted sampled bank, including failed candidates.

Exact initial-block tokens are invalidated by slice updates and copied on accepted exchanges. Their disappearance means an initial block no longer survives verbatim; it does not prove equilibrium or independent sampling. Modal persistence can survive after every exact initial coordinate block is gone. Physical-sweep switching includes burn-in; retained switching counts omit within-interval excursions. Counts per physical sweep need not increase just because a longer schedule samples more transitions.

## Primary robustness answers by budget

| Budget | A: all attempted selection ESS >=20 | B: all reached calibration ESS >=20 | C: worst largest survivor share | D: minimum retained switches per bank | E: heterogeneity trend | F: full six levels |
|---|---|---|---:|---:|---|---|
| 1x | True | True | 0.157039 | 0 | See frozen common-level comparisons above | True |
| 2x | True | True | 0.151385 | 0 | See frozen common-level comparisons above | True |
| 4x | True | True | 0.146082 | 0 | See frozen common-level comparisons above | True |

A passed ESS gate is a reliability diagnostic, not proof that every walker traversed both modes. Calibration unreached after a selection failure is not counted as a calibration success for that attempted level.

## Restart and scope

{
  "checkpoint_level": 1,
  "checks": {
    "candidate_and_ESS_and_provenance_diagnostics": true,
    "candidate_record_2": true,
    "candidate_record_3": true,
    "candidate_record_4": true,
    "candidate_record_5": true,
    "candidate_record_6": true,
    "checkpoint_payload_2": true,
    "checkpoint_payload_3": true,
    "checkpoint_payload_4": true,
    "checkpoint_payload_5": true,
    "checkpoint_payload_6": true,
    "log_masses": true,
    "new_process": false,
    "records": true,
    "retained_populations_2": true,
    "retained_populations_3": true,
    "retained_populations_4": true,
    "retained_populations_5": true,
    "retained_populations_6": true,
    "thresholds": true
  },
  "code_design_seeds_and_budgets_unchanged": true,
  "corrected_command": [
    "/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python",
    "-m",
    "examples.dns_ladder_robustness_study",
    "--worker",
    "4",
    "/tmp/dns-stage9a-development/4x/ladder/attempt_000000/checkpoint"
  ],
  "factor": 4,
  "launcher_failure_reason": "Direct script import resolved another installed checkout; ImportError before branch() or sampling.",
  "no_stochastic_retry": true,
  "numerical_restart_branches": 1,
  "passed": false,
  "pre_sampling_launcher_failure": {
    "no_retry": true,
    "passed": false,
    "returncode": 1
  }
}

The saved continuation uses the predeclared completed level-1 checkpoint of the highest budget that completed six levels. A genuinely fresh OS Python process is NOT PROVEN by the saved provenance. It is a reproducibility check, not a repeated development trial. No alternate seeds or stochastic configuration/restart retries are included. The original direct-script restart launcher failed to import the intended checkout before any sampling; the one numerical restart used the module entry point from this workspace. That launch-only harness error and corrected command are recorded explicitly; no scientific source or frozen design was modified.

All pinned scientific sources, original reports and design hashes were checked unchanged after execution. No Stage-8H trajectory or stopping decision was regenerated. There is no termination-validity conclusion, production configuration change or LISA authorization from this development study. Any new independently held-out end-to-end validation would require a separately frozen protocol and explicit authorization; this study cannot retroactively validate the failed held-out run.

## Saved-output schedule verification and scientific limitation

The frozen design predates sampling. Initial banks and first-level preflights match exactly across factors. The common PCG64 label/MH prefixes match exactly. The first-level retained positions and both caches also match exactly at shared physical sweep times: 384 times/walker for 1x versus 2x, and 64 for 1x versus 4x, independently for both banks. Complete first-level physical-sweep mode-label prefixes match. These checks use saved arrays only; no transition was rerun.

The individual likelihood peaks are both `1/sqrt(2*pi)=0.3989422804014327`, because weight/width is one for each component. Similar conditional within-mode tail fractions can therefore give high compression ESS even when modal allocations are poorly explored. This is a limitation of what this development toy and the compression diagnostic reveal, not a new gate definition. The toy was frozen before these outcomes and is not replaced.

Passing ESS and finishing six levels is not evidence that walkers mixed across disconnected modes; direct occupancy/switching diagnostics must be read separately.

This development model did not reproduce Stage-8H ESS collapse; it cannot establish that extra sweeps resolve that historical failure.

CASE C does not justify a production change or a new end-to-end termination validation on current evidence.

No tested budget is established as robust across modes merely by completing the six ESS-gated levels. In particular, the smallest factor passing ESS gates must not be conflated with a smallest factor demonstrating modal robustness. Historical Stage-8H FAIL and termination UNVALIDATED remain unchanged.

## Development disposition

All three configurations completed six levels, with selection and calibration ESS >=20 at every attempt, and with every walker supplying candidate survivors. No failed level or calibration-unreached case occurred in this study. This does not reproduce the Stage-8H collapse. Selection ESS at level 5 was 1575.368934 (1x), 2793.750244 (2x), and 1705.207344 (4x), giving a nonmonotonic response. Calibration survivor CV at level 2 also rose from 0.094282 (1x) to 0.097393 (2x) before falling to 0.078079 (4x).

Longer schedules increased early-level mode switching: retained totals at level 1 were selection [225,389,706] and calibration [168,352,726] for [1x,2x,4x]. Yet levels 4, 5 and 6 had zero mode switches in both banks under every budget, even across the entire physical schedule including burn-in and the initial promoted-start-to-first-sweep transition. The original post-sweep switch counts omit that initial transition; the JSON now also records the saved-data reduction including it. This distinction changes no candidate, ESS, classification or trajectory.

CASE C is retained exactly as predeclared. 1x is the smallest factor completing all ESS gates; no tested factor establishes robust inter-mode mixing. Additional transitions show a plausible early-level benefit but do not resolve the deep modal persistence observed here. Compression ESS alone is not a certificate of modal exploration. No production setting change or new independently held-out end-to-end validation is justified by this ambiguous development result. Further scientific work would require separate authorization; no additional trajectory is part of this study.

Common-level comparisons are between the same construction index, not identical likelihood contours: the frozen automatic candidates can differ across budgets. This is a single matched-seed development comparison, not a statistical proof of a population-average monotonic budget response. The equal component peak heights are an explicit interpretive limitation, not a reason to replace or tune this development toy after observing it.

## Final saved-artifact restart audit

**NUMERICALLY MATCHED / PROCESS PROVENANCE NOT ESTABLISHED. Restart reproducibility: NOT PROVEN.**

The comparator computes `new_process = other['pid'] != result['pid']` (study script line 236). Both saved result files record PID `3`; sampling_started.json also records `3`. No parent PID, process start time, PID namespace identity, or independent launch receipt is saved. All 56 4x and 46 restart JSON files were inspected recursively. Checkpoint metadata contains scientific/environment/seed/stream/hash records, without OS process identity. The module launch command is saved, but no corrected-launch subprocess identity is recorded. The initial `subprocess.run` attempt failed before sampling and cannot establish the process identity of the later successful continuation.

The baseline result mtime is 2026-10-06 11:18:10 UTC; the continuation result mtime is 11:27:15 UTC. The separate continuation log records levels 2–6. These establish later saved outputs, not an independently identified OS process. File birth times are unavailable; modification times are not process creation times. PID reuse or separate PID namespaces are possible explanations, not established facts. Numerical equality is not used as evidence of a new process. A comparator/provenance-check defect is therefore not established; the original `new_process=false` and `passed=false` checks are preserved as historical comparator results, without treating them as numerical divergence. No sampler or comparator launcher was executed.

| Saved comparison | Exact result |
|---|---|
| Thresholds and log masses | Identical |
| Construction records | Identical |
| Candidate records/thresholds, levels 2–6 | Identical |
| ESS and full continuation diagnostic records, levels 2–6 | Identical |
| Promotion/provenance, preflight, candidate hashes and decisions, levels 2–6 | Identical |
| Checkpoint envelopes/payloads, levels 2–6 | Identical |
| Retained population arrays, levels 2–6 | Identical key sets, dtype, shape and bytes |

Every numerical quantity compared so far is exactly identical. JSON comparisons are canonical and have no numerical tolerance. The audit rechecked the saved comparator results and additional saved provenance records; it did not generate another trajectory. Evidence hashes and precise artifact mtimes are included in the JSON report.

## Final Stage-9A answers

1. **Is baseline mixing insufficient?** Baseline cross-mode mixing is insufficient at deep levels: no switching at levels 4–6. Baseline passed every ESS gate, so an ESS-budget deficiency is not established by this study.

2. **Does increasing only the mixing budget solve the ESS/mixing problem?** Not demonstrated. All budgets pass ESS gates; 2x and 4x still have zero deep modal switching. This development model did not reproduce the historical Stage-8H ESS collapse.

3. **Is improvement monotonic and physically interpretable?** No robust monotonic improvement under the predeclared criteria. Early switching improves, but level-5 selection ESS and level-2 calibration survivor CV are nonmonotonic; deep persistence remains.

4. **What is the smallest robust tested budget?** None established. 1x is the smallest tested budget passing all six ESS-gated levels, but no tested budget establishes cross-mode robustness.

5. **Is another independently held-out end-to-end validation justified?** No, current CASE-C evidence does not justify another independently held-out end-to-end validation or a production budget change.

Compression ESS can be high even when walkers remain trapped in separate modes. Therefore ESS reliability and cross-mode mixing diagnostics are different requirements. The fact that all walkers contribute candidate survivors does not prove cross-mode ergodicity.

Final CASE C is frozen: nonmonotonic / ambiguous; no production change justified. All three budgets completed all six requested levels, and all three lacked deep modal switching at levels 4–6. No new sampling, kernel changes, sweep experiments, termination validation, LISA, or production action was performed. The frozen design JSON, design SHA256 file, and study script are unchanged.
