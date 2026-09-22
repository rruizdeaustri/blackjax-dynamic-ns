# Stage-4J: controlled frozen differential-direction benchmark

This stage resumes the three existing untracked Stage-4J implementation/test files. No earlier sampling was repeated. The partial implementation already contained the reference snapshot, baseline masks, post-mask normalization, fallback, start/cache equality checks, and a bounded runner. An independent fair pair reversal was added before execution to guarantee exact sign symmetry even in the presence of finite-precision bounded-integer RNG effects.

Only the slice-direction orientation changes. The generic DNS core, constrained-slice wrapper, slice stepping/shrinkage, and frozen production proposals remain byte-identical to HEAD; the production-proposal source also matches the saved Stage-4H hash. All 111 relevant tests (107 existing plus four differential tests) passed before GPU execution. The four tests cover frozen snapshots, bank separation, distinct reference indices, pair-reversal symmetry, baseline component masks, post-mask norm, deterministic isotropic fallback, and independence from current position.

The frozen reference comprises all 753 distinct retained `level9_selection_trace.npz` states with logL strictly above ell9 = −110252.99476697217. This trace originally constructed level 9 at level 8; its strict survivors supply the reference orientations at level 9. They are not IID samples or guaranteed equilibrated samples at that contour. There are no shared rows with evaluation starts. Selection and calibration used separate walker streams, but inherit the same adaptive ladder; independence is conditional on that construction history, not an unconditional claim.

The eight evaluation starts, including prior/likelihood caches, match the Stage-4H continuation inputs both numerically and byte-for-byte. The baseline is the saved Stage-4H isotropic continuation and is not rerun. Both runs use seeds 810100–810107, giving the same component masks and outer slice RNG streams. The reference-pair RNG uses a separate fixed domain tag `fold_in(direction_key, 0x4D1F)`. Slice loops can consume different numbers of variates as their paths diverge; this is common-random-number control, not independent replicate experiments.

For each proposal, two distinct frozen reference indices are drawn independently of the current walker. A fair reversal symmetrizes their ordered pair, guaranteeing equal probability for (a,b) and (b,a). The existing 20/40/40 global/single-label/unordered-pair mask is applied to reference[a]−reference[b], then the active difference is normalized to Euclidean norm pi/sqrt(3). Reversing the pair reverses every nondegenerate direction exactly. A masked norm ≤1e-12 invokes the ordinary isotropic direction drawn from the baseline RNG; the criterion is sign-invariant and the fallback distribution is symmetric. Fallback uses the same key deterministically and is counted. No matching, sorting, current-state-dependent reference selection, covariance tuning, or multiplier is introduced into the kernel.

The differential run is exactly 512 steps per walker, with no samples discarded and no follow-up sampling. The rejected ell10 = −109401.11425363769 is used only to report exceedances. Every step checks finite prior/likelihood and coordinates, strict ell9 membership, scalar/shape parameter contract, direct prior/likelihood cache consistency, slice acceptance, and direction norm.

The fixed finite reference mixture gives symmetric, current-state-independent straight-line slice orientations. This is the same constrained-prior invariance argument as the baseline mixture; it does not guarantee adequate mixing or coverage. Reference geometry may itself reflect limited selection-bank exploration.

Read-only comparison reports first 256, second 256, and all 512 steps. Frequency ESS uses the existing short-history autocorrelation diagnostic. Permutation-invariant frequency distances sort each actual catalogue for analysis only and use RMS distances in Fourier-bin units, df = 1/31536000 Hz. Cross-walker window distances average squared matched distances over all cross-time pairs; they include within-walker variation. Half-to-half source-set movement also reports distance between the means of sorted catalogues. Neither ESS nor geometric proximity establishes global mixing.

## Result: D — worse on likelihood agreement

The frozen empirical orientations did not improve constrained-walker communication in this benchmark. Exceedance disagreement is essentially unchanged over all 512 steps and slightly higher in the second half. Between-walker logL-mean disagreement increases substantially, despite a slightly lower slice likelihood cost proxy. This earns D under the predeclared “increased disagreement” category. It does not establish that every empirical orientation bank is worse, or that the isotropic kernel is adequate.

| Kernel / window | Exceedance SD | logL-mean SD | Median / minimum f0_u ESS | Likelihood cost proxy | Mean num_steps / num_shrink |
|---|---:|---:|---:|---:|---:|
| isotropic / first256 | 0.264798 | 439.037 | 5.386 / 2.806 | 13449 | 3.49268 / 3.07422 |
| isotropic / second256 | 0.270649 | 559.401 | 5.248 / 2.810 | 13893 | 3.55957 / 3.22412 |
| isotropic / all512 | 0.252070 | 446.210 | 6.923 / 2.895 | 27342 | 3.52612 / 3.14917 |
| differential / first256 | 0.249690 | 874.388 | 5.638 / 2.957 | 13316 | 3.52979 / 2.97217 |
| differential / second256 | 0.291656 | 815.480 | 6.276 / 2.991 | 13354 | 3.61816 / 2.90234 |
| differential / all512 | 0.250991 | 767.274 | 6.656 / 3.207 | 26670 | 3.57397 / 2.93726 |

Across all 512 steps, exceedance SD changes 0.25207 → 0.25099 (−0.4%), while logL-mean SD changes 446.21 → 767.27 (+72.0%). In the second half, exceedance SD changes 0.27065 → 0.29166 (+7.8%) and logL-mean SD changes 559.40 → 815.48 (+45.8%). Differential second-half fractions span 0–0.86719; the baseline spans 0.015625–0.70703125. Median frequency ESS improves in the second half but declines slightly over the full 512 steps. No consistent multi-metric convergence improvement is observed.

## Per-walker comparisons

All samples are used; the window split is descriptive and no new burn-in cutoff is applied.

| Walker | Isotropic exceedance first / second / all | Differential exceedance first / second / all |
|---|---:|---:|
| 0 | 0.441406 / 0.246094 / 0.343750 | 0.734375 / 0.621094 / 0.677734 |
| 1 | 0.554688 / 0.582031 / 0.568359 | 0.152344 / 0.613281 / 0.382812 |
| 2 | 0.652344 / 0.707031 / 0.679688 | 0.664062 / 0.730469 / 0.697266 |
| 3 | 0.023438 / 0.015625 / 0.019531 | 0.277344 / 0.207031 / 0.242188 |
| 4 | 0.148438 / 0.449219 / 0.298828 | 0.414062 / 0.699219 / 0.556641 |
| 5 | 0.691406 / 0.585938 / 0.638672 | 0.566406 / 0.867188 / 0.716797 |
| 6 | 0.375000 / 0.078125 / 0.226562 | 0.234375 / 0.414062 / 0.324219 |
| 7 | 0.054688 / 0.085938 / 0.070312 | 0.046875 / 0.000000 / 0.023438 |

| Walker | Isotropic mean logL first / second / all | Differential mean logL first / second / all |
|---|---:|---:|
| 0 | -109388.161 / -109688.027 / -109538.094 | -108595.458 / -108788.288 / -108691.873 |
| 1 | -109332.351 / -108401.521 / -108866.936 | -109645.301 / -108772.692 / -109208.996 |
| 2 | -109074.875 / -108972.706 / -109023.790 | -107200.428 / -107986.910 / -107593.669 |
| 3 | -109897.264 / -109986.700 / -109941.982 | -109540.264 / -109713.225 / -109626.744 |
| 4 | -109794.438 / -109161.683 / -109478.060 | -109375.400 / -108671.836 / -109023.618 |
| 5 | -108632.630 / -108980.737 / -108806.683 | -109091.400 / -107693.034 / -108392.217 |
| 6 | -109527.296 / -109788.525 / -109657.911 | -109643.118 / -109436.956 / -109540.037 |
| 7 | -109895.590 / -109864.805 / -109880.197 | -109894.468 / -110055.414 / -109974.941 |

| Walker | Isotropic early-to-late mean logL change | Differential early-to-late mean logL change |
|---|---:|---:|
| 0 | -299.866 | -192.830 |
| 1 | +930.830 | +872.608 |
| 2 | +102.169 | -786.482 |
| 3 | -89.436 | -172.962 |
| 4 | +632.755 | +703.565 |
| 5 | -348.107 | +1398.365 |
| 6 | -261.228 | +206.163 |
| 7 | +30.785 | -160.947 |

There are individual gains: walker 3 has higher exceedance than its nearly stationary baseline, and walkers 4 and 6 also exceed more often. However, walker 7 has zero second-half exceedances, while walkers 2 and 5 remain high. Walker 5’s mean logL rises by 1398.37 between halves and walker 2’s falls by 786.48. This is not coherent convergence toward common behavior. The likelihood-spread result is not solely a single extreme walker: second-half median absolute deviation of walker mean logL rises 442.03 → 725.02, and IQR rises 828.87 → 1005.42. These are descriptive checks, not independent hypothesis tests.

## Permutation-invariant physical source-set diagnostics

| Window | Isotropic / differential median pairwise cross-state f0 RMS (bins) | Isotropic / differential median distance between mean ordered f0 vectors (bins) |
|---|---:|---:|
| first256 | 94.397 / 101.948 | 44.330 / 51.928 |
| second256 | 103.429 / 88.245 | 45.808 / 41.670 |
| all512 | 108.162 / 99.069 | 37.090 / 35.925 |

The final-half median matched frequency distance decreases 103.43 → 88.24 bins (−14.7%). Actual final-state median pair distance decreases 136.90 → 67.92 bins; single endpoints are more variable than window summaries. Frequency-set proximity improves somewhat while likelihood agreement deteriorates, consistent with Stage-4I’s finding that frequency similarity is not sufficient for likelihood similarity. This benchmark does not infer physical catalogue families.

| Walker | Half-to-half ordered-f0 mean movement, isotropic / differential (bins) | Start-to-end matched f0 RMS, isotropic / differential (bins) |
|---|---:|---:|
| 0 | 28.995 / 24.109 | 85.808 / 57.694 |
| 1 | 56.860 / 20.613 | 106.705 / 72.155 |
| 2 | 16.107 / 47.735 | 123.167 / 80.868 |
| 3 | 31.521 / 16.896 | 123.261 / 47.127 |
| 4 | 68.203 / 76.306 | 172.797 / 121.598 |
| 5 | 121.095 / 94.678 | 100.958 / 65.851 |
| 6 | 25.776 / 42.176 | 105.564 / 88.835 |
| 7 | 120.182 / 39.804 | 162.591 / 114.816 |

All 28 final walker pairs are reported below using both the final 256-step window and actual final states. Distances are permutation-invariant frequency RMS in bins. Full pairwise arrays for every window are in comparison.json.

| Pair | Final-window isotropic / differential | Final-state isotropic / differential |
|---|---:|---:|
| 0–1 | 90.772 / 76.814 | 88.242 / 21.951 |
| 0–2 | 85.587 / 92.874 | 85.486 / 102.340 |
| 0–3 | 101.416 / 88.797 | 185.501 / 66.955 |
| 0–4 | 87.099 / 82.091 | 61.632 / 48.061 |
| 0–5 | 132.379 / 105.074 | 134.888 / 55.639 |
| 0–6 | 81.463 / 100.661 | 138.903 / 50.647 |
| 0–7 | 153.204 / 68.405 | 122.106 / 61.449 |
| 1–2 | 95.583 / 91.465 | 72.700 / 100.643 |
| 1–3 | 111.102 / 76.088 | 185.236 / 68.675 |
| 1–4 | 95.637 / 71.297 | 92.029 / 36.327 |
| 1–5 | 126.236 / 94.892 | 114.727 / 49.800 |
| 1–6 | 87.909 / 87.692 | 165.380 / 49.783 |
| 1–7 | 144.184 / 65.114 | 148.885 / 53.258 |
| 2–3 | 101.817 / 113.931 | 226.261 / 106.244 |
| 2–4 | 84.644 / 96.383 | 94.934 / 101.065 |
| 2–5 | 151.384 / 124.277 | 159.668 / 77.176 |
| 2–6 | 83.488 / 122.141 | 182.026 / 107.033 |
| 2–7 | 174.955 / 89.646 | 175.403 / 97.474 |
| 3–4 | 104.066 / 73.680 | 216.835 / 86.130 |
| 3–5 | 154.095 / 85.731 | 112.010 / 48.784 |
| 3–6 | 101.591 / 75.430 | 102.925 / 106.323 |
| 3–7 | 175.623 / 70.963 | 96.558 / 93.577 |
| 4–5 | 141.023 / 93.986 | 162.919 / 67.165 |
| 4–6 | 80.065 / 87.474 | 175.164 / 51.185 |
| 4–7 | 164.348 / 66.840 | 155.070 / 75.001 |
| 5–6 | 128.656 / 97.276 | 139.955 / 81.126 |
| 5–7 | 102.793 / 91.710 | 117.381 / 62.056 |
| 6–7 | 151.030 / 85.420 | 47.447 / 75.151 |

## Cost and integrity

| Kernel | Slice likelihood proxy | Direct cache-check evaluations | Extra initialization likelihood calls | Fallbacks |
|---|---:|---:|---:|---:|
| Saved isotropic baseline | 27342 | 4104 | 0 | N/A |
| New differential benchmark | 26670 | 4104 | 0 | 16 |

Only the differential row is newly incurred. Its likelihood proxy is 2.46% lower. Direct checks include all 4096 parameter steps and 8 cached initialization states; each check evaluates both likelihood and prior. These proxy counts do not measure the reference-generation overhead or establish a wall-clock speedup. Slice counter quantiles are in comparison.json.

The predetermined isotropic fallback occurred 16/4096 times (0.390625%). Maximum direction-norm error was 4.44e-16. All sampled reference pairs have distinct indices. The reference snapshot is unchanged, evaluation starts match byte-for-byte, and every proposal’s component and label mask matches the saved baseline. All input hashes and protected code files were rechecked after execution.

Contour, cache, nonfinite, parameter-contract, and slice failure counts are all zero for both baseline and differential. No integrity failure or retry occurred.

## Interpretation and artifacts

This controlled result does not support the current isotropic orientation distribution being the dominant remaining limitation that this frozen empirical bank resolves. It shows no overall mixing improvement and worse likelihood agreement for the tested bank and budget. Reference-bank limitations, continuing transients, and finite-run variability remain possible explanations; one matched eight-walker experiment cannot distinguish them or establish a universal ranking of kernels. No multiplier, bank, seed, or budget was tuned after observing the result.

Level 10 remains rejected. No new level, production trajectory, evidence, or posterior reconstruction was run. The experiment ends here.

Artifacts: `/tmp/lisa_dns_stage4j_differential/report.json`, `comparison.json`, `trace.npz`, `reference_bank.npz`, `initial_states.npz`, and `blocks.csv`. The original Stage-4H baseline files are unchanged. `compare_differential_benchmark.py` reproduces the read-only quantitative comparison without likelihood or MCMC calls.
