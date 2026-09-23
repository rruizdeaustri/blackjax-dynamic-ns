# Stage-4O: fixed slice-then-refresh composition

**Hybrid S: D — worse. Hybrid P: B — modest, mixed improvement.** The pair hybrid lowers cross-walker exceedance disagreement and pairwise likelihood KS distance while retaining large physical steps. Its full-run logL-mean spread and logL ESS do not improve, and seven of eight formerly sticky blocks are never refreshed. This is limited complementarity, not established global convergence. Level 10 remains rejected.

## Saved-trace bottleneck audit (before sampling)

The audit used only saved Stage-4H and Stage-4N data and was written before the new runner started. The same label was never successfully refreshed under both standalone kernels in every walker. All eight sticky labels were repeatedly attempted and remained byte-identical to their starts throughout Stage-4N. Counts below are attempts; every corresponding Stage-4N acceptance count is zero. Physical f0 is the same at initialization and throughout both standalone refresh chains.

| Walker | Sticky label | S attempts / accepted | P attempts / accepted | Initial/current f0 (Hz) | Stage-4H final f0 (Hz) | Stage-4H f0 range (bins) | Stage-4H block changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 53 / 0 | 121 / 0 | 0.0018407766558671 | 0.0018405198700397 | 10.211 | 182 |
| 1 | 3 | 49 / 0 | 112 / 0 | 0.0018408791753215 | 0.0018410290271471 | 15.362 | 184 |
| 2 | 3 | 64 / 0 | 109 / 0 | 0.0018408578564392 | 0.0018438910867939 | 106.501 | 160 |
| 3 | 0 | 58 / 0 | 101 / 0 | 0.0018412004191430 | 0.0018413789381436 | 6.809 | 189 |
| 4 | 8 | 64 / 0 | 113 / 0 | 0.0018407004817041 | 0.0018410989332589 | 15.547 | 173 |
| 5 | 2 | 58 / 0 | 123 / 0 | 0.0018409388651064 | 0.0018409550017948 | 17.527 | 172 |
| 6 | 7 | 59 / 0 | 117 / 0 | 0.0018412012426603 | 0.0018411384899818 | 8.129 | 177 |
| 7 | 4 | 41 / 0 | 107 / 0 | 0.0018411970565605 | 0.0018413107660991 | 10.357 | 165 |

| Walker | Stage-4H f0 net change (bins) | f0 path length (bins) | Latent-block net L2 | Latent-block path length | Maximum latent distance from start |
|---|---:|---:|---:|---:|---:|
| 0 | -8.098 | 103.925 | 1.0793 | 5.4776 | 1.1091 |
| 1 | 4.726 | 109.601 | 0.7385 | 4.5224 | 0.7400 |
| 2 | 95.656 | 203.338 | 0.9502 | 4.3499 | 0.9502 |
| 3 | 5.630 | 42.278 | 0.2090 | 1.3482 | 0.2457 |
| 4 | 12.566 | 81.708 | 0.7282 | 2.9853 | 0.7692 |
| 5 | 0.509 | 132.139 | 0.9355 | 5.8069 | 1.0885 |
| 6 | -1.979 | 81.926 | 1.1913 | 3.7414 | 1.2025 |
| 7 | 3.586 | 70.943 | 0.4720 | 2.7842 | 0.4911 |

The sticky starting frequencies span 0.0018407004817–0.0018412012427 Hz. Their pairwise separations range from 0.026 to 15.792 Fourier bins. Whole-catalogue Hungarian matching pairs sticky-to-sticky in 9/28 full-coordinate assignments and 7/28 frequency-only assignments. Thus they occupy a similar frequency region, but matching does not consistently identify them as the same physical source or establish equal likelihood importance. No named catalogue families or source-gain claims are made.

The audit supports testing complementarity: slice moves already changed all eight blocks. It did not select labels, probabilities, starts, or kernel parameters for the hybrid.

## Exact composition and frozen design

Let P be the unchanged implemented scalar prior multiplied by the strict indicator `logL > -110252.99476697217`. The validated constrained isotropic slice kernel satisfies `P K_slice = P`. Exact Logistic independence MH, with actual full `pi_impl` evaluations and analytic block proposal q, satisfies `P K_refresh = P`. Consequently:

```text
P K_slice K_refresh = (P K_slice) K_refresh = P K_refresh = P
```

This is `K_refresh o K_slice` in function-composition notation. Each constituent preserves the same target; reversibility of their deterministic composition is not required. A cheap finite-state toy test verifies two invariant MH kernels whose ordered product remains invariant while being nonreversible.

Hybrid S uses one ordinary isotropic slice transition followed by one exact single-source refresh. Hybrid P uses the same slice transition followed by one exact pair refresh. Refresh is mandatory every sweep. Both run eight walkers for exactly 512 sweeps, with no burn-in removal or extension. Stage-4H and both Stage-4N baselines were loaded, never rerun.

All starting positions and cached scalars match Stage-4H and both Stage-4N starts byte for byte. Slice uses the unchanged `proposals.build_parameter_step`, readonly scales pi/sqrt(3), 20/40/40 global/single/pair mixture, max_steps=10 and max_shrinkage=100. Refresh calls the unchanged Stage-4N `transition`, with uniform labels/pairs, direct JAX Logistic draws, stable analytic q, full implemented-prior MH ratio, strict contour rejection, and exact state/cache updates. Negative-infinite proposed prior means rejection; nonfinite draws are recorded and rejected without repair or redraw.

Single-hybrid seeds are 870100–870107; pair-hybrid seeds are 870200–870207. Each key splits into independent slice and refresh streams; refresh uses the unchanged proposal/MH-uniform splitting. Label selection has no current-state input. The saved bottleneck labels enter analysis only.

## Primary cross-walker communication

SD uses ddof=1 across eight walkers. KS is descriptive across 28 walker pairs, with no IID p-values. MAD and IQR below describe the distribution of walker mean logL; per-walker logL median/MAD/IQR and all pairwise KS distances are also retained in comparison.json. Robust summaries alone can hide an isolated walker, as the standalone refresh results demonstrate.

| Kernel | Window | LogL-mean SD | Exceedance SD | Median pairwise logL KS | Walker-mean MAD | Walker-mean IQR |
|---|---|---:|---:|---:|---:|---:|
| Isotropic | first256 | 439.037 | 0.264798 | 0.410156 | 359.782 | 551.744 |
| Isotropic | second256 | 559.401 | 0.270649 | 0.433594 | 442.034 | 828.866 |
| Isotropic | all512 | 446.210 | 0.252070 | 0.337891 | 403.013 | 728.905 |
| Refresh S | first256 | 1141.235 | 0.344681 | 0.703125 | 118.350 | 398.368 |
| Refresh S | second256 | 1133.546 | 0.338710 | 0.609375 | 82.011 | 415.727 |
| Refresh S | all512 | 1136.648 | 0.339276 | 0.652344 | 68.951 | 410.924 |
| Refresh P | first256 | 1127.876 | 0.330508 | 0.597656 | 123.235 | 351.496 |
| Refresh P | second256 | 1150.622 | 0.339834 | 0.531250 | 119.028 | 308.184 |
| Refresh P | all512 | 1138.871 | 0.333667 | 0.544922 | 121.131 | 329.840 |
| Hybrid S | first256 | 807.674 | 0.299604 | 0.425781 | 426.167 | 897.938 |
| Hybrid S | second256 | 837.112 | 0.327911 | 0.441406 | 594.561 | 1142.306 |
| Hybrid S | all512 | 801.318 | 0.300604 | 0.408203 | 599.394 | 952.588 |
| Hybrid P | first256 | 490.850 | 0.273746 | 0.402344 | 334.761 | 628.838 |
| Hybrid P | second256 | 598.350 | 0.202413 | 0.300781 | 251.241 | 443.729 |
| Hybrid P | all512 | 471.850 | 0.222468 | 0.298828 | 291.811 | 488.809 |

Hybrid S worsens all three primary agreement measures in both halves and overall. Hybrid P improves all-512 exceedance SD and median KS by about 12%; the improvements are larger in the second half (about 25% and 31%). Its logL-mean SD remains about 6% higher overall and 7% higher in the second half. First-half agreement is mostly similar or worse. The result is mixed, not a concordant strong improvement in every diagnostic.

### Hybrid per-walker first256

| Walker | Hybrid S exceedance | Hybrid S mean logL | Hybrid P exceedance | Hybrid P mean logL |
|---|---:|---:|---:|---:|
| 0 | 0.585938 | -109028.804 | 0.480469 | -109344.834 |
| 1 | 0.183594 | -109711.502 | 0.531250 | -108980.194 |
| 2 | 0.781250 | -107691.603 | 0.847656 | -108521.813 |
| 3 | 0.007812 | -109956.987 | 0.050781 | -109893.116 |
| 4 | 0.402344 | -109262.607 | 0.121094 | -109840.080 |
| 5 | 0.742188 | -108337.478 | 0.445312 | -109295.159 |
| 6 | 0.207031 | -109664.597 | 0.222656 | -109686.504 |
| 7 | 0.085938 | -109881.138 | 0.105469 | -109860.780 |
### Hybrid per-walker second256

| Walker | Hybrid S exceedance | Hybrid S mean logL | Hybrid P exceedance | Hybrid P mean logL |
|---|---:|---:|---:|---:|
| 0 | 0.523438 | -108991.875 | 0.320312 | -109303.321 |
| 1 | 0.492188 | -109027.913 | 0.667969 | -108049.250 |
| 2 | 0.750000 | -107482.452 | 0.464844 | -109405.915 |
| 3 | 0.062500 | -109904.054 | 0.019531 | -109985.174 |
| 4 | 0.820312 | -108546.201 | 0.328125 | -109177.496 |
| 5 | 0.789062 | -108664.867 | 0.304688 | -109569.728 |
| 6 | 0.195312 | -109735.324 | 0.218750 | -109679.978 |
| 7 | 0.015625 | -109968.576 | 0.105469 | -109822.442 |
### Hybrid per-walker all512

| Walker | Hybrid S exceedance | Hybrid S mean logL | Hybrid P exceedance | Hybrid P mean logL |
|---|---:|---:|---:|---:|
| 0 | 0.554688 | -109010.340 | 0.400391 | -109324.078 |
| 1 | 0.337891 | -109369.708 | 0.599609 | -108514.722 |
| 2 | 0.765625 | -107587.027 | 0.656250 | -108963.864 |
| 3 | 0.035156 | -109930.521 | 0.035156 | -109939.145 |
| 4 | 0.611328 | -108904.404 | 0.224609 | -109508.788 |
| 5 | 0.765625 | -108501.172 | 0.375000 | -109432.443 |
| 6 | 0.201172 | -109699.961 | 0.220703 | -109683.241 |
| 7 | 0.050781 | -109924.857 | 0.105469 | -109841.611 |

All corresponding per-walker baseline and standalone-refresh values are preserved in comparison.json. First-to-second-half likelihood KS distances are:

| Kernel | Walkers 0–7 |
|---|---|
| Isotropic | 0.332, 0.453, 0.168, 0.230, 0.477, 0.238, 0.328, 0.113 |
| Refresh S | 0.305, 0.168, 0.164, 0.215, 0.223, 0.133, 0.211, 0.289 |
| Refresh P | 0.246, 0.062, 0.258, 0.137, 0.137, 0.160, 0.176, 0.062 |
| Hybrid S | 0.113, 0.320, 0.141, 0.121, 0.422, 0.207, 0.086, 0.195 |
| Hybrid P | 0.246, 0.254, 0.461, 0.184, 0.258, 0.188, 0.094, 0.152 |

## Autocorrelation and physical exploration

ESS uses the same short-chain initial-positive monotone estimator as earlier stages. f0_u summaries span 72 labelled walker/coordinate series; logL summaries span eight walkers. They are descriptive and do not certify convergence.

| Kernel | Window | Median f0_u ESS | Minimum f0_u ESS | Median logL ESS | Unique-state fraction range |
|---|---|---:|---:|---:|---|
| Isotropic | first256 | 5.386 | 2.806 | 33.528 | 1.000000–1.000000 |
| Isotropic | second256 | 5.248 | 2.810 | 29.681 | 1.000000–1.000000 |
| Isotropic | all512 | 6.923 | 2.895 | 33.529 | 1.000000–1.000000 |
| Refresh S | first256 | 16.063 | 0.000 | 22.538 | 0.632812–0.914062 |
| Refresh S | second256 | 16.503 | 0.000 | 18.029 | 0.687500–0.890625 |
| Refresh S | all512 | 28.039 | 0.000 | 32.636 | 0.660156–0.898438 |
| Refresh P | first256 | 27.521 | 0.000 | 32.937 | 0.468750–0.808594 |
| Refresh P | second256 | 29.533 | 0.000 | 38.854 | 0.492188–0.781250 |
| Refresh P | all512 | 57.150 | 0.000 | 58.311 | 0.480469–0.787109 |
| Hybrid S | first256 | 20.075 | 3.369 | 31.979 | 1.000000–1.000000 |
| Hybrid S | second256 | 19.992 | 3.023 | 25.768 | 1.000000–1.000000 |
| Hybrid S | all512 | 37.584 | 3.676 | 40.171 | 1.000000–1.000000 |
| Hybrid P | first256 | 29.944 | 2.868 | 46.260 | 1.000000–1.000000 |
| Hybrid P | second256 | 30.704 | 3.607 | 23.645 | 1.000000–1.000000 |
| Hybrid P | all512 | 55.277 | 3.852 | 25.548 | 1.000000–1.000000 |

Both hybrids eliminate the exactly frozen coordinate series seen in standalone refresh. Hybrid P increases median coordinate ESS but lowers full-run median logL ESS from 33.53 to 25.55; its mixing improvement cannot be inferred from coordinate ESS alone. All eight unique-state fractions and coordinate/logL autocorrelation details are saved.

Permutation-invariant metrics reuse Stage-4I: frequency-only optimal matching equals sorting, with 1/31536000 Hz per Fourier bin; full-coordinate Hungarian matching uses the same normalized widths and circular angular differences. Frequency cross-window RMS uses every time pair. Full-coordinate RMS uses every 32nd endpoint, eight per half.

| Kernel | Window | Cross-walker frequency RMS median (bins) | Sorted-frequency mean distance median (bins) | Full-coordinate assignment RMS median |
|---|---|---:|---:|---:|
| Isotropic | first256 | 94.928 | 45.055 | 0.249369 |
| Isotropic | second256 | 104.066 | 48.034 | 0.259783 |
| Isotropic | all512 | 109.341 | 38.564 | 0.256396 |
| Refresh S | first256 | 104.265 | 25.813 | 0.259344 |
| Refresh S | second256 | 102.234 | 31.000 | 0.258018 |
| Refresh S | all512 | 104.261 | 21.434 | 0.258437 |
| Refresh P | first256 | 109.066 | 16.987 | 0.258878 |
| Refresh P | second256 | 106.946 | 15.224 | 0.256388 |
| Refresh P | all512 | 106.487 | 12.333 | 0.257445 |
| Hybrid S | first256 | 105.766 | 19.689 | 0.258989 |
| Hybrid S | second256 | 108.158 | 19.795 | 0.260621 |
| Hybrid S | all512 | 106.879 | 11.107 | 0.259050 |
| Hybrid P | first256 | 105.145 | 18.568 | 0.260853 |
| Hybrid P | second256 | 107.593 | 18.799 | 0.257231 |
| Hybrid P | all512 | 106.550 | 12.639 | 0.258628 |

All 28 pairwise final-window (second-half) distances, corresponding first-half values, and differences from Stage-4H are saved. Sorted-frequency means agree more closely under both hybrids; full-coordinate cross-walker RMS remains broadly similar. Cross-time RMS combines dispersion and separation and should not be interpreted as a standalone convergence score.

| Hybrid | Walker | Within first-half frequency RMS | Within second-half frequency RMS | Between-half sorted mean shift | Start-to-end frequency-set distance | Maximum frequency-set distance from start |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid S | 0 | 96.710 | 100.304 | 15.127 | 79.993 | 167.454 |
| Hybrid S | 1 | 112.673 | 114.595 | 38.497 | 79.076 | 208.361 |
| Hybrid S | 2 | 111.073 | 103.974 | 26.166 | 53.322 | 170.712 |
| Hybrid S | 3 | 99.987 | 105.097 | 18.242 | 75.942 | 213.499 |
| Hybrid S | 4 | 96.064 | 100.535 | 18.486 | 103.411 | 206.611 |
| Hybrid S | 5 | 96.524 | 105.583 | 32.294 | 62.013 | 237.127 |
| Hybrid S | 6 | 112.157 | 105.245 | 19.220 | 66.538 | 191.250 |
| Hybrid S | 7 | 99.691 | 112.720 | 16.342 | 183.488 | 218.457 |
| Hybrid P | 0 | 105.628 | 105.205 | 13.520 | 84.590 | 171.828 |
| Hybrid P | 1 | 99.932 | 103.387 | 11.240 | 30.762 | 209.140 |
| Hybrid P | 2 | 101.798 | 105.126 | 12.898 | 67.348 | 160.786 |
| Hybrid P | 3 | 104.309 | 102.740 | 46.179 | 105.083 | 196.898 |
| Hybrid P | 4 | 101.896 | 108.118 | 21.740 | 56.760 | 265.736 |
| Hybrid P | 5 | 104.793 | 102.080 | 12.040 | 85.697 | 222.720 |
| Hybrid P | 6 | 101.498 | 116.508 | 5.096 | 84.458 | 224.211 |
| Hybrid P | 7 | 107.416 | 96.393 | 12.052 | 119.673 | 225.484 |

All distances in this table are Fourier bins. Both hybrids retain substantial source-set exploration. Matching removes label permutations but does not establish communication in every likelihood-constrained direction.

## Sticky-source test

Counts track the fixed labelled block identified in the saved Stage-4N audit, without changing proposal selection. Path length sums both constituent displacements; net L2 is the start-to-final displacement. Frequency range includes both intermediate and final states.

| Hybrid | Walker | Label | Slice changes | Refresh attempts | Accepted refreshes | First accepted refresh sweep (1-based) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid S | 0 | 7 | 168 | 58 | 0 | none |
| Hybrid S | 1 | 3 | 183 | 57 | 0 | none |
| Hybrid S | 2 | 3 | 158 | 56 | 0 | none |
| Hybrid S | 3 | 0 | 161 | 50 | 0 | none |
| Hybrid S | 4 | 8 | 181 | 54 | 0 | none |
| Hybrid S | 5 | 2 | 167 | 43 | 0 | none |
| Hybrid S | 6 | 7 | 161 | 55 | 0 | none |
| Hybrid S | 7 | 4 | 176 | 60 | 45 | 97 |
| Hybrid P | 0 | 7 | 179 | 133 | 0 | none |
| Hybrid P | 1 | 3 | 167 | 109 | 0 | none |
| Hybrid P | 2 | 3 | 167 | 110 | 0 | none |
| Hybrid P | 3 | 0 | 158 | 113 | 0 | none |
| Hybrid P | 4 | 8 | 186 | 103 | 0 | none |
| Hybrid P | 5 | 2 | 161 | 116 | 0 | none |
| Hybrid P | 6 | 7 | 149 | 125 | 0 | none |
| Hybrid P | 7 | 4 | 192 | 113 | 10 | 469 |

| Hybrid | Walker | Latent net L2 | Latent total path | Slice latent path | Refresh latent path | f0 net displacement (bins) | f0 range (bins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid S | 0 | 1.7132 | 6.7265 | 6.7265 | 0.0000 | 8.618 | 14.908 |
| Hybrid S | 1 | 0.4602 | 5.3180 | 5.3180 | 0.0000 | -4.896 | 23.982 |
| Hybrid S | 2 | 0.5397 | 3.3928 | 3.3928 | 0.0000 | 5.826 | 12.425 |
| Hybrid S | 3 | 0.1262 | 0.9581 | 0.9581 | 0.0000 | 1.451 | 2.647 |
| Hybrid S | 4 | 1.5090 | 4.2655 | 4.2655 | 0.0000 | 7.137 | 10.403 |
| Hybrid S | 5 | 0.5282 | 3.2597 | 3.2597 | 0.0000 | 0.226 | 8.853 |
| Hybrid S | 6 | 0.7643 | 3.3699 | 3.3699 | 0.0000 | 3.217 | 7.159 |
| Hybrid S | 7 | 3.4263 | 364.8552 | 57.3790 | 307.4762 | -276.577 | 626.983 |
| Hybrid P | 0 | 0.3828 | 3.9031 | 3.9031 | 0.0000 | 6.493 | 17.518 |
| Hybrid P | 1 | 0.9371 | 4.3974 | 4.3974 | 0.0000 | -1.763 | 10.982 |
| Hybrid P | 2 | 0.5094 | 3.4820 | 3.4820 | 0.0000 | -4.153 | 6.760 |
| Hybrid P | 3 | 0.1884 | 1.2761 | 1.2761 | 0.0000 | 1.557 | 2.616 |
| Hybrid P | 4 | 0.7472 | 4.2303 | 4.2303 | 0.0000 | 5.166 | 20.674 |
| Hybrid P | 5 | 1.4437 | 5.6840 | 5.6840 | 0.0000 | 11.153 | 13.964 |
| Hybrid P | 6 | 0.2813 | 2.1040 | 2.1040 | 0.0000 | 6.495 | 8.237 |
| Hybrid P | 7 | 4.5036 | 79.6408 | 6.8219 | 72.8188 | -297.768 | 347.295 |

**Does slice move the blocks that standalone refresh could not replace? Yes, for all eight walkers under both hybrids. Do later refreshes become accepted? Only for walker 7, label 4: 45 acceptances under Hybrid S and 10 under Hybrid P.** Every one follows a slice-induced change on an earlier sweep; first successful refreshes occur at sweeps 97 and 469 respectively. Seven other blocks still have zero refresh acceptances. This is temporal evidence of complementarity in one labelled slot, not proof that a particular slice move caused the acceptance or that all physical bottlenecks are resolved.

## Refresh acceptance and numerical correction

| Hybrid | Accepted / 4096 | Acceptance | Exact mean alpha | Contour rejection fraction | MH rejections / contour survivors |
|---|---:|---:|---:|---:|---:|
| Hybrid S | 3507 | 0.856201172 | 0.856201171874988 | 0.143798828 | 0 / 3507 |
| Hybrid P | 2893 | 0.706298828 | 0.706298828124987 | 0.293701172 | 0 / 2893 |

These rates were measured after the slice transition, not copied from Stage-4N. No binomial intervals are assigned to sequential-chain acceptance fractions. The exact prior/proposal correction remains present even though no MH uniform rejected a contour survivor.

| Walker | Hybrid S refresh acceptance | Hybrid P refresh acceptance |
|---|---:|---:|
| 0 | 0.869141 | 0.654297 |
| 1 | 0.841797 | 0.746094 |
| 2 | 0.882812 | 0.755859 |
| 3 | 0.847656 | 0.695312 |
| 4 | 0.863281 | 0.658203 |
| 5 | 0.910156 | 0.728516 |
| 6 | 0.835938 | 0.726562 |
| 7 | 0.798828 | 0.685547 |

All Hybrid S refresh categories, including zero categories if present:

| Label(s) | Attempts | Accepted | Acceptance |
|---|---:|---:|---:|
| [0] | 480 | 411 | 0.856250 |
| [1] | 485 | 469 | 0.967010 |
| [2] | 434 | 369 | 0.850230 |
| [3] | 453 | 323 | 0.713024 |
| [4] | 455 | 428 | 0.940659 |
| [5] | 409 | 390 | 0.953545 |
| [6] | 469 | 456 | 0.972281 |
| [7] | 444 | 320 | 0.720721 |
| [8] | 467 | 341 | 0.730193 |

All Hybrid P refresh categories, including zero categories if present:

| Label(s) | Attempts | Accepted | Acceptance |
|---|---:|---:|---:|
| [0, 1] | 120 | 102 | 0.850000 |
| [0, 2] | 111 | 75 | 0.675676 |
| [0, 3] | 109 | 59 | 0.541284 |
| [0, 4] | 107 | 77 | 0.719626 |
| [0, 5] | 117 | 95 | 0.811966 |
| [0, 6] | 99 | 76 | 0.767677 |
| [0, 7] | 107 | 54 | 0.504673 |
| [0, 8] | 110 | 81 | 0.736364 |
| [1, 2] | 105 | 85 | 0.809524 |
| [1, 3] | 114 | 74 | 0.649123 |
| [1, 4] | 132 | 106 | 0.803030 |
| [1, 5] | 101 | 95 | 0.940594 |
| [1, 6] | 95 | 85 | 0.894737 |
| [1, 7] | 108 | 77 | 0.712963 |
| [1, 8] | 122 | 102 | 0.836066 |
| [2, 3] | 124 | 73 | 0.588710 |
| [2, 4] | 107 | 66 | 0.616822 |
| [2, 5] | 120 | 92 | 0.766667 |
| [2, 6] | 124 | 91 | 0.733871 |
| [2, 7] | 116 | 64 | 0.551724 |
| [2, 8] | 130 | 89 | 0.684615 |
| [3, 4] | 109 | 62 | 0.568807 |
| [3, 5] | 122 | 78 | 0.639344 |
| [3, 6] | 106 | 71 | 0.669811 |
| [3, 7] | 113 | 52 | 0.460177 |
| [3, 8] | 105 | 58 | 0.552381 |
| [4, 5] | 107 | 84 | 0.785047 |
| [4, 6] | 111 | 88 | 0.792793 |
| [4, 7] | 122 | 65 | 0.532787 |
| [4, 8] | 119 | 93 | 0.781513 |
| [5, 6] | 123 | 106 | 0.861789 |
| [5, 7] | 111 | 72 | 0.648649 |
| [5, 8] | 117 | 97 | 0.829060 |
| [6, 7] | 128 | 84 | 0.656250 |
| [6, 8] | 110 | 94 | 0.854545 |
| [7, 8] | 115 | 71 | 0.617391 |

For both hybrids there were zero proposed negative-infinite implemented priors, zero nonfinite Logistic draws/coordinates, zero nonfinite proposed likelihoods, and zero absolute log corrections greater than 1e-8. This does not invalidate the tail counterexamples from Stage-4L. The recorded pre-truncation log MH ratio has distribution:

| Hybrid | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| Hybrid S | -7.3541173e-13 | -5.5067062e-14 | 0 | 5.3290705e-14 | 1.3127277e-12 |
| Hybrid P | -1.7017499e-12 | -6.3948846e-14 | 0 | 6.3948846e-14 | 1.2931878e-12 |

Accepted refresh geometry remains nonlocal:

| Hybrid / metric | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| Hybrid S / latent_l2 | 1.39437 | 3.07231 | 5.85989 | 9.54674 | 14.6443 |
| Hybrid S / physical_f0_rms_bins | 0.0539324 | 15.9147 | 188.111 | 496.354 | 618.387 |
| Hybrid S / source_set_frequency_rms_bins | 0.0179775 | 5.06387 | 41.9164 | 81.9616 | 121.436 |
| Hybrid S / source_set_full_rms | 0.0334675 | 0.0692886 | 0.116939 | 0.153822 | 0.186888 |
| Hybrid P / latent_l2 | 2.93554 | 5.65722 | 8.52495 | 12.1503 | 18.7025 |
| Hybrid P / physical_f0_rms_bins | 2.00352 | 62.4298 | 226.221 | 414.935 | 572.58 |
| Hybrid P / source_set_frequency_rms_bins | 0.944467 | 18.2972 | 55.8214 | 103.836 | 170.78 |
| Hybrid P / source_set_full_rms | 0.0681306 | 0.112352 | 0.153604 | 0.192139 | 0.228448 |

Median frequency-set refresh movement is 41.92 bins (Hybrid S) and 55.82 bins (Hybrid P), compared with 5.62/5.26 bins for Stage-4H single/pair slice components. The pooled Stage-4H step median is 2.73 bins; the hybrids’ slice components remain about 2.97/2.91 bins and their complete-sweep medians are 39.23/43.97 bins. Full distributions for slice, accepted refresh and complete sweeps are saved separately.

## Cost and mechanically selected prefixes

Costs retain the historical slice `num_steps + num_shrink` proxy. The current slice source evaluates both scalar functions once per stepping/shrink loop; prior counts derived from these counters are labelled proxies rather than independently instrumented calls. Refresh and direct checks are counted actual scalar calls. Every current state is checked after both constituents, including refresh rejection.

| Chain | Slice likelihood/prior proxy | Refresh likelihood calls | Refresh prior calls | Direct likelihood checks incl. starts | Direct prior checks incl. starts | Total likelihood proxy+actual | Total prior proxy+actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isotropic | 27342 | 0 | 0 | 4104 | 4104 | 31446 | 31446 (same loop-counter convention) |
| Refresh S | 0 | 4096 | 4096 | 4104 | 4104 | 8200 actual | 8200 actual |
| Refresh P | 0 | 4096 | 4096 | 4104 | 4104 | 8200 actual | 8200 actual |
| Hybrid S | 27183 | 4096 | 4096 | 8200 | 8200 | 39479 | 39479 |
| Hybrid P | 27222 | 4096 | 4096 | 8200 | 8200 | 39518 | 39518 |

Measured CPU loop times were 71.69 seconds for Hybrid S and 69.82 seconds for Hybrid P, including per-step checks but excluding start checks, schedules and trace writes. Slice compilation took 1.95 seconds separately. Stage-4H used GPU and has no comparable timing, so no wall-clock speedup is asserted.

The primary cost-matching rule, saved before sampling, uses the largest **common sweep prefix across all eight walkers** whose pooled cumulative `slice proxy + refresh likelihood calls + both direct cache checks`, plus eight start checks, fits the Stage-4H total of 31446. Prefix length depends only on cost. To make audit overhead explicit, a secondary proposal-only comparison uses the budget 27342 and excludes all direct checks.

| Hybrid | Cost convention | Prefix sweeps / walker | Cost | Cost at next sweep | Budget | LogL-mean SD | Exceedance SD | Median KS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid S | including_audits | 407 | 31408 | 31487 | 31446 | 917.768 | 0.325780 | 0.459459 |
| Hybrid S | proposal_only | 447 | 27301 | 27359 | 27342 | 894.917 | 0.322979 | 0.458613 |
| Hybrid P | including_audits | 408 | 31436 | 31510 | 31446 | 434.020 | 0.235478 | 0.294118 |
| Hybrid P | proposal_only | 446 | 27294 | 27367 | 27342 | 464.233 | 0.230370 | 0.329596 |

The reference remains all 512 Stage-4H steps: SD(logL means) 446.210, exceedance SD 0.252070, median KS 0.337891. Hybrid S remains worse under both cost conventions. Hybrid P improves all three at the 408-sweep primary prefix, although the logL-SD gain is small; at the 446-sweep proposal-only prefix it still improves exceedance and KS but has slightly larger logL-mean spread. No prefix was chosen from diagnostics and no chain was extended. Full prefix ESS, per-walker summaries and source-set pair distances are saved; no combined efficiency score is constructed.

## Integrity, tests, classification and scope

All starting positions and scalars matched exactly. Both constituent outputs of all 8192 hybrid sweeps had finite current priors and likelihoods, strict ell9 membership and direct cache agreement. Maximum absolute cache differences were 4.27e-14 in prior and 3.48e-8 in likelihood, within the inherited 1e-9 and 1e-7 tolerances. Every slice transition succeeded. **Zero sampling structural failures occurred.**

A separate saved-trace audit checked refresh untouched blocks, selected draws, q terms, log ratios, alphas and MH uniforms; rejection retains the exact post-slice position/caches and acceptance advances to the proposed caches. It also checked slice block masks, cumulative cost accounting and protected hashes. The prior, configuration, accepted checkpoint, old traces, original proposals.py, generic DNS code, and Stage-4N refresh implementation remained unchanged.

**All 141 existing tests and seven new tests passed (148 total).** New tests cover target invariance with a nonreversible composition, exact ordering, post-slice state transfer, acceptance/rejection cache contracts, stopping before refresh on a failed intermediate check, frozen scale/default settings, cost accounting and deterministic maximal prefixes. The initial toy-example choice happened to yield a reversible product; correcting the toy proposal exposed nonreversibility while preserving target invariance. No sampling code changed and no LISA trajectory is run in pytest.

**Hybrid S: D — worse.** The full-run and both cost-matched comparisons worsen all primary communication measures. Higher coordinate ESS and large physical movement are insufficient.

**Hybrid P: B — modest improvement, with mixed evidence.** Lower exceedance disagreement, lower pairwise KS, improved robust between-walker summaries and a favorable primary cost-matched comparison support a modest benefit. Full-run logL-mean SD and logL ESS do not improve, the first half is not convincing, and seven sticky slots still resist refresh. These eight short trajectories do not establish global equilibrium or a broadly resolved bottleneck.

Artifacts: `/tmp/lisa_dns_stage4o_hybrid/bottleneck_audit.json`, `report.json`, `comparison.json`, exact `initial_states.npz`, and `hybrid_single/trace.npz` / `hybrid_pair/trace.npz`. Traces save the post-slice and final states/caches, slice component/labels/counters, refresh proposal state/caches, Logistic draws, MH uniforms, q terms, ratios, alphas, decisions and per-sweep costs.

No probability tuning, prior or slice modification, DNS production integration, production run, evidence work, or level construction was performed. **Level 10 remains rejected; level 11 was not constructed.**
