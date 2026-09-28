# Stage 4Y: fixed-contour population validation at level 11

**Classification B — usable but residual heterogeneity.** Yes: the fixed level-11 population supports exactly ONE controlled Stage-4Z level-12 construction attempt in a later stage. No level 12 is constructed here.

Communication remains qualitatively viable on the tighter contour. Historical low lineages 2 and 7 have first/second-half diagnostic exceedances 0.32421875/0.375 and 0.41015625/0.3671875; both receive exact realizations from all seven donors (259 and 276 accepted transfers). Full-run mean-logL SD is 45.4636, versus Stage-4W 173.6039; median/max KS are 0.0703125/0.138671875, versus 0.1005859375/0.376953125. These full-run improvements are supported by reasonable within-half overlap, not persistent near-zero upper tails. The classification is B rather than A: first-to-second-half mean-logL SD rises 94.4583 to 110.4969, maximum KS rises 0.2109375 to 0.23046875, and exceedance SD rises 0.0503718 to 0.0578543, even though median KS, MAD and IQR improve. Full-run averaging partly cancels lineage differences. Second-half sorted-frequency mean distance is 68.1890 bins, versus Stage-4W 46.7360, while six-coordinate RMS remains similar (0.255983 versus 0.256824). Physical catalogue equilibrium is not established and labelled-frequency ESS remains limited. The collective likelihood, upper-tail, provenance and source-set movement evidence nevertheless supports a single independently gated construction next.

The frozen endpoint remains exactly `ell_11 = -108868.60077896297`, `log X_11 = -9.643899891984567`. Levels 0–10 match Stage 4V, levels 0–9 match Stage 4G, and the copied Stage-4X level-11 checkpoint is unchanged byte for byte. No threshold or log mass was modified.

[Diagnostic figure (PNG)](/tmp/lisa_dns_stage4y_level11_validation/mixing_diagnostics.png) · [PDF](/tmp/lisa_dns_stage4y_level11_validation/mixing_diagnostics.pdf)

The figure uses 32-sweep means only for display; the recovery criterion uses unsmoothed contemporaneous likelihoods. Both focal walkers already satisfy that criterion at the recovered sweep-0 ensemble, so their first entry at sweep 0 is not a claim of transition-induced recovery.

## Frozen design and starting ensemble

Exactly eight walkers ran 512 population sweeps with no burn-in removal, extension or second bank. Every sweep used one unchanged isotropic constrained-slice transition per walker, then four disjoint reciprocal source exchanges on the fixed seven-round schedule. Both the slice target and both exchange contour checks used ell11. Full implemented joint prior and recomputed reverse/forward selection probabilities entered one joint MH decision. Both walkers advanced or neither did.

Selector constants remain f_star = 0.0018409808 Hz and sigma_f = 2.558e-7 Hz, with stable logsumexp and no clipping. Slice scales remain pi/sqrt(3), mixture 20/40/40, max_steps 10 and max_shrinkage 100. Exchange frequency and source-selection formula are unchanged. No source_gain enters the kernel. The exact Stage-4W run-body/MH source equivalence is tested apart from contour, checkpoint source and metadata names.

Starting rule was fixed before sampling: use each final Stage-4X calibration state if it strictly exceeds ell11; otherwise use that walker’s **last** retained strict survivor, never its maximum-logL state. Positions and both caches are exact copies from the saved trace, and every start also passed direct model/cache checks before the first sweep.

| walker | fallback | absolute trace index (0-based) | retained index (0-based) | initial logL |
| --- | --- | --- | --- | --- |
| 0 | False | 383 | 255 | -108615.50843659024 |
| 1 | False | 383 | 255 | -105888.61895776488 |
| 2 | True | 222 | 94 | -108218.73287424145 |
| 3 | True | 379 | 251 | -108842.17649887555 |
| 4 | True | 382 | 254 | -108368.31002360515 |
| 5 | True | 372 | 244 | -108106.80476661834 |
| 6 | True | 381 | 253 | -108228.74174357545 |
| 7 | False | 383 | 255 | -108375.78618604878 |

Frozen random streams: slice seeds `[920100, 920101, 920102, 920103, 920104, 920105, 920106, 920107]`; PCG64 Gumbel label seed `2026092811`; independent joint-MH uniform seed `2026092812`. Designs, starts and draws were saved before sampling. These are new fixed diagnostic streams, without seed retries.

## Cross-walker logL agreement and comparison with level 10

| run | window | mean SD | mean MAD | mean IQR | median pairwise KS | maximum pairwise KS |
| --- | --- | --- | --- | --- | --- | --- |
| Stage-4W / level 10 | first256 | 289.732878 | 41.5344745 | 117.22607 | 0.115234375 | 0.578125 |
| Stage-4W / level 10 | second256 | 128.718885 | 65.6756962 | 129.518603 | 0.115234375 | 0.23828125 |
| Stage-4W / level 10 | all512 | 173.603906 | 60.6317565 | 130.885905 | 0.100585938 | 0.376953125 |
| Stage-4Y / level 11 | first256 | 94.4583351 | 81.9101715 | 164.430353 | 0.1328125 | 0.2109375 |
| Stage-4Y / level 11 | second256 | 110.496873 | 45.2556631 | 66.5836 | 0.095703125 | 0.23046875 |
| Stage-4Y / level 11 | all512 | 45.4636098 | 21.829034 | 44.6263202 | 0.0703125 | 0.138671875 |

MAD is the unscaled median absolute deviation of the eight walker means; SD uses ddof=1. KS values are descriptive distances without IID p-value interpretation. The targets differ, so level-10 comparisons assess qualitative persistence of communication rather than identical distributions.

| walker | mean logL / first256 | mean logL / second256 | mean logL / all512 |
| --- | --- | --- | --- |
| 0 | -107872.929 | -107903.242 | -107888.085 |
| 1 | -107909.435 | -108163.371 | -108036.403 |
| 2 | -108036.749 | -107888.307 | -107962.528 |
| 3 | -108070.401 | -107799.468 | -107934.935 |
| 4 | -108094.624 | -107839.005 | -107966.814 |
| 5 | -107864.6 | -107981.713 | -107923.156 |
| 6 | -107998.097 | -107903.906 | -107951.001 |
| 7 | -107883.333 | -107929.516 | -107906.424 |

| walker paired with 2 | first256 | second256 | all512 |
| --- | --- | --- | --- |
| 0 | 0.140625 | 0.078125 | 0.091796875 |
| 1 | 0.12109375 | 0.19140625 | 0.0625 |
| 3 | 0.078125 | 0.0859375 | 0.068359375 |
| 4 | 0.07421875 | 0.06640625 | 0.033203125 |
| 5 | 0.140625 | 0.07421875 | 0.0703125 |
| 6 | 0.1328125 | 0.0703125 | 0.064453125 |
| 7 | 0.15234375 | 0.0625 | 0.080078125 |

| walker paired with 7 | first256 | second256 | all512 |
| --- | --- | --- | --- |
| 0 | 0.04296875 | 0.05078125 | 0.033203125 |
| 1 | 0.05859375 | 0.203125 | 0.12890625 |
| 2 | 0.15234375 | 0.0625 | 0.080078125 |
| 3 | 0.14453125 | 0.109375 | 0.05859375 |
| 4 | 0.16796875 | 0.09375 | 0.095703125 |
| 5 | 0.078125 | 0.06640625 | 0.0625 |
| 6 | 0.2109375 | 0.06640625 | 0.099609375 |

## Diagnostic upper-tail threshold, never a DNS level

`ell_12_diag = -107828.19633247243`. This was computed only after completion from the pooled 4096 saved logL values: ascending order index `2589 = floor(N*(1-exp(-1)))`, with strict exceedance. This is the historical exp(-1) **upper-tail** convention (CDF quantile 1-exp(-1)), with no interpolation or jitter. The same diagnostic value is applied to all three windows. It is not independently calibrated, never enters a transition, has no log mass, and cannot be frozen by either Stage-4Y entry point.

| walker | fraction above ell_12_diag / first256 | fraction above ell_12_diag / second256 | fraction above ell_12_diag / all512 |
| --- | --- | --- | --- |
| 0 | 0.42578125 | 0.36328125 | 0.39453125 |
| 1 | 0.39453125 | 0.24609375 | 0.3203125 |
| 2 | 0.32421875 | 0.375 | 0.349609375 |
| 3 | 0.30859375 | 0.44140625 | 0.375 |
| 4 | 0.30859375 | 0.4140625 | 0.361328125 |
| 5 | 0.421875 | 0.3828125 | 0.40234375 |
| 6 | 0.359375 | 0.33984375 | 0.349609375 |
| 7 | 0.41015625 | 0.3671875 | 0.388671875 |

| window | pooled fraction | fraction SD | fraction min–max | conservative ESS | between-walker SE | block SE | IID SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first256 | 0.369140625 | 0.050371804 | 0.30859375 – 0.42578125 | 589.961804 | 0.0178091221 | 0.0198678217 | 0.0106634375 |
| second256 | 0.366210938 | 0.0578543375 | 0.24609375 – 0.44140625 | 396.488971 | 0.0204545972 | 0.0241948242 | 0.0106456712 |
| all512 | 0.367675781 | 0.0276558638 | 0.3203125 – 0.40234375 | 956.287847 | 0.00977782441 | 0.0155922253 | 0.00753394501 |

Conservative ESS uses the unchanged Bernoulli-equivalent estimator based on the maximum of IID, block-means and between-walker SE, block size 32. It is a mixing summary for a data-selected diagnostic threshold, not a construction gate or independent calibration. Interacting walkers are dependent; this ESS is not proof of convergence.

Level-10 versus level-11 diagnostic upper-tail and logL-autocorrelation comparison (each uses its own pooled post-hoc threshold):

| run | window | exceedance SD | exceedance range | conservative exceedance ESS | median logL ESS |
| --- | --- | --- | --- | --- | --- |
| Stage-4W / level 10 | first256 | 0.153856529 | 0.03125 – 0.515625 | 79.7698329 | 80.2591989 |
| Stage-4W / level 10 | second256 | 0.0762326623 | 0.2890625 – 0.50390625 | 314.611403 | 80.4308873 |
| Stage-4W / level 10 | all512 | 0.0954650944 | 0.162109375 – 0.466796875 | 204.082429 | 142.817338 |
| Stage-4Y / level 11 | first256 | 0.050371804 | 0.30859375 – 0.42578125 | 589.961804 | 68.2798912 |
| Stage-4Y / level 11 | second256 | 0.0578543375 | 0.24609375 – 0.44140625 | 396.488971 | 62.8319336 |
| Stage-4Y / level 11 | all512 | 0.0276558638 | 0.3203125 – 0.40234375 | 956.287847 | 128.25745 |

## Recovery of historical low-exceedance lineages 2 and 7

The criterion was saved before sampling and applied identically, separately, to both walkers: **For each of walkers 2 and 7 separately: inclusive contemporaneous 25th-75th percentiles of the OTHER seven walkers, numpy linear quantiles; sweep 0 is the recovered ensemble; sweeps 1..512 are post-exchange states; no smoothing or criterion retuning.**

**Walker 2:** initial logL -108218.73287424145; initially in the other-seven IQR: True. First entry including sweep 0: 0; first post-transition entry: 1. Fraction of all 512 sweeps in range: 0.380859375.

| window | fraction in other-seven IQR | ell_12_diag exceedance | logL ESS |
| --- | --- | --- | --- |
| first256 | 0.3828125 | 0.32421875 | 114.349766 |
| second256 | 0.37890625 | 0.375 | 38.0873457 |
| all512 | 0.380859375 | 0.349609375 | 107.802799 |

Accepted reciprocal exchanges / exact transfers received: 259 / 259; 248 distinct exact realizations received; 7 distinct other donor walkers: `[0, 1, 3, 4, 5, 6, 7]`. Start-to-end source-set displacement: 97.4187286 Fourier bins and 0.242450824 normalized six-coordinate RMS.

**Walker 7:** initial logL -108375.78618604878; initially in the other-seven IQR: True. First entry including sweep 0: 0; first post-transition entry: 2. Fraction of all 512 sweeps in range: 0.451171875.

| window | fraction in other-seven IQR | ell_12_diag exceedance | logL ESS |
| --- | --- | --- | --- |
| first256 | 0.48046875 | 0.41015625 | 83.7583201 |
| second256 | 0.421875 | 0.3671875 | 120.391303 |
| all512 | 0.451171875 | 0.388671875 | 225.403943 |

Accepted reciprocal exchanges / exact transfers received: 276 / 276; 267 distinct exact realizations received; 7 distinct other donor walkers: `[0, 1, 2, 3, 4, 5, 6]`. Start-to-end source-set displacement: 138.202891 Fourier bins and 0.264481565 normalized six-coordinate RMS.

First IQR entry is a descriptive hit, not a declaration of sustained recovery. Labels denote ensemble lineages, not permanent physical modes.

## Half-to-half behavior of every lineage

| walker | first-half upper-tail fraction | second-half upper-tail fraction | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- |
| 0 | 0.42578125 | 0.36328125 | -107872.929 | -107903.242 |
| 1 | 0.39453125 | 0.24609375 | -107909.435 | -108163.371 |
| 2 | 0.32421875 | 0.375 | -108036.749 | -107888.307 |
| 3 | 0.30859375 | 0.44140625 | -108070.401 | -107799.468 |
| 4 | 0.30859375 | 0.4140625 | -108094.624 | -107839.005 |
| 5 | 0.421875 | 0.3828125 | -107864.6 | -107981.713 |
| 6 | 0.359375 | 0.33984375 | -107998.097 | -107903.906 |
| 7 | 0.41015625 | 0.3671875 | -107883.333 | -107929.516 |

Severe low-exceedance behavior largely disappears for the historically low lineages 2 and 7, rather than persisting in them. Relative low-tail ranking migrates: walkers 3 and 4 are tied lowest in the first half at 0.30859375, while walker 1 becomes lowest in the second at 0.24609375 (from 0.39453125). Walkers 3 and 4 rise to 0.44140625 and 0.4140625. Walker 2 rises from 0.32421875 to 0.375; walker 7 changes from 0.41015625 to 0.3671875. Thus no lineage remains nearly excluded from the diagnostic upper tail, but there is residual time-dependent heterogeneity. First-half/second-half ranges are 0.30859375–0.42578125 and 0.24609375–0.44140625. This description is not an added gate.

## Autocorrelation and unique states

| window | median f0_u ESS | minimum f0_u ESS | median logL ESS |
| --- | --- | --- | --- |
| first256 | 5.51291823 | 3.01253317 | 68.2798912 |
| second256 | 6.12479996 | 2.83841486 | 62.8319336 |
| all512 | 7.21454813 | 2.82502053 | 128.25745 |

| walker | logL ESS / first256 | logL ESS / second256 | logL ESS / all512 | unique fraction / first256 | unique fraction / second256 | unique fraction / all512 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 48.5628032 | 90.0008794 | 163.052981 | 1 | 1 | 1 |
| 1 | 128.076887 | 87.5765214 | 194.955641 | 1 | 1 | 1 |
| 2 | 114.349766 | 38.0873457 | 107.802799 | 1 | 1 | 1 |
| 3 | 37.274465 | 118.766583 | 130.960151 | 1 | 1 | 1 |
| 4 | 46.8417913 | 27.6917096 | 52.8284644 | 1 | 1 | 1 |
| 5 | 65.9651941 | 30.7154639 | 67.9418219 | 1 | 1 | 1 |
| 6 | 70.5945882 | 34.5130117 | 125.554748 | 1 | 1 | 1 |
| 7 | 83.7583201 | 120.391303 | 225.403943 | 1 | 1 | 1 |

Autocorrelation ESS uses the established initial-positive monotone paired estimator, with constant-coordinate ESS zero and no antithetic bonus. Labelled f0_u ESS alone is not interpreted as physical convergence.

## Permutation-invariant source-set geometry

Stage-4I geometry is unchanged: Hungarian source matching; frequency RMS in Fourier bins, DF=1/31536000 Hz; six-coordinate RMS normalized by prior widths, with psi period pi and lam period 2pi. Frequency cross-window distances use all cross-time pairs; six-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. The final window is the second 256 sweeps. These finite-trajectory distances describe movement and separation, not distributional convergence by themselves.

| run | window | median frequency-set RMS (bins) | median sorted-frequency mean distance (bins) | median six-coordinate source-set RMS |
| --- | --- | --- | --- | --- |
| Stage-4W / level 10 | first256 | 105.566416 | 58.9003017 | 0.256131953 |
| Stage-4W / level 10 | second256 | 96.7113719 | 46.7359812 | 0.256823562 |
| Stage-4W / level 10 | all512 | 97.5332267 | 40.6709785 | 0.257350817 |
| Stage-4Y / level 11 | first256 | 105.004603 | 52.0726335 | 0.253674752 |
| Stage-4Y / level 11 | second256 | 114.577621 | 68.1890044 | 0.255982746 |
| Stage-4Y / level 11 | all512 | 114.750126 | 54.9993314 | 0.255917663 |

| walker | start→end frequency bins | max frequency from start | start→end six-coordinate RMS | max six-coordinate RMS | between-halves frequency-set RMS |
| --- | --- | --- | --- | --- | --- |
| 0 | 57.1949512 | 175.104994 | 0.245114546 | 0.300032774 | 92.6075201 |
| 1 | 58.9696262 | 136.802505 | 0.276998377 | 0.301266321 | 99.4476011 |
| 2 | 97.4187286 | 207.416 | 0.242450824 | 0.283056293 | 153.990576 |
| 3 | 103.479721 | 164.128678 | 0.254940357 | 0.283131713 | 113.728063 |
| 4 | 49.4858783 | 136.484729 | 0.234385427 | 0.261597241 | 79.5702001 |
| 5 | 38.318535 | 235.926666 | 0.293127218 | 0.293127218 | 128.890337 |
| 6 | 71.809176 | 138.757522 | 0.267283346 | 0.279927873 | 73.0581941 |
| 7 | 138.202891 | 228.850792 | 0.264481565 | 0.285407152 | 132.680671 |

| walker | within-walker frequency RMS / first256 | within-walker frequency RMS / second256 | within-walker six-coordinate RMS / first256 | within-walker six-coordinate RMS / second256 |
| --- | --- | --- | --- | --- |
| 0 | 80.4822396 | 81.2248066 | 0.221848655 | 0.232641886 |
| 1 | 99.9214338 | 91.4819943 | 0.228101228 | 0.219307573 |
| 2 | 72.5344651 | 115.545363 | 0.215833635 | 0.227547888 |
| 3 | 104.082339 | 97.5729366 | 0.21178583 | 0.224197435 |
| 4 | 83.2015232 | 71.9835188 | 0.221992321 | 0.213712789 |
| 5 | 103.095702 | 70.627793 | 0.219003466 | 0.211806267 |
| 6 | 55.1361645 | 73.3262294 | 0.210712523 | 0.217178156 |
| 7 | 93.332615 | 81.1385386 | 0.217906399 | 0.2233169 |

| pair | W level-10 final-window frequency RMS | Y level-11 final-window frequency RMS | W level-10 sorted-frequency mean distance | Y level-11 sorted-frequency mean distance | W level-10 six-coordinate RMS | Y level-11 six-coordinate RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 123.076916 | 103.911854 | 86.7195445 | 57.5713572 | 0.261272007 | 0.267675105 |
| 0,2 | 116.112512 | 140.809993 | 80.6926371 | 99.2640631 | 0.267577039 | 0.263643958 |
| 0,3 | 83.6422878 | 165.819428 | 29.1849031 | 139.417033 | 0.267277819 | 0.269969332 |
| 0,4 | 90.4790933 | 115.17842 | 33.4175273 | 85.8866716 | 0.269376858 | 0.254826507 |
| 0,5 | 106.244942 | 192.314431 | 63.3599963 | 176.612467 | 0.269349782 | 0.2692393 |
| 0,6 | 117.658838 | 140.866671 | 69.9365852 | 117.712856 | 0.26388791 | 0.265270645 |
| 0,7 | 105.74654 | 181.423706 | 45.2313828 | 162.247019 | 0.275279915 | 0.273093724 |
| 1,2 | 80.2356784 | 115.537809 | 15.9959865 | 49.8913037 | 0.245190012 | 0.263048948 |
| 1,3 | 101.253094 | 131.731538 | 69.9774004 | 91.6977721 | 0.257240743 | 0.266147663 |
| 1,4 | 101.166998 | 94.7225011 | 62.8804223 | 46.8728181 | 0.2501914 | 0.257805711 |
| 1,5 | 86.1174005 | 155.222487 | 30.5226052 | 131.967422 | 0.250584549 | 0.262151311 |
| 1,6 | 100.793625 | 107.707703 | 44.6743758 | 68.761208 | 0.254372461 | 0.261280544 |
| 1,7 | 106.971964 | 138.098322 | 55.6369434 | 107.679792 | 0.255489783 | 0.25997765 |
| 2,3 | 92.478163 | 116.454754 | 62.0781143 | 46.1097101 | 0.257763717 | 0.255017469 |
| 2,4 | 93.5928702 | 99.6878759 | 55.996835 | 25.913195 | 0.259069292 | 0.255079943 |
| 2,5 | 81.7595099 | 131.728299 | 29.2724167 | 90.459033 | 0.256138335 | 0.255492179 |
| 2,6 | 96.1911184 | 101.922898 | 41.7852902 | 32.0084937 | 0.259495407 | 0.248638805 |
| 2,7 | 100.867658 | 119.25289 | 49.8355991 | 65.2238847 | 0.25452999 | 0.261300086 |
| 3,4 | 70.4339091 | 105.617115 | 12.7865446 | 61.6759476 | 0.251269335 | 0.255923907 |
| 3,5 | 85.6058212 | 99.489813 | 48.2405797 | 51.4182977 | 0.250790812 | 0.248783846 |
| 3,6 | 100.698621 | 91.5214935 | 58.8220323 | 30.456146 | 0.256104753 | 0.253787013 |
| 3,7 | 87.3079024 | 99.7359527 | 27.5411036 | 43.5349292 | 0.253230432 | 0.25319862 |
| 4,5 | 86.5557068 | 128.576531 | 39.5629348 | 106.990505 | 0.25640638 | 0.248985207 |
| 4,6 | 103.499483 | 89.3938369 | 55.7359922 | 52.077602 | 0.25859996 | 0.250690665 |
| 4,7 | 90.1130527 | 113.976822 | 18.186351 | 84.3099712 | 0.263982789 | 0.256041584 |
| 5,6 | 93.9279675 | 98.7650864 | 31.8285651 | 67.6168008 | 0.248165927 | 0.255687772 |
| 5,7 | 97.2316253 | 103.777805 | 38.2178859 | 70.5971605 | 0.250337149 | 0.245884398 |
| 6,7 | 116.020431 | 96.4024218 | 61.5498219 | 57.5615127 | 0.264673248 | 0.254730362 |

Physical pair distances involving focal walkers 2 or 7:

| window | pair | frequency-set RMS (bins) | sorted-frequency mean distance (bins) | six-coordinate RMS |
| --- | --- | --- | --- | --- |
| second256 | 0,2 | 140.809993 | 99.2640631 | 0.263643958 |
| second256 | 0,7 | 181.423706 | 162.247019 | 0.273093724 |
| second256 | 1,2 | 115.537809 | 49.8913037 | 0.263048948 |
| second256 | 1,7 | 138.098322 | 107.679792 | 0.25997765 |
| second256 | 2,3 | 116.454754 | 46.1097101 | 0.255017469 |
| second256 | 2,4 | 99.6878759 | 25.913195 | 0.255079943 |
| second256 | 2,5 | 131.728299 | 90.459033 | 0.255492179 |
| second256 | 2,6 | 101.922898 | 32.0084937 | 0.248638805 |
| second256 | 2,7 | 119.25289 | 65.2238847 | 0.261300086 |
| second256 | 3,7 | 99.7359527 | 43.5349292 | 0.25319862 |
| second256 | 4,7 | 113.976822 | 84.3099712 | 0.256041584 |
| second256 | 5,7 | 103.777805 | 70.5971605 | 0.245884398 |
| second256 | 6,7 | 96.4024218 | 57.5615127 | 0.254730362 |
| all512 | 0,2 | 114.388552 | 32.427972 | 0.260560071 |
| all512 | 0,7 | 143.04081 | 101.83041 | 0.264479671 |
| all512 | 1,2 | 114.706383 | 11.6691984 | 0.260022188 |
| all512 | 1,7 | 125.212768 | 67.8143035 | 0.25322403 |
| all512 | 2,3 | 140.993923 | 76.4643386 | 0.261386553 |
| all512 | 2,4 | 116.114781 | 46.1735987 | 0.255925469 |
| all512 | 2,5 | 153.974127 | 96.6848177 | 0.266656155 |
| all512 | 2,6 | 122.000689 | 65.1377747 | 0.255070442 |
| all512 | 2,7 | 141.341712 | 73.5517692 | 0.263641906 |
| all512 | 3,7 | 113.473816 | 28.0310959 | 0.249199141 |
| all512 | 4,7 | 105.430514 | 41.3143908 | 0.253451439 |
| all512 | 5,7 | 120.919263 | 46.9395062 | 0.250160171 |
| all512 | 6,7 | 101.176279 | 39.2087727 | 0.248599253 |

## Exact source-realization provenance

| diagnostic | Stage-4W / level 10 | Stage-4Y / level 11 |
| --- | --- | --- |
| accepted directional transfers | 1558 | 1990 |
| distinct exact realizations visiting another walker | 853 | 1012 |
| foreign-birth arrivals | 1487 | 1899 |
| first recipient visits | 1417 | 1825 |
| exact initial-token transfers | 19 | 4 |
| initial ancestries visiting other walkers | 51 | 44 |

| recipient walker | Y accepted transfers received | Y distinct received realizations | Y distinct donor walkers | Y donor IDs | W distinct donor walkers | W distinct received realizations |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 249 | 233 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 | 197 |
| 1 | 249 | 244 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 | 210 |
| 2 | 259 | 248 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 | 169 |
| 3 | 237 | 227 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 | 168 |
| 4 | 250 | 240 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 | 241 |
| 5 | 246 | 236 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 | 213 |
| 6 | 224 | 220 | 7 | [0, 1, 2, 3, 4, 5, 7] | 7 | 118 |
| 7 | 276 | 267 | 7 | [0, 1, 2, 3, 4, 5, 6] | 7 | 172 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 36 | 37 | 34 | 32 | 35 | 30 | 45 |
| 1 | 36 | 0 | 39 | 27 | 33 | 39 | 29 | 46 |
| 2 | 37 | 39 | 0 | 38 | 39 | 37 | 28 | 41 |
| 3 | 34 | 27 | 38 | 0 | 38 | 27 | 35 | 38 |
| 4 | 32 | 33 | 39 | 38 | 0 | 40 | 33 | 35 |
| 5 | 35 | 39 | 37 | 27 | 40 | 0 | 33 | 35 |
| 6 | 30 | 29 | 28 | 35 | 33 | 33 | 0 | 36 |
| 7 | 45 | 46 | 41 | 38 | 35 | 35 | 36 | 0 |

Initial ancestry IDs 0–71 are assigned to the recovered starting blocks for this diagnostic. As in Stage 4U, ancestry survives slice descendants, but an exact-realization token is newly minted whenever the slice targets its block, conservatively even for a roundoff-identical update. Untouched tokens persist; accepted joint exchanges swap tokens and rejection preserves them. Independent replay verifies copied six-coordinate values, all tokens, all cache outcomes and lineage permutations. Full event records are saved in provenance_transfers.json and token-by-token migration counts in comparison.json. These tags are descriptive and never affect pairing or selection.

## Exchange diagnostics

| run | proposals | joint contour survivors | survival fraction | accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage-4W / level 10 | 2048 | 795 | 0.388183594 | 779 | 0.380371094 | 16 | 0.0201257862 |
| Stage-4Y / level 11 | 2048 | 1009 | 0.492675781 | 995 | 0.485839844 | 14 | 0.0138751239 |

| distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- |
| selection_log_ratio | -0.440105503 | -0.00300346176 | 1.34700704e-05 | 0.0475329508 | 5.57222044 | 0 |
| joint_prior_ratio | -1.42108547e-13 | -2.84217094e-14 | 0 | 2.84217094e-14 | 1.70530257e-13 | 0 |

Nonfinite proposed priors: 0; likelihoods: 0; full MH ratios: 0. Minimum finite selector log probability: -919.3236910973995. Selection and MH stay in log space; no epsilon clipping.

| pair | proposals | joint survivors | accepted | acceptance fraction | MH rejection among survivors |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 73 | 36 | 36 | 0.493150685 | 0 |
| 0,2 | 73 | 37 | 37 | 0.506849315 | 0 |
| 0,3 | 73 | 34 | 34 | 0.465753425 | 0 |
| 0,4 | 73 | 32 | 32 | 0.438356164 | 0 |
| 0,5 | 73 | 35 | 35 | 0.479452055 | 0 |
| 0,6 | 73 | 32 | 30 | 0.410958904 | 0.0625 |
| 0,7 | 74 | 46 | 45 | 0.608108108 | 0.0217391304 |
| 1,2 | 73 | 40 | 39 | 0.534246575 | 0.025 |
| 1,3 | 73 | 28 | 27 | 0.369863014 | 0.0357142857 |
| 1,4 | 73 | 33 | 33 | 0.452054795 | 0 |
| 1,5 | 73 | 40 | 39 | 0.534246575 | 0.025 |
| 1,6 | 74 | 29 | 29 | 0.391891892 | 0 |
| 1,7 | 73 | 47 | 46 | 0.630136986 | 0.0212765957 |
| 2,3 | 73 | 40 | 38 | 0.520547945 | 0.05 |
| 2,4 | 73 | 39 | 39 | 0.534246575 | 0 |
| 2,5 | 74 | 38 | 37 | 0.5 | 0.0263157895 |
| 2,6 | 73 | 29 | 28 | 0.383561644 | 0.0344827586 |
| 2,7 | 73 | 41 | 41 | 0.561643836 | 0 |
| 3,4 | 74 | 38 | 38 | 0.513513514 | 0 |
| 3,5 | 73 | 28 | 27 | 0.369863014 | 0.0357142857 |
| 3,6 | 73 | 37 | 35 | 0.479452055 | 0.0540540541 |
| 3,7 | 73 | 38 | 38 | 0.520547945 | 0 |
| 4,5 | 73 | 40 | 40 | 0.547945205 | 0 |
| 4,6 | 73 | 33 | 33 | 0.452054795 | 0 |
| 4,7 | 73 | 35 | 35 | 0.479452055 | 0 |
| 5,6 | 73 | 33 | 33 | 0.452054795 | 0 |
| 5,7 | 73 | 35 | 35 | 0.479452055 | 0 |
| 6,7 | 73 | 36 | 36 | 0.493150685 | 0 |

The fixed schedule executes 73 complete seven-round cycles plus the first round. Four pairs have 74 opportunities and the remaining 24 have 73; every walker has 512 pair opportunities.

## Cost, integrity and stopping point

| cost counter | Stage-4W / level 10 | Stage-4Y / level 11 |
| --- | --- | --- |
| slice_likelihood_proxy | 27925 | 27868 |
| exchange_likelihood | 4096 | 4096 |
| slice_prior_proxy | 27925 | 27868 |
| exchange_prior | 4096 | 4096 |
| direct_cache_likelihood | 8200 | 8200 |
| direct_cache_prior | 8200 | 8200 |
| total_likelihood_proxy | 40221 | 40164 |
| total_prior_proxy | 40221 | 40164 |

Slice counters retain the historical num_steps+num_shrink proxy; scalar prior and likelihood are evaluated together. Exchange evaluations and direct cache checks are counted separately, including the eight initial-state checks. There was no cost-based truncation. CPU loop wall time 69.5667173 s is descriptive only.

All 4096 slice outcomes and 4096 pair-member exchange outcomes passed finite-coordinate/prior/likelihood, strict ell11 and direct cache checks. Maximum absolute prior-cache error 0.0; likelihood-cache error 0.0 (atol 1e-9 and 1e-7 respectively, rtol 0). Selected six-coordinate blocks swapped exactly, untouched blocks were unchanged, rejection restored both post-slice states, and acceptance committed both jointly. Saved selector probabilities, full MH decisions, state/cache outcomes and provenance were independently replayed. Structural failures: **0**. Protected inputs and the full frozen ladder retained their hashes.

**248 tests passed before sampling:** all previous 235 tests preserved plus 13 cheap Stage-4Y cases covering exact frozen checkpoint, last-valid fallback/cache copying, no-survivor stop, strict ell11 enforcement for both walkers, unchanged transition settings/source equivalence, diagnostic-only threshold with no construction/freeze path, walker-2/7 criteria and sweep numbering, and accepted/rejected provenance replay with tamper detection. No real LISA trajectory is run in pytest.

Artifacts are in `/tmp/lisa_dns_stage4y_level11_validation/`: frozen design.json, copied checkpoint_level11.json, initial_states.npz, random_schedule.npz, report.json, trace.npz, exchanges.npz, provenance.json, provenance_transfers.json, comparison.json, decision.json and validation.json. The trace includes all 512 post-exchange states plus constituent states and caches.

No level 12 was constructed or independently calibrated. No threshold, mass, selector, pairing, exchange frequency, slice kernel or prior was tuned. No production or evidence implementation was run. Evidence remains unvalidated and out of scope. Yes: the fixed level-11 population supports exactly ONE controlled Stage-4Z level-12 construction attempt in a later stage. No level 12 is constructed here. Stop here.
