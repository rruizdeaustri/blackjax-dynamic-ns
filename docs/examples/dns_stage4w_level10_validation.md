# Stage 4W: fixed-contour population validation at level 10

**Classification B — usable but residual heterogeneity.** Yes: communication at the frozen level 10 is sufficient to justify ONE controlled level-11 construction attempt in a later stage; none is performed here.

Walker 6 has serious first-half separation but recovers in the second half: its diagnostic upper-tail fraction rises from 0.03125 to 0.29296875, within the other seven walkers' second-half range 0.2890625–0.50390625. Its pairwise KS distances fall from 0.47265625–0.578125 to 0.05078125–0.2265625. Global mean-logL SD falls from 289.7329 to 128.7189, maximum pairwise KS from 0.578125 to 0.23828125, and exceedance SD from 0.1538565 to 0.0762327. All eight walkers receive exact source realizations from all seven other walkers; walker 6 receives 123 and moves 48.9189 Fourier bins / 0.275081 six-coordinate RMS from its start. Final-window source-set geometry is comparable to the level-9 benchmark, with no distinctive walker-6 isolation in the reported final-window distances. These concordant observations support B rather than C. It is not A: the full-run walker-6 tail fraction remains 0.162109375 versus 0.3359375–0.466796875 for the others, all-run mean-logL SD is 173.6039 versus Stage-4U 54.3648, temporal shifts persist, and labelled-frequency ESS and whole-catalogue exploration remain limited.

The frozen endpoint remains exactly `ell_10 = -109780.28875081123`, `log X_10 = -8.394305725653265`. Levels 0–9 match Stage 4G byte for byte, and the copied Stage-4V level-10 checkpoint is unchanged. No threshold or log mass was modified.

[Diagnostic figure (PNG)](/tmp/lisa_dns_stage4w_level10_validation/mixing_diagnostics.png) · [PDF](/tmp/lisa_dns_stage4w_level10_validation/mixing_diagnostics.pdf)

The figure uses 32-sweep means only for display. The predeclared recovery criterion uses unsmoothed contemporaneous values. First entry at sweep 11 is an initial hit, not evidence of sustained recovery; the first-half separation is retained in all full-run diagnostics.

## Frozen design and starting ensemble

Exactly eight walkers ran 512 population sweeps with no burn-in removal, extension or second bank. Every sweep used one unchanged isotropic constrained-slice transition per walker, then four disjoint reciprocal source exchanges on the fixed seven-round schedule. Both the slice target and both exchange contour checks used ell10. Full implemented joint prior and recomputed reverse/forward selection probabilities entered one joint MH decision. Both walkers advanced or neither did.

Selector constants remain f_star = 0.0018409808 Hz and sigma_f = 2.558e-7 Hz, with stable logsumexp and no clipping. Slice scales remain pi/sqrt(3), mixture 20/40/40, max_steps 10 and max_shrinkage 100. Exchange frequency and source-selection formula are unchanged. No source_gain enters the kernel. The exact Stage-4U run-body/MH source equivalence is tested apart from contour, checkpoint source and metadata names.

Starting rule was fixed before sampling: use each final Stage-4V calibration state if it strictly exceeds ell10; otherwise use that walker’s **last** retained strict survivor, never its maximum-logL state. Positions and both caches are exact copies from the saved trace, and every start also passed direct model/cache checks before the first sweep.

| walker | fallback | absolute trace index (0-based) | retained index (0-based) | initial logL |
| --- | --- | --- | --- | --- |
| 0 | False | 383 | 255 | -107861.51300179477 |
| 1 | False | 383 | 255 | -109604.57077308747 |
| 2 | False | 383 | 255 | -109550.11122971782 |
| 3 | False | 383 | 255 | -108988.73167866454 |
| 4 | True | 382 | 254 | -109013.57187485501 |
| 5 | True | 377 | 249 | -109261.12086012302 |
| 6 | True | 376 | 248 | -109713.97700038827 |
| 7 | True | 382 | 254 | -109133.6817948025 |

Frozen random streams: slice seeds `[900100, 900101, 900102, 900103, 900104, 900105, 900106, 900107]`; PCG64 Gumbel label seed `2026092805`; independent joint-MH uniform seed `2026092806`. Designs, starts and draws were saved before sampling. These are new fixed diagnostic streams, without seed retries.

## Cross-walker logL agreement and comparison with level 9

| run | window | mean SD | mean MAD | mean IQR | median pairwise KS | maximum pairwise KS |
| --- | --- | --- | --- | --- | --- | --- |
| Stage-4U / level 9 | first256 | 45.0530315 | 37.2754858 | 56.3281173 | 0.08203125 | 0.125 |
| Stage-4U / level 9 | second256 | 92.45061 | 47.9837293 | 174.579989 | 0.087890625 | 0.13671875 |
| Stage-4U / level 9 | all512 | 54.3647596 | 50.321155 | 94.1777898 | 0.0625 | 0.1015625 |
| Stage-4W / level 10 | first256 | 289.732878 | 41.5344745 | 117.22607 | 0.115234375 | 0.578125 |
| Stage-4W / level 10 | second256 | 128.718885 | 65.6756962 | 129.518603 | 0.115234375 | 0.23828125 |
| Stage-4W / level 10 | all512 | 173.603906 | 60.6317565 | 130.885905 | 0.100585938 | 0.376953125 |

MAD is the unscaled median absolute deviation of the eight walker means; SD uses ddof=1. KS values are descriptive distances without IID p-value interpretation. The targets differ, so level-9 comparisons assess qualitative persistence of communication rather than identical distributions.

| walker | mean logL / first256 | mean logL / second256 | mean logL / all512 |
| --- | --- | --- | --- |
| 0 | -108750.325 | -109072.297 | -108911.311 |
| 1 | -108833.394 | -109194.215 | -109013.804 |
| 2 | -109002.285 | -109062.864 | -109032.574 |
| 3 | -108777.851 | -109160.662 | -108969.256 |
| 4 | -108642.017 | -108975.412 | -108808.714 |
| 5 | -108797.761 | -108835.261 | -108816.511 |
| 6 | -109563.127 | -109162.621 | -109362.874 |
| 7 | -108761.079 | -109216.422 | -108988.75 |

| walker paired with 6 | first256 | second256 | all512 |
| --- | --- | --- | --- |
| 0 | 0.578125 | 0.09375 | 0.3203125 |
| 1 | 0.5625 | 0.05859375 | 0.27734375 |
| 2 | 0.47265625 | 0.1015625 | 0.26953125 |
| 3 | 0.50390625 | 0.08203125 | 0.275390625 |
| 4 | 0.56640625 | 0.19140625 | 0.376953125 |
| 5 | 0.55078125 | 0.2265625 | 0.361328125 |
| 7 | 0.57421875 | 0.05078125 | 0.287109375 |

## Diagnostic upper-tail threshold, never a DNS level

`ell_11_diag = -109102.11830523466`. This was computed only after completion from the pooled 4096 saved logL values: ascending order index `2589 = floor(N*(1-exp(-1)))`, with strict exceedance. This is the historical exp(-1) **upper-tail** convention (CDF quantile 1-exp(-1)), with no interpolation or jitter. The same diagnostic value is applied to all three windows. It is not independently calibrated, never enters a transition, has no log mass, and cannot be frozen by either Stage-4W entry point.

| walker | fraction above ell_11_diag / first256 | fraction above ell_11_diag / second256 | fraction above ell_11_diag / all512 |
| --- | --- | --- | --- |
| 0 | 0.4375 | 0.37109375 | 0.404296875 |
| 1 | 0.4296875 | 0.2890625 | 0.359375 |
| 2 | 0.3046875 | 0.3671875 | 0.3359375 |
| 3 | 0.4296875 | 0.3046875 | 0.3671875 |
| 4 | 0.515625 | 0.41015625 | 0.462890625 |
| 5 | 0.4296875 | 0.50390625 | 0.466796875 |
| 6 | 0.03125 | 0.29296875 | 0.162109375 |
| 7 | 0.4765625 | 0.2890625 | 0.3828125 |

| window | pooled fraction | fraction SD | fraction min–max | conservative ESS | between-walker SE | block SE | IID SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first256 | 0.381835938 | 0.153856529 | 0.03125 – 0.515625 | 79.7698329 | 0.0543964974 | 0.0263786672 | 0.0107355748 |
| second256 | 0.353515625 | 0.0762326623 | 0.2890625 – 0.50390625 | 314.611403 | 0.0269523162 | 0.0212804855 | 0.0105637557 |
| all512 | 0.367675781 | 0.0954650944 | 0.162109375 – 0.466796875 | 204.082429 | 0.0337520078 | 0.016926042 | 0.00753394501 |

Conservative ESS uses the unchanged Bernoulli-equivalent estimator based on the maximum of IID, block-means and between-walker SE, block size 32. It is a mixing summary for a data-selected diagnostic threshold, not a construction gate or independent calibration. Interacting walkers are dependent; this ESS is not proof of convergence.

## Historical walker 6

The criterion was saved before sampling: **Inclusive contemporaneous 25th-75th percentiles of the other seven walkers, numpy linear quantiles; sweep 0 is the recovered ensemble; sweeps 1..512 are post-exchange states; no temporal smoothing or retuning.**

Initial logL = -109713.97700038827; initially in the other-seven IQR: False. First entry including sweep 0: 11; first post-transition entry: 11. Fraction of the 512 sweeps in range: 0.28515625. Window fractions: `{'first256': 0.19140625, 'second256': 0.37890625, 'all512': 0.28515625}`.

Walker 6 received 123 accepted source-realization transfers in 123 accepted pair exchanges, from 7 distinct other walkers: `[0, 1, 2, 3, 4, 5, 7]`. Its start-to-end frequency-set displacement was 48.9189317 Fourier bins (maximum 139.374862); six-coordinate source-set RMS displacement was 0.275080573 (maximum 0.276722189). These are permutation-invariant distances, not labelled coordinate movement.

## Autocorrelation and unique states

| window | median f0_u ESS | minimum f0_u ESS | median logL ESS |
| --- | --- | --- | --- |
| first256 | 6.98297366 | 2.7345512 | 80.2591989 |
| second256 | 6.01002887 | 3.0493297 | 80.4308873 |
| all512 | 7.48545144 | 2.92538147 | 142.817338 |

| walker | logL ESS / first256 | logL ESS / second256 | logL ESS / all512 | unique fraction / first256 | unique fraction / second256 | unique fraction / all512 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 68.1743563 | 86.0962674 | 137.27999 | 1 | 1 | 1 |
| 1 | 106.035224 | 63.7290199 | 148.354686 | 1 | 1 | 1 |
| 2 | 81.8270055 | 90.5131531 | 170.052267 | 1 | 1 | 1 |
| 3 | 106.909935 | 59.6952283 | 183.184026 | 1 | 1 | 1 |
| 4 | 78.6913922 | 136.74036 | 153.660419 | 1 | 1 | 1 |
| 5 | 31.0854205 | 114.010372 | 93.2709256 | 1 | 1 | 1 |
| 6 | 91.2321641 | 74.7655073 | 41.1084825 | 1 | 1 | 1 |
| 7 | 37.7053103 | 44.3007066 | 44.2739174 | 1 | 1 | 1 |

Autocorrelation ESS uses the established initial-positive monotone paired estimator, with constant-coordinate ESS zero and no antithetic bonus. Labelled f0_u ESS alone is not interpreted as physical convergence.

## Permutation-invariant source-set geometry

Stage-4I geometry is unchanged: Hungarian source matching; frequency RMS in Fourier bins, DF=1/31536000 Hz; six-coordinate RMS normalized by prior widths, with psi period pi and lam period 2pi. Frequency cross-window distances use all cross-time pairs; six-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. The final window is the second 256 sweeps. These finite-trajectory distances describe movement and separation, not distributional convergence by themselves.

| run | window | median frequency-set RMS (bins) | median sorted-frequency mean distance (bins) | median six-coordinate source-set RMS |
| --- | --- | --- | --- | --- |
| Stage-4U / level 9 | first256 | 94.1538316 | 46.3818613 | 0.256021555 |
| Stage-4U / level 9 | second256 | 101.483954 | 64.0018392 | 0.260728835 |
| Stage-4U / level 9 | all512 | 101.303309 | 42.0729606 | 0.257219051 |
| Stage-4W / level 10 | first256 | 105.566416 | 58.9003017 | 0.256131953 |
| Stage-4W / level 10 | second256 | 96.7113719 | 46.7359812 | 0.256823562 |
| Stage-4W / level 10 | all512 | 97.5332267 | 40.6709785 | 0.257350817 |

| walker | start→end frequency bins | max frequency from start | start→end six-coordinate RMS | max six-coordinate RMS | between-halves frequency-set RMS |
| --- | --- | --- | --- | --- | --- |
| 0 | 63.8992259 | 138.296469 | 0.230832238 | 0.287731235 | 91.7468722 |
| 1 | 46.1302952 | 163.608148 | 0.228443486 | 0.267353909 | 112.058243 |
| 2 | 69.9450817 | 120.970545 | 0.236941261 | 0.278569932 | 77.1949844 |
| 3 | 68.9813092 | 136.232209 | 0.239333717 | 0.290496916 | 81.3958463 |
| 4 | 170.245082 | 197.495274 | 0.280333149 | 0.314948057 | 103.571911 |
| 5 | 79.4717811 | 127.30279 | 0.235152113 | 0.269863476 | 96.393595 |
| 6 | 48.9189317 | 139.374862 | 0.275080573 | 0.276722189 | 83.0127466 |
| 7 | 173.42563 | 180.748293 | 0.307679656 | 0.307679656 | 97.5961791 |

| walker | within-walker frequency RMS / first256 | within-walker frequency RMS / second256 | within-walker six-coordinate RMS / first256 | within-walker six-coordinate RMS / second256 |
| --- | --- | --- | --- | --- |
| 0 | 71.3818392 | 91.7418612 | 0.23822975 | 0.217276644 |
| 1 | 105.964669 | 82.696594 | 0.225326332 | 0.206872166 |
| 2 | 71.5219703 | 74.3307362 | 0.219119454 | 0.222824365 |
| 3 | 86.1686449 | 62.2252229 | 0.222333452 | 0.212198559 |
| 4 | 76.3539224 | 75.6498599 | 0.217285603 | 0.22403423 |
| 5 | 96.7835375 | 78.2970486 | 0.218972211 | 0.220899715 |
| 6 | 60.6896696 | 97.4083324 | 0.205727461 | 0.220616861 |
| 7 | 65.1952449 | 99.2790839 | 0.206987513 | 0.216084746 |

| pair | U final-window frequency RMS | W final-window frequency RMS | U sorted-frequency mean distance | W sorted-frequency mean distance | U six-coordinate RMS | W six-coordinate RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0,1 | 101.434346 | 123.076916 | 68.2358179 | 86.7195445 | 0.265230417 | 0.261272007 |
| 0,2 | 124.322342 | 116.112512 | 92.4472311 | 80.6926371 | 0.27704938 | 0.267577039 |
| 0,3 | 82.0163346 | 83.6422878 | 32.0896432 | 29.1849031 | 0.260631196 | 0.267277819 |
| 0,4 | 115.239574 | 90.4790933 | 97.81483 | 33.4175273 | 0.260826475 | 0.269376858 |
| 0,5 | 68.1710488 | 106.244942 | 10.8723758 | 63.3599963 | 0.24884783 | 0.269349782 |
| 0,6 | 122.951921 | 117.658838 | 102.608308 | 69.9365852 | 0.266877899 | 0.26388791 |
| 0,7 | 87.2335107 | 105.74654 | 31.6374764 | 45.2313828 | 0.261797008 | 0.275279915 |
| 1,2 | 107.730196 | 80.2356784 | 56.1836878 | 15.9959865 | 0.264684677 | 0.245190012 |
| 1,3 | 95.7734249 | 101.253094 | 43.9997815 | 69.9774004 | 0.262455627 | 0.257240743 |
| 1,4 | 96.347984 | 101.166998 | 63.4892216 | 62.8804223 | 0.255522127 | 0.2501914 |
| 1,5 | 100.749014 | 86.1174005 | 63.8877627 | 30.5226052 | 0.25561545 | 0.250584549 |
| 1,6 | 96.6557978 | 100.793625 | 56.6921428 | 44.6743758 | 0.261566741 | 0.254372461 |
| 1,7 | 112.615847 | 106.971964 | 67.3345325 | 55.6369434 | 0.262905079 | 0.255489783 |
| 2,3 | 112.357347 | 92.478163 | 64.1159158 | 62.0781143 | 0.275103105 | 0.257763717 |
| 2,4 | 81.6957915 | 93.5928702 | 12.0579837 | 55.996835 | 0.254570601 | 0.259069292 |
| 2,5 | 122.558773 | 81.7595099 | 87.6081091 | 29.2724167 | 0.256139184 | 0.256138335 |
| 2,6 | 89.787614 | 96.1911184 | 25.6264866 | 41.7852902 | 0.254651026 | 0.259495407 |
| 2,7 | 143.841958 | 100.867658 | 106.138422 | 49.8355991 | 0.2834586 | 0.25452999 |
| 3,4 | 100.600263 | 70.4339091 | 69.3125234 | 12.7865446 | 0.257802798 | 0.251269335 |
| 3,5 | 83.5838827 | 85.6058212 | 29.2132434 | 48.2405797 | 0.251472714 | 0.250790812 |
| 3,6 | 106.807946 | 100.698621 | 72.2181332 | 58.8220323 | 0.272354218 | 0.256104753 |
| 3,7 | 101.533562 | 87.3079024 | 45.7879927 | 27.5411036 | 0.256197289 | 0.253230432 |
| 4,5 | 113.147151 | 86.5557068 | 93.0265625 | 39.5629348 | 0.242157657 | 0.25640638 |
| 4,6 | 69.1417231 | 103.499483 | 23.9293701 | 55.7359922 | 0.254124189 | 0.25859996 |
| 4,7 | 137.042024 | 90.1130527 | 112.037975 | 18.186351 | 0.265920982 | 0.263982789 |
| 5,6 | 120.520154 | 93.9279675 | 97.4704044 | 31.8285651 | 0.251328012 | 0.248165927 |
| 5,7 | 89.4298059 | 97.2316253 | 30.8709702 | 38.2178859 | 0.25556159 | 0.250337149 |
| 6,7 | 139.263692 | 116.020431 | 110.860615 | 61.5498219 | 0.273322235 | 0.264673248 |

## Exact source-realization provenance

| diagnostic | Stage-4U / level 9 | Stage-4W / level 10 |
| --- | --- | --- |
| accepted directional transfers | 2004 | 1558 |
| distinct exact realizations visiting another walker | 1077 | 853 |
| foreign-birth arrivals | 1916 | 1487 |
| first recipient visits | 1841 | 1417 |
| exact initial-token transfers | 7 | 19 |
| initial ancestries visiting other walkers | 51 | 51 |

| recipient walker | W accepted transfers received | W distinct donor walkers | W donor IDs | U distinct donor walkers |
| --- | --- | --- | --- | --- |
| 0 | 210 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 |
| 1 | 219 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 |
| 2 | 175 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 |
| 3 | 173 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 |
| 4 | 256 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 |
| 5 | 221 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 |
| 6 | 123 | 7 | [0, 1, 2, 3, 4, 5, 7] | 7 |
| 7 | 181 | 7 | [0, 1, 2, 3, 4, 5, 6] | 7 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 26 | 27 | 30 | 43 | 37 | 16 | 31 |
| 1 | 26 | 0 | 27 | 28 | 49 | 38 | 21 | 30 |
| 2 | 27 | 27 | 0 | 23 | 31 | 26 | 17 | 24 |
| 3 | 30 | 28 | 23 | 0 | 29 | 24 | 12 | 27 |
| 4 | 43 | 49 | 31 | 29 | 0 | 45 | 25 | 34 |
| 5 | 37 | 38 | 26 | 24 | 45 | 0 | 24 | 27 |
| 6 | 16 | 21 | 17 | 12 | 25 | 24 | 0 | 8 |
| 7 | 31 | 30 | 24 | 27 | 34 | 27 | 8 | 0 |

Initial ancestry IDs 0–71 are assigned to the recovered starting blocks for this diagnostic. As in Stage 4U, ancestry survives slice descendants, but an exact-realization token is newly minted whenever the slice targets its block, conservatively even for a roundoff-identical update. Untouched tokens persist; accepted joint exchanges swap tokens and rejection preserves them. Independent replay verifies copied six-coordinate values, all tokens, all cache outcomes and lineage permutations. Full event records are saved in provenance_transfers.json and token-by-token migration counts in comparison.json. These tags are descriptive and never affect pairing or selection.

## Exchange diagnostics

| run | proposals | joint contour survivors | survival fraction | accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage-4U / level 9 | 2048 | 1034 | 0.504882812 | 1002 | 0.489257812 | 32 | 0.0309477756 |
| Stage-4W / level 10 | 2048 | 795 | 0.388183594 | 779 | 0.380371094 | 16 | 0.0201257862 |

| distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- |
| selection_log_ratio | -1.54083908 | -0.0165476125 | 1.36021396e-10 | 0.0480499683 | 8.04306636 | 0 |
| joint_prior_ratio | -1.42108547e-13 | -2.84217094e-14 | 0 | 2.84217094e-14 | 1.70530257e-13 | 0 |

Nonfinite proposed priors: 0; likelihoods: 0; full MH ratios: 0. Minimum finite selector log probability: -914.8153123107547. Selection and MH stay in log space; no epsilon clipping.

| pair | proposals | joint survivors | accepted | acceptance fraction | MH rejection among survivors |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 73 | 28 | 26 | 0.356164384 | 0.0714285714 |
| 0,2 | 73 | 27 | 27 | 0.369863014 | 0 |
| 0,3 | 73 | 30 | 30 | 0.410958904 | 0 |
| 0,4 | 73 | 43 | 43 | 0.589041096 | 0 |
| 0,5 | 73 | 38 | 37 | 0.506849315 | 0.0263157895 |
| 0,6 | 73 | 16 | 16 | 0.219178082 | 0 |
| 0,7 | 74 | 32 | 31 | 0.418918919 | 0.03125 |
| 1,2 | 73 | 27 | 27 | 0.369863014 | 0 |
| 1,3 | 73 | 28 | 28 | 0.383561644 | 0 |
| 1,4 | 73 | 49 | 49 | 0.671232877 | 0 |
| 1,5 | 73 | 38 | 38 | 0.520547945 | 0 |
| 1,6 | 74 | 21 | 21 | 0.283783784 | 0 |
| 1,7 | 73 | 30 | 30 | 0.410958904 | 0 |
| 2,3 | 73 | 25 | 23 | 0.315068493 | 0.08 |
| 2,4 | 73 | 31 | 31 | 0.424657534 | 0 |
| 2,5 | 74 | 26 | 26 | 0.351351351 | 0 |
| 2,6 | 73 | 20 | 17 | 0.232876712 | 0.15 |
| 2,7 | 73 | 25 | 24 | 0.328767123 | 0.04 |
| 3,4 | 74 | 29 | 29 | 0.391891892 | 0 |
| 3,5 | 73 | 24 | 24 | 0.328767123 | 0 |
| 3,6 | 73 | 13 | 12 | 0.164383562 | 0.0769230769 |
| 3,7 | 73 | 30 | 27 | 0.369863014 | 0.1 |
| 4,5 | 73 | 45 | 45 | 0.616438356 | 0 |
| 4,6 | 73 | 25 | 25 | 0.342465753 | 0 |
| 4,7 | 73 | 34 | 34 | 0.465753425 | 0 |
| 5,6 | 73 | 24 | 24 | 0.328767123 | 0 |
| 5,7 | 73 | 28 | 27 | 0.369863014 | 0.0357142857 |
| 6,7 | 73 | 9 | 8 | 0.109589041 | 0.111111111 |

The fixed schedule executes 73 complete seven-round cycles plus the first round. Four pairs have 74 opportunities and the remaining 24 have 73; every walker has 512 pair opportunities.

## Cost, integrity and stopping point

| cost counter | Stage-4U / level 9 | Stage-4W / level 10 |
| --- | --- | --- |
| slice_likelihood_proxy | 27152 | 27925 |
| exchange_likelihood | 4096 | 4096 |
| slice_prior_proxy | 27152 | 27925 |
| exchange_prior | 4096 | 4096 |
| direct_cache_likelihood | 8200 | 8200 |
| direct_cache_prior | 8200 | 8200 |
| total_likelihood_proxy | 39448 | 40221 |
| total_prior_proxy | 39448 | 40221 |

Slice counters retain the historical num_steps+num_shrink proxy; scalar prior and likelihood are evaluated together. Exchange evaluations and direct cache checks are counted separately, including the eight initial-state checks. There was no cost-based truncation. CPU loop wall time 69.9163862 s is descriptive only.

All 4096 slice outcomes and 4096 pair-member exchange outcomes passed finite-coordinate/prior/likelihood, strict ell10 and direct cache checks. Maximum absolute prior-cache error 0.0; likelihood-cache error 0.0 (atol 1e-9 and 1e-7 respectively, rtol 0). Selected six-coordinate blocks swapped exactly, untouched blocks were unchanged, rejection restored both post-slice states, and acceptance committed both jointly. Saved selector probabilities, full MH decisions, state/cache outcomes and provenance were independently replayed. Structural failures: **0**. Protected inputs and the full frozen ladder retained their hashes.

**223 tests passed before sampling:** all previous 212 tests preserved plus 11 cheap Stage-4W cases covering exact frozen checkpoint, last-valid fallback/cache copying, no-survivor stop, strict ell10 enforcement for both walkers, unchanged transition settings/source equivalence, diagnostic-only threshold with no construction/freeze path, walker-6 criterion and sweep numbering, and accepted/rejected provenance replay with tamper detection. No real LISA trajectory is run in pytest.

Artifacts are in `/tmp/lisa_dns_stage4w_level10_validation/`: frozen design.json, copied checkpoint_level10.json, initial_states.npz, random_schedule.npz, report.json, trace.npz, exchanges.npz, provenance.json, provenance_transfers.json, comparison.json, decision.json and validation.json. The trace includes all 512 post-exchange states plus constituent states and caches.

No level 11 was constructed or independently calibrated. No threshold, mass, selector, pairing, exchange frequency, slice kernel or prior was tuned. No production or evidence implementation was run. Evidence remains unvalidated and out of scope. Yes: communication at the frozen level 10 is sufficient to justify ONE controlled level-11 construction attempt in a later stage; none is performed here. Stop here.
