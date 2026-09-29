# Stage 5A: fixed level-12 population mixing validation

**Classification B — usable but residual heterogeneity.** The frozen level-12 population is sufficiently well mixed to justify exactly ONE controlled level-13 construction attempt in a later stage. This does not authorize further growth or establish physical convergence.

Cross-walker logL agreement is strong: full-run mean-logL SD 49.2881, median/max pairwise KS 0.05859375/0.09765625, and diagnostic upper-tail fractions 0.330078125–0.40625 (SD 0.0301087; conservative ESS 1425.46). Every walker receives 273–288 accepted transfers from all seven donors; exact replay and all structural checks pass. However, the mean-logL SD rises from 29.6283 in the first half to 88.7502 in the second, and pooled mean logL shifts upward by about 97.44. Second-half tail fractions widen to 0.28515625–0.4140625. These residual temporal and cross-walker differences support a convincing B rather than an unqualified convergence claim. Exchange acceptance is supporting evidence, not the classification criterion.

## Frozen ladder and design

Loaded the exact Stage-4Z checkpoint with checksum, settings, state and PRNG-schema validation. Its entire level-0..11 prefix matches Stage 4X, whose prefix matches Stage 4V and Stage 4G. Saved Stage-4Z levels.npz also matches byte for byte. The original checkpoint and all protected inputs retained their hashes. Stage 4Z had no validation.json at handoff; its completed reports, protected hashes, checkpoint payload checksum and recorded exact replay results were checked directly. No historical artifact was changed.

| level | ell | log X |
| --- | --- | --- |
| 0 | -inf | 0 |
| 1 | -113725.6589 | -1.034317938 |
| 2 | -113448.9963 | -1.999643004 |
| 3 | -113172.8482 | -3.11847793 |
| 4 | -112838.3656 | -4.172213954 |
| 5 | -112388.109 | -5.014640884 |
| 6 | -111915.6905 | -5.840204009 |
| 7 | -111344.9657 | -6.561075737 |
| 8 | -110809.8214 | -7.156607159 |
| 9 | -110252.9948 | -7.675440016 |
| 10 | -109780.2888 | -8.394305726 |
| 11 | -108868.6008 | -9.643899892 |
| 12 | -108187.9971 | -10.41839271 |

Authoritative endpoints remain exactly `ell_11 = -108868.60077896297`, `log X_11 = -9.643899891984567`, `ell_12 = -108187.99712765554`, `log X_12 = -10.418392711998465`.

Exactly eight walkers, 512 sweeps, no burn-in removal, no extension. Each sweep applies eight isotropic constrained-slice updates, followed by four disjoint reciprocal frequency-guided exchanges on the unchanged seven-round schedule. Both slice and exchange checks use strict ell12. Full implemented joint-prior change and recomputed log q_reverse − log q_forward enter one joint MH decision; both walkers advance or neither does. No diagnostic threshold enters the transition.

Frozen settings: scales pi/sqrt(3); mixture 20% global, 40% labelled single source, 40% labelled unordered pair; max_steps=10, max_shrinkage=100; f_star=0.0018409808, sigma_f=2.558e-07. No source_gain calls. Transition source equivalence to Stage 4Y is tested apart from contour/checkpoint metadata; shared selector, swap, commit and schedule functions are identical objects.

Slice seeds: `[940100, 940101, 940102, 940103, 940104, 940105, 940106, 940107]`; Gumbel label seed 2026092917; joint-MH uniform seed 2026092918. Starts and random schedules were saved before sampling, with no retries.

## Starting ensemble

Use each final Stage-4Z calibration state if valid; otherwise the same walker’s LAST retained strict ell12 survivor, never its maximum-likelihood state. Indices are zero-based; retained indices subtract the original 128-sweep calibration burn-in. Positions and both caches copy exactly from that trace.

| walker | fallback required | retained index | absolute trace index | starting logL |
| --- | --- | --- | --- | --- |
| 0 | False | 255 | 383 | -107961.74030917077 |
| 1 | True | 254 | 382 | -108095.61134281836 |
| 2 | True | 254 | 382 | -107615.99568137276 |
| 3 | False | 255 | 383 | -107854.32441057786 |
| 4 | True | 253 | 381 | -107938.52776163933 |
| 5 | True | 246 | 374 | -107896.709403001 |
| 6 | True | 254 | 382 | -107714.82856569697 |
| 7 | False | 255 | 383 | -107942.4107417032 |

## Cross-walker logL agreement

| run | window | mean SD | mean MAD | mean IQR | median KS | maximum KS |
| --- | --- | --- | --- | --- | --- | --- |
| 4W / level 10 | first256 | 289.732878 | 41.53447451 | 117.2260699 | 0.115234375 | 0.578125 |
| 4W / level 10 | second256 | 128.7188847 | 65.67569622 | 129.5186029 | 0.115234375 | 0.23828125 |
| 4W / level 10 | all512 | 173.6039059 | 60.63175651 | 130.8859049 | 0.1005859375 | 0.376953125 |
| 4Y / level 11 | first256 | 94.45833508 | 81.91017152 | 164.4303527 | 0.1328125 | 0.2109375 |
| 4Y / level 11 | second256 | 110.4968727 | 45.25566312 | 66.58359997 | 0.095703125 | 0.23046875 |
| 4Y / level 11 | all512 | 45.46360982 | 21.82903396 | 44.62632017 | 0.0703125 | 0.138671875 |
| 5A / level 12 | first256 | 29.62826432 | 18.39333104 | 26.71584735 | 0.08203125 | 0.12890625 |
| 5A / level 12 | second256 | 88.75020205 | 79.69802022 | 130.940008 | 0.08984375 | 0.18359375 |
| 5A / level 12 | all512 | 49.28813107 | 41.26843159 | 68.84209646 | 0.05859375 | 0.09765625 |

SD uses ddof=1, MAD is the unscaled median absolute deviation of the eight walker means, and IQR uses linear quartiles. KS is descriptive; no IID p-value interpretation is used.

| walker | mean logL / first256 | mean logL / second256 | mean logL / all512 |
| --- | --- | --- | --- |
| 0 | -107506.9627 | -107280.7773 | -107393.87 |
| 1 | -107543.0695 | -107316.4121 | -107429.7408 |
| 2 | -107492.4996 | -107478.0745 | -107485.2871 |
| 3 | -107447.7752 | -107364.9998 | -107406.3875 |
| 4 | -107532.3922 | -107529.3234 | -107530.8578 |
| 5 | -107508.1068 | -107393.6415 | -107450.8741 |
| 6 | -107529.2862 | -107500.948 | -107515.1171 |
| 7 | -107509.5971 | -107425.9851 | -107467.7911 |

| pair | KS / first256 | KS / second256 | KS / all512 |
| --- | --- | --- | --- |
| 0,1 | 0.1015625 | 0.07421875 | 0.08203125 |
| 0,2 | 0.05859375 | 0.0859375 | 0.05078125 |
| 0,3 | 0.08203125 | 0.0625 | 0.046875 |
| 0,4 | 0.0546875 | 0.15234375 | 0.08984375 |
| 0,5 | 0.08203125 | 0.08203125 | 0.072265625 |
| 0,6 | 0.07421875 | 0.15234375 | 0.09375 |
| 0,7 | 0.09765625 | 0.0625 | 0.0546875 |
| 1,2 | 0.12890625 | 0.09375 | 0.095703125 |
| 1,3 | 0.10546875 | 0.07421875 | 0.08984375 |
| 1,4 | 0.09375 | 0.1171875 | 0.056640625 |
| 1,5 | 0.1015625 | 0.06640625 | 0.052734375 |
| 1,6 | 0.046875 | 0.13671875 | 0.064453125 |
| 1,7 | 0.06640625 | 0.07421875 | 0.052734375 |
| 2,3 | 0.1015625 | 0.1015625 | 0.05078125 |
| 2,4 | 0.0703125 | 0.140625 | 0.087890625 |
| 2,5 | 0.125 | 0.08203125 | 0.052734375 |
| 2,6 | 0.09765625 | 0.12109375 | 0.09765625 |
| 2,7 | 0.12890625 | 0.0625 | 0.060546875 |
| 3,4 | 0.08203125 | 0.15625 | 0.0859375 |
| 3,5 | 0.08203125 | 0.0625 | 0.0625 |
| 3,6 | 0.09765625 | 0.18359375 | 0.09765625 |
| 3,7 | 0.0859375 | 0.078125 | 0.05859375 |
| 4,5 | 0.078125 | 0.140625 | 0.0546875 |
| 4,6 | 0.06640625 | 0.08203125 | 0.046875 |
| 4,7 | 0.07421875 | 0.1484375 | 0.05859375 |
| 5,6 | 0.09765625 | 0.15234375 | 0.0546875 |
| 5,7 | 0.08203125 | 0.05078125 | 0.029296875 |
| 6,7 | 0.0546875 | 0.13671875 | 0.0546875 |

## Diagnostic upper tail and lineage behavior

Only after all 512 sweeps completed: `ell_13_diag = -107587.9625651654`, pooled ascending order index 2589 of 4096, floor(N*(1-exp(-1))). Exceedances are strict. This is not level 13; it has no log mass, creates no checkpoint, enters no transition, and receives no independent calibration. The same threshold is applied to both halves.

| walker | full exceedance | first-half exceedance | second-half exceedance | first-half mean logL | second-half mean logL |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.40625 | 0.3984375 | 0.4140625 | -107506.9627 | -107280.7773 |
| 1 | 0.3359375 | 0.31640625 | 0.35546875 | -107543.0695 | -107316.4121 |
| 2 | 0.384765625 | 0.3828125 | 0.38671875 | -107492.4996 | -107478.0745 |
| 3 | 0.396484375 | 0.40234375 | 0.390625 | -107447.7752 | -107364.9998 |
| 4 | 0.333984375 | 0.3828125 | 0.28515625 | -107532.3922 | -107529.3234 |
| 5 | 0.37890625 | 0.34765625 | 0.41015625 | -107508.1068 | -107393.6415 |
| 6 | 0.330078125 | 0.34375 | 0.31640625 | -107529.2862 | -107500.948 |
| 7 | 0.375 | 0.33984375 | 0.41015625 | -107509.5971 | -107425.9851 |

| window | pooled fraction | min | max | SD | conservative ESS | IID SE | block-means SE | between-walker SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first256 | 0.3642578125 | 0.31640625 | 0.40234375 | 0.03137183282 | 643.226212 | 0.01063359162 | 0.01897418307 | 0.01109161786 |
| second256 | 0.37109375 | 0.28515625 | 0.4140625 | 0.04802350441 | 785.3178484 | 0.01067504708 | 0.01723899888 | 0.01697887281 |
| all512 | 0.3676757812 | 0.330078125 | 0.40625 | 0.03010872892 | 1425.455461 | 0.007533945011 | 0.01277102026 | 0.0106450432 |

The unchanged conservative Bernoulli-equivalent ESS uses the maximum of IID, block-means and between-walker SE; block size 32. Interacting walkers are dependent, and this data-selected diagnostic is not independent calibration.

Severe low-tail behavior largely disappears. The lowest first-half diagnostic fraction belongs to walker 1 (0.31640625); the lowest second-half fraction belongs to walker 4 (0.28515625), so the relative deficit changes lineage. Walker 6 remains modestly below the pooled fraction in both halves (0.34375 then 0.31640625), but is not isolated: it receives 279 transfers from all seven donors, enters the other-seven IQR at sweep 2, and spends 42.38% of post-transition sweeps there. Walker 3 is central in both halves (0.40234375 then 0.390625). These are symmetric post-run observations, not preselected exceptions. Mean-logL changes differ across lineages, from about +3.1 for walker 4 to +226.7 for walker 1, so complete stationarity is not established.

| run | window | upper-tail min–max | upper-tail SD | upper-tail ESS | median logL ESS |
| --- | --- | --- | --- | --- | --- |
| 4W / level 10 | first256 | [0.03125, 0.515625] | 0.1538565286 | 79.76983287 | 80.2591989 |
| 4W / level 10 | second256 | [0.2890625, 0.50390625] | 0.07623266233 | 314.6114029 | 80.43088735 |
| 4W / level 10 | all512 | [0.162109375, 0.466796875] | 0.09546509439 | 204.0824289 | 142.817338 |
| 4Y / level 11 | first256 | [0.30859375, 0.42578125] | 0.050371804 | 589.961804 | 68.27989116 |
| 4Y / level 11 | second256 | [0.24609375, 0.44140625] | 0.05785433754 | 396.4889711 | 62.83193356 |
| 4Y / level 11 | all512 | [0.3203125, 0.40234375] | 0.02765586379 | 956.2878466 | 128.2574497 |
| 5A / level 12 | first256 | [0.31640625, 0.40234375] | 0.03137183282 | 643.226212 | 118.2962698 |
| 5A / level 12 | second256 | [0.28515625, 0.4140625] | 0.04802350441 | 785.3178484 | 140.9576694 |
| 5A / level 12 | all512 | [0.330078125, 0.40625] | 0.03010872892 | 1425.455461 | 259.3456851 |

## Symmetric recovery diagnostic

For each of all eight walkers separately: inclusive contemporaneous 25th-75th percentiles of the OTHER seven walkers, numpy linear quantiles; sweep 0 is the recovered ensemble; sweeps 1..512 are post-exchange states; no smoothing or criterion retuning. The occupancy denominator is exactly the 512 post-transition sweeps; the initial state can establish first entry at sweep 0. A first hit is descriptive, not proof of sustained recovery.

| walker | initially inside | first entry sweep | fraction of sweeps inside | first-half fraction inside | second-half fraction inside |
| --- | --- | --- | --- | --- | --- |
| 0 | False | 1 | 0.349609375 | 0.328125 | 0.37109375 |
| 1 | False | 2 | 0.416015625 | 0.45703125 | 0.375 |
| 2 | False | 4 | 0.416015625 | 0.44921875 | 0.3828125 |
| 3 | True | 0 | 0.388671875 | 0.34765625 | 0.4296875 |
| 4 | True | 0 | 0.376953125 | 0.359375 | 0.39453125 |
| 5 | True | 0 | 0.3828125 | 0.3515625 | 0.4140625 |
| 6 | False | 2 | 0.423828125 | 0.44140625 | 0.40625 |
| 7 | True | 0 | 0.4296875 | 0.40625 | 0.453125 |

## Autocorrelation

| window | median logL ESS | median labelled f0_u ESS | minimum labelled f0_u ESS |
| --- | --- | --- | --- |
| first256 | 118.2962698 | 6.351666898 | 2.816754978 |
| second256 | 140.9576694 | 6.768810635 | 2.963610408 |
| all512 | 259.3456851 | 8.119773103 | 3.157856197 |

| walker | logL ESS / first256 | logL ESS / second256 | logL ESS / all512 | unique-state fraction / first256 | unique-state fraction / second256 | unique-state fraction / all512 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 107.8916786 | 39.90566356 | 82.71238965 | 1 | 1 | 1 |
| 1 | 127.3364202 | 139.1250788 | 254.2999856 | 1 | 1 | 1 |
| 2 | 146.7831106 | 155.7982921 | 309.2256242 | 1 | 1 | 1 |
| 3 | 130.9145438 | 185.2359862 | 314.0062181 | 1 | 1 | 1 |
| 4 | 81.99623297 | 118.6390456 | 197.9958537 | 1 | 1 | 1 |
| 5 | 159.4752359 | 174.4880865 | 320.3314857 | 1 | 1 | 1 |
| 6 | 109.2561194 | 142.7902599 | 264.3913846 | 1 | 1 | 1 |
| 7 | 74.83778106 | 85.84065471 | 154.0418992 | 1 | 1 | 1 |

The established initial-positive monotone paired autocorrelation estimator is reused. Labelled coordinate ESS alone is not interpreted as physical convergence.

## Permutation-invariant physical communication

Unchanged Stage-4I Hungarian source matching: frequency RMS in Fourier bins (DF=1/31536000 Hz), sorted-frequency mean distance in bins, and six-coordinate source RMS normalized by prior widths, with psi period pi and lam period 2pi. Frequency summaries use all cross-time pairs; six-coordinate summaries use every 32nd endpoint and all cross-pairs of these endpoints. Final window means the second 256 sweeps. There is no geometry gate.

| run | window | frequency-set RMS | sorted-frequency mean distance | six-coordinate RMS |
| --- | --- | --- | --- | --- |
| 4W / level 10 | first256 | 105.5664159 | 58.90030167 | 0.2561319527 |
| 4W / level 10 | second256 | 96.71137187 | 46.73598124 | 0.2568235615 |
| 4W / level 10 | all512 | 97.5332267 | 40.67097855 | 0.2573508172 |
| 4Y / level 11 | first256 | 105.0046026 | 52.07263352 | 0.2536747523 |
| 4Y / level 11 | second256 | 114.5776211 | 68.18900439 | 0.2559827457 |
| 4Y / level 11 | all512 | 114.7501261 | 54.99933142 | 0.2559176632 |
| 5A / level 12 | first256 | 100.7049204 | 45.90095994 | 0.2589726914 |
| 5A / level 12 | second256 | 99.21436365 | 57.74749894 | 0.2538485427 |
| 5A / level 12 | all512 | 98.69434335 | 38.72077218 | 0.2587586273 |

| final-window pair | frequency-set RMS | sorted-frequency mean distance | six-coordinate RMS |
| --- | --- | --- | --- |
| 0,1 | 87.433352 | 18.79865795 | 0.2636469311 |
| 0,2 | 96.6528556 | 70.63438521 | 0.2562646076 |
| 0,3 | 79.52893298 | 35.18912287 | 0.2401191825 |
| 0,4 | 91.19829492 | 64.68883314 | 0.2509986426 |
| 0,5 | 100.6185762 | 57.47763618 | 0.2504177971 |
| 0,6 | 94.56161431 | 58.01736171 | 0.2574405536 |
| 0,7 | 105.8209473 | 63.43874121 | 0.260026347 |
| 1,2 | 104.1188192 | 61.63131697 | 0.2406875829 |
| 1,3 | 90.28784574 | 19.37955389 | 0.2539920037 |
| 1,4 | 105.1180139 | 65.01970332 | 0.256229906 |
| 1,5 | 106.2621044 | 42.20739792 | 0.2523256612 |
| 1,6 | 114.5652276 | 69.71051254 | 0.2772276904 |
| 1,7 | 113.7059733 | 55.36874747 | 0.2636091433 |
| 2,3 | 85.39016193 | 49.5357967 | 0.2337443759 |
| 2,4 | 75.91738571 | 43.35732696 | 0.2473992424 |
| 2,5 | 97.81015107 | 54.72932162 | 0.2480925599 |
| 2,6 | 135.5947318 | 114.2757142 | 0.2692100515 |
| 2,7 | 86.50087129 | 23.61554461 | 0.2499549851 |
| 3,4 | 92.15635616 | 62.250973 | 0.2414866622 |
| 3,5 | 89.81867784 | 27.60161574 | 0.2400775724 |
| 3,6 | 115.942972 | 85.91717895 | 0.2572964776 |
| 3,7 | 97.65991857 | 43.34436779 | 0.2540741607 |
| 4,5 | 110.6080344 | 76.69866056 | 0.2476966813 |
| 4,6 | 124.654335 | 102.1361105 | 0.268280697 |
| 4,7 | 91.92334611 | 41.77641718 | 0.2596901508 |
| 5,6 | 141.6003114 | 110.7058927 | 0.2537050817 |
| 5,7 | 106.7762799 | 44.81747491 | 0.2450169453 |
| 6,7 | 145.9658821 | 114.7083495 | 0.2714501389 |

## Exact source-block provenance

| run | directional transfers | distinct migrated realizations | foreign-birth arrivals | first recipient visits | initial ancestries visiting another walker |
| --- | --- | --- | --- | --- | --- |
| 4W / level 10 | 1558 | 853 | 1487 | 1417 | 51 |
| 4Y / level 11 | 1990 | 1012 | 1899 | 1825 | 44 |
| 5A / level 12 | 2250 | 1057 | 2140 | 2019 | 30 |

| walker | accepted transfers received | distinct donor walkers | donor IDs | distinct exact realizations received |
| --- | --- | --- | --- | --- |
| 0 | 275 | 7 | [1, 2, 3, 4, 5, 6, 7] | 244 |
| 1 | 282 | 7 | [0, 2, 3, 4, 5, 6, 7] | 262 |
| 2 | 284 | 7 | [0, 1, 3, 4, 5, 6, 7] | 267 |
| 3 | 288 | 7 | [0, 1, 2, 4, 5, 6, 7] | 271 |
| 4 | 281 | 7 | [0, 1, 2, 3, 5, 6, 7] | 266 |
| 5 | 288 | 7 | [0, 1, 2, 3, 4, 6, 7] | 276 |
| 6 | 279 | 7 | [0, 1, 2, 3, 4, 5, 7] | 267 |
| 7 | 273 | 7 | [0, 1, 2, 3, 4, 5, 6] | 264 |

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 41 | 45 | 36 | 36 | 41 | 39 | 37 |
| 1 | 41 | 0 | 39 | 43 | 40 | 45 | 34 | 40 |
| 2 | 45 | 39 | 0 | 43 | 40 | 36 | 42 | 39 |
| 3 | 36 | 43 | 43 | 0 | 41 | 40 | 41 | 44 |
| 4 | 36 | 40 | 40 | 41 | 0 | 46 | 40 | 38 |
| 5 | 41 | 45 | 36 | 40 | 46 | 0 | 44 | 36 |
| 6 | 39 | 34 | 42 | 41 | 40 | 44 | 0 | 39 |
| 7 | 37 | 40 | 39 | 44 | 38 | 36 | 39 | 0 |

The established provenance convention is unchanged: 72 initial ancestry IDs follow slice descendants; each slice-targeted block mints a new exact-realization token, conservatively even for roundoff-identical updates. Untouched tokens persist. Joint acceptance swaps tokens and ancestry; rejection preserves both. Exact token IDs for every arrival are in provenance_transfers.json, with migration counts in comparison.json. Provenance demonstrates communication, not convergence by itself.

## Exchange diagnostics

| run | proposals | contour survivors | survival fraction | accepted | acceptance fraction | MH rejected survivors | MH rejection fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4W / level 10 | 2048 | 795 | 0.3881835938 | 779 | 0.3803710938 | 16 | 0.02012578616 |
| 4Y / level 11 | 2048 | 1009 | 0.4926757812 | 995 | 0.4858398438 | 14 | 0.01387512389 |
| 5A / level 12 | 2048 | 1130 | 0.5517578125 | 1125 | 0.5493164062 | 5 | 0.004424778761 |

| distribution | min | q25 | median | q75 | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- |
| selection_log_ratio | -0.2533415025 | -0.0004259924404 | 4.044252354e-07 | 0.006342355489 | 4.082995353 | 0 |
| joint_prior_ratio | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.98951966e-13 | 0 |

Nonfinite proposed priors: 0; proposed likelihoods: 0; MH ratios: 0. All realized coordinates, priors and likelihoods are finite; selector probabilities are finite in log space. Minimum selector log probability -903.8337635784936.

| pair | proposals | contour survivors | accepted | acceptance fraction | MH rejection among survivors |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 73 | 41 | 41 | 0.5616438356 | 0 |
| 0,2 | 73 | 45 | 45 | 0.6164383562 | 0 |
| 0,3 | 73 | 36 | 36 | 0.4931506849 | 0 |
| 0,4 | 73 | 36 | 36 | 0.4931506849 | 0 |
| 0,5 | 73 | 41 | 41 | 0.5616438356 | 0 |
| 0,6 | 73 | 39 | 39 | 0.5342465753 | 0 |
| 0,7 | 74 | 37 | 37 | 0.5 | 0 |
| 1,2 | 73 | 39 | 39 | 0.5342465753 | 0 |
| 1,3 | 73 | 43 | 43 | 0.5890410959 | 0 |
| 1,4 | 73 | 41 | 40 | 0.5479452055 | 0.0243902439 |
| 1,5 | 73 | 46 | 45 | 0.6164383562 | 0.02173913043 |
| 1,6 | 74 | 34 | 34 | 0.4594594595 | 0 |
| 1,7 | 73 | 41 | 40 | 0.5479452055 | 0.0243902439 |
| 2,3 | 73 | 43 | 43 | 0.5890410959 | 0 |
| 2,4 | 73 | 41 | 40 | 0.5479452055 | 0.0243902439 |
| 2,5 | 74 | 36 | 36 | 0.4864864865 | 0 |
| 2,6 | 73 | 42 | 42 | 0.5753424658 | 0 |
| 2,7 | 73 | 39 | 39 | 0.5342465753 | 0 |
| 3,4 | 74 | 41 | 41 | 0.5540540541 | 0 |
| 3,5 | 73 | 41 | 40 | 0.5479452055 | 0.0243902439 |
| 3,6 | 73 | 41 | 41 | 0.5616438356 | 0 |
| 3,7 | 73 | 44 | 44 | 0.602739726 | 0 |
| 4,5 | 73 | 46 | 46 | 0.6301369863 | 0 |
| 4,6 | 73 | 40 | 40 | 0.5479452055 | 0 |
| 4,7 | 73 | 38 | 38 | 0.5205479452 | 0 |
| 5,6 | 73 | 44 | 44 | 0.602739726 | 0 |
| 5,7 | 73 | 36 | 36 | 0.4931506849 | 0 |
| 6,7 | 73 | 39 | 39 | 0.5342465753 | 0 |

73 complete seven-round cycles plus round 0: four pairs have 74 proposals and the other 24 pairs have 73. Every walker has exactly 512 exchange opportunities.

## Cost and integrity

| cost counter | 4W / level 10 | 4Y / level 11 | 5A / level 12 |
| --- | --- | --- | --- |
| slice_likelihood_proxy | 27925 | 27868 | 28767 |
| exchange_likelihood | 4096 | 4096 | 4096 |
| direct_cache_likelihood | 8200 | 8200 | 8200 |
| slice_prior_proxy | 27925 | 27868 | 28767 |
| exchange_prior | 4096 | 4096 | 4096 |
| direct_cache_prior | 8200 | 8200 | 8200 |
| total_likelihood_proxy | 40221 | 40164 | 41063 |
| total_prior_proxy | 40221 | 40164 | 41063 |

Slice cost remains num_steps+num_shrink. Exchange and direct cache evaluations, including starts, are separate. All three runs use 512 sweeps: differences in totals reflect slice work, not truncation or changed exchange frequency. Stage-5A CPU loop wall time 71.29015307 seconds is descriptive only.

All 4096 slice outcomes and 4096 pair-member exchange outcomes passed finite coordinate/prior/likelihood, strict ell12 and direct cache checks. Disjoint operations are evaluated in batches; each individual outcome is checked. Maximum prior-cache error 0.0; likelihood-cache error 0.0 (rtol 0; atol 1e-9 and 1e-7). Every selected block swapped exactly, untouched coordinates stayed unchanged, rejection restored both states exactly, and acceptance advanced both jointly. Saved operation and provenance replay passed exactly. No structural failures. Levels 0–12, selector and pairing remained unchanged.

215 example-level DNS tests passed before sampling, including 19 new cheap Stage-5A cases. Core DNS test results are recorded separately in tests_core.log. Existing tests were preserved; no real LISA trajectory runs in pytest. New coverage includes immutable frozen prefix, exact endpoint, last-valid recovery/cache copying, missing-survivor STOP, strict contour use for both walkers, frozen transition source equivalence, diagnostic-only schema with no construction path, all eight recovery rules and sweep numbering, and accepted/rejected provenance replay with tamper detection.

Artifacts: `/tmp/lisa_dns_stage5a_level12_validation/` contains design.json, copied checkpoint_level12.json, initial_states.npz, random_schedule.npz, report.json, full trace.npz, exchanges.npz, provenance.json, provenance_transfers.json, comparison.json, decision.json, validation.json and test logs. The copy is the existing level-12 checkpoint, not a newly constructed level.

Across frozen levels 10, 11 and 12, the same kernel retains qualitative communication. Full-run mean-logL SD is 173.60, 45.46 and 49.29; maximum pairwise KS is 0.3770, 0.1387 and 0.0977; diagnostic-tail ESS is 204.1, 956.3 and 1425.5. Median logL ESS is 142.8, 128.3 and 259.3. Full-window six-coordinate source-set RMS remains about 0.2574, 0.2559 and 0.2588, with frequency-set RMS about 97.5, 114.8 and 98.7 bins. These physical distances do not prove whole-catalogue convergence, but show no new gross separation on level 12. All-to-all donor communication continues. Total likelihood-cost proxy is 40221, 40164 and 41063: level 12 is about 2.1% above level 10 and 2.2% above level 11, entirely from slice work. No identical-number requirement or new gate is applied.

The frozen level-12 population is sufficiently well mixed to justify exactly ONE controlled level-13 construction attempt in a later stage. This does not authorize further growth or establish physical convergence. No level 13, independent calibration, tuning, extended run, production or evidence implementation was performed. Evidence reconstruction remains unvalidated and out of scope. STOP.
