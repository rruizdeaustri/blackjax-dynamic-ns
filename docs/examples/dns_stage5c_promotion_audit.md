# Stage 5C: read-only promotion audit

**Stage-5B remains FAILED under its promotion/survivor contract.** No level 13 exists. The ladder remains `ell_12 = -108187.99712765554`, `log X_12 = -10.418392711998465`; the unchanged `ell_13_candidate = -107123.66036615883` is only a failed-attempt candidate. Stage 5C neither changes that decision nor substitutes a sensitivity threshold.

The evidence favors an unusually poor finite selection ensemble and inadequate mixing within the fixed budget, exposed by a conservative same-walker promotion rule. It does not establish an incorrect MH kernel or a general inability to communicate at ell12. Same-walker ancestry is not a mathematical invariance requirement. One bank-local pooled-with-replacement protocol is worth validating separately; it does not retroactively make Stage 5B successful.

This report resumes the completed saved calculations in `/tmp/lisa_dns_stage5c_readonly_diagnosis/diagnostics.json`; that file was not regenerated or overwritten after the interruption. Inputs are saved Stage-5B states, starts, exchanges, provenance and the copied level-12 checkpoint, plus saved Stage-5A comparison data. No historical validation other than the requested Stage-5A comparison was revisited for this audit. No LISA model, sampling routine, or promotion routine is imported by the diagnostic module.

## A. Exact survivor accounting

Membership is strictly `logL > -107123.66036615883`. The primary sample is exactly 256 retained states per walker, absolute zero-based trace indices 128–383. First/last indices below are retained indices 0–255; add 128 for absolute trace indices. A block is one of eight nonoverlapping contiguous 32-state blocks. Longest zero interval counts consecutive retained states, including leading/trailing runs. Burn-in is excluded.

**5B selection — retained:**

| walker | survivors | fraction | first index | last index | occupied 32-state blocks | longest zero interval | first-128 fraction | second-128 fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 134 | 0.5234375 | 4 | 254 | 8 | 8 | 0.4296875 | 0.6171875 |
| 1 | 138 | 0.5390625 | 2 | 255 | 8 | 10 | 0.421875 | 0.65625 |
| 2 | 131 | 0.51171875 | 10 | 255 | 8 | 14 | 0.453125 | 0.5703125 |
| 3 | 1 | 0.00390625 | 32 | 32 | 1 | 223 | 0.0078125 | 0 |
| 4 | 122 | 0.4765625 | 8 | 255 | 8 | 16 | 0.3671875 | 0.5859375 |
| 5 | 109 | 0.42578125 | 0 | 255 | 8 | 16 | 0.3359375 | 0.515625 |
| 6 | 0 | 0 | — | — | 0 | 256 | 0 | 0 |
| 7 | 118 | 0.4609375 | 0 | 255 | 8 | 12 | 0.3984375 | 0.5234375 |

**5B calibration — retained:**

| walker | survivors | fraction | first index | last index | occupied 32-state blocks | longest zero interval | first-128 fraction | second-128 fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 37 | 0.14453125 | 12 | 246 | 7 | 87 | 0.0546875 | 0.234375 |
| 1 | 46 | 0.1796875 | 30 | 255 | 7 | 36 | 0.0703125 | 0.2890625 |
| 2 | 34 | 0.1328125 | 25 | 242 | 7 | 77 | 0.078125 | 0.1875 |
| 3 | 29 | 0.11328125 | 9 | 247 | 8 | 37 | 0.0625 | 0.1640625 |
| 4 | 31 | 0.12109375 | 4 | 223 | 7 | 35 | 0.0859375 | 0.15625 |
| 5 | 19 | 0.07421875 | 26 | 243 | 6 | 82 | 0.03125 | 0.1171875 |
| 6 | 24 | 0.09375 | 34 | 229 | 6 | 92 | 0.015625 | 0.171875 |
| 7 | 46 | 0.1796875 | 23 | 255 | 8 | 40 | 0.1484375 | 0.2109375 |

**Selection walker 6 has zero retained survivors**, no first/last survivor, zero occupied blocks and a 256-state zero interval. Walker 3 has one retained survivor, index 32 (absolute index 160), followed by 223 consecutive failures. Those two labels remain low across both retained halves. Calibration walkers 3 and 6 instead have 29 and 24 retained survivors, with second-half fractions 0.1640625 and 0.171875.

Burn-in accounting is separate. These 128 states are not added to the construction sample and cannot rescue promotion:

| 5B selection walker | burn-in survivors | fraction | first absolute index | last absolute index | occupied / 4 blocks | longest zero interval |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 34 | 0.265625 | 45 | 125 | 3 | 45 |
| 1 | 28 | 0.21875 | 41 | 115 | 3 | 41 |
| 2 | 20 | 0.15625 | 47 | 123 | 3 | 47 |
| 3 | 2 | 0.015625 | 12 | 13 | 1 | 114 |
| 4 | 23 | 0.1796875 | 13 | 122 | 4 | 26 |
| 5 | 24 | 0.1875 | 14 | 127 | 4 | 26 |
| 6 | 19 | 0.1484375 | 25 | 92 | 3 | 35 |
| 7 | 21 | 0.1640625 | 47 | 126 | 3 | 47 |

| 5B calibration walker | burn-in survivors | fraction | first absolute index | last absolute index | occupied / 4 blocks | longest zero interval |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13 | 0.1015625 | 49 | 119 | 3 | 49 |
| 1 | 7 | 0.0546875 | 69 | 111 | 2 | 69 |
| 2 | 13 | 0.1015625 | 16 | 123 | 4 | 31 |
| 3 | 6 | 0.046875 | 7 | 115 | 2 | 96 |
| 4 | 9 | 0.0703125 | 10 | 123 | 3 | 61 |
| 5 | 11 | 0.0859375 | 56 | 115 | 3 | 56 |
| 6 | 4 | 0.03125 | 50 | 112 | 3 | 50 |
| 7 | 7 | 0.0546875 | 48 | 124 | 3 | 48 |

Selection walker 6 had **19 burn-in survivors**, last at absolute index 92, before its retained zero-survivor interval. Thus it was not intrinsically incapable of reaching the candidate support. Its later residence at low logL is the observed failure. No burn-in state is used as a replacement.

## B. Three saved populations on the same ell12 contour

Stage-5B summaries use retained 256 states and halves of 128; Stage-5A uses all 512 states and halves of 256, without burn-in removal. Different trajectory lengths and random streams prevent a controlled causal attribution to starts alone. All KS entries are descriptive distances, without IID p-value claims. SD uses ddof=1; MAD is unscaled; quartiles are linear.

| run | starting mean logL | starting SD | starting MAD | starting IQR | starting min | starting max |
| --- | --- | --- | --- | --- | --- | --- |
| 5B selection | -107944.4467 | 255.0388604 | 103.6986285 | 276.0471487 | -108162.2845 | -107505.0544 |
| 5B calibration | -107877.5185 | 150.4086767 | 53.7079493 | 127.7926842 | -108095.6113 | -107615.9957 |
| 5A validation | -107877.5185 | 150.4086767 | 53.7079493 | 127.7926842 | -108095.6113 | -107615.9957 |

| walker | 5B selection starting logL | 5B calibration starting logL | 5A validation starting logL |
| --- | --- | --- | --- |
| 0 | -107505.0544 | -107961.7403 | -107961.7403 |
| 1 | -108162.2845 | -108095.6113 | -108095.6113 |
| 2 | -108137.138 | -107615.9957 | -107615.9957 |
| 3 | -108138.5725 | -107854.3244 | -107854.3244 |
| 4 | -107950.7938 | -107938.5278 | -107938.5278 |
| 5 | -107593.4166 | -107896.7094 | -107896.7094 |
| 6 | -108051.7897 | -107714.8286 | -107714.8286 |
| 7 | -108016.5236 | -107942.4107 | -107942.4107 |

Stage-5A and Stage-5B calibration starts, positions and both caches are byte-identical; their sampling streams differ. Selection uses its separate Stage-4Z selection history. This provides two trajectories from the calibration-origin ensemble, not two independent ensembles of starting catalogues. No Stage-5A endpoint initialized Stage-5B.

| run | window | states/walker | mean-logL SD | MAD | IQR | median KS | maximum KS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5B selection | first_half | 128 | 365.5791332 | 103.086092 | 331.1701336 | 0.203125 | 0.734375 |
| 5B selection | second_half | 128 | 503.731612 | 130.4652823 | 428.19221 | 0.1875 | 0.8671875 |
| 5B selection | retained | 256 | 432.0439562 | 105.6981434 | 339.1289785 | 0.14453125 | 0.78515625 |
| 5B calibration | first_half | 128 | 33.83197119 | 26.62735235 | 48.38398628 | 0.11328125 | 0.203125 |
| 5B calibration | second_half | 128 | 102.159833 | 93.21076666 | 143.5566474 | 0.15234375 | 0.2734375 |
| 5B calibration | retained | 256 | 54.4834341 | 46.03588611 | 71.8451209 | 0.1015625 | 0.15234375 |
| 5A validation | first_half | 256 | 29.62826432 | 18.39333104 | 26.71584735 | 0.08203125 | 0.12890625 |
| 5A validation | second_half | 256 | 88.75020205 | 79.69802022 | 130.940008 | 0.08984375 | 0.18359375 |
| 5A validation | full | 512 | 49.28813107 | 41.26843159 | 68.84209646 | 0.05859375 | 0.09765625 |

| 5B selection walker | first-half mean logL | second-half mean logL | primary-window mean logL | first-half candidate fraction | second-half candidate fraction | primary candidate fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -107159.3771 | -106948.3431 | -107053.8601 | 0.4296875 | 0.6171875 | 0.5234375 |
| 1 | -107137.8314 | -106786.0987 | -106961.965 | 0.421875 | 0.65625 | 0.5390625 |
| 2 | -107067.1484 | -106842.6385 | -106954.8934 | 0.453125 | 0.5703125 | 0.51171875 |
| 3 | -107908.6325 | -107845.6603 | -107877.1464 | 0.0078125 | 0 | 0.00390625 |
| 4 | -107213.5408 | -106806.1161 | -107009.8285 | 0.3671875 | 0.5859375 | 0.4765625 |
| 5 | -107344.0036 | -106914.6558 | -107129.3297 | 0.3359375 | 0.515625 | 0.42578125 |
| 6 | -108016.0083 | -108069.2314 | -108042.6198 | 0 | 0 | 0 |
| 7 | -107246.8332 | -107067.0467 | -107156.94 | 0.3984375 | 0.5234375 | 0.4609375 |

| 5B calibration walker | first-half mean logL | second-half mean logL | primary-window mean logL | first-half candidate fraction | second-half candidate fraction | primary candidate fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -107804.9721 | -107522.9427 | -107663.9574 | 0.0546875 | 0.234375 | 0.14453125 |
| 1 | -107841.1011 | -107431.2303 | -107636.1657 | 0.0703125 | 0.2890625 | 0.1796875 |
| 2 | -107782.7637 | -107544.6243 | -107663.694 | 0.078125 | 0.1875 | 0.1328125 |
| 3 | -107774.6905 | -107685.1408 | -107729.9157 | 0.0625 | 0.1640625 | 0.11328125 |
| 4 | -107751.8433 | -107583.8097 | -107667.8265 | 0.0859375 | 0.15625 | 0.12109375 |
| 5 | -107814.2302 | -107685.2556 | -107749.7429 | 0.03125 | 0.1171875 | 0.07421875 |
| 6 | -107826.7602 | -107629.7147 | -107728.2375 | 0.015625 | 0.171875 | 0.09375 |
| 7 | -107750.2025 | -107425.1448 | -107587.6736 | 0.1484375 | 0.2109375 | 0.1796875 |

| 5A validation walker | first-half mean logL | second-half mean logL | primary-window mean logL | first-half candidate fraction | second-half candidate fraction | primary candidate fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -107506.9627 | -107280.7773 | -107393.87 | 0.18359375 | 0.21484375 | 0.19921875 |
| 1 | -107543.0695 | -107316.4121 | -107429.7408 | 0.15234375 | 0.1953125 | 0.173828125 |
| 2 | -107492.4996 | -107478.0745 | -107485.2871 | 0.1875 | 0.14453125 | 0.166015625 |
| 3 | -107447.7752 | -107364.9998 | -107406.3875 | 0.1953125 | 0.19140625 | 0.193359375 |
| 4 | -107532.3922 | -107529.3234 | -107530.8578 | 0.171875 | 0.13671875 | 0.154296875 |
| 5 | -107508.1068 | -107393.6415 | -107450.8741 | 0.1796875 | 0.16796875 | 0.173828125 |
| 6 | -107529.2862 | -107500.948 | -107515.1171 | 0.15625 | 0.1640625 | 0.16015625 |
| 7 | -107509.5971 | -107425.9851 | -107467.7911 | 0.19140625 | 0.1640625 | 0.177734375 |

The following fractions all use the **same Stage-5B candidate**, avoiding comparison of different tail events:

| run | window | walker fraction min | max | SD |
| --- | --- | --- | --- | --- |
| 5B selection | first_half | 0 | 0.453125 | 0.1874505841 |
| 5B selection | second_half | 0 | 0.65625 | 0.2715014367 |
| 5B selection | retained | 0 | 0.5390625 | 0.2286068242 |
| 5B calibration | first_half | 0.015625 | 0.1484375 | 0.03999991281 |
| 5B calibration | second_half | 0.1171875 | 0.2890625 | 0.05298695299 |
| 5B calibration | retained | 0.07421875 | 0.1796875 | 0.03768497441 |
| 5A validation | first_half | 0.15234375 | 0.1953125 | 0.01589300018 |
| 5A validation | second_half | 0.13671875 | 0.21484375 | 0.0264471547 |
| 5A validation | full | 0.154296875 | 0.19921875 | 0.0154143162 |

For context only, Stage-5A’s different official diagnostic threshold is `-107587.9625651654`; its full-run fractions span 0.330078125–0.40625 and its conservative tail ESS is 1425.45546104352. At the tighter **Stage-5B candidate**, Stage-5A fractions span 0.154296875–0.19921875. Neither diagnostic threshold is a frozen level.

Selection’s retained mean-logL SD (432.04) and maximum KS (0.78516) distinguish it sharply from calibration (54.48; 0.15234) and Stage-5A (49.29; 0.09766). The selection first-to-second-half SD grows from 365.58 to 503.73. Calibration also drifts upward: its pooled candidate fraction rises from 0.068359375 to 0.19140625. Passing its numerical ESS gate is not evidence of stationary whole-catalogue sampling.

| saved bank comparison | pooled logL KS |
| --- | --- |
| selection/calibration | 0.318359375 |
| selection/stage5a | 0.2346191406 |
| calibration/stage5a | 0.1376953125 |

All 28 within-bank pairwise KS/geometry rows per window and all 64 cross-bank pairs per comparison are retained in diagnostics.json. Pooled distributions are dependent MCMC histories, so these KS values are not inferential significance tests.

**Exchange and provenance comparisons:** all-window rates use 384 sweeps for Stage-5B and 512 for Stage-5A; retained/half rows use the windows stated above.

| run | window | accepted / proposals | acceptance | directional arrivals | distinct migrated realizations | donor-count range |
| --- | --- | --- | --- | --- | --- | --- |
| 5B selection | full | 433 / 1536 | 0.2819010417 | 866 | 534 | [4, 7] |
| 5B selection | first_half | 155 / 512 | 0.302734375 | 310 | 189 | [1, 6] |
| 5B selection | second_half | 161 / 512 | 0.314453125 | 322 | 198 | [3, 7] |
| 5B selection | retained | 316 / 1024 | 0.30859375 | 632 | 386 | [4, 7] |
| 5B calibration | full | 648 / 1536 | 0.421875 | 1296 | 672 | [7, 7] |
| 5B calibration | first_half | 240 / 512 | 0.46875 | 480 | 226 | [7, 7] |
| 5B calibration | second_half | 192 / 512 | 0.375 | 384 | 224 | [7, 7] |
| 5B calibration | retained | 432 / 1024 | 0.421875 | 864 | 446 | [7, 7] |
| 5A validation | full | 1125 / 2048 | 0.5493164062 | 2250 | 1057 | [7, 7] |
| 5A validation | first_half | 573 / 1024 | 0.5595703125 | 1146 | 529 | [7, 7] |
| 5A validation | second_half | 552 / 1024 | 0.5390625 | 1104 | 532 | [7, 7] |

| 5B selection walker | full-run transfers received | full-run donor IDs | primary-window transfers | primary-window donor IDs |
| --- | --- | --- | --- | --- |
| 0 | 136 | [1, 2, 3, 4, 5, 7] | 103 | [1, 2, 3, 4, 5, 7] |
| 1 | 139 | [0, 2, 3, 4, 5, 6, 7] | 105 | [0, 2, 4, 5, 6, 7] |
| 2 | 147 | [0, 1, 3, 4, 5, 7] | 104 | [0, 1, 4, 5, 7] |
| 3 | 21 | [0, 1, 2, 5, 6, 7] | 16 | [0, 5, 6, 7] |
| 4 | 145 | [0, 1, 2, 5, 7] | 107 | [0, 1, 2, 5, 7] |
| 5 | 132 | [0, 1, 2, 3, 4, 6, 7] | 90 | [0, 1, 2, 3, 4, 6, 7] |
| 6 | 18 | [1, 3, 5, 7] | 15 | [1, 3, 5, 7] |
| 7 | 128 | [0, 1, 2, 3, 4, 5, 6] | 92 | [0, 1, 2, 3, 4, 5, 6] |

| 5B calibration walker | full-run transfers received | full-run donor IDs | primary-window transfers | primary-window donor IDs |
| --- | --- | --- | --- | --- |
| 0 | 177 | [1, 2, 3, 4, 5, 6, 7] | 118 | [1, 2, 3, 4, 5, 6, 7] |
| 1 | 165 | [0, 2, 3, 4, 5, 6, 7] | 102 | [0, 2, 3, 4, 5, 6, 7] |
| 2 | 152 | [0, 1, 3, 4, 5, 6, 7] | 103 | [0, 1, 3, 4, 5, 6, 7] |
| 3 | 154 | [0, 1, 2, 4, 5, 6, 7] | 118 | [0, 1, 2, 4, 5, 6, 7] |
| 4 | 166 | [0, 1, 2, 3, 5, 6, 7] | 120 | [0, 1, 2, 3, 5, 6, 7] |
| 5 | 148 | [0, 1, 2, 3, 4, 6, 7] | 91 | [0, 1, 2, 3, 4, 6, 7] |
| 6 | 169 | [0, 1, 2, 3, 4, 5, 7] | 103 | [0, 1, 2, 3, 4, 5, 7] |
| 7 | 165 | [0, 1, 2, 3, 4, 5, 6] | 109 | [0, 1, 2, 3, 4, 5, 6] |

| 5A validation walker | full-run transfers received | full-run donor IDs | primary-window transfers | primary-window donor IDs |
| --- | --- | --- | --- | --- |
| 0 | 275 | [1, 2, 3, 4, 5, 6, 7] | 275 | [1, 2, 3, 4, 5, 6, 7] |
| 1 | 282 | [0, 2, 3, 4, 5, 6, 7] | 282 | [0, 2, 3, 4, 5, 6, 7] |
| 2 | 284 | [0, 1, 3, 4, 5, 6, 7] | 284 | [0, 1, 3, 4, 5, 6, 7] |
| 3 | 288 | [0, 1, 2, 4, 5, 6, 7] | 288 | [0, 1, 2, 4, 5, 6, 7] |
| 4 | 281 | [0, 1, 2, 3, 5, 6, 7] | 281 | [0, 1, 2, 3, 5, 6, 7] |
| 5 | 288 | [0, 1, 2, 3, 4, 6, 7] | 288 | [0, 1, 2, 3, 4, 6, 7] |
| 6 | 279 | [0, 1, 2, 3, 4, 5, 7] | 279 | [0, 1, 2, 3, 4, 5, 7] |
| 7 | 273 | [0, 1, 2, 3, 4, 5, 6] | 273 | [0, 1, 2, 3, 4, 5, 6] |

**Physical geometry:** unchanged Stage-4I source matching; frequency-set RMS and sorted-frequency mean distance in Fourier bins (DF=1/31536000 Hz), six-coordinate Hungarian RMS normalized by prior widths, psi period pi and lam period 2pi. Frequencies use all cross-time pairs; six-coordinate distances use every 32nd endpoint and all cross-pairs of those endpoints. No geometry threshold is introduced.

| run | window | median frequency-set RMS | median sorted-frequency mean distance | median six-coordinate RMS |
| --- | --- | --- | --- | --- |
| 5B selection | first_half | 93.28725086 | 54.58653485 | 0.2548292899 |
| 5B selection | second_half | 94.79763027 | 54.92694059 | 0.2517703768 |
| 5B selection | retained | 94.26622186 | 33.11483555 | 0.2515979041 |
| 5B calibration | first_half | 106.339669 | 61.06530844 | 0.2540505685 |
| 5B calibration | second_half | 103.1961341 | 78.17498806 | 0.2584117904 |
| 5B calibration | retained | 105.3386093 | 50.1829384 | 0.2564770172 |
| 5A validation | first_half | 100.7049204 | 45.90095994 | 0.2589726914 |
| 5A validation | second_half | 99.21436365 | 57.74749894 | 0.2538485427 |
| 5A validation | full | 98.69434335 | 38.72077218 | 0.2587586273 |

The implemented frequency interval differs slightly from nominal configuration limits. The diagnostic recovered the affine logistic-to-frequency map from saved selected-block coordinates and recorded frequencies, checking every recorded selection to 1e-18 Hz. Recovered bounds are `[0.0018300038051750386, 0.001850012683916793]`. Recomputed bank geometry agrees with saved Stage-5B geometry within 1e-8; no likelihood, decoder, or physical model call was used. Similar aggregate six-coordinate RMS across all three runs does not erase the much larger likelihood separation in selection.

## C. Candidate sensitivity, without replacement of the official candidate

Every diagnostic uses the unchanged strict empirical rule, ascending order index floor(N*(1-exp(-1))), with no interpolation. Official full selection has N=2048, index=1294. Leave-one-out has N=1792, index=1132; each half has N=1024, index=647. None of the values below enters a transition, mass, or checkpoint.

| omitted selection walker | diagnostic candidate | shift from official |
| --- | --- | --- |
| 0 | -107168.36944351123 | -44.70907735 |
| 1 | -107173.03168951336 | -49.37132335 |
| 2 | -107170.20991274703 | -46.54954659 |
| 3 | -107026.34902706592 | 97.31133909 |
| 4 | -107153.36948472615 | -29.70911857 |
| 5 | -107144.19324936366 | -20.5328832 |
| 6 | -107026.34902706592 | 97.31133909 |
| 7 | -107151.90497518318 | -28.24460902 |

| selection subset | diagnostic candidate | shift from official |
| --- | --- | --- |
| first_half | -107244.31647449388 | -120.6561083 |
| second_half | -106957.97870893264 | 165.6816572 |

Omitting either low lineage 3 or 6 raises the threshold by 97.3113; omitting another lineage lowers it by 20.5329–49.3713. Thus the upper tail is supplied almost entirely by six lineages, while the two low lineages pull the pooled order statistic downward. Half-only candidates differ by 286.3378. This is meaningful composition and temporal sensitivity, but there is no reference sampling distribution here that licenses calling it statistically “unusual” or selecting a substitute threshold.

## D. Implementation and repository history

| stage / date | commit | behavior and evidence |
| --- | --- | --- |
| Stage 1 / 2026-09-11 | d9c2e27 | Frozen DNS initialization requires valid support, not stationary or same-ancestor states; [blackjax/ns/dns.py](/r5/home/rruiz/projects/sbi/blackjax-dns/blackjax/ns/dns.py:144) |
| Stage 2 / 2026-09-11 | 0bf4898 | Constrained-prior transition kernel; no level-promotion ancestry requirement |
| Stage 3 / 2026-09-11 | 2f8cb04cdd2a3ea5281bbf3cff6b8fed41454385 | _survivors flattens time and walkers within one bank; strict eligible indices; jax.random.choice(..., replace=True); errors only for an empty eligible pool; [blackjax/ns/dns_levels.py](/r5/home/rruiz/projects/sbi/blackjax-dns/blackjax/ns/dns_levels.py:156) |
| LISA direct-prior bootstrap / 2026-09-21 | 9104fe5257e9dbf951740a036ca0af9d04ab2fd8 | survivor_indices selects eight distinct eligible bank indices without replacement; no preceding walker ancestry; [examples/lisa_dns_stage4/prior_bootstrap.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/prior_bootstrap.py:76) |
| LISA Stage 4F / 2026-09-22 | 81666f1f3698f526a9b50e20bfcb2d8099922469 | Introduces promote(position, logL, threshold, seed), one random survivor per original walker; raises if any row has none; [examples/lisa_dns_stage4/run_ladder_extension.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/run_ladder_extension.py:33) |
| LISA Stage 4G / 2026-09-22 | 58b0f139ccc51cc50dbdf4a726958aa21948160e | promoted_states calls promote independently for each bank; checkpoint metadata records own-walker rule; [examples/lisa_dns_stage4/run_ladder_batch.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/run_ladder_batch.py:84) |
| Stage 5B | bf46fe8 | freeze_level13 calls promoted_states before save_checkpoint; missing selection survivor raises before any level-13 file is written; [examples/lisa_dns_stage4/population_level13_construction.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/population_level13_construction.py:82) |

Generic Stage 3 uses this pooled resampling in both `construct_levels` and `calibrate_level_masses`. The first operates on two separately initialized banks and invokes _survivors separately for each bank; the second operates on its own independent calibration history. Exact positions and cached prior/likelihood values are selected together as a PyTree. Fresh burn-in follows promotion. No deduplication or donor balancing is performed. The [docs/examples/dns_stage3.md](/r5/home/rruiz/projects/sbi/blackjax-dns/docs/examples/dns_stage3.md:11) algorithm explicitly documents own-bank resampling and warns about survivor genealogy and finite mixing. The analytic-ladder, independent mass-refinement, and multimodal toy tests exercise this generic construction, including [tests/ns/test_dns_levels.py](/r5/home/rruiz/projects/sbi/blackjax-dns/tests/ns/test_dns_levels.py) and [tests/ns/test_dns_adaptive_multimodal.py](/r5/home/rruiz/projects/sbi/blackjax-dns/tests/ns/test_dns_adaptive_multimodal.py). Those toy passes are not validation of pooled LISA population exchange.

The Stage-4F docstring states: “One survivor per original walker preserves separate walker genealogies.” The introduction commit, [docs/examples/dns_stage4_ladder_extension.md](/r5/home/rruiz/projects/sbi/blackjax-dns/docs/examples/dns_stage4_ladder_extension.md:5), and [tests/experimental/test_dns_gb_ladder_extension.py](/r5/home/rruiz/projects/sbi/blackjax-dns/tests/experimental/test_dns_gb_ladder_extension.py:17) explicitly preserve row identity and reject a missing row survivor. They document and test an **algorithmic design choice with a conservative genealogy-preservation rationale**, not a theorem. No assertion that same-walker ancestry is mathematically necessary was found in the inspected source, tests, documentation, introduction commits, or repository-history search for mathematical promotion requirements. It is not classified as mere naming or convenience; the stated genealogy rationale is evidence.

The related ordinary NS adapter [blackjax/ns/from_mcmc.py](/r5/home/rruiz/projects/sbi/blackjax-dns/blackjax/ns/from_mcmc.py:53) also selects surviving particles with replacement. It is not the DNS construction algorithm and its no-survivor fallback must not be imported into this proposal. Only generic Stage-3 DNS supplies the relevant empty-pool STOP behavior.

A second independent LISA restriction matters for any future protocol: [examples/lisa_dns_stage4/run_ladder_batch.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/run_ladder_batch.py:31) currently requires eight distinct positions within each checkpoint bank and no shared positions across banks. With-replacement promotion can violate the within-bank distinctness check. A separately validated protocol would need explicit versioned promotion/checkpoint semantics that allow within-bank duplicates while retaining bank-origin separation, valid support and exact caches. Stage 5C changes none of these rules.

## E. Mathematics: support, invariance, stationarity and mixing are different

For a fixed threshold, write π_new(du)=Z_new⁻¹ π_impl(u) 1{logL(u)>ell_new} du. The population target for eight walkers is Π_new=π_new⊗8. Each valid single-walker update preserves π_new; the exact reciprocal MH exchanges preserve Π_new; composition in the fixed pairing schedule preserves Π_new. This invariance statement concerns the transition and target—not genealogical labels on its initial states.

An invariant kernel and its starting distribution are distinct mathematical objects. Convergence from a nonstationary start additionally needs appropriate irreducibility/recurrence and aperiodicity assumptions; invariance alone gives no finite burn-in guarantee. This standard distinction is stated in [Tierney (1994), Sections 2.1 and 3](https://www.stat.rice.edu/~dcox/Stat552/Mcmc/tierneyMCMCAnnStat94.pdf). Applying it here: no term in Π_new K=Π_new requires that slot w inherit a state from old slot w.

Every promoted state must be in the implemented prior support, have finite valid coordinates/caches, and satisfy the strict new likelihood constraint. That supplies admissible initialization. It does **not** imply that its initial law μ equals Π_new or that μK^128 equals Π_new. Exact stationary sampling from the first transition would require the joint initial distribution to be Π_new, a much stronger condition than support membership or same-walker ancestry. Sufficient support validity therefore answers the initialization question, not the finite-sample equilibrium question. For this interacting population, an eightfold clone of one random π_new state even has the correct individual marginals but the wrong joint product law.

| alternative / issue | invariance and initialization | finite-run implication |
| --- | --- | --- |
| Pooled unique survivors | Any selected valid state supplies a supported start; no ancestry theorem forbids it | Deduplicating a trajectory changes empirical residence weights when exact states repeat; no-replacement choices are dependent |
| Pooled entries with replacement | Samples the empirical conditional history within a bank; preserves eligibility and leaves the subsequent kernel unchanged | Shared donors/states introduce genealogy and correlation; the empirical history may be biased or nonstationary |
| Duplicate starts | Coincident positions are valid in product support; the slice and exchange formulas remain defined with finite caches | Independent fresh random streams can separate copies, but separation speed and effective diversity are not guaranteed; copies are not independent observations |
| Donor diversity | No specified donor count is required by invariance | More labels need not mean more physical modes; forced balancing changes empirical donor weights and can overweight one rare transient survivor |
| Existing burn-in | Applies the same invariant kernel to a nonstationary supported initial law | 128 sweeps is a fixed budget, not a theorem of convergence; no increased burn-in is proposed or run here |
| Same-walker rule | Neither necessary nor sufficient for stationary initialization | Protects lineage representation and exposes missing-survivor rows, but can reject an otherwise nonempty admissible support pool |

Even if a saved pool were drawn at equilibrium on the old contour, its finite empirical conditional distribution would only approximate the new constrained target. Here the threshold is adaptively selected and the histories are correlated and visibly heterogeneous. Pooling cannot make the failed compression estimate unbiased, recover missing catalogue families, or establish whole-catalogue physical convergence. It is a principled initialization design to test, not a proof that this LISA run is already correct statistically.

## F. Counterfactual support feasibility — no populations instantiated

Only Stage-5B **selection retained survivors** enter this calculation. The 753 eligible entries are 753 distinct saved coordinate states, supplied by seven donor walkers; counts by donor are `[134,138,131,1,122,109,0,118]`. No calibration or burn-in state is included. No eight-state array was drawn, instantiated, saved, promoted or run.

| hypothetical rule | eligible retained entries | eligible donor walkers | can form 8 supported starts? | duplicate states necessary? | donor duplication |
| --- | --- | --- | --- | --- | --- |
| A: same walker | 753 | 7 | No: walker 6 has none | not applicable | one per walker required |
| B: pooled unique | 753 | 7 | Yes | No | at least one donor repeated |
| C: donor-balanced unique | 753 | 7 | Yes; all 7 donors can be represented | No | one of the 6 donors with >=2 states supplies an extra |
| D: pooled with replacement | 753 | 7 | Yes | No, but allowed | repeated donors allowed; at least one unavoidable |

An unspecified pool does not determine a unique starting-distance value. Therefore the completed calculation reports the **analytic distribution of an unordered pair of start slots**, averaged over all possible allocations under stated hypothetical laws—not distances from a chosen ensemble. B is uniform among eight distinct unique states. C includes every eligible donor once, chooses the extra donor uniformly among the six having at least two states, then chooses distinct states uniformly within donor. D draws independently and uniformly from the 753 eligible retained entries. These conventions define read-only comparison weights only; no promotion algorithm is implemented.

| rule | pairwise distance | minimum | median | maximum | mean | root mean square |
| --- | --- | --- | --- | --- | --- | --- |
| B | frequency_set_RMS_bins | 0.0004590132818 | 77.22900203 | 255.3699683 | 81.65798852 | 87.62059308 |
| B | six_coordinate_source_set_RMS | 8.949964599e-07 | 0.2473335224 | 0.3232019626 | 0.2447514559 | 0.2464562807 |
| C | frequency_set_RMS_bins | 0.0004590132818 | 78.08336726 | 255.3699683 | 82.00640621 | 87.03902285 |
| C | six_coordinate_source_set_RMS | 8.949964599e-07 | 0.2520449087 | 0.3232019626 | 0.2520238333 | 0.2530538313 |
| D | frequency_set_RMS_bins | 0 | 77.18474452 | 255.3699683 | 81.54954497 | 87.56239275 |
| D | six_coordinate_source_set_RMS | 0 | 0.2473031784 | 0.3232019626 | 0.2444264207 | 0.2462925767 |

All 283128 distinct unordered pairs were evaluated with the established frequency and six-coordinate matching metrics. For D, diagonal zero-distance probability is 1/753; nonidentical pairs have probability 2/753². Probability of any repeated state among eight independent draws is 0.03662127235067014. Repeats are therefore possible but not needed. Candidate C forces the lone walker-3 survivor into every hypothetical population, whereas D gives that donor its empirical weight 1/753. The similar distance summaries do not demonstrate that any method mixes or justify choosing a method by these numbers alone.

## G. Bank independence must remain explicit

Future selection promotion may pool **SELECTION survivors only**. Future calibration promotion may pool **CALIBRATION survivors only**. Calibration may never rescue selection. Whole states and caches must remain bank-local, in separate mutable storage, with independent bank promotion streams and independent subsequent slice, label-selection and joint-MH streams. Within-bank donor reuse is allowed by the candidate protocol; cross-bank reuse is prohibited.

The threshold must still be determined only by selection and saved before its independent calibration assessment. Calibration results must not choose selection donors, revise the threshold, choose favorable seeds, or trigger retries. Separate data origins and streams are the required operational independence; shared adaptive thresholds, finite histories and acceptance conditioning do not turn either bank into IID equilibrium replications. Within-bank clones in particular must not be counted as independent evidence. No bank was pooled or rescued during this audit.

## H. Selection walker 6: what the saved evidence supports

| run / walker | start logL | primary mean logL | candidate fraction | median KS to peers | max KS | full exchanges | full donors | median frequency RMS to peers | median full RMS to peers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5B selection / 6 | -108051.7897 | -108042.6198 | 0 | 0.73828125 | 0.78515625 | 18/384 | [1, 3, 5, 7] | 99.07108235 | 0.2578433259 |
| 5B selection / 3 | -108138.5725 | -107877.1464 | 0.00390625 | 0.59375 | 0.6328125 | 21/384 | [0, 1, 2, 5, 6, 7] | 110.3000626 | 0.2519536567 |
| 5B calibration / 6 | -107714.8286 | -107728.2375 | 0.09375 | 0.08984375 | 0.10546875 | 169/384 | [0, 1, 2, 3, 4, 5, 7] | 99.05907212 | 0.254964019 |
| 5A validation / 6 | -107714.8286 | -107515.1171 | 0.16015625 | 0.064453125 | 0.09765625 | 279/512 | [0, 1, 2, 3, 4, 5, 7] | 114.7761236 | 0.2634106822 |

| selection pair | retained KS | frequency-set RMS | sorted-frequency mean distance | six-coordinate RMS |
| --- | --- | --- | --- | --- |
| 0,3 | 0.6328125 | 107.7140978 | 47.92699904 | 0.2650392314 |
| 0,6 | 0.78515625 | 91.81002622 | 9.060670311 | 0.2469239355 |
| 1,3 | 0.6171875 | 103.4902722 | 25.52390424 | 0.2821405653 |
| 1,6 | 0.75390625 | 103.2936495 | 39.61257031 | 0.2642709262 |
| 2,3 | 0.62890625 | 112.6042527 | 53.76156042 | 0.251669082 |
| 2,6 | 0.75390625 | 99.07108235 | 31.36895709 | 0.2589239666 |
| 3,4 | 0.59375 | 103.537197 | 32.96634085 | 0.2519536567 |
| 3,5 | 0.57421875 | 110.3000626 | 56.18470613 | 0.2510337906 |
| 3,6 | 0.36328125 | 118.5596022 | 44.71848885 | 0.2510547186 |
| 3,7 | 0.57421875 | 121.0397089 | 54.60688388 | 0.2572786185 |
| 4,6 | 0.73828125 | 96.42192893 | 24.94717015 | 0.2578433259 |
| 5,6 | 0.71875 | 92.11051479 | 20.81607054 | 0.2416300661 |
| 6,7 | 0.72265625 | 105.2166208 | 18.97847209 | 0.2609034599 |

| selection group / walker | first-half mean logL | second-half mean logL | first-half candidate fraction | second-half candidate fraction |
| --- | --- | --- | --- | --- |
| 3 | -107908.6325 | -107845.6603 | 0.0078125 | 0 |
| 6 | -108016.0083 | -108069.2314 | 0 | 0 |
| 0 | -107159.3771 | -106948.3431 | 0.4296875 | 0.6171875 |
| 1 | -107137.8314 | -106786.0987 | 0.421875 | 0.65625 |
| 2 | -107067.1484 | -106842.6385 | 0.453125 | 0.5703125 |
| 4 | -107213.5408 | -106806.1161 | 0.3671875 | 0.5859375 |
| 5 | -107344.0036 | -106914.6558 | 0.3359375 | 0.515625 |
| 7 | -107246.8332 | -107067.0467 | 0.3984375 | 0.5234375 |

Walker 6 starts lower than its calibration/Stage-5A counterpart, but selection walkers 1, 2 and 3 start lower still; initial logL alone does not explain its fate. Its first three 32-sweep burn-in blocks have candidate fractions 0.09375, 0.21875 and 0.28125; all later blocks are zero. Its retained half means worsen from -108016.0083 to -108069.2314. Selection walker 3 also stays low, with one first-half survivor and none in the second. The other six selection walkers have second-half fractions 0.515625–0.65625. This is persistent separation within this saved run, not demonstrated topological disconnection of constrained support.

By contrast, calibration walker 6 increases from 0.015625 to 0.171875; Stage-5A walker 6 has fractions 0.15625 then 0.1640625 at this same Stage-5B candidate. The latter two start from identical saved states but independent streams. Labels are bookkeeping slots, not identified physical source modes.

| run / walker | retained or primary proposals | joint survivors | accepted | own proposed logL passes ell12 | partner proposed logL passes ell12 | selected f median (Hz) | selected f IQR (Hz) | inside frozen selector interval | dominant selected label / count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5B selection / 6 | 256 | 15 | 15 | 190 | 78 | 0.001841173733 | [0.0018411029247019665, 0.0018411808679820601] | 0.93359375 | 8 / 222 |
| 5B selection / 3 | 256 | 16 | 16 | 206 | 62 | 0.001841106489 | [0.0018410993158986056, 0.0018411719990994223] | 0.91796875 | 7 / 217 |
| 5B calibration / 6 | 256 | 103 | 103 | 174 | 184 | 0.001841035854 | [0.0018409883135257412, 0.0018410539435078685] | 0.95703125 | 3 / 238 |
| 5A validation / 6 | 512 | 279 | 279 | 388 | 403 | 0.001841012068 | [0.0018409481520981817, 0.0018410433415208607] | 0.962890625 | 3 / 446 |

All four cases have selected-rank median 1 and IQR [1,1]. The selector therefore does focus near its frozen frequency region. Selection walker 6 has 190/256 own proposed contour passes and 78/256 partner passes, but only 15 jointly surviving proposals, all accepted. Walker 3 similarly has only 16 jointly surviving retained exchanges. Joint contour compatibility, rather than MH selection-ratio rejection among survivors, is the immediate bottleneck in their retained exchanges. This does not identify which physical source is wrong; complete block transfers and labels cannot establish source identity or causal likelihood contribution.

In the first retained half, walker 3 receives eight transfers, all from walker 6; walker 6 receives nine, from walkers 1 and 3. Thus much of their limited communication circulates within the low-tail pair. Over all 384 sweeps their received transfers are only 21 and 18, compared with 128–147 for the other six selection walkers. Calibration walker 6 receives 169, and Stage-5A walker 6 receives 279, both from all seven donors. Exact provenance proves those transfers occurred, not equilibrium.

Walker 6’s median frequency-set distance to other selection walkers (99.07 bins) and six-coordinate RMS (0.25784) are not extraordinary relative to the other populations. Aggregate permutation-invariant geometry therefore provides weaker discrimination than logL, tail residence and reciprocal communication. The matching metric is not a physical mode classifier.

**Diagnosis:** evidence favors insufficient finite-time mixing in a poor selection-origin ensemble. Inherited catalogue quality is plausible, supported by the broader starting ensemble and the separate bank history, but start quality and stream effects are confounded. Walker 6’s earlier burn-in crossings rule out a simple claim that its lineage could never reach the candidate region. Persistent constrained-region separation is observed over the retained window; permanent disconnected support or a specific source mismatch remains unproven. This is evidence against uniform finite-budget robustness, not against the exact invariance of the population kernel.

## I. Exactly one future promotion design

Recommend **generic Stage-3-style within-bank pooled resampling with replacement**: after selecting a threshold and applying the unchanged statistical/integrity decision, form each bank’s pool from all of its own retained states strictly above the threshold; uniformly sample the required promoted walker slots with replacement using independent bank streams; copy complete states and caches into separate arrays; then apply the existing fixed burn-in with fresh independent walker streams. An empty bank pool remains a STOP condition. No donor balancing, deduplication, calibration-to-selection rescue, favorable seed selection, or manual walker exception is part of this design.

This choice is based on the target mathematics, existing generic implementation and preservation of empirical residence weights—not a small difference in distance summaries. It requires separate validation and explicit handling of allowable within-bank duplicates in the versioned checkpoint contract. The current same-walker promotion function, current checkpoint validator, generic DNS code and all Stage-5B artifacts remain unchanged. Stage 5C implements only diagnostics and tests, not this design.

## J. End of the manual methodology-development sequence

Pooled within-bank promotion is principled, so the recommended follow-up permission is bounded to exactly: **(1) one cheap predeclared toy validation; (2) one controlled predeclared LISA validation; (3) freeze the algorithm if those validations pass.** Neither validation is run in Stage 5C. The toy must check support/cache copying, bank isolation, duplicate starts and known constrained-distribution behavior; the controlled LISA validation must assess finite-run agreement and communication without treating the failed Stage-5B candidate as a frozen level. Budgets, criteria and random streams must be fixed before either experiment.

After that there must be **no manual level-13 → level-14 → level-15 methodology-development loop**. If a validation fails, stop ladder development at level 12 rather than tune, retry, or extend this sequence. Passing the bounded validation path would freeze an algorithmic protocol, not retroactively freeze Stage-5B level 13 or establish evidence reconstruction. Production and evidence remain out of scope.

## K. Read-only verification and tests

The complete saved level-0..12 prefix and copied checkpoint checksum were verified against Stage-5B preflight, the stored original-checkpoint hash and authoritative endpoint. Both saved 384-sweep traces are complete; their state/provenance replays were recorded as passed before the failed promotion. Candidate hash is unchanged. Every Stage-5B and compared Stage-5A artifact hash matches the pre-interruption Stage-5C input manifest. No checkpoint_level13.json exists in Stage-5B or Stage-5C. The original level-12 checkpoint also retains its recorded SHA256.

**11 cheap read-only tests pass.** They cover strict survivor counts and runs, zero/all survivors, burn-in exclusion, half splitting, leave-one-walker-out order statistics, support feasibility, output-only diagnostics with historical-directory guards, absence of sampling/model/promotion imports and calls, saved affine physical mapping, and analytic pair-distance weights without allocating a starting population. Existing tests are preserved unchanged. No existing model-based tests were run for this audit. No new likelihood, sampler, or promotion call occurred.

Artifacts: [completed diagnostics JSON](/tmp/lisa_dns_stage5c_readonly_diagnosis/diagnostics.json), [read-only validation manifest](/tmp/lisa_dns_stage5c_readonly_diagnosis/validation.json), and [final cheap-test log](/tmp/stage5c_final_tests.log). The diagnostic code is [examples/lisa_dns_stage4/diagnose_population_level13.py](/r5/home/rruiz/projects/sbi/blackjax-dns/examples/lisa_dns_stage4/diagnose_population_level13.py) and the tests are [tests/experimental/test_dns_gb_readonly_promotion_diagnosis.py](/r5/home/rruiz/projects/sbi/blackjax-dns/tests/experimental/test_dns_gb_readonly_promotion_diagnosis.py). All initial-state distance summaries are analytic diagnostics; there is no saved promoted population.

## L. Five scientific answers

1. **Was Stage-5B primarily evidence of failure of the population kernel?** No general kernel failure is established. It exposes a poor finite selection ensemble, limited finite-time communication, and a conservative promotion bottleneck; the same kernel communicates substantially better in calibration and Stage-5A. The poor selection trajectory remains a real mixing warning.

2. **Is same-walker promotion mathematically required for DNS correctness?** No. It is a later LISA genealogy-preservation design choice. Support-valid initialization and an invariant transition are required; same-walker ancestry neither proves stationarity nor is required by invariance.

3. **Is pooled within-bank survivor promotion mathematically principled?** Yes, as supported initialization followed by the unchanged invariant kernel. It does not guarantee finite-time equilibrium or unbiased finite-run mass estimates, and it does not reverse Stage-5B’s failure.

4. **Which single promotion rule should be validated next?** Uniform sampling with replacement from each bank’s own eligible retained-state pool, independently across banks, followed by the existing fixed burn-in.

5. **Does the manual level-by-level methodology-development loop stop after this audit?** Yes. Permit only one cheap toy validation and one controlled LISA validation of that single protocol, then freeze the algorithm if they pass; otherwise stop at level 12. No repeated level-by-level manual development loop. STOP.
