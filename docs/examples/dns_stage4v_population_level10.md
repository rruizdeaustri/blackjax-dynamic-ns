# Stage 4V: one population level-10 construction/calibration attempt

**Yes.** The single attempt passed the unchanged independent calibration gate: ESS = 52.27700333532884, required ≥20. Candidate ell10 = -109780.28875081123.

Exactly one new level is frozen: `ell_10 = -109780.28875081123`, `log X_10 = -8.394305725653265`. The mass increment is the log of the independent calibration compression; selection compression does not set the mass.

## Frozen setup and historical recovery

The complete ten-level prefix was loaded from the Stage-4G checkpoint and verified byte for byte. The lower checkpoint was copied unchanged and its SHA256 remains `c476746952ee6660f6adf8d634dfb639d5a79dc643431b3ba008b1da62e5e319`. No lower level was reconstructed.

| level | ell | log X |
| --- | --- | --- |
| 0 | -inf | 0.0 |
| 1 | -113725.65889538352 | -1.0343179379627125 |
| 2 | -113448.99632193986 | -1.9996430044384734 |
| 3 | -113172.84824147604 | -3.118477930469643 |
| 4 | -112838.36560726895 | -4.172213954289457 |
| 5 | -112388.10901787043 | -5.0146408844420645 |
| 6 | -111915.6904902109 | -5.840204008542667 |
| 7 | -111344.96574156903 | -6.561075737117467 |
| 8 | -110809.82135735315 | -7.156607159127204 |
| 9 | -110252.99476697217 | -7.67544001580533 |

Promotion machinery replay recovered selection indices `[133, 209, 107, 226, 248, 45, 58, 181]` and calibration indices `[168, 137, 95, 193, 219, 173, 144, 142]`. Every position, cached prior and cached likelihood matched both the historical checkpoint and Stage-4P initial NPZs exactly. Banks are disjoint and have separate walker, label and MH random streams.

Each bank used exactly 8 walkers, 128 burn-in population sweeps and 256 retained population sweeps. Block size 32, target compression exp(-1), min_ess 20. One sweep applies the validated isotropic slice to every walker followed by four disjoint exact joint exchanges in the frozen seven-round schedule. The source-equivalence test verifies that the Stage-4U run body changes only output location, slice seeds and sweep count.

The selector is the saved Stage-4T/4U design: f_star = 0.0018409808 Hz, sigma_f = 2.558e-7 Hz, log weights -0.5*((f0-f_star)/sigma_f)^2 with logsumexp normalization, no clipping. The full implemented prior ratio and recomputed reverse/forward selection ratio enter one joint MH decision; both proposed walkers must exceed ell9. No source_gain is evaluated by the transition. Slice settings, source-selection formula, exchange frequency and pairing are unchanged.

Frozen slice seeds: selection `[890100, 890101, 890102, 890103, 890104, 890105, 890106, 890107]`; calibration `[890200, 890201, 890202, 890203, 890204, 890205, 890206, 890207]`. Independent PCG64 label/MH seeds: `{'selection': [2026092801, 2026092802], 'calibration': [2026092803, 2026092804]}`. All draws and both designs were saved before sampling. The candidate was saved after selection and before calibration sampling; neither historical failed threshold was reused as a target.

## Three-attempt comparison

| diagnostic | original isotropic | Hybrid P | Stage-4U population |
| --- | --- | --- | --- |
| candidate threshold | -109401.11425363769 | -109301.45870884096 | -109780.28875081123 |
| selection compression | 0.3676757812 | 0.3676757812 | 0.3676757812 |
| selection ESS | 24.27235681 | 25.32714806 | 32.20894459 |
| calibration compression | 0.3642578125 | 0.3525390625 | 0.4873046875 |
| calibration ESS | 16.79821519 | 18.27464305 | 52.27700334 |
| selection IID / Bernoulli SE | 0.01065460721 | 0.01065460721 | 0.01065460721 |
| selection block-means SE | 0.03991955486 | 0.0385553775 | 0.03537767973 |
| selection between-walker SE | 0.09786929189 | 0.0958096548 | 0.08495993527 |
| selection walker exceedance range | 0 – 0.7421875 | 0.01171875 – 0.72265625 | 0.05078125 – 0.66796875 |
| selection walker exceedance SD | 0.2768161599 | 0.2709906265 | 0.2403029854 |
| selection walker logL-mean SD | 687.8969865 | 754.4955032 | 275.8816872 |
| selection walker logL-mean MAD | 581.8831542 | 481.0933223 | 218.5796885 |
| selection walker logL-mean IQR | 1037.219862 | 840.9466341 | 486.8528254 |
| calibration IID / Bernoulli SE | 0.01063359162 | 0.01055711937 | 0.01104498147 |
| calibration block-means SE | 0.04444675061 | 0.04157628748 | 0.02871102186 |
| calibration between-walker SE | 0.1174122426 | 0.1117598878 | 0.06913128436 |
| calibration walker exceedance range | 0.01171875 – 0.81640625 | 0.01953125 – 0.7734375 | 0.01953125 – 0.6171875 |
| calibration walker exceedance SD | 0.3320919719 | 0.3161046981 | 0.1955327999 |
| calibration walker logL-mean SD | 622.7920421 | 641.9882211 | 311.4313296 |
| calibration walker logL-mean MAD | 591.1017867 | 451.8136653 | 71.97596207 |
| calibration walker logL-mean IQR | 1111.490913 | 941.6427825 | 126.0180293 |

Stage-4V selection per-walker retained summaries:

| walker | fraction logL > candidate | mean logL |
| --- | --- | --- |
| 0 | 0.609375 | -109390.5171 |
| 1 | 0.37890625 | -109682.0345 |
| 2 | 0.66796875 | -109424.9088 |
| 3 | 0.51171875 | -109552.0438 |
| 4 | 0.48046875 | -109570.5507 |
| 5 | 0.12109375 | -110011.1596 |
| 6 | 0.05078125 | -110065.1593 |
| 7 | 0.12109375 | -110005.7639 |

Stage-4V calibration per-walker retained summaries:

| walker | fraction logL > candidate | mean logL |
| --- | --- | --- |
| 0 | 0.59765625 | -109152.8072 |
| 1 | 0.53125 | -109356.0961 |
| 2 | 0.6171875 | -109255.7469 |
| 3 | 0.52734375 | -109255.2776 |
| 4 | 0.53125 | -109341.6083 |
| 5 | 0.60546875 | -109183.5944 |
| 6 | 0.01953125 | -110125.452 |
| 7 | 0.46875 | -109385.2109 |

All tail ESS and standard errors use the existing machinery unchanged. These are interacting walkers, so agreement does not make them independent convergence replications. The gate is the historically specified operational criterion; broad physical convergence and evidence accuracy are not established by passing it.

Calibration between-walker SE changed from 0.11741224 (isotropic) and 0.11175989 (Hybrid P) to 0.069131284. Walker-fraction SD changed from 0.33209197 and 0.3161047 to 0.1955328; walker-logL-mean SD changed from 622.79204 and 641.98822 to 311.43133. These comparisons address the prior between-walker blocker directly, rather than relying only on ESS crossing 20.

Between-walker calibration uncertainty was **materially reduced relative to both failures**: the SE fell by 41.1% versus isotropic and 38.1% versus Hybrid P; logL-mean SD fell by about 50%. Residual disagreement remains substantial. Calibration walker 6 had exceedance fraction 0.01953125 and mean logL -110125.45204687324, versus fractions 0.46875–0.6171875 for the other seven walkers. Its 18 accepted pair exchanges over 384 opportunities show that some communication occurred without resolving its likelihood discrepancy. Selection walkers 5–7 also remain low. Thus the stronger agreement of the earlier Stage-4U benchmark is not fully reproduced from these historical construction starts.

The new selection threshold is lower than both failed candidates, and independent calibration compression 0.4873046875 differs from selection compression 0.36767578125. Comparisons of exceedance uncertainty therefore use each attempt's own selected threshold; the unthresholded logL-mean SD/MAD/IQR provide additional evidence of improvement. The existing gate passes comfortably, but this is not a claim that the whole population has converged physically.

## Exchange diagnostics and physical communication

| bank | proposals | joint survivors | survival fraction | joint accepted | acceptance fraction | MH rejections among survivors | MH rejection fraction | nonfinite priors | nonfinite logL | nonfinite ratios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 1536 | 278 | 0.1809895833 | 255 | 0.166015625 | 23 | 0.08273381295 | 0 | 0 | 0 |
| calibration | 1536 | 635 | 0.4134114583 | 600 | 0.390625 | 35 | 0.05511811024 | 0 | 0 | 0 |

| bank | distribution | min | 25% | median | 75% | max | nonfinite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selection | selection_log_ratio | -4.718716625 | -0.09082916772 | 6.30392895e-12 | 0.0664232476 | 3.493242909 | 0 |
| selection | joint_prior_log_ratio | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.421085472e-13 | 0 |
| calibration | selection_log_ratio | -1.437236636 | -0.05885567585 | 0 | 0.05696126752 | 3.591062749 | 0 |
| calibration | joint_prior_log_ratio | -1.421085472e-13 | -2.842170943e-14 | 0 | 2.842170943e-14 | 1.98951966e-13 | 0 |

| walker pair | selection accepted / proposed | selection fraction | calibration accepted / proposed | calibration fraction |
| --- | --- | --- | --- | --- |
| 0,1 | 24 / 54 | 0.4444444444 | 21 / 54 | 0.3888888889 |
| 0,2 | 17 / 55 | 0.3090909091 | 27 / 55 | 0.4909090909 |
| 0,3 | 19 / 55 | 0.3454545455 | 21 / 55 | 0.3818181818 |
| 0,4 | 15 / 55 | 0.2727272727 | 31 / 55 | 0.5636363636 |
| 0,5 | 6 / 55 | 0.1090909091 | 29 / 55 | 0.5272727273 |
| 0,6 | 3 / 55 | 0.05454545455 | 1 / 55 | 0.01818181818 |
| 0,7 | 4 / 55 | 0.07272727273 | 29 / 55 | 0.5272727273 |
| 1,2 | 14 / 55 | 0.2545454545 | 29 / 55 | 0.5272727273 |
| 1,3 | 22 / 55 | 0.4 | 21 / 55 | 0.3818181818 |
| 1,4 | 24 / 55 | 0.4363636364 | 30 / 55 | 0.5454545455 |
| 1,5 | 8 / 55 | 0.1454545455 | 23 / 55 | 0.4181818182 |
| 1,6 | 1 / 55 | 0.01818181818 | 5 / 55 | 0.09090909091 |
| 1,7 | 7 / 55 | 0.1272727273 | 34 / 55 | 0.6181818182 |
| 2,3 | 13 / 55 | 0.2363636364 | 27 / 55 | 0.4909090909 |
| 2,4 | 17 / 55 | 0.3090909091 | 34 / 55 | 0.6181818182 |
| 2,5 | 4 / 55 | 0.07272727273 | 31 / 55 | 0.5636363636 |
| 2,6 | 1 / 55 | 0.01818181818 | 2 / 55 | 0.03636363636 |
| 2,7 | 3 / 54 | 0.05555555556 | 32 / 54 | 0.5925925926 |
| 3,4 | 15 / 55 | 0.2727272727 | 28 / 55 | 0.5090909091 |
| 3,5 | 5 / 55 | 0.09090909091 | 27 / 55 | 0.4909090909 |
| 3,6 | 4 / 54 | 0.07407407407 | 4 / 54 | 0.07407407407 |
| 3,7 | 2 / 55 | 0.03636363636 | 21 / 55 | 0.3818181818 |
| 4,5 | 6 / 54 | 0.1111111111 | 27 / 54 | 0.5 |
| 4,6 | 2 / 55 | 0.03636363636 | 2 / 55 | 0.03636363636 |
| 4,7 | 3 / 55 | 0.05454545455 | 35 / 55 | 0.6363636364 |
| 5,6 | 3 / 55 | 0.05454545455 | 4 / 55 | 0.07272727273 |
| 5,7 | 4 / 55 | 0.07272727273 | 25 / 55 | 0.4545454545 |
| 6,7 | 9 / 55 | 0.1636363636 | 0 / 55 | 0 |

Both banks executed 54 complete seven-round cycles and the first six rounds. Every pair had 54 or 55 predetermined opportunities; the schedule was not adapted.

**selection:** 510 accepted directional inter-walker transfers; 390 distinct exact realizations migrated; 500 arrivals were to walkers other than the realization's birth walker; 497 were first recipient visits. 1 transfers involved exact initial tokens, and 48/72 initial ancestries visited other walkers.

| recipient walker | distinct direct donor walkers | donor IDs | distinct foreign birth walkers |
| --- | --- | --- | --- |
| 0 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 |
| 1 | 7 | [0, 2, 3, 4, 5, 6, 7] | 6 |
| 2 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 |
| 3 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 |
| 4 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 |
| 5 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 |
| 6 | 7 | [0, 1, 2, 3, 4, 5, 7] | 7 |
| 7 | 7 | [0, 1, 2, 3, 4, 5, 6] | 7 |

Directional migration counts (rows = donor, columns = recipient):

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 24 | 17 | 19 | 15 | 6 | 3 | 4 |
| 1 | 24 | 0 | 14 | 22 | 24 | 8 | 1 | 7 |
| 2 | 17 | 14 | 0 | 13 | 17 | 4 | 1 | 3 |
| 3 | 19 | 22 | 13 | 0 | 15 | 5 | 4 | 2 |
| 4 | 15 | 24 | 17 | 15 | 0 | 6 | 2 | 3 |
| 5 | 6 | 8 | 4 | 5 | 6 | 0 | 3 | 4 |
| 6 | 3 | 1 | 1 | 4 | 2 | 3 | 0 | 9 |
| 7 | 4 | 7 | 3 | 2 | 3 | 4 | 9 | 0 |

Exact-realization migration-count histogram, including zero migrations: `{'0': 8833, '1': 303, '2': 64, '3': 15, '4': 7, '5': 0, '6': 1}`. Full token-by-token counts are in `comparison.json → population → selection → provenance → source_realization_migration_counts`; each transfer is in `selection/provenance_transfers.json`.

**calibration:** 1200 accepted directional inter-walker transfers; 663 distinct exact realizations migrated; 1151 arrivals were to walkers other than the realization's birth walker; 1104 were first recipient visits. 9 transfers involved exact initial tokens, and 48/72 initial ancestries visited other walkers.

| recipient walker | distinct direct donor walkers | donor IDs | distinct foreign birth walkers |
| --- | --- | --- | --- |
| 0 | 7 | [1, 2, 3, 4, 5, 6, 7] | 7 |
| 1 | 7 | [0, 2, 3, 4, 5, 6, 7] | 7 |
| 2 | 7 | [0, 1, 3, 4, 5, 6, 7] | 7 |
| 3 | 7 | [0, 1, 2, 4, 5, 6, 7] | 7 |
| 4 | 7 | [0, 1, 2, 3, 5, 6, 7] | 7 |
| 5 | 7 | [0, 1, 2, 3, 4, 6, 7] | 7 |
| 6 | 6 | [0, 1, 2, 3, 4, 5] | 6 |
| 7 | 6 | [0, 1, 2, 3, 4, 5] | 6 |

Directional migration counts (rows = donor, columns = recipient):

| donor / recipient | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 21 | 27 | 21 | 31 | 29 | 1 | 29 |
| 1 | 21 | 0 | 29 | 21 | 30 | 23 | 5 | 34 |
| 2 | 27 | 29 | 0 | 27 | 34 | 31 | 2 | 32 |
| 3 | 21 | 21 | 27 | 0 | 28 | 27 | 4 | 21 |
| 4 | 31 | 30 | 34 | 28 | 0 | 27 | 2 | 35 |
| 5 | 29 | 23 | 31 | 27 | 27 | 0 | 4 | 25 |
| 6 | 1 | 5 | 2 | 4 | 2 | 4 | 0 | 0 |
| 7 | 29 | 34 | 32 | 21 | 35 | 25 | 0 | 0 |

Exact-realization migration-count histogram, including zero migrations: `{'0': 8375, '1': 397, '2': 127, '3': 69, '4': 36, '5': 22, '6': 4, '7': 4, '8': 3, '9': 0, '10': 0, '11': 0, '12': 1}`. Full token-by-token counts are in `comparison.json → population → calibration → provenance → source_realization_migration_counts`; each transfer is in `calibration/provenance_transfers.json`.

Provenance follows Stage 4U exactly: ancestry persists through slice descendants, while a new exact-realization token is minted for every slice-targeted block, conservatively including roundoff-identical changes. Joint acceptance swaps both tokens; rejection preserves both. Saved operations were replayed independently with exact six-coordinate block checks. These are communication diagnostics only, not adaptive selection rules or claims of whole-catalogue convergence.

## Cost and integrity

| attempt | bank | slice likelihood proxy | exchange/refresh likelihood calls | slice prior proxy | exchange/refresh prior calls | direct cache likelihood checks incl starts | direct cache prior checks incl starts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original isotropic | selection | 21008 | 0 | 21008 | 0 | 3080 | 3080 |
| original isotropic | calibration | 20837 | 0 | 20837 | 0 | 3080 | 3080 |
| Hybrid P | selection | 21104 | 3072 | 21104 | 3072 | 6152 | 6152 |
| Hybrid P | calibration | 20797 | 3072 | 20797 | 3072 | 6152 | 6152 |
| Stage-4U population | selection | 21468 | 3072 | 21468 | 3072 | 6152 | 6152 |
| Stage-4U population | calibration | 20951 | 3072 | 20951 | 3072 | 6152 | 6152 |

The slice proxy retains the historical num_steps+num_shrink convention; scalar prior and likelihood are evaluated together. Exchange prior and likelihood calls and direct cache checks are counted separately. No cost-based truncation or optimization was used. Full original counter dictionaries are preserved in comparison.json.

selection: 384 sweeps, zero structural failures; maximum direct prior-cache error 5.684341886080802e-14, maximum logL-cache error 2.028536982834339e-08. Minimum selector log probability -919.2865907326775. CPU loop time 52.549 s (not a cross-hardware performance comparison).

calibration: 384 sweeps, zero structural failures; maximum direct prior-cache error 2.842170943040401e-14, maximum logL-cache error 9.851646609604359e-09. Minimum selector log probability -913.1972115562244. CPU loop time 51.859 s (not a cross-hardware performance comparison).

Every initial state and every constituent outcome passed finite-prior, finite-likelihood, strict ell9 and direct cache checks (prior atol 1e-9, likelihood atol 1e-7, rtol 0). Selected blocks were exact copies, untouched coordinates unchanged, rejection restored both post-slice states, and acceptance updated both jointly. Saved-draw replay reconstructed selector probabilities, full MH ratios, decisions, state caches and token histories before the freeze gate. Protected historical artifacts, prior/model/configuration, slice, selector and complete lower ladder retained their hashes.

**212 pre-run tests passed:** 147 experimental tests (all previous 137 plus 10 Stage-4V tests) and 65 generic DNS tests. New tests cover exact start recovery, frozen prefix, bank/stream independence, burn-in bookkeeping, selection/calibration separation, ESS boundary, no checkpoint on failure, exactly one checkpoint on success, no level-11 path, and source equivalence to the frozen Stage-4U body. No real LISA construction is in pytest.

Artifacts: `/tmp/lisa_dns_stage4v_population_level10/` contains preflight.json, both designs and random schedules, exact initial banks, all 384-sweep traces, exchange records, token births and transfer histories, selection_candidate.json, report.json, comparison.json, levels.npz, the unchanged checkpoint_level9.json, and checkpoint_level10.json.

Exactly one attempt was performed. No retry, level 11, population tuning, prior/slice change, production, seed44/66/88 scientific chain, or evidence reconstruction was performed. Evidence remains unvalidated and out of scope. Stop here.
