# Isotropic LISA ladder extension from Stage-4E

Both bounded extensions passed the unchanged ESS gate and were frozen. This validates shallow adaptive extension beyond level 2 under the declared budget; it does not establish global catalogue mixing or adequate ladder depth.

Fixed settings: two separate banks, eight walkers each, 128 burn-in and 256 retained steps per walker, block size 32, minimum ESS 20, target compression exp(-1), all 54 scales pi/sqrt(3), unchanged 20/40/40 proposal. Levels 0–2 and their IID/rejection calibration were consumed unchanged. Initialization used eight distinct unused IID survivors from each bank (seeds 5100/5101); previous retained calibration/selection points were excluded. Level-3 streams were 5200–5207 and 5300–5307. Level-4 starts used one survivor per original walker (promotion seeds 5400/5401), followed by the same burn-in; streams were 5500–5507 and 5600–5607. Banks never exchanged construction samples. Conditional on the selected adaptive thresholds, their random streams remain separate; short MCMC and shared adaptive thresholds do not imply exact IID sampling.

| Level | Frozen threshold | Estimated log X | Selection compression / ESS | Calibration compression / ESS | Gap below seed44 |
|---|---:|---:|---:|---:|---:|
| 3 | -113172.84824147603649 | -3.118477930469643 | 0.3676757812 / 178.47008 | 0.3266601562 / 317.99050 | 22208.81891064 |
| 4 | -112838.36560726894822 | -4.172213954289457 | 0.3676757812 / 108.54302 | 0.3486328125 / 43.85595 | 21874.33627643 |

The log masses multiply only independent-bank calibration fractions into the fixed previous mass. Intervals below are conditional compression diagnostics, not joint propagated uncertainty in log X. Selection diagnostics use the selected threshold and are not independent mass estimates.

| Level / bank | Within-walker SE | Between-walker SE | Block-means SE | SD of walker logL means | Median f0_u variance ratio | Median f0_u ESS |
|---|---:|---:|---:|---:|---:|---:|
| 3 / selection | 0.026078 | 0.036093 | 0.027372 | 177.647 | 0.288697 | 5.464 |
| 3 / calibration | 0.026563 | 0.024090 | 0.026300 | 118.861 | 0.328692 | 4.835 |
| 4 / selection | 0.027754 | 0.046281 | 0.030375 | 154.677 | 0.305118 | 5.580 |
| 4 / calibration | 0.026836 | 0.071959 | 0.034864 | 246.342 | 0.230903 | 5.470 |

Within-walker SE pools each walker’s block-mean variance divided by its block count; between-walker SE is the SD of walker fractions divided by sqrt(8). Block-means SE uses all 64 within-walker block means and includes between-walker variation. The unchanged core ESS uses the largest IID, block-means, or between-walker SE. These are approximate short-chain diagnostics. f0_u ratios compare each walker/label variance to opposite-bank retained IID level-2 survivors, then take the median over 72 ratios. For level 4 this is a lower-contour comparison, not coverage against an IID level-3 reference.

| Level | Walker | Selection exceedance | Calibration exceedance | Selection mean logL | Calibration mean logL |
|---|---:|---:|---:|---:|---:|
| 3 | 0 | 0.25781250 | 0.27343750 | -113227.22827 | -113044.06136 |
| 3 | 1 | 0.53125000 | 0.30078125 | -112855.07123 | -113219.79873 |
| 3 | 2 | 0.26953125 | 0.45312500 | -113224.10213 | -112886.40356 |
| 3 | 3 | 0.27734375 | 0.35937500 | -113214.83855 | -113146.40038 |
| 3 | 4 | 0.31250000 | 0.23828125 | -113133.26930 | -113253.79055 |
| 3 | 5 | 0.44140625 | 0.33593750 | -112761.80455 | -113134.85535 |
| 3 | 6 | 0.40234375 | 0.37109375 | -113152.75693 | -113127.95969 |
| 3 | 7 | 0.44921875 | 0.28125000 | -113100.57287 | -113222.13616 |
| 4 | 0 | 0.33984375 | 0.17968750 | -112840.55023 | -112957.61452 |
| 4 | 1 | 0.50390625 | 0.15234375 | -112498.18746 | -112982.16139 |
| 4 | 2 | 0.15234375 | 0.42578125 | -112987.66498 | -112734.13559 |
| 4 | 3 | 0.20312500 | 0.60937500 | -112948.74396 | -112501.58875 |
| 4 | 4 | 0.45312500 | 0.68359375 | -112753.01030 | -112320.75010 |
| 4 | 5 | 0.49609375 | 0.18750000 | -112759.79293 | -112965.65848 |
| 4 | 6 | 0.42578125 | 0.26953125 | -112780.09666 | -112852.49482 |
| 4 | 7 | 0.36718750 | 0.28125000 | -112672.04503 | -112928.74036 |

The earlier bad-scale calibration had ESS 13.73685, walker fractions 0.03125–0.99609, between-walker SE 0.13438, and median within-walker f0_u variance about 1.44e-6 of the IID reference. At level 3, isotropic calibration has ESS 317.99, fractions 0.23828–0.45313, between-walker SE 0.02409, and median variance ratio 0.32869. This is substantial practical improvement, though the contours and reference samples differ, so the ratio is not a controlled causal estimate.

Level 4 passes with ESS 43.86, but disagreement increases: calibration fractions span 0.15234–0.68359 and between-walker SE 0.07196 dominates block SE 0.03486. Low frequency ESS, substantial half-history fluctuations (saved in JSON), and occasional high-logL excursions leave residual autocorrelation and finite-burn-in uncertainty. There is no common direction of half-history drift proving insufficient burn-in, and these runs cannot isolate contour geometry from short-chain mixing. No automatic tuning or further sampling was performed.

All 12,288 parameter steps and 32 initializations passed finite-prior/likelihood, strict-contour, scalar/shape contract, and direct prior/likelihood cache checks. Contour, cache, nonfinite, parameter-contract, and slice failures were all zero.

| Added level | Selection likelihood calls | Calibration likelihood calls | Direct cache-check likelihood evaluations | Initialization likelihood evaluations |
|---|---:|---:|---:|---:|
| 3 | 18382 | 18317 | 6160 | 16 |
| 4 | 18775 | 18796 | 6160 | 16 |
| Total | 37157 | 37113 | 12320 | 32 |

Each cache check also evaluates the prior. Slice likelihood calls are counted as num_steps + num_shrink: inspection of the unchanged scalar slice implementation shows one slice_fn likelihood call per stepping/shrinking iteration. These are algorithmic evaluation counts, not GPU hardware-operation counts. Previous production costs are excluded.

Raw artifacts: `/tmp/lisa_dns_stage4_ladder_extension/report.json`, `levels.npz`, `frozen_inputs.npz`, `level4_starts.npz`, and four bank traces. Traces include burn-in, retained positions and scalar caches, proposal labels, acceptance, and slice counters. JSON records configuration and input/source hashes, stream seeds, all coordinate variance ratios, half-history diagnostics, and conditional compression intervals.

No DNS production, seed66/88 trajectory, multi-start, fixed-high control, evidence, or posterior reconstruction was run. The highest threshold remains far below all cited historical best likelihoods; catalogue-family connectivity is not established. Evidence reconstruction remains unvalidated and out of scope.
Validation: all 93 existing DNS/adapter tests plus four new cheap extension tests passed (97 total, CPU, float64). New tests cover exclusion of prior calibration samples, strict per-walker promotion, rejection at the unchanged calibration ESS gate, and use of independent calibration in log-mass bookkeeping. Expensive LISA construction is outside pytest. Generic DNS and proposal source files are unchanged.
