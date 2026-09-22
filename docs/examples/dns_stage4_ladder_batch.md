# Stage-4G: resumable bounded LISA ladder batch

This construction-only adapter resumes the frozen Stage-4F ladder. It neither changes generic DNS files nor calls DNS production, evidence, posterior weighting, or catalogue classification.

Stage-4F persisted complete separate selection/calibration traces. Each original walker has retained strict level-4 survivors: selection counts `[87,129,39,52,116,127,109,94]`, calibration counts `[46,39,109,156,175,48,69,72]`. A dedicated resume checkpoint had not been saved. The adapter imports one randomly selected survivor from each original walker, including its cached prior and likelihood, using separate promotion streams. It does not reconstruct levels 0–4 or generate replacement banks. All resumed cached states are directly checked against the real target before stepping.

The fixed settings are eight walkers in each bank, 128 burn-in and 256 retained steps per walker, block size 32, minimum ESS 20, target exp(-1), 54 scales pi/sqrt(3), and the unchanged 20/40/40 proposal. The batch attempts only levels 5–10 and stops at the first failed selection/calibration or integrity failure. No retry, extra sampling, or tuning follows a failed gate.

Checkpoint files `checkpoint_levelN.json` are versioned, digest-protected, atomically saved, and never overwritten. Each contains all frozen thresholds and log masses, accepted diagnostics, separate named banks with eight positions and scalar caches each, settings, configuration/source provenance, survivor indices, and the predeclared next-level PRNG schedule. Loading checks strict support, finite values, shapes, bank separation, digest, settings, and the fixed Stage-4F prefix. Loaded arrays are read-only. A resumed job appends accepted levels and verifies that its entire loaded prefix remains unchanged.

Promotion selects one retained survivor per original walker, without exchanging states between banks. NumPy promotion seeds are `800000 + accepted_level*10 + bank`. JAX construction streams are `700000 + proposed_level*100 + bank*10 + walker`, with bank 0 selection and bank 1 calibration. Independence means separate banks and random streams conditional on the adaptively selected contour; finite MCMC histories are not IID samples or proof of global mixing.

Run from the repository root:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=. MPLCONFIGDIR=/tmp/lisa_dns_mpl \
/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python \
-m examples.lisa_dns_stage4.run_ladder_batch \
--stage4f /tmp/lisa_dns_stage4_ladder_extension \
--output /tmp/lisa_dns_stage4_ladder_batch
```

For a later explicitly authorized continuation within this runner's level-10 ceiling, use `--resume /path/checkpoint_levelN.json` and a new output directory. An accepted checkpoint can resume without reconstructing lower levels. This does not authorize rerunning a failed batch.

Within-walker uncertainty pools each walker's eight block means. Block-means uncertainty includes all 64 within-walker block means and can include between-walker variation. The unchanged core diagnostic takes the maximum of IID, block-means, and between-walker SE to obtain Bernoulli-equivalent ESS. Log-mass estimates use only independent-bank calibration fractions; reported compression uncertainties are conditional, not joint propagated mass intervals.

Per-step scalar slice cost is recorded as `num_steps + num_shrink`. Direct cache evaluations are separate (one likelihood and one prior each), including the 16 resumed initial states per level. Initialization likelihood cost is zero because persisted scalar caches are reused; their revalidation is charged to direct cache checks. Previous construction and production costs are excluded. Trace files retain burn-in and retained samples, proposal labels, and slice counters. CPU tests contain no real-target construction.

## Results

Five consecutive higher levels (5–9) passed without intervention. Level 10 failed independent calibration ESS at 16.7982 and was not frozen. The highest accepted checkpoint is `checkpoint_level9.json`. All settings and lower levels remain unchanged.

| Level | Accepted | Threshold | Selection ratio / ESS | Calibration ratio / ESS | Estimated log X | Increment | Gap below seed44 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | True | -112388.10901787043258 | 0.367675781 / 29.31616 | 0.430664062 / 55.65861 | -5.014640884442064 | 450.25659 | 21424.07969 |
| 6 | True | -111915.69049021089450 | 0.367675781 / 23.84235 | 0.437988281 / 26.91157 | -5.840204008542667 | 472.41853 | 20951.66116 |
| 7 | True | -111344.96574156903080 | 0.367675781 / 22.59835 | 0.486328125 / 43.12003 | -6.561075737117467 | 570.72475 | 20380.93641 |
| 8 | True | -110809.82135735315387 | 0.367675781 / 23.14766 | 0.551269531 / 43.27895 | -7.156607159127204 | 535.14438 | 19845.79203 |
| 9 | True | -110252.99476697217324 | 0.367675781 / 21.39372 | 0.595214844 / 29.74021 | -7.675440015805330 | 556.82659 | 19288.96544 |
| 10 | False | -109401.11425363768649 | 0.367675781 / 24.27236 | 0.364257812 / 16.79822 | -8.685333401760968 | 851.88051 | 18437.08492 |

Level-10 threshold and log X are rejected diagnostic estimates only. The frozen log X at level 9 is −7.67544001580533. Accepted increments are 450, 472, 571, 535, and 557: a modest overall increase with nonmonotonic variation. The rejected next proposal jumps by 852. No guaranteed number of levels to the catalogue region can be inferred.

| Level / bank | Within-walker SE | Block-means SE | Between-walker SE | Between dominates calibration? |
|---|---:|---:|---:|---|
| 5 / selection | 0.023480 | 0.037030 | 0.089053 | — |
| 5 / calibration | 0.027877 | 0.034355 | 0.066372 | True |
| 6 / selection | 0.027194 | 0.041723 | 0.098748 | — |
| 6 / calibration | 0.030153 | 0.042714 | 0.095639 | True |
| 7 / selection | 0.026901 | 0.042265 | 0.101429 | — |
| 7 / calibration | 0.028448 | 0.036920 | 0.076115 | True |
| 8 / selection | 0.023988 | 0.040342 | 0.100219 | — |
| 8 / calibration | 0.032307 | 0.039533 | 0.075603 | True |
| 9 / selection | 0.026679 | 0.042897 | 0.104246 | — |
| 9 / calibration | 0.024195 | 0.037689 | 0.090007 | True |
| 10 / selection | 0.024402 | 0.039920 | 0.097869 | — |
| 10 / calibration | 0.022344 | 0.044447 | 0.117412 | True |

Between-walker uncertainty dominates every calibration in this batch. Level 10 has between-walker SE 0.11741 versus within-walker 0.02234 and block-means 0.04445; calibration fractions span 0.01172–0.81641. This is directly observed walker disagreement, not merely a within-walker autocorrelation penalty.

| Level / bank | Walker fractions, labels 0–7 |
|---|---|
| 5 / selection | 0.28906250, 0.80468750, 0.19140625, 0.31640625, 0.19921875, 0.21875000, 0.73046875, 0.19140625 |
| 5 / calibration | 0.62109375, 0.57812500, 0.37890625, 0.47265625, 0.28125000, 0.32031250, 0.12500000, 0.66796875 |
| 6 / selection | 0.86328125, 0.21875000, 0.14843750, 0.33593750, 0.70312500, 0.26171875, 0.36718750, 0.04296875 |
| 6 / calibration | 0.70312500, 0.57812500, 0.13671875, 0.20703125, 0.16015625, 0.66015625, 0.26953125, 0.78906250 |
| 7 / selection | 0.58593750, 0.60937500, 0.10156250, 0.64062500, 0.69531250, 0.15625000, 0.11328125, 0.03906250 |
| 7 / calibration | 0.52734375, 0.60156250, 0.53515625, 0.76171875, 0.35546875, 0.33593750, 0.09375000, 0.67968750 |
| 8 / selection | 0.32812500, 0.77734375, 0.14062500, 0.44531250, 0.79296875, 0.20703125, 0.05078125, 0.19921875 |
| 8 / calibration | 0.61718750, 0.45312500, 0.75000000, 0.57421875, 0.54687500, 0.87109375, 0.16406250, 0.43359375 |
| 9 / selection | 0.28906250, 0.67578125, 0.05468750, 0.64843750, 0.80859375, 0.17187500, 0.15234375, 0.14062500 |
| 9 / calibration | 0.76953125, 0.80078125, 0.49609375, 0.40625000, 0.80078125, 0.89062500, 0.17578125, 0.42187500 |
| 10 / selection | 0.24609375, 0.74218750, 0.00000000, 0.65625000, 0.21093750, 0.06250000, 0.46875000, 0.55468750 |
| 10 / calibration | 0.47656250, 0.22656250, 0.81640625, 0.02734375, 0.57421875, 0.74609375, 0.01171875, 0.03515625 |

| Level / bank | Range of walker mean logL | Median / minimum f0_u ESS | Median between-walker f0 mean SD | Range across labels of f0 mean SD |
|---|---:|---:|---:|---:|
| 5 / selection | -112560.65297 … -111793.70559 | 4.663 / 2.869 | 1.16482 | 0.41742 … 1.60792 |
| 5 / calibration | -112601.00793 … -111434.36255 | 5.394 / 2.802 | 1.27263 | 0.37659 … 1.95418 |
| 6 / selection | -112245.81849 … -110567.68372 | 5.194 / 2.810 | 1.10652 | 0.52680 … 1.70172 |
| 6 / calibration | -112147.16299 … -110434.33820 | 6.031 / 3.043 | 1.33942 | 0.68032 … 2.49944 |
| 7 / selection | -111723.26063 … -110740.03028 | 6.078 / 2.971 | 0.76949 | 0.40959 … 1.93116 |
| 7 / calibration | -111665.75986 … -110045.64560 | 5.653 / 2.917 | 0.96228 | 0.56383 … 1.51558 |
| 8 / selection | -111166.48958 … -109884.36990 | 5.223 / 2.842 | 1.28529 | 0.96188 … 1.63643 |
| 8 / calibration | -111018.36012 … -109737.10274 | 6.168 / 2.842 | 1.01429 | 0.58359 … 1.52776 |
| 9 / selection | -110635.59484 … -108829.83005 | 4.766 / 2.921 | 1.16022 | 0.64783 … 1.77767 |
| 9 / calibration | -110478.00395 … -108647.09074 | 5.879 / 2.900 | 1.18992 | 0.46951 … 1.84397 |
| 10 / selection | -110138.38711 … -108330.42545 | 5.571 / 2.744 | 0.98052 | 0.50466 … 2.27919 |
| 10 / calibration | -110036.44452 … -108516.85763 | 5.847 / 2.854 | 1.14674 | 0.61652 … 1.45230 |

Full per-walker/per-label f0 latent means, SDs and ranges are in `report.json` under each row’s `mixing` field; all eight walker logL means and half-history summaries are under the uncertainty fields. No finite-ESS diagnostic is interpreted as global mixing.

Calibration quality is degrading overall, with fluctuations rather than a monotonic ESS trend. Selection ESS remains close to the gate (about 21–29). Calibration ESS ranges 27–56 before failing at 17. Calibration fractions rise from 0.431 at level 5 to 0.595 at level 9, while the selection fraction remains 0.368. This sustained bank mismatch is an additional warning of imperfect constrained exploration, even for accepted levels. It is not an independent significance test because consecutive levels share adaptive thresholds and walker ancestry. At the failed level, the bank-wide fraction returns near the target but severe within-bank disagreement still invalidates calibration.

Frequency ESS remains low (median about 5–6, minimum about 3 per 256-step history). Residual autocorrelation and persistent walker heterogeneity are observed. Insufficient burn-in and contour geometry may contribute, but this fixed-budget run does not identify their separate effects or justify automatic tuning.

| Level | Selection cost proxy | Calibration cost proxy | Cache-check evaluations | Initialization evaluations | Elapsed seconds | Cumulative selection / calibration / cache |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 19152 | 18711 | 6160 | 0 | 30.66 | 19152 / 18711 / 6160 |
| 6 | 19960 | 19757 | 6160 | 0 | 27.45 | 39112 / 38468 / 12320 |
| 7 | 20182 | 19733 | 6160 | 0 | 22.34 | 59294 / 58201 / 18480 |
| 8 | 20417 | 19761 | 6160 | 0 | 22.23 | 79711 / 77962 / 24640 |
| 9 | 21242 | 20072 | 6160 | 0 | 22.60 | 100953 / 98034 / 30800 |
| 10 | 21008 | 20837 | 6160 | 0 | 23.15 | 121961 / 118871 / 36960 |
| Total | 121961 | 118871 | 36960 | 0 | 148.43 | — |

All 36,864 parameter steps plus 96 resumed initial states passed strict contour, finite prior/likelihood, parameter-contract, and direct-cache checks. Cumulative failures: contour 0, cache 0, nonfinite 0, parameter-contract 0, slice 0. Initialization costs are zero additional likelihood evaluations because saved caches are reused; the 96 initialization checks are included in the 36,960 direct checks.

The six attempts took about 148 seconds in their measured sampling/diagnostic sections on the V100 (including first-step compilation and trace writes, excluding initial model setup). Subsequent attempts took about 22–27 seconds each. Per-level parameter likelihood proxy costs rose modestly, roughly 38k–41k, with 6,160 direct checks per attempt. Local runtime is manageable; the current fixed strategy is not validated as a practical route all the way to the catalogue region because calibration already fails with a remaining accepted-threshold gap of 19,288.97 logL. Continuing unchanged is unsupported by this result. No depth or time-to-catalogue extrapolation is made.

All 97 previous tests plus four new cheap checkpoint/stop tests passed (101 total). The new tests cover round-trip, separation on resume, immutable accepted prefix/overwrite rejection, and first-failure stopping. Actual saved checkpoints were also loaded read-only to verify their separation and frozen prefix.

Artifacts are in `/tmp/lisa_dns_stage4_ladder_batch`: `report.json`, `checkpoint_level4.json` through `checkpoint_level9.json`, and the twelve bank trace files for attempted levels 5–10. No level-10 checkpoint exists. No production, seed44/66/88 trajectory, evidence, posterior weights, or catalogue classification ran. Evidence reconstruction remains unvalidated and out of scope.
