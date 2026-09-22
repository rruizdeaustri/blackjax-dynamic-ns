# Stage-4I: arbitrary source labels versus physical source-set differences

**Classification B: label correspondence contributes substantially, but genuine physical differences and likelihood disagreement remain.** This is not evidence for named catalogue families or a proof of disjoint constrained regions. It does not resolve the Stage-4H question of transient relaxation versus persistent separation.

Only the saved failed-attempt calibration trace and its 512-step continuation were used. No MCMC, likelihood, residual reconstruction, production, or kernel edits were performed. Level 9 and the rejected diagnostic threshold are unchanged. Sorting and matching exist only in this analysis adapter.

## Coordinate decoding and fixed distances

All 896 saved states per walker were decoded to nine sources with six coordinates: f0, fdot, iota, psi, lambda, beta. Decoding uses the inspected existing box transform, lo + sigmoid(u)*(hi−lo), with the recorded audited physical prior bounds. Arrays of every decoded catalogue are saved. Neither amplitude nor phase is reconstructed or changed; their marginalization remains untouched.

Primary distances are RMS frequency differences over nine matched sources, measured in Fourier bins: df = 1/31536000 Hz = 3.1709791983764586e-8 Hz. The true prior frequency width is about 631 bins. This normalization is fixed by the observation duration, not chosen to maximize separation.

A. Labelled distance pairs source i with source i. B. Frequency-canonical distance sorts each actual physical catalogue by f0 before comparing it. C. Optimal assignment uses SciPy linear_sum_assignment on squared frequency differences. In one dimension with quadratic cost, sorting is already the optimal assignment, so B and C coincide mathematically. Hungarian matching on all actual final-state pairs independently confirms this equality. These are not two independent pieces of evidence.

For each 256-step window, frequency distances aggregate **all 256×256 cross-time pairs** of actual catalogues per walker pair: square distances are averaged and then square-rooted. First/second moments compute this exactly. No artificial catalogue is made by averaging labelled sources and then sorting. Cross-time distances include within-walker variability and are not distances between independent statistical observations. We also report distance between mean ordered frequency vectors and within-walker cross-time RMS for context.

The supplementary six-coordinate metric uses every 32nd saved state (eight per half) and all 8×8 cross-state pairs per walker pair. Each source cost is the mean of six squared normalized differences. Linear dimensions use their prior widths: f0 2.000887874175568e-5 Hz, fdot 2e-13 Hz/s, and iota/beta approximately pi radians. Psi uses shortest circular distance with period pi (the inspected response uses cos(2 psi), sin(2 psi)); lambda uses period 2 pi. Circular differences are normalized by their periods. Iota and beta are bounded inclination/latitude coordinates and are not wrapped across their interval endpoints. All six dimensions have equal normalized weight, fixed before these comparisons. This is a prior-scale geometric metric, not a waveform mismatch, and does not quotient every possible physical degeneracy.

## How much matching removes

| Window | Median frequency matched/labelled | Range | Median six-coordinate matched/labelled |
|---|---:|---:|---:|
| failed_retained | 0.3690 | 0.2584–0.4784 | 0.7109 |
| continuation_first | 0.4158 | 0.3021–0.5487 | 0.7252 |
| continuation_second | 0.4332 | 0.3431–0.6693 | 0.7262 |

In the final half, matching removes a median **56.7% of frequency RMS distance** (81.2% of squared distance), a factor of about 2.3. For all six coordinates, it removes 27.4% of RMS distance. This is a substantial/moderate reduction, **not orders of magnitude**. The metric-dependent reduction cannot be interpreted as “57% of the calibration failure is caused by labels.” It quantifies how arbitrary correspondence inflates geometric distances, not actual label-swap event counts or a causal share of ESS failure.

Final-half labelled frequency RMS spans 203.66–290.01 bins; matched RMS spans 80.07–175.62 bins. Within-walker matched RMS spans 74.60–117.10 bins, so many between-walker distances are comparable to ordinary within-walker spread. The remaining distance does not by itself prove different disconnected regions. Six-coordinate matched RMS spans 0.2479–0.2769, versus labelled RMS 0.3167–0.3981.

All 28 final-half walker pairs are below. CSV files also contain every pair for the original failed retained window and the first continuation half, along with full-coordinate labelled/sorted/assigned values. Final actual-state pairs and their Hungarian permutations are in report.json.

| Pair | Labelled f0 RMS (bins) | Sorted f0 RMS | Assignment f0 RMS | Matched/labelled | Full-coordinate matched/labelled |
|---|---:|---:|---:|---:|---:|
| 0–1 | 248.570 | 90.772 | 90.772 | 0.3652 | 0.7298 |
| 0–2 | 225.294 | 85.587 | 85.587 | 0.3799 | 0.7252 |
| 0–3 | 241.097 | 101.416 | 101.416 | 0.4206 | 0.6883 |
| 0–4 | 206.927 | 87.099 | 87.099 | 0.4209 | 0.7678 |
| 0–5 | 227.642 | 132.379 | 132.379 | 0.5815 | 0.7541 |
| 0–6 | 237.450 | 81.463 | 81.463 | 0.3431 | 0.7016 |
| 0–7 | 260.757 | 153.204 | 153.204 | 0.5875 | 0.7352 |
| 1–2 | 233.059 | 95.583 | 95.583 | 0.4101 | 0.7007 |
| 1–3 | 241.218 | 111.102 | 111.102 | 0.4606 | 0.6644 |
| 1–4 | 245.954 | 95.637 | 95.637 | 0.3888 | 0.7306 |
| 1–5 | 263.484 | 126.236 | 126.236 | 0.4791 | 0.7284 |
| 1–6 | 236.412 | 87.909 | 87.909 | 0.3718 | 0.6793 |
| 1–7 | 280.156 | 144.184 | 144.184 | 0.5147 | 0.7282 |
| 2–3 | 241.650 | 101.817 | 101.817 | 0.4213 | 0.7304 |
| 2–4 | 238.974 | 84.644 | 84.644 | 0.3542 | 0.7067 |
| 2–5 | 242.337 | 151.384 | 151.384 | 0.6247 | 0.7204 |
| 2–6 | 227.100 | 83.488 | 83.488 | 0.3676 | 0.7068 |
| 2–7 | 261.403 | 174.955 | 174.955 | 0.6693 | 0.7802 |
| 3–4 | 245.903 | 104.066 | 104.066 | 0.4232 | 0.7178 |
| 3–5 | 261.976 | 154.095 | 154.095 | 0.5882 | 0.7195 |
| 3–6 | 260.560 | 101.591 | 101.591 | 0.3899 | 0.6615 |
| 3–7 | 290.009 | 175.623 | 175.623 | 0.6056 | 0.7053 |
| 4–5 | 235.983 | 141.023 | 141.023 | 0.5976 | 0.7898 |
| 4–6 | 203.660 | 80.065 | 80.065 | 0.3931 | 0.7132 |
| 4–7 | 245.783 | 164.348 | 164.348 | 0.6687 | 0.7763 |
| 5–6 | 243.494 | 128.656 | 128.656 | 0.5284 | 0.7439 |
| 5–7 | 231.912 | 102.793 | 102.793 | 0.4432 | 0.7272 |
| 6–7 | 231.021 | 151.030 | 151.030 | 0.6537 | 0.7619 |

## Source sets over time

For each walker the half-to-half shift below compares means of sorted catalogues; the spread columns are RMS optimally matched distances between actual states within each half. This distinguishes moving physical frequency sets from relabelling identical sets. No threshold for a mode transition was tuned.

| Walker | Half-to-half ordered-frequency mean shift (bins) | Within first / second RMS (bins) | Descriptive temporal reading |
|---|---:|---:|---|
| 0 | 28.995 | 82.166 / 79.266 | Moving within a broadly overlapping frequency-set region |
| 1 | 56.860 | 81.958 / 84.709 | Appreciable redistribution; shift smaller than within-half spread |
| 2 | 16.107 | 78.603 / 79.837 | Broadly stable ordered-frequency summary with ongoing movement |
| 3 | 31.521 | 82.568 / 117.103 | Broadly overlapping summary; physical spread increases despite low exceedance |
| 4 | 68.203 | 95.262 / 80.640 | Coherent upward frequency redistribution; not a proven region crossing |
| 5 | 121.095 | 97.691 / 99.980 | Large downward physical source-set shift; not merely a permutation |
| 6 | 25.776 | 66.546 / 74.605 | Moving within a broadly overlapping frequency-set region |
| 7 | 120.182 | 78.967 / 90.377 | Large downward physical source-set shift despite persistently low exceedance |

Walkers 5 and 7 shift all nine ordered-frequency means downward, with vector RMS shifts about 121 and 120 bins. These are large changes of the physical frequency set, not just exchanging names. Walker 4 shifts all ordered means upward, about 68 bins RMS. Block summaries and step distances show continuing motion; these data do not identify an abrupt transition into a different physical region. “Large shift” here describes magnitude relative to other walkers and within-half spread, not a named or disconnected mode. In particular, nearly stationary logL for walker 7 does not imply a stationary source set.

## Relation to likelihood and the two low-exceedance walkers

Across the 28 final-half pairs, descriptive Spearman correlations between matched frequency distance and absolute differences in mean logL / exceedance are −0.061 / −0.088. Pair observations share walkers and are not independent, so no p-values are reported. Frequency-set similarity does not reliably predict either statistic.

| Pair | Matched cross-state f0 RMS (bins) | Distance between ordered-frequency means (bins) | Exceedances | Absolute mean-logL difference | Full matched/labelled |
|---|---:|---:|---:|---:|---:|
| 2–3 | 101.817 | 17.977 | 0.70703 / 0.01562 | 1013.994 | 0.7304 |
| 1–3 | 111.102 | 43.581 | 0.58203 / 0.01562 | 1585.179 | 0.6644 |
| 3–5 | 154.095 | 109.045 | 0.01562 / 0.58594 | 1005.963 | 0.7195 |
| 5–7 | 102.793 | 38.528 | 0.58594 / 0.08594 | 884.068 | 0.7272 |
| 2–7 | 174.955 | 152.769 | 0.70703 / 0.08594 | 892.099 | 0.7802 |
| 1–7 | 144.184 | 114.530 | 0.58203 / 0.08594 | 1463.283 | 0.7282 |

Walker 3 is **not cleanly distinct from all high-exceedance walkers in frequency-set space**: its ordered means are only 17.98 bins RMS from walker 2, compared with their within-walker RMS 117.10 and 79.84 bins. Yet exceedance is 0.015625 versus 0.70703125 and mean logL differs by 1013.99. Other physical coordinates and joint configurations matter; frequency matching alone does not establish equivalence.

Walker 7 differs substantially from high-exceedance walker 2 (152.77-bin mean-set distance), but overlaps high-exceedance walker 5 much more closely (38.53 bins). Their exceedances remain 0.0859375 versus 0.5859375, and mean logL differs by 884.07. Thus low-exceedance walkers do not form one uniformly separated physical frequency-set group. The full-coordinate assignment leaves 66–79% of labelled distance across final-half pairs, but this coordinate metric is also not proof of distinct waveform families.

An exact source relabelling cannot change a symmetric unordered target’s likelihood. Consequently, the observed logL/exceedance differences are not explained by arbitrary labels alone. What remains unresolved is whether their physical differences are transient explorations of the same constrained region or slow communication between different regions. This preserves, rather than overturns, Stage-4H’s ambiguity.

## Residuals, consequence, and scope

No residual chi2 or source_gain arrays are stored for these calibration/continuation states. Obtaining them would require fresh model evaluations, so they were skipped. The cached logL values are sufficient for the likelihood comparisons above. No likelihood calls were made for this stage.

Choose **B**, interpreted as substantial label-related distance inflation plus remaining physical/likelihood differences. Neither A (collapse with overlapping likelihood behavior) nor C (little effect of matching) fits. The extent of disconnected physical-region separation remains indeterminate. Before changing proposal topology, distinguish label communication from physical constrained exploration; an exact label permutation alone leaves logL and ell10 exceedance unchanged. No swap or other kernel change was implemented.

Artifacts: `/tmp/lisa_dns_stage4i_permutations/report.json`, both full decoded NPZ arrays, per-block sorted-frequency CSV summaries, and all-pair CSV tables for three windows. Input hashes, physical bounds, normalizations, assignments, pairwise likelihood differences, within-walker spread, and temporal shifts are recorded.

New MCMC steps: 0. New likelihood calls: 0. Level 10 remains rejected; no new levels, production, evidence, posterior weights, or named catalogue-family classification.
Validation: all 104 previous tests plus three cheap source-set sorting, permutation-assignment, and circular-distance tests passed (107 total). All recorded source input hashes remain unchanged. There is no target sampling in pytest.
