# Stage 9E: bounded top-level stationarity development study

**CASE F — STILL INCONCLUSIVE.** No systematic first-K/equal-walker distortion or reproducible promoted-start left-mode deficit was established. The historical calibration trace improves later, but this eight-replicate control does not resolve the cause or calibrate its rare-event probability. Stage 9D remains **FAILED-CONDITIONAL**; its model is now development-only and can never again be held out.

## Scope and preregistration

Design SHA256: `7b73003d4e9263f3faaa025dbfb0c1236b5010692a13ddf53701439e9f61fcb5`. The [design](dns_top_level_stationarity_design.json) and [checksum](dns_top_level_stationarity_design.sha256) were saved before any new transition. The [diagnostic driver](../../examples/dns_top_level_stationarity_study.py) performs separate existing / prepare / run / summarize phases. [Machine-readable results](dns_top_level_stationarity_study.json) include every replicate, available cut/window, observable ESS, exact truth and trace hash.

Exactly eight seeds: [910701, 910702, 910703, 910704, 910705, 910706, 910707, 910708]. Each seed has two paired initialization arms, eight independent walkers and 32,768 physical sweeps per walker: 16 fixed traces total (4,194,304 walker sweeps). Both arms use matched transition-key seeds, with independent seeds across replicates. No further seeds, lengths or cut points were added after inspection. All existing scientific/core hashes and the preregistered diagnostic-driver hash were verified before/after trajectories and final reporting.

The promoted arm uses the eight actual saved candidate-5 calibration promoted positions, frozen at preregistration. It does not redo promotion, reconstruct a ladder, or characterize uncertainty in the upstream ladder. The stationary arm initializes each walker from the exact joint target; those initial states are independent. Both arms call the same original `dns.build_kernel`, the same constrained slice callback and zero log-visitation weights. Exchange remains identity, as in Stage 9D. No reference draws enter any subsequent transition. CPU/x64 is required and checked. No core source, estimator, gate, historical report or design changed. No LISA, reconstruction, termination, validation or production run occurred.

## Exact reference and frozen target

The model is the existing Uniform[-8,8] weighted-normal upper envelope with centers [-3.8, -0.7, 2.8], widths [0.27, 0.39, 0.32] and weights [0.2755102040816327, 0.3979591836734694, 0.326530612244898]. Candidate 5 uses assigned levels 0..4, thresholds ['-inf', -5.698424514857786, -1.5326205100193282, -0.9827770262941096, -0.9101304544651906] and calibrated log masses [0.0, -1.0205057992077717, -2.035438671723301, -3.028220584432101, -4.048218468620065]. The candidate threshold is -0.90031045187278591; it remains unaccepted historically.

For each level, solve the Gaussian-envelope threshold inequality analytically, clip and merge the resulting intervals. Uniform-prior constrained mass is their total length / 16. CDF is the sum of clipped interval lengths below theta / total length; first and second moments integrate theta and theta² over these intervals. This reference does not use a DNS empirical histogram.

At J=4, X=0.018492736146886027, mean=-0.41122448979591553, second moment=6.7342337262831036; exact mode probabilities are 0.27551 / 0.39796 / 0.32653. Intervals: [[-3.8407595000788506, -3.759240499921149], [-0.758874833447229, -0.641125166552771], [2.7516924443509914, 2.8483075556490083]].

Full-joint initialization is practical and exact: q(j) proportional to a_j X_j with a_j=exp(-log Xhat_j), giving probabilities 0.19208 / 0.20227 / 0.20282 / 0.19931 / 0.20352. Given j, select an interval with probability proportional to length and sample uniformly within it. The same interval procedure supplies an exact conditional sampler for any frozen level. It is used only as a stationary initialization reference, never to rescue DNS trajectories.

## Physical time and visit time

Burn-in is the original 1024 physical sweeps; K=2048 assigned-top occupation events per walker, including self-loops, with no thinning. Collection is post-transition at the same point in each joint sweep as the original adapter. Independent physical cut points are 1024, 8192, 16384, 24576; no separately tuned runs are launched for them. Physical windows are nonoverlapping [0,1024), [1024,8192), [8192,16384), [16384,24576), [24576,32768). The first is burn-in only. Cumulative rows begin at 1024.

Visit windows use per-walker top-visit indices [0,K), [K,2K), [2K,3K) after each cut. They are mutually disjoint within that cut, but windows at different cuts overlap and are not additional independent replicates. Only complete K-windows for all eight walkers are reported; none is padded. A cut at 24576 generally leaves too little physical time for K visits. A first-K window extends to different physical times for different walkers. An all-top physical-window sample has random counts; comparing its variance with equal K is not a same-N experiment.

Round trips in physical tables mean top -> any lower assigned level -> top, completed entirely within the window; excursions crossing a window boundary are not counted. They do not themselves certify correct mode weights.

## Existing Stage-9D calibration trace, analyzed before new sampling

| Physical window | Top events | Top occupancy | KS | Mode probabilities | Mean | Second moment | Round trips |
|---|---:|---:|---:|---|---:|---:|---:|
| [0,1024) | 1593 | 0.19446 | 0.14802 | 0.25173 / 0.27558 / 0.47269 | 0.17442 | 7.47957 | 290 |
| [1024,8192) | 11333 | 0.19763 | 0.09322 | 0.18653 / 0.39848 / 0.41498 | 0.17412 | 6.14588 | 2065 |
| [8192,16384) | 13714 | 0.20926 | 0.04527 | 0.28358 / 0.43481 / 0.28161 | -0.59372 | 6.51622 | 2540 |
| [16384,24576) | 13709 | 0.20918 | 0.01732 | 0.26107 / 0.39901 / 0.33992 | -0.31946 | 6.63284 | 2452 |
| [24576,32768) | 12982 | 0.19809 | 0.02475 | 0.27361 / 0.42389 / 0.30250 | -0.48938 | 6.52970 | 2329 |

| Cumulative post-burn endpoint | KS | Modes | Mean | Second moment |
|---|---:|---|---:|---:|
| 8192 | 0.09322 | 0.18653 / 0.39848 / 0.41498 | 0.17412 | 6.14588 |
| 16384 | 0.03615 | 0.23967 / 0.41837 / 0.34196 | -0.24629 | 6.34865 |
| 24576 | 0.02871 | 0.24724 / 0.41152 / 0.34124 | -0.27218 | 6.44918 |
| 32768 | 0.02191 | 0.25386 / 0.41463 / 0.33152 | -0.32668 | 6.46938 |

| Physical cut | First K KS | Second disjoint K KS | Third disjoint K KS |
|---|---:|---:|---:|
| 1024 | 0.09116 | 0.02167 | unavailable |
| 8192 | 0.04648 | 0.02977 | unavailable |
| 16384 | 0.00937 | unavailable | unavailable |
| 24576 | unavailable | unavailable | unavailable |

The historical first K has KS 0.09116473409724596 and modal weights 0.18469 / 0.43787 / 0.37744. The second disjoint K has KS 0.02167 and weights 0.29498 / 0.39655 / 0.30847. A third complete K-window is unavailable. Later physical windows and cumulative estimates approach truth, with fluctuations rather than strict monotone convergence. This is evidence of an unusually deficient early realized occupation window, not proof of a reproducible burn-in transient. Historical samples are neither replaced nor reclassified.

## Independent replicate distributions

| Seed | Promoted first K KS | Promoted second K KS | Stationary first K KS | Stationary second K KS |
|---|---:|---:|---:|---:|
| 910701 | 0.03235 | 0.02306 | 0.01513 | 0.02206 |
| 910702 | 0.05798 | 0.00354 | 0.03821 | 0.02164 |
| 910703 | 0.05295 | 0.04468 | 0.03876 | 0.04591 |
| 910704 | 0.02412 | 0.03903 | 0.04673 | 0.01274 |
| 910705 | 0.01934 | 0.06705 | 0.02621 | 0.08506 |
| 910706 | 0.01543 | 0.01314 | 0.02255 | 0.03099 |
| 910707 | 0.00773 | 0.03013 | 0.01445 | 0.05983 |
| 910708 | 0.02436 | 0.04851 | 0.03116 | 0.00871 |

| Seed / initialization | First K mode probabilities | Second K mode probabilities | First K mean | First K second moment |
|---|---|---|---:|---:|
| 910701 / saved_promoted_start | 0.25092 / 0.39331 / 0.35577 | 0.29749 / 0.38806 / 0.31445 | -0.23250 | 6.60458 |
| 910701 / exact_joint_stationary | 0.29041 / 0.37958 / 0.33002 | 0.29120 / 0.40234 / 0.30646 | -0.44507 | 6.96707 |
| 910702 / saved_promoted_start | 0.29211 / 0.43915 / 0.26874 | 0.27496 / 0.39825 / 0.32678 | -0.66524 | 6.54261 |
| 910702 / exact_joint_stationary | 0.29913 / 0.41199 / 0.28888 | 0.25494 / 0.41614 / 0.32892 | -0.61591 | 6.78703 |
| 910703 / saved_promoted_start | 0.32806 / 0.39313 / 0.27881 | 0.23126 / 0.42065 / 0.34808 | -0.74097 | 7.11496 |
| 910703 / exact_joint_stationary | 0.30286 / 0.40918 / 0.28796 | 0.23010 / 0.40997 / 0.35992 | -0.63097 | 6.83021 |
| 910704 / saved_promoted_start | 0.25525 / 0.40521 / 0.33954 | 0.27606 / 0.35864 / 0.36530 | -0.30263 | 6.54556 |
| 910704 / exact_joint_stationary | 0.24249 / 0.38641 / 0.37109 | 0.27698 / 0.38660 / 0.33643 | -0.15295 | 6.60026 |
| 910705 / saved_promoted_start | 0.29395 / 0.38153 / 0.32452 | 0.23535 / 0.37140 / 0.39325 | -0.47545 | 6.97470 |
| 910705 / exact_joint_stationary | 0.30164 / 0.38257 / 0.31580 | 0.23682 / 0.35181 / 0.41138 | -0.52959 | 7.01699 |
| 910706 / saved_promoted_start | 0.26221 / 0.41064 / 0.32715 | 0.26385 / 0.41632 / 0.31982 | -0.36774 | 6.55295 |
| 910706 / exact_joint_stationary | 0.28351 / 0.41217 / 0.30432 | 0.24500 / 0.44580 / 0.30920 | -0.51363 | 6.68097 |
| 910707 / saved_promoted_start | 0.26929 / 0.40393 / 0.32678 | 0.30536 / 0.35828 / 0.33636 | -0.39092 | 6.64833 |
| 910707 / exact_joint_stationary | 0.27441 / 0.38501 / 0.34058 | 0.33521 / 0.36481 / 0.29999 | -0.35853 | 6.82069 |
| 910708 / saved_promoted_start | 0.28619 / 0.40930 / 0.30450 | 0.22711 / 0.43829 / 0.33459 | -0.52129 | 6.72191 |
| 910708 / exact_joint_stationary | 0.29004 / 0.41339 / 0.29657 | 0.26862 / 0.40265 / 0.32874 | -0.56097 | 6.71630 |

## Ensemble centering and collection comparison

Intervals below are descriptive Student-t intervals over the eight independent replicates (7 degrees of freedom), not newly imposed scientific acceptance criteria. Initialization arms are paired and cannot be counted as 16 independent replicates of one process. Intervals assume sufficiently regular replicate-level variability; they cannot establish a tiny rare-event probability.

| Arm / estimator | Mean mode probabilities | Signed mean error ± 95% halfwidth | Signed second-moment error ± 95% halfwidth |
|---|---|---:|---:|
| saved_promoted_start / first_K | 0.27975 / 0.40453 / 0.31573 | -0.05087 ± 0.14646 | -0.02104 ± 0.18118 |
| saved_promoted_start / second_K | 0.26393 / 0.39374 / 0.34233 | +0.09114 ± 0.13119 | -0.04510 ± 0.31526 |
| saved_promoted_start / all_postburn_occupation | 0.27115 / 0.40161 / 0.32724 | +0.01598 ± 0.07065 | -0.05572 ± 0.12612 |
| exact_joint_stationary / first_K | 0.28556 / 0.39754 / 0.31690 | -0.06473 ± 0.13189 | +0.06821 ± 0.11718 |
| exact_joint_stationary / second_K | 0.26736 / 0.39751 / 0.33513 | +0.05536 ± 0.17737 | -0.05000 ± 0.30666 |
| exact_joint_stationary / all_postburn_occupation | 0.27228 / 0.39952 / 0.32821 | +0.01592 ± 0.08316 | -0.03297 ± 0.17060 |

Stationary-control first-K mode errors are +0.01005 / -0.00042 / -0.00963, with 95% halfwidths approximately 0.01663 / 0.01278 / 0.02420: all include zero. Promoted-start first-K errors are +0.00424 / +0.00657 / -0.01080; all likewise include zero. No ensemble recreates the historical left-mode deficit. Paired first-vs-second and first-vs-all-occupation modal differences include zero in both arms. This supports no detected systematic distortion, not a proof of exact finite-K unbiasedness.

Equal-per-walker weighting of all post-burn top observations differs from natural occupation pooling by at most 0.00166 (promoted) or 0.00213 (stationary) in any mode. Ensemble-average differences are under 0.00049. Equalization alone is not a demonstrated explanation of the historical 0.09082 left-mode deficit.

Only one promoted replicate has a complete third K-window (seed 910707, KS 0.01881); no stationary replicate does. Missing windows are explicitly marked unavailable in JSON. These quota limitations do not mean the original first-K collector failed its quota.

## Physical-time settling across controls

| Arm | Window start | Mean / maximum KS | Mean modal probabilities |
|---|---:|---|---|
| saved_promoted_start | 1024 | 0.03887 / 0.05913 | 0.27806 / 0.40053 / 0.32141 |
| saved_promoted_start | 8192 | 0.03174 / 0.08173 | 0.27041 / 0.41068 / 0.31891 |
| saved_promoted_start | 16384 | 0.03464 / 0.07146 | 0.26093 / 0.40370 / 0.33537 |
| saved_promoted_start | 24576 | 0.02486 / 0.04868 | 0.27628 / 0.39133 / 0.33239 |
| exact_joint_stationary | 1024 | 0.03974 / 0.05487 | 0.29187 / 0.39275 / 0.31538 |
| exact_joint_stationary | 8192 | 0.02768 / 0.04131 | 0.27227 / 0.40251 / 0.32522 |
| exact_joint_stationary | 16384 | 0.04329 / 0.08990 | 0.24916 / 0.41453 / 0.33631 |
| exact_joint_stationary | 24576 | 0.03810 / 0.06590 | 0.27743 / 0.38807 / 0.33450 |

The exact-stationary arm has no physical-time burn-in bias in expectation, yet its finite windows fluctuate substantially. One second-K stationary window reaches KS 0.08506; a stationary physical window reaches 0.08990. Later windows are not universally closer: neither arm demonstrates the consistent replicated early bias required for CASE A. They also do not demonstrate persistent failure to settle required for CASE D. The historical recovery alone cannot distinguish those causes.

## Observable-specific ESS

The frozen compression diagnostic remains unchanged: Bernoulli tail-indicator variance with the maximum of IID, block64 and between-walker SE. Other quantities use diagnostic block256 variance-equivalent ESS with the same maximum principle and within-walker spectral ESS from initial positive monotone autocovariance pairs. These estimates assume stationarity and adequate block/lag truncation; neither is a convergence certificate. Within-walker centering can hide shared drift. Every observable has a different autocorrelation function; no common whole-distribution ESS is implied.

| Arm | Observable | Median spectral ESS | Spectral range | Median block256 ESS |
|---|---|---:|---|---:|
| saved_promoted_start | compression | 16238.3 | 15733.8–16384.0 | 16333.1 |
| saved_promoted_start | mode0 | 288.1 | 224.2–322.9 | 278.6 |
| saved_promoted_start | mode1 | 370.0 | 290.9–434.7 | 305.0 |
| saved_promoted_start | mode2 | 269.5 | 244.5–370.1 | 256.1 |
| saved_promoted_start | theta | 256.7 | 194.7–310.4 | 251.8 |
| saved_promoted_start | theta2 | 345.1 | 303.6–376.6 | 266.7 |
| exact_joint_stationary | compression | 16362.9 | 15825.5–16384.0 | 12731.9 |
| exact_joint_stationary | mode0 | 282.6 | 243.5–325.9 | 276.1 |
| exact_joint_stationary | mode1 | 367.1 | 344.5–410.1 | 272.3 |
| exact_joint_stationary | mode2 | 267.9 | 241.4–341.3 | 285.1 |
| exact_joint_stationary | theta | 247.5 | 221.4–288.6 | 269.5 |
| exact_joint_stationary | theta2 | 363.4 | 303.4–387.3 | 276.7 |

| Seed | Frozen compression ESS, promoted | Frozen compression ESS, stationary |
|---|---:|---:|
| 910701 | 15836.0 | 14024.5 |
| 910702 | 16119.1 | 11437.8 |
| 910703 | 7045.8 | 15738.2 |
| 910704 | 16384.0 | 10827.8 |
| 910705 | 16281.2 | 9950.9 |
| 910706 | 16384.0 | 16384.0 |
| 910707 | 16384.0 | 14964.6 |
| 910708 | 10594.0 | 7804.3 |

The historical compression ESS is 13,329 while conditional-observable diagnostics were roughly 300–540; controls likewise give hundreds for mode/moment ESS and thousands for compression. At these equal-peak envelope contours, each mode has the same within-mode candidate survivor fraction. Compression therefore remains insensitive to modal allocation error and decorrelates rapidly within a mode. These diagnostics would warn that compression ESS does not certify conditional precision; they would not have predicted the exact historical realization or established a new acceptance gate.

## Probability of a historical-size discrepancy

saved_promoted_start: 0/8 first-K KS exceedances of the historical 0.09116473409724596; empirical frequency 0, exact binomial 95% interval [0, 0.36942]. The two arms are not pooled because paired trajectories are dependent.
exact_joint_stationary: 0/8 first-K KS exceedances of the historical 0.09116473409724596; empirical frequency 0, exact binomial 95% interval [0, 0.36942]. The two arms are not pooled because paired trajectories are dependent.

A sensitivity calculation using stationary-control median mode-0 spectral ESS 282.6, variance p0(1-p0), and a stationary Gaussian modal-mean approximation puts the historical left-mode deficit at 3.42 standard errors, two-sided probability approximately 0.00063268. This is for one modal observable, **not a KS probability**. Slow-chain tail behavior, ESS estimation and finite first-hit initialization may invalidate the Gaussian approximation; the controls cannot verify such a small tail probability. Different long-run-variance estimators give different standardized discrepancies. It is therefore not justified to declare the historical KS either impossible or an established ordinary fluctuation. IID N=16,384 KS significance is not used.

## Collection mathematics and causal classification

A stationary joint sweep observed at a fixed physical time, conditional on assigned level J, has the exact constrained-prior law. The occupation trace chain retains every top observation, including stays, and has this law as its invariant distribution. Fixed K observations starting with that invariant trace law are unbiased in expectation, regardless of their autocorrelation; equal K from identically targeted walkers requires no visit-frequency weights. Entry-only samples are a different process.

Exact full-joint initialization makes the physical chain stationary at every deterministic sweep. It does not, by itself, prove that the first future top visit after a fixed cutoff starts in the invariant trace law: first-hit probabilities can depend on outside excursion durations. Thus the control deliberately tests the actual first-K procedure under stationary physical dynamics, rather than assuming its mathematical unbiasedness. No systematic distortion was resolved by eight replicates, but that does not remove the finite-first-hit caveat established in Stage 9D-F.

**Classification: CASE F — STILL INCONCLUSIVE.** CASE A lacks replicated consistent early signed bias; CASE B lacks demonstrated stationary-control systematic distortion; CASE C lacks historical-size exceedances or a calibrated autocorrelation-tail model; CASE D lacks consistent persistent failure of later samples; CASE E has no new deterministic defect evidence. The study demonstrates correlated modal fluctuations and later recovery, but cannot assign a unique cause at the frozen sample size. This is a bounded negative diagnostic result, not validation of the collector.

**Single next issue: stop and reassess how to establish stationary modal weights and calibrated finite-first-K conditional uncertainty.** This includes the unresolved distinction between first-hit initialization effects and rare slow-modal fluctuations. No fix, reliability gate, burn-in change or new validation is authorized by this result. Stage 9D remains FAILED-CONDITIONAL and its model remains development-only. STOP.
