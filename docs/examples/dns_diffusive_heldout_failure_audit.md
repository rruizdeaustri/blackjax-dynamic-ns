# Stage-9D held-out conditional failure: saved-artifact audit

**Stage 9D remains FAILED-CONDITIONAL. Primary causal category for this audit: F — INCONCLUSIVE.** The saved evidence establishes a shared early-window modal-occupation error, but does not uniquely separate first-visit/collection initialization, incomplete finite-time mixing and a correlated fluctuation. It does not establish an incorrect stationary joint target or extreme level-weight imbalance.

This audit is read-only with respect to scientific artifacts and code. Only this document and temporary saved-data reductions were written. No trajectory, resampling, bootstrap, alternate acceptance test, kernel change, seed/model/tolerance/budget change, LISA run or termination evaluation occurred. The held-out report/design and frozen source hashes were verified unchanged. Later saved observations are diagnostic comparisons, not replacements for the officially failed sample.

Source artifacts: [held-out report](dns_diffusive_ladder_heldout.md), [held-out JSON](dns_diffusive_ladder_heldout.json), [preregistered design](dns_diffusive_ladder_heldout_design.json), and `/tmp/dns-stage9d-heldout/original/attempt_000004/{preflight,promotion,candidate}.json`, `calibration_heldout_checks.json`, `selection_heldout_checks.json`, and each bank's `*_trace.npz` / `*_top_observations.npz`. Design SHA256: `6749554b9649995174317303318315eb34cd58f3af42539d5b74bab83b880ad7`. Temporary reproducible reductions: `/tmp/stage9d_failure_diagnostics.py` and `/tmp/stage9d_failure_diagnostics.json`; neither calls a stochastic transition.

## 1. Exact failed construction attempt

Candidate number 5 was generated under **J=4**, the prefix containing levels 0–4. Four candidates had already been accepted. The failed fifth candidate was never appended, the sixth was never attempted, and restart was never launched.

| Quantity | Saved value |
|---|---|
| Generating log threshold | -0.9101304544651906 |
| Candidate log threshold, unaccepted | -0.9003104518727859 |
| Physical sweeps | 32,768 per walker; eight walkers; 262,144 sweep/walker states |
| Burn-in | 1,024 joint sweeps per walker |
| Top visits including burn-in | 53,331 |
| Post-burn top visits | 51,738 |
| Retained top events | 16,384 = 2,048 per walker; first quota events, no thinning |
| Compression ESS | 13329.007831709318 |
| Conditional KS | 0.09116473409724596 |
| KS frozen limit | 0.08 |
| Sample / exact conditional mean | 0.048236413711541665 / -0.4112244897959155 |
| Mean signed error | 0.45946090350745716 |
| Mean absolute-error limit | 0.3074700721890876 |
| Sample / exact second moment | 5.844473738659277 / 6.734233726283104 |
| Second-moment signed error | -0.8897599876238269 |
| Second-moment relative error | -0.13212490444921363 |
| Second-moment absolute relative-error limit | 0.12 |
| Exact mode probabilities | 0.2755 / 0.3980 / 0.3265 |
| Retained mode probabilities | 0.1847 / 0.4379 / 0.3774 |
| Mode probability errors | -0.0908 / 0.0399 / 0.0509 |
| Mode absolute-error limit | 0.07 |
| Candidate calibration compression / truth | 0.37225341796875 / 0.3717394229196642 |
| Proposed log mass / truth, NOT accepted | -5.036398894162843 / -4.979939410889623 |

Thresholds are log likelihoods. The prefix already available at this attempt was:

| j | ell_j | calibrated log Xhat_j | Xhat_j | exact X_j | log weight | log a_j | a_j |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -inf | 0 | 1 | 1 | 0 | -0 | 1 |
| 1 | -5.698424514857786 | -1.02050579921 | 0.360412597656 | 0.379540059516 | 0 | 1.02050579921 | 2.77459779848 |
| 2 | -1.5326205100193282 | -2.03543867172 | 0.130623169243 | 0.137929199528 | 0 | 2.03543867172 | 7.65560968849 |
| 3 | -0.9827770262941096 | -3.02822058443 | 0.0484016882615 | 0.0502223707845 | 0 | 3.02822058443 | 20.6604363591 |
| 4 | -0.9101304544651906 | -4.04821846862 | 0.0174534408111 | 0.0184927361469 | 0 | 4.04821846862 | 57.2952927061 |

The exact top set is the union of three narrow intervals [[-3.8407595000788506, -3.759240499921149], [-0.758874833447229, -0.641125166552771], [2.7516924443509914, 2.8483075556490083]]. Geometric mode regions are separated at -2.25 and 1.05; mode 0 is the peak at -3.8, mode 1 at -0.7, and mode 2 at 2.8. All later mode vectors use that order, not the frozen adapter's binary sign labels.

## 2. Per-walker retained conditional distributions

Top occupancy is post-burn top visits divided by 31,744 physical sweeps. Every retained count is exactly 2,048. Sweep indices are zero-based stored trace indices. All quantities below are reductions of the frozen retained events, not reselected observations.

### Selection

| Walker | Post-burn top visits | Retained | Top occupancy | Mode counts 0/1/2 | Mode fractions 0/1/2 | Mean | Second moment | Last retained sweep |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 0 | 6381 | 2048 | 0.2010 | 476/894/678 | 0.2324 / 0.4365 / 0.3311 | -0.26193 | 6.16748 | 10330 |
| 1 | 6710 | 2048 | 0.2114 | 529/1019/500 | 0.2583 / 0.4976 / 0.2441 | -0.64760 | 5.89074 | 11289 |
| 2 | 6121 | 2048 | 0.1928 | 513/810/725 | 0.2505 / 0.3955 / 0.3540 | -0.23661 | 6.58690 | 12191 |
| 3 | 6498 | 2048 | 0.2047 | 442/790/816 | 0.2158 / 0.3857 / 0.3984 | 0.02592 | 6.42854 | 10343 |
| 4 | 5912 | 2048 | 0.1862 | 672/932/444 | 0.3281 / 0.4551 / 0.2168 | -0.95761 | 6.66329 | 11515 |
| 5 | 6549 | 2048 | 0.2063 | 652/771/625 | 0.3184 / 0.3765 / 0.3052 | -0.61908 | 7.17870 | 11078 |
| 6 | 7234 | 2048 | 0.2279 | 378/974/696 | 0.1846 / 0.4756 / 0.3398 | -0.08351 | 5.56538 | 11007 |
| 7 | 6706 | 2048 | 0.2113 | 561/799/688 | 0.2739 / 0.3901 / 0.3359 | -0.37336 | 6.77794 | 10274 |

| Walker | Direct physical top-to-top mode changes | Retained top-visit mode changes | All post-burn top-visit mode changes | Lower departures | Complete top-lower-top | Complete top-0-top | Cross-mode return witnesses |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 35 | 143 | 1149 | 1149 | 286 | 99 |
| 1 | 0 | 58 | 155 | 1220 | 1219 | 286 | 112 |
| 2 | 0 | 58 | 148 | 1156 | 1155 | 297 | 101 |
| 3 | 0 | 41 | 149 | 1185 | 1184 | 295 | 90 |
| 4 | 0 | 46 | 148 | 1035 | 1034 | 267 | 100 |
| 5 | 0 | 51 | 147 | 1156 | 1155 | 295 | 100 |
| 6 | 0 | 57 | 134 | 1247 | 1247 | 284 | 101 |
| 7 | 0 | 39 | 136 | 1226 | 1226 | 282 | 94 |

### Calibration

| Walker | Post-burn top visits | Retained | Top occupancy | Mode counts 0/1/2 | Mode fractions 0/1/2 | Mean | Second moment | Last retained sweep |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 0 | 6256 | 2048 | 0.1971 | 423/837/788 | 0.2065 / 0.4087 / 0.3848 | 0.00575 | 6.20186 | 10810 |
| 1 | 6594 | 2048 | 0.2077 | 500/793/755 | 0.2441 / 0.3872 / 0.3687 | -0.16684 | 6.61310 | 11502 |
| 2 | 6367 | 2048 | 0.2006 | 481/1000/567 | 0.2349 / 0.4883 / 0.2769 | -0.45935 | 5.80961 | 11749 |
| 3 | 5969 | 2048 | 0.1880 | 261/1029/758 | 0.1274 / 0.5024 / 0.3701 | 0.20042 | 4.99123 | 11265 |
| 4 | 6262 | 2048 | 0.1973 | 409/951/688 | 0.1997 / 0.4644 / 0.3359 | -0.14296 | 5.75115 | 11383 |
| 5 | 6913 | 2048 | 0.2178 | 262/882/904 | 0.1279 / 0.4307 / 0.4414 | 0.44762 | 5.52326 | 9663 |
| 6 | 6638 | 2048 | 0.2091 | 398/890/760 | 0.1943 / 0.4346 / 0.3711 | -0.00413 | 5.92747 | 11579 |
| 7 | 6739 | 2048 | 0.2123 | 292/792/964 | 0.1426 / 0.3867 / 0.4707 | 0.50538 | 5.93810 | 10435 |

| Walker | Direct physical top-to-top mode changes | Retained top-visit mode changes | All post-burn top-visit mode changes | Lower departures | Complete top-lower-top | Complete top-0-top | Cross-mode return witnesses |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 49 | 167 | 1107 | 1107 | 303 | 122 |
| 1 | 0 | 54 | 149 | 1203 | 1203 | 277 | 99 |
| 2 | 0 | 52 | 163 | 1189 | 1188 | 294 | 101 |
| 3 | 0 | 49 | 156 | 1068 | 1067 | 277 | 106 |
| 4 | 0 | 49 | 153 | 1151 | 1150 | 289 | 100 |
| 5 | 0 | 28 | 137 | 1266 | 1265 | 298 | 96 |
| 6 | 0 | 47 | 142 | 1207 | 1206 | 290 | 92 |
| 7 | 0 | 37 | 149 | 1223 | 1222 | 289 | 100 |

A direct top-to-top change means adjacent physical sweeps both have assigned index 4 and different mode labels. All such counts are zero. Changes between top-visit observations can span a lower-level excursion, and are not direct fixed-contour theta jumps. Lower departures count a transition from top to any lower assigned index after the first post-burn top visit; a return completes one trip. A departure near the end can be incomplete. The top-0-top column uses the original saved round-trip definition. Witnesses additionally require a physical mode change with the preceding and current assigned index zero and a different-mode top return. Full physical-trace witness counts must not be confused with counts inside the shorter retained-event window.

**Calibration is not an outlier-walker failure. All eight walkers underrepresent mode 0:** their fractions range from 0.1274 to 0.2441 versus exact 0.27551. Walkers 3, 5 and 7 have the largest deficits, but removing any one walker would not explain away the shared effect. This is systematic within this realized retained window; one run cannot establish systematic bias of the algorithm over repeated independent realizations. Selection has both signs of mode-0 error: walkers 4 and 5 overrepresent it; six others underrepresent it, leaving a much smaller pooled deficit.

## 3. Selection versus calibration: same target, different finite paths

| Quantity | Selection | Calibration |
|---|---|---|
| KS | 0.018436235790322664 | 0.09116473409724596 |
| Mean | -0.39422247364501234 | 0.048236413711541665 |
| Second moment | 6.407370484982368 | 5.844473738659277 |
| Mode fractions | 0.2578 / 0.4266 / 0.3157 | 0.1847 / 0.4379 / 0.3774 |
| Post-burn level occupancies | 0.1885 / 0.2011 / 0.2028 / 0.2025 / 0.2052 | 0.1908 / 0.1999 / 0.2044 / 0.2011 / 0.2037 |
| Post-burn top visits | 52111 | 51738 |
| Top-0-top trips | 2292 | 2317 |
| Cross-mode witnesses | 797 | 816 |
| Compression ESS | 14763.718574670898 | 13329.007831709318 |
| Promoted starting mode labels | [1, 2, 1, 0, 0, 2, 0, 0] | [0, 2, 0, 0, 1, 2, 0, 0] |
| Donor walker labels | [3, 2, 3, 4, 6, 6, 3, 1] | [7, 0, 2, 2, 2, 7, 7, 4] |

Both banks have the same frozen prefix, weights, parameter settings, top quota and invariant constrained target. Their RNG namespaces, promotion histories, source draws and realized mode paths differ. Neither has duplicate promoted coordinate states. The starting mode counts are 4/2/2 for selection and 5/1/2 for calibration. Thus the calibration deficit is not simply an initial absence of the left mode: it started with more left-mode walkers. Repeated donor labels identify common earlier histories, not equal promoted coordinates. Separate banks/random streams do not make promoted starts stationary or independent equilibrium draws.

No structural difference in transition or collection rules was found. Calibration had slightly more full-trace cross-mode witnesses and essentially the same visitation as selection, yet failed. The structural statistical issue is that useful conditional observations are a short correlated visit-order prefix of each chain, not 16,384 independent observations drawn from a proved stationary initialization.

## 4. Evolution over candidate levels 1–5

Candidate n means construction observations generated at J=n-1. Candidate 5 was rejected. Exact mode fractions at J=0 are 0.359375 / 0.206250 / 0.434375 under the full prior; at J=1–4 they are 0.275510 / 0.397959 / 0.326531 because the separated interval lengths are proportional to the widths.

| Candidate / generating J | Bank | KS | Sample mode fractions 0/1/2 | Mean signed error | Second-moment signed / relative error | Top occupancy | Top-0-top trips | Cross-mode witnesses | Compression ESS |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| 1 / 0 | selection | 0.03323 | 0.3456 / 0.1902 / 0.4642 | +0.23955 | +0.53338 / +0.02500 | 1.0000 | 0 | 0 | 3776.2 |
| 1 / 0 | calibration | 0.02980 | 0.3547 / 0.1871 / 0.4583 | +0.19939 | +1.35086 / +0.06332 | 1.0000 | 0 | 0 | 3019.9 |
| 2 / 1 | selection | 0.01206 | 0.2643 / 0.4037 / 0.3320 | +0.04408 | -0.11835 / -0.01667 | 0.5128 | 23386 | 4601 | 14197.4 |
| 2 / 1 | calibration | 0.00592 | 0.2755 / 0.4019 / 0.3226 | -0.00963 | +0.01340 / +0.00189 | 0.5117 | 23595 | 4585 | 14509.8 |
| 3 / 2 | selection | 0.04528 | 0.2326 / 0.4076 / 0.3598 | +0.24903 | -0.35001 / -0.05161 | 0.3415 | 7774 | 2160 | 11243.4 |
| 3 / 2 | calibration | 0.01996 | 0.2805 / 0.4100 / 0.3094 | -0.07345 | -0.06067 / -0.00895 | 0.3413 | 7646 | 2017 | 16223.4 |
| 4 / 3 | selection | 0.01328 | 0.2793 / 0.4071 / 0.3136 | -0.05545 | -0.04433 / -0.00658 | 0.2489 | 3829 | 1236 | 13621.1 |
| 4 / 3 | calibration | 0.04535 | 0.3207 / 0.3365 / 0.3428 | -0.08322 | +0.74550 / +0.11061 | 0.2505 | 3969 | 1221 | 16384.0 |
| 5 / 4 | selection | 0.01844 | 0.2578 / 0.4266 / 0.3157 | +0.01700 | -0.32686 / -0.04854 | 0.2052 | 2292 | 797 | 14763.7 |
| 5 / 4 | calibration | 0.09116 | 0.1847 / 0.4379 / 0.3774 | +0.45946 | -0.88976 / -0.13212 | 0.2037 | 2317 | 816 | 13329.0 |

Calibration KS rises from 0.00592 to 0.01996, 0.04535 and 0.09116 over candidates 2–5. Useful top-0-top trips decline with depth for a fixed physical budget. However the signed modal/moment errors do not deteriorate monotonically in one direction: calibration overrepresented mode 0 at candidate 4 (0.32074), then underrepresented it at candidate 5 (0.18469); its second-moment error flips sign. Selection KS is also nonmonotonic. Candidate 5 is the largest excursion in a depth-dependent, increasingly correlation-sensitive sequence, not evidence of a single stable progressively wrong modal weight.

## 5. What the retained-event rule guarantees, and what it does not

The frozen [collector](../../blackjax/ns/dns_automatic_diffusive.py), `collect_top`, takes the first 2,048 indices per walker with assigned index J after physical sweep 1,024. It includes consecutive top stays and rejection self-loops, excludes all lower-level states, and fails rather than pads an insufficient quota. The saved arrays agree with those exact trace indices. All walkers share the same target; equal K is a deliberate equal allocation, not evidence of IID observations.

Let S be the augmented-state subset j=J and P the fixed physical-sweep kernel. For stationary joint q, one observation at a fixed physical time, conditioned on membership in S, has distribution q_S=q(.|S). The formal conditional cancels a_J/C and is the normalized constrained prior. This statement alone does not identify the initial law of the first future hit of S after a fixed-time burn-in.

The trace/censored chain Y_n=X_(T_n), where T_n enumerates **every** visit to S including stays, has invariant measure q_S. Formally, partitioning state space into S and O, its kernel is

```text
P^S = P_SS + P_SO (I-P_OO)^(-1) P_OS.
```

The stationary equations q_O=q_S P_SO+q_O P_OO and q_S=q_S P_SS+q_O P_OS imply q_S P^S=q_S (with q_S normalized after restriction). This derivation does not require full-sweep reversibility.

If **Y_0 itself has q_S**, every Y_n has q_S, and the mean of a fixed K observations is unbiased for any integrable conditional observable despite autocorrelation. Equal K contributions from walkers with the same stationary conditional target introduce no extra modal weighting, and independence is not required for this expectation identity. Independence and autocorrelation do matter for uncertainty. No visit-frequency importance weighting is needed for that stationary visit sample.

Entry-only states are generally governed by incoming flux, not residence occupancy. In this particular frozen DNS top-boundary kernel there is only a neighboring upward entry from J-1, with a constant eligible MH factor. At stationarity its incoming density is proportional to pi(theta) I[A_(J-1)] a_(J-1) × (1/2) × I[A_J] × min(1,a_J/a_(J-1)), which normalizes to the same constrained prior on A_J. Top departure probability is also theta-independent: (1/2) min(1,a_(J-1)/a_J), about 0.18030 at J=4. Thus entries happen to have q_S at stationary flux for this specific kernel; that does not justify treating them as independent or modifying the collector. In a generic augmented kernel the entry law need not be q_S.

A first future hit from a fixed physical-time snapshot can still have a different law from a typical stationary entry or a conditional physical-time observation. Outside excursions are inspected in proportion to their durations. Consequently a physically stationary chain does not automatically initialize its future visit sequence in stationary visit law; nonstationary promoted starts make the guarantee weaker still. The first K after that hit have expected average (1/K) sum_(n=0)^(K-1) nu (P^S)^n f, where nu is the first-hit law. It equals q_S f exactly only under suitable initialization (or a special cancellation), not from the invariant-measure statement alone.

### Deterministic counterexample to automatic finite-sample unbiasedness

No toy trajectory was generated for this counterexample. Consider a reversible four-state Markov matrix, S={a,b}:

```text
       a       b       c       d
a    0.9       0     0.1       0
b      0     0.9       0     0.1
c   0.25       0    0.71    0.04
d      0  0.0625    0.01  0.9275
q = (0.25, 0.25, 0.10, 0.40).
```

Direct matrix multiplication verifies stationarity and detailed balance. Exit probability from either a or b is the same 0.1; stationary incoming flux to a and b is equal, and q(.|S)=(0.5,0.5). However first-hit probabilities from c and d are (29/33,4/33) and (4/33,29/33). Starting from stationary q at a physical time, the first hit, including an immediate hit when already in S, is (17/44,27/44), not (0.5,0.5). Its trace kernel still has the correct invariant target. A fixed finite K prefix retains some initialization effect. This establishes the absence of a universal unbiasedness guarantee; it is not a demonstration that this mechanism explains the held-out magnitude.

For the actual held-out run, promoted empirical histories are not certified stationary, and 1,024 joint sweeps are not a proof of joint or first-visit stationarity. First-K collection consistently estimates q_S in the ergodic limit, but it is **not unconditionally finite-sample unbiased in this protocol**. Nor is a histogram of all selected physical-time observations generally an exactly unbiased finite-sample ratio: its denominator is random. Later/full-trace comparisons below are therefore informative diagnostics, not a validated alternative collection rule.

## 6. Frozen level transition and parameter target audit

The [level kernel](../../blackjax/ns/dns.py), `level_move_probability` / `build_level_kernel`, proposes each sign with probability 1/2 even at an endpoint. Invalid endpoints are rejected self-loops, without renormalizing neighbor probabilities. For an eligible neighbor k, its acceptance is

```text
min(1, exp(log_weight[k]-log_weight[j]-log_mass[k]+log_mass[j]))
 = min(1, a_k/a_j).
```

Eligibility is a valid neighboring index and strict likelihood exceedance at the proposed nonzero level; level 0 is the unconstrained prior. The prior factor cancels in a theta-fixed index move. Estimated masses are part of a_j: replacing them by true masses in the code would change the frozen target, not correct an error. With zero log weights, eligible upward acceptance is one. Downward acceptances on the four edges are 0.360412597656, 0.362426757812, 0.370544433594, 0.360595703125. A downward proposal is always eligible for a valid current state, but may be rejected.

| Saved proposal audit | Selection | Calibration |
|---|---:|---:|
| down_direction_fraction | 0.49839019775390625 | 0.4991912841796875 |
| up_direction_fraction | 0.5016098022460938 | 0.5008087158203125 |
| boundary_attempts | 51859 | 51544 |
| boundary_accepts | 0 | 0 |
| max_probability_difference | 5.551115123125783e-17 | 5.551115123125783e-17 |
| lower_assigned_parameter_sweeps | 207986 | 208807 |

All 262,144 sweep/walker records per bank match the independently evaluated eligibility expression. Every recorded acceptance probability agrees to at most 5.55e-17; every committed index equals proposed index on acceptance and old index on rejection; every recorded previous index matches the preceding sweep's index. All out-of-bounds attempts rejected. The observed direction fractions are consistent with the source's half/half proposal; they are not used to infer an unrecorded random uniform. No level-transition detailed-balance error was found.

In `dns.build_kernel`, the parameter threshold is `levels.loglikelihood[state.level_index]`, read from the **pre-parameter assigned index** each sweep. Parameter moves occur first, then one index step. The stored final assigned index can therefore differ from the threshold actually used for that parameter move. The diffusive adapter calls this existing kernel; the scalar fixed-contour branch is not used. The model callback in [the held-out harness](../../examples/dns_diffusive_ladder_heldout.py) captures model constants and the frozen symmetric ±0.35 direction, but receives the threshold dynamically. It uses prior density for the vertical slice, likelihood for strict horizontal eligibility, and the original finite stepping-out/shrinkage kernel. There is no state-dependent selector or cross-walker exchange in this run.

Every saved post-parameter theta satisfies the preceding assigned contour and the final assigned contour, and recomputation agrees with prior/likelihood caches under the unchanged CPU/x64 contract. There are 177,803 selection and 178,602 calibration parameter sweeps at lower assigned levels whose resulting likelihood is **outside** the top contour. This rules out an accidental permanent top-level threshold in the realized lower-level trajectory. The original `parameter_valid` and `parameter_accepted` flags are all true. An invariance proof cannot be obtained from contour membership alone, but the audited source is the existing constrained-prior-preserving slice transition; no contradictory code path or cache evidence was found.

Under that existing parameter-kernel contract, each parameter substep preserves q conditional on j, each index substep preserves q by MH, and their composition preserves q even if the composed sweep is not reversible. Its stationary conditional really is pi(theta) I[A_J]/X_J. The observed finite conditional histogram need not already equal this invariant law.

## 7. Connectivity does not determine stationary mode probabilities

The failed calibration bank's full post-burn directed cross-mode return counts are

```text
        to 0   to 1   to 2
from 0     0    160     74
from 1   178      0    186
from 2    67    151      0
```

There are 816 saved witnesses. Every mode can reach every other mode. These full-trace counts are not expected occupancy weights or independent draws, and they include returns after retained quotas were filled. Correct conditional weights depend on dwell times, correlations, transient initialization and the stationary measure, not just an edge being nonzero. All three modes are reachable and all are present in retained events; their **retained** proportions are wrong. Incorrect invariant stationary weights have not been established.

## 8. What compression ESS measures

The reported ESS comes from [tail_diagnostics](../../blackjax/ns/dns_levels.py) for I_nw = I[log L_nw > candidate threshold]. It computes p=mean(I), N=16,384 and a standard error equal to the maximum of Bernoulli IID SE, block-mean SE with 64 consecutive visit-order events per walker, and between-walker SE. ESS=min(N,p(1-p)/SE²). This is an approximate Bernoulli variance-equivalent **compression-indicator** ESS, not an ESS for mode labels, theta, theta², a CDF supremum or the entire distribution. For this bank the between-walker term is the maximum; its SE is 0.0041870930642738426.

The distinction is especially sharp on the held-out equal-peak envelope. At these disconnected contours every mode has an interval of width proportional to sigma_k. Moving the threshold rescales all three intervals by the same factor, so the analytic survivor fraction **within each mode** is the same 0.3717394229196642. Incorrect relative mode allocation can therefore leave compression almost correct: observed compression is 0.37225341796875. Rapid within-mode coordinate motion makes the compression indicator decorrelate even while the mode label changes slowly. High compression ESS and a failed conditional KS/mode/moment distribution are fully compatible.

The global mean and second-moment discrepancies are predominantly modal-allocation errors: using the mode centers, delta_p dot mu is about +0.460, and delta_p dot mu² is about -0.893, close to the actual +0.45946 and -0.88976 errors. As a diagnostic only, per-mode KS against each uniform top interval is approximately 0.03372 / 0.00587 / 0.01442 for calibration; no new retrospective gate is applied.

## 9. Alternative observable-specific ESS, diagnostic only

For each saved retained observable y (2048 visit events × 8 walkers), a general variance-equivalent diagnostic uses v=sample variance and SE²=max(v/N, var(block means)/(number of block means), var(walker means)/8), ESS=min(N,v/SE²). The block sizes 64/128/256 are sensitivity checks, not new criteria. An independent within-walker FFT autocovariance calculation averages lag covariances and uses the initial positive, monotonically decreasing adjacent-lag pair sequence to estimate tau; spectral ESS=N/tau. This spectral calculation de-means each walker and cannot detect a bias common to all walkers. Both estimates assume sufficient stationarity/block length and can overstate ESS under an unresolved slow transient. The small difference from original compression ESS in the general variance formula is its ddof=1 variance rather than p(1-p).

| Calibration observable | Block ESS 64 | Block ESS 128 | Block ESS 256 | Spectral ESS | Estimated visit-order tau | Unranked split Rhat |
|---|---:|---:|---:|---:|---:|---:|
| theta | 444.1 | 343.7 | 382.6 | 313.0 | 52.34 | 1.01660 |
| theta2 | 537.3 | 426.4 | 319.8 | 350.4 | 46.75 | 1.02007 |
| mode0 | 540.8 | 391.6 | 353.4 | 354.5 | 46.22 | 1.02079 |
| mode1 | 495.5 | 402.4 | 320.4 | 323.5 | 50.64 | 1.01725 |
| mode2 | 456.2 | 344.5 | 364.6 | 298.2 | 54.94 | 1.01457 |
| compression | 13329.8 | 13329.8 | 13329.8 | 16237.7 | 1.01 | 1.00000 |

Mode/theta ESS is roughly 300–540 rather than 13,329. For selection, spectral ESS is about 247 / 425 / 333 for mode indicators, 262 for theta and 352 for theta². The ordinary unranked split-Rhat calculation divides each visit sequence into halves and compares within/between means; it is descriptive only and not a convergence certificate. Calibration's mode-0 discrepancy is about 4.40 of its 256-event block SE (using the observed rather than exact variance). That is not comfortable evidence for category E, but a Gaussian p-value would be unjustified because stationarity and the relevant long-run variance have not been established. None of these quantities changes the frozen held-out acceptance decision.

## 10. Level weights and actual visitation

The intended marginal occupancy is q(j) proportional to a_j X_j = X_j/Xhat_j, not exactly uniform unless estimates equal truth. Exact held-out interval masses give:

| Level | Relative marginal weight X/Xhat | Expected normalized occupancy | Selection observed | Calibration observed |
|---|---:|---:|---:|---:|
| 0 | 1.000000 | 0.192080 | 0.188453 | 0.190792 |
| 1 | 1.053071 | 0.202274 | 0.201089 | 0.199935 |
| 2 | 1.055932 | 0.202823 | 0.202798 | 0.204401 |
| 3 | 1.037616 | 0.199305 | 0.202460 | 0.201140 |
| 4 | 1.059547 | 0.203518 | 0.205200 | 0.203731 |

All levels receive approximately 19–20.5% of post-burn time; top visitation is close to its exact expected marginal, and every walker far exceeds its 2048 quota. There is no extreme occupancy imbalance supporting category D. Balanced level occupancy does not prove equilibration of theta or sufficient independent modal excursions. More levels also reduce root round-trip rates within a fixed cost even when occupancies are balanced; this is a mixing-efficiency observation, not evidence that weights violate their target. No weight change is proposed or made.

## 11. Saved physical-time evolution and stationarity evidence

The first row is the actual 0–1024 burn-in. The four subsequent equal-duration windows partition the existing post-burn physical trace. These histograms use **all assigned-top observations in that fixed window**, with their natural random counts. They are not reruns, a proposed retention rule, or replacement acceptance results.

| Bank | Physical indices [start,end) | Top events | Mode fractions 0/1/2 | Mean | Second moment | Level occupancies 0..4 |
|---|---|---:|---|---:|---:|---|
| selection | [0,1024) | 2042 | 0.2909 / 0.4594 / 0.2498 | -0.72789 | 6.38615 | 0.1541 / 0.1680 / 0.2064 / 0.2223 / 0.2493 |
| selection | [1024,8960) | 13014 | 0.2391 / 0.4415 / 0.3194 | -0.32300 | 6.17459 | 0.1857 / 0.2066 / 0.2008 / 0.2018 / 0.2050 |
| selection | [8960,16896) | 13248 | 0.2320 / 0.4176 / 0.3503 | -0.19363 | 6.30281 | 0.1895 / 0.1990 / 0.2006 / 0.2022 / 0.2087 |
| selection | [16896,24832) | 13078 | 0.2891 / 0.3869 / 0.3240 | -0.46260 | 6.90600 | 0.1901 / 0.1989 / 0.2030 / 0.2020 / 0.2060 |
| selection | [24832,32768) | 12771 | 0.2629 / 0.3538 / 0.3834 | -0.17309 | 6.97501 | 0.1884 / 0.1999 / 0.2068 / 0.2038 / 0.2012 |
| calibration | [0,1024) | 1593 | 0.2517 / 0.2756 / 0.4727 | 0.17442 | 7.47957 | 0.1906 / 0.2059 / 0.2079 / 0.2012 / 0.1945 |
| calibration | [1024,8960) | 12613 | 0.1920 / 0.4118 / 0.3962 | 0.09114 | 6.08433 | 0.1939 / 0.1996 / 0.2071 / 0.2008 / 0.1987 |
| calibration | [8960,16896) | 13271 | 0.2849 / 0.4323 / 0.2828 | -0.59349 | 6.54310 | 0.1917 / 0.1960 / 0.2010 / 0.2022 / 0.2090 |
| calibration | [16896,24832) | 13239 | 0.2619 / 0.3940 / 0.3441 | -0.30726 | 6.67518 | 0.1829 / 0.1970 / 0.2085 / 0.2031 / 0.2085 |
| calibration | [24832,32768) | 12615 | 0.2746 / 0.4205 / 0.3049 | -0.48411 | 6.56085 | 0.1947 / 0.2071 / 0.2010 / 0.1985 / 0.1987 |

Calibration's early post-burn top sample has mode-0 fraction 0.1920 and mean +0.0911; the final physical quarter has mode-0 fraction 0.2746 and mean -0.4841, much closer to exact 0.27551 and -0.41122. Pooling all saved post-burn top visits gives calibration fractions 0.25386 / 0.41463 / 0.33152 and mean -0.32668. Its discrepancy is smaller than the frozen quota sample, but it is not proof of stationarity or a new PASS. Per-walker final-quarter mode-0 fractions still range from approximately 0.1741 to 0.3714. Selection also fluctuates materially across physical quarters.

Equal per-walker weighting of **all** post-burn visit means gives calibration mode fractions 0.2535 / 0.4142 / 0.3323; natural pooling gives 0.2539 / 0.4146 / 0.3315. Their small difference, versus the much larger first-quota deficit, does not suggest that per-walker equalization alone explains the failure.

The saved retained-event chronology is also not a simple monotone burn-in relaxation. Across four successive groups of 512 retained events per walker, calibration's pooled mode fractions are approximately 0.2791/0.3408/0.3801; 0.1492/0.4858/0.3650; 0.1343/0.4587/0.4070; and 0.1763/0.4661/0.3577. The first visit-quarter is near the correct left-mode weight, followed by a sustained deficient segment. Selection's mode-0 quarters are 0.2861, 0.1536, 0.2759, 0.3154. These fluctuations and later recovery argue against claiming stable incorrect invariant modal weights. They are compatible with slow modal occupation dynamics and finite-window effects, but do not prove a deterministic initial transient rather than a rare correlated realization.

## 12. Primary failure class and consequence

**Primary category: F — INCONCLUSIVE.** What is established is that the frozen finite retained sample had a mode-allocation error shared across walkers, with much lower conditional-observable information than its compression ESS suggests. Code/record audits support the intended joint target (no B evidence), and actual occupancies rule out extreme visitation imbalance (no D evidence). The stationary trace-chain collection has the correct invariant target, while the actual first-K initialization has no unconditional finite-sample unbiasedness guarantee. Hence later recovery cannot uniquely identify A versus nonstationarity/insufficient mixing (C), and a single correlated record with uncertain long-run variance does not quantitatively justify E. The audit does not select an algorithmic fix from those unresolved alternatives.

The logically relevant next *methodological question*, before any further validation, is: **What is the finite-sample law of the first-K top-visit process initialized after the specified joint burn-in, and what evidence establishes stationary modal weights at collection time?** A separately authorized development study would need to distinguish collection initialization effects from modal equilibration and calibrate uncertainty for conditional observables. This is a collection/visit-process stationarity question, not an authorization to alter weights, rerun the held-out toy, or change the frozen tests. No implementation or tuning is performed or proposed here.

Stage 9D remains FAILED-CONDITIONAL; broader construction robustness remains unvalidated. No new held-out, end-to-end, termination, LISA or production run is authorized by this audit. STOP.
