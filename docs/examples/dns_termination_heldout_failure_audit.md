# Stage 8H held-out failure audit

**Stage 8H held-out validation remains FAIL. Automatic termination remains
UNVALIDATED.** Historical Stage 8 also remains FAIL. This document is a
read-only diagnosis of the completed, failed run. No trajectory, checkpoint
continuation, reconstruction stream or fresh-process restart was rerun.
No code, protocol, seed, budget, gate or stopping criterion was changed.

## Evidence and an important persistence limitation

All numerical observations come from saved artifacts under
`/tmp/dns-termination-heldout-validation/`, particularly
`continuous/ladder/attempt_000000..000003/`, their accepted checkpoints,
`continuous/decision_3.json`, and the existing held-out validation report.
The frozen `dns_levels.tail_diagnostics` implementation establishes how the
recorded uncertainty and ESS were defined; it was inspected, not modified.

**The failed level-4 selection history was not serialized.** The failed
attempt contains candidate statistics, promotion/preflight provenance and
`stop.json`, but no checkpoint or raw retained bank archive. Accepted
attempts contain `checkpoint/banks.npz`. Full level-4 per-walker likelihood
quantiles, time sequences, mode occupancy and switching counts therefore
cannot be recovered. They are not replaced by a regenerated stream.

Available failed-bank information is unusually diagnostic nonetheless:
8,192 total retained events, the candidate threshold, global and per-walker
survivor counts, uncertainty/ESS, and the exact eight promoted starting
states recoverable by saved donor/time indices from the preceding checkpoint.
The source hashes of those states were checked against recorded provenance.

For accepted construction level j, the saved bank histories were sampled
under the previous contour ell_(j-1), then used to propose/calibrate ell_j.
They are not new inference records sampled at the newly accepted contour.
Each bank has 1,024 retained times for each of eight walkers. All statistics
below respect that time-by-walker layout.

## Exact first failure: attempted level 4

The first failure remains `automatic_construction.selection_ESS`.
Zero-based attempt 3 requested level 4, but rejected the candidate with
`status=insufficient_tail_information` and `stop_reason=selection_gate`.

| Quantity | Saved value |
|---|---:|
| Current accepted log threshold | -1.3299343712381886 |
| Candidate log threshold | -1.2715287082049687 |
| Candidate likelihood threshold | 0.28040264015136135 |
| Raw strict compression fraction | 0.36767578125 |
| Target compression | exp(-1), approximately 0.36787944117 |
| Total retained selection events | 8192 |
| Candidate survivors | 3012 |
| Selection ESS | 12.0093475208749 |
| Required ESS | 20 |
| Between-walker SE | 0.1391370252157515 |
| Recorded conservative SE | 0.1391370252157515 |
| Naive IID SE | 0.005327303606180806 |
| Recorded conservative interval | [0.03866899713035732, 0.6966825653696427] |
| Naive IID Wilson interval | [0.3572987425141107, 0.3781768629020103] |
| Complete retained blocks | 128, 16 per walker, block size 64 |
| Selection sweeps completed | 1280, including burn-in |
| Constituent numerical checks | PASS |

There is no global shortage of survivors: 3,012 events exceed the candidate
threshold, and the raw fraction closely matches the target. Selection uses
an order statistic, so a pooled fraction near target does not establish
reliable mode mixing or independent tail information.

## Walker survival and recoverable starting provenance

Walker and donor indices below are zero-based. The survivor count is exact,
recovered from the saved fraction times 1,024. Starting coordinates and
cached log likelihoods were copied from saved source indices only.
Components are inferred dominant likelihood components, not stored latent
component IDs. M1/M2/M3 correspond to centers -3.0/0.2/3.4.

| Walker | Survivors | Survival fraction | Donor walker | Source retained index | Starting component | Starting theta | Starting cached log L |
|---|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 0 | 6 | 457 | M2 | 0.217824676384 | -1.32436580069 |
| 1 | 0 | 0 | 5 | 784 | M2 | 0.233719434513 | -1.32546846176 |
| 2 | 776 | 0.7578125 | 3 | 378 | M1 | -3.15181406141 | -1.27864391225 |
| 3 | 0 | 0 | 7 | 54 | M2 | 0.212403280287 | -1.32414961089 |
| 4 | 731 | 0.7138671875 | 4 | 1018 | M1 | -2.95162249402 | -1.21393338924 |
| 5 | 723 | 0.7060546875 | 3 | 528 | M1 | -3.11546710242 | -1.24828492532 |
| 6 | 0 | 0 | 2 | 488 | M2 | 0.18390347252 | -1.32435122169 |
| 7 | 782 | 0.763671875 | 4 | 107 | M1 | -2.90278438405 | -1.23615316543 |

Four walkers contribute **zero** survivors; walkers 2, 4, 5 and 7 contribute
all 3,012. Every survivor-producing walker started in M1; every zero-survivor
walker started in M2. This exact alignment supports persistent modal
separation, although the missing raw failed history prevents proving every
walker's full retained component membership or transition history.

A coarse per-walker likelihood distribution is recoverable: all retained
values passed the current strict-contour check, so exceed -1.3299343712381886.
For walkers 0, 1, 3 and 6, all 1,024 values are at or below the candidate
threshold -1.2715287082049687. For walkers 2, 4, 5 and 7, respectively
248, 293, 301 and 242 values lie in that lower interval; the remaining
776, 731, 723 and 782 lie above the candidate. Finer distributions, means,
minima/maxima and quantiles for this failed bank are **not persisted**.
The starting cached likelihoods above are individual starting points, not
substitutes for retained distributions.

The SE was independently recovered from saved per-walker fractions:

```
between_walker_SE = std(fraction_w, ddof=1) / sqrt(8)
conservative_SE = max(IID_SE, within-walker-block_SE, between_walker_SE)
ESS = min(8192, p*(1-p)/conservative_SE**2)
```

Between-walker SE equals the recorded maximum exactly and gives ESS
12.0093475208749. The block SE is not individually saved for failed
selection; it cannot be recovered without the time sequence, but cannot
exceed the recorded maximum under the inspected diagnostic. The conservative
SE is about 26.1 times the IID SE. This ESS is a Bernoulli-variance equivalent
for compression, not an actual count of independent events or the posterior
importance-weight ESS.

## Calibration and donor provenance at the failure

**Calibration for attempted level 4 was never run.** Candidate selection
failed before the construction loop called the calibration kernel.
There is no calibration ESS, compression estimate or retained calibration
history for that attempted level. The calibration history in the preceding
checkpoint belongs to the previously accepted level-3 construction attempt.

Both banks were promoted before selection, so prospective calibration starts
and RNG schedules were recorded even though their sampling was not reached.
Selection donor IDs were `[6,5,3,7,4,3,2,4]`: six distinct donor walkers,
with donors 3 and 4 contributing twice each. Those two M1 donor histories
supplied all four M1 starts. The eight selected full states were distinct,
with zero exact duplicates, so the issue is not literal duplicate particles.
Distinct points from the same correlated history do not imply independent
modal ancestry.

Prospective calibration donor IDs were `[4,2,2,2,3,5,7,4]`, with retained
indices `[11,840,569,370,427,69,744,898]`. All eight copied calibration
starts were M1; there were five distinct donor walkers and eight distinct
full states. This is a saved initialization imbalance, **not a measurement
of an unexecuted calibration trajectory**. No calibration behavior, ESS or
future bias is assigned to it.

Recorded source-history hashes, donor IDs, flat pool indices, retained-time
indices, donor multiplicities and per-state hashes establish immediate
provenance. They do not provide persistent ancestor IDs through all kernel
updates or complete genealogies. Persistent *modal affiliation* is directly
measurable in previous retained histories; persistent exact lineage through
the failed history is not recoverable.

## Previous accepted levels: reliability deterioration

Compression S denotes the selection fraction at the candidate; compression
C denotes the independent calibration fraction used to freeze mass.
Every listed bank has 8,192 retained events. Failed level 4 has no C row.

| Level | S compression | S survivors | S ESS | S between-walker SE | S block SE | C compression | C survivors | C ESS | C between-walker SE | C block SE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.367065430 | 3007 | 490.859011 | 0.021755685 | 0.020604433 | 0.395019531 | 3236 | 609.763885 | 0.019333189 | 0.019796988 |
| 2 | 0.367797852 | 3013 | 123.945185 | 0.043312948 | 0.017284239 | 0.310546875 | 2544 | 61.283575 | 0.059107679 | 0.019592669 |
| 3 | 0.367675781 | 3012 | 41.255349 | 0.075069286 | 0.021855807 | 0.550537109 | 4510 | 31.067553 | 0.089245576 | 0.024139844 |
| 4 rejected | 0.367675781 | 3012 | 12.009348 | 0.139137025 | Not saved | Not run | — | — | — | — |

Selection ESS declined 490.9 → 123.9 → 41.3 → 12.0. Calibration ESS declined
609.8 → 61.3 → 31.1. Between-walker SE increasingly dominated within-walker
block SE. Passing level 3 did not prove complete mode exploration; it met
the gate with considerably less information than earlier levels.

| Construction level | Selection M1/M2/M3 event counts | Calibration M1/M2/M3 event counts | Selection retained component switches | Calibration retained component switches |
|---|---|---|---:|---:|
| 1 | [2370, 3853, 1969] | [1467, 4567, 2158] | 320 | 308 |
| 2 | [1537, 5895, 760] | [1779, 4385, 2028] | 11 | 12 |
| 3 | [2295, 5897, 0] | [5120, 3072, 0] | 1 | 0 |

These are unweighted occupancy counts under the generating contour, not
posterior probabilities. Labels use the largest saved-coordinate Gaussian
likelihood contribution, `c_k*NormalPDF(theta;mu_k,s_k)`; fixed-region labels
at -1.4 and 1.8 give the same qualitative deterioration. Switch counts compare
successive retained states and omit burn-in and within-sweep transitions.
A retained switch is not automatically an independent sample.

At level 2, calibration walker 7 spent all 1,024 retained events in M3 and
had zero candidate survivors. At level 3, selection walkers 3 and 4 stayed
entirely in M1; walker 1 had 247 M1 and 777 M2 events with one switch; the
other five stayed entirely in M2. All eight level-3 calibration walkers
stayed in a single retained component: five in M1 and three in M2. Thus
seven selection walkers and all calibration walkers had no retained
component switches at that level.

The level-3 banks had very different modal allocations: selection M1
occupancy 2295/8192 (28.0%) versus calibration 5120/8192 (62.5%). Their
candidate compression estimates differed correspondingly, 0.36767578125
versus 0.550537109375. This is direct saved-state evidence of heterogeneous,
poorly equilibrated ensemble allocations, without requiring a guessed
counterfactual calibration at level 4.

## Missing modes versus legitimately excluded contours

The absence of M3 in level-3 construction histories is not by itself a
missing-mode bug. Those histories were generated above the accepted level-2
likelihood threshold 0.23607512850891021. On the fixed right region
`theta>=1.8`, the sum of componentwise likelihood upper bounds is only
0.1484004818046059, so that region is excluded by the contour.

Similarly, the failed level-4 candidate threshold is 0.28040264015136135.
Bounding each Gaussian component by its maximum on the interval gives
likelihood bounds 0.2710044454563706 on `[-1.4,1.2]` and
0.09026254274747902 on `[1.2,1.8]`. Both are below the candidate, as is
the right-region bound. Hence candidate survivors necessarily lie in the
left region M1. The previous contour still permits M1 and M2; estimating
how much conditional mass lies in M1 requires their relative representation,
not just many samples within the surviving island.

These bounds are elementary calculations from the saved model parameters,
not new likelihood trajectories or new truth-reference runs. They explain
why central walkers can have zero candidate survivors without an arithmetic
error. The failure is the modal clustering and insufficient effective
information to estimate the mixture's compression reliably.

No global absence of the surviving left mode is evident: four failed-bank
walkers supply survivors and the source banks contain M1. The missing full
failed history prevents proving all occupation details. There is also no
basis for equating a component excluded from a deep contour with loss of
that posterior component: fixed-contour reconstruction retains lower strata.
The saved level-3 reconstruction has all three nonzero component probabilities
approximately 0.2480 / 0.6195 / 0.1324; its mean/q84 diagnostics nonetheless
failed their partial accuracy limits, as recorded in the original report.
These inference diagnostics do not override the construction-gate failure.

## Safety gate and separation from termination

Rejecting ESS 12.0093 below the unchanged requirement 20 was scientifically
appropriate. The pooled survivor count masks extreme clustering: four zero
walkers, four high-survival walkers, donor-mode persistence, and a conservative
compression interval spanning approximately 0.039 to 0.697. Freezing a new
compression mass on the strength of the narrow IID interval would ignore
this dependence and modal-allocation uncertainty.

The gate is an approximate reliability check, not proof of unbiased sampling
or complete exploration. Passing earlier ESS values did not certify those
properties. Nevertheless it correctly prevented accepting this candidate
and proceeding to independent calibration with insufficient selection
information. There is no justification here for lowering ESS=20.

The stopping criterion was evaluated at accepted levels 0..3 and returned
CONTINUE. At the final accepted level, U_3=0.09590719912457871,
D_3=0.05038250568540154 and D_3-U_3=-0.045524693439177165. The required
positive normalization was absent; no valid stopping ratio was declared.
Nothing in the saved evidence shows an incorrect stopping decision.
Automatic-stop accuracy, deeper-tail coverage and restart-to-stop identity
were never reached and remain unvalidated.

## Final blocker classification and further work boundary

**Primary blocker: automatic-ladder robustness problem in this frozen
multimodal regime and budget.** The immediate failure is independent
selection reliability, quantified by the monotone ESS decline, sharply
reduced retained modal switching, mismatched bank allocations, and ESS
12.0093 below 20. It is not a demonstrated termination-method failure.

A finite-budget/toy interaction may contribute; one failed, fixed-budget
run cannot distinguish an intrinsic algorithm limitation from insufficient
mixing at these particular settings. That uncertainty does not make the
immediate blocker ambiguous: automatic level construction failed its own
gate before termination could be assessed. No alternate budget or algorithm
was tested, and no tuning proposal is made.

This held-out validation stays closed and FAIL. Further validation of this
same run by tuning, reseeding or retrying would invalidate its held-out role
and is not justified or authorized. An independently predeclared robustness
study using independent evidence could be scientifically justified, but
would require separate authorization and could not retroactively validate
Stage 8H. No changes to walker counts, sweeps, reconstruction counts,
promotion, population exchange, compression target or stopping tolerance
are proposed here. No LISA or production validation is authorized.

## Appendix: saved successful per-walker profiles

The vector in the last column is cached **log likelihood**
`[minimum, q05, median, q95, maximum]`. M1/M2/M3 counts total 1,024 for each
row. All distributions are computed directly from accepted bank archives.
They do not describe the unavailable failed level-4 history.

### Accepted construction level 1

| Bank | Walker | Candidate survivors | Survival fraction | M1/M2/M3 counts | Retained component switches | Log likelihood profile |
|---|---:|---:|---:|---|---:|---|
| selection | 0 | 352 | 0.343750000 | [351, 576, 97] | 40 | [-90.997007, -31.046832, -2.917118, -1.289802, -1.209209] |
| selection | 1 | 362 | 0.353515625 | [268, 472, 284] | 43 | [-91.745840, -67.034793, -2.637271, -1.350164, -1.235381] |
| selection | 2 | 353 | 0.344726562 | [184, 381, 459] | 30 | [-13.852893, -5.285021, -2.544565, -1.395408, -1.207058] |
| selection | 3 | 453 | 0.442382812 | [307, 467, 250] | 47 | [-36.929812, -10.444461, -2.362538, -1.332946, -1.206638] |
| selection | 4 | 344 | 0.335937500 | [348, 594, 82] | 48 | [-6.245895, -5.685381, -2.481629, -1.329651, -1.207261] |
| selection | 5 | 307 | 0.299804688 | [254, 480, 290] | 47 | [-77.347780, -8.435465, -2.638141, -1.334956, -1.207345] |
| selection | 6 | 493 | 0.481445312 | [136, 492, 396] | 30 | [-5.726977, -4.585730, -2.192790, -1.325032, -1.206627] |
| selection | 7 | 343 | 0.334960938 | [522, 391, 111] | 35 | [-62.003041, -37.624478, -3.069570, -1.344405, -1.210321] |
| calibration | 0 | 320 | 0.312500000 | [39, 662, 323] | 35 | [-15.901784, -8.739292, -2.992632, -1.379140, -1.323966] |
| calibration | 1 | 362 | 0.353515625 | [262, 539, 223] | 39 | [-8.815451, -5.712535, -2.533378, -1.323972, -1.207048] |
| calibration | 2 | 429 | 0.418945312 | [53, 639, 332] | 36 | [-9.012716, -4.630567, -2.356788, -1.341589, -1.323968] |
| calibration | 3 | 405 | 0.395507812 | [300, 602, 122] | 31 | [-17.062572, -10.110352, -2.737757, -1.326894, -1.207493] |
| calibration | 4 | 364 | 0.355468750 | [349, 541, 134] | 27 | [-18.718908, -8.298370, -2.644981, -1.324330, -1.206626] |
| calibration | 5 | 416 | 0.406250000 | [212, 447, 365] | 52 | [-28.234607, -5.712568, -2.512017, -1.353938, -1.246852] |
| calibration | 6 | 502 | 0.490234375 | [171, 579, 274] | 48 | [-5.727778, -5.391572, -2.201449, -1.315661, -1.206844] |
| calibration | 7 | 438 | 0.427734375 | [81, 558, 385] | 40 | [-20.037995, -5.152542, -2.409307, -1.330818, -1.206952] |

### Accepted construction level 2

| Bank | Walker | Candidate survivors | Survival fraction | M1/M2/M3 counts | Retained component switches | Log likelihood profile |
|---|---:|---:|---:|---|---:|---|
| selection | 0 | 401 | 0.391601562 | [177, 847, 0] | 1 | [-2.145011, -2.059993, -1.543705, -1.261394, -1.206672] |
| selection | 1 | 307 | 0.299804688 | [0, 796, 228] | 1 | [-2.144172, -2.110372, -1.667228, -1.332044, -1.323980] |
| selection | 2 | 536 | 0.523437500 | [872, 152, 0] | 3 | [-2.134768, -2.057991, -1.415207, -1.209080, -1.206620] |
| selection | 3 | 410 | 0.400390625 | [200, 824, 0] | 1 | [-2.145236, -2.096646, -1.528740, -1.323986, -1.206942] |
| selection | 4 | 397 | 0.387695312 | [95, 929, 0] | 1 | [-2.142753, -2.077969, -1.533689, -1.245725, -1.206963] |
| selection | 5 | 494 | 0.482421875 | [4, 1020, 0] | 1 | [-2.133022, -2.019402, -1.459355, -1.324491, -1.323962] |
| selection | 6 | 342 | 0.333984375 | [189, 835, 0] | 2 | [-2.146333, -2.088301, -1.578249, -1.313402, -1.206790] |
| selection | 7 | 126 | 0.123046875 | [0, 492, 532] | 1 | [-2.145913, -2.098990, -1.962942, -1.340026, -1.323967] |
| calibration | 0 | 371 | 0.362304688 | [0, 960, 64] | 2 | [-2.146078, -2.100262, -1.558231, -1.327595, -1.323961] |
| calibration | 1 | 491 | 0.479492188 | [848, 176, 0] | 3 | [-2.139330, -2.045361, -1.465626, -1.211750, -1.206620] |
| calibration | 2 | 398 | 0.388671875 | [0, 1024, 0] | 0 | [-2.146211, -2.068203, -1.509895, -1.327497, -1.323993] |
| calibration | 3 | 154 | 0.150390625 | [0, 387, 637] | 1 | [-2.147502, -2.119774, -1.973344, -1.337026, -1.324024] |
| calibration | 4 | 274 | 0.267578125 | [0, 795, 229] | 2 | [-2.142403, -2.111216, -1.785611, -1.328580, -1.323967] |
| calibration | 5 | 351 | 0.342773438 | [0, 950, 74] | 2 | [-2.145176, -2.096384, -1.556535, -1.327133, -1.323960] |
| calibration | 6 | 505 | 0.493164062 | [931, 93, 0] | 2 | [-2.147388, -2.024902, -1.456084, -1.208284, -1.206620] |
| calibration | 7 | 0 | 0.000000000 | [0, 0, 1024] | 0 | [-2.146468, -2.131802, -2.009990, -1.960669, -1.960412] |

### Accepted construction level 3

| Bank | Walker | Candidate survivors | Survival fraction | M1/M2/M3 counts | Retained component switches | Log likelihood profile |
|---|---:|---:|---:|---|---:|---|
| selection | 0 | 249 | 0.243164062 | [0, 1024, 0] | 0 | [-1.443420, -1.429089, -1.356268, -1.324204, -1.323960] |
| selection | 1 | 356 | 0.347656250 | [247, 777, 0] | 1 | [-1.443560, -1.431664, -1.342892, -1.214707, -1.206622] |
| selection | 2 | 232 | 0.226562500 | [0, 1024, 0] | 0 | [-1.443557, -1.432703, -1.356087, -1.324247, -1.323960] |
| selection | 3 | 734 | 0.716796875 | [1024, 0, 0] | 0 | [-1.441193, -1.410374, -1.261839, -1.207233, -1.206621] |
| selection | 4 | 711 | 0.694335938 | [1024, 0, 0] | 0 | [-1.442916, -1.423148, -1.270143, -1.207369, -1.206620] |
| selection | 5 | 250 | 0.244140625 | [0, 1024, 0] | 0 | [-1.443077, -1.436027, -1.352466, -1.324222, -1.323960] |
| selection | 6 | 226 | 0.220703125 | [0, 1024, 0] | 0 | [-1.443560, -1.432834, -1.356975, -1.324141, -1.323960] |
| selection | 7 | 254 | 0.248046875 | [0, 1024, 0] | 0 | [-1.443557, -1.421085, -1.346535, -1.324072, -1.323961] |
| calibration | 0 | 266 | 0.259765625 | [0, 1024, 0] | 0 | [-1.443520, -1.429177, -1.345321, -1.324175, -1.323960] |
| calibration | 1 | 262 | 0.255859375 | [0, 1024, 0] | 0 | [-1.442820, -1.429730, -1.347375, -1.324509, -1.323961] |
| calibration | 2 | 739 | 0.721679688 | [1024, 0, 0] | 0 | [-1.443388, -1.423274, -1.262716, -1.207798, -1.206620] |
| calibration | 3 | 757 | 0.739257812 | [1024, 0, 0] | 0 | [-1.442909, -1.405897, -1.264209, -1.206988, -1.206623] |
| calibration | 4 | 748 | 0.730468750 | [1024, 0, 0] | 0 | [-1.442024, -1.421931, -1.265339, -1.208180, -1.206636] |
| calibration | 5 | 748 | 0.730468750 | [1024, 0, 0] | 0 | [-1.443449, -1.415274, -1.260752, -1.206946, -1.206621] |
| calibration | 6 | 228 | 0.222656250 | [0, 1024, 0] | 0 | [-1.443318, -1.423362, -1.350735, -1.324431, -1.323961] |
| calibration | 7 | 762 | 0.744140625 | [1024, 0, 0] | 0 | [-1.442784, -1.410095, -1.265630, -1.206986, -1.206623] |

## Audit integrity

All saved run files, official reports and inspected frozen source files
retain their hashes, sizes and modification times from the audit start.
Only this new audit document is added to the repository. All candidate,
compression and provenance calculations are read-only reductions of saved
arrays/records; no sampling calls, trajectory regeneration, source changes,
staging or commits occurred.
