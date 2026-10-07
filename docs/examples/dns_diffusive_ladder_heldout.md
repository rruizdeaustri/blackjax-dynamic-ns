# Stage 9D: independent held-out diffusive-ladder validation

**FAILED-CONDITIONAL** — Frozen exact conditional-truth checks failed: {'KS': False, 'mean': False, 'second_moment': False, 'mode_probabilities': False, 'strict_membership': True, 'assigned_level': True, 'unique_events': True, 'exact_trace_events': True}

Design SHA256: `6749554b9649995174317303318315eb34cd58f3af42539d5b74bab83b880ad7`. The [preregistered design](dns_diffusive_ladder_heldout_design.json) and [checksum](dns_diffusive_ladder_heldout_design.sha256) precede sampling. This is one held-out trajectory plus, when eligible, its one exact-continuation check. No tuning or retry.

## Frozen methodology and analytic model

The committed Stage-9C ConditionalKernel, original DNS parameter/index kernels, strict membership, pooled-with-replacement promotion, compression estimator and ESS>=20 gate are unchanged. A read-only validation wrapper audits each returned bank before the frozen orchestrator can advance to calibration or the next attempt. It creates no transition and modifies no observation. Numeric budgets and tolerances are copied unchanged from Stage 9C, before evaluating this new three-mode model.

The prior is Uniform[-8,8]. Locations are [-3.8,-0.7,2.8], widths [0.27,0.39,0.32], and weights sigma/sum(sigma). Likelihood is max_k w_k NormalPDF(theta;mu_k,sigma_k), an equal-peak weighted-normal envelope, not a Gaussian sum. This model has not appeared in Stages 9A, 9C or 8H. Modes use fixed geometric regions bounded at -2.25 and 1.05.

At a log threshold ell below the common peak, A_ell is the clipped union of intervals mu_k +/- sigma_k sqrt(2(log_peak-ell)). Exact prior masses and CDFs are interval lengths; moments use polynomial integrals; mode probabilities use interval-region intersections. Exact posterior mode probabilities/evidence are computed by analytic Gaussian integrals on pairwise dominance cells, without reconstruction sampling.

For each attempt the accepted prefix 0..J has frozen log weights zero and log a_j=-log Xhat_j. In q(theta,j)=pi(theta) I[A_j]a_j/C, conditioning on assigned j=J cancels a_J/C and gives pi(theta)I[A_J]/X_J. Only those observations enter selection/calibration. Separate banks retain disjoint random namespaces, starts, promotion pools and histories; shared adaptive levels do not make finite samples IID.

The schedule remains eight walkers, six candidate levels, 32768 physical sweeps per bank, 1024 burn-in sweeps, no thinning, and the first 2048 post-burn assigned-top observations per walker. Rejection self-loops remain observations; no padding. Cross-walker exchange remains the Stage-9C identity transition.

The adapter's historical sign-based mode summaries remain saved. A separate read-only reduction of raw traces labels all three analytic regions and requires, at every disconnected generating contour and for both banks, at least five physical level-zero cross-mode returns, at least ten retained-top mode changes, and a strongly connected directed return graph across all three modes. Each witness records its actual theta-changing physical step with both adjacent assigned levels zero and its top-mode endpoints. Mode balance or ESS alone cannot certify this.

Inherited tolerances: KS<=0.08; each mode probability error<=0.07; mean error<=0.12 conditional SD; second-moment relative error<=0.12; selection/calibration compression error<=0.055; log-mass error<=0.30. Every analytic mode must remain represented. Failure is immediate at a bank boundary; no later trajectory is run to rescue it.

## Saved results

| Candidate | Threshold | Calibrated log X | Exact X | Bank | KS | Mode probabilities | ESS | Physical return witnesses |
|---|---:|---:|---:|---|---:|---|---:|---:|
| 1 | -5.698424514857786 | -1.0205057992077717 | 0.37954005951642455 | selection | 0.03322722456156946 | [0.3455810546875, 0.19024658203125, 0.46417236328125] | 3776.241305521124 | 0 |
| 1 | -5.698424514857786 | -1.0205057992077717 | 0.37954005951642455 | calibration | 0.02980274440043873 | [0.35467529296875, 0.18707275390625, 0.458251953125] | 3019.949604080593 | 0 |
| 2 | -1.5326205100193282 | -2.035438671723301 | 0.1379291995282975 | selection | 0.012058743303999642 | [0.26434326171875, 0.4036865234375, 0.33197021484375] | 14197.377842928612 | 4601 |
| 2 | -1.5326205100193282 | -2.035438671723301 | 0.1379291995282975 | calibration | 0.0059233462985298635 | [0.2755126953125, 0.40185546875, 0.3226318359375] | 14509.781361112844 | 4585 |
| 3 | -0.9827770262941096 | -3.028220584432101 | 0.050222370784479844 | selection | 0.04527985915492694 | [0.23260498046875, 0.4075927734375, 0.35980224609375] | 11243.379898618223 | 2160 |
| 3 | -0.9827770262941096 | -3.028220584432101 | 0.050222370784479844 | calibration | 0.019959844036614993 | [0.280517578125, 0.4100341796875, 0.3094482421875] | 16223.380152238924 | 2017 |
| 4 | -0.9101304544651906 | -4.048218468620065 | 0.018492736146886027 | selection | 0.013281935810161016 | [0.279296875, 0.4071044921875, 0.3135986328125] | 13621.106424763866 | 1236 |
| 4 | -0.9101304544651906 | -4.048218468620065 | 0.018492736146886027 | calibration | 0.045351923043195075 | [0.3206787109375, 0.33648681640625, 0.34283447265625] | 16383.999999999998 | 1221 |
| 5 | not accepted | not accepted | not accepted | selection | 0.018436235790322664 | [0.25775146484375, 0.42657470703125, 0.315673828125] | 14763.718574670898 | 797 |
| 5 | not accepted | not accepted | not accepted | calibration | 0.09116473409724596 | [0.1846923828125, 0.4378662109375, 0.37744140625] | 13329.007831709318 | 816 |

The [complete JSON](dns_diffusive_ladder_heldout.json) reports the complete threshold/log-mass/exact-mass table; exact posterior probabilities; every conditional check, compression estimate and truth error; survivor counts/CV/between-walker errors; per-level/per-walker occupancy, accepted up/down moves, physical mode changes, round trips and saved cross-mode witnesses. Raw states, event provenance, traces and full joint checkpoints are under `/tmp/dns-stage9d-heldout`.

Checks apply to generating levels J=0..5. The accepted final ell_6 is a new boundary; construction histories there were generated at ell_5. Its mass is checked against truth. No extra trajectory at the final boundary is authorized or silently treated as an ell_6 conditional stream.

## Restart, isolation and disposition

**Restart NOT PERFORMED.** Conditional failure stopped validation before restart eligibility; numerical reproducibility and child-process provenance are unassessed. The preregistered completed level-3 checkpoint and attempt-4 selection midpoint at sweep 8192 preserve theta, assigned index, caches, exact keys, weights, collection counters/ragged histories and both retained banks. Child PID/PPID, independent startup UUIDs, timestamps and parent launch receipt are recorded. Exact comparisons include candidates, ESS/check records, thresholds/masses/weights, retained arrays, full traces, RNG/collection bundles and checkpoint payloads. No retry.

Protected hashes before/after: True. No reconstruction import, export or invocation, LISA, termination evaluation or production occurred. Compression ESS can be high while walkers remain trapped in modes; survivor contributions do not prove ergodicity.

A VALIDATED result freezes this construction architecture for the tested analytic multimodal scope and authorizes one new independently designed end-to-end analytic validation combining diffusive construction, frozen Stage-7B reconstruction and frozen Stage-8R termination. That end-to-end run is not performed here. A FAILED-* result authorizes no termination or LISA follow-up.

## Immediate failure and frozen disposition

Validation stopped in candidate 5, generating level J=4, calibration bank. Four candidate levels were accepted. Candidate 5 was not appended, candidate 6 was not attempted, and no child process was launched. The saved level-3 checkpoint and selection midpoint were not used to repair or repeat the run.

| Failed conditional check | Observed discrepancy | Frozen tolerance |
|---|---:|---:|
| KS | 0.09116473 | 0.08 |
| Mean absolute error | 0.45946090 | 0.30747007 |
| Second-moment relative error | 0.13212490 | 0.12 |
| Maximum mode-probability absolute error | 0.09081782 | 0.07 |

The exact conditional mode probabilities were [0.2755102040816322, 0.39795918367346944, 0.3265306122448984], while the retained calibration frequencies were [0.1846923828125, 0.4378662109375, 0.37744140625]. Exact mean was -0.4112244897959155; observed mean was 0.048236413711541665. Exact second moment was 6.734233726283104; observed second moment was 5.844473738659277.

The failed bank nevertheless had compression ESS 13329.007831709318, 816 genuine level-zero cross-mode return witnesses, all three modes represented, and a strongly connected return graph. Minimum post-burn top visits were 5969 per walker. Its compression error was 0.0005139950490857803 and proposed (unaccepted) log-mass error was 0.056459483273219924. These diagnostics do not override the failed conditional CDF/moment/mode tests. The predeclared early retained observations were used exactly; later saved visits were not substituted to rescue the result.

This is **FAILED-CONDITIONAL**, an empirical conditional-sampling failure under the frozen protocol. The invariant-target identity remains unchanged; the result does not identify whether finite equilibration, correlation or another limitation caused the discrepancy. No alternate window, budget, weighting, proposal, mode definition, or seed was tried.

### Accepted ladder only

| Level | Log threshold | Calibrated log X | Calibrated X | Exact X |
|---|---:|---:|---:|---:|
| 0 | -inf | 0 | 1 | 1 |
| 1 | -5.698424514857786 | -1.02050579921 | 0.360412597656 | 0.379540059516 |
| 2 | -1.5326205100193282 | -2.03543867172 | 0.130623169243 | 0.137929199528 |
| 3 | -0.9827770262941096 | -3.02822058443 | 0.0484016882615 | 0.0502223707845 |
| 4 | -0.9101304544651906 | -4.04821846862 | 0.0174534408111 | 0.0184927361469 |

Failed candidate threshold (not frozen): `-0.9003104518727859`. Its analytic mass and proposed log-mass checks are reported as unaccepted diagnostics in JSON, not as an extension of the accepted prefix.

A saved-only audit verified 163840 retained events and 17433 physical excursion witnesses, the design/driver/protected hashes, and absence of reconstruction imports, a sixth attempt, or a restart launch. Raw output is retained under `/tmp/dns-stage9d-heldout`. The [validation harness](../../examples/dns_diffusive_ladder_heldout.py) is frozen by the preregistered design; all Stage-9C implementation files remain unchanged.

**Diffusive automatic construction is not validated by this held-out run.** Stage-9C CASE A remains a development result. This failure does not freeze a validated architecture or authorize the proposed end-to-end analytic validation. No termination validation, LISA or production follow-up is authorized by Stage 9D. STOP.
