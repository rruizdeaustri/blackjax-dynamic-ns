# Stage 8: automatic DNS termination on analytic toys

## Derivation and scope

The frozen Stage-7B deterministic-mixture reconstruction uses
q(u) = prior(u) sum_i alpha_i 1[L(u)>L_i]/X_i and importance ratio
L(u)/sum_i alpha_i 1[L(u)>L_i]/X_i. Level zero has full prior support.
Consequently the terminal contour is already accounted for: adding a level
changes the proposal and finite-sample estimate, rather than adding a missing
shell. There is no deterministic bound on Monte Carlo error from X_j alone.

Let C_j={L>L_j}. With a global L_upper satisfying L(u)<=L_upper everywhere,
Z_cap=integral_Cj prior(u)L(u)du <= X_j L_upper. An observed maximum is a
lower bound on the supremum and is not accepted as its justification.
For these toys L_upper=sum_k weight_k/(sqrt(2*pi)*width_k). Each component
normal density is bounded by its peak; their positive weighted sum is bounded
by the sum of peaks. This is exact for the single Gaussian and conservative
for the asymmetric mixture, without locating or observing either mode.

Writing L_0=0, the layer-cake inequality gives
Z >= Z_lower=sum_{i=1}^j X_i (L_i-L_{i-1}). The generic criterion uses
rho=X_j_upper L_upper / sum_i X_i_lower (L_i-L_{i-1}), and stops when
rho<=epsilon. Also report delta_logZ=log(1+rho). If supplied mass envelopes
are true bounds, the cap contributes at most rho of total evidence.
Deleting that cap changes the posterior by total variation <=rho; bounded
observables have error <=rho times their range. This does not bound unbounded
means/variances, quantiles without a density condition, or reconstruction
noise. Those quantities require empirical checks below.

Calibrated X is a plug-in estimate, not a certified mass bound. Products of
calibration ratio +/- two recorded conservative SE (clipped to (0,1)) provide
a descriptive envelope, with no independence assumption, stochastic model,
or simultaneous confidence claim. This envelope drives the candidate criterion.
It is a conservative production *candidate*, not a proven production guarantee.
A construction gate failure or maximum-level exhaustion is FAIL, not termination.

## Predeclared protocol (before dynamic Stage-8 tests)

The immutable generic configuration uses epsilon=0.05, with diagnostic
fractions 0.20 and 0.005. Diagnostic decisions never alter sampling. These
values will not be tuned after observing results. Stage-7B model parameters,
seeds, construction budgets, ESS gate, reconstruction budgets, numerical
contract and analytic accuracy tolerances remain frozen.

Reuse the saved three-level Stage-7B endpoints and reconstruction records.
Continue one accepted automatic level at a time; add a fresh independent
fixed-contour inference stream at each new level. Reconstruct every prefix
with the frozen estimator. Stop automatically or fail at level 12. No
Stage-7B artifact is overwritten. This validates termination beginning from
an already completed Stage-7B ladder, rather than rebuilding either toy.

After stopping, continue exactly two more accepted levels as a separate
reference. Its predeclared maximum changes are: log Z 0.05, posterior CDF
sup distance 0.04, mode probability 0.04, mean 0.08, variance 0.15, and
any reported quantile 0.15. Exact-truth accuracy uses Stage-7B's already
predeclared tolerances verbatim. Both absolute evidence differences (truth
and deeper reference) and the independent exact terminal-contour integral
must be no greater than the descriptive cap estimate. Failure is reported
without retuning. Full per-level mean, variance, quantiles and component/
region probabilities are retained, including the secondary mixture mode.

A fresh interpreter reloads the three-level checkpoint and existing inference
records and independently continues to the stop, using identical immutable
configuration and random stream recipe. Arrays require identical dtype,
shape and bytes; decisions, metadata and summaries require canonical JSON
identity. A separate reload exactly at the stop must add no levels. Termination
configuration in the immutable design file and reconstruction-record sidecars
accompany existing checkpoints; the frozen checkpoint format and ladder
algorithm are unchanged. The wrapper stops scheduling construction calls; it
does not modify the ladder state or redefine construction-gate failures.

Run with CPU/x64 using `python -m examples.dns_automatic_termination --output
/tmp/dns-stage8-validation --predeclare`, then the same command without
`--predeclare`. The design is exclusively written before dynamic validation;
its hash is included in the report. No LISA model or production run is used.

## Frozen inference acceptance values

| Quantity | Gaussian | Asymmetric mixture |
|---|---:|---:|
| Absolute log Z error | 0.22 | 0.22 |
| Relative Z error | 0.25 | 0.25 |
| Absolute mean error | 0.18 | 0.30 |
| Absolute variance error | 0.16 | 0.80 |
| Maximum quantile error | 0.30 | 0.50 |
| Posterior CDF discrepancy | 0.10 | 0.10 |
| Component and region probability error | — | 0.08 |
| Weight normalization error | 1e-12 | 1e-12 |

These are the Stage-7B frozen values, copied before dynamic Stage-8 validation.
The independent deeper-reference limits and stopping fractions above were
also exclusively written before the run. The predeclared design SHA-256 is
`70a7f1acf763e76b6311e7683aaf3270e975f773c48928f52b2d3c777da4b898`.

## Results: Stage 8 FAIL

Both toys terminate automatically, meet all frozen inference accuracy limits,
and pass deeper-reference posterior stability and exact restart checks.
Both fail the required conservative comparison with total evidence error to
analytic truth. No tolerance was retuned and no second posterior criterion
was introduced.

| Toy | Stop level | X_stop | Threshold_stop | rho | Estimated cap evidence | Absolute Z error to truth |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 6 | 0.00256364254 | -0.562308625849 | 0.0286685165 | 0.00263121125 | 0.0131511662 |
| mixture | 7 | 0.000482216308 | -0.551223439988 | 0.0261649152 | 0.00107026379 | 0.00405588443 |

The primary tolerance is 0.05. The reported delta_logZ bound is 0.0282653 for
the Gaussian and 0.0258285 for the mixture. The first observed loose diagnostic
trigger is level 4 for the Gaussian and level 5 for the mixture; the tight
0.005 diagnostic first triggers at levels 8 and 9 respectively, on the
predeclared deeper references. Diagnostic decisions never drive the run.

| Toy | Exact log Z | Stopped log Z | Exact mean | Stopped mean | Exact variance | Stopped variance |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | -1.60539924523 | -1.67313305314 | 0.821167883212 | 0.816202950747 | 0.402372262774 | 0.411244571502 |
| mixture | -2.14529842107 | -2.18056905357 | -0.634640837592 | -0.638916182993 | 3.56526845228 | 3.55777448325 |

Quantiles are ordered q05, q16, median, q84, q95.

- gaussian: exact `[-0.222209154, 0.190355206, 0.821167883, 1.45198056, 1.86454492]`; stopped `[-0.244608841, 0.188470162, 0.816674709, 1.463178543, 1.859595679]`.
- mixture: exact `[-2.537313643, -2.216181318, -1.614999881, 1.981223473, 2.638397538]`; stopped `[-2.550310347, -2.218701971, -1.61652359, 1.960701953, 2.585796684]`.

Mixture component probabilities: exact `[0.671547841742549, 0.3284521582574509]`; stopped `[0.667913294233111, 0.3320867057669006]`. Both component and region probability checks pass. The saved strata retain 7891 nonnegative events, so secondary-mode inference is represented even though the deepest contour contains only the dominant mode. Per-origin counts and region probabilities are recorded in the JSON report.

## Deeper continuation and posterior stability

| Toy | Deeper level | Absolute change log Z | Posterior CDF distance | Component probability change | Mean change | Variance change | Maximum quantile change |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 8 | 8.32315405e-05 | 8.06837151e-05 | 6.66133815e-16 | 1.52618289e-05 | 3.14287588e-05 | 0 |
| mixture | 9 | 1.20889063e-05 | 2.09930578e-05 | 4.01454077e-06 | 1.64573199e-05 | 2.06051902e-05 | 0 |

All changes meet the predeclared deeper-reference limits. The JSON records
mean, variance, all five quantiles, component probabilities and region
probabilities at each tested level. Evidence-based termination passes the
posterior stability requirement on these two trajectories; a second
posterior stopping condition is not justified by these results.

## Remainder calibration and compression uncertainty

| Toy | Nominal X_stop | Exact contour mass | Descriptive upper X | Exact terminal integral | Estimated cap | Absolute Z change to deeper |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 0.00256364254 | 0.00282965757 | 0.00461682797 | 0.00161264728 | 0.00263121125 | 1.56197311e-05 |
| mixture | 0.000482216308 | 0.000479403403 | 0.00137970178 | 0.000276255301 | 0.00107026379 | 1.36577931e-06 |

The two-SE descriptive mass envelopes enclose the true stopping-contour mass
for both toys, and both uncertainty-expanded caps exceed the exact terminal
integral and the evidence change to the deeper reference. With plug-in X alone,
the Gaussian cap (0.00146106) falls below the actual terminal integral
(0.00161265). Including mass uncertainty matters even for this simple toy.
These SE envelopes remain descriptive; no confidence guarantee is claimed.

The decisive failure is different: the absolute evidence error to truth is
0.0131512 for the Gaussian and 0.00405588 for the mixture, greater than the
respective cap estimates 0.00263121 and 0.00107026. Further levels hardly
change the estimates; they do not remove existing calibrated-mass or
finite-sample reconstruction error. The mathematical cap bound remains
valid for the terminal integral under true mass bounds, but interpreting it
as a bound on the total reconstruction error fails this validation.

## Restart and failure checks

Both toys resumed in a fresh interpreter from the saved level-3 checkpoint
and continued to the identical stop. Identical dtype, shape and bytes were
required for thresholds, log masses, reconstruction coordinates, likelihoods,
priors, origin levels, walker/draw indices, retained-event IDs, normalized
weights and importance ratios. Metadata, stopping decisions, summaries,
final checkpoint payloads and hashes also match exactly. Retained construction
bank archives are byte-identical. Reloading at the stopping boundary adds
zero levels and reproduces the same decision and inference summary.

The focused suite passes **53 tests**, covering the new termination module
and frozen reconstruction unit tests. Cheap failures cover nonfinite metrics,
invalid X, negative remainder estimates, missing deepest-level samples,
missing or violated finite likelihood bounds, an unseen-high-peak synthetic
case, restart before stopping, an exact stopping boundary, immutable
configuration and checkpoint/record disagreement. The full regression suite
was not rerun.

Frozen source hashes and the predeclared design remain unchanged. No LISA
model calls, LISA level 13, production run, kernel/promotion/ladder/estimator/
record-format changes, ESS tuning, staging or commits occurred.

## Artifacts and API

- [Generic termination module](../../blackjax/ns/dns_termination.py): frozen
  `TerminationConfig`, `stopping_decision`, and validated remainder normalization.
- [Analytic validation example](../../examples/dns_automatic_termination.py):
  independent fixed-contour streams, automatic outer loop, deeper continuation,
  fresh-interpreter restart and read-only saved-result audit.
- [Cheap tests](../../tests/ns/test_dns_termination.py).
- [Full machine-readable validation](dns_automatic_termination_validation.json):
  exact truth, immutable design, inference errors, complete trajectories,
  cap calibration, mass envelopes, deeper comparisons and restart identity.

Sampling artifacts and the original runtime result are saved under
`/tmp/dns-stage8-validation/`. Stage-7B saved artifacts are reused read-only.

**Final decision: Stage 8 FAIL.** The candidate resolves the deepest contour
and exhibits stable toy posterior inference, but fails the requested
conservative total-evidence-error requirement. It is not approved as an
authoritative production termination guarantee.
