# DNS inference reconstruction: Stage 7B

## Measure audit and derivation (recorded before implementation)

The audit covers `dns.py`, `dns_levels.py`, `dns_automatic.py`,
`dns_population.py`, and the Stage-7A automatic toy output and checkpoints.
Let `h(theta)=log L(theta)`, `A_j={h>ell_j}`, `A_0` be the full prior, and
`X_j=integral pi(theta) 1[A_j] dtheta`, with normalized prior density `pi`.

- An unconditioned prior draw has density `pi`; level zero also targets `pi`.
- A fixed-contour population at level j targets the product of constrained
  priors over its walkers; each stationary walker has marginal
  `p_j(theta)=pi(theta) 1[A_j]/X_j`. Slice and reciprocal exchange preserve
  this target. Promotion gives correlated empirical starts; finite burn-in
  does not prove stationarity. Retained samples have serial and population
  dependence and must not be treated as IID.
- `dns.py` can instead sample an explicit augmented diffusive target
  `q(theta,j)=pi(theta) 1[A_j] a_j/C`, where
  `a_j=exp(log_weight_j-log_mass_hat_j)` and `C=sum_j a_j X_j`.
  Its actual occupancy is `a_j X_j/C`, not necessarily the requested weights
  when masses are estimated. The marginal is `pi(theta) sum_j a_j 1[A_j]/C`.
  That kernel is **not** what the automatic construction outputs represent.
- At construction iteration j, both selection and calibration retained
  histories target the **old** contour `ell_j`. Selection chooses `ell_(j+1)`;
  calibration estimates its conditional exceedance, using its own samples.
  The checkpoint appends the new threshold but stores histories sampled at
  the preceding threshold. The final checkpoint's last level is therefore
  **not** the generating level of its stored histories.
- Selection data determine future thresholds. Calibration data determine
  future mass estimates and influence later promotion. Conditional on the
  final adaptive ladder, neither complete construction archive is a set of
  independent draws from preselected fixed proposals. Combining them naively
  in a retrospective deterministic-mixture estimator has no finite-sample
  unbiasedness justification. Initial states, promotion copies, burn-in, and
  candidate thresholds are not additional independent inference observations.

For this stage, both construction banks are used only for construction and
starting-state provenance. After freezing the automatic ladder, generate fresh,
separately seeded reconstruction trajectories at every fixed contour, including
level zero and the highest level. Initial populations are promoted from their
own saved calibration history; a fresh fixed burn-in follows. Conditional on
the frozen ladder, the **target** of reconstruction stratum j is `p_j`.
This is an invariant-measure assertion, not a claim that finite MCMC is IID or
exactly equilibrated. The fresh streams do not feed back into the ladder.

With fixed counts `n_j>0`, `N=sum_j n_j`, `alpha_j=n_j/N`, define the deterministic
mixture

```
q_mix(theta) = sum_j alpha_j p_j(theta)
             = pi(theta) S(theta),
S(theta) = sum_j alpha_j 1[A_j](theta) / X_j.
```

Level zero must be included: it gives support everywhere the prior does.
For all fresh retained draws `theta_ji`, use

```
r_ji = L(theta_ji) / S(theta_ji)
I_f_hat = (1/N) sum_(j,i) r_ji f(theta_ji)
Z_hat = I_1_hat
W_ji = r_ji / sum_(k,l) r_kl
E_hat[f | d] = sum_(j,i) W_ji f(theta_ji).
```

The importance ratio is `pi L/q_mix=L/S`; the normalized prior cancels, but
likelihood normalization constants **must** be retained. For fixed exact masses
and stationary marginal samples,

```
E[I_f_hat] = sum_j alpha_j integral p_j(theta) L(theta) f(theta)/S(theta) dtheta
           = integral pi(theta) L(theta) f(theta) dtheta.
```

Correlation changes variance, not this marginal-expectation identity. The ratio
posterior estimate has ordinary finite-sample self-normalization bias. Overlap is
handled by **all** eligible levels in `S`, not just a draw's originating level.
Different levels may be pooled with their actual fixed allocation fractions.
Each retained event appears once. Equal coordinates from rejection or promotion
are legitimate repeated events; repeating the same event ID is an error.

The implementable primary estimate substitutes recorded calibrated `X_hat_j`
for unknown `X_j`. This is a **plug-in estimator**. Conditioning on the recorded
masses does not make `pi 1[A_j]/X_hat_j` a normalized true proposal. Consequently
there is no claim of exact unbiasedness under mass error: conditional on those
numbers, the working denominator is fixed, but its discrepancy from the actual
proposal remains. Primary results use these recorded masses without fitting to
analytic truth. An exact-contour-mass diagnostic isolates that discrepancy on
the toys; a separately reported compression perturbation analysis assesses
sensitivity without changing samples, thresholds, or the estimator.

For an actual diffusive trajectory, the distinct ratio identity would be
`Z=E_q[L/sum_j a_j 1[A_j]] / E_q[1/sum_j a_j 1[A_j]]`, because the unknown `C`
cancels and the prior is normalized. Applying this ratio to automatic
fixed-allocation population histories would be incorrect. Stage 7B implements
the fixed-stratum estimator above, not a diffusive-trajectory estimator.

Both selection and calibration could contribute if they were newly generated
inference-only streams after the ladder was fixed, with explicit allocations and
no selection feedback; their shared contour would simply have combined `n_j`.
That is not the construction archive here. Only the fresh reconstruction stream
enters the authoritative estimator; no construction draw is double counted.

## Shell diagnostic, not the authoritative estimator

For `B_j=A_j minus A_(j+1)` and terminal `B_K=A_K`,

```
Z = sum_(j<K) (X_j-X_(j+1)) E_pi[L | B_j] + X_K E_pi[L | A_K].
```

The diagnostic obtains each representative shell likelihood as the arithmetic
mean of fresh level-j likelihoods that also satisfy `h<=ell_(j+1)`. The terminal
representative is the fresh top-level mean. Thus no arbitrary midpoint or
threshold likelihood stands in for a shell mean. It uses the recorded mass
differences, is noisy when shells are sparsely occupied, and is undefined for
an empty sampled shell. The top contribution is retained, not dropped. This
empirical-shell diagnostic is reported alongside MIS, including failures;
ordinary live-point shrinkage or trapezoid quadrature is not assumed.

## Predeclared validation design

The design was saved before reconstruction implementation or dynamic toy runs
at `/tmp/dns-stage7b-validation/design.json`, SHA-256
`4d7fbd0125837fb751b8ef5f314e4316802eb8259b9be7b92e9462d01f9ed82d`.

Toy A: `theta~N(0,1.5^2)`, observation `y=1`, likelihood `N(y|theta,0.7^2)`.
Toy B: `theta~N(0,2^2)`, likelihood
`0.65 N(theta|-2,0.45^2) + 0.35 N(theta|2.2,0.7^2)`.
The likelihood component probabilities are validated via posterior expectations
of analytic component responsibilities; the predefined regions `theta<0` and
`theta>=0` are checked separately.

Both models use latent standard-normal coordinates with the deterministic map
`theta=sigma_prior*u[0]`. The other 53 independent standard-normal coordinates
are nuisance variables that integrate to one, allowing the unchanged validated
54-coordinate population kernel. Prior and likelihood caches include their
normalizing constants. No LISA model is loaded.

Fixed builder seeds: 7101 / 7102; reconstruction seeds: 7201 / 7202.
Each toy constructs exactly three new levels through the Stage-7A builder,
with eight walkers, 256 burn-in and 1,024 retained sweeps, block size 64,
compression target `exp(-1)`, and ESS gate 20. Reconstruction at levels 0–3
uses fresh streams, eight walkers, 512 burn-in and 1,024 retained sweeps per
level. Counts are fixed in advance, not adapted to observed errors. Restart
validation repeats the reconstruction schedule after rebuilding the final
construction iteration from its completed checkpoint. No production stopping
criterion is introduced.

Predeclared final-estimate tolerances:

| Check | Absolute tolerance unless specified |
|---|---:|
| log Z, both toys | 0.22 |
| Relative Z error, both toys | 0.25 |
| Gaussian mean / variance | 0.18 / 0.16 |
| Gaussian quantiles / weighted-CDF discrepancy | 0.30 / 0.10 |
| Mixture mean / variance | 0.30 / 0.80 |
| Mixture quantiles / weighted-CDF discrepancy | 0.50 / 0.10 |
| Each mixture component and each sign-region probability | 0.08 |
| Normalized weight sum | 1e-12 |
| Restart inputs | Byte-exact |
| Restart inference outputs | Numerically identical |

Quantile probabilities are 0.05, 0.16, 0.50, 0.84, and 0.95. These conservative
finite-MCMC tolerances account for correlated populations and estimated mass
normalizers; raw count alone is not an IID precision guarantee. No threshold
will be adjusted after viewing the answers. All levels-0..k results are reported,
while the predeclared binary accuracy gate applies to the final k=3 estimator.

## Recording and generic API

`blackjax/ns/dns_reconstruction.py` implements the estimator for explicitly
labelled fixed-contour reconstruction records. Its main operations are
`reconstruct_evidence`, `posterior_weights`, `posterior_expectation`,
`posterior_quantiles`, `weighted_cdf_discrepancy`, and `shell_diagnostic`.
`validate_caches` accepts a caller-supplied evaluator and numerical contract;
it has no model-loading logic. The core contains no analytic toy truth.

`ReconstructionRecords` carries the complete threshold/calibrated-log-mass table,
positions, prior/likelihood caches, originating level, walker, retained time,
unique event ID, per-level target/stream metadata, mass calibration status,
contract, and checkpoint/RNG provenance. Likelihood caches must be finite under
the frozen contract. Level indices and strict support are validated. Missing
levels, uncalibrated candidate masses, nonmonotonic tables, negative shell mass,
wrong sampling-target labels, accidental duplicate events, and non-normalized
posterior weights are rejected. Repeated coordinates with distinct retained-event
identities remain valid MCMC observations.

The automatic builder already saves each successful attempt's retained histories
and frozen prefix. Its latest checkpoint alone does not contain all earlier
histories; the chain of attempt checkpoints does. No builder change is required
or made. Fresh inference receives its own `samples.npz` / `records.json` archive,
with SHA-256 checksums and a separate target label. The per-level inference
preflight fixes the old generating contour, source calibration-history hash,
promotion indices, and all RNG streams before sampling. Neither burn-in nor
promotion copies enter the inference sample count.

Restart validation reloads the completed automatic checkpoint after two levels,
finishes the third with the same schedule, and then repeats all four fresh
inference streams with the same recorded inputs. It compares thresholds, masses,
coordinates/caches, event metadata, posterior weights, log Z, and summaries.
The archive also supports a checksum-verified load without regenerating samples.

## Analytic references

For Toy A, with `v = (sigma_p^-2 + sigma_l^-2)^-1`,
`m = v*y/sigma_l^2`, the exact posterior is `N(m,v)` and the evidence is
`N(y|0,sigma_p^2+sigma_l^2)`.

For Toy B, each product of prior and likelihood component gives

```
z_k = c_k N(mu_k | 0, sigma_p^2+s_k^2)
Z = sum_k z_k
P(component=k | d) = z_k/Z
v_k = (sigma_p^-2+s_k^-2)^-1
m_k = v_k mu_k/s_k^2.
```

The exact mean is `sum P_k m_k`, variance is
`sum P_k (v_k+m_k^2) - mean^2`, and the exact CDF is
`sum P_k Phi((theta-m_k)/sqrt(v_k))`. Mixture quantiles are roots of that analytic
CDF, not reconstructions from samples. A separate high-accuracy quadrature check
of prior × likelihood agrees with these closed-form evidence/moment formulas in
unit tests. Reconstruction routines are not used to generate reference truths.

For the exact-mass diagnostic only, Gaussian contour masses are normal-CDF
interval differences. Mixture contour boundaries are found numerically from the
likelihood, and normal-CDF differences sum the enclosed prior probability. This
additional root calculation is independent of the evidence reconstruction and
is not supplied to the primary estimator.

## Correlation and mass uncertainty

Weight ESS is `1/sum W_i^2`; maximum weight and entropy are also reported. These
measure importance-weight concentration, not the number of independent MCMC
samples. Separate observable-specific MCMC diagnostics use contiguous time
blocks averaged over the whole population, retaining simultaneous exchange
dependence. The maximum of IID, population-block, and between-walker mean SE is
converted to a rough variance-equivalent ESS, capped by raw sample count.
Evidence, posterior-mean residual, mode-region residual, and log-likelihood
integrands are reported separately. ESS sums across levels are descriptive;
shared initialization and finite burn-in are additional limitations.

Mass sensitivity holds samples and thresholds fixed. It perturbs each recorded
compression on the logit scale using its conservative calibration SE, forms
strictly decreasing cumulative masses, and recomputes the same MIS estimator.
The 128 predeclared draws give sensitivity quantiles, **not** calibrated confidence
intervals: correlations among compression estimates are not modelled. Comparison
with true contour masses separates normalization error from residual sampling
and reconstruction error without fitting any parameter to the exact answer.

## Recovered validation results and final decision

**Stage 7B PASS.** Both toys pass the predeclared numerical validation; 419 previously completed regression tests pass. The single bounded fresh-process Gaussian checkpoint continuation also passes exact comparison with the saved continuous endpoint. No ladder was rebuilt and no reconstruction stream or test suite was rerun.

All Gaussian reconstruction levels 0–3 completed, as did all four repeated streams. No Gaussian reconstruction levels remain incomplete. The Gaussian ladder is the saved ladder with thresholds `[-inf, -1.694779510430953, -0.7081386045518494, -0.5815898827700305]`. The asymmetric multimodal ladder and both sets of reconstruction streams also completed.

The predeclared design file and hash remain unchanged. Results below are transcribed from saved artifacts. The authoritative estimator remains the overlapping-stratum multiple-importance estimator, including every supporting stratum in each sample’s proposal denominator.

### Primary estimates conditional on recorded calibrated masses

| Model | Exact log Z | Reconstructed log Z | Absolute logZ error | Relative Z error |
|---|---:|---:|---:|---:|
| gaussian | -1.605399245229 | -1.671771431805 | 0.066372187 | 0.064217486 |
| mixture | -2.145298421067 | -2.183665675496 | 0.038367254 | 0.037640555 |

| Model | Mean | Variance | q05 | q16 | Median | q84 | q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian exact | 0.821167883 | 0.402372263 | -0.222209154 | 0.190355206 | 0.821167883 | 1.451980560 | 1.864544920 |
| gaussian reconstructed | 0.816593529 | 0.410784055 | -0.244608841 | 0.188726105 | 0.816869662 | 1.463178543 | 1.859595679 |
| mixture exact | -0.634640838 | 3.565268452 | -2.537313643 | -2.216181318 | -1.614999881 | 1.981223473 | 2.638397538 |
| mixture reconstructed | -0.634693419 | 3.563041318 | -2.550310347 | -2.219970809 | -1.616062189 | 1.966349926 | 2.589302313 |

| Model | Weighted-CDF discrepancy | Weight ESS | Maximum normalized weight | Weight entropy |
|---|---:|---:|---:|---:|
| gaussian | 0.0118201787 | 20031.9394 | 0.000119066845 | 10.0960825 |
| mixture | 0.0104813426 | 18775.1096 | 0.0001259882 | 10.0639555 |

Mixture component probabilities are a primary accuracy criterion. Exact probabilities are `[0.671547842, 0.328452158]`; reconstructed probabilities are `[0.666883353, 0.333116647]`, with maximum absolute error `0.004664488` against the frozen `0.08` limit. Exact sign-region probabilities are `[0.672037778, 0.327962222]`; reconstructed values are `[0.667357219, 0.332642781]`, with maximum absolute error `0.004680559`. All saved final-estimate accuracy checks pass.

### Cumulative reconstruction

| Model | Included levels | log Z | Absolute logZ error | Relative Z error | Mean | Variance | CDF discrepancy | Weight ESS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 0 | -1.67114387 | 0.0657446231 | 0.063630039 | 0.805777251 | 0.415533127 | 0.0370379312 | 3770.12397 |
| gaussian | 0..1 | -1.67018835 | 0.0647891017 | 0.0627348899 | 0.820154802 | 0.411639984 | 0.0112515996 | 12207.0582 |
| gaussian | 0..2 | -1.67988472 | 0.0744854765 | 0.071779045 | 0.814495982 | 0.413627371 | 0.0156045786 | 18486.229 |
| gaussian | 0..3 | -1.67177143 | 0.0663721866 | 0.0642174862 | 0.816593529 | 0.410784055 | 0.0118201787 | 20031.9394 |
| mixture | 0 | -2.19325584 | 0.0479574202 | 0.0468256278 | -0.880839678 | 3.19122303 | 0.0703828141 | 2752.47562 |
| mixture | 0..1 | -2.17980311 | 0.034504689 | 0.0339161903 | -0.637076852 | 3.54857543 | 0.0122653786 | 10288.273 |
| mixture | 0..2 | -2.18721681 | 0.0419183879 | 0.0410519608 | -0.629193635 | 3.56726487 | 0.00908725559 | 17641.4822 |
| mixture | 0..3 | -2.18366568 | 0.0383672544 | 0.0376405548 | -0.634693419 | 3.56304132 | 0.0104813426 | 18775.1096 |

The JSON report retains every cumulative posterior summary, quantile, mode probability, error, and weight diagnostic without recomputation. The final levels-0..3 estimates alone receive the predeclared binary accuracy gate.

### Predeclared mass sensitivity and shell comparison

Primary results above condition on recorded calibrated masses. The saved 128 logit-normal compression perturbations (seed 7301) were already completed; no uncertainty model was added. Their logZ q05/median/q95 are:

- gaussian: `[-1.8234157217059455, -1.6739246119234847, -1.5509282134334137]`.

- mixture: `[-2.3826444568514202, -2.1720495505870323, -1.9961203240187657]`.

These are sensitivity quantiles, not confidence intervals. Posterior-mean and component sensitivity results, exact-contour-mass diagnostics, and correlation diagnostics remain fully recorded in the JSON.

Both predeclared shell diagnostics are defined, with accounted prior mass 1. Gaussian shell logZ is `-1.675152069913857`; mixture shell logZ is `-2.1829245556290284`. They are comparison diagnostics and do not replace the authoritative estimator.

### Bookkeeping, structural checks, and restart limitation

Fresh-process archive loading verified checksum integrity, identical continuous/restarted checkpoint payloads, identical thresholds and log masses, byte-identical reconstruction arrays, event IDs and origin levels, and identical per-level stream metadata. Each toy has 32,768 retained events (8,192 per level) and four distinct stream IDs. Record validation enforces unique events and level/walker/draw triples, finite caches, strict origin-contour support, X0=1, increasing thresholds, decreasing masses, and nonnegative shell masses. Saved posterior weights are finite, nonnegative and sum to one within 1e-12. Only fresh reconstruction records enter inference; adaptive construction banks provide starting provenance and are not silently pooled.

The fresh-process Gaussian continuation loaded
`gaussian/continuous/attempt_000001/checkpoint` and completed only the one
remaining iteration to the saved `attempt_000002` endpoint. Sampling occurred
exactly once. The initial comparison harness then failed by accessing a
nonexistent `DNSParticleState.logprior` attribute. The serialized prior cache
is `selection_logdensity` / `calibration_logdensity`. Correcting the read-only
comparison did not retry sampling or change restart semantics.

The completed comparison passes 42 exact checks: candidate and decision
records/hashes, promotion indices and RNG preflight, all retained selection
and calibration coordinates/prior caches/likelihood caches, final checkpoint
payload and checksum, reconstruction origin levels, stream IDs, event IDs,
and mixture metadata. Checkpoint hashing uses the existing sorted compact
JSON encoding. All already-written fresh-process files retain their SHA-256
hashes unchanged.

No new reconstruction archive was generated. Existing continuous and repeated
independent reconstruction records are reused under the byte-identical
fresh-process endpoint; thresholds, masses, and source-bank archive hash match
exactly. Applying the frozen estimator to these saved inputs reproduces the
saved log Z, normalized weights, posterior mean, variance, median, q05, q16,
q84 and q95 exactly. Analytic truth was not recalculated.

A further read-only verification completed 45 checks directly against the saved
artifacts, using identical array dtype, shape and bytes without new tolerances.
Both saved reconstruction archives produced identical inference results under
the frozen estimator. Artifact SHA-256 hashes, sizes and modification times
remained unchanged. The final checkpoint payload hash is
`1ecdcb2a1988ea39c740d02732553e558d0146ca340f63207d39338ff82ab2d9`.
This comparator verification is not a restart retry.

**Final decision: Stage 7B PASS.**

- Evidence reconstruction PASS.
- Posterior reconstruction PASS.
- Multimodal mode probabilities PASS.
- Fresh-process checkpoint continuation PASS.
- 419 previously completed tests PASS.

Read-only comparison evidence is saved at
`/tmp/dns-stage7b-validation/fresh_process_saved_comparison.json`. Original
fresh-process outputs remain immutable; their original harness-error record is
retained as provenance. No new tolerances, estimator changes, core changes,
LISA calls, termination implementation, staging, or commits occurred.

The complete machine-readable report is [dns_inference_reconstruction_validation.json](dns_inference_reconstruction_validation.json). Stop here.
