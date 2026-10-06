# Stage 8R: DNS termination semantics audit

**Stage 8 remains officially FAIL under its original predeclared criterion.**
This audit changes neither its implementation nor its decision. It uses only
saved Stage-8 numerical results and the frozen estimator source. No estimator
evaluation, analytic-truth calculation, trajectory, restart, test suite, or
LISA calculation is run for this audit.

Numerical evidence comes from
[dns_automatic_termination_validation.json](dns_automatic_termination_validation.json)
and the existing `/tmp/dns-stage8-validation/` artifacts. Mathematical
semantics are tied to
[dns_reconstruction.py](../../blackjax/ns/dns_reconstruction.py) and
[dns_termination.py](../../blackjax/ns/dns_termination.py).

## 1. What is reconstructed at level J?

Let pi(u) be the normalized prior, L(u) the nonnegative likelihood,
ell_j the likelihood threshold (the saved threshold is log ell_j), and
C_j={u:L(u)>ell_j}. Write ell_0=0, C_0 equal to prior support, and
X_j=integral_Cj pi(u)du. These X_j denote true masses; saved calibrated
masses are written Xhat_j. Define

\[
 Z=\int\pi L,\qquad T_J=\int_{C_J}\pi L,
 \qquad Z_{\mathrm{out},J}=Z-T_J.
\]

The independent fixed-contour proposal at level j is
q_j=pi 1_Cj/X_j. For allocation alpha_j=n_j/N, the actual ideal proposal
mixture and its calibrated counterpart are

\[
 S_J=\sum_{j=0}^J\alpha_j\,1_{C_j}/X_j,\quad
 \widehat S_J=\sum_{j=0}^J\alpha_j\,1_{C_j}/\widehat X_j,
 \quad q_J=\pi S_J.
\]

The implemented evidence estimator is exactly

\[
 \widehat Z_J=\frac1N\sum_{j=0}^J\sum_{t=1}^{n_j}
 \frac{L(U_{jt})}{\widehat S_J(U_{jt})}.
\]

It does not discard samples above ell_J. Because alpha_0>0 and X_0=1,
its proposal has full prior support at every included depth. With true
masses and exact stationary stratum sampling, its expected evidence is Z
for every J, including J=0. Adding a level changes proposal allocation,
importance weights, and sampling precision, rather than adding an omitted
piece of the target integral. Correlation changes variance; stationary
marginals suffice for this expectation identity and do not imply independent
events. Finite burn-in can leave nonstationary marginals.

## 2. Explicit error decomposition without double counting

Condition on the frozen ladder, calibration output, allocation and starting
provenance, denoted F. For any retained region R, put

\[
 Z_R=\int_R\pi L,\quad
 f_R(u)=1_R(u)L(u)/\widehat S_J(u),\quad
 \mu_R=\int q_J f_R,\quad
 m_R=E[\widehat Z_R\mid F].
\]

Here q_J denotes the ideal mixture of true constrained priors, whereas m_R
uses the actual finite-stream sampling law. The following telescoping
identity is exact whenever these quantities exist:

\[
 Z-\widehat Z_R=
 \underbrace{(Z-Z_R)}_{A:\ \mathrm{omitted\ target}}
 +\underbrace{(Z_R-\mu_R)}_{B:\ \mathrm{mass\ plug\!\!\!-in}}
 +\underbrace{(\mu_R-m_R)}_{E:\ \mathrm{sampling\ bias}}
 +\underbrace{(m_R-\widehat Z_R)}_{C/D:\ \mathrm{sampling\ fluctuation}}.
\]

- **A, terminal omission:** If R is the complement of C_J, A=T_J. For
  the actual Stage-7B full-support reconstruction, R is all prior support
  and A=0. A poorly sampled terminal contour is a precision issue, not
  automatically an omitted target region.
- **B, calibrated-mass error:** Explicitly,
  `Z_R-mu_R = integral_R pi L (1-S_J/Shat_J)`.
  This affects every region whose denominator depends on erroneous masses,
  including already reconstructed regions. It vanishes if Xhat=X.
- **C, finite reconstruction-sample error:** The realized centered
  fluctuation m_R-Zhat_R includes finite retained counts and dependence
  between events.
- **D, importance-weight Monte Carlo error:** This is part of C, not an
  independent additive error. The integrand being averaged is f_R, and
  its weight variability contributes to the same finite-sample fluctuation.
  Adding a separate D to C would double count it. Posterior expectations
  also use random self-normalization and therefore can have ratio bias.
- **E, possible estimator bias:** Nonstationary starts, inadequate mixing,
  or an incorrect constrained sampling law can make mu_R differ from m_R.
  Repeated-calibration plug-in bias is the unconditional average of B,
  not a second copy of B. Taking log Z or normalizing posterior weights
  can introduce additional nonlinear bias even when the evidence estimator
  itself is unbiased under oracle masses and stationary sampling.

For the actual estimator, therefore,

\[
 Z-\widehat Z_J=(Z-\mu_{\mathrm{all}})
 +(\mu_{\mathrm{all}}-E[\widehat Z_J\mid F])
 +(E[\widehat Z_J\mid F]-\widehat Z_J),\qquad A=0.
\]

These signed terms can cancel. Saved aggregate errors do not uniquely
identify how much came from B, C/D or E; this audit does not infer that split.
An equivalent realized identity uses an oracle-mass estimator on the same
samples: `Z-Zhat_plugin = (Z-Zhat_oracle)+(Zhat_oracle-Zhat_plugin)`.
No oracle estimator is evaluated here.

If L(u)<=L_upper globally and X_J<=X_J_upper, then

\[
 0\le T_J\le U_J:=X_J^{\mathrm{upper}}L_{\mathrm{upper}}.
\]

This controls A for an actually omitted terminal contour, or the target
mass of a contour treated as potentially unresolved. It does not bound B,
C/D or E outside that contour, nor their total. An observed likelihood
maximum is not a certified global upper bound. The saved toys used the
justified bound `sum_k c_k/(sqrt(2*pi)*s_k)` on their Gaussian-component
likelihood densities. The saved two-SE mass envelopes are descriptive,
not simultaneous probabilistic confidence bounds.

## 3. Termination error versus estimator error

Define the hypothetical truncation functional

\[
 \Delta_{\mathrm{tail}}(J):=Z-Z_{\mathrm{out},J}=T_J.
\]

It measures what would be lost by omitting C_J entirely. Calling it a
resolution budget for an unresolved contour is also legitimate, provided
it is not described as evidence actually omitted by this implementation.
For the ideal full-support DNS estimator, stopping at finite J changes no
target integral: its ideal target bias due to truncation is zero.

For a specific saved continuation to K>J, define instead the observable

\[
 d_Z(J,K)=|\widehat Z_K-\widehat Z_J|,\qquad
 d_{\log Z}(J,K)=|\log\widehat Z_K-\log\widehat Z_J|.
\]

These measure practical sensitivity to continuing that experiment. They
include changed mixture denominators outside C_J, additional samples and
mass calibration effects; they are not equal to T_J. No deterministic
inequality `d_Z<=U_J` follows merely from the terminal-integral bound.
The saved deeper reference is a finite comparison, not an exact
infinite-depth counterfactual, and shared prefix records make it correlated
with the stopped estimate. Even perfect stability can coexist with a
shared error relative to truth.

Thus “error caused by stopping” has two precise uses: an omitted target
integral, if truncation is performed; or empirical sensitivity to foregoing
further proposal refinement, when full-support reconstruction is retained.
The present implementation has the second behavior. These uses must not
be silently interchanged.

## 4. Gaussian: saved level 6 versus saved level 8

| Quantity | Saved value |
|---|---:|
| Stop level / reference level | 6 / 8 |
| Calibrated X_stop | 0.0025636425354013657 |
| Absolute log Z error to analytic truth | 0.0677338079131502 |
| Absolute Z error to analytic truth | 0.0131511661895381 |
| Uncertainty-expanded terminal-cap estimate U_J | 0.0026312112537538635 |
| Recorded actual terminal integral T_J | 0.001612647283574832 |
| Absolute Z change after deeper continuation | 0.000015619731144694837 |
| Absolute log Z change after deeper continuation | 0.00008323154054679094 |
| Absolute posterior mean change | 0.000015261828865242855 |
| Absolute posterior variance change | 0.000031428758758389463 |
| Posterior CDF sup-distance | 0.00008068371513880734 |
| q05, q16, median, q84, q95 changes | 0, 0, 0, 0, 0 |

The actual terminal integral and the observed evidence continuation effect
are both below the cap, while total evidence error is above it. The cap
is about 168 times the observed absolute Z continuation change. The saved
Stage-8 metric rho=U_J/D_J was 0.0286685165264942, and its reported
`log(1+rho)` was 0.028265263579564746; these are dimensionless quantities,
whereas U_J is in evidence units and cannot directly be compared with a
log Z difference.

**Answer:** The cap conservatively covers the observed evidence effect of
the predeclared level-6-to-level-8 continuation, and posterior changes are
practically small. It is not a theorem about every possible future
reconstruction run or an attribution of all the observed change to C_6.
Unchanged empirical quantiles mean the same retained order statistics were
selected, not that their uncertainty is zero.

## 5. Asymmetric mixture: saved level 7 versus saved level 9

| Quantity | Saved value |
|---|---:|
| Stop level / reference level | 7 / 9 |
| Calibrated X_stop | 0.0004822163078971016 |
| Absolute log Z error to analytic truth | 0.03527063249994056 |
| Absolute Z error to analytic truth | 0.004055884434416693 |
| Uncertainty-expanded terminal-cap estimate U_J | 0.0010702637853929567 |
| Recorded actual terminal integral T_J | 0.00027625530098043024 |
| Absolute Z change after deeper continuation | 0.0000013657793103277749 |
| Absolute log Z change after deeper continuation | 0.000012088906309770664 |
| Absolute posterior mean change | 0.000016457319869989107 |
| Absolute posterior variance change | 0.00002060519021851448 |
| Maximum component probability change | 0.0000040145407739222705 |
| Maximum region probability change | 0.000004008830040702627 |
| Posterior CDF sup-distance | 0.000020993057814389093 |
| q05, q16, median, q84, q95 changes | 0, 0, 0, 0, 0 |

Stopped component probabilities are
`[0.667913294233111, 0.3320867057669006]`; the deeper probabilities are
`[0.6679173087738633, 0.33208269122612666]`. The saved exact component
probabilities are `[0.671547841742549, 0.3284521582574509]`. Component and
region probabilities passed the frozen Stage-7B criteria. Lower proposal
strata retain secondary-mode support even when the deepest contour excludes
that mode.

**Answer:** The observed evidence continuation effect is smaller than the
cap by a factor of about 784. Mean, variance, quantiles and mode probabilities
also remain stable. The saved metric rho was 0.026164915197649026, with
reported `log(1+rho)=0.02582846989200815`. Coverage of this particular
continuation effect is empirical, not a general bound on proposal-change
Monte Carlo noise.

For both toys the saved mass envelope encloses the true stopping-contour
mass. The Gaussian nominal plug-in cap, 0.0014610648560101872, did not
cover its actual terminal integral; the uncertainty-expanded cap did.
This distinction must be preserved in a future protocol.

## 6. Ordinary nested sampling and the DNS mapping

For ordinary shell-quadrature nested sampling, a partial evidence sum
accounts for removed prior shells. If the retained prior region has mass X,
its unaccounted evidence is `integral_retained pi L`, bounded by
`X L_upper` when a valid global upper likelihood bound is available.
Using the maximum observed live likelihood as that scale is a heuristic
unless it is independently known to bound the entire remaining region.
Even a rigorous remainder bound does not control errors in shell weights
or quadrature contributions that have already been accumulated.

In Stage-7B DNS reconstruction the deepest region is included in the
importance-sampling target and proposal support. The analogous quantity is
its terminal integral T_J, not the whole residual `Z-Zhat_J`. A small T_J
can justify a contour-resolution budget or hypothetical truncation; it
cannot alone certify the accuracy of the full importance estimator.

**It is intended to bound remaining unresolved target evidence, not the
entire accumulated Monte Carlo error of the evidence estimate.** In this
DNS implementation “remaining” describes a resolution budget, because
that region is already accounted for in expectation.

## 7. Independent validation responsibilities and failure classification

**Reconstruction validity:** The estimator, calibrated masses and actual
sampling must give sufficiently accurate evidence and posterior inference.
Stage 7B validated this on its declared toy designs and tolerances; it did
not prove accuracy for every future model or budget.

**Termination validity:** Stopping further ladder refinement must respect
a predeclared terminal-resolution budget, and held-out continuation must
show sufficiently small practical inference changes. This requires its
own validation. These are separate acceptance responsibilities, not a
claim of statistical independence: mass calibration affects both.

One terminal-integral bound cannot certify both responsibilities because
B, C/D and E persist outside the terminal contour. Conversely, a tiny
stopped/deeper difference does not certify reconstruction accuracy when
the estimates share calibration and retained data.

**Primary classification: validation-target mismatch.** The original
Stage-8 test required a terminal-cap estimate to cover total evidence error
to analytic truth. It failed that test for both toys, and therefore remains
FAIL. However, its uncertainty-expanded caps covered the actual terminal
integrals and the observed continuation changes. Saved log Z errors of
0.0677338 and 0.0352706 coexist with continuation changes of only
0.0000832315 and 0.0000120889. All recorded posterior-stability and restart
checks pass. These observations support target mismatch rather than a
demonstrated premature-stop failure on these particular trajectories.
They do not establish universal correctness of a stopping rule or identify
the exact source of the remaining estimator error.

## 8. Exactly one corrected candidate for future validation

The single proposed candidate is a **terminal-cap fraction relative to a
lower bound on already resolved evidence**. No second posterior stopping
condition is proposed. Posterior stability remains a validation requirement.

Require a justified global L_upper and mass envelopes
`X_i_lower <= X_i <= X_i_upper`. Set

\[
 U_J=X_J^{\mathrm{upper}}L_{\mathrm{upper}},\qquad
 D_J=\sum_{i=1}^J X_i^{\mathrm{lower}}(\ell_i-\ell_{i-1}),
 \qquad B_J=D_J-U_J.
\]

The layer-cake bound gives `D_J<=Z`. Since `T_J<=U_J`,
`Z_out,J=Z-T_J>=D_J-U_J=B_J`. If B_J<=0, do not stop. Otherwise

\[
 \rho_{\mathrm{tail}}(J)=U_J/B_J,\qquad
 \boxed{\mathrm{stop\ when}\quad \rho_{\mathrm{tail}}(J)\le\epsilon_{\mathrm{stop}}.}
\]

This is one criterion with a positive-normalization prerequisite. It bounds
`T_J/Z_out,J`, and therefore gives the actual truncation statements

\[
 \log Z-\log Z_{\mathrm{out},J}
 \le\log(1+\rho_{\mathrm{tail}}),\qquad
 \mathrm{TV}(P,P_{\mathrm{out},J})
 \le\rho_{\mathrm{tail}}/(1+\rho_{\mathrm{tail}}).
\]

The second result follows because deleting a region changes the posterior
by its posterior probability `T_J/Z`. Bounded observables inherit a bound
using their range. Means and variances on unbounded domains, and quantiles
without additional density assumptions, still require empirical checks.

This normalization matters: Stage 8 used D_J, which bounds total evidence,
not necessarily resolved evidence. In general `log(1+U_J/D_J)` is not a
bound on the log error caused by deleting C_J. Using B_J supplies that
missing distinction. This audit neither evaluates the new criterion on
the observed Stage-8 toys nor changes their existing tolerance or decision.

With descriptive two-SE envelopes, the formula is a candidate diagnostic
whose mathematical bounds are conditional on envelope validity. They are
not automatically certified confidence bounds. A rigorous production
claim would require justified mass coverage as well as the global
likelihood bound. The candidate is not represented as a theorem bounding
`|Zhat_K-Zhat_J|` or total estimator error.

## 9. One held-out validation: protocol frozen here, not executed

**Exactly one held-out toy validation is justified and authorized for a
future execution. No execution is authorized within this read-only audit.**
It comprises one model, its continuous branch, one deeper reference and
one fresh-process restart branch. No alternative toy, seed search, trajectory
retry, tolerance tuning or fallback validation is included. A failed gate,
invalid bound, numerical ambiguity or exhausted budget is FAIL.

This is an empirical held-out test of the conditional candidate, not a
production certification of mass uncertainty. Neither of the observed
Stage-8 toy runs can independently validate it.

### Model and fixed execution design

| Item | Frozen value |
|---|---|
| Identifier | Stage-8R-heldout-trimodal-1 |
| Prior | u distributed N(0,I_54); theta=2.5*u[0] |
| Likelihood | sum_k c_k NormalPDF(theta; mu_k, s_k) |
| Component weights c | [0.30, 0.40, 0.30] |
| Component centers mu | [-3.0, 0.2, 3.4] |
| Component widths s | [0.40, 0.60, 0.85] |
| Global L_upper | sum_k c_k/(sqrt(2*pi)*s_k), fixed analytically |
| Construction seed | 81301 |
| Reconstruction seed | 81302 |
| RNG recipes | Existing frozen SeedSequence-v1 schedules, PCG64 and threefry2x32 |
| Walkers | 8 in each construction bank and each reconstruction stratum |
| Initial bank histories | 16 independent prior states per walker in each bank; existing initial-state SeedSequence construction |
| Construction budget per attempt | burn-in 256; retained 1024; block size 64 |
| Frozen construction gates | target compression exp(-1); minimum ESS 20 |
| Reconstruction budget per included level | burn-in 512; retained 1024; block size 64 |
| Maximum automatic stopping level | 12 accepted levels beyond level zero |
| Deeper reference | Exactly 3 additional accepted levels after automatic stop |
| Restart point | Checkpoint after first accepted level, with saved reconstruction prefix |
| Numerical environment | Frozen CPU/float64/x64; JAX 0.10.0, jaxlib 0.10.0, NumPy 2.4.4; same interpreter/environment as Stage 8 |
| Kernel/promotion/estimator | Frozen Stage-7B implementations; only this new analytic likelihood callback |
| Allocation | Equal retained counts at every included fixed-contour level, including level zero |
| Mass envelope | Products of clip(r_i +/- 2*SE_i, 1e-12, 1-1e-12), using recorded calibration SE |
| Decision | The single rho_tail=U/(D-U) criterion above; B<=0 means continue |
| epsilon_stop | 0.01, a predeclared 1% terminal-to-resolved-evidence budget |
| Allowed absolute continuation log Z change | log(1.01) |

All existing kernel settings and source versions are pinned to the Stage-8
design with SHA-256
`70a7f1acf763e76b6311e7683aaf3270e975f773c48928f52b2d3c777da4b898`;
the new model parameters above are the only likelihood change. The two
construction banks remain independent selection and calibration banks.

The 1% budget is an explicit future accuracy policy, not a fit to Stage-8
stop levels, errors or deeper-reference outcomes. It is frozen before this
held-out model is evaluated. It does not alter Stage 8's 5% criterion.

Reconstruction starts at level zero. At each accepted level the wrapper
appends a fresh fixed-contour reconstruction stream, invokes the frozen
estimator and evaluates the candidate using only current information.
The saved initial state, stream keys, configuration and reconstruction
prefix must be portable to the restart branch. Since B_1<=0 for valid
bounds, an accepted stop cannot precede the prescribed level-1 restart
checkpoint. The deeper reference is generated only after stopping and
must not influence the stop decision.

Analytic normal-product identities provide exact evidence, posterior
component probabilities, mean and variance; deterministic CDF inversion
provides quantiles. These references are diagnostics only. Origin-contour
mass and terminal-integral calculations use independent deterministic
contour bracketing/integration, with verified absolute numerical uncertainty
at most 1e-10 in each mass or evidence integral. If that uncertainty cannot
be justified, the experiment cannot PASS. No exact mass or reference value
is fed back into the stopping decision or reconstruction estimator.

### Predeclared PASS/FAIL responsibilities

All following conditions are required; there is no criterion selection
based on the held-out outcome.

1. **Automatic behavior:** Stop by level 12 without manual intervention,
   gate changes or repeated attempts. Retain all required deepest-level
   samples and valid origin, stream and event metadata. Complete all three
   deeper levels under the same frozen gates.
2. **Reconstruction validity, separately:** Stopped absolute log Z error
   <=0.22, relative Z error <=0.25, mean error <=0.25, variance error
   <=0.625, maximum q05/q16/median/q84/q95 error <=0.50, CDF discrepancy
   <=0.10, and maximum component and region probability error <=0.08.
   These are independent acceptance gates; none must be bounded by U_J.
   Region boundaries are the fixed center midpoints -1.4 and 1.8. Weights
   must be finite, nonnegative and normalized within 1e-12.
3. **Continuation evidence:** Absolute stopped/deeper log Z difference
   <=log(1.01), and absolute stopped/deeper Z difference <=U_J evaluated
   at the original stop. The latter is an empirical coverage test,
   explicitly not an implication of the cap theorem for noisy estimators.
4. **Posterior stability:** Absolute stopped/deeper mean change <=0.05,
   variance change <=0.125, every reported quantile change <=0.10,
   weighted posterior CDF sup-distance <=0.02, and maximum component
   and region probability change <=0.02. These fixed scales correspond
   to 0.02 prior SD for mean, 0.02 prior variance for variance and 0.04
   prior SD for quantiles. They do not drive a second stop decision.
5. **Tail and normalization validity:** Independent terminal-integral
   upper uncertainty endpoint <=U_J. Independently computed contour-mass
   uncertainty intervals at all levels entering D_J and U_J must be
   contained within their descriptive mass envelopes. Their use as true
   mass bounds is checked on this toy, not assumed statistically universal.
   Validate positive B_J and the resolved-evidence lower bound against
   independent deterministic references with their stated uncertainties.
6. **Restart identity:** A fresh interpreter reloads the level-1 checkpoint
   and prefix records and reaches identical stopping level, thresholds,
   calibrated log masses, decision, retained construction banks,
   reconstruction coordinates/caches/origin and event metadata, weights
   and summaries. Require equal dtype, shape and bytes for arrays and
   canonical identity for metadata and decisions, with no new numerical
   tolerance. Loading a checkpoint exactly at stop must request no new
   construction or reconstruction samples.

Record the full protocol and its checksum before execution, including
source hashes and environment, and preserve every branch artifact. One
failed requirement yields held-out FAIL with the first failure reported;
no reseeding, loosened tolerance or second held-out model is authorized.
Even a PASS would establish evidence only for this single held-out design,
not authorize production, LISA, or modification of frozen core code.

## 10. Explicit answers and disposition

1. **What should termination bound?** The unresolved terminal-contour
   evidence T_J, or its fraction relative to resolved evidence. It should
   not purport to bound pre-existing calibration, sampling or estimator error.
2. **What primarily failed in Stage 8?** The validation target: it tested
   a terminal-integral bound against total evidence error. Stage 8 stays
   FAIL under that original test; the recorded trajectories do not show a
   genuine premature-stop failure.
3. **Was stopping already practically stable?** Yes, for both saved
   predeclared deeper comparisons. Log Z changes were about 8.32e-5 and
   1.21e-5, with small posterior and mode-probability changes and exact
   restart identity. This is finite empirical evidence, not a universal guarantee.
4. **What single corrected criterion should be validated?** Stop when
   `U_J/(D_J-U_J)<=epsilon_stop`, requiring positive denominator, a justified
   global likelihood bound, and valid mass bounds. It is a terminal-resolution
   candidate; posterior stability and reconstruction accuracy are separate
   acceptance checks.
5. **Is exactly one held-out validation justified?** Yes: only the newly
   predeclared trimodal model and protocol above, in a future execution.
   No trajectories were run for Stage 8R.

Only this audit document is added. Stage-8 results, implementation and
existing tolerances remain unchanged. No LISA, production, LISA level 13,
new numerical trajectory, estimator change, staging or commit is performed.
