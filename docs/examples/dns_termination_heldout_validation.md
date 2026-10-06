# Frozen Stage-8R held-out termination validation

**Decision: FAIL.** The one authorized trimodal validation failed the
unchanged automatic selection ESS gate while proposing level 4. Three
levels were accepted, but the stopping criterion never triggered. This is
an automatic-construction information failure before termination could be
validated, rather than the historical Stage-8 total-error/tail-target mismatch.

Historical **Stage 8 remains FAIL**. Its result and the
[Stage-8R protocol](dns_termination_semantics_audit.md) were not modified.

## Frozen design and held-out verification

The machine-readable design was exclusively saved **before sampling** at
`/tmp/dns-termination-heldout-validation/design.json`, with checksum file
`design.sha256`. Its SHA-256 is
`2796a941c82c2c424b89f6fc228f45bd237f8342d8d1d8c6232e6b00055104d7`.

The [full validation JSON](dns_termination_heldout_validation.json) embeds
the design, verbatim protocol, protocol hash, source hashes, harness hashes,
environment, exact truth, failure evidence and saved partial diagnostics.
The protocol was complete; no missing scientific field was invented.

Before execution, historical Stage-7A/7B/8 JSON artifacts and repository
saved example results were checked for this exact model and identifier.
There were no prior model matches. Source inspection found the identifier
and parameter set only in the predeclared audit, which explicitly recorded
that this model had not been evaluated. The historical file list is retained
in the design. The model was not used to derive or tune the criterion,
1% stopping budget, sampling budgets, deeper depth or acceptance thresholds.
No held-out outcome was inspected before those choices were frozen.

| Field | Frozen value |
|---|---|
| Identifier | Stage-8R-heldout-trimodal-1 |
| Prior | u~N(0,I_54), theta=2.5*u[0] |
| Likelihood | sum_k c_k NormalPDF(theta; mu_k,s_k) |
| Component weights | [0.30, 0.40, 0.30] |
| Centers | [-3.0, 0.2, 3.4] |
| Widths | [0.40, 0.60, 0.85] |
| Construction / reconstruction seeds | 81301 / 81302 |
| Walkers | 8 per bank and reconstruction stratum |
| Initial bank history | 16 independent prior states per walker |
| Construction burn-in / retained / block size | 256 / 1024 / 64 |
| Reconstruction burn-in / retained / block size | 512 / 1024 / 64 |
| Target compression / ESS gate | exp(-1) / 20 |
| Allocation | Equal retained counts at every level, including level zero |
| epsilon_stop | 0.01 |
| Safety maximum | 12 accepted levels beyond level zero |
| Deeper reference | Exactly 3 additional accepted levels after automatic stop |
| Restart point | Checkpoint after first accepted level and its saved reconstruction prefix |
| RNG | SeedSequence-v1, PCG64, threefry2x32 |
| Environment | CPU/float64/x64; JAX 0.10.0, jaxlib 0.10.0, NumPy 2.4.4 |

The frozen population kernel, pooled-with-replacement promotion, ladder,
ESS gate, numerical contract, reconstruction estimator and record format
were used without modification. An external, checksum-pinned validation
wrapper implements the Stage-8R decision without changing Stage-8 code.

## Criterion used without substitution

For likelihood thresholds ell_i, with ell_0=0, the frozen definitions are

```
X_i_lower = product_{k<=i} clip(r_k - 2*SE_k, 1e-12, 1-1e-12)
X_i_upper = product_{k<=i} clip(r_k + 2*SE_k, 1e-12, 1-1e-12)
X_0_lower = X_0_upper = 1
L_upper = sum_k c_k/(sqrt(2*pi)*s_k)
U_J = X_J_upper * L_upper
D_J = sum_{i=1..J} X_i_lower * (ell_i - ell_(i-1))
B_J = D_J - U_J
stop iff B_J > 0 and U_J/B_J <= 0.01
```

The global likelihood bound was fixed analytically and conservatively
rounded upward using independent interval arithmetic:
`L_upper=0.7059713883574373`. No observed maximum was substituted.
Mass envelopes use the recorded conservative calibration SE and remain
descriptive, not universal confidence bounds. Independent references are
never supplied to the stopping decision or frozen estimator.

The predeclared continuation evidence limits were
`abs(delta log Z)<=log(1.01)` and `abs(delta Z)<=U_J` at the original stop.
The bound was not required to cover total analytic-truth estimator error.

## First failure and automatic execution

One continuous run produced accepted levels 1, 2 and 3. While requesting
level 4 (zero-based construction attempt 3), the frozen selection gate
returned:

| Field | Value |
|---|---:|
| Candidate selection ESS | 12.0093475208749 |
| Required selection ESS | 20 |
| Candidate status | insufficient_tail_information |
| Construction stop reason | selection_gate |
| Candidate log likelihood threshold | -1.2715287082049687 |
| Retained selection events | 8192 |
| Selection sweeps completed | 1280 |
| Constituent numerical checks | PASS |

The candidate walker exceedance fractions were
`[0, 0, 0.7578125, 0, 0.7138671875, 0.7060546875, 0, 0.763671875]`.
Their marked between-walker variation explains the low reported selection
information; this is not an integrity-check exception or a comparator
failure. The candidate was not accepted or independently calibrated.
No level-4 checkpoint or level-4 inference stratum is claimed.

Evidence is preserved in
`continuous/ladder/attempt_000003/candidate.json`, `candidate_hash.json`
and `stop.json`. The predeclared protocol explicitly counts a failed
construction gate as FAIL. There was no manual intervention between levels,
ESS tuning, reseeding or retry.

## No automatic stopping state: last accepted state only

There is **no stopping level J**. These quantities describe the last
accepted level and must not be mistaken for triggered termination.

| Quantity at last accepted level 3 | Value |
|---|---:|
| ell_3 | 0.2644946191845924 |
| log ell_3 | -1.3299343712381886 |
| log X_3 | -2.695101406702748 |
| X_3 | 0.06753553287126123 |
| U_3 | 0.09590719912457871 |
| D_3 | 0.05038250568540154 |
| D_3-U_3 | -0.045524693439177165 |
| Stopping ratio | Undefined: denominator nonpositive |
| epsilon_stop | 0.01 |
| Criterion decision | CONTINUE |
| Descriptive lower X_3 | 0.025432889097455516 |
| Descriptive upper X_3 | 0.13585139668014468 |

The precise reason was `D_J-U_J is not positive; continue`. Construction
failure prevented further progress and was not relabelled as termination.
The complete log envelopes, calibrated masses, calibration ratios and SE
values are retained in the JSON report.

Independent interval references for the saved levels have uncertainty
at most 4.19e-13, below the required 1e-10. Each computed mass interval
lies within its descriptive envelope. At level 3, the terminal integral
interval is `[0.014420797502996055, 0.014420797503233619]`, below U_3.
The references use 90-digit outward-rounded Decimal arithmetic, a bounded
Machin series for pi, an explicit normal-CDF series remainder or Mills
bound, and interval contour subdivision including unresolved boxes and
exterior prior tails. This is a partial contour check, not a successful
termination or deeper-reference test.

## Saved partial inference and analytic truth

There is no inference at an automatic stop. For completeness, the
**last accepted level-3** reconstruction contains 32,768 retained events,
8,192 at each of levels 0..3. These diagnostics were recovered from saved
records after failure using the unchanged estimator, without new sampling.

| Quantity | Exact analytic value | Saved level-3 reconstruction |
|---|---:|---:|
| log Z | -2.252920204213016 | -2.2657167656185404 |
| Posterior mean | 0.03303576424897807 | -0.24283206151706832 |
| Posterior variance | 4.033058922117459 | 3.54645121895725 |

Absolute log Z error: `0.012796561405524542`.
Relative Z error: `0.012715033542914908`.
Weighted CDF discrepancy: `0.06406782135802108`.
Importance-weight ESS: `19395.367571944596`; this differs from the
construction candidate ESS that failed.
Weights are finite and nonnegative, with sum `0.9999999999999998`.

| Quantile | Exact | Level 3 | Absolute error |
|---|---:|---:|---:|
| q05 | -3.2244144286834016 | -3.198345149608919 | 0.026069279074482754 |
| q16 | -2.6975434029965633 | -2.8258870241166867 | 0.12834362112012343 |
| median | 0.1459509358102303 | 0.0006879154569276163 | 0.14526302035330269 |
| q84 | 2.2166921395411046 | 1.1131432156648835 | 1.103548923876221 |
| q95 | 3.5516188308777346 | 3.128296797939394 | 0.4233220329383407 |

| Inference diagnostic | Frozen limit | Partial result |
|---|---:|---|
| Absolute log Z error | 0.22 | PASS |
| Relative Z error | 0.25 | PASS |
| Absolute mean error | 0.25 | FAIL: 0.2758678257660464 |
| Absolute variance error | 0.625 | PASS |
| Maximum quantile error | 0.50 | FAIL: q84 error 1.103548923876221 |
| Weighted CDF discrepancy | 0.10 | PASS |
| Component and region probability error | 0.08 | PASS |
| Weight normalization error | 1e-12 | PASS |

These are diagnostics at a nonterminal intermediate ladder. They do not
establish what inference accuracy would have been at a future stop and
are not substituted for the required reconstruction-at-stop test.

## All three posterior components

Probabilities follow the Stage-7B convention: posterior component
responsibilities, with a separate three-region check at boundaries -1.4
and 1.8. Initial likelihood weights are not posterior mode probabilities.

| Component | Exact posterior probability | Last accepted level 3 | Absolute error | Deeper reference |
|---|---:|---:|---:|---|
| 1 | 0.22291854352877752 | 0.24801932763599116 | 0.02510078410721364 | Not reached |
| 2 | 0.5888247920411369 | 0.6195395963529322 | 0.030714804311795296 | Not reached |
| 3 | 0.18825666443008562 | 0.13244107601105662 | 0.055815588419029005 | Not reached |

All three probabilities are nonzero and their partial errors are within
0.08. The tertiary component is underestimated, and the partial mean/q84
failures remain material. Mode preservation at an automatic stop and
stability against a deeper reference were **not validated**.

## Deeper reference, tail continuation effect and restart

The frozen continuation limits were mean change <=0.05, variance change
<=0.125, each quantile change <=0.10, posterior CDF distance <=0.02,
and each component and region probability change <=0.02, in addition
to the evidence limits above. Exact restart required identical dtype,
shape and bytes for arrays and canonical identity for decisions and
metadata, with no numerical tolerance or retry.

| Required comparison | Status |
|---|---|
| Automatic stop by level 12 | FAIL: construction rejected level 4 before any stop |
| Stopped versus exactly 3 deeper levels | NOT REACHED |
| Delta log Z, mean, variance, quantiles and three component probabilities | NOT AVAILABLE |
| Tail bound covers deeper-continuation evidence effect | NOT REACHED |
| Fresh-process restart reaches identical stop | NOT REACHED |
| Checkpoint exactly at stop requests zero new samples | NOT REACHED |

The prescribed level-1 checkpoint and reconstruction prefix were saved
(`continuous/restart_point.json`). No fresh interpreter was launched after
the failure, and no alternative restart-to-construction-failure experiment
was substituted. There was no valid stop from which to generate the
predeclared deeper reference. Unavailable comparisons are left null rather
than invented.

## Disposition and artifact integrity

**Held-out validation FAIL. Automatic termination remains unvalidated.**
The first genuine failure is the unchanged selection ESS gate, not a claim
that the revised terminal bound exceeded a measured continuation effect.
The failed prerequisite prevents that termination comparison.

The frozen design, protocol, harness and scientific source hashes are
unchanged. Sampling artifacts and the original runtime failure result are
preserved under `/tmp/dns-termination-heldout-validation/`. The two
requested reports add read-only diagnostics from the completed partial run.

No additional held-out model, stochastic retry, tolerance change,
reconstruction change, kernel or promotion tuning, ESS change, LISA,
manual LISA level 13 or production run occurred. No files were staged or
committed. Historical Stage 8 stays FAIL.

Per the predeclared consequence, **do not tune termination further in this
study and stop before LISA production**. The first end-to-end automatic
LISA validation is **not authorized** by this result.
