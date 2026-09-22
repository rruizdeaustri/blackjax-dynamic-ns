# Stage-4K: exact prior-refresh prerequisite audit

**Stopped before feasibility probes: the configured target factorizes, but the actual prior sampler is not exactly the same continuous prior measure.** No likelihood evaluations, prior draws, base-state evaluations, or refresh proposals were run. Single-source and pair-source feasibility remain unmeasured; neither can honestly be classified as promising, marginal, or impractical from this audit alone.

## Detailed balance for an exact block-prior refresh

Let B contain one source label or two unordered labels, chosen with fixed state-independent probability w(B). With the other coordinates unchanged, the intended constrained target is

\[
\pi_9(u) = Z_9^{-1}\,\pi_{-B}(u_{-B})\pi_B(u_B)
\mathbf 1\{\log L(u)>\ell_9\}.
\]

If the proposal is **exactly** q_B(u'_B | u_B)=pi_B(u'_B), its reverse proposal is pi_B(u_B). The label-selection probability is identical in both directions. Thus

\[
R=\frac{\pi_{-B}(u_{-B})\pi_B(u'_B)I(u')\,w(B)\pi_B(u_B)}
{\pi_{-B}(u_{-B})\pi_B(u_B)I(u)\,w(B)\pi_B(u'_B)}
=\frac{I(u')}{I(u)}.
\]

For a valid current state I(u)=1, MH accepts precisely when logL(u')>ell9. The argument applies equally to one or two whole source blocks. Equality with the contour is rejection. This algebra was verified by cheap numerical log-ratio tests with a factorized normalized uniform-box latent prior. It is a conditional proof: it requires the proposal measure to equal the target prior factor.

## What the actual model does

The model/configuration source hash matches the accepted checkpoint. Kmax=9, marg_Aphi=True, use_gates=False, order_f0=False, and no Gaussian prior overrides are configured. `logprior` adds individual physical prior terms and individual coordinate Jacobians; its support tests are coordinatewise. There are no source-order constraints or cross-source prior factors in this configuration. The likelihood can couple sources, but that does not invalidate prior factorization.

The actual `sample_prior` closure splits a key into six streams, samples independent physical arrays of shape (n,9), applies `box_to_u`, stacks `[f0_u, fdot_u, iota_u, psi_u, lam_u, beta_u]` on the last axis, and reshapes to (n,54). Label b therefore occupies indices `6*b:6*(b+1)`. The legacy `sample_prior_flat` is not the sampler under consideration.

The audited `box_to_u` implementation explicitly computes:

```python
z = (x - lo) / (hi - lo)
z = jnp.clip(z, 1e-9, 1.0 - 1e-9)
return jnp.log(z) - jnp.log1p(-z)
```

This is the same clipping already documented in the earlier prior audit; it was not newly introduced. The new requirement for an **exact independence-MH cancellation** makes the distinction consequential.

Ignoring ordinary RNG discretization and working in the continuous idealization, define epsilon=1e-9 and c_-=logit(epsilon), c_+=logit(1-epsilon). The actual coordinate proposal measure is

\[
Q(du) = f(u)\mathbf 1\{c_-<u<c_+\}\,du
+\epsilon\delta_{c_-}(du)+\epsilon\delta_{c_+}(du),
\]

where f is the ideal logistic density induced by a physical uniform and the inverse box transform. The target prior has a continuous coordinate density, including below c_- and above c_+, and has no point masses at the clipping boundaries. The numerical Jacobian guard does not turn that target into the sampler's mixed continuous/atomic measure. Consequently Q is not the required exact prior factor.

A deterministic test executes only the existing `box_to_u` function extracted from the actual source; it does not import/build the waveform model. Unit-box inputs epsilon/4, epsilon/2, and epsilon all produce the same latent value, approximately −20.72326583594641. Their unclipped inverse-transform values are distinct. This supplies a concrete counterexample without random sampling or likelihood calls.

For a proposal at a clipping atom, a correct measure-aware MH rule against a continuous target cannot simply accept on contour membership. For a current selected coordinate beyond the sampler's truncated support, the reverse proposal has zero probability density. Those cases refute the requested global contour-only rule for this sampler. The cancellation is valid for ideal exact-prior draws, and on the shared nonclipped continuous support, but not unconditionally for the supplied machinery.

## Numerical scale and stop decision

For the idealized clipped uniform sampler, the probability of at least one clipped coordinate in a selected six-coordinate block is approximately 1.2e-8, and in two selected blocks approximately 2.4e-8. Across the requested 4096 single and 4096 pair probes, the ideal probability of any selected coordinate hitting clipping is about 1.47e-4 (0.0147%). Thus this is a small numerical discrepancy, not evidence that prior stages suffered appreciable sampling error. No previous ladder values or checkpoint files are changed or invalidated by this audit.

Nevertheless, a small discrepancy is not an exact cancellation. This task explicitly requires cancellation before interpreting acceptance fractions, and requires using the existing sampler/transform. The adapter therefore stops before likelihood probes instead of silently replacing the sampler, ignoring clipping, changing the target, or introducing a different MH rule. No approximate feasibility rate is substituted for the requested exact rate.

The requested 128 fixed bases, 4096 single probes, and 4096 pair probes were not executed. No global control was run. Probe integrity counts are not applicable because no bases or proposals were evaluated; there is one explicit prerequisite sampler/target mismatch. The helper code for source indexing and nonsequential replacement is isolated and not wired into DNS.

## Validation and artifacts

Cheap tests cover exact six-coordinate indexing, unchanged unselected coordinates, independence from the current selected block, bijective uniform categorical mappings for nine labels and 36 unordered pairs, ideal prior/proposal cancellation and strict contour handling, independent fixed-base proposals without cumulative updates, and the actual-transform clipping counterexample. The algebraic tests do not misrepresent the clipped sampler as exact.

Artifacts: `/tmp/lisa_dns_stage4k_prior_refresh_audit/report.json`, `examples/lisa_dns_stage4/prior_refresh_audit.py`, and `tests/experimental/test_dns_gb_prior_refresh_audit.py`. The report records configuration/source provenance, the extracted transform, deterministic counterexample, exactness failure, and zero evaluation counts. The model, isotropic proposals, DNS core, and frozen ladder are unchanged.

**Answer:** exact single- and pair-source prior refresh have the desired contour-only acceptance algebra under an exact factor sampler. Their level-9 feasibility has not been established here because the supplied sampler fails that exactness prerequisite. Resolve sampler/target agreement before assigning feasibility classifications.

Level 10 remains rejected. No level 11, production, burn-in change, direction tuning, evidence, or posterior reconstruction was performed. Evidence reconstruction remains unvalidated and out of scope.
Regression result: all 111 existing tests plus seven new cheap prerequisite tests passed (118 total). Model/checkpoint hashes and protected code were verified unchanged after the audit.
