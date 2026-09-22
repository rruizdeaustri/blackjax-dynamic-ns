# Stage-4L: direct latent Logistic gate

**Gate 1 failed against the actual numerically implemented scalar prior. Gate 2 was not run.** The intended analytic prior is factorized Logistic, and JAX provides a direct sampler without the old clipping. However, actual positive-tail density evaluation does not agree with the stable analytic log density. No likelihood probes or MCMC steps were taken.

## Analytic prior and cancellation

For s(u)=sigmoid(u), the derivative of the unit-logit map is s(u)(1−s(u)). The standard Logistic density is

\[p(u)=\frac{e^{-u}}{(1+e^{-u})^2},\qquad \log p(u)=-\operatorname{softplus}(u)-\operatorname{softplus}(-u).\]

With the configured independent uniform physical boxes, each physical-width normalization cancels the corresponding box Jacobian width. No ordering, gates, Gaussian override, or cross-source prior factor is active. Ignoring numerical evaluation details, the intended density is the product of 54 independent standard Logistic coordinates. A labelled source block b is indices 6b through 6b+5 in the order f0_u, fdot_u, iota_u, psi_u, lambda_u, beta_u. The six-factor product is the single-source prior; a two-source refresh uses the product of twelve factors. No sorting or matching enters the proposal.

For B containing either one or two source blocks, state-independent label selection cancels in the forward/reverse ratio. If q_B is exactly the target factor pi_B,

\[\frac{P(u')q_B(u_B)}{P(u)q_B(u'_B)}=\frac{\pi_{-B}(u_{-B})\pi_B(u'_B)I(u')\pi_B(u_B)}{\pi_{-B}(u_{-B})\pi_B(u_B)I(u)\pi_B(u'_B)}=\frac{I(u')}{I(u)}.\]

The intended contour-only rule follows for a valid base. Cheap tests evaluate the target and forward/reverse proposal log densities separately and verify cancellation for both single and pair blocks. These tests prove the analytic rule; they do not replace checking the actual model prior.

## Actual model evaluation

The audit loads the real configured BayesLISAx problem, obtains the same scalar prior via audit.scalar_functions(problem), and evaluates it in float64 on CPU. The source/configuration hashes match the accepted level-9 checkpoint. The likelihood function is never called. Deterministic vectors deliberately cover ordinary magnitudes, increasing positive/negative tails, and a positive-tail saturation case.

| Test vector | Actual scalar log prior | Stable analytic log prior | Actual minus analytic |
|---|---:|---:|---:|
| zero | -74.8598955004741 | -74.8598955004741 | 1.4210854715202e-14 |
| ramp8 | -230.944195269092 | -230.944195269092 | -2.27373675443232e-13 |
| ramp12 | -337.340554250361 | -337.340554250352 | -8.24229573481716e-12 |
| ramp20 | -554.484861371039 | -554.484861433372 | 6.23327878201962e-08 |
| ramp30 | -828.096097934821 | -828.094729537094 | -0.00136839772676467 |
| positive36 | -109.517254528471 | -109.473601139354 | -0.043653389117182 |
| positive40 | -inf | -113.473601139354 | -inf |
| negative40 | -113.473601139354 | -113.473601139354 | -2.8421709430404e-14 |

`rampN` means 54 evenly spaced values from −N to +N. `positive36`, `positive40`, and `negative40` have just their first coordinate set to the indicated value; the other 53 coordinates are zero. Ordinary-region results agree closely, consistent with earlier finite numerical audits. The tail tests expose a limitation those checks did not establish away.

The source evaluates each coordinate Jacobian using

```python
s = jax.nn.sigmoid(u)
log(width) + log(s + EPS) + log1p(-s + EPS)
```

where EPS is float64’s smallest positive normal value. The `1−s` subtraction is ill-conditioned in the positive tail; eventually sigmoid(u) rounds to 1. At u=40, the EPS addition is too small to change −1, and log1p(−1) is −inf. The stable analytic coordinate density at +40 is approximately exp(−40)=4.25e−18, readily representable in float64. Thus the finite-versus-infinite result is not unavoidable representational underflow of the requested density. It is an instability of the implementation. At −40 the actual prior remains finite and agrees with the symmetric analytic density.

No arbitrary tolerance choice is needed to identify the +40 failure. The +36 result also gives a finite, nonzero cancellation residual inside the implemented finite-prior support.

## Block-change counterexamples

All unchanged coordinates are exactly zero. Each selected block has its first coordinate changed to +36.

| Selected blocks | Actual full-prior change | Reverse minus forward analytic block-proposal log density | Uncancelled log ratio |
|---|---:|---:|---:|
| [0] | -34.65735902799727 | 34.61370563888009 | -0.04365338911718 |
| [1, 7] | -69.31471805599456 | 69.22741127776023 | -0.08730677823434 |

These residuals would be zero if the actual prior factors equalled the analytic proposal density. This audit evaluates only prior/proposal terms; it does not claim that these deliberately constructed vectors meet ell9, because their likelihoods were not evaluated. The requested global prior identity and block-delta prerequisites already fail. All four tested label sets and magnitudes 4, 12, 20, 30, 36, and 40 are recorded in report.json.

## Direct JAX Logistic generator

Installed JAX version: 0.10.0. The inspected jax.random.logistic implementation delegates to

```python
x = uniform(key, shape, dtype, minval=finfo(dtype).tiny, maxval=1.)
return log(x) - log1p(-x)
```

It does not call problem.sample_prior, box_to_u, physical transforms, or a fixed 1e-9 clipping operation. The lower bound is the dtype’s smallest normal value, part of the native generator’s finite-precision semantics. This is a suitable candidate for the intended analytic Logistic prior. Small cheap tests confirm float64 shape, finite values, replay, no deterministic old clipping-boundary values in those draws, and independence from the current state through an API accepting only a key and dimensions. They are not a claim that a finite test proves the distribution. The full fixed-budget sampler audit and likelihood feasibility probes were not executed after the prior gate failed.

## Decision and scope

**For the intended analytic parameterization, yes: direct Logistic draws have exactly the desired block factorization and MH cancellation. For the current numerically implemented prior, the requested equality fails; an exact contour-only refresh has not been established.** Therefore single- and pair-source level-9 viability remain unmeasured, with no promising/marginal/impractical classification fabricated.

This follows the explicit Gate-1 stop rule. No 128-base selection, 4096 single proposals, 4096 pair proposals, optional full-prior control, or likelihood evaluations were performed. No sequential updating, production integration, isotropic-kernel change, level-10 retry/acceptance, level 11, or evidence work occurred. The earlier ladder checkpoints are unchanged; this audit does not quantify any impact of tail evaluation on earlier trajectories.

Artifacts: `/tmp/lisa_dns_stage4l_latent_gate/report.json`, `examples/lisa_dns_stage4/latent_logistic_gate.py`, and `tests/experimental/test_dns_gb_latent_logistic_gate.py`. The report includes actual scalar-prior results, block deltas, native generator source, and provenance hashes. Real target likelihood probes are absent from pytest.

Validation: all 118 existing tests plus seven new cheap tests passed (125 total). Model, configuration, checkpoint, and protected code hashes remain unchanged.
