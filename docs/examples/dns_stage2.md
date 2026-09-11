# DNS Stage 2: frozen-ladder multimodal validation

Stage 2 keeps the Stage-1 DNS target and likelihood ladder frozen. It adds a
constrained-prior slice callback, offline diagnostics, and a controlled
multimodal mechanism test. It does not construct levels, calculate evidence,
modify NSS, or integrate BayesLISAx.

## Constrained-prior slice callback

`blackjax.ns.dns_kernels.build_constrained_slice_kernel` uses the log prior as
the vertical-slice density and the DNS likelihood threshold as an additional
horizontal-slice eligibility condition. Therefore at assigned level `j` its
target is

```text
pi(theta) * 1[logL(theta) > threshold[j]].
```

The default direction is drawn from a frozen covariance. A custom direction
generator is permitted only under the same invariance requirement: its
selection law must be valid for the fixed constrained-prior target. The
adapter does not repair state-dependent block-selection probabilities.

## Exact double-well mechanism target

The validation target is one-dimensional, with

```text
theta ~ Uniform[-6, 6]
logL(theta) = -0.5 * ((abs(theta) - 1) / 0.3)^2.
```

The two maxima are at `theta=-1` and `theta=+1`. For a likelihood threshold
corresponding to radius `r`, the allowed set is

```text
||theta| - 1| < r.
```

For `r < 1` this is two disconnected intervals. For `r >= 1` the two
intervals meet and the contour is connected.

The frozen ladder uses exact masses

```text
X_j = exp(-j),  j = 0,...,4.
```

For each nonzero level the radius is chosen analytically from the uniform-prior
mass:

```text
r = 6 X_j - 1       if X_j >= 1/3   (connected contour)
r = 3 X_j           if X_j <  1/3   (two disconnected contours)
threshold_j = -0.5 * (r / 0.3)^2.
```

Hence level 1 is connected while levels 2--4 are disconnected. With exact
masses and uniform DNS weights the requested assigned-level occupancy is
uniform.

## Why the test uses a local direction

The ordinary HRSS direction scale can sometimes land directly in the opposite
disconnected component during stepping out. That is a valid global move, but
it would obscure the mechanism comparison. The Stage-2 toy test therefore
uses a position-independent direction of magnitude `0.5` and the same slice
kernel in both arms.

At the fixed high threshold the first rejected point blocks traversal of the
gap and the chain remains in its starting mode. DNS can instead descend to
level 1 or 0, move through the connected region over successive parameter
steps, and climb back to high likelihood in the other mode.

## Diagnostics

`blackjax.ns.dns_diagnostics` provides:

- highest eligible level, distinct from assigned level;
- assigned-level occupancy;
- neighboring-edge attempt/eligibility/acceptance counts;
- low-high-low and high-low-high round-trip counts;
- high-likelihood mode-switch counts and the minimum assigned level reached
  between the two high-mode visits.

A switch at high level is accepted as evidence for the intended mechanism only
when the recorded backtracking depth reaches level 0 or 1, the connected part
of this toy ladder.

## Distributional checks

`tests/ns/test_dns_kernels.py` checks both a bounded uniform prior and a
bounded exponential prior. The exponential test is specifically intended to
fail if the slice callback samples uniformly in coordinate volume instead of
according to the prior.

`tests/ns/test_dns_multimodal.py` checks:

- uniform assigned-level occupancy with exact masses and uniform weights;
- conditional first/second moments inside every level;
- equal modal fractions within statistical tolerance;
- independence from starting in the left or right mode after burn-in;
- repeated A->B and B->A high-mode switching across independent seeds;
- switch backtracking through only the connected levels 0 or 1;
- a matched slice-evaluation-cost fixed-high-threshold control that remains
  trapped in the initial mode.

Run the focused validation with

```bash
pytest tests/ns/test_dns.py tests/ns/test_dns_kernels.py tests/ns/test_dns_multimodal.py -q
```

and the mechanism report with

```bash
python examples/dns_multimodal_validation.py
```

Stage 2 deliberately stops here. Adaptive level construction, evidence
reconstruction, and BayesLISAx integration belong to later stages.

## Later DNS-safe Galactic-Binary direction mixture

The old BayesLISAx 20/40/40 proposal selected the pair component from sources
adjacent in the current frequency ordering. That selection probability changes
with the current point and can become zero for the reverse state after a move,
so it must not be imported unchanged into DNS.

The first production-safe candidate is:

- 20% global direction;
- 40% one source label selected uniformly;
- 40% one unordered labeled pair selected uniformly from all `K(K-1)/2` pairs;
- fixed, positive-definite covariance/direction parameters during stationary
  production.

For `K=9` there are 36 unordered labeled pairs. Conditional on any fixed
subset, the underlying straight-line slice transition must preserve the same
constrained prior. Because the mixture probabilities are then independent of
`theta`, a fixed mixture of those invariant subset kernels is itself invariant.
This statement also assumes that any stepper Jacobian/reversibility conditions
are satisfied; the default straight-line stepper has no extra Jacobian.
