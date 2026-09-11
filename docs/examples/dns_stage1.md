# Frozen DNS core: scope, invariants, and GB proposal audit

This is an experimental internal module, `blackjax.ns.dns`, not a supported
top-level algorithm. Stage 1 contains fixed levels and persistent single-walker
transitions. Independent walkers can share the same frozen kernel through
`jax.vmap`. It does not construct levels, adapt proposals, calculate evidence,
or use nested-sampling live points, replacement, or birth/death histories.

## Mathematical contract

Let `A_0` be the full prior support, and for `j > 0` let
`A_j = {theta: loglikelihood(theta) > threshold[j]}`. Write `X_j` for the true
prior mass of `A_j`, and `Xhat_j = exp(levels.log_mass[j])` for a fixed estimate.
With positive weights `w_j = exp(levels.log_weight[j])`, the target is

```
q(theta, j) = pi(theta) * 1[A_j](theta) * a_j / C
a_j = w_j / Xhat_j
C = sum_j a_j * X_j
```

Consequently `q(theta | j) = pi(theta) 1[A_j](theta) / X_j` and
`q(j) = a_j X_j / C`. Incorrect mass estimates change occupancy; they do not
invalidate the frozen target. Weight normalization is unnecessary. Arbitrary
changes to the table during a run are outside this implementation's contract.
If a level has zero true prior mass, its occupancy is zero and its conditional
distribution is undefined; it cannot be reached or validly initialized.

Each level step proposes either `j-1` or `j+1` with probability one half.
Out-of-bounds attempts are rejected self-loops, rather than redrawing or
renormalizing over available neighbors. For an eligible neighbor `k`, acceptance
is `min(1, a_k/a_j)`. An ineligible upward proposal has acceptance zero. Level
zero explicitly includes zero-likelihood prior points (`loglikelihood = -inf`).
NaN and positive-infinite likelihoods are not supported valid states.

For exact masses, uniform weights, and `X_(j+1)/X_j = exp(-1)`, eligible upward
acceptance is one, downward acceptance is `exp(-1)`, and upward eligibility
conditional on level `j` is `exp(-1)`. Successful transitions across either
direction of an edge occur with probability `exp(-1)/2` per visit to its origin.

The parameter callback must preserve the *constrained prior*, not the posterior:

```
parameter_step_fn(key, particle, threshold) -> (particle, info)
```

`particle` is a `DNSParticleState(position, logdensity, loglikelihood)`;
`logdensity` is the cached log prior. A `-inf` threshold means the entire prior.
Both returned caches must correspond to the returned position. Rejection must
return the existing particle. Settings belong in a frozen callback closure.
No additional generic rejection rule can repair a callback with the wrong
invariant distribution. `DNSInfo.parameter_is_valid` reports violations visible
in cached values after *every* inner step. A false entry invalidates the run; it
is not a normal rejected move, and it is not silently repaired. This check cannot
detect incorrect caches or certify the callback's invariant distribution.

The sweep composes conditional parameter transitions and level transitions.
Each preserves `q`, so their composition preserves `q`; the composition itself
does not need to satisfy detailed balance.

## Internal API and compilation

```python
import jax
from blackjax.ns import dns

levels = dns.create_levels(thresholds, log_mass, log_weight)
state = dns.init(position, logprior_fn, loglikelihood_fn, levels)
step = dns.build_kernel(levels, parameter_step_fn,
                        num_inner_steps=1, num_level_steps=1)
state, info = jax.jit(step)(key, state)
```

`create_levels` and `init` validate concrete inputs on the host. Use these
factories rather than constructing unchecked tables or invalid states manually.
They are intentionally not jitted initializers. They validate after conversion
to the configured JAX precision, so rounded duplicate thresholds are rejected.
Initialize a batch by stacking single-walker states, then use `jax.vmap(step)`.
The level table is shared in the closure; walker states do not duplicate it.
Checkpoint the table alongside walker states; a walker state alone does not
identify its target.
Array shapes, loop lengths, and callback PyTree structures remain fixed.
`lax.scan` returns only the requested caller-selected trace; the core accumulates
no history. Per-sweep info retains inner-step and level-step axes.

## Validation

`tests/ns/test_dns.py` starts with a finite four-position, three-level problem
with a nonuniform prior. It constructs the transition matrices explicitly and
checks row normalization, exact stationarity, and level-subkernel detailed
balance, with both correct/incorrect masses and uniform/nonuniform weights.
Randomized implementations are then checked against independent matrix oracles,
including the composed parameter-and-level sweep. Additional tests cover strict
eligibility, endpoint self-loops, a single level, zero likelihood at the prior
level, invalid inputs, broken callback reporting, PyTrees, independent walker
keys, and eager/jit/vmap agreement.

The continuous test uses `pi = Uniform[-1,1]`, `logL(x) = -x^2/2`, and radii
`X_j = exp(-j)`. Conditional draws are exact uniform draws from `[-X_j, X_j]`;
this isolates DNS mechanics from slice-kernel correctness. Thirty-two walkers
run for 6500 sweeps; the first 500 are discarded. Tests check occupancy,
conditional means/second moments/CDFs, eligibility, acceptance, and directional
crossing frequencies. This is not a multimodal-exploration validation or a
validation of the existing NSS slice kernel. No evidence estimator is tested.

Validation on 2026-09-11 used the `blackjax-ns` environment, JAX 0.10.0, CPU:

| Tests | Result |
| --- | --- |
| New DNS tests, default float32 | 34 passed |
| New DNS tests, `JAX_ENABLE_X64=1` | 34 passed |
| Existing `tests/ns/test_nested_sampling.py` | All 54 passed in fresh processes |

The combined existing NS suite exhausted memory in JAX/LLVM compilation;
limiting CPU affinity alone did not resolve accumulated memory use. Each of its
54 collected test cases was therefore run unchanged in a fresh pytest process,
with four-core CPU affinity and at most two processes concurrently. All exited
successfully. No sampler, test assertion, or JAX numerical setting was patched
to obtain these regression results. GPU execution was not tested.

The new suite can be reproduced with:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ns/test_dns.py -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ns/test_dns.py -q
```

## Read-only BayesLISAx audit

Inspected in the neighboring checkout:

- `src/jax_samplers/samplers/blackjax_ns.py`,
  `_make_gb_slice_direction_old` (line 450), especially `single_move` and
  `pair_move` (the latter sorts `pos2[:, 0]` at lines 655–660).
- The sampler's selection of that factory near line 2135.
- `src/jax_samplers/problems/lisa_gb_transdim_problem.py`,
  `unpack_theta_u` near line 342 and `logprior` near line 1438.

No BayesLISAx files were modified or executed.

The factory selects a fixed mixture of 20% global, 40% single-source, and 40%
pair-source directions. Conditional on a *fixed* source subset and fixed
positive-definite covariance, the zero-mean Gaussian direction and its
Mahalanobis normalization are sign symmetric and do not otherwise depend on
position. Global direction generation uses position only for its PyTree shape;
single-source selection is uniform over labels and is position independent.

Pair selection is different: it sorts the current first coordinate in each
source block and picks one of the `K-1` adjacent pairs. The selected pair's full
blocks, including the ordering coordinate, can move. For example, starting
coordinates `(0,1,2)` make labels `{0,1}` adjacent. A line move of those two
blocks can yield `(0.1,3,2)`, where that same labeled pair is no longer adjacent.
Its forward selection probability was `1/(K-1)` and its reverse selection
probability is now zero. The old factory adds no selection-probability correction
and imposes no adjacency-preservation constraint during the slice update.

Sign symmetry at a given position is therefore insufficient to justify this
state-dependent mixture. Practical success of the old proposal is not a proof
that it preserves the constrained prior. The other mixture components do not
provide a general correction. This audit identifies a missing invariance
argument; it does not quantify the bias of a particular production run.

There is also a coordinate-layout caveat: `unpack_theta_u` places `f0_u` in column
zero only when `marg_Aphi=True`. With `marg_Aphi=False`, column zero is `lnA_u`
and `f0_u` is column one. The factory always sorts column zero, and its caller
does not guard on `marg_Aphi`. The same position-dependence issue exists in both
cases, but the interpretation as frequency adjacency is then incorrect.

For `order_f0=False`, the inspected prior includes physical-space prior terms
and logistic-transform Jacobians. A future DNS callback must retain this latent
log prior, rather than treating latent coordinates as uniform. The old factory
is selected only when `order_f0=False`. The ordered-frequency path has its own
Jacobian caveat in the problem file and is outside this audit's proposed reuse.

### Recommended later equivalent

Keep the 20/40/40 mixture, but select the single source uniformly over fixed
labels and select an unordered pair uniformly over the `K*(K-1)/2` labeled pairs.
For `K=9` there are 36 pairs. Freeze covariance (possibly one covariance per
level) for production. Each fixed-subset constrained-prior slice kernel then has
a position-independent mixing probability, so their mixture preserves the same
conditional target, provided the underlying fixed-subset slice kernel is valid.
Retain the existing prior density and straight-line stepper assumptions. This
changes pair locality while preserving the intended source-block move sizes.

A graph of preferred labeled pairs learned during warmup and then frozen is
also position independent in production. Recomputing adjacency at the start of
each move and merely holding it fixed for that move does **not** solve the
reverse-selection problem.

An alternative preserving current adjacency is an explicit auxiliary-subset
correction: select `S` with probability `s(S|theta)`, propose through a validated
reversible fixed-`S` kernel, then accept with
`min(1, s(S|theta_new)/s(S|theta_old))`. For uniform adjacent pairs this rejects
endpoints where the chosen labels are no longer adjacent. This correction
requires reversibility of the fixed-subset kernel, not just invariance, and
needs its own tests. It is not implemented in Stage 1. Continuously adapted
live-population covariances from NSS are likewise not imported into frozen DNS.

The old GB proposal must not be connected unchanged to DNS and described as a
validated constrained-prior kernel. No LISA integration or LISA run is included.
