# Copyright 2026 The BlackJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental frozen-level Diffusive Nested Sampling transitions.

For a normalized prior pi and nested sets A_j, this module preserves

    q(theta, j) proportional to pi(theta) 1[A_j](theta) a_j,
    a_j = exp(log_weight[j] - log_mass[j]).

The supplied masses are fixed estimates, not assumed exact. Actual occupancy is
proportional to weight[j] * X[j] / estimated_X[j]; conditional on j the target is
always the constrained prior. Level zero is the entire prior, including points
with zero likelihood. All other levels use strict log-likelihood exceedance.

There is no adaptation, live population, birth/death bookkeeping, or evidence
estimator. Factories and validated initialization run on the host; returned
single-walker kernels support jit, scan, and vmap over independent walkers. Keep
the level table and parameter-kernel settings frozen for the whole trajectory.
"""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.types import Array, ArrayLikeTree, PRNGKey


class DNSLevels(NamedTuple):
    """Fixed arrays of equal shape (number of levels,); log_weight need not normalize."""

    loglikelihood: Array
    log_mass: Array
    log_weight: Array


class DNSParticleState(NamedTuple):
    """One persistent position with cached log prior and log likelihood; no birth field."""

    position: ArrayLikeTree
    logdensity: Array
    loglikelihood: Array


class DNSState(NamedTuple):
    """One walker and its assigned (not necessarily highest eligible) level."""

    particle: DNSParticleState
    level_index: Array


class DNSLevelInfo(NamedTuple):
    """One index proposal, including raw out-of-range proposals at the endpoints.

    Boundary attempts are rejected self-loops: eligibility and acceptance are
    zero. Downward proposals can be eligible but rejected by Metropolis.
    """

    previous_level: Array
    proposed_level: Array
    is_eligible: Array
    acceptance_probability: Array
    is_accepted: Array


class DNSInfo(NamedTuple):
    """Per-sweep traces with leading inner-step and level-step axes respectively.

    parameter_is_valid checks cached support/constraint values after every
    parameter transition. False indicates a broken callback contract, not an
    ordinary rejection; discard that run and fix the callback. Caches and
    invariance cannot be verified automatically without a model-specific audit.
    """

    parameter_info: Any
    parameter_is_valid: Array
    level_info: DNSLevelInfo


def create_levels(loglikelihood, log_mass, log_weight) -> DNSLevels:
    """Validate a fixed ladder on the host, after conversion to JAX precision.

    The first threshold must be -inf and log_mass[0] must be zero. Remaining
    thresholds must be finite and strictly increasing. Mass estimates must be
    finite and nonincreasing; equal masses are permitted as provisional values.
    All weights must be positive (finite in log space). No levels are padded.
    """
    arrays = tuple(jnp.asarray(x) for x in (loglikelihood, log_mass, log_weight))
    if any(x.dtype.kind not in "fiu" for x in arrays):
        raise ValueError("Level arrays must be real numeric arrays")
    arrays = tuple(x.astype(jnp.result_type(x, float)) for x in arrays)
    threshold, mass, weight = (np.asarray(x) for x in arrays)
    if any(x.ndim != 1 for x in (threshold, mass, weight)):
        raise ValueError("Level arrays must be one-dimensional")
    if (
        threshold.size == 0
        or mass.shape != threshold.shape
        or weight.shape != threshold.shape
    ):
        raise ValueError("Level arrays must have the same nonempty shape")
    if not np.isneginf(threshold[0]) or not np.all(np.isfinite(threshold[1:])):
        raise ValueError("Level zero must be -inf; other thresholds must be finite")
    if not np.all(np.diff(threshold) > 0):
        raise ValueError("Likelihood thresholds must strictly increase")
    if not np.all(np.isfinite(mass)) or mass[0] != 0 or np.any(np.diff(mass) > 0):
        raise ValueError(
            "Log masses must be finite and nonincreasing, starting at zero"
        )
    if not np.all(np.isfinite(weight)):
        raise ValueError("Log weights must be finite")
    return DNSLevels(*arrays)


def is_eligible(levels: DNSLevels, loglikelihood: Array, level_index: Array) -> Array:
    """JAX membership test; invalid indices are ineligible, level zero is unconstrained."""
    in_bounds = (level_index >= 0) & (level_index < levels.loglikelihood.shape[0])
    safe_index = jnp.clip(level_index, 0, levels.loglikelihood.shape[0] - 1)
    return in_bounds & (
        (level_index == 0) | (loglikelihood > levels.loglikelihood[safe_index])
    )


def _particle_is_valid(levels, particle, level_index):
    return (
        jnp.isfinite(particle.logdensity)
        & ~jnp.isnan(particle.loglikelihood)
        & ~jnp.isposinf(particle.loglikelihood)
        & is_eligible(levels, particle.loglikelihood, level_index)
    )


def init(
    position: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    levels: DNSLevels,
    level_index: int = 0,
) -> DNSState:
    """Initialize and validate one walker on the host (not a jitted initializer).

    Positions need not initially be stationary, but must have finite log prior,
    non-NaN/non-positive-infinite log likelihood, and satisfy their assigned
    constraint. Zero likelihood is allowed at level zero. For a batch, stack
    initialized states and vmap the transition kernel; keep levels shared.
    """
    index = np.asarray(level_index)
    if index.shape != () or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("level_index must be a scalar integer")
    if not 0 <= index < levels.loglikelihood.size:
        raise ValueError("Initial level index is out of bounds")
    particle = DNSParticleState(
        position,
        jnp.asarray(logprior_fn(position)),
        jnp.asarray(loglikelihood_fn(position)),
    )
    if particle.logdensity.shape != () or particle.loglikelihood.shape != ():
        raise ValueError("Prior and likelihood functions must return scalars")
    state = DNSState(particle, jnp.asarray(index, dtype=jnp.int32))
    if not bool(_particle_is_valid(levels, particle, state.level_index)):
        raise ValueError(
            "Initial particle violates the prior or assigned likelihood constraint"
        )
    return state


def level_move_probability(
    levels: DNSLevels, loglikelihood: Array, level_index: Array, proposed_level: Array
) -> tuple[Array, Array]:
    """Eligibility and MH probability for symmetric neighboring-level proposals.

    A sign is drawn with probability 1/2 at every level. Out-of-bounds attempts
    stay put instead of being renormalized over available neighbors. Therefore
    no endpoint Hastings correction is needed. Current state must be valid.
    """
    eligible = is_eligible(levels, loglikelihood, proposed_level) & (
        jnp.abs(proposed_level - level_index) == 1
    )
    safe_index = jnp.clip(proposed_level, 0, levels.loglikelihood.shape[0] - 1)
    # Difference before subtraction reduces cancellation from common offsets.
    log_ratio = (levels.log_weight[safe_index] - levels.log_weight[level_index]) - (
        levels.log_mass[safe_index] - levels.log_mass[level_index]
    )
    probability = jnp.where(eligible, jnp.exp(jnp.minimum(0.0, log_ratio)), 0.0)
    return eligible, probability


def build_level_kernel(levels: DNSLevels) -> Callable:
    """Build a single neighboring-level MH transition; no likelihood evaluations."""

    def kernel(rng_key: PRNGKey, state: DNSState) -> tuple[DNSState, DNSLevelInfo]:
        direction_key, accept_key = jax.random.split(rng_key)
        direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
        proposed = state.level_index + direction
        eligible, probability = level_move_probability(
            levels, state.particle.loglikelihood, state.level_index, proposed
        )
        accepted = jax.random.uniform(accept_key, dtype=probability.dtype) < probability
        next_level = jnp.where(accepted, proposed, state.level_index)
        return DNSState(state.particle, next_level), DNSLevelInfo(
            state.level_index, proposed, eligible, probability, accepted
        )

    return kernel


def build_kernel(
    levels: DNSLevels,
    parameter_step_fn: Callable,
    num_inner_steps: int = 1,
    num_level_steps: int = 1,
) -> Callable:
    """Build a frozen DNS sweep: parameter transitions followed by index transitions.

    ``parameter_step_fn(key, particle, loglikelihood_threshold) -> (particle, info)``
    must preserve the prior conditional on strict threshold exceedance. A -inf
    threshold denotes the full prior, including zero-likelihood points. It must
    return a DNSParticleState with consistent caches and fixed-shape info; on
    rejection it returns the current particle. Freeze any parameters in its
    closure. State-dependent proposal selection needs its own invariance proof.

    The callback need only preserve the conditional target; reversibility of
    the full sweep is not required. No generic rejection correction is applied
    to broken callbacks. num_inner_steps and num_level_steps are positive static
    integers. Histories are returned in info, not accumulated in the state.
    """
    for name, value in (
        ("num_inner_steps", num_inner_steps),
        ("num_level_steps", num_level_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive Python integer")
    level_kernel = build_level_kernel(levels)

    def kernel(rng_key: PRNGKey, state: DNSState) -> tuple[DNSState, DNSInfo]:
        parameter_key, level_key = jax.random.split(rng_key)
        threshold = levels.loglikelihood[state.level_index]

        def parameter_body(particle, key):
            particle, info = parameter_step_fn(key, particle, threshold)
            valid = _particle_is_valid(levels, particle, state.level_index)
            return particle, (info, valid)

        particle, (parameter_info, parameter_valid) = jax.lax.scan(
            parameter_body,
            state.particle,
            jax.random.split(parameter_key, num_inner_steps),
        )

        def level_body(current, key):
            return level_kernel(key, current)

        new_state, level_info = jax.lax.scan(
            level_body,
            DNSState(particle, state.level_index),
            jax.random.split(level_key, num_level_steps),
        )
        return new_state, DNSInfo(parameter_info, parameter_valid, level_info)

    return kernel
