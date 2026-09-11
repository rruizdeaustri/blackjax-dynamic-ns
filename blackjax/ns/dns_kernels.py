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
"""Constrained-prior transition kernels for frozen-level DNS.

These kernels are independent of nested-sampling live-point deletion and
replacement. At a fixed DNS level the required invariant density is

    pi(theta) * 1[loglikelihood(theta) > threshold].

The slice sampler therefore uses the log prior as its vertical-slice density
and treats the likelihood threshold as an additional horizontal-slice
eligibility condition.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.mcmc import ss
from blackjax.ns import dns
from blackjax.types import PRNGKey


def default_stepper(position, direction, t):
    """Straight-line stepper used by the default constrained slice kernel."""
    return jax.tree.map(lambda x, d: x + t * d, position, direction), jnp.array(True)


def _validated_covariance(cov):
    cov = jnp.asarray(cov)
    host = np.asarray(cov)
    if host.ndim != 2 or host.shape[0] != host.shape[1] or host.shape[0] == 0:
        raise ValueError("cov must be a nonempty square matrix")
    if not np.all(np.isfinite(host)):
        raise ValueError("cov must be finite")
    try:
        np.linalg.cholesky(host)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc
    return cov


def build_constrained_slice_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    cov=None,
    *,
    generate_slice_direction_fn: Optional[Callable] = None,
    stepper_fn: Callable = default_stepper,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a DNS parameter callback preserving a constrained prior.

    The returned callable has the Stage-1 DNS contract

        step(key, particle, threshold) -> (particle, info).

    particle.logdensity is the log prior. threshold=-inf denotes the full
    prior and does not reject points with zero likelihood.

    If generate_slice_direction_fn is omitted, directions are drawn with
    blackjax.mcmc.ss.sample_direction_from_covariance using a frozen,
    positive-definite covariance matrix. A custom direction generator must
    have signature (key, position) -> direction. Its selection law must
    itself be valid for the constrained-prior target; in particular,
    state-dependent block selection is not automatically corrected here.

    stepper_fn(position, direction, t) returns (new_position, is_valid).
    The default is a straight line. Nonlinear steppers require their own
    invariance/reversibility argument.

    Proposal settings are captured in the closure and must remain frozen
    during a stationary production trajectory.
    """
    for name, value in (("max_steps", max_steps), ("max_shrinkage", max_shrinkage)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive Python integer")

    if generate_slice_direction_fn is None:
        if cov is None:
            raise ValueError("Provide cov or generate_slice_direction_fn")
        frozen_cov = _validated_covariance(cov)

        def direction_fn(key, position):
            return ss.sample_direction_from_covariance(key, position, frozen_cov)

    else:
        if cov is not None:
            raise ValueError(
                "Pass either cov or generate_slice_direction_fn, not both"
            )
        direction_fn = generate_slice_direction_fn

    def parameter_step(
        rng_key: PRNGKey, particle: dns.DNSParticleState, threshold
    ):
        slice_key, direction_key = jax.random.split(rng_key)
        direction = direction_fn(direction_key, particle.position)

        def slice_fn(t):
            position, step_is_valid = stepper_fn(particle.position, direction, t)
            logdensity = jnp.asarray(logprior_fn(position))
            loglikelihood = jnp.asarray(loglikelihood_fn(position))
            proposed = dns.DNSParticleState(position, logdensity, loglikelihood)
            in_contour = jnp.isneginf(threshold) | (loglikelihood > threshold)
            return proposed, jnp.asarray(step_is_valid) & in_contour

        slice_kernel = ss.build_kernel(
            slice_fn,
            max_steps=max_steps,
            max_shrinkage=max_shrinkage,
        )
        return slice_kernel(slice_key, particle)

    return parameter_step
