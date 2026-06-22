# Copyright 2020- The Blackjax Authors.
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
"""Nested Slice Sampling (NSS) algorithm.
A specific implementation of Nested Sampling that uses
Hit-and-Run Slice Sampling (HRSS) as the inner MCMC kernel.
"""

from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import sample_direction_from_covariance
from blackjax.ns import diagnostics
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import ArrayTree

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
    "cluster_aware_update_with_mcmc_take_last",
]


def default_stepper_fn(x: ArrayTree, d: ArrayTree, t: float) -> tuple[ArrayTree, bool]:
    """A simple stepper function that moves from `x` along direction `d` by `t` units.

    Implements the operation: `x_new = x + t * d`.

    Parameters
    ----------
    x
        The starting position (PyTree).
    d
        The direction of movement (PyTree, same structure as `x`).
    t
        The scalar step size or distance along the direction.

    Returns
    -------
    tuple[ArrayTree, bool]
        A tuple containing the new position and whether the step was accepted.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d), True


def update_inner_kernel_params(
    rng_key: jax.random.PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Update inner kernel parameters from current particles.

    Computes the empirical covariance matrix from the live particles
    for use in slice direction proposals.

    Parameters
    ----------
    state
        The current NSState containing live particles.
    info
        Information from the last NS step (unused but kept for interface consistency).
    inner_kernel_params
        Previous inner kernel parameters (unused but kept for interface consistency).

    Returns
    -------
    Dict[str, ArrayTree]
        Dictionary containing updated 'cov' (covariance matrix).
    """
    return {
        "cov": jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
    }


def _cluster_positions(position: ArrayTree, indices) -> ArrayTree:
    """Select a cluster subset from every position PyTree leaf."""
    return jax.tree.map(lambda leaf: leaf[jnp.asarray(indices)], position)


def _contains_jax_tracer(value) -> bool:
    """Return True when ``value`` contains a JAX tracer leaf.

    The experimental cluster-aware replacement uses NumPy clustering diagnostics,
    so it must only inspect concrete live points. When callers wrap NSS steps in
    ``jax.jit``/``lax.scan``, live particles are tracers at trace time; detecting
    that up front lets us take the standard global replacement path without
    triggering ``TracerArrayConversionError`` from NumPy/Python conversions.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def cluster_aware_update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
    num_delete,
    *,
    radius: Optional[float] = None,
    min_cluster_size: int = 3,
    max_condition_number: float = 1e12,
    print_diagnostics: bool = True,
):
    """Experimental cluster-aware nested-sampling replacement strategy.

    This opt-in prototype mirrors :func:`update_with_mcmc_take_last`, but first
    clusters the current live points, samples one sufficiently populated cluster
    proportional to its size, starts replacements from that cluster, and passes
    the cluster-local covariance to compatible constrained kernels (for NSS this
    controls hit-and-run directions). Any unsafe condition falls back to the
    existing global replacement strategy. This function is intentionally not the
    default and is not intended for JIT compilation. When
    ``print_diagnostics`` is enabled, each attempted replacement emits a concise
    ``[cluster-aware]`` line with cumulative attempt, success, and fallback
    counters plus the fallback reason or selected cluster metadata.
    """
    fallback_update = update_with_mcmc_take_last(
        constrained_mcmc_step_fn, num_mcmc_steps, num_delete
    )
    diagnostic_counts = {"attempts": 0, "successes": 0, "fallbacks": 0}

    def emit_diagnostic(
        *,
        reason,
        summary=None,
        cluster_id=None,
        cluster_size=None,
        exception=None,
    ):
        if not print_diagnostics:
            return
        sizes = None
        condition_numbers = None
        num_clusters = None
        if summary is not None:
            num_clusters = summary.num_clusters
            sizes = jnp.asarray(summary.cluster_sizes).tolist()
            condition_numbers = jnp.asarray(
                summary.covariance_condition_number
            ).tolist()
        fields = [
            f"attempt={diagnostic_counts['attempts']}",
            f"success={diagnostic_counts['successes']}",
            f"fallback={diagnostic_counts['fallbacks']}",
            f"reason={reason}",
            f"num_clusters={num_clusters}",
            f"sizes={sizes}",
            f"cov_cond={condition_numbers}",
            f"selected_cluster={cluster_id}",
            f"selected_size={cluster_size}",
        ]
        if exception is not None:
            fields.append(f"exception={type(exception).__name__}")
            fields.append(f"message={exception}")
        print("[cluster-aware] " + ", ".join(fields))

    def fallback(
        reason, rng_key, state, loglikelihood_0, summary=None, **step_parameters
    ):
        diagnostic_counts["fallbacks"] += 1
        emit_diagnostic(reason=reason, summary=summary)
        return fallback_update(rng_key, state, loglikelihood_0, **step_parameters)

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        diagnostic_counts["attempts"] += 1
        try:
            particles = state.particles
            if _contains_jax_tracer(
                (rng_key, particles, loglikelihood_0, step_parameters)
            ):
                return fallback(
                    "cluster_aware_requires_concrete_state",
                    rng_key,
                    state,
                    loglikelihood_0,
                    **step_parameters,
                )

            summary = diagnostics.diagnose_live_point_clusters(
                particles,
                radius=radius,
                include_covariance_condition=True,
            )
            valid = (
                (summary.num_clusters > 1)
                & (summary.cluster_sizes >= min_cluster_size)
                & (summary.covariance_condition_number < max_condition_number)
                & np.isfinite(summary.covariance_condition_number)
            )
            valid = np.asarray(valid, dtype=bool)
            if not np.any(valid):
                return fallback(
                    "no_valid_clusters",
                    rng_key,
                    state,
                    loglikelihood_0,
                    summary=summary,
                    **step_parameters,
                )

            choice_key, sample_key = jax.random.split(rng_key)
            labels = jnp.asarray(summary.labels)
            cluster_ids = jnp.arange(summary.num_clusters)
            cluster_weights = jnp.where(valid, jnp.asarray(summary.cluster_sizes), 0)
            cluster_id = jax.random.choice(
                choice_key, cluster_ids, p=cluster_weights / cluster_weights.sum()
            )
            cluster_id_int = int(np.asarray(jax.device_get(cluster_id)))
            cluster_mask = labels == cluster_id_int
            survivor_mask = np.asarray(
                jax.device_get(particles.loglikelihood > loglikelihood_0), dtype=bool
            )
            weights = jnp.asarray(cluster_mask & survivor_mask, dtype=jnp.float32)
            selected_cluster_size = int(np.sum(cluster_mask))
            if float(np.asarray(jax.device_get(weights.sum()))) <= 0.0:
                diagnostic_counts["fallbacks"] += 1
                emit_diagnostic(
                    reason="no_surviving_points_in_selected_cluster",
                    summary=summary,
                    cluster_id=cluster_id_int,
                    cluster_size=selected_cluster_size,
                )
                return fallback_update(
                    rng_key, state, loglikelihood_0, **step_parameters
                )

            start_key, mcmc_key = jax.random.split(sample_key)
            start_idx = jax.random.choice(
                start_key,
                len(weights),
                shape=(num_delete,),
                p=weights / weights.sum(),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            cluster_indices = jnp.nonzero(
                cluster_mask, size=int(cluster_mask.shape[0])
            )[0][:selected_cluster_size]
            cluster_position = _cluster_positions(particles.position, cluster_indices)
            cluster_cov = jnp.atleast_2d(particles_covariance_matrix(cluster_position))
            if not np.all(np.asarray(jax.device_get(jnp.isfinite(cluster_cov)))):
                diagnostic_counts["fallbacks"] += 1
                emit_diagnostic(
                    reason="non_finite_cluster_covariance",
                    summary=summary,
                    cluster_id=cluster_id_int,
                    cluster_size=selected_cluster_size,
                )
                return fallback_update(
                    rng_key, state, loglikelihood_0, **step_parameters
                )

            local_params = dict(step_parameters)
            local_params["cov"] = cluster_cov
            shared_mcmc_step_fn = partial(
                constrained_mcmc_step_fn,
                loglikelihood_0=loglikelihood_0,
                **local_params,
            )

            def mcmc_kernel(rng_key, state):
                keys = jax.random.split(rng_key, num_mcmc_steps)

                def body_fn(state, rng_key):
                    new_state, info = shared_mcmc_step_fn(rng_key, state)
                    return new_state, info

                return jax.lax.scan(body_fn, state, keys)

            sample_keys = jax.random.split(mcmc_key, num_delete)
            diagnostic_counts["successes"] += 1
            emit_diagnostic(
                reason="cluster_aware",
                summary=summary,
                cluster_id=cluster_id_int,
                cluster_size=selected_cluster_size,
            )
            return jax.vmap(mcmc_kernel)(sample_keys, start_state)
        except Exception as exc:
            diagnostic_counts["fallbacks"] += 1
            emit_diagnostic(reason="exception", exception=exc)
            return fallback_update(rng_key, state, loglikelihood_0, **step_parameters)

    return update_function


def build_kernel(
    init_state_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Builds the Nested Slice Sampling kernel.

    see `as_top_level_api` for parameter descriptions.
    """

    def constrained_mcmc_slice_fn(rng_key, state, loglikelihood_0, **params):
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key, state.position, **params)

        def slice_fn(t) -> tuple[NSState, bool]:
            x, step_accepted = stepper_fn(state.position, d, t)
            new_state = init_state_fn(x, loglikelihood_birth=loglikelihood_0)
            in_contour = new_state.loglikelihood > loglikelihood_0
            is_accepted = in_contour & step_accepted
            return new_state, is_accepted

        slice_kernel = build_slice_kernel(
            slice_fn,
            max_steps=max_steps,
            max_shrinkage=max_shrinkage,
        )
        new_slice_state, slice_info = slice_kernel(rng_key, state)
        return new_slice_state, slice_info

    inner_kernel = update_strategy(
        constrained_mcmc_slice_fn, num_inner_steps, num_delete
    )

    delete_fn = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates a Nested Slice Sampling (NSS) algorithm.

    This function configures a Nested Sampling algorithm that uses Hit-and-Run
    Slice Sampling (HRSS) as its inner kernel. The parameters for the HRSS
    direction proposal (specifically, the covariance matrix) are managed
    externally using `init_inner_kernel_params` and `update_inner_kernel_params`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_inner_steps
        The number of HRSS steps to run for each new particle generation.
        This should be a multiple of the dimension of the parameter space.
    num_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> (x_new, is_accepted)` for the HRSS kernel.
        Defaults to `default_stepper_fn`.
    generate_slice_direction_fn
        A function `(rng_key, position, **kwargs) -> direction_pytree` that generates a
        normalized direction for HRSS. Keyword arguments are unpacked from the
        inner_kernel_params dict. Defaults to `sample_direction_from_covariance`.
    init_state_strategy_fn
        A function to initialize NSState from positions.
        Defaults to `init_state_strategy`.
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase. Defaults to 10.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.
        Defaults to 100.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Slice Sampler. The step function signature is
        `step(rng_key, state, inner_kernel_params) -> (new_state, info)`.
    """
    init_state_fn = partial(
        init_state_strategy_fn,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        stepper_fn=stepper_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        update_strategy=update_strategy,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        # Vectorize the functions for parallel evaluation over particles
        # vmap maps over positional args, keyword args (like loglikelihood_birth) are broadcast
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
            rng_key=rng_key,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
