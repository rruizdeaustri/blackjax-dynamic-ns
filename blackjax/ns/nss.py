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
from types import SimpleNamespace
from typing import Callable, Dict, Optional, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import time
from jax.flatten_util import ravel_pytree

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


class ReplacementDiagnostics(NamedTuple):
    """Per-NS-step diagnostics for constrained-MCMC replacement."""

    mcmc_infos: NamedTuple
    strategy: str
    used_strategy: str
    success_rate: jnp.ndarray
    accepted_proposals: jnp.ndarray
    proposals_per_accepted_replacement: jnp.ndarray
    final_loglikelihood_improvement: jnp.ndarray
    selected_cluster_size: jnp.ndarray
    covariance_mode: str
    condition_number_before_regularization: jnp.ndarray
    condition_number_after_regularization: jnp.ndarray
    runtime_seconds: jnp.ndarray
    fallback_to_global: jnp.ndarray
    diagonal_covariance_fallback: jnp.ndarray


def _replacement_acceptance_diagnostics(
    final_particles, mcmc_infos, loglikelihood_0, num_mcmc_steps
):
    accepted = getattr(mcmc_infos, "is_accepted", None)
    if accepted is None:
        accepted = jnp.ones(
            (final_particles.loglikelihood.shape[0], num_mcmc_steps), dtype=bool
        )
    accepted = jnp.asarray(accepted).astype(bool)
    per_replacement_accepted = jnp.sum(accepted, axis=-1)
    success = final_particles.loglikelihood > loglikelihood_0
    denom = jnp.maximum(per_replacement_accepted, 1)
    proposals_per_accepted = jnp.where(success, num_mcmc_steps / denom, jnp.inf)
    improvement = final_particles.loglikelihood - loglikelihood_0
    return success, per_replacement_accepted, proposals_per_accepted, improvement


def diagnostic_update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
    num_delete,
    *,
    strategy_name: str = "global",
    print_diagnostics: bool = True,
):
    """Global replacement strategy with per-step replacement diagnostics."""
    base_update = update_with_mcmc_take_last(
        constrained_mcmc_step_fn, num_mcmc_steps, num_delete
    )
    counts = {"attempts": 0, "successes": 0, "runtime": 0.0}

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        counts["attempts"] += 1
        t0 = time.perf_counter()
        new_particles, mcmc_infos = base_update(
            rng_key, state, loglikelihood_0, **step_parameters
        )
        jax.block_until_ready(new_particles.loglikelihood)
        runtime = time.perf_counter() - t0
        success, accepted, proposals_per_accepted, improvement = (
            _replacement_acceptance_diagnostics(
                new_particles, mcmc_infos, loglikelihood_0, num_mcmc_steps
            )
        )
        counts["successes"] += int(np.asarray(jnp.sum(success)))
        counts["runtime"] += runtime
        diag = ReplacementDiagnostics(
            mcmc_infos=mcmc_infos,
            strategy=strategy_name,
            used_strategy="global",
            success_rate=jnp.mean(success.astype(jnp.float32)),
            accepted_proposals=accepted,
            proposals_per_accepted_replacement=proposals_per_accepted,
            final_loglikelihood_improvement=improvement,
            selected_cluster_size=jnp.full((num_delete,), -1),
            covariance_mode="global",
            condition_number_before_regularization=jnp.nan,
            condition_number_after_regularization=jnp.nan,
            runtime_seconds=jnp.asarray(runtime),
            fallback_to_global=jnp.asarray(False),
            diagonal_covariance_fallback=jnp.asarray(False),
        )
        if print_diagnostics:
            print(
                "[replacement] "
                f"strategy={strategy_name}, used=global, attempt={counts['attempts']}, "
                f"success_rate={float(diag.success_rate):.4f}, "
                f"accepted_proposals={np.asarray(accepted).tolist()}, "
                f"proposals_per_accepted={np.asarray(proposals_per_accepted).tolist()}, "
                f"final_logL_minus_threshold={np.asarray(improvement).tolist()}, "
                f"runtime_s={runtime:.6f}"
            )
        return new_particles, diag

    update_function.diagnostic_counts = counts
    return update_function


__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
    "cluster_aware_update_with_mcmc_take_last",
    "diagnostic_update_with_mcmc_take_last",
    "ReplacementDiagnostics",
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


def _standardize_positions(
    position: ArrayTree, *, scale_floor: float
) -> tuple[ArrayTree, jnp.ndarray, jnp.ndarray]:
    """Robustly standardize batched positions without changing their PyTree shape.

    The experimental cluster-aware path does NumPy clustering outside JIT.  It
    should make scale-sensitive choices (Euclidean neighbourhoods and covariance
    condition checks) in dimensionless coordinates while keeping the actual
    replacement state in the caller's original parameterization.
    """
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")

    rows = jax.vmap(lambda x: ravel_pytree(x)[0])(position)
    _, unravel_fn = ravel_pytree(jax.tree.map(lambda leaf: leaf[0], position))
    center = jnp.median(rows, axis=0)
    absolute_deviation = jnp.abs(rows - center)
    mad = 1.4826 * jnp.median(absolute_deviation, axis=0)
    std = jnp.std(rows, axis=0)
    scale_floor_array = jnp.asarray(scale_floor, dtype=rows.dtype)
    scale = jnp.where(mad > scale_floor_array, mad, std)
    scale = jnp.maximum(scale, scale_floor_array)
    standardized_rows = (rows - center) / scale
    return jax.vmap(unravel_fn)(standardized_rows), center, scale


def _covariance_condition_numbers_for_labels(
    position: ArrayTree, labels: np.ndarray, num_clusters: int
) -> np.ndarray:
    """Compute per-label covariance condition numbers in the original coordinates."""
    rows = np.asarray(jax.device_get(jax.vmap(lambda x: ravel_pytree(x)[0])(position)))
    condition_numbers = np.full(num_clusters, np.inf, dtype=float)
    labels = np.asarray(labels)
    for cluster_id in range(num_clusters):
        cluster_rows = rows[labels == cluster_id]
        if cluster_rows.shape[0] > 1:
            covariance = np.atleast_2d(np.cov(cluster_rows, rowvar=False))
            condition_numbers[cluster_id] = float(np.linalg.cond(covariance))
    return condition_numbers


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
    standardize: bool = True,
    scale_floor: float = 1e-12,
    print_diagnostics: bool = True,
    eager: bool = False,
    covariance_regularization: float = 1e-6,
    auto_fallback: bool = False,
    warmup_attempts: int = 25,
    min_success_rate: float = 0.5,
    max_runtime_ratio: float = 2.0,
):
    """Experimental cluster-aware nested-sampling replacement strategy.

    This opt-in prototype mirrors :func:`update_with_mcmc_take_last`, but first
    clusters the current live points, samples one sufficiently populated cluster
    proportional to its size, starts replacements from that cluster, and passes
    the cluster-local covariance to compatible constrained kernels (for NSS this
    controls hit-and-run directions). Any unsafe condition falls back to the
    existing global replacement strategy. This function is intentionally not the
    default. By default it preserves the traced/JIT-compatible fallback behavior
    used by the standard replacement path. Set ``eager=True`` for validation runs
    that operate on concrete live points: the cluster selection and per-particle
    MCMC loop are executed from Python instead of ``vmap``/``lax.scan``. When
    ``print_diagnostics`` is enabled, each attempted replacement emits a concise
    ``[cluster-aware]`` line with cumulative attempt, success, and fallback
    counters plus the fallback reason or selected cluster metadata. When
    ``standardize=True``, clustering, covariance condition diagnostics, and the
    cluster-local proposal covariance are computed after robust median/MAD
    standardization with ``scale_floor``; replacement states remain in the
    original parameterization. By default (``eager=False``), cluster selection,
    validity checks, and covariance construction are JAX-native and compatible
    with ``jax.jit``/``lax.scan``. Set ``eager=True`` to retain the original
    Python/NumPy diagnostics path.
    """
    fallback_update = update_with_mcmc_take_last(
        constrained_mcmc_step_fn, num_mcmc_steps, num_delete
    )
    diagnostic_counts = {
        "attempts": 0,
        "successes": 0,
        "fallbacks": 0,
        "cluster_runtime": 0.0,
        "cluster_attempts": 0,
        "global_runtime": 0.0,
        "global_attempts": 0,
        "auto_fallback_active": False,
    }

    def emit_diagnostic(
        *,
        reason,
        summary=None,
        raw_condition_numbers=None,
        standardized_condition_numbers=None,
        standardization_applied=False,
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
        if raw_condition_numbers is not None:
            raw_condition_numbers = jnp.asarray(raw_condition_numbers).tolist()
        if standardized_condition_numbers is not None:
            standardized_condition_numbers = jnp.asarray(
                standardized_condition_numbers
            ).tolist()
        fields = [
            f"attempt={diagnostic_counts['attempts']}",
            f"success={diagnostic_counts['successes']}",
            f"fallback={diagnostic_counts['fallbacks']}",
            f"reason={reason}",
            f"standardized={standardization_applied}",
            f"num_clusters={num_clusters}",
            f"sizes={sizes}",
            f"cov_cond={condition_numbers}",
            f"cov_cond_raw={raw_condition_numbers}",
            f"cov_cond_standardized={standardized_condition_numbers}",
            f"selected_cluster={cluster_id}",
            f"selected_size={cluster_size}",
        ]
        if exception is not None:
            fields.append(f"exception={type(exception).__name__}")
            fields.append(f"message={exception}")
        print("[cluster-aware] " + ", ".join(fields))

    def fallback(
        reason,
        rng_key,
        state,
        loglikelihood_0,
        summary=None,
        raw_condition_numbers=None,
        standardized_condition_numbers=None,
        standardization_applied=False,
        **step_parameters,
    ):
        diagnostic_counts["fallbacks"] += 1
        emit_diagnostic(
            reason=reason,
            summary=summary,
            raw_condition_numbers=raw_condition_numbers,
            standardized_condition_numbers=standardized_condition_numbers,
            standardization_applied=standardization_applied,
        )
        if auto_fallback and diagnostic_counts["attempts"] >= warmup_attempts:
            fallback_fraction = diagnostic_counts["fallbacks"] / max(
                diagnostic_counts["attempts"], 1
            )
            if fallback_fraction > (1.0 - min_success_rate):
                diagnostic_counts["auto_fallback_active"] = True
                if print_diagnostics:
                    print(
                        "[cluster-aware] auto_fallback_enabled "
                        f"reason=fallback_fraction, fallback_fraction={fallback_fraction:.4f}"
                    )
        return fallback_update(rng_key, state, loglikelihood_0, **step_parameters)

    def _regularized_weighted_covariance(rows, weights):
        weights = weights.astype(rows.dtype)
        total = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=rows.dtype))
        normalized = weights / total
        mean = jnp.sum(rows * normalized[:, None], axis=0)
        centered = rows - mean
        covariance = (centered * normalized[:, None]).T @ centered
        dimension = rows.shape[-1]
        diag = jnp.diag(covariance)
        diag_floor = jnp.maximum(
            jnp.mean(diag), jnp.asarray(scale_floor, dtype=rows.dtype)
        )
        regularization = jnp.asarray(covariance_regularization, dtype=rows.dtype)
        covariance = covariance + regularization * diag_floor * jnp.eye(
            dimension, dtype=rows.dtype
        )
        covariance = jnp.where(
            jnp.all(jnp.isfinite(covariance)),
            covariance,
            diag_floor * jnp.eye(dimension, dtype=rows.dtype),
        )
        return covariance

    def _condition_number(covariance):
        eigenvalues = jnp.linalg.eigvalsh(covariance)
        max_eigenvalue = jnp.max(eigenvalues)
        min_eigenvalue = jnp.maximum(
            jnp.min(eigenvalues), jnp.asarray(scale_floor, dtype=eigenvalues.dtype)
        )
        return max_eigenvalue / min_eigenvalue

    def _jax_cluster_update(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, start_key, sample_key = jax.random.split(rng_key, 3)
        particles = state.particles
        rows = jax.vmap(lambda x: ravel_pytree(x)[0])(particles.position)
        if standardize:
            center = jnp.median(rows, axis=0)
            absolute_deviation = jnp.abs(rows - center)
            mad = 1.4826 * jnp.median(absolute_deviation, axis=0)
            std = jnp.std(rows, axis=0)
            floor = jnp.asarray(scale_floor, dtype=rows.dtype)
            scale = jnp.maximum(jnp.where(mad > floor, mad, std), floor)
            cluster_rows = (rows - center) / scale
        else:
            cluster_rows = rows

        distances = jnp.linalg.norm(
            cluster_rows[:, None, :] - cluster_rows[None, :, :], axis=-1
        )
        if radius is None:
            positive_distances = jnp.where(
                distances > 0, distances, jnp.asarray(jnp.inf, dtype=distances.dtype)
            )
            radius_value = jnp.median(jnp.min(positive_distances, axis=1)) * 2.0
            radius_value = jnp.maximum(
                radius_value, jnp.asarray(scale_floor, dtype=rows.dtype)
            )
        else:
            radius_value = jnp.asarray(radius, dtype=rows.dtype)

        neighbor_mask = distances <= radius_value
        cluster_sizes = jnp.sum(neighbor_mask, axis=1)
        survivor_mask = particles.loglikelihood > loglikelihood_0

        def candidate_cov(mask):
            return _regularized_weighted_covariance(
                cluster_rows, mask.astype(rows.dtype)
            )

        covariances = jax.vmap(candidate_cov)(neighbor_mask)
        condition_numbers = jax.vmap(_condition_number)(covariances)
        valid = (
            (cluster_sizes >= min_cluster_size)
            & jnp.isfinite(condition_numbers)
            & (condition_numbers < max_condition_number)
            & (jnp.sum(neighbor_mask & survivor_mask[None, :], axis=1) > 0)
        )
        weights = jnp.where(valid, cluster_sizes.astype(jnp.float32), 0.0)
        has_valid_cluster = jnp.sum(weights) > 0.0

        def cluster_branch(_):
            seed_idx = jax.random.choice(
                choice_key,
                rows.shape[0],
                p=weights / jnp.sum(weights),
            )
            selected_mask = neighbor_mask[seed_idx]
            start_weights = jnp.where(selected_mask & survivor_mask, 1.0, 0.0)
            start_idx = jax.random.choice(
                start_key,
                rows.shape[0],
                shape=(num_delete,),
                p=start_weights / jnp.sum(start_weights),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)
            cluster_cov = covariances[seed_idx]
            diagonal_cov = jnp.diag(jnp.maximum(jnp.diag(cluster_cov), scale_floor))
            cluster_cov = jnp.where(
                (_condition_number(cluster_cov) < max_condition_number)
                & jnp.all(jnp.isfinite(cluster_cov)),
                cluster_cov,
                diagonal_cov,
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

            sample_keys = jax.random.split(sample_key, num_delete)
            return jax.vmap(mcmc_kernel)(sample_keys, start_state)

        return jax.lax.cond(
            has_valid_cluster,
            cluster_branch,
            lambda _: fallback_update(
                rng_key, state, loglikelihood_0, **step_parameters
            ),
            operand=None,
        )

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        if not eager:
            return _jax_cluster_update(
                rng_key, state, loglikelihood_0, **step_parameters
            )
        diagnostic_counts["attempts"] += 1
        if auto_fallback and diagnostic_counts["auto_fallback_active"]:
            t0 = time.perf_counter()
            new_particles, mcmc_infos = fallback_update(
                rng_key, state, loglikelihood_0, **step_parameters
            )
            jax.block_until_ready(new_particles.loglikelihood)
            runtime = time.perf_counter() - t0
            diagnostic_counts["global_runtime"] += runtime
            diagnostic_counts["global_attempts"] += 1
            success, accepted, proposals_per_accepted, improvement = (
                _replacement_acceptance_diagnostics(
                    new_particles, mcmc_infos, loglikelihood_0, num_mcmc_steps
                )
            )
            diag = ReplacementDiagnostics(
                mcmc_infos,
                "cluster_aware",
                "global",
                jnp.mean(success.astype(jnp.float32)),
                accepted,
                proposals_per_accepted,
                improvement,
                jnp.full((num_delete,), -1),
                "global_auto_fallback",
                jnp.nan,
                jnp.nan,
                jnp.asarray(runtime),
                jnp.asarray(True),
                jnp.asarray(False),
            )
            if print_diagnostics:
                print(
                    "[replacement] strategy=cluster_aware, used=global_auto_fallback, "
                    f"attempt={diagnostic_counts['attempts']}, success_rate={float(diag.success_rate):.4f}, runtime_s={runtime:.6f}"
                )
            return new_particles, diag
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

            standardization_applied = bool(standardize)
            standardized_position = particles.position
            if standardization_applied:
                standardized_position, _, _ = _standardize_positions(
                    particles.position, scale_floor=scale_floor
                )

            standardized_particles = SimpleNamespace(
                position=standardized_position, loglikelihood=particles.loglikelihood
            )
            summary = diagnostics.diagnose_live_point_clusters(
                standardized_particles,
                radius=radius,
                include_covariance_condition=True,
            )
            standardized_condition_numbers = summary.covariance_condition_number
            raw_condition_numbers = _covariance_condition_numbers_for_labels(
                particles.position, summary.labels, summary.num_clusters
            )
            valid = (
                (summary.num_clusters > 1)
                & (summary.cluster_sizes >= min_cluster_size)
                & (standardized_condition_numbers < max_condition_number)
                & np.isfinite(standardized_condition_numbers)
            )
            valid = np.asarray(valid, dtype=bool)
            if not np.any(valid):
                return fallback(
                    "no_valid_clusters",
                    rng_key,
                    state,
                    loglikelihood_0,
                    summary=summary,
                    raw_condition_numbers=raw_condition_numbers,
                    standardized_condition_numbers=standardized_condition_numbers,
                    standardization_applied=standardization_applied,
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
                    raw_condition_numbers=raw_condition_numbers,
                    standardized_condition_numbers=standardized_condition_numbers,
                    standardization_applied=standardization_applied,
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
            cluster_position = _cluster_positions(
                standardized_position, cluster_indices
            )
            cluster_cov = jnp.atleast_2d(particles_covariance_matrix(cluster_position))
            covariance_mode = "full_covariance"
            diagonal_covariance_fallback = False
            condition_before_regularization = jnp.linalg.cond(cluster_cov)
            if covariance_regularization > 0.0:
                diag_mean = jnp.maximum(
                    jnp.mean(jnp.diag(cluster_cov)),
                    jnp.asarray(scale_floor, dtype=cluster_cov.dtype),
                )
                cluster_cov = (
                    cluster_cov
                    + covariance_regularization
                    * diag_mean
                    * jnp.eye(cluster_cov.shape[0], dtype=cluster_cov.dtype)
                )
                covariance_mode = "regularized_covariance"
            condition_after_regularization = jnp.linalg.cond(cluster_cov)
            if (
                not np.isfinite(float(np.asarray(condition_after_regularization)))
            ) or float(
                np.asarray(condition_after_regularization)
            ) >= max_condition_number:
                diagonal_covariance_fallback = True
                covariance_mode = "diagonal_fallback"
                cluster_cov = jnp.diag(jnp.maximum(jnp.diag(cluster_cov), scale_floor))
                condition_after_regularization = jnp.linalg.cond(cluster_cov)
            if not np.all(np.asarray(jax.device_get(jnp.isfinite(cluster_cov)))):
                diagnostic_counts["fallbacks"] += 1
                emit_diagnostic(
                    reason="non_finite_cluster_covariance",
                    summary=summary,
                    raw_condition_numbers=raw_condition_numbers,
                    standardized_condition_numbers=standardized_condition_numbers,
                    standardization_applied=standardization_applied,
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

                if not eager:
                    return jax.lax.scan(body_fn, state, keys)

                infos = []
                for key in keys:
                    state, info = body_fn(state, key)
                    infos.append(info)
                return state, jax.tree.map(lambda *xs: jnp.stack(xs), *infos)

            sample_keys = jax.random.split(mcmc_key, num_delete)
            diagnostic_counts["successes"] += 1
            emit_diagnostic(
                reason="cluster_aware",
                summary=summary,
                raw_condition_numbers=raw_condition_numbers,
                standardized_condition_numbers=standardized_condition_numbers,
                standardization_applied=standardization_applied,
                cluster_id=cluster_id_int,
                cluster_size=selected_cluster_size,
            )
            t0 = time.perf_counter()
            outputs = [
                mcmc_kernel(key, jax.tree.map(lambda x, i=i: x[i], start_state))
                for i, key in enumerate(sample_keys)
            ]
            final_states, infos = zip(*outputs)
            new_particles = jax.tree.map(lambda *xs: jnp.stack(xs), *final_states)
            mcmc_infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
            jax.block_until_ready(new_particles.loglikelihood)
            runtime = time.perf_counter() - t0
            diagnostic_counts["cluster_runtime"] += runtime
            diagnostic_counts["cluster_attempts"] += 1
            success, accepted, proposals_per_accepted, improvement = (
                _replacement_acceptance_diagnostics(
                    new_particles, mcmc_infos, loglikelihood_0, num_mcmc_steps
                )
            )
            success_rate = float(jnp.mean(success.astype(jnp.float32)))
            before_cond = float(np.asarray(condition_before_regularization))
            after_cond = float(np.asarray(condition_after_regularization))
            diag = ReplacementDiagnostics(
                mcmc_infos=mcmc_infos,
                strategy="cluster_aware",
                used_strategy="cluster_aware",
                success_rate=jnp.asarray(success_rate),
                accepted_proposals=accepted,
                proposals_per_accepted_replacement=proposals_per_accepted,
                final_loglikelihood_improvement=improvement,
                selected_cluster_size=jnp.full((num_delete,), selected_cluster_size),
                covariance_mode=covariance_mode,
                condition_number_before_regularization=jnp.asarray(before_cond),
                condition_number_after_regularization=jnp.asarray(after_cond),
                runtime_seconds=jnp.asarray(runtime),
                fallback_to_global=jnp.asarray(False),
                diagonal_covariance_fallback=jnp.asarray(diagonal_covariance_fallback),
            )
            if print_diagnostics:
                print(
                    "[replacement] strategy=cluster_aware, used=cluster_aware, "
                    f"attempt={diagnostic_counts['attempts']}, success_rate={success_rate:.4f}, "
                    f"accepted_proposals={np.asarray(accepted).tolist()}, proposals_per_accepted={np.asarray(proposals_per_accepted).tolist()}, "
                    f"final_logL_minus_threshold={np.asarray(improvement).tolist()}, selected_cluster_size={selected_cluster_size}, "
                    f"covariance_mode={covariance_mode}, diagonal_fallback={diagonal_covariance_fallback}, "
                    f"cond_before={before_cond:.6g}, cond_after={after_cond:.6g}, runtime_s={runtime:.6f}"
                )
            if auto_fallback and diagnostic_counts["attempts"] >= warmup_attempts:
                cluster_attempts = max(diagnostic_counts["cluster_attempts"], 1)
                mean_cluster_runtime = (
                    diagnostic_counts["cluster_runtime"] / cluster_attempts
                )
                mean_global_runtime = (
                    diagnostic_counts["global_runtime"]
                    / max(diagnostic_counts["global_attempts"], 1)
                    if diagnostic_counts["global_attempts"]
                    else np.nan
                )
                runtime_bad = (
                    np.isfinite(mean_global_runtime)
                    and mean_cluster_runtime > max_runtime_ratio * mean_global_runtime
                )
                success_bad = success_rate < min_success_rate
                if runtime_bad or success_bad:
                    diagnostic_counts["auto_fallback_active"] = True
                    if print_diagnostics:
                        print(
                            "[cluster-aware] auto_fallback_enabled reason="
                            + ("runtime" if runtime_bad else "success_rate")
                        )
            return new_particles, diag
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
