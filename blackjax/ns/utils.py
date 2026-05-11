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
"""Utility functions for Nested Sampling post-processing."""

from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.types import Array, ArrayTree, PRNGKey


class NSBatchMetadata(NamedTuple):
    """Metadata collected while running a bounded NS batch.

    Attributes
    ----------
    num_steps
        Number of NS transitions executed.
    num_dead
        Number of dead particles retained in the bounded interval.
    terminated_reason
        Integer code describing why the batch ended:
        0 => reached ``num_steps``
        1 => crossed ``loglikelihood_upper``
    reached_loglikelihood_upper
        Whether an upper bound was specified and reached.
    """

    num_steps: int
    requested_num_steps: int
    num_dead: int
    is_empty: bool
    interval_width: float
    terminated_reason: int
    reached_loglikelihood_upper: bool


class NSBatchResult(NamedTuple):
    """Result container for a bounded nested-sampling batch."""

    dead_points: ArrayTree
    dead_point_loglikelihoods: Array
    dead_particles: Any
    final_live_points: Any
    num_live_points: int
    loglikelihood_lower: float
    loglikelihood_upper: float
    metadata: NSBatchMetadata


class NSMergedMetadata(NamedTuple):
    """Metadata produced when reweighting one or more bounded NS batches."""

    num_batches: int
    num_dead: int
    loglikelihood_min: float
    loglikelihood_max: float
    min_num_live: float
    max_num_live: float


class NSMergedResult(NamedTuple):
    """Merged and reweighted dead-point information from bounded NS batches."""

    dead_points: ArrayTree
    dead_point_loglikelihoods: Array
    dead_particles: Any
    num_live_points: Array
    logX: Array
    logdX: Array
    log_weights: Array
    logZ: Array
    posterior_log_weights: Array
    posterior_weights: Array
    ess: Array
    metadata: NSMergedMetadata


class NSDynamicMetadata(NamedTuple):
    """Metadata for the posterior-focused dynamic NS scheduler."""

    objective: str
    max_batches: int
    initial_num_steps: int
    refinement_num_steps: int
    num_requested_batches: int
    num_executed_batches: int
    num_empty_batches: int
    refinement_interval_diagnostics: tuple["NSRefinementIntervalDiagnostics", ...]


class NSRefinementIntervalDiagnostics(NamedTuple):
    """Diagnostics for dynamic interval selection per refinement batch."""

    selected_lower_threshold: float
    selected_upper_threshold: float
    selected_upper_is_finite: bool
    interval_width: float
    upper_none_reason_code: str
    posterior_weight_at_selected_dead_point: float
    selected_dead_point_index: int
    num_candidate_dead_points_considered: int


class NSDynamicResult(NamedTuple):
    """Structured result for dynamic nested sampling orchestration."""

    batches: tuple[NSBatchResult, ...]
    merged: NSMergedResult
    logZ: Array
    posterior_weights: Array
    ess: Array
    metadata: NSDynamicMetadata


def run_dynamic_posterior_scheduler(
    rng_key: PRNGKey,
    state: NSState,
    step_fn: Callable,
    initial_num_steps: int,
    refinement_num_steps: int,
    max_batches: int,
    objective: str = "posterior",
    min_loglikelihood_interval_width: float = 1e-6,
) -> tuple[NSState, NSDynamicResult]:
    """Run a posterior-focused dynamic bounded nested-sampling schedule."""
    if objective != "posterior":
        raise NotImplementedError(
            "Only objective='posterior' is currently supported"
        )
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    batches = []
    rng_key, batch_key = jax.random.split(rng_key)
    state, first_batch = run_bounded_batch(
        rng_key=batch_key,
        state=state,
        step_fn=step_fn,
        num_steps=initial_num_steps,
        loglikelihood_lower=-jnp.inf,
        loglikelihood_upper=None,
    )
    batches.append(first_batch)
    merged = merge_bounded_batches(batches)

    if min_loglikelihood_interval_width < 0.0:
        raise ValueError("min_loglikelihood_interval_width must be non-negative")

    interval_diagnostics = []
    for _ in range(1, max_batches):
        posterior_weights = merged.posterior_weights
        max_idx = int(jnp.argmax(posterior_weights))
        logL = merged.dead_point_loglikelihoods
        lower = float(logL[max_idx])
        upper = None
        upper_none_reason_code = "unknown"
        if max_idx + 1 < logL.shape[0]:
            candidate_upper = float(logL[max_idx + 1])
            if (candidate_upper - lower) >= min_loglikelihood_interval_width:
                upper = candidate_upper
            else:
                upper_none_reason_code = "next_gap_below_min_width"
        else:
            upper_none_reason_code = "posterior_peak_at_last_dead_point"

        # If we can already detect an invalid scheduling range, stop early.
        if not jnp.isfinite(lower):
            upper_none_reason_code = "no_valid_candidate_interval"
            break

        rng_key, batch_key = jax.random.split(rng_key)
        state, new_batch = run_bounded_batch(
            rng_key=batch_key,
            state=state,
            step_fn=step_fn,
            num_steps=refinement_num_steps,
            loglikelihood_lower=lower,
            loglikelihood_upper=upper,
        )

        # Fallback to a broader lower-tail interval when a selected interval is
        # effectively empty in finite runs.
        if new_batch.metadata.num_dead == 0:
            state, new_batch = run_bounded_batch(
                rng_key=batch_key,
                state=state,
                step_fn=step_fn,
                num_steps=refinement_num_steps,
                loglikelihood_lower=lower,
                loglikelihood_upper=None,
            )
            upper = None
            upper_none_reason_code = "finite_window_empty_fallback"
            if new_batch.metadata.num_dead == 0:
                break
        elif upper is None and upper_none_reason_code == "unknown":
            upper_none_reason_code = "upper_tail_by_design"

        selected_upper_threshold = float(jnp.inf if upper is None else upper)
        interval_diagnostics.append(
            NSRefinementIntervalDiagnostics(
                selected_lower_threshold=float(lower),
                selected_upper_threshold=selected_upper_threshold,
                selected_upper_is_finite=bool(jnp.isfinite(selected_upper_threshold)),
                interval_width=float(selected_upper_threshold - lower),
                upper_none_reason_code=upper_none_reason_code,
                posterior_weight_at_selected_dead_point=float(posterior_weights[max_idx]),
                selected_dead_point_index=max_idx,
                num_candidate_dead_points_considered=int(logL.shape[0]),
            )
        )
        batches.append(new_batch)
        merged = merge_bounded_batches(batches)

    dynamic_result = NSDynamicResult(
        batches=tuple(batches),
        merged=merged,
        logZ=merged.logZ,
        posterior_weights=merged.posterior_weights,
        ess=merged.ess,
        metadata=NSDynamicMetadata(
            objective=objective,
            max_batches=max_batches,
            initial_num_steps=initial_num_steps,
            refinement_num_steps=refinement_num_steps,
            num_requested_batches=max_batches,
            num_executed_batches=len(batches),
            num_empty_batches=sum(int(batch.metadata.is_empty) for batch in batches),
            refinement_interval_diagnostics=tuple(interval_diagnostics),
        ),
    )
    return state, dynamic_result


def summarize_dynamic_result(result: NSDynamicResult) -> str:
    """Return a compact human-readable summary for dynamic NS results."""
    lines = [
        "Dynamic nested sampling summary:",
        f"- logZ: {float(result.logZ):.6f}",
        f"- ESS: {float(result.ess):.2f}",
        f"- batches: {result.metadata.num_executed_batches}/{result.metadata.num_requested_batches}",
        f"- empty batches: {result.metadata.num_empty_batches}",
    ]
    for idx, batch in enumerate(result.batches):
        lines.append(
            "  "
            f"batch {idx}: requested_steps={batch.metadata.requested_num_steps}, "
            f"executed_steps={batch.metadata.num_steps}, dead={batch.metadata.num_dead}, "
            f"empty={batch.metadata.is_empty}, width={batch.metadata.interval_width:.3e}, "
            f"term_reason={batch.metadata.terminated_reason}"
        )
    return "\n".join(lines)


def run_bounded_batch(
    rng_key: PRNGKey,
    state: NSState,
    step_fn: Callable,
    num_steps: int,
    loglikelihood_lower: float,
    loglikelihood_upper: Optional[float] = None,
) -> tuple[NSState, NSBatchResult]:
    """Run a bounded nested-sampling batch.

    This orchestrates repeated calls to an existing NS transition kernel while
    retaining only dead particles with death log-likelihoods in the interval
    ``[loglikelihood_lower, loglikelihood_upper)``. If ``loglikelihood_upper``
    is not provided, only the lower bound is enforced.

    Notes
    -----
    - The constrained replacement-kernel interface is reused unchanged:
      ``step_fn(rng_key, state) -> (new_state, info)``.
    - Python-level orchestration is used for dynamic bookkeeping; each inner
      NS step remains JAX-compatible.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    upper = jnp.inf if loglikelihood_upper is None else float(loglikelihood_upper)
    dead_batches = []
    num_executed_steps = 0
    reached_upper = False

    for _ in range(num_steps):
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step_fn(step_key, state)
        num_executed_steps += 1

        dead_loglik = info.particles.loglikelihood
        within_bounds = (dead_loglik >= loglikelihood_lower) & (dead_loglik < upper)
        dead_batches.append(jax.tree.map(lambda x: x[within_bounds], info.particles))

        if loglikelihood_upper is not None:
            min_live = jnp.min(state.particles.loglikelihood)
            if bool(min_live >= upper):
                reached_upper = True
                break

    if dead_batches:
        dead_particles = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *dead_batches
        )
    else:
        dead_particles = jax.tree.map(lambda x: x[:0], state.particles)

    metadata = NSBatchMetadata(
        num_steps=num_executed_steps,
        requested_num_steps=num_steps,
        num_dead=int(dead_particles.loglikelihood.shape[0]),
        is_empty=bool(dead_particles.loglikelihood.shape[0] == 0),
        interval_width=float(upper - loglikelihood_lower),
        terminated_reason=1 if reached_upper else 0,
        reached_loglikelihood_upper=reached_upper,
    )

    result = NSBatchResult(
        dead_points=dead_particles.position,
        dead_point_loglikelihoods=dead_particles.loglikelihood,
        dead_particles=dead_particles,
        final_live_points=state.particles,
        num_live_points=int(state.particles.loglikelihood.shape[0]),
        loglikelihood_lower=float(loglikelihood_lower),
        loglikelihood_upper=float(upper),
        metadata=metadata,
    )
    return state, result


def merge_bounded_batches(
    batches: Sequence[NSBatchResult],
    beta: float = 1.0,
) -> NSMergedResult:
    """Merge and reweight one or more bounded NS batches.

    The merged dead points are sorted by death log-likelihood, effective live-point
    counts are recomputed from the birth/death process, and deterministic
    shrinkage-based log-weights are recomputed.
    """
    if len(batches) == 0:
        raise ValueError("Expected at least one NSBatchResult")

    dead_particles = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *[batch.dead_particles for batch in batches],
    )
    if dead_particles.loglikelihood.shape[0] == 0:
        raise ValueError("Cannot merge empty batches with zero dead points")

    sort_idx = jnp.argsort(dead_particles.loglikelihood)
    dead_particles = jax.tree.map(lambda x: x[sort_idx], dead_particles)
    merged_info = NSInfo(particles=dead_particles, update_info={})
    num_live = compute_num_live(merged_info).astype(jnp.float32)

    # Deterministic nested-sampling shrinkage approximation:
    # E[log t_i] = -1 / n_live,i  and  dX_i = X_{i-1} - X_i.
    delta_logX = -1.0 / num_live
    logX = jnp.cumsum(delta_logX)
    logX_prev = jnp.concatenate([jnp.array([0.0], dtype=logX.dtype), logX[:-1]])
    logdX = logX_prev + log1mexp(logX - logX_prev)

    log_weights = logdX + beta * dead_particles.loglikelihood
    logZ = jax.scipy.special.logsumexp(log_weights)
    posterior_log_weights = log_weights - logZ
    posterior_weights = jnp.exp(posterior_log_weights)
    ess = jnp.exp(-jax.scipy.special.logsumexp(2.0 * posterior_log_weights))

    metadata = NSMergedMetadata(
        num_batches=len(batches),
        num_dead=int(dead_particles.loglikelihood.shape[0]),
        loglikelihood_min=float(dead_particles.loglikelihood[0]),
        loglikelihood_max=float(dead_particles.loglikelihood[-1]),
        min_num_live=float(jnp.min(num_live)),
        max_num_live=float(jnp.max(num_live)),
    )

    return NSMergedResult(
        dead_points=dead_particles.position,
        dead_point_loglikelihoods=dead_particles.loglikelihood,
        dead_particles=dead_particles,
        num_live_points=num_live,
        logX=logX,
        logdX=logdX,
        log_weights=log_weights,
        logZ=logZ,
        posterior_log_weights=posterior_log_weights,
        posterior_weights=posterior_weights,
        ess=ess,
        metadata=metadata,
    )


def log1mexp(x: Array) -> Array:
    """Computes log(1 - exp(x)) in a numerically stable way."""
    return jnp.where(
        x > -0.6931472,  # approx log(2)
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_num_live(info: NSInfo) -> Array:
    """Compute the effective number of live points at each death contour.

    When doing batch deletions, the jump in energy level can be smoothed by
    transforming 1 jump of size k into k jumps of size 1. This function computes
    the effective population size associated with this transformation.

    Returns
    -------
    Array
        An array where each element `num_live[j]` is the effective number of live
        points `m*_i` when the j-th particle (in the sorted list of dead particles)
        was considered "dead".
    """
    birth_logL = info.particles.loglikelihood_birth
    death_logL = info.particles.loglikelihood

    birth_events = jnp.column_stack((birth_logL, jnp.ones_like(birth_logL, dtype=int)))
    death_events = jnp.column_stack((death_logL, -jnp.ones_like(death_logL, dtype=int)))
    combined = jnp.concatenate([birth_events, death_events], axis=0)
    logL_col = combined[:, 0]
    n_col = combined[:, 1]
    not_nan_sort_key = ~jnp.isnan(logL_col)
    logL_sort_key = logL_col
    n_sort_key = n_col
    sorted_indices = jnp.lexsort((n_sort_key, logL_sort_key, not_nan_sort_key))
    sorted_n_col = n_col[sorted_indices]
    cumsum = jnp.cumsum(sorted_n_col)
    cumsum = jnp.maximum(cumsum, 0)
    death_mask_sorted = sorted_n_col == -1
    num_live = cumsum[death_mask_sorted] + 1
    return num_live


def logX(rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100) -> tuple[Array, Array]:
    """Simulate the stochastic evolution of log prior volumes.

    Wraps the effective population size in `compute_num_live`, along with stochastic
    simulation of the log prior shrinkage associated with each deleted particle.


    Parameters
    ----------
    rng_key
        A JAX PRNG key for generating uniform random variates.
    dead_info
        An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
        and `loglikelihood` for all dead particles accumulated during an NS run.
        It's assumed these particles are already sorted by their death log-likelihood.
    shape
        The shape of Monte Carlo samples to generate for the stochastic
        log-volume sequence. Each sample represents one possible path of
        volume shrinkage. Default is 100.

    Returns
    -------
    tuple[Array, Array]
        - `logX_cumulative`: An array of shape `(num_dead_particles, *shape)`
          containing `shape` simulated sequences of cumulative log prior volumes `log(X_i)`.
        - `log_dX_elements`: An array of shape `(num_dead_particles, *shape)`
          containing `shape` simulated sequences of log prior volume elements `log(dX_i)`.
          `dX_i` is approximately `X_i - X_{i+1}`.
    """
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(
        subkey,
        shape=(dead_info.particles.loglikelihood.shape[0], shape),
    )
    r = jax.lax.log1p(jax.lax.neg(u))
    num_live = compute_num_live(dead_info)
    t = r / num_live[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate([logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0)
    log_diff = logXm - logXp
    logdX = log1mexp(log_diff) + logXp - jnp.log(2)
    return logX, logdX


def log_weights(
    rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100, beta: float = 1.0
) -> Array:
    """Calculate the log importance weights for Nested Sampling results.

    Parameters
    ----------
    rng_key
        A JAX PRNG key for simulating `log(dX_i)`.
    dead_info
        An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
        and `loglikelihood` for all dead particles.
    shape
        The shape of Monte Carlo samples to use for simulating `log(dX_i)`.
        Default is 100.
    beta
        The inverse temperature. Typically 1.0 for standard evidence calculation.
        Allows for reweighting to different temperatures.

    Returns
    -------
    Array
        An array of log importance weights, shape `(num_dead_particles, *shape)`.
        The original order of particles in `dead_info` is preserved.
    """
    sort_indices = jnp.argsort(dead_info.particles.loglikelihood)
    unsort_indices = jnp.empty_like(sort_indices)
    unsort_indices = unsort_indices.at[sort_indices].set(jnp.arange(len(sort_indices)))
    dead_info_sorted = jax.tree.map(lambda x: x[sort_indices], dead_info)
    _, log_dX = logX(rng_key, dead_info_sorted, shape)
    log_w = log_dX + beta * dead_info_sorted.particles.loglikelihood[..., jnp.newaxis]
    return log_w[unsort_indices]


def finalise(live: NSState, dead: list[NSInfo], update_info: bool = True) -> NSInfo:
    """Combines the history of dead particle information with the final live points.

    Parameters
    ----------
    live
        The final `NSState` of the Nested Sampler, containing the live particles.
    dead
        A list of `NSInfo` objects, where each object contains information
        about the particles that "died" at one step of the NS algorithm.

    Returns
    -------
    NSInfo
        A single `NSInfo` object where all fields are concatenations of the
        corresponding fields from `dead` and the final live points.
        The `update_info` from the last element of `dead` is used
        for the final live points' `update_info` (as a placeholder).
    """

    if update_info:
        update_infos = [d.update_info for d in dead]
        final_update_info = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *update_infos
        )
    else:
        final_update_info = None

    particles = [d.particles for d in dead] + [live.particles]
    final_particles = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *particles
    )
    return NSInfo(final_particles, final_update_info)


def ess(rng_key: PRNGKey, dead: NSInfo) -> Array:
    """Computes the Effective Sample Size (ESS) from log-weights.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used by `log_weights`.
    dead
        An `NSInfo` object containing the full set of dead (and final live)
        particles, typically the output of `finalise`.

    Returns
    -------
    Array
        The mean Effective Sample Size, a scalar float.
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    ess = jnp.exp(2 * l_sum_w - l_sum_w_sq)
    return ess


def sample(rng_key: PRNGKey, dead: NSInfo, shape: int = 1000) -> ArrayTree:
    """Resamples particles according to their importance weights.

    Returns
    -------
    ArrayTree
        A PyTree of resampled particles, where each leaf has `shape`.
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead.particles.loglikelihood.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(shape,),
        replace=True,
    )
    return jax.tree.map(lambda leaf: leaf[indices], dead.particles)


def get_first_row(x: ArrayTree) -> ArrayTree:
    """Extracts the first "row" (element along the leading axis) of each leaf in a PyTree.

    This is typically used to get a single particle's structure or values from
    a PyTree representing a collection of particles, where the leading dimension
    of each leaf array corresponds to the particle index.

    Parameters
    ----------
    x
        A PyTree of arrays, where each leaf array has a leading dimension.

    Returns
    -------
    ArrayTree
        A PyTree with the same structure as `x`, but where each leaf is the
        first slice `leaf[0]` of the corresponding leaf in `x`.
    """
    return jax.tree.map(lambda x: x[0], x)


def uniform_prior(
    rng_key: PRNGKey, num_live: int, bounds: Dict[str, Tuple[float, float]]
) -> Tuple[ArrayTree, Callable]:
    """Helper function to create a uniform prior for parameters.

    This function generates a set of initial parameter samples uniformly
    distributed within specified bounds. It also provides a log-prior
    function that computes the log-prior probability for a given set of
    parameters.

    Parameters
    ----------
    rng_key
        A JAX PRNG key for random number generation.
    num_live
        The number of live particles to sample.
    bounds
        A dictionary mapping parameter names to their bounds (tuples of min and max).
        Each parameter will be sampled uniformly within these bounds.
        Example: {'param1': (0.0, 1.0), 'param2': (-5.0, 5.0)}

    Returns
    -------
    tuple
        - `particles`: A PyTree of sampled parameters, where each leaf has shape `(num_live,)`.
        - `logprior_fn`: A function that computes the log-prior probability
          for a given set of parameters.
    """

    def logprior_fn(params):
        logprior = 0.0
        for p, (a, b) in bounds.items():
            x = params[p]
            logprior += jax.scipy.stats.uniform.logpdf(x, a, b - a)
        return logprior

    def prior_sample(rng_key):
        init_keys = jax.random.split(rng_key, len(bounds))
        params = {}
        for rng_key, (p, (a, b)) in zip(init_keys, bounds.items()):
            params[p] = jax.random.uniform(rng_key, minval=a, maxval=b)
        return params

    init_keys = jax.random.split(rng_key, num_live)
    particles = jax.vmap(prior_sample)(init_keys)

    return particles, logprior_fn
