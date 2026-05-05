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
"""Gradient-guided Nested Sampling replacement kernel.

This module provides a minimal constrained replacement kernel inspired by
GGNS-like ideas: combine random momentum with local likelihood gradients to
propose moves that stay above the current NS likelihood contour.
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.types import ArrayTree, PRNGKey

__all__ = [
    "ConstrainedGradientGuidedInfo",
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
]


class ConstrainedGradientGuidedInfo(NamedTuple):
    """Info for one constrained Hamiltonian-style constrained proposal."""

    accepted: jnp.ndarray
    crossed_boundary: jnp.ndarray
    start_loglikelihood: jnp.ndarray
    final_loglikelihood: jnp.ndarray
    num_integration_steps: jnp.ndarray
    num_reflections: jnp.ndarray
    reflection_failures: jnp.ndarray


def _sample_tree_normal(rng_key: PRNGKey, tree: ArrayTree, scale: float) -> ArrayTree:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(rng_key, len(leaves))
    sampled = [
        scale * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, sampled)


def _tree_dot(x: ArrayTree, y: ArrayTree) -> jnp.ndarray:
    return sum(jnp.sum(a * b) for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y)))


def _tree_l2_norm(x: ArrayTree) -> jnp.ndarray:
    return jnp.sqrt(_tree_dot(x, x))


def _interpolate_tree(start: ArrayTree, end: ArrayTree, alpha: jnp.ndarray) -> ArrayTree:
    return jax.tree.map(lambda a, b: a + alpha * (b - a), start, end)


def _find_boundary_point(
    loglikelihood_fn: Callable,
    start_position: ArrayTree,
    end_position: ArrayTree,
    loglikelihood_0: jnp.ndarray,
    num_bisection_steps: int = 8,
) -> Tuple[ArrayTree, jnp.ndarray]:
    lo = jnp.asarray(0.0)
    hi = jnp.asarray(1.0)

    def body(carry, _):
        lo_t, hi_t = carry
        mid = 0.5 * (lo_t + hi_t)
        theta_mid = _interpolate_tree(start_position, end_position, mid)
        ll_mid = loglikelihood_fn(theta_mid)
        above = ll_mid > loglikelihood_0
        lo_new = jnp.where(above, mid, lo_t)
        hi_new = jnp.where(above, hi_t, mid)
        return (lo_new, hi_new), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), xs=None, length=num_bisection_steps)
    alpha = 0.5 * (lo + hi)
    theta_boundary = _interpolate_tree(start_position, end_position, alpha)
    ll_boundary = loglikelihood_fn(theta_boundary)
    return theta_boundary, ll_boundary


def update_inner_kernel_params(
    rng_key: PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """No-op updater for the first GGNS-style implementation."""
    del rng_key, state, info
    if inner_kernel_params is None:
        return {}
    return inner_kernel_params


def build_kernel(
    init_state_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    step_size: float = 0.1,
    momentum_weight: float = 0.5,
    num_integration_steps: int = 4,
    momentum_scale: float = 1.0,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
) -> Callable:
    """Build a constrained Hamiltonian-style NS replacement kernel.

    Notes
    -----
    This uses a safe intermediate strategy: if any leapfrog point crosses the
    nested-sampling likelihood boundary, the full trajectory is rejected.
    True reflective boundary handling is future work.
    """

    loglikelihood_grad_fn = jax.grad(loglikelihood_fn)
    _ = momentum_weight

    def constrained_gradient_guided_step_fn(rng_key, state, loglikelihood_0, **params):
        del params

        momentum = _sample_tree_normal(rng_key, state.position, momentum_scale)

        def leapfrog_step(carry, _):
            position, momentum_t, boundary_ok, num_reflections, reflection_failures, crossed_boundary = carry
            grad_ll = loglikelihood_grad_fn(position)

            momentum_half = jax.tree.map(
                lambda p, g: p + 0.5 * step_size * g,
                momentum_t,
                grad_ll,
            )
            proposed_position = jax.tree.map(
                lambda q, p: q + step_size * p,
                position,
                momentum_half,
            )
            ll_new = loglikelihood_fn(proposed_position)
            crossed = ll_new <= loglikelihood_0

            def reflect(_):
                theta_boundary, ll_boundary = _find_boundary_point(
                    loglikelihood_fn, position, proposed_position, loglikelihood_0
                )
                grad_boundary = loglikelihood_grad_fn(theta_boundary)
                grad_norm = _tree_l2_norm(grad_boundary)
                finite_norm = jnp.isfinite(grad_norm) & (grad_norm > 0.0)

                n_hat = jax.tree.map(lambda g: g / grad_norm, grad_boundary)
                projection = _tree_dot(momentum_half, n_hat)
                reflected_momentum = jax.tree.map(
                    lambda p, n: p - 2.0 * projection * n, momentum_half, n_hat
                )
                alpha = jnp.asarray(1.0)
                theta_reflected = jax.tree.map(
                    lambda qb, pr: qb + alpha * step_size * pr, theta_boundary, reflected_momentum
                )
                ll_reflected = loglikelihood_fn(theta_reflected)
                reflected_valid = ll_reflected > loglikelihood_0
                reflection_success = finite_norm & reflected_valid & jnp.isfinite(ll_boundary)

                new_position = jax.tree.map(
                    lambda qr, qp: jnp.where(reflection_success, qr, qp), theta_reflected, position
                )
                new_momentum = jax.tree.map(
                    lambda mr, mp: jnp.where(reflection_success, mr, mp), reflected_momentum, momentum_half
                )
                new_ll = jnp.where(reflection_success, ll_reflected, loglikelihood_fn(position))
                return (new_position, new_momentum, new_ll, reflection_success)

            def no_reflect(_):
                return (proposed_position, momentum_half, ll_new, jnp.asarray(True))

            position_after, momentum_after, ll_after, step_ok = jax.lax.cond(crossed, reflect, no_reflect, operand=None)

            grad_new = loglikelihood_grad_fn(position_after)
            momentum_new = jax.tree.map(
                lambda p, g: p + 0.5 * step_size * g,
                momentum_after,
                grad_new,
            )
            return (
                position_after,
                momentum_new,
                boundary_ok & step_ok & (ll_after > loglikelihood_0),
                num_reflections + crossed.astype(jnp.int32) * step_ok.astype(jnp.int32),
                reflection_failures + crossed.astype(jnp.int32) * (~step_ok).astype(jnp.int32),
                crossed_boundary | crossed,
            ), ll_after

        (final_position, _, boundary_ok, num_reflections, reflection_failures, crossed_boundary), ll_history = jax.lax.scan(
            leapfrog_step,
            (state.position, momentum, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
            xs=None,
            length=num_integration_steps,
        )
        start_loglikelihood = state.loglikelihood
        final_loglikelihood = ll_history[-1]
        proposed_state = init_state_fn(final_position, loglikelihood_birth=loglikelihood_0)

        accepted = boundary_ok & (final_loglikelihood > loglikelihood_0)
        new_state = jax.lax.cond(accepted, lambda _: proposed_state, lambda _: state, operand=None)
        info = ConstrainedGradientGuidedInfo(
            accepted=accepted,
            crossed_boundary=crossed_boundary,
            start_loglikelihood=start_loglikelihood,
            final_loglikelihood=final_loglikelihood,
            num_integration_steps=jnp.asarray(num_integration_steps),
            num_reflections=num_reflections,
            reflection_failures=reflection_failures,
        )
        return new_state, info

    inner_kernel = update_strategy(
        constrained_gradient_guided_step_fn,
        num_mcmc_steps=num_inner_steps,
        num_delete=num_delete,
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
    init_state_strategy_fn: Callable = init_state_strategy,
    step_size: float = 0.1,
    momentum_weight: float = 0.5,
    num_integration_steps: int = 4,
    momentum_scale: float = 1.0,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
) -> SamplingAlgorithm:
    """Create a minimal GGNS-style nested sampling algorithm."""

    init_state_fn = partial(
        init_state_strategy_fn,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn=init_state_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        step_size=step_size,
        momentum_weight=momentum_weight,
        num_integration_steps=num_integration_steps,
        momentum_scale=momentum_scale,
        delete_fn=delete_fn,
        update_strategy=update_strategy,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
            rng_key=rng_key,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
