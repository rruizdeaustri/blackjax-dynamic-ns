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
from typing import Callable, Dict, NamedTuple, Optional

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


def _sample_tree_normal(rng_key: PRNGKey, tree: ArrayTree, scale: float) -> ArrayTree:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(rng_key, len(leaves))
    sampled = [
        scale * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, sampled)


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
            position, momentum_t, boundary_ok = carry
            grad_ll = loglikelihood_grad_fn(position)

            momentum_half = jax.tree.map(
                lambda p, g: p + 0.5 * step_size * g,
                momentum_t,
                grad_ll,
            )
            position_new = jax.tree.map(
                lambda q, p: q + step_size * p,
                position,
                momentum_half,
            )
            ll_new = loglikelihood_fn(position_new)
            step_ok = ll_new > loglikelihood_0

            grad_new = loglikelihood_grad_fn(position_new)
            momentum_new = jax.tree.map(
                lambda p, g: p + 0.5 * step_size * g,
                momentum_half,
                grad_new,
            )
            return (position_new, momentum_new, boundary_ok & step_ok), ll_new

        (final_position, _, boundary_ok), ll_history = jax.lax.scan(
            leapfrog_step,
            (state.position, momentum, jnp.asarray(True)),
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
            crossed_boundary=~boundary_ok,
            start_loglikelihood=start_loglikelihood,
            final_loglikelihood=final_loglikelihood,
            num_integration_steps=jnp.asarray(num_integration_steps),
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
