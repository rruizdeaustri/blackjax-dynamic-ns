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
    """Info for one constrained gradient-guided proposal."""

    is_accepted: jnp.ndarray
    gradient_norm: jnp.ndarray


def _tree_l2_norm(tree: ArrayTree) -> jnp.ndarray:
    sq_norm = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        tree,
        initializer=jnp.asarray(0.0),
    )
    return jnp.sqrt(sq_norm)


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
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
) -> Callable:
    """Build a minimal gradient-guided constrained NS replacement kernel."""

    loglikelihood_grad_fn = jax.grad(loglikelihood_fn)

    def constrained_gradient_guided_step_fn(rng_key, state, loglikelihood_0, **params):
        del params
        noise_key = rng_key
        random_direction = jax.tree.map(
            lambda x: jax.random.normal(noise_key, x.shape, dtype=x.dtype),
            state.position,
        )

        gradient = loglikelihood_grad_fn(state.position)
        guided_direction = jax.tree.map(
            lambda g, r: momentum_weight * r + (1.0 - momentum_weight) * g,
            gradient,
            random_direction,
        )

        proposed_position = jax.tree.map(
            lambda x, d: x + step_size * d,
            state.position,
            guided_direction,
        )
        proposed_state = init_state_fn(
            proposed_position, loglikelihood_birth=loglikelihood_0
        )

        is_accepted = proposed_state.loglikelihood > loglikelihood_0
        new_state = jax.lax.cond(
            is_accepted,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        info = ConstrainedGradientGuidedInfo(
            is_accepted=is_accepted,
            gradient_norm=_tree_l2_norm(gradient),
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
