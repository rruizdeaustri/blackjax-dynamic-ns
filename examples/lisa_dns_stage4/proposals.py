"""Frozen labelled directions. Evidence reconstruction is unvalidated/out of scope."""
from itertools import combinations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns_kernels import build_constrained_slice_kernel


class ProposalInfo(NamedTuple):
    slice: object
    component: object
    labels: object


def build_direction(scales, k=9, per=6):
    """Symmetric fixed-scale directions, independent of current position.

    Constant mixture of invariant straight-line slice kernels preserves the
    constrained prior. Masking selects fixed labels; no sorting or adaptation.
    """
    scales = np.array(scales, copy=True)
    if scales.shape != (k * per,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Require positive finite scales with shape (k*per,)')
    frozen = jnp.asarray(scales)
    pairs = jnp.asarray(list(combinations(range(k), 2)), dtype=jnp.int32)
    if k < 2:
        raise ValueError('Require at least two source labels')

    def draw(key):
        kc, kl, kn = jax.random.split(key, 3)
        component = jax.random.categorical(kc, jnp.log(jnp.array([.2, .4, .4])))
        # A separate selection draw for each component, independent of theta.
        single = jax.random.randint(kl, (), 0, k)
        pair = pairs[jax.random.randint(kl, (), 0, len(pairs))]
        labels = jnp.where(component == 1, jnp.array([single, -1]), pair)
        labels = jnp.where(component == 0, jnp.array([-1, -1]), labels)
        blocks = jnp.arange(k)
        mask = (component == 0) | (blocks == labels[0]) | (blocks == labels[1])
        mask = jnp.repeat(mask, per)
        normal = jax.random.normal(kn, (k * per,), dtype=frozen.dtype) * mask
        direction = frozen * normal / jnp.maximum(jnp.linalg.norm(normal), 1e-30)
        return direction, component, labels

    def direction(key, position):
        return draw(key)[0]
    return direction, draw


def build_parameter_step(prior, likelihood, scales, max_steps=10, max_shrinkage=100):
    direction, draw = build_direction(scales)
    kernel = build_constrained_slice_kernel(prior, likelihood,
        generate_slice_direction_fn=direction, max_steps=max_steps, max_shrinkage=max_shrinkage)
    def step(key, particle, threshold):
        result, info = kernel(key, particle, threshold)
        # Replay only inexpensive proposal RNG, exactly as the reference adapter.
        _, direction_key = jax.random.split(key)
        _, component, labels = draw(direction_key)
        return result, ProposalInfo(info, component, labels)
    return step
