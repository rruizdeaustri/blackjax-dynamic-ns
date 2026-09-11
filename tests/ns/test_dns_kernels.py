import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.ns import dns, dns_kernels


def _run_parameter_chain(step, state, num_steps, seed):
    keys = jax.random.split(jax.random.key(seed), num_steps)

    def body(particle, key):
        particle, info = step(key, particle, state[1])
        return particle, (particle.position, info)

    return jax.jit(lambda particle, ks: jax.lax.scan(body, particle, ks))(
        state[0], keys
    )


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5])
def test_uniform_constrained_prior_distribution(radius):
    def logprior(x):
        return jnp.where(jnp.abs(x) <= 2.0, 0.0, -jnp.inf)

    def loglikelihood(x):
        return -(x**2)

    threshold = jnp.array(-(radius**2))
    particle = dns.DNSParticleState(jnp.array(0.0), logprior(0.0), loglikelihood(0.0))
    step = dns_kernels.build_constrained_slice_kernel(
        logprior, loglikelihood, cov=jnp.array([[1.0]]), max_steps=12
    )
    (_, (positions, info)) = _run_parameter_chain(
        step, (particle, threshold), 5000, 13
    )
    samples = np.asarray(positions[500:])
    assert np.max(np.abs(samples)) < radius
    assert abs(samples.mean()) < 0.05
    assert abs(np.mean(samples**2) - radius**2 / 3.0) < 0.05
    assert np.mean(np.asarray(info.is_accepted[500:])) > 0.95


def test_nonuniform_prior_is_not_sampled_uniformly():
    beta = 2.0
    radius = 0.8

    def logprior(x):
        return jnp.where(jnp.abs(x) <= 1.0, beta * x, -jnp.inf)

    def loglikelihood(x):
        return -(x**2)

    threshold = jnp.array(-(radius**2))
    particle = dns.DNSParticleState(jnp.array(0.0), logprior(0.0), loglikelihood(0.0))
    step = dns_kernels.build_constrained_slice_kernel(
        logprior, loglikelihood, cov=jnp.array([[1.0]]), max_steps=12
    )
    (_, (positions, _)) = _run_parameter_chain(
        step, (particle, threshold), 9000, 29
    )
    samples = np.asarray(positions[1000:])
    expected_mean = radius / np.tanh(beta * radius) - 1.0 / beta
    assert np.max(np.abs(samples)) < radius
    assert abs(samples.mean() - expected_mean) < 0.04
    assert samples.mean() > 0.25


def test_eager_jit_and_vmap_agree():
    def logprior(x):
        return jnp.where(jnp.abs(x) <= 2.0, -0.5 * x**2, -jnp.inf)

    def loglikelihood(x):
        return -0.25 * x**2

    step = dns_kernels.build_constrained_slice_kernel(
        logprior, loglikelihood, cov=jnp.array([[1.0]])
    )
    particle = dns.DNSParticleState(jnp.array(0.2), logprior(0.2), loglikelihood(0.2))
    threshold = jnp.array(-0.5)
    key = jax.random.key(8)
    with jax.disable_jit():
        eager = step(key, particle, threshold)
    compiled = jax.jit(step)(key, particle, threshold)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    keys = jax.random.split(key, 6)
    mapped = jax.jit(jax.vmap(step, in_axes=(0, None, None)))(
        keys, particle, threshold
    )
    assert mapped[0].position.shape == (6,)
    assert mapped[1].is_accepted.shape == (6,)


@pytest.mark.parametrize(
    "cov",
    [jnp.array([1.0]), jnp.array([[1.0, 0.0]]), jnp.array([[0.0]]), jnp.array([[np.nan]])],
)
def test_invalid_covariance_is_rejected(cov):
    with pytest.raises(ValueError):
        dns_kernels.build_constrained_slice_kernel(
            lambda x: -x**2, lambda x: -x**2, cov=cov
        )


def test_custom_direction_and_covariance_are_mutually_exclusive():
    with pytest.raises(ValueError):
        dns_kernels.build_constrained_slice_kernel(
            lambda x: -x**2,
            lambda x: -x**2,
            cov=jnp.array([[1.0]]),
            generate_slice_direction_fn=lambda key, position: position,
        )
