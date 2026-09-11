"""Independent finite-state oracles for the frozen DNS transition kernel."""

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.ns import dns


def make_discrete_problem(incorrect_masses=False, nonuniform_weights=False):
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    loglikelihood = np.array([-3.0, -1.0, 1.0, 2.0])
    thresholds = np.array([-np.inf, -2.0, 0.0])
    membership = loglikelihood[:, None] > thresholds
    masses = prior @ membership
    estimated = np.array([1.0, 0.5, 0.15]) if incorrect_masses else masses
    weights = np.array([0.2, 0.3, 0.5]) if nonuniform_weights else np.ones(3) / 3
    levels = dns.create_levels(thresholds, np.log(estimated), np.log(weights))
    states = [(x, j) for x, j in itertools.product(range(4), range(3)) if membership[x, j]]
    target = np.array([prior[x] * weights[j] / estimated[j] for x, j in states])
    target /= target.sum()
    return prior, loglikelihood, membership, masses, weights, levels, states, target


@pytest.mark.parametrize("incorrect_masses", [False, True])
@pytest.mark.parametrize("nonuniform_weights", [False, True])
def test_exact_finite_state_stationarity(incorrect_masses, nonuniform_weights):
    prior, ll, membership, masses, weights, levels, states, target = make_discrete_problem(
        incorrect_masses, nonuniform_weights
    )
    size = len(states)
    lookup = {state: i for i, state in enumerate(states)}
    parameter_matrix = np.zeros((size, size))
    level_matrix = np.zeros((size, size))
    for row, (x, j) in enumerate(states):
        for y in range(len(prior)):
            if membership[y, j]:
                parameter_matrix[row, lookup[y, j]] = prior[y] / masses[j]
        for k in (j - 1, j + 1):
            eligible, probability = dns.level_move_probability(levels, ll[x], j, k)
            # Independent oracle: do not derive the expected answer from the helper.
            expected = 0.0
            if 0 <= k < 3 and membership[x, k]:
                expected = min(1.0, (weights[k] / np.exp(levels.log_mass[k])) /
                               (weights[j] / np.exp(levels.log_mass[j])))
            np.testing.assert_allclose(probability, expected, rtol=2e-6)
            assert bool(eligible) == (0 <= k < 3 and membership[x, max(0, min(k, 2))])
            if expected:
                level_matrix[row, lookup[x, k]] += 0.5 * float(probability)
            level_matrix[row, row] += 0.5 * (1.0 - float(probability))

    for matrix in (parameter_matrix, level_matrix, parameter_matrix @ level_matrix):
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=2e-7)
        np.testing.assert_allclose(target @ matrix, target, atol=2e-7)
    flow = target[:, None] * level_matrix
    np.testing.assert_allclose(flow, flow.T, atol=2e-7)
    occupancy = np.array([sum(target[i] for i, (_, k) in enumerate(states) if k == j)
                          for j in range(3)])
    expected_occupancy = weights * masses / np.exp(levels.log_mass)
    expected_occupancy /= expected_occupancy.sum()
    np.testing.assert_allclose(occupancy, expected_occupancy, atol=2e-7)
    if incorrect_masses:
        assert not np.allclose(occupancy, weights)


def test_exact_e_fold_acceptance_and_boundaries():
    levels = dns.create_levels([-np.inf, -2.0, -1.0], [0.0, -1.0, -2.0], [0.0] * 3)
    for j in (0, 1):
        eligible, probability = dns.level_move_probability(levels, 0.0, j, j + 1)
        assert eligible
        assert probability == 1.0
    for j in (1, 2):
        eligible, probability = dns.level_move_probability(levels, 0.0, j, j - 1)
        assert eligible
        np.testing.assert_allclose(probability, np.exp(-1), rtol=1e-6)
    for ll, j, k in [(-2.0, 0, 1), (-3.0, 0, 1), (0.0, 0, -1), (0.0, 2, 3)]:
        eligible, probability = dns.level_move_probability(levels, ll, j, k)
        assert not eligible
        assert probability == 0.0


def test_sampled_level_kernel_matches_finite_state_oracle():
    _, ll, _, _, _, levels, states, _ = make_discrete_problem(True, True)
    step = dns.build_level_kernel(levels)
    keys = jax.random.split(jax.random.key(19), 24000)
    for x, j in states:
        state = dns.DNSState(dns.DNSParticleState(jnp.array(x), jnp.array(0.0), jnp.array(ll[x])),
                             jnp.array(j, dtype=jnp.int32))
        output, info = jax.jit(jax.vmap(step, in_axes=(0, None)))(keys, state)
        expected = np.zeros(3)
        for k in (j - 1, j + 1):
            _, probability = dns.level_move_probability(levels, ll[x], j, k)
            if 0 <= k < 3:
                expected[k] += 0.5 * float(probability)
            expected[j] += 0.5 * (1 - float(probability))
        observed = np.bincount(np.asarray(output.level_index), minlength=3) / len(keys)
        np.testing.assert_allclose(observed, expected, atol=0.015)
        assert np.all(np.asarray(output.particle.position) == x)
        assert np.all(np.asarray(output.particle.loglikelihood) == ll[x])
        np.testing.assert_array_equal(info.previous_level, jnp.full(len(keys), j))


def test_one_level_and_zero_likelihood_prior_support():
    levels = dns.create_levels([-np.inf], [0.0], [0.0])
    state = dns.init(jnp.array(0.5), lambda x: jnp.array(0.0), lambda x: -jnp.inf, levels)
    step = dns.build_level_kernel(levels)
    output, info = jax.jit(jax.vmap(step, in_axes=(0, None)))(
        jax.random.split(jax.random.key(2), 100), state
    )
    assert np.all(output.level_index == 0)
    assert not np.any(info.is_accepted)
    assert not np.any(info.is_eligible)
    assert np.all(info.acceptance_probability == 0)
    assert dns.is_eligible(levels, -jnp.inf, 0)


@pytest.mark.parametrize("thresholds,masses,weights", [
    ([], [], []),
    ([0.0], [0.0], [0.0]),
    ([-np.inf, 0.0, 0.0], [0.0, -1.0, -2.0], [0.0] * 3),
    ([-np.inf, np.nan], [0.0, -1.0], [0.0] * 2),
    ([-np.inf, np.inf], [0.0, -1.0], [0.0] * 2),
    ([-np.inf, 0.0], [0.0, 1.0], [0.0] * 2),
    ([-np.inf], [-1.0], [0.0]),
    ([-np.inf, 0.0], [0.0, -np.inf], [0.0] * 2),
    ([-np.inf], [0.0], [-np.inf]),
    ([-np.inf], [0.0], [0.0, 0.0]),
])
def test_invalid_levels(thresholds, masses, weights):
    with pytest.raises(ValueError):
        dns.create_levels(thresholds, masses, weights)


@pytest.mark.parametrize("index,lp,ll", [(-1, 0.0, 0.0), (2, 0.0, 0.0),
    (1, 0.0, -1.0), (0, -np.inf, 0.0), (0, 0.0, np.nan), (0, 0.0, np.inf)])
def test_invalid_initial_state(index, lp, ll):
    levels = dns.create_levels([-np.inf, -1.0], [0.0, -1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        dns.init(jnp.array(0.0), lambda x: jnp.array(lp), lambda x: jnp.array(ll),
                 levels, level_index=index)


class ParameterInfo(NamedTuple):
    previous_position: object


def test_full_discrete_sweep_matches_independent_transition_matrix():
    prior, ll, membership, masses, weights, levels, states, _ = make_discrete_problem(True, True)

    def parameter_step(key, particle, threshold):
        logits = jnp.where(jnp.asarray(ll) > threshold, jnp.log(jnp.asarray(prior)), -jnp.inf)
        x = jax.random.categorical(key, logits)
        new = dns.DNSParticleState(x, jnp.log(jnp.asarray(prior)[x]), jnp.asarray(ll)[x])
        return new, ParameterInfo(particle.position)

    step = dns.build_kernel(levels, parameter_step)
    batched_step = jax.jit(jax.vmap(step, in_axes=(0, None)))
    keys = jax.random.split(jax.random.key(902), 24000)
    for x, j in states:
        state = dns.init(jnp.array(x), lambda x: jnp.log(jnp.asarray(prior)[x]),
                         lambda x: jnp.asarray(ll)[x], levels, j)
        output, info = batched_step(keys, state)
        expected = np.zeros((4, 3))
        for y in range(4):
            if not membership[y, j]:
                continue
            conditional = prior[y] / masses[j]
            for k in (j - 1, j + 1):
                probability = 0.0
                if 0 <= k < 3 and membership[y, k]:
                    probability = min(1.0, (weights[k] / np.exp(levels.log_mass[k])) /
                                      (weights[j] / np.exp(levels.log_mass[j])))
                    expected[y, k] += conditional * 0.5 * probability
                expected[y, j] += conditional * 0.5 * (1.0 - probability)
        observed = np.zeros((4, 3))
        np.add.at(observed, (np.asarray(output.particle.position), np.asarray(output.level_index)),
                  1.0 / len(keys))
        np.testing.assert_allclose(observed, expected, atol=0.015)
        assert np.all(info.parameter_is_valid)
        assert np.all(info.parameter_info.previous_position == x)


def gaussian_problem():
    """pi=Uniform[-1,1], L=exp(-x*x/2), radii=X_j=exp(-j)."""
    radii = np.exp(-np.arange(4))
    thresholds = -0.5 * radii**2
    thresholds[0] = -np.inf
    levels = dns.create_levels(thresholds, -np.arange(4, dtype=float), np.zeros(4))

    def logprior(x):
        return jnp.where(jnp.abs(x) <= 1, -jnp.log(2.0), -jnp.inf)

    def loglikelihood(x):
        return -0.5 * x**2

    def parameter_step(key, particle, threshold):
        radius = jnp.minimum(1.0, jnp.sqrt(-2.0 * threshold))
        # An exact constrained-prior draw, independent of the current position.
        x = jax.random.uniform(key, minval=-radius, maxval=radius)
        # Float arithmetic can hit a strict boundary. Keep the previous point in
        # that null-continuum event, as a rejection transition would.
        eligible = jnp.isneginf(threshold) | (loglikelihood(x) > threshold)
        x = jnp.where(eligible, x, particle.position)
        return dns.DNSParticleState(x, logprior(x), loglikelihood(x)), ParameterInfo(
            particle.position
        )

    return levels, radii, logprior, loglikelihood, parameter_step


def test_continuous_analytic_occupancy_conditionals_and_flows():
    levels, radii, logprior, loglikelihood, parameter_step = gaussian_problem()
    step = dns.build_kernel(levels, parameter_step)
    state = dns.init(jnp.array(0.0), logprior, loglikelihood, levels)
    initial = jax.tree.map(lambda x: jnp.broadcast_to(x, (32,) + x.shape), state)

    def sweep(states, key):
        new, info = jax.vmap(step)(jax.random.split(key, 32), states)
        return new, (new.particle.position, new.level_index, info)

    _, (positions, indices, info) = jax.jit(lambda s, ks: jax.lax.scan(sweep, s, ks))(
        initial, jax.random.split(jax.random.key(746), 6500)
    )
    positions = np.asarray(positions[500:])
    indices = np.asarray(indices[500:])
    level_info = jax.tree.map(lambda x: np.asarray(x[500:, :, 0]), info.level_info)
    assert np.all(info.parameter_is_valid)
    occupancy = np.bincount(indices.ravel(), minlength=4) / indices.size
    np.testing.assert_allclose(occupancy, 0.25, atol=0.015)

    for j in range(4):
        samples = positions[indices == j]
        # Conditional prior is uniform within the analytic radius, not Gaussian.
        assert np.max(np.abs(samples)) <= radii[j] + 1e-7
        normalized = samples / radii[j]
        assert abs(normalized.mean()) < 0.035
        assert abs(np.mean(normalized**2) - 1 / 3) < 0.02
        grid = np.linspace(-0.9, 0.9, 19)
        empirical_cdf = np.array([np.mean(normalized <= value) for value in grid])
        np.testing.assert_allclose(empirical_cdf, (grid + 1) / 2, atol=0.025)

    for j in range(3):
        up = (level_info.previous_level == j) & (level_info.proposed_level == j + 1)
        down = (level_info.previous_level == j + 1) & (level_info.proposed_level == j)
        assert up.sum() > 10000 and down.sum() > 10000
        assert abs(level_info.is_eligible[up].mean() - np.exp(-1)) < 0.02
        assert np.all(level_info.is_accepted[up & level_info.is_eligible])
        assert np.all(level_info.acceptance_probability[up & level_info.is_eligible] == 1)
        assert np.all(level_info.is_eligible[down])
        np.testing.assert_allclose(level_info.acceptance_probability[down], np.exp(-1), rtol=1e-6)
        assert abs(level_info.is_accepted[down].mean() - np.exp(-1)) < 0.02
        # Per visit to either end of an edge, successful crossing probability is r/2.
        for current, mask in ((j, up), (j + 1, down)):
            frequency = level_info.is_accepted[mask].sum() / (
                level_info.previous_level == current
            ).sum()
            assert abs(frequency - np.exp(-1) / 2) < 0.015


def test_eager_jit_and_vmap_agree_with_pytree_positions():
    levels = dns.create_levels([-np.inf, -1.0], [0.0, -1.0], [0.0, 0.0])

    def logprior(position):
        return jnp.where(jnp.abs(position["x"]) <= 1, -jnp.log(2.0), -jnp.inf)

    def loglikelihood(position):
        return -0.5 * position["x"]**2

    def parameter_step(key, particle, threshold):
        position = {"x": jax.random.uniform(key, minval=-1.0, maxval=1.0)}
        return dns.DNSParticleState(position, logprior(position), loglikelihood(position)), ()

    state = dns.init({"x": jnp.array(0.0)}, logprior, loglikelihood, levels)
    step = dns.build_kernel(levels, parameter_step, num_inner_steps=2, num_level_steps=3)
    key = jax.random.key(42)
    with jax.disable_jit():
        eager = step(key, state)
    compiled = jax.jit(step)(key, state)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    assert compiled[1].parameter_is_valid.shape == (2,)
    assert compiled[1].level_info.is_accepted.shape == (3,)
    keys = jax.random.split(key, 4)
    mapped = jax.jit(jax.vmap(step, in_axes=(0, None)))(keys, state)
    serial = [step(k, state) for k in keys]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *serial)
    for a, b in zip(jax.tree.leaves(mapped), jax.tree.leaves(stacked)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    assert len(np.unique(np.asarray(mapped[0].particle.position["x"]))) == 4


def test_parameter_contract_violation_is_reported():
    levels = dns.create_levels([-np.inf, 0.0], [0.0, -1.0], [0.0, 0.0])
    state = dns.init(jnp.array(1.0), lambda x: jnp.array(0.0), lambda x: x, levels, 1)

    def broken_parameter_step(key, particle, threshold):
        return particle._replace(position=jnp.array(-1.0), loglikelihood=jnp.array(-1.0)), ()

    _, info = jax.jit(dns.build_kernel(levels, broken_parameter_step))(jax.random.key(0), state)
    assert not np.any(info.parameter_is_valid)


def test_globally_inaccessible_upper_level():
    levels = dns.create_levels([-np.inf, 1.0], [0.0, -1.0], [0.0, 0.0])
    state = dns.init(jnp.array(0.0), lambda x: jnp.array(0.0), lambda x: -x**2, levels)

    def parameter_step(key, particle, threshold):
        return particle, ()

    step = dns.build_kernel(levels, parameter_step)
    output, info = jax.jit(jax.vmap(step, in_axes=(0, None)))(
        jax.random.split(jax.random.key(193), 1000), state
    )
    assert np.all(output.level_index == 0)
    assert np.any(info.level_info.proposed_level == 1)
    assert not np.any(info.level_info.is_eligible)
    assert not np.any(info.level_info.is_accepted)


def test_eager_jit_level_edges_agree():
    levels = dns.create_levels([-np.inf, -2.0, -1.0], [0.0, -1.0, -2.0], [0.0] * 3)
    step = dns.build_level_kernel(levels)
    compiled_step = jax.jit(step)
    for j, ll in [(0, -np.inf), (0, -2.0), (0, 0.0), (1, -1.0), (2, 0.0)]:
        state = dns.init(jnp.array(ll), lambda x: jnp.array(0.0), lambda x: x, levels, j)
        for seed in range(5):
            key = jax.random.key(seed)
            with jax.disable_jit():
                eager = step(key, state)
            compiled = compiled_step(key, state)
            for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("inner,level", [(0, 1), (1, 0), (-1, 1), (1, 1.5), (True, 1)])
def test_invalid_step_counts(inner, level):
    levels = dns.create_levels([-np.inf], [0.0], [0.0])
    with pytest.raises(ValueError):
        dns.build_kernel(levels, lambda *args: None, inner, level)
