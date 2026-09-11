import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns, dns_diagnostics, dns_kernels


BOUND = 6.0
MODE = 1.0
SIGMA = 0.3
NUM_LEVELS = 5
LOCAL_DIRECTION_SCALE = 0.5


def double_well_problem():
    masses = np.exp(-np.arange(NUM_LEVELS, dtype=float))
    radii = np.zeros(NUM_LEVELS)
    thresholds = np.full(NUM_LEVELS, -np.inf)
    for j in range(1, NUM_LEVELS):
        if masses[j] >= 2.0 * MODE / BOUND:
            radii[j] = BOUND * masses[j] - MODE
        else:
            radii[j] = 0.5 * BOUND * masses[j]
        thresholds[j] = -0.5 * (radii[j] / SIGMA) ** 2

    levels = dns.create_levels(thresholds, np.log(masses), np.zeros(NUM_LEVELS))

    def logprior(x):
        return jnp.where(jnp.abs(x) <= BOUND, -jnp.log(2.0 * BOUND), -jnp.inf)

    def loglikelihood(x):
        return -0.5 * ((jnp.abs(x) - MODE) / SIGMA) ** 2

    def local_direction(key, position):
        del position
        return jnp.where(
            jax.random.bernoulli(key), LOCAL_DIRECTION_SCALE, -LOCAL_DIRECTION_SCALE
        )

    parameter_step = dns_kernels.build_constrained_slice_kernel(
        logprior,
        loglikelihood,
        generate_slice_direction_fn=local_direction,
        max_steps=10,
        max_shrinkage=100,
    )
    return levels, masses, radii, logprior, loglikelihood, parameter_step


def run_dns(seed, start_sign, num_walkers=8, num_sweeps=4000):
    levels, masses, radii, logprior, loglikelihood, parameter_step = double_well_problem()
    state = dns.init(
        jnp.array(float(start_sign) * MODE),
        logprior,
        loglikelihood,
        levels,
        level_index=NUM_LEVELS - 1,
    )
    states = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_walkers,) + x.shape), state
    )
    step = dns.build_kernel(levels, parameter_step)

    def sweep(current, key):
        keys = jax.random.split(key, num_walkers)
        new, info = jax.vmap(step)(keys, current)
        return new, (
            new.particle.position,
            new.particle.loglikelihood,
            new.level_index,
            info,
        )

    _, trace = jax.jit(lambda s, ks: jax.lax.scan(sweep, s, ks))(
        states, jax.random.split(jax.random.key(seed), num_sweeps)
    )
    return levels, masses, radii, trace


def run_fixed_high(seed, num_walkers=8, num_sweeps=6500, high_level=3):
    levels, _, _, logprior, loglikelihood, parameter_step = double_well_problem()
    particle = dns.DNSParticleState(
        jnp.array(-MODE), logprior(-MODE), loglikelihood(-MODE)
    )
    particles = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_walkers,) + x.shape), particle
    )
    threshold = levels.loglikelihood[high_level]

    def sweep(current, key):
        keys = jax.random.split(key, num_walkers)
        new, info = jax.vmap(parameter_step, in_axes=(0, 0, None))(
            keys, current, threshold
        )
        return new, (new.position, info)

    _, trace = jax.jit(lambda s, ks: jax.lax.scan(sweep, s, ks))(
        particles, jax.random.split(jax.random.key(seed), num_sweeps)
    )
    return trace


def _expected_second_moment(level, masses, radii):
    if level == 0:
        return BOUND**2 / 3.0
    if masses[level] >= 2.0 * MODE / BOUND:
        outer = MODE + radii[level]
        return outer**2 / 3.0
    return MODE**2 + radii[level] ** 2 / 3.0


def test_frozen_dns_stationarity_and_initial_mode_independence():
    burn = 500
    fractions = []
    for seed, start_sign in ((11, -1), (17, 1)):
        levels, masses, radii, trace = run_dns(seed, start_sign)
        positions, loglikelihood, indices, info = trace
        positions = np.asarray(positions[burn:])
        loglikelihood = np.asarray(loglikelihood[burn:])
        indices = np.asarray(indices[burn:])
        assert np.all(np.asarray(info.parameter_is_valid))

        occupancy = dns_diagnostics.level_occupancy(indices, NUM_LEVELS)
        np.testing.assert_allclose(
            occupancy, np.full(NUM_LEVELS, 1.0 / NUM_LEVELS), atol=0.04
        )

        for level in range(NUM_LEVELS):
            samples = positions[indices == level]
            assert samples.size > 1000
            expected_second = _expected_second_moment(level, masses, radii)
            tolerance = max(0.08, 0.06 * expected_second)
            assert abs(np.mean(samples**2) - expected_second) < tolerance
            assert abs(np.mean(samples > 0) - 0.5) < 0.08

        fractions.append(float(np.mean(positions > 0)))
        highest = np.asarray(
            dns_diagnostics.highest_eligible_level(levels, loglikelihood)
        )
        assert highest.shape == positions.shape

        level_info = jax.tree.map(
            lambda x: np.asarray(x[burn:]), info.level_info
        )
        edge = dns_diagnostics.edge_statistics(level_info, NUM_LEVELS)
        expected = np.exp(-1.0)
        for j in range(NUM_LEVELS - 1):
            assert edge["up_attempts"][j] > 200
            assert edge["down_attempts"][j] > 200
            up_eligibility = edge["up_eligible"][j] / edge["up_attempts"][j]
            down_acceptance = edge["down_accepted"][j] / edge["down_attempts"][j]
            assert abs(up_eligibility - expected) < 0.06
            assert abs(down_acceptance - expected) < 0.06
            assert edge["up_accepted"][j] == edge["up_eligible"][j]

    assert abs(fractions[0] - 0.5) < 0.08
    assert abs(fractions[1] - 0.5) < 0.08
    assert abs(fractions[0] - fractions[1]) < 0.08


def test_backtracking_causes_repeated_high_mode_switches_across_seeds():
    burn = 400
    total_switches = []
    for seed in (21, 22, 23):
        levels, _, _, trace = run_dns(seed, -1, num_walkers=6, num_sweeps=3200)
        positions, loglikelihood, indices, _ = trace
        positions = np.asarray(positions[burn:])
        loglikelihood = np.asarray(loglikelihood[burn:])
        indices = np.asarray(indices[burn:])
        highest = np.asarray(
            dns_diagnostics.highest_eligible_level(levels, loglikelihood)
        )

        switches = 0
        a_to_b = 0
        b_to_a = 0
        depths = []
        trips = 0
        for walker in range(positions.shape[1]):
            summary = dns_diagnostics.mode_switch_summary(
                np.where(positions[:, walker] >= 0, 1, -1),
                indices[:, walker],
                highest[:, walker],
                high_level=3,
            )
            switches += summary["num_switches"]
            a_to_b += summary["a_to_b"]
            b_to_a += summary["b_to_a"]
            depths.extend(summary["backtracking_minima"].tolist())
            round_trips = dns_diagnostics.round_trip_counts(
                indices[:, walker], low_level=0, high_level=NUM_LEVELS - 1
            )
            trips += round_trips["low_high_low"] + round_trips["high_low_high"]

        assert switches > 20
        assert a_to_b > 5
        assert b_to_a > 5
        assert trips > 5
        assert depths
        assert np.max(depths) <= 1
        total_switches.append(switches)

    assert min(total_switches) > 20


def test_matched_cost_high_constraint_remains_trapped():
    levels, _, _, dns_trace = run_dns(31, -1)
    dns_positions, _, _, dns_info = dns_trace
    dns_positions = np.asarray(dns_positions[500:])
    dns_eval_proxy = np.sum(
        np.asarray(dns_info.parameter_info.num_steps)
        + np.asarray(dns_info.parameter_info.num_shrink)
    )

    fixed_positions, fixed_info = run_fixed_high(32)
    fixed_positions = np.asarray(fixed_positions)
    fixed_eval_proxy = np.sum(
        np.asarray(fixed_info.num_steps) + np.asarray(fixed_info.num_shrink)
    )

    assert np.any(dns_positions > 0)
    assert not np.any(fixed_positions > 0)
    ratio = fixed_eval_proxy / dns_eval_proxy
    assert 0.8 < ratio < 1.2

    highest = np.asarray(
        dns_diagnostics.highest_eligible_level(
            levels, np.asarray(dns_trace[1][500:])
        )
    )
    switches = 0
    for walker in range(dns_positions.shape[1]):
        switches += dns_diagnostics.mode_switch_summary(
            np.where(dns_positions[:, walker] >= 0, 1, -1),
            np.asarray(dns_trace[2][500:, walker]),
            highest[:, walker],
            high_level=3,
        )["num_switches"]
    assert switches > 20
