"""Frozen-ladder multimodal validation for experimental DNS Stage 2.

This is a mechanism/validation example, not an evidence calculation.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns, dns_diagnostics, dns_kernels


BOUND = 6.0
MODE = 1.0
SIGMA = 0.3
NUM_LEVELS = 5
LOCAL_DIRECTION_SCALE = 0.5


def build_problem():
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


def run_dns(seed, num_walkers=6, num_sweeps=3200):
    levels, masses, radii, logprior, loglikelihood, parameter_step = build_problem()
    state = dns.init(
        jnp.array(-MODE), logprior, loglikelihood, levels, level_index=NUM_LEVELS - 1
    )
    states = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_walkers,) + x.shape), state
    )
    step = dns.build_kernel(levels, parameter_step)

    def sweep(current, key):
        new, info = jax.vmap(step)(jax.random.split(key, num_walkers), current)
        return new, (new.particle.position, new.particle.loglikelihood, new.level_index, info)

    _, trace = jax.jit(lambda s, ks: jax.lax.scan(sweep, s, ks))(
        states, jax.random.split(jax.random.key(seed), num_sweeps)
    )
    return levels, masses, radii, trace


def summarize(seed, burn=400):
    levels, masses, radii, trace = run_dns(seed)
    positions, loglikelihood, indices, info = trace
    positions = np.asarray(positions[burn:])
    loglikelihood = np.asarray(loglikelihood[burn:])
    indices = np.asarray(indices[burn:])
    highest = np.asarray(dns_diagnostics.highest_eligible_level(levels, loglikelihood))
    result = {
        "seed": seed,
        "mode_fraction_B": float(np.mean(positions > 0)),
        "level_occupancy": dns_diagnostics.level_occupancy(indices, NUM_LEVELS).tolist(),
        "a_to_b": 0,
        "b_to_a": 0,
        "switches": 0,
        "round_trips": 0,
        "first_passage": None,
        "backtracking_minima": [],
        "parameter_failure_count": int(np.size(info.parameter_is_valid) - np.count_nonzero(info.parameter_is_valid)),
    }
    passages = []
    for walker in range(positions.shape[1]):
        switch = dns_diagnostics.mode_switch_summary(
            np.where(positions[:, walker] >= 0, 1, -1),
            indices[:, walker],
            highest[:, walker],
            high_level=3,
        )
        result["a_to_b"] += switch["a_to_b"]
        result["b_to_a"] += switch["b_to_a"]
        result["switches"] += switch["num_switches"]
        result["backtracking_minima"].extend(switch["backtracking_minima"].tolist())
        if switch["first_passage_index"] is not None:
            passages.append(switch["first_passage_index"])
        trips = dns_diagnostics.round_trip_counts(
            indices[:, walker], low_level=0, high_level=NUM_LEVELS - 1
        )
        result["round_trips"] += trips["low_high_low"] + trips["high_low_high"]
    result["first_passage"] = min(passages) if passages else None
    result["max_switch_backtracking_level"] = (
        max(result["backtracking_minima"]) if result["backtracking_minima"] else None
    )
    result["slice_eval_proxy"] = int(
        np.sum(np.asarray(info.parameter_info.num_steps) + np.asarray(info.parameter_info.num_shrink))
    )
    result["masses"] = masses.tolist()
    result["radii"] = radii.tolist()
    result["thresholds"] = np.asarray(levels.loglikelihood).tolist()
    return result


def fixed_high_control(seed=32, num_walkers=8, num_sweeps=6500, high_level=3):
    levels, _, _, logprior, loglikelihood, parameter_step = build_problem()
    particle = dns.DNSParticleState(jnp.array(-MODE), logprior(-MODE), loglikelihood(-MODE))
    particles = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_walkers,) + x.shape), particle
    )

    def sweep(current, key):
        new, info = jax.vmap(parameter_step, in_axes=(0, 0, None))(
            jax.random.split(key, num_walkers), current, levels.loglikelihood[high_level]
        )
        return new, (new.position, info)

    _, (positions, info) = jax.jit(lambda s, ks: jax.lax.scan(sweep, s, ks))(
        particles, jax.random.split(jax.random.key(seed), num_sweeps)
    )
    positions = np.asarray(positions)
    return {
        "seed": seed,
        "high_level": high_level,
        "ever_reached_mode_B": bool(np.any(positions > 0)),
        "slice_eval_proxy": int(
            np.sum(np.asarray(info.num_steps) + np.asarray(info.num_shrink))
        ),
    }


if __name__ == "__main__":
    report = {
        "dns": [summarize(seed) for seed in (21, 22, 23)],
        "fixed_high_control": fixed_high_control(),
    }
    print(json.dumps(report, indent=2))
