"""Validation script comparing static vs dynamic nested sampling schedules.

This script is intentionally orchestration-only: it reuses existing NSS/GGNS
kernels and existing bounded/dynamic scheduler utilities without altering
sampler behavior.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from blackjax.ns import ggns, nss, utils


def uniform_logprior_2d(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x: jax.Array) -> jax.Array:
    m1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    m2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(m1, m2)


def build_algo(args: argparse.Namespace):
    common_kwargs = dict(
        logprior_fn=uniform_logprior_2d,
        loglikelihood_fn=gaussian_mixture_loglikelihood,
        num_inner_steps=args.ggns_num_inner_steps,
        num_delete=2,
    )
    if args.sampler == "nss":
        if args.replacement_strategy == "cluster_aware":
            common_kwargs["update_strategy"] = partial(
                nss.cluster_aware_update_with_mcmc_take_last,
                eager=args.cluster_aware_eager,
            )
        elif args.replacement_strategy != "global":
            raise ValueError(f"Unknown replacement strategy: {args.replacement_strategy}")
        return nss.as_top_level_api(**common_kwargs)
    if args.sampler == "ggns":
        return ggns.as_top_level_api(step_size=args.ggns_step_size, **common_kwargs)
    raise ValueError(f"Unknown sampler: {args.sampler}")


def init_positions(seed: int, num_live: int, dim: int = 2) -> jax.Array:
    return jax.random.uniform(
        jax.random.key(seed), shape=(num_live, dim), minval=-4.0, maxval=4.0
    )


def time_static_run(
    seed: int,
    algo,
    positions: jax.Array,
    static_num_steps: int,
) -> tuple[utils.NSMergedResult, utils.NSBatchResult, float]:
    state = algo.init(positions, rng_key=jax.random.key(seed + 10_000))
    t0 = time.perf_counter()
    _, batch = utils.run_bounded_batch(
        rng_key=jax.random.key(seed + 20_000),
        state=state,
        step_fn=algo.step,
        num_steps=static_num_steps,
        loglikelihood_lower=-jnp.inf,
    )
    merged = utils.merge_bounded_batches([batch])
    jax.block_until_ready(merged.logZ)
    dt = time.perf_counter() - t0
    return merged, batch, dt


def time_dynamic_run(
    seed: int,
    algo,
    positions: jax.Array,
    initial_num_steps: int,
    refinement_num_steps: int,
    max_batches: int,
    target: str,
) -> tuple[utils.NSDynamicResult, float]:
    state = algo.init(positions, rng_key=jax.random.key(seed + 30_000))
    t0 = time.perf_counter()
    _, result = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(seed + 40_000),
        state=state,
        step_fn=algo.step,
        initial_num_steps=initial_num_steps,
        refinement_num_steps=refinement_num_steps,
        max_batches=max_batches,
        objective=target,
    )
    jax.block_until_ready(result.merged.logZ)
    dt = time.perf_counter() - t0
    return result, dt


def print_static_summary(seed: int, merged: utils.NSMergedResult, batch, dt: float):
    steps = max(batch.metadata.num_steps, 1)
    print(f"seed={seed} mode=static")
    print(f"  logZ: {float(merged.logZ):.6f}")
    print(f"  ESS: {float(merged.ess):.2f}")
    print("  number of batches: 1")
    print(f"  number of dead points: {merged.metadata.num_dead}")
    print(f"  runtime (s): {dt:.6f}")
    print(f"  ms per NS step: {1e3 * dt / steps:.4f}")
    print(f"  posterior weight sum: {float(jnp.sum(merged.posterior_weights)):.8f}")


def print_dynamic_summary(seed: int, result: utils.NSDynamicResult, dt: float):
    merged = result.merged
    steps = max(sum(batch.metadata.num_steps for batch in result.batches), 1)
    print(f"seed={seed} mode=dynamic")
    print(f"  logZ: {float(merged.logZ):.6f}")
    print(f"  ESS: {float(merged.ess):.2f}")
    print(f"  number of batches: {merged.metadata.num_batches}")
    print(f"  number of dead points: {merged.metadata.num_dead}")
    print(f"  runtime (s): {dt:.6f}")
    print(f"  ms per NS step: {1e3 * dt / steps:.4f}")
    print(f"  posterior weight sum: {float(jnp.sum(merged.posterior_weights)):.8f}")

    print(f"  initial batch steps: {result.metadata.initial_num_steps}")
    print(f"  refinement batch steps: {result.metadata.refinement_num_steps}")
    intervals = [
        (batch.loglikelihood_lower, batch.loglikelihood_upper)
        for batch in result.batches[1:]
    ]
    print("  selected refinement interval [logL_min, logL_max):")
    if intervals:
        for i, (lower, upper) in enumerate(intervals):
            print(f"    batch {i + 1}: [{lower:.6f}, {upper:.6f})")
    else:
        print("    none")
    print(f"  number of empty batches: {result.metadata.num_empty_batches}")
    widths = [batch.metadata.interval_width for batch in result.batches]
    print(f"  dynamic batch widths: {[float(w) for w in widths]}")
    print("  refinement interval diagnostics:")
    if result.metadata.refinement_interval_diagnostics:
        for i, diag in enumerate(result.metadata.refinement_interval_diagnostics):
            print(
                f"    batch {i + 1}: lower={diag.selected_lower_threshold:.6f}, "
                f"upper={diag.selected_upper_threshold:.6f}, "
                f"upper_is_finite={diag.selected_upper_is_finite}, "
                f"width={diag.interval_width:.6f}, "
                f"upper_none_reason={diag.upper_none_reason_code}, "
                f"posterior_weight={diag.posterior_weight_at_selected_dead_point:.8f}, "
                f"dead_point_idx={diag.selected_dead_point_index}, "
                f"candidate_dead_points={diag.num_candidate_dead_points_considered}, "
                f"finite_interval_selected={diag.finite_interval_selected}, "
                f"posterior_mass_in_interval={diag.posterior_mass_in_interval:.8f}, "
                f"lower_idx={diag.selected_lower_index}, "
                f"upper_idx={diag.selected_upper_index}, "
                f"orig_upper={diag.original_attempted_upper_threshold:.6f}, "
                f"widened_upper={diag.widened_upper_threshold:.6f}, "
                f"finite_retry_used={diag.finite_retry_used}, "
                f"finite_retry_succeeded={diag.finite_retry_succeeded}, "
                f"no_wider_finite_upper_available={diag.no_wider_finite_upper_available}, "
                f"attempted_finite_dead_within={diag.attempted_finite_dead_within}, "
                f"attempted_finite_dead_above={diag.attempted_finite_dead_above}, "
                f"attempted_finite_dead_min_logL={diag.attempted_finite_dead_min_logL:.6f}, "
                f"attempted_finite_dead_max_logL={diag.attempted_finite_dead_max_logL:.6f}"
            )
    else:
        print("    none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=["nss", "ggns"], default="nss")
    parser.add_argument("--target", default="posterior")
    parser.add_argument("--num-live", type=int, default=32)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--static-num-steps", type=int, default=32)
    parser.add_argument("--initial-num-steps", type=int, default=16)
    parser.add_argument("--refinement-num-steps", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--ggns-step-size", type=float, default=0.05)
    parser.add_argument("--ggns-num-inner-steps", type=int, default=4)
    parser.add_argument(
        "--replacement-strategy",
        choices=["global", "cluster_aware"],
        default="global",
        help="NSS replacement strategy. The default global path is unchanged.",
    )
    parser.add_argument(
        "--cluster-aware-eager",
        action="store_true",
        help="Run cluster-aware replacement's Python validation path for concrete live points.",
    )
    parser.add_argument(
        "--nss-eager",
        "--disable-nss-jit",
        dest="nss_eager",
        action="store_true",
        help="Disable JAX JIT while running NSS validation loops.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "Compatibility flag for validation workflows that normally create a "
            "final corner plot; this script does not plot, so the flag is accepted "
            "and reported as disabled plotting."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo = build_algo(args)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"sampler: {args.sampler}")
    if args.no_plot:
        print("plotting: disabled (--no-plot)")
    print(
        "config: "
        f"num_live={args.num_live}, num_seeds={args.num_seeds}, "
        f"static_steps={args.static_num_steps}, initial_steps={args.initial_num_steps}, "
        f"refinement_steps={args.refinement_num_steps}, max_batches={args.max_batches}"
    )

    eager_context = (
        jax.disable_jit()
        if (args.sampler == "nss" and args.nss_eager)
        else nullcontext()
    )
    with eager_context:
        for seed in range(args.num_seeds):
            positions = init_positions(seed=seed, num_live=args.num_live)
            static_merged, static_batch, static_dt = time_static_run(
                seed=seed,
                algo=algo,
                positions=positions,
                static_num_steps=args.static_num_steps,
            )
            print_static_summary(seed, static_merged, static_batch, static_dt)

            dynamic_result, dynamic_dt = time_dynamic_run(
                seed=seed,
                algo=algo,
                positions=positions,
                initial_num_steps=args.initial_num_steps,
                refinement_num_steps=args.refinement_num_steps,
                max_batches=args.max_batches,
                target=args.target,
            )
            print_dynamic_summary(seed, dynamic_result, dynamic_dt)


if __name__ == "__main__":
    main()
