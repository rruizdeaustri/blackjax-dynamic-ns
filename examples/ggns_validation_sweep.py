"""Smoke validation sweep for dynamic NSS vs dynamic GGNS on a 2D Gaussian mixture.

This script is intended as a quick regression/sanity check for constrained proposals
and scheduler integration. It is **not** a statistical benchmark.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from blackjax.ns import ggns, nss, utils


@dataclass
class RunMetrics:
    logz: float
    ess: float
    run_seconds: float
    ms_per_step: float
    failed: bool = False


def uniform_logprior_2d(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x: jax.Array) -> jax.Array:
    m1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    m2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(m1, m2)


def make_positions(seed: int, num_live: int) -> jax.Array:
    return jax.random.uniform(
        jax.random.key(seed), shape=(num_live, 2), minval=-4.0, maxval=4.0
    )


def run_one(
    sampler_name: str,
    seed: int,
    num_live: int,
    initial_num_steps: int,
    refinement_num_steps: int,
    max_batches: int,
) -> RunMetrics:
    positions = make_positions(seed, num_live)

    if sampler_name == "nss":
        algo = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_mixture_loglikelihood,
            num_inner_steps=4,
            num_delete=2,
        )
    elif sampler_name == "ggns":
        algo = ggns.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_mixture_loglikelihood,
            num_inner_steps=4,
            num_delete=2,
            step_size=0.05,
        )
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'")

    state = algo.init(positions, rng_key=jax.random.key(seed + 10_000))

    t0 = time.perf_counter()
    _, dynamic_result = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(seed + 20_000),
        state=state,
        step_fn=algo.step,
        initial_num_steps=initial_num_steps,
        refinement_num_steps=refinement_num_steps,
        max_batches=max_batches,
    )
    # Timing requirement: materialize merged.logZ before stopping timer.
    jax.block_until_ready(dynamic_result.merged.logZ)
    run_seconds = time.perf_counter() - t0

    merged = dynamic_result.merged
    total_steps = sum(batch.metadata.num_steps for batch in dynamic_result.batches)
    ms_per_step = 1e3 * run_seconds / max(total_steps, 1)

    return RunMetrics(
        logz=float(merged.logZ),
        ess=float(merged.ess),
        run_seconds=run_seconds,
        ms_per_step=ms_per_step,
        failed=False,
    )


def summarize_sampler(name: str, results: list[RunMetrics]) -> None:
    failures = sum(result.failed for result in results)
    ok = [result for result in results if not result.failed]

    if not ok:
        print(f"\n{name.upper()}: all runs failed ({failures}/{len(results)}).")
        return

    logzs = jnp.array([r.logz for r in ok])
    esses = jnp.array([r.ess for r in ok])
    run_seconds = jnp.array([r.run_seconds for r in ok])
    ms_per_step = jnp.array([r.ms_per_step for r in ok])

    print(f"\n{name.upper()} summary ({len(ok)} successful / {len(results)} total):")
    print(f"  mean logZ: {float(jnp.mean(logzs)):.6f}")
    print(f"  std logZ: {float(jnp.std(logzs)):.6f}")
    print(f"  mean ESS: {float(jnp.mean(esses)):.3f}")
    print(f"  mean run-only wall time (s): {float(jnp.mean(run_seconds)):.4f}")
    print(f"  mean ms per NS step: {float(jnp.mean(ms_per_step)):.4f}")
    print(f"  failures: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke validation sweep for Dynamic NSS and Dynamic GGNS on a 2D "
            "Gaussian mixture target."
        )
    )
    parser.add_argument("--num-live", type=int, default=40)
    parser.add_argument("--initial-num-steps", type=int, default=12)
    parser.add_argument("--refinement-num-steps", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument(
        "--samplers",
        type=str,
        default="nss,ggns",
        help="Comma-separated list from {nss,ggns}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samplers = [s.strip().lower() for s in args.samplers.split(",") if s.strip()]

    print("Smoke validation only (not a statistical benchmark).")
    print(f"JAX backend: {jax.default_backend()}")
    print(
        f"Config: num_live={args.num_live}, initial_num_steps={args.initial_num_steps}, "
        f"refinement_num_steps={args.refinement_num_steps}, max_batches={args.max_batches}, "
        f"num_seeds={args.num_seeds}, samplers={samplers}"
    )

    for sampler in samplers:
        results: list[RunMetrics] = []
        for seed in range(args.num_seeds):
            try:
                metrics = run_one(
                    sampler_name=sampler,
                    seed=seed,
                    num_live=args.num_live,
                    initial_num_steps=args.initial_num_steps,
                    refinement_num_steps=args.refinement_num_steps,
                    max_batches=args.max_batches,
                )
                results.append(metrics)
            except Exception as exc:  # validation script: continue and report failures
                print(f"  {sampler} seed={seed}: FAILED ({type(exc).__name__}: {exc})")
                results.append(
                    RunMetrics(
                        logz=float("nan"),
                        ess=float("nan"),
                        run_seconds=float("nan"),
                        ms_per_step=float("nan"),
                        failed=True,
                    )
                )

        summarize_sampler(sampler, results)


if __name__ == "__main__":
    main()
