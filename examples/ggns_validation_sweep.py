"""Smoke validation sweep for NSS vs GGNS in static/dynamic NS modes."""
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
    num_steps: int
    failed: bool = False
    acceptance_rate: float = float("nan")
    boundary_crossing_rate: float = float("nan")
    fallback_rejection_rate: float = float("nan")
    mean_constraint_gap: float = float("nan")
    median_constraint_gap: float = float("nan")
    max_constraint_gap: float = float("nan")


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
    mode: str,
    seed: int,
    num_live: int,
    initial_num_steps: int,
    refinement_num_steps: int,
    max_batches: int,
    static_num_steps: int,
    ggns_step_size: float,
    ggns_num_integration_steps: int | None,
    ggns_num_inner_steps: int,
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
        ggns_kwargs = dict(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_mixture_loglikelihood,
            num_inner_steps=ggns_num_inner_steps,
            num_delete=2,
        )
        ggns_kwargs["step_size"] = ggns_step_size
        if ggns_num_integration_steps is not None:
            ggns_kwargs["num_integration_steps"] = ggns_num_integration_steps
        algo = ggns.as_top_level_api(**ggns_kwargs)
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'")

    state = algo.init(positions, rng_key=jax.random.key(seed + 10_000))

    ggns_accepted = []
    ggns_crossed = []
    ggns_gap = []

    def instrumented_step(rng_key, ns_state):
        new_state, info = algo.step(rng_key, ns_state)
        if sampler_name == "ggns":
            inner_info = getattr(info.update_info, 'mcmc_infos', info.update_info)
            # For GGNS update_with_mcmc_take_last, mcmc_infos has shape
            # [num_delete, num_inner_steps] and fields from ConstrainedGradientGuidedInfo.
            ggns_accepted.append(jnp.ravel(inner_info.accepted))
            ggns_crossed.append(jnp.ravel(inner_info.crossed_boundary))
            ggns_gap.append(jnp.ravel(inner_info.final_loglikelihood) - info.particles.loglikelihood.max())
        return new_state, info

    t0 = time.perf_counter()
    if mode == "dynamic":
        _, result = utils.run_dynamic_posterior_scheduler(
            rng_key=jax.random.key(seed + 20_000),
            state=state,
            step_fn=instrumented_step,
            initial_num_steps=initial_num_steps,
            refinement_num_steps=refinement_num_steps,
            max_batches=max_batches,
        )
        merged = result.merged
        total_steps = sum(batch.metadata.num_steps for batch in result.batches)
    elif mode == "static":
        _, batch = utils.run_bounded_batch(
            rng_key=jax.random.key(seed + 20_000),
            state=state,
            step_fn=instrumented_step,
            num_steps=static_num_steps,
            loglikelihood_lower=-jnp.inf,
            loglikelihood_upper=None,
        )
        merged = utils.merge_bounded_batches([batch])
        total_steps = batch.metadata.num_steps
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    jax.block_until_ready(merged.logZ)
    run_seconds = time.perf_counter() - t0
    ms_per_step = 1e3 * run_seconds / max(total_steps, 1)

    metrics = RunMetrics(
        logz=float(merged.logZ),
        ess=float(merged.ess),
        run_seconds=run_seconds,
        ms_per_step=ms_per_step,
        num_steps=int(total_steps),
        failed=False,
    )

    if sampler_name == "ggns" and ggns_accepted:
        accepted = jnp.concatenate(ggns_accepted)
        crossed = jnp.concatenate(ggns_crossed)
        gaps = jnp.concatenate(ggns_gap)
        metrics.acceptance_rate = float(jnp.mean(accepted.astype(jnp.float32)))
        metrics.boundary_crossing_rate = float(jnp.mean(crossed.astype(jnp.float32)))
        metrics.fallback_rejection_rate = float(
            jnp.mean((~accepted.astype(bool)).astype(jnp.float32))
        )
        metrics.mean_constraint_gap = float(jnp.mean(gaps))
        metrics.median_constraint_gap = float(jnp.median(gaps))
        metrics.max_constraint_gap = float(jnp.max(gaps))
    elif sampler_name == "ggns":
        # TODO(ns-ggns): expose per-batch GGNS diagnostics in NSBatchMetadata/NSDynamicMetadata
        # so diagnostics do not require step-function instrumentation from this script.
        pass

    return metrics


def summarize(label: str, sampler: str, results: list[RunMetrics]) -> None:
    failures = sum(result.failed for result in results)
    ok = [result for result in results if not result.failed]
    if not ok:
        print(f"\n{label} {sampler.upper()}: all runs failed ({failures}/{len(results)}).")
        return

    logzs = jnp.array([r.logz for r in ok])
    esses = jnp.array([r.ess for r in ok])
    run_seconds = jnp.array([r.run_seconds for r in ok])
    ms_per_step = jnp.array([r.ms_per_step for r in ok])

    print(f"\n{label} {sampler.upper()} summary ({len(ok)} successful / {len(results)} total):")
    print(f"  mean logZ: {float(jnp.mean(logzs)):.6f}")
    print(f"  std logZ: {float(jnp.std(logzs)):.6f}")
    print(f"  mean ESS: {float(jnp.mean(esses)):.3f}")
    print(f"  mean runtime (s): {float(jnp.mean(run_seconds)):.4f}")
    print(f"  mean ms per NS step: {float(jnp.mean(ms_per_step)):.4f}")
    print(f"  failures: {failures}")

    if sampler == "ggns":
        ar = jnp.array([r.acceptance_rate for r in ok])
        bcr = jnp.array([r.boundary_crossing_rate for r in ok])
        fr = jnp.array([r.fallback_rejection_rate for r in ok])
        mean_gap = jnp.array([r.mean_constraint_gap for r in ok])
        med_gap = jnp.array([r.median_constraint_gap for r in ok])
        max_gap = jnp.array([r.max_constraint_gap for r in ok])
        print(f"  mean acceptance rate: {float(jnp.nanmean(ar)):.4f}")
        print(f"  mean boundary-crossing rate: {float(jnp.nanmean(bcr)):.4f}")
        print(f"  mean fallback/rejection rate: {float(jnp.nanmean(fr)):.4f}")
        print(f"  mean(final_logL - constraint): {float(jnp.nanmean(mean_gap)):.6f}")
        print(f"  median(final_logL - constraint): {float(jnp.nanmean(med_gap)):.6f}")
        print(f"  max(final_logL - constraint): {float(jnp.nanmean(max_gap)):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke validation sweep for static/dynamic NSS and GGNS.")
    parser.add_argument("--num-live", type=int, default=40)
    parser.add_argument("--initial-num-steps", type=int, default=10)
    parser.add_argument("--refinement-num-steps", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--static-num-steps", type=int, default=16)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--samplers", type=str, default="nss,ggns")
    parser.add_argument("--modes", type=str, default="static,dynamic")
    parser.add_argument("--ggns-step-size", type=float, default=0.05)
    parser.add_argument("--ggns-num-integration-steps", type=int, default=None)
    parser.add_argument("--ggns-num-inner-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samplers = [s.strip().lower() for s in args.samplers.split(",") if s.strip()]
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    print("Smoke validation only (not a statistical benchmark).")
    print(f"JAX backend: {jax.default_backend()}")
    print(
        f"Config: num_live={args.num_live}, initial_num_steps={args.initial_num_steps}, "
        f"refinement_num_steps={args.refinement_num_steps}, max_batches={args.max_batches}, "
        f"static_num_steps={args.static_num_steps}, num_seeds={args.num_seeds}, "
        f"samplers={samplers}, modes={modes}"
    )
    print(
        "GGNS config: "
        f"step_size={args.ggns_step_size}, "
        f"num_integration_steps={args.ggns_num_integration_steps}, "
        f"num_inner_steps={args.ggns_num_inner_steps}"
    )

    for mode in modes:
        for sampler in samplers:
            results: list[RunMetrics] = []
            for seed in range(args.num_seeds):
                try:
                    metrics = run_one(
                        sampler_name=sampler,
                        mode=mode,
                        seed=seed,
                        num_live=args.num_live,
                        initial_num_steps=args.initial_num_steps,
                        refinement_num_steps=args.refinement_num_steps,
                        max_batches=args.max_batches,
                        static_num_steps=args.static_num_steps,
                        ggns_step_size=args.ggns_step_size,
                        ggns_num_integration_steps=args.ggns_num_integration_steps,
                        ggns_num_inner_steps=args.ggns_num_inner_steps,
                    )
                    results.append(metrics)
                    print(
                        f"  {mode} {sampler} seed={seed}: logZ={metrics.logz:.6f}, "
                        f"ESS={metrics.ess:.2f}, runtime={metrics.run_seconds:.3f}s, "
                        f"ms/step={metrics.ms_per_step:.3f}, failures=0"
                    )
                except Exception as exc:
                    print(f"  {mode} {sampler} seed={seed}: FAILED ({type(exc).__name__}: {exc})")
                    results.append(
                        RunMetrics(
                            logz=float("nan"),
                            ess=float("nan"),
                            run_seconds=float("nan"),
                            ms_per_step=float("nan"),
                            num_steps=0,
                            failed=True,
                        )
                    )
            summarize(mode.upper(), sampler, results)


if __name__ == "__main__":
    main()
