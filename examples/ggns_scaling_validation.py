"""Scaling/statistical validation for NSS vs reflected GGNS."""
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
    mean_delta_logl: float = float("nan")
    fraction_proposals_with_reflection: float = float("nan")
    reflection_failure_rate: float = float("nan")


@dataclass
class TargetSpec:
    name: str
    dimension: int
    logprior_fn: callable
    loglikelihood_fn: callable
    init_fn: callable


def uniform_box_logprior(x: jax.Array, bound: float = 6.0) -> jax.Array:
    return jnp.where(jnp.all(jnp.abs(x) <= bound), 0.0, -jnp.inf)


def make_gaussian_mixture_2d() -> TargetSpec:
    def logprior_fn(x: jax.Array) -> jax.Array:
        return uniform_box_logprior(x, bound=6.0)

    def loglikelihood_fn(x: jax.Array) -> jax.Array:
        means = jnp.array([[2.0, 0.0], [-2.0, 0.0]])
        ll0 = stats.norm.logpdf(x - means[0]).sum()
        ll1 = stats.norm.logpdf(x - means[1]).sum()
        return jnp.logaddexp(ll0, ll1)

    def init_fn(key: jax.Array, num_live: int) -> jax.Array:
        return jax.random.uniform(key, (num_live, 2), minval=-4.0, maxval=4.0)

    return TargetSpec("gaussian_mixture", 2, logprior_fn, loglikelihood_fn, init_fn)


def make_correlated_gaussian(dimension: int) -> TargetSpec:
    idx = jnp.arange(dimension)
    corr = 0.7 ** jnp.abs(idx[:, None] - idx[None, :])
    cov = corr + 1e-3 * jnp.eye(dimension)
    precision = jnp.linalg.inv(cov)
    logdet = jnp.linalg.slogdet(cov)[1]

    def logprior_fn(x: jax.Array) -> jax.Array:
        return uniform_box_logprior(x, bound=6.0)

    def loglikelihood_fn(x: jax.Array) -> jax.Array:
        quad = x @ precision @ x
        return -0.5 * (dimension * jnp.log(2.0 * jnp.pi) + logdet + quad)

    def init_fn(key: jax.Array, num_live: int) -> jax.Array:
        return jax.random.normal(key, (num_live, dimension))

    return TargetSpec("correlated_gaussian", dimension, logprior_fn, loglikelihood_fn, init_fn)


def make_banana(dimension: int) -> TargetSpec:
    if dimension < 2:
        raise ValueError("banana target requires dimension >= 2")

    def logprior_fn(x: jax.Array) -> jax.Array:
        return uniform_box_logprior(x, bound=8.0)

    def loglikelihood_fn(x: jax.Array) -> jax.Array:
        x0 = x[0]
        x1 = x[1]
        a = 1.0
        b = 0.1
        rb = -((a - x0) ** 2 + 100.0 * (x1 - x0**2) ** 2)
        tail = -0.5 * jnp.sum((x[2:] / b) ** 2)
        return rb + tail

    def init_fn(key: jax.Array, num_live: int) -> jax.Array:
        return jax.random.uniform(key, (num_live, dimension), minval=-2.0, maxval=2.0)

    return TargetSpec("banana", dimension, logprior_fn, loglikelihood_fn, init_fn)


def build_target(target: str, dimension: int) -> TargetSpec:
    if target == "gaussian_mixture":
        if dimension != 2:
            raise ValueError("gaussian_mixture is only implemented in 2D")
        return make_gaussian_mixture_2d()
    if target == "correlated_gaussian":
        return make_correlated_gaussian(dimension)
    if target in {"banana", "rosenbrock"}:
        return make_banana(dimension)
    raise ValueError(f"Unknown target '{target}'")


def run_one(target_spec: TargetSpec, sampler_name: str, mode: str, seed: int, args: argparse.Namespace) -> RunMetrics:
    key = jax.random.key(seed)
    positions = target_spec.init_fn(key, args.num_live)

    if sampler_name == "nss":
        algo = nss.as_top_level_api(
            logprior_fn=target_spec.logprior_fn,
            loglikelihood_fn=target_spec.loglikelihood_fn,
            num_inner_steps=4,
            num_delete=2,
        )
    elif sampler_name == "ggns":
        algo = ggns.as_top_level_api(
            logprior_fn=target_spec.logprior_fn,
            loglikelihood_fn=target_spec.loglikelihood_fn,
            num_inner_steps=args.ggns_num_inner_steps,
            step_size=args.ggns_step_size,
            num_delete=2,
        )
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'")

    state = algo.init(positions, rng_key=jax.random.key(seed + 1000))

    all_delta_logl = []
    all_reflections = []
    all_reflect_failures = []

    def instrumented_step(rng_key, ns_state):
        new_state, info = algo.step(rng_key, ns_state)
        if sampler_name == "ggns":
            update_info = getattr(info, "update_info", None)
            inner = getattr(update_info, "mcmc_infos", update_info)
            start = getattr(inner, "start_loglikelihood", None)
            end = getattr(inner, "final_loglikelihood", None)
            num_ref = getattr(inner, "num_reflections", None)
            ref_fail = getattr(inner, "reflection_failures", None)
            if ref_fail is None:
                ref_fail = getattr(inner, "reflection_failed", None)
            if start is not None and end is not None:
                all_delta_logl.append(jnp.ravel(end) - jnp.ravel(start))
            if num_ref is not None:
                all_reflections.append(jnp.ravel(num_ref))
            if ref_fail is not None:
                all_reflect_failures.append(jnp.ravel(ref_fail))
        return new_state, info

    t0 = time.perf_counter()
    if mode == "dynamic":
        _, result = utils.run_dynamic_posterior_scheduler(
            rng_key=jax.random.key(seed + 2000),
            state=state,
            step_fn=instrumented_step,
            initial_num_steps=args.initial_num_steps,
            refinement_num_steps=args.refinement_num_steps,
            max_batches=args.max_batches,
        )
        merged = result.merged
        total_steps = sum(batch.metadata.num_steps for batch in result.batches)
    elif mode == "static":
        _, batch = utils.run_bounded_batch(
            rng_key=jax.random.key(seed + 2000),
            state=state,
            step_fn=instrumented_step,
            num_steps=args.static_num_steps,
            loglikelihood_lower=-jnp.inf,
            loglikelihood_upper=None,
        )
        merged = utils.merge_bounded_batches([batch])
        total_steps = batch.metadata.num_steps
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    jax.block_until_ready(merged.logZ)
    elapsed = time.perf_counter() - t0

    metrics = RunMetrics(
        logz=float(merged.logZ),
        ess=float(merged.ess),
        run_seconds=float(elapsed),
        ms_per_step=float(1e3 * elapsed / max(total_steps, 1)),
        num_steps=int(total_steps),
    )

    if sampler_name == "ggns":
        if all_delta_logl:
            metrics.mean_delta_logl = float(jnp.mean(jnp.concatenate(all_delta_logl)))
        if all_reflections:
            refs = jnp.concatenate(all_reflections)
            metrics.fraction_proposals_with_reflection = float(jnp.mean((refs > 0).astype(jnp.float32)))
        if all_reflect_failures:
            fails = jnp.concatenate(all_reflect_failures)
            metrics.reflection_failure_rate = float(jnp.mean((fails > 0).astype(jnp.float32)))

    return metrics


def summarize(sampler: str, results: list[RunMetrics]) -> None:
    ok = [r for r in results if not r.failed]
    logz = jnp.array([r.logz for r in ok])
    ess = jnp.array([r.ess for r in ok])
    runtime = jnp.array([r.run_seconds for r in ok])
    mstep = jnp.array([r.ms_per_step for r in ok])

    print(f"\n{sampler.upper()} summary ({len(ok)}/{len(results)} successful)")
    print(f"  mean/std logZ: {float(jnp.mean(logz)):.6f} / {float(jnp.std(logz)):.6f}")
    print(f"  mean ESS: {float(jnp.mean(ess)):.3f}")
    print(f"  mean runtime (s): {float(jnp.mean(runtime)):.4f}")
    print(f"  mean ms per NS step: {float(jnp.mean(mstep)):.4f}")

    if sampler == "ggns":
        delta = jnp.array([r.mean_delta_logl for r in ok])
        frac_ref = jnp.array([r.fraction_proposals_with_reflection for r in ok])
        fail = jnp.array([r.reflection_failure_rate for r in ok])
        print(f"  mean delta_logL: {float(jnp.nanmean(delta)):.6f}")
        print(f"  fraction proposals with reflection: {float(jnp.nanmean(frac_ref)):.6f}")
        print(f"  reflection failure rate: {float(jnp.nanmean(fail)):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaling/statistical validation for NSS vs reflected GGNS.")
    parser.add_argument(
        "--target",
        default="gaussian_mixture",
        choices=["gaussian_mixture", "correlated_gaussian", "banana", "rosenbrock", "all"],
    )
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--num-live", type=int, default=40)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--mode", default="static", choices=["static", "dynamic"])
    parser.add_argument("--samplers", default="nss,ggns")
    parser.add_argument("--initial-num-steps", type=int, default=10)
    parser.add_argument("--refinement-num-steps", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--static-num-steps", type=int, default=16)
    parser.add_argument("--ggns-step-size", type=float, default=0.001)
    parser.add_argument("--ggns-num-inner-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]

    print(f"mode={args.mode}, num_live={args.num_live}, num_seeds={args.num_seeds}, samplers={samplers}")
    print(f"GGNS: step_size={args.ggns_step_size}, num_inner_steps={args.ggns_num_inner_steps}")

    if args.target == "all":
        target_grid = [
            ("gaussian_mixture", 2),
            ("correlated_gaussian", 2),
            ("correlated_gaussian", 5),
            ("correlated_gaussian", 10),
            ("banana", 2),
        ]
    else:
        target_grid = [(args.target, args.dimension)]

    for target_name, dimension in target_grid:
        target_spec = build_target(target_name, dimension)
        print(f"\n=== Target={target_spec.name}, dimension={target_spec.dimension} ===")

        for sampler in samplers:
            results = []
            for seed in range(args.num_seeds):
                result = run_one(target_spec, sampler, args.mode, seed, args)
                results.append(result)
                print(
                    f"  {sampler} seed={seed}: logZ={result.logz:.5f}, ESS={result.ess:.2f}, "
                    f"runtime={result.run_seconds:.3f}s, ms/step={result.ms_per_step:.3f}"
                )
            summarize(sampler, results)


if __name__ == "__main__":
    main()
