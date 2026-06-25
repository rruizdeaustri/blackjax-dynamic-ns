"""Smoke validation sweep for NSS vs GGNS in static/dynamic NS modes."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

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
    mean_start_constraint_gap: float = float("nan")
    median_start_constraint_gap: float = float("nan")
    mean_accepted_constraint_gap: float = float("nan")
    median_accepted_constraint_gap: float = float("nan")
    mean_delta_logl: float = float("nan")
    median_delta_logl: float = float("nan")
    fallback_rejection_gap: float = float("nan")
    mean_reflections_per_proposal: float = float("nan")
    max_reflections_per_proposal: float = float("nan")
    fraction_proposals_with_reflection: float = float("nan")
    reflection_failure_rate: float = float("nan")


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
    replacement_strategy: str,
    cluster_aware_eager: bool,
    cluster_aware_standardize: bool,
    cluster_aware_scale_floor: float,
    cluster_aware_radius: float | None,
    cluster_aware_min_cluster_size: int,
    cluster_aware_max_condition_number: float,
    cluster_aware_covariance_regularization: float,
    cluster_aware_auto_fallback: bool,
    cluster_aware_warmup_attempts: int,
    cluster_aware_min_success_rate: float,
    cluster_aware_max_runtime_ratio: float,
    replacement_diagnostics: bool,
    nss_eager: bool,
) -> RunMetrics:
    positions = make_positions(seed, num_live)

    if sampler_name == "nss":
        nss_kwargs = dict(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_mixture_loglikelihood,
            num_inner_steps=4,
            num_delete=2,
        )
        if replacement_strategy == "cluster_aware":
            nss_kwargs["update_strategy"] = partial(
                nss.cluster_aware_update_with_mcmc_take_last,
                eager=cluster_aware_eager,
                standardize=cluster_aware_standardize,
                scale_floor=cluster_aware_scale_floor,
                radius=cluster_aware_radius,
                min_cluster_size=cluster_aware_min_cluster_size,
                max_condition_number=cluster_aware_max_condition_number,
                covariance_regularization=cluster_aware_covariance_regularization,
                auto_fallback=cluster_aware_auto_fallback,
                warmup_attempts=cluster_aware_warmup_attempts,
                min_success_rate=cluster_aware_min_success_rate,
                max_runtime_ratio=cluster_aware_max_runtime_ratio,
            )
        elif replacement_strategy == "global" and replacement_diagnostics:
            nss_kwargs["update_strategy"] = partial(
                nss.diagnostic_update_with_mcmc_take_last
            )
        elif replacement_strategy != "global":
            raise ValueError(f"Unknown replacement strategy '{replacement_strategy}'")
        algo = nss.as_top_level_api(**nss_kwargs)
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
    ggns_start_gap = []
    ggns_accepted_gap = []
    ggns_delta_logl = []
    ggns_fallback_gap = []
    ggns_num_reflections = []
    ggns_reflection_failures = []

    def instrumented_step(rng_key, ns_state):
        new_state, info = algo.step(rng_key, ns_state)
        if sampler_name == "ggns":
            update_info = getattr(info, "update_info", None)
            inner_info = getattr(update_info, "mcmc_infos", update_info)
            # For GGNS update_with_mcmc_take_last, mcmc_infos has shape
            # [num_delete, num_inner_steps] and fields from ConstrainedGradientGuidedInfo.
            accepted = getattr(inner_info, "accepted", None)
            crossed_boundary = getattr(inner_info, "crossed_boundary", None)
            start_logl = getattr(inner_info, "start_loglikelihood", None)
            final_logl = getattr(inner_info, "final_loglikelihood", None)
            constraint = getattr(info.particles, "loglikelihood", None)
            num_reflections = getattr(inner_info, "num_reflections", None)
            reflection_failures = getattr(inner_info, "reflection_failures", None)
            if reflection_failures is None:
                reflection_failures = getattr(inner_info, "reflection_failed", None)
            if accepted is not None:
                accepted_flat = jnp.ravel(accepted).astype(bool)
                ggns_accepted.append(accepted_flat)
            else:
                accepted_flat = None
            if crossed_boundary is not None:
                ggns_crossed.append(jnp.ravel(crossed_boundary))
            if num_reflections is not None:
                ggns_num_reflections.append(jnp.ravel(num_reflections))
            if reflection_failures is not None:
                ggns_reflection_failures.append(jnp.ravel(reflection_failures))
            if final_logl is not None and constraint is not None:
                constraint_threshold = jnp.max(constraint)
                final_gap = jnp.ravel(final_logl) - constraint_threshold
                ggns_gap.append(final_gap)
                if accepted_flat is not None:
                    ggns_accepted_gap.append(final_gap[accepted_flat])
                    ggns_fallback_gap.append(final_gap[~accepted_flat])
            if start_logl is not None and constraint is not None:
                constraint_threshold = jnp.max(constraint)
                ggns_start_gap.append(jnp.ravel(start_logl) - constraint_threshold)
            if start_logl is not None and final_logl is not None:
                ggns_delta_logl.append(jnp.ravel(final_logl) - jnp.ravel(start_logl))
        return new_state, info

    t0 = time.perf_counter()
    eager_context = (
        jax.disable_jit() if (sampler_name == "nss" and nss_eager) else nullcontext()
    )
    with eager_context:
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
        accepted = jnp.concatenate(ggns_accepted) if ggns_accepted else None
        crossed = jnp.concatenate(ggns_crossed) if ggns_crossed else None
        gaps = jnp.concatenate(ggns_gap) if ggns_gap else None
        if accepted is not None:
            metrics.acceptance_rate = float(jnp.mean(accepted.astype(jnp.float32)))
            metrics.fallback_rejection_rate = float(
                jnp.mean((~accepted.astype(bool)).astype(jnp.float32))
            )
        if crossed is not None:
            metrics.boundary_crossing_rate = float(
                jnp.mean(crossed.astype(jnp.float32))
            )
        if gaps is not None:
            metrics.mean_constraint_gap = float(jnp.mean(gaps))
            metrics.median_constraint_gap = float(jnp.median(gaps))
            metrics.max_constraint_gap = float(jnp.max(gaps))
        if ggns_start_gap:
            start_gaps = jnp.concatenate(ggns_start_gap)
            metrics.mean_start_constraint_gap = float(jnp.mean(start_gaps))
            metrics.median_start_constraint_gap = float(jnp.median(start_gaps))
        if ggns_accepted_gap:
            accepted_gaps = jnp.concatenate(ggns_accepted_gap)
            if accepted_gaps.size > 0:
                metrics.mean_accepted_constraint_gap = float(jnp.mean(accepted_gaps))
                metrics.median_accepted_constraint_gap = float(
                    jnp.median(accepted_gaps)
                )
        if ggns_delta_logl:
            delta_logl = jnp.concatenate(ggns_delta_logl)
            metrics.mean_delta_logl = float(jnp.mean(delta_logl))
            metrics.median_delta_logl = float(jnp.median(delta_logl))
        if ggns_fallback_gap:
            fallback_gaps = jnp.concatenate(ggns_fallback_gap)
            if fallback_gaps.size > 0:
                metrics.fallback_rejection_gap = float(jnp.mean(fallback_gaps))
        if ggns_num_reflections:
            reflections = jnp.concatenate(ggns_num_reflections)
            if reflections.size > 0:
                reflections_f = reflections.astype(jnp.float32)
                metrics.mean_reflections_per_proposal = float(jnp.mean(reflections_f))
                metrics.max_reflections_per_proposal = float(jnp.max(reflections_f))
                metrics.fraction_proposals_with_reflection = float(
                    jnp.mean((reflections > 0).astype(jnp.float32))
                )

        if ggns_reflection_failures:
            reflection_failures = jnp.concatenate(ggns_reflection_failures)
            if reflection_failures.size > 0:
                metrics.reflection_failure_rate = float(
                    jnp.mean((reflection_failures > 0).astype(jnp.float32))
                )
    elif sampler_name == "ggns":
        # TODO(ns-ggns): expose per-batch GGNS diagnostics in NSBatchMetadata/NSDynamicMetadata
        # so diagnostics do not require step-function instrumentation from this script.
        pass

    return metrics


def summarize(label: str, sampler: str, results: list[RunMetrics]) -> None:
    failures = sum(result.failed for result in results)
    ok = [result for result in results if not result.failed]
    if not ok:
        print(
            f"\n{label} {sampler.upper()}: all runs failed ({failures}/{len(results)})."
        )
        return

    logzs = jnp.array([r.logz for r in ok])
    esses = jnp.array([r.ess for r in ok])
    run_seconds = jnp.array([r.run_seconds for r in ok])
    ms_per_step = jnp.array([r.ms_per_step for r in ok])

    print(
        f"\n{label} {sampler.upper()} summary ({len(ok)} successful / {len(results)} total):"
    )
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
        start_mean_gap = jnp.array([r.mean_start_constraint_gap for r in ok])
        start_median_gap = jnp.array([r.median_start_constraint_gap for r in ok])
        accepted_mean_gap = jnp.array([r.mean_accepted_constraint_gap for r in ok])
        accepted_median_gap = jnp.array([r.median_accepted_constraint_gap for r in ok])
        mean_delta = jnp.array([r.mean_delta_logl for r in ok])
        median_delta = jnp.array([r.median_delta_logl for r in ok])
        fallback_gap = jnp.array([r.fallback_rejection_gap for r in ok])
        mean_reflections = jnp.array([r.mean_reflections_per_proposal for r in ok])
        max_reflections = jnp.array([r.max_reflections_per_proposal for r in ok])
        frac_reflected = jnp.array([r.fraction_proposals_with_reflection for r in ok])

        refl_fail_rate = jnp.array([r.reflection_failure_rate for r in ok])

        def fmt_nanmean(arr: jax.Array, precision: int = 6) -> str:
            value = float(jnp.nanmean(arr))
            return "n/a" if jnp.isnan(value) else f"{value:.{precision}f}"

        def fmt_nanmax(arr: jax.Array, precision: int = 4) -> str:
            if bool(jnp.all(jnp.isnan(arr))):
                return "n/a"
            value = float(jnp.nanmax(arr))
            return "n/a" if jnp.isnan(value) else f"{value:.{precision}f}"

        print(f"  mean acceptance rate: {float(jnp.nanmean(ar)):.4f}")
        print(f"  mean boundary-crossing rate: {float(jnp.nanmean(bcr)):.4f}")
        print(f"  mean fallback/rejection rate: {float(jnp.nanmean(fr)):.4f}")
        print(f"  mean(start_logL - constraint): {fmt_nanmean(start_mean_gap)}")
        print(f"  median(start_logL - constraint): {fmt_nanmean(start_median_gap)}")
        print(f"  mean(final_logL - constraint): {float(jnp.nanmean(mean_gap)):.6f}")
        print(f"  median(final_logL - constraint): {float(jnp.nanmean(med_gap)):.6f}")
        print(
            f"  mean(final_logL - constraint, accepted): {fmt_nanmean(accepted_mean_gap)}"
        )
        print(
            f"  median(final_logL - constraint, accepted): {fmt_nanmean(accepted_median_gap)}"
        )
        print(
            f"  mean(delta_logL = final_logL - start_logL): {fmt_nanmean(mean_delta)}"
        )
        print(f"  median(delta_logL): {fmt_nanmean(median_delta)}")
        print(f"  fallback/rejection gap: {fmt_nanmean(fallback_gap)}")
        print(
            f"  mean reflections per proposal: {fmt_nanmean(mean_reflections, precision=4)}"
        )
        print(
            f"  max reflections per proposal: {fmt_nanmax(max_reflections, precision=4)}"
        )
        print(
            f"  fraction proposals with reflection: {fmt_nanmean(frac_reflected, precision=4)}"
        )
        print(f"  reflection failure rate: {fmt_nanmean(refl_fail_rate, precision=4)}")
        print(f"  max(final_logL - constraint): {float(jnp.nanmean(max_gap)):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke validation sweep for static/dynamic NSS and GGNS. "
            "For current Hamiltonian-style GGNS, conservative settings "
            "step_size=0.001 and num_inner_steps=1 are recommended."
        )
    )
    parser.add_argument("--num-live", type=int, default=40)
    parser.add_argument("--initial-num-steps", type=int, default=10)
    parser.add_argument("--refinement-num-steps", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--static-num-steps", type=int, default=16)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--samplers", type=str, default="nss,ggns")
    parser.add_argument("--modes", type=str, default="static,dynamic")
    parser.add_argument("--ggns-step-size", type=float, default=0.001)
    parser.add_argument("--ggns-num-integration-steps", type=int, default=None)
    parser.add_argument("--ggns-num-inner-steps", type=int, default=1)
    parser.add_argument(
        "--replacement-diagnostics",
        action="store_true",
        help="Print per-replacement diagnostics for the global NSS path.",
    )
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
        "--cluster-aware-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cluster-aware-scale-floor", type=float, default=1e-12)
    parser.add_argument("--cluster-aware-radius", type=float, default=None)
    parser.add_argument("--cluster-aware-min-cluster-size", type=int, default=3)
    parser.add_argument(
        "--cluster-aware-max-condition-number", type=float, default=1e12
    )
    parser.add_argument(
        "--cluster-aware-covariance-regularization", type=float, default=1e-6
    )
    parser.add_argument("--cluster-aware-auto-fallback", action="store_true")
    parser.add_argument("--cluster-aware-warmup-attempts", type=int, default=25)
    parser.add_argument("--cluster-aware-min-success-rate", type=float, default=0.5)
    parser.add_argument("--cluster-aware-max-runtime-ratio", type=float, default=2.0)
    parser.add_argument(
        "--nss-eager",
        "--disable-nss-jit",
        dest="nss_eager",
        action="store_true",
        help="Disable JAX JIT while running NSS validation loops.",
    )
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
    print(
        "Note: with conservative GGNS smoke settings "
        "(step_size=0.001, num_inner_steps=1), delta_logL is often ~0, "
        "so proposals are not expected to be strong likelihood climbers."
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
                        replacement_strategy=args.replacement_strategy,
                        cluster_aware_eager=args.cluster_aware_eager,
                        cluster_aware_standardize=args.cluster_aware_standardize,
                        cluster_aware_scale_floor=args.cluster_aware_scale_floor,
                        cluster_aware_radius=args.cluster_aware_radius,
                        cluster_aware_min_cluster_size=args.cluster_aware_min_cluster_size,
                        cluster_aware_max_condition_number=args.cluster_aware_max_condition_number,
                        cluster_aware_covariance_regularization=args.cluster_aware_covariance_regularization,
                        cluster_aware_auto_fallback=args.cluster_aware_auto_fallback,
                        cluster_aware_warmup_attempts=args.cluster_aware_warmup_attempts,
                        cluster_aware_min_success_rate=args.cluster_aware_min_success_rate,
                        cluster_aware_max_runtime_ratio=args.cluster_aware_max_runtime_ratio,
                        replacement_diagnostics=args.replacement_diagnostics,
                        nss_eager=args.nss_eager,
                    )
                    results.append(metrics)
                    print(
                        f"  {mode} {sampler} seed={seed}: logZ={metrics.logz:.6f}, "
                        f"ESS={metrics.ess:.2f}, runtime={metrics.run_seconds:.3f}s, "
                        f"ms/step={metrics.ms_per_step:.3f}, failures=0"
                    )
                except Exception as exc:
                    print(
                        f"  {mode} {sampler} seed={seed}: FAILED ({type(exc).__name__}: {exc})"
                    )
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
