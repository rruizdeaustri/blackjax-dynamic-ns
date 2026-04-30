"""Minimal demo for static/dynamic nested sampling with NSS and GGNS kernels."""
import time

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from blackjax.ns import ggns, nss, utils


def uniform_logprior_2d(x):
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x):
    m1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    m2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(m1, m2)


def summarize(name, merged, batches):
    print(f"\n{name}")
    print(f"  logZ: {float(merged.logZ):.4f}")
    print(f"  posterior weight sum: {float(jnp.sum(merged.posterior_weights)):.6f}")
    print(f"  ESS: {float(merged.ess):.2f}")
    print(f"  num batches: {merged.metadata.num_batches}")
    print(f"  num dead points: {merged.metadata.num_dead}")
    for i, batch in enumerate(batches):
        print(
            "  "
            f"batch {i}: [{batch.loglikelihood_lower:.3f}, {batch.loglikelihood_upper:.3f}) "
            f"dead={batch.metadata.num_dead} empty={batch.metadata.is_empty} "
            f"req_steps={batch.metadata.requested_num_steps} "
            f"exec_steps={batch.metadata.num_steps} "
            f"width={batch.metadata.interval_width:.3e} "
            f"term={batch.metadata.terminated_reason}"
        )


def print_profile_summary(name, run_only_seconds, executed_steps, num_batches):
    avg_step_ms = 1e3 * run_only_seconds / max(executed_steps, 1)
    avg_batch_ms = 1e3 * run_only_seconds / max(num_batches, 1)
    print(
        f"{name} profiling:\n"
        f"  executed steps: {executed_steps}\n"
        f"  run-only wall time: {run_only_seconds:.3f}s\n"
        f"  avg time / NS step: {avg_step_ms:.2f}ms\n"
        f"  avg time / bounded batch: {avg_batch_ms:.2f}ms"
    )


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    num_live = 40
    positions = jax.random.uniform(
        jax.random.key(0), shape=(num_live, 2), minval=-4.0, maxval=4.0
    )

    nss_algo = nss.as_top_level_api(
        logprior_fn=uniform_logprior_2d,
        loglikelihood_fn=gaussian_mixture_loglikelihood,
        num_inner_steps=4,
        num_delete=2,
    )
    ggns_algo = ggns.as_top_level_api(
        logprior_fn=uniform_logprior_2d,
        loglikelihood_fn=gaussian_mixture_loglikelihood,
        num_inner_steps=4,
        num_delete=2,
        step_size=0.05,
    )

    nss_state = nss_algo.init(positions, rng_key=jax.random.key(1))

    # Warm-up compile + run timing.
    t0 = time.perf_counter()
    _, static_warmup_batch = utils.run_bounded_batch(
        rng_key=jax.random.key(21),
        state=nss_state,
        step_fn=nss_algo.step,
        num_steps=20,
        loglikelihood_lower=-jnp.inf,
    )
    jax.block_until_ready(static_warmup_batch.dead_point_loglikelihoods)
    static_compile_dt = time.perf_counter() - t0

    # Second call approximates run-only timing.
    t0 = time.perf_counter()
    _, static_nss_batch = utils.run_bounded_batch(
        rng_key=jax.random.key(2),
        state=nss_state,
        step_fn=nss_algo.step,
        num_steps=20,
        loglikelihood_lower=-jnp.inf,
    )
    jax.block_until_ready(static_nss_batch.dead_point_loglikelihoods)
    static_nss_dt = time.perf_counter() - t0

    static_nss_merged = utils.merge_bounded_batches([static_nss_batch])
    summarize("Static NSS", static_nss_merged, (static_nss_batch,))
    print(f"  compile+run: {static_compile_dt:.3f}s")
    print(f"  run-only: {static_nss_dt:.3f}s")
    print_profile_summary(
        "Static NSS",
        run_only_seconds=static_nss_dt,
        executed_steps=static_nss_batch.metadata.num_steps,
        num_batches=1,
    )

    nss_state = nss_algo.init(positions, rng_key=jax.random.key(3))

    t0 = time.perf_counter()
    _, dynamic_nss_warmup = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(31),
        state=nss_state,
        step_fn=nss_algo.step,
        initial_num_steps=12,
        refinement_num_steps=8,
        max_batches=2,
    )
    jax.block_until_ready(dynamic_nss_warmup.merged.logZ)
    dynamic_nss_compile_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, dynamic_nss = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(4),
        state=nss_state,
        step_fn=nss_algo.step,
        initial_num_steps=12,
        refinement_num_steps=8,
        max_batches=2,
    )
    jax.block_until_ready(dynamic_nss.merged.logZ)
    dynamic_nss_dt = time.perf_counter() - t0

    summarize("Dynamic NSS", dynamic_nss.merged, dynamic_nss.batches)
    print(utils.summarize_dynamic_result(dynamic_nss))
    print(f"  compile+run: {dynamic_nss_compile_dt:.3f}s")
    print(f"  run-only: {dynamic_nss_dt:.3f}s")

    dynamic_nss_steps = sum(batch.metadata.num_steps for batch in dynamic_nss.batches)
    print_profile_summary(
        "Dynamic NSS",
        run_only_seconds=dynamic_nss_dt,
        executed_steps=dynamic_nss_steps,
        num_batches=dynamic_nss.merged.metadata.num_batches,
    )

    ggns_state = ggns_algo.init(positions, rng_key=jax.random.key(5))

    t0 = time.perf_counter()
    _, dynamic_ggns_warmup = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(41),
        state=ggns_state,
        step_fn=ggns_algo.step,
        initial_num_steps=12,
        refinement_num_steps=8,
        max_batches=2,
    )
    jax.block_until_ready(dynamic_ggns_warmup.merged.logZ)
    dynamic_ggns_compile_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, dynamic_ggns = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(6),
        state=ggns_state,
        step_fn=ggns_algo.step,
        initial_num_steps=12,
        refinement_num_steps=8,
        max_batches=2,
    )
    jax.block_until_ready(dynamic_ggns.merged.logZ)
    dynamic_ggns_dt = time.perf_counter() - t0

    summarize("Dynamic GGNS", dynamic_ggns.merged, dynamic_ggns.batches)
    print(utils.summarize_dynamic_result(dynamic_ggns))
    print(f"  compile+run: {dynamic_ggns_compile_dt:.3f}s")
    print(f"  run-only: {dynamic_ggns_dt:.3f}s")

    dynamic_ggns_steps = sum(batch.metadata.num_steps for batch in dynamic_ggns.batches)
    print_profile_summary(
        "Dynamic GGNS",
        run_only_seconds=dynamic_ggns_dt,
        executed_steps=dynamic_ggns_steps,
        num_batches=dynamic_ggns.merged.metadata.num_batches,
    )


if __name__ == "__main__":
    main()
