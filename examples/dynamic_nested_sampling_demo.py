"""Minimal demo for static/dynamic nested sampling with NSS and GGNS kernels."""

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
            f"dead={batch.metadata.num_dead}"
        )


def main():
    num_live = 80
    positions = jax.random.uniform(
        jax.random.key(0), shape=(num_live, 2), minval=-4.0, maxval=4.0
    )

    nss_algo = nss.as_top_level_api(
        logprior_fn=uniform_logprior_2d,
        loglikelihood_fn=gaussian_mixture_loglikelihood,
        num_inner_steps=8,
        num_delete=2,
    )
    ggns_algo = ggns.as_top_level_api(
        logprior_fn=uniform_logprior_2d,
        loglikelihood_fn=gaussian_mixture_loglikelihood,
        num_inner_steps=8,
        num_delete=2,
        step_size=0.05,
    )

    nss_state = nss_algo.init(positions, rng_key=jax.random.key(1))
    _, static_nss_batch = utils.run_bounded_batch(
        rng_key=jax.random.key(2),
        state=nss_state,
        step_fn=nss_algo.step,
        num_steps=50,
        loglikelihood_lower=-jnp.inf,
    )
    static_nss_merged = utils.merge_bounded_batches([static_nss_batch])
    summarize("Static NSS", static_nss_merged, (static_nss_batch,))

    nss_state = nss_algo.init(positions, rng_key=jax.random.key(3))
    _, dynamic_nss = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(4),
        state=nss_state,
        step_fn=nss_algo.step,
        initial_num_steps=30,
        refinement_num_steps=20,
        max_batches=3,
    )
    summarize("Dynamic NSS", dynamic_nss.merged, dynamic_nss.batches)

    ggns_state = ggns_algo.init(positions, rng_key=jax.random.key(5))
    _, dynamic_ggns = utils.run_dynamic_posterior_scheduler(
        rng_key=jax.random.key(6),
        state=ggns_state,
        step_fn=ggns_algo.step,
        initial_num_steps=30,
        refinement_num_steps=20,
        max_batches=3,
    )
    summarize("Dynamic GGNS", dynamic_ggns.merged, dynamic_ggns.batches)


if __name__ == "__main__":
    main()
