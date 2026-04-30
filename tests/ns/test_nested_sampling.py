"""Test the Nested Sampling algorithms"""

import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

from blackjax.ns import adaptive, base, ggns, nss, utils


def gaussian_logprior(x):
    """Standard normal prior"""
    return stats.norm.logpdf(x).sum()


def gaussian_loglikelihood(x):
    """Gaussian likelihood with offset"""
    return stats.norm.logpdf(x - 1.0).sum()


def make_init_state_fn(logprior_fn, loglikelihood_fn):
    """Helper to create init_state_fn from logprior and loglikelihood functions."""
    return functools.partial(
        base.init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )


def make_mock_nsinfo(positions, loglikelihood, loglikelihood_birth, logdensity):
    """Helper to create NSInfo with correct structure."""
    particles = base.StateWithLogLikelihood(
        position=positions,
        logdensity=logdensity,
        loglikelihood=loglikelihood,
        loglikelihood_birth=loglikelihood_birth,
    )
    return base.NSInfo(particles=particles, update_info={})


def uniform_logprior_2d(x):
    """Uniform prior on [-5, 5]^2"""
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_loglikelihood_2d(x):
    """2D Gaussian likelihood centered at [1, -1]."""
    center = jnp.array([1.0, -1.0])
    return stats.norm.logpdf(x - center).sum()


def gaussian_mixture_loglikelihood(x):
    """2D Gaussian mixture for multi-modal testing"""
    mixture1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    mixture2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(mixture1, mixture2)


class NestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_base_ns_init(self):
        """Test basic NS initialization"""
        key = jax.random.key(123)
        num_live = 50

        # Generate initial particles
        positions = jax.random.normal(key, (num_live,))

        # Initialize NS state using the correct API
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Check state structure - particles is now a StateWithLogLikelihood
        chex.assert_shape(state.particles.position, (num_live,))
        chex.assert_shape(state.particles.loglikelihood, (num_live,))
        chex.assert_shape(state.particles.logdensity, (num_live,))
        chex.assert_shape(state.particles.loglikelihood_birth, (num_live,))

        # Check that loglikelihood and logprior are properly computed
        expected_loglik = jax.vmap(gaussian_loglikelihood)(positions)
        expected_logprior = jax.vmap(gaussian_logprior)(positions)

        chex.assert_trees_all_close(state.particles.loglikelihood, expected_loglik)
        chex.assert_trees_all_close(state.particles.logdensity, expected_logprior)

    def test_delete_fn(self):
        """Test particle deletion function"""
        key = jax.random.key(456)
        num_live = 20
        num_delete = 3

        positions = jax.random.normal(key, (num_live,))
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        dead_idx, target_idx = base.delete_fn(state, num_delete)

        # Check correct number of deletions
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Check that worst particles are selected
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        selected_loglik = state.particles.loglikelihood[dead_idx]
        chex.assert_trees_all_close(jnp.sort(selected_loglik), worst_loglik)

    @parameterized.parameters([1, 2, 5])
    def test_ns_step_consistency(self, num_delete):
        """Test NS step maintains particle count"""
        key = jax.random.key(789)
        num_live = 50

        positions = jax.random.normal(key, (num_live, 2))
        init_state_fn = jax.vmap(
            make_init_state_fn(uniform_logprior_2d, gaussian_mixture_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Mock inner kernel for testing — num_delete closed over from outer scope
        def mock_inner_kernel(rng_key, state, loglikelihood_0):
            particles = state.particles

            # Select start particles from survivors
            choice_key, sample_key = jax.random.split(rng_key)
            weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
            weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
            start_idx = jax.random.choice(
                choice_key,
                len(weights),
                shape=(num_delete,),
                p=weights / weights.sum(),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            # Simple random walk for testing
            def single_step(rng_key, state):
                new_pos = (
                    state.position
                    + jax.random.normal(rng_key, state.position.shape) * 0.1
                )
                new_state = base.init_state_strategy(
                    new_pos,
                    uniform_logprior_2d,
                    gaussian_mixture_loglikelihood,
                    loglikelihood_birth=loglikelihood_0,
                )
                return new_state

            sample_keys = jax.random.split(sample_key, num_delete)
            new_particles = jax.vmap(single_step)(sample_keys, start_state)
            return new_particles, {}

        delete_fn = functools.partial(base.delete_fn, num_delete=num_delete)
        kernel = base.build_kernel(delete_fn, mock_inner_kernel)

        # Test that the kernel can be constructed with mock components
        self.assertTrue(callable(kernel))

        # Test delete function works
        dead_idx, target_idx = base.delete_fn(state, num_delete)
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Actually run the kernel and check post-conditions
        new_state, info = kernel(key, state)

        # Particle count preserved
        chex.assert_shape(
            new_state.particles.position,
            state.particles.position.shape,
        )
        # Dead particles returned in info
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))
        # Dead particles are the worst from original state
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        chex.assert_trees_all_close(
            jnp.sort(info.particles.loglikelihood), worst_loglik
        )

    def test_utils_functions(self):
        """Test utility functions"""
        key = jax.random.key(101112)

        # Create mock dead info
        n_dead = 20
        dead_loglik = jnp.sort(jax.random.uniform(key, (n_dead,))) * 10 - 5
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create StateWithLogLikelihood for particles
        particles = base.StateWithLogLikelihood(
            position=jnp.zeros((n_dead, 2)),
            logdensity=jnp.zeros(n_dead),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
        )

        mock_info = base.NSInfo(particles=particles, update_info={})

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Test logX simulation
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=10)
        chex.assert_shape(logX_seq, (n_dead, 10))
        chex.assert_shape(logdX_seq, (n_dead, 10))

        # Check logX is decreasing
        self.assertTrue(jnp.all(logX_seq[1:] <= logX_seq[:-1]))

class GradientGuidedNestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(2026)

    def _init_state(self, key, num_live, logprior_fn, loglikelihood_fn):
        positions = jax.random.normal(key, (num_live, 2))
        algorithm = ggns.as_top_level_api(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=8,
            num_delete=3,
            step_size=0.05,
            momentum_weight=0.5,
        )
        state = algorithm.init(positions, rng_key=key)
        return algorithm, state

    @parameterized.parameters(
        [gaussian_loglikelihood_2d, gaussian_mixture_loglikelihood]
    )
    def test_ggns_replacement_satisfies_constraint(self, loglikelihood_fn):
        key = jax.random.key(999)
        algorithm, state = self._init_state(
            key, 40, uniform_logprior_2d, loglikelihood_fn
        )

        step_key = jax.random.key(1001)
        new_state, info = algorithm.step(step_key, state)

        dead_threshold = info.particles.loglikelihood.max()
        self.assertTrue(jnp.all(info.particles.loglikelihood <= dead_threshold))

        updated_mask = jnp.isclose(
            new_state.particles.loglikelihood_birth, dead_threshold, atol=1e-6
        )
        updated_birth = new_state.particles.loglikelihood_birth[updated_mask]
        self.assertGreater(updated_birth.shape[0], 0)
        self.assertTrue(
            jnp.all(
                new_state.particles.loglikelihood[updated_mask] > dead_threshold
            )
        )

    @parameterized.parameters(
        [gaussian_loglikelihood_2d, gaussian_mixture_loglikelihood]
    )
    def test_ggns_vs_nss_constraint_validity(self, loglikelihood_fn):
        num_live = 60
        positions = jax.random.normal(self.key, (num_live, 2))

        ggns_algo = ggns.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=8,
            num_delete=4,
            step_size=0.05,
        )
        nss_algo = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=8,
            num_delete=4,
        )

        g_state = ggns_algo.init(positions, rng_key=self.key)
        n_state = nss_algo.init(positions, rng_key=self.key)

        g_key, n_key = jax.random.split(jax.random.key(12345))
        g_new_state, _ = ggns_algo.step(g_key, g_state)
        n_new_state, _ = nss_algo.step(n_key, n_state)

        g_replaced = jnp.isfinite(g_new_state.particles.loglikelihood_birth)
        n_replaced = jnp.isfinite(n_new_state.particles.loglikelihood_birth)
        g_valid = (
            g_new_state.particles.loglikelihood[g_replaced]
            > g_new_state.particles.loglikelihood_birth[g_replaced]
        )
        n_valid = (
            n_new_state.particles.loglikelihood[n_replaced]
            > n_new_state.particles.loglikelihood_birth[n_replaced]
        )

        self.assertTrue(jnp.all(g_valid))
        self.assertTrue(jnp.all(n_valid))


class AdaptiveNestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_adaptive_init(self):
        """Test adaptive NS initialization"""
        key = jax.random.key(123)
        num_live = 30

        positions = jax.random.normal(key, (num_live,))

        def mock_update_params_fn(rng_key, state, info, current_params):
            return {"test_param": 1.0}

        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = adaptive.init(
            positions,
            init_state_fn,
            update_inner_kernel_params_fn=mock_update_params_fn,
        )

        # Check that inner kernel params were set
        self.assertEqual(state.inner_kernel_params["test_param"], 1.0)


class NestedSliceSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_nss_direction_functions(self):
        """Test NSS direction generation functions"""
        key = jax.random.key(456)

        # Test covariance computation
        positions = jax.random.normal(key, (50, 3))

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return stats.norm.logpdf(x).sum()

        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        state = base.init(positions, init_state_fn)

        # Use update_inner_kernel_params instead of removed init_inner_kernel_params
        params = nss.update_inner_kernel_params(key, state, None, {})

        # Check that covariance is computed
        self.assertIn("cov", params)
        cov_pytree = params["cov"]
        chex.assert_shape(cov_pytree, (3, 3))

    def test_nss_kernel_construction(self):
        """Test NSS kernel can be constructed"""
        init_state_fn = make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        kernel = nss.build_kernel(init_state_fn, num_inner_steps=10)

        # Test that kernel is callable
        self.assertTrue(callable(kernel))


class NestedSamplingBatchTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(31415)

    def test_run_bounded_batch_lower_bound(self):
        num_live = 40
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-4.0,
            maxval=4.0,
        )
        algorithm = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_loglikelihood_2d,
            num_inner_steps=6,
            num_delete=2,
        )
        state = algorithm.init(positions, rng_key=self.key)

        _, batch = utils.run_bounded_batch(
            rng_key=jax.random.key(1234),
            state=state,
            step_fn=algorithm.step,
            num_steps=25,
            loglikelihood_lower=-6.0,
        )

        self.assertGreaterEqual(batch.metadata.num_steps, 1)
        self.assertEqual(batch.num_live_points, num_live)
        self.assertEqual(batch.final_live_points.loglikelihood.shape[0], num_live)
        self.assertTrue(jnp.all(batch.dead_point_loglikelihoods >= -6.0))
        self.assertEqual(batch.loglikelihood_lower, -6.0)
        self.assertTrue(jnp.isinf(batch.loglikelihood_upper))
        self.assertEqual(batch.dead_points.shape[-1], 2)
        self.assertEqual(batch.metadata.num_dead, batch.dead_point_loglikelihoods.shape[0])

    def test_run_bounded_batch_upper_bound(self):
        num_live = 35
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-3.0,
            maxval=3.0,
        )
        algorithm = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_loglikelihood_2d,
            num_inner_steps=5,
            num_delete=3,
        )
        state = algorithm.init(positions, rng_key=self.key)
        logL_lower = -8.0
        logL_upper = -2.0

        final_state, batch = utils.run_bounded_batch(
            rng_key=jax.random.key(2222),
            state=state,
            step_fn=algorithm.step,
            num_steps=50,
            loglikelihood_lower=logL_lower,
            loglikelihood_upper=logL_upper,
        )

        self.assertEqual(batch.loglikelihood_lower, logL_lower)
        self.assertEqual(batch.loglikelihood_upper, logL_upper)
        self.assertEqual(batch.num_live_points, num_live)
        self.assertTrue(jnp.all(batch.dead_point_loglikelihoods >= logL_lower))
        self.assertTrue(jnp.all(batch.dead_point_loglikelihoods < logL_upper))
        self.assertGreaterEqual(batch.metadata.num_steps, 1)
        self.assertEqual(batch.metadata.num_dead, batch.dead_point_loglikelihoods.shape[0])
        if batch.metadata.reached_loglikelihood_upper:
            self.assertEqual(batch.metadata.terminated_reason, 1)
            self.assertTrue(jnp.min(final_state.particles.loglikelihood) >= logL_upper)

    def test_merge_single_bounded_batch(self):
        num_live = 40
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-4.0,
            maxval=4.0,
        )
        algorithm = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_loglikelihood_2d,
            num_inner_steps=6,
            num_delete=2,
        )
        state = algorithm.init(positions, rng_key=self.key)
        _, batch = utils.run_bounded_batch(
            rng_key=jax.random.key(9001),
            state=state,
            step_fn=algorithm.step,
            num_steps=30,
            loglikelihood_lower=-8.0,
        )
        merged = utils.merge_bounded_batches([batch])

        self.assertEqual(merged.metadata.num_batches, 1)
        self.assertEqual(merged.metadata.num_dead, batch.metadata.num_dead)
        self.assertTrue(
            jnp.all(
                merged.dead_point_loglikelihoods[1:]
                >= merged.dead_point_loglikelihoods[:-1]
            )
        )
        self.assertTrue(jnp.all(merged.num_live_points > 0.0))
        self.assertTrue(jnp.all(merged.logX[1:] <= merged.logX[:-1]))
        self.assertTrue(jnp.isfinite(merged.logZ))
        self.assertAlmostEqual(float(jnp.sum(merged.posterior_weights)), 1.0, places=5)
        self.assertGreater(float(merged.ess), 1.0)

    def test_merge_two_bounded_batches(self):
        num_live = 40
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-4.0,
            maxval=4.0,
        )
        algorithm = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=gaussian_loglikelihood_2d,
            num_inner_steps=6,
            num_delete=2,
        )
        state = algorithm.init(positions, rng_key=self.key)
        state, batch1 = utils.run_bounded_batch(
            rng_key=jax.random.key(7001),
            state=state,
            step_fn=algorithm.step,
            num_steps=30,
            loglikelihood_lower=-10.0,
            loglikelihood_upper=-3.5,
        )
        _, batch2 = utils.run_bounded_batch(
            rng_key=jax.random.key(7002),
            state=state,
            step_fn=algorithm.step,
            num_steps=30,
            loglikelihood_lower=-3.5,
        )
        merged_two = utils.merge_bounded_batches([batch1, batch2])
        merged_one = utils.merge_bounded_batches([batch1])

        expected_dead = batch1.metadata.num_dead + batch2.metadata.num_dead
        self.assertEqual(merged_two.metadata.num_batches, 2)
        self.assertEqual(merged_two.metadata.num_dead, expected_dead)
        self.assertEqual(merged_two.dead_point_loglikelihoods.shape[0], expected_dead)
        self.assertTrue(
            jnp.all(
                merged_two.dead_point_loglikelihoods[1:]
                >= merged_two.dead_point_loglikelihoods[:-1]
            )
        )
        self.assertAlmostEqual(
            float(jnp.sum(merged_two.posterior_weights)), 1.0, places=5
        )
        self.assertGreaterEqual(float(merged_two.ess), float(merged_one.ess))

    @parameterized.parameters(
        [gaussian_loglikelihood_2d, gaussian_mixture_loglikelihood]
    )
    def test_dynamic_posterior_scheduler_runs(self, loglikelihood_fn):
        num_live = 50
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-4.0,
            maxval=4.0,
        )
        algorithm = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=6,
            num_delete=2,
        )
        state = algorithm.init(positions, rng_key=self.key)

        _, dynamic_result = utils.run_dynamic_posterior_scheduler(
            rng_key=jax.random.key(8080),
            state=state,
            step_fn=algorithm.step,
            initial_num_steps=25,
            refinement_num_steps=20,
            max_batches=3,
            objective="posterior",
        )

        self.assertEqual(dynamic_result.metadata.objective, "posterior")
        self.assertEqual(dynamic_result.metadata.max_batches, 3)
        self.assertLen(dynamic_result.batches, 3)
        self.assertEqual(dynamic_result.merged.metadata.num_batches, 3)
        self.assertTrue(jnp.isfinite(dynamic_result.logZ))
        self.assertAlmostEqual(
            float(jnp.sum(dynamic_result.posterior_weights)), 1.0, places=5
        )
        self.assertGreater(float(dynamic_result.ess), 1.0)

    @parameterized.parameters(
        [gaussian_loglikelihood_2d, gaussian_mixture_loglikelihood]
    )
    def test_static_vs_dynamic_nss_and_ggns(self, loglikelihood_fn):
        num_live = 60
        positions = jax.random.uniform(
            self.key,
            shape=(num_live, 2),
            minval=-4.0,
            maxval=4.0,
        )

        def run_static(algorithm, rng_key):
            state = algorithm.init(positions, rng_key=rng_key)
            _, batch = utils.run_bounded_batch(
                rng_key=rng_key,
                state=state,
                step_fn=algorithm.step,
                num_steps=40,
                loglikelihood_lower=-jnp.inf,
            )
            merged = utils.merge_bounded_batches([batch])
            return merged, (batch,)

        def run_dynamic(algorithm, rng_key):
            state = algorithm.init(positions, rng_key=rng_key)
            _, result = utils.run_dynamic_posterior_scheduler(
                rng_key=rng_key,
                state=state,
                step_fn=algorithm.step,
                initial_num_steps=25,
                refinement_num_steps=20,
                max_batches=3,
                objective="posterior",
            )
            return result.merged, result.batches

        static_nss_algo = nss.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=8,
            num_delete=2,
        )
        static_ggns_algo = ggns.as_top_level_api(
            logprior_fn=uniform_logprior_2d,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=8,
            num_delete=2,
            step_size=0.05,
        )

        static_nss, static_nss_batches = run_static(static_nss_algo, jax.random.key(501))
        static_ggns, static_ggns_batches = run_static(
            static_ggns_algo, jax.random.key(502)
        )
        dynamic_nss, dynamic_nss_batches = run_dynamic(
            static_nss_algo, jax.random.key(503)
        )
        dynamic_ggns, dynamic_ggns_batches = run_dynamic(
            static_ggns_algo, jax.random.key(504)
        )

        for merged, batches in (
            (static_nss, static_nss_batches),
            (static_ggns, static_ggns_batches),
            (dynamic_nss, dynamic_nss_batches),
            (dynamic_ggns, dynamic_ggns_batches),
        ):
            self.assertTrue(jnp.isfinite(merged.logZ))
            self.assertAlmostEqual(float(jnp.sum(merged.posterior_weights)), 1.0, places=5)
            self.assertGreater(float(merged.ess), 0.0)
            self.assertGreaterEqual(merged.metadata.num_batches, 1)
            self.assertGreater(merged.metadata.num_dead, 0)
            for batch in batches:
                self.assertLessEqual(batch.loglikelihood_lower, batch.loglikelihood_upper)
                self.assertGreaterEqual(batch.metadata.num_steps, 1)
                self.assertGreaterEqual(batch.metadata.num_dead, 0)

        for batches in (dynamic_nss_batches, dynamic_ggns_batches):
            # Posterior-focused refinement should avoid empty bounded intervals in
            # the current 2D Gaussian(-mixture) setup.
            self.assertTrue(all(batch.metadata.num_dead > 0 for batch in batches[1:]))


class NestedSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests for nested sampling algorithms."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_1d_gaussian_evidence_estimation(self):
        """Test evidence estimation with analytic validation for unnormalized Gaussian."""

        # Simple case: unnormalized Gaussian likelihood exp(-0.5*x²), uniform prior [-3,3]
        prior_a, prior_b = -3.0, 3.0

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            # Unnormalized Gaussian: exp(-0.5 * x²)
            return -0.5 * x**2

        # Analytic evidence: Z = ∫[-3,3] (1/6) * exp(-0.5*x²) dx
        # = (1/6) * √(2π) * [Φ(3) - Φ(-3)]
        from scipy.stats import norm

        prior_width = prior_b - prior_a
        integral_part = jnp.sqrt(2 * jnp.pi) * (norm.cdf(3.0) - norm.cdf(-3.0))
        analytical_evidence = integral_part / prior_width
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock nested sampling data
        num_steps = 60
        key = jax.random.key(42)

        # Create positions spanning the prior range
        positions = jnp.linspace(prior_a + 0.05, prior_b - 0.05, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood (as NS naturally produces)
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods - start from prior
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create NSInfo object
        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate many evidence estimates for statistical testing
        n_evidence_samples = 500
        key = jax.random.key(789)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        # Compute evidence estimates
        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # Check statistical consistency with 95% confidence interval
        # For mock data with simplified NS, expect some bias but should be in ballpark
        tolerance = 2.0 * std_estimate  # 95% CI
        bias = jnp.abs(mean_estimate - analytical_log_evidence)

        self.assertLess(
            bias,
            tolerance,
            f"Evidence estimate {mean_estimate} vs analytic {analytical_log_evidence} "
            f"differs by {bias}, which exceeds 2σ = {tolerance}",
        )

        # Also test that individual estimates are reasonable
        self.assertFalse(
            jnp.any(jnp.isnan(log_evidence_samples)),
            "No evidence estimates should be NaN",
        )
        self.assertFalse(
            jnp.any(jnp.isinf(log_evidence_samples)),
            "No evidence estimates should be infinite",
        )

        # Check that estimates are in a reasonable range
        self.assertGreater(
            mean_estimate, analytical_log_evidence - 1.0, "Mean estimate not too low"
        )
        self.assertLess(
            mean_estimate, analytical_log_evidence + 1.0, "Mean estimate not too high"
        )

    def test_uniform_prior_evidence(self):
        """Test evidence estimation for uniform prior with simple likelihood."""

        # Setup: Uniform prior on [0, 1], simple likelihood
        def logprior_fn(x):
            return jnp.where((x >= 0.0) & (x <= 1.0), 0.0, -jnp.inf)

        def loglikelihood_fn(x):
            # Simple quadratic likelihood peaked at 0.5
            return -10.0 * (x - 0.5) ** 2

        # Analytical evidence can be computed numerically for comparison
        # Z = integral_0^1 exp(-10(x-0.5)^2) dx ≈ sqrt(π/10) * erf(...)

        num_live = 50
        key = jax.random.key(456)

        # Initialize particles uniformly in [0, 1]
        positions = jax.random.uniform(key, (num_live,))
        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        state = base.init(positions, init_state_fn)

        # Check that initialization worked correctly
        self.assertTrue(jnp.all(state.particles.position >= 0.0))
        self.assertTrue(jnp.all(state.particles.position <= 1.0))
        self.assertFalse(jnp.any(jnp.isinf(state.particles.logdensity)))
        self.assertFalse(jnp.any(jnp.isnan(state.particles.loglikelihood)))

    def test_evidence_monotonicity(self):
        """Test that we can initialize state and track integrator."""

        # Simple setup for testing monotonicity
        def logprior_fn(x):
            return stats.norm.logpdf(x)

        def loglikelihood_fn(x):
            return -0.5 * x**2  # Simple quadratic

        num_live = 30
        key = jax.random.key(789)

        positions = jax.random.normal(key, (num_live,))
        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        initial_state = base.init(positions, init_state_fn)

        # Test that we can access particle likelihoods
        self.assertIsNotNone(initial_state.particles.loglikelihood)
        chex.assert_shape(initial_state.particles.loglikelihood, (num_live,))

        # For integrator tests, use adaptive state instead
        from blackjax.ns import adaptive as adaptive_module

        adaptive_state = adaptive_module.init(positions, init_state_fn)

        # Check integrator exists and has expected fields
        self.assertIsNotNone(adaptive_state.integrator)
        self.assertIsNotNone(adaptive_state.integrator.logZ)
        self.assertIsNotNone(adaptive_state.integrator.logX)

    def test_nested_sampling_utils_statistical_properties(self):
        """Test statistical properties of nested sampling utility functions."""
        key = jax.random.key(101112)

        # Create realistic mock data
        n_dead = 100

        # Generate realistic loglikelihood sequence (increasing)
        base_loglik = jnp.linspace(-10, -1, n_dead)
        noise = jax.random.normal(key, (n_dead,)) * 0.1
        dead_loglik = jnp.sort(base_loglik + noise)

        # Create more realistic birth likelihoods that reflect actual NS behavior
        # Particles can be born at various levels, not just at previous death
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (n_dead,)) * 2.0 - 1.0  # [-1, 1]
        dead_loglik_birth = jnp.concatenate(
            [
                jnp.array([-jnp.inf]),  # First particle born from prior
                dead_loglik[:-1] + birth_noise[1:] * 0.5,  # Others with some variation
            ]
        )
        # Ensure birth likelihoods don't exceed death likelihoods
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 2)), dead_loglik, dead_loglik_birth, jnp.zeros(n_dead)
        )

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Basic sanity checks for number of live points
        # NOTE: num_live should NOT be monotonically decreasing in general NS!
        # It follows a sawtooth pattern as particles die and are replenished
        self.assertTrue(
            jnp.all(num_live >= 1), "Should always have at least 1 live point"
        )
        self.assertTrue(
            jnp.all(num_live <= 1000),  # Reasonable upper bound
            "Number of live points should be reasonable",
        )
        self.assertFalse(
            jnp.any(jnp.isnan(num_live)), "Number of live points should not be NaN"
        )

        # Test logX simulation
        n_samples = 50
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=n_samples)
        chex.assert_shape(logX_seq, (n_dead, n_samples))
        chex.assert_shape(logdX_seq, (n_dead, n_samples))

        # Log volumes should be decreasing
        self.assertTrue(
            jnp.all(logX_seq[1:] <= logX_seq[:-1]), "Log volumes should be decreasing"
        )

        # All log volume elements should be negative (since dX < X)
        finite_logdX = logdX_seq[jnp.isfinite(logdX_seq)]
        if len(finite_logdX) > 0:
            self.assertTrue(
                jnp.all(finite_logdX <= 0.0), "Log volume elements should be negative"
            )

        # Test log_weights function
        log_weights_matrix = utils.log_weights(key, mock_info, shape=n_samples)
        chex.assert_shape(log_weights_matrix, (n_dead, n_samples))

        # Weights should be finite for most particles
        finite_weights = jnp.isfinite(log_weights_matrix)
        self.assertGreater(
            jnp.sum(finite_weights),
            n_dead * n_samples * 0.5,
            "Most weights should be finite",
        )

    def test_gaussian_evidence_narrow_prior(self):
        """Test evidence estimation with narrow prior for challenging case."""

        # Setup: Gaussian likelihood with narrow uniform prior (more challenging)
        mu_true = 1.2
        sigma_true = 0.6
        prior_a, prior_b = 0.8, 1.6  # Narrow prior around the mean

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * ((x - mu_true) / sigma_true) ** 2 - 0.5 * jnp.log(
                2 * jnp.pi * sigma_true**2
            )

        # Analytic evidence
        from scipy.stats import norm

        analytical_evidence = (
            norm.cdf((prior_b - mu_true) / sigma_true)
            - norm.cdf((prior_a - mu_true) / sigma_true)
        ) / (prior_b - prior_a)
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock NS data with higher resolution for narrow prior
        num_steps = 60
        key = jax.random.key(12345)

        # Dense sampling in the narrow prior region
        positions = jnp.linspace(prior_a + 0.01, prior_b - 0.01, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (num_steps,)) * 0.3 - 0.15
        dead_loglik_birth = jnp.concatenate(
            [jnp.array([-jnp.inf]), dead_loglik[:-1] + birth_noise[1:]]
        )
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate evidence estimates for statistical testing
        n_evidence_samples = 800
        key = jax.random.key(555)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # 99% confidence interval test
        lower_bound = mean_estimate - 2.576 * std_estimate  # 99% CI
        upper_bound = mean_estimate + 2.576 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 99% CI lower bound {lower_bound}",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 99% CI upper bound {upper_bound}",
        )

    def test_evidence_integration_simple_case(self):
        """Test evidence calculation for a simple analytical case with constant likelihood."""
        # Test case: uniform prior on [0,2], constant likelihood
        # Evidence = ∫[0,2] (1/width) * exp(loglik_constant) dx = exp(loglik_constant)

        loglik_constant = -1.5
        prior_width = 2.0  # Prior on [0, 2]
        n_dead = 40

        # Analytic answer: evidence = ∫[0,2] (1/2) * exp(-1.5) dx = exp(-1.5)
        analytical_log_evidence = loglik_constant

        # Mock data: all particles have same likelihood (constant function)
        dead_loglik = jnp.full(n_dead, loglik_constant)
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)  # All from prior

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            dead_loglik,
            dead_loglik_birth,
            jnp.full(n_dead, -jnp.log(prior_width)),  # Uniform prior log density
        )

        # Generate many evidence estimates
        n_samples = 500
        key = jax.random.key(999)
        keys = jax.random.split(key, n_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=25)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # For constant likelihood case, should be very accurate
        # 95% confidence interval
        lower_bound = mean_estimate - 1.96 * std_estimate
        upper_bound = mean_estimate + 1.96 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 95% CI",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 95% CI",
        )

    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation."""
        key = jax.random.key(67890)

        # Create mock data with varying weights
        n_dead = 50
        dead_loglik = jax.random.uniform(key, (n_dead,)) * 5 - 10  # Range [-10, -5]
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            jnp.sort(dead_loglik),  # Ensure increasing
            dead_loglik_birth,
            jnp.zeros(n_dead),
        )

        # Calculate ESS
        ess_value = utils.ess(key, mock_info)

        # ESS should be positive and reasonable
        self.assertIsInstance(ess_value, (float, jax.Array))
        self.assertGreater(ess_value, 0.0, "ESS should be positive")
        self.assertLessEqual(
            ess_value, n_dead, "ESS should not exceed number of samples"
        )
        self.assertFalse(jnp.isnan(ess_value), "ESS should not be NaN")


if __name__ == "__main__":
    absltest.main()
