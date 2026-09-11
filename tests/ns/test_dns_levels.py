"""Stage-3 independent calibration, correlation, and frozen-target checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.ns import dns_levels, dns_kernels
from examples.dns_level_construction_validation import build, problem, production, true_mass


def test_quantile_strict_tail_and_ties():
    x = np.arange(256.0).reshape(128, 2)
    result = dns_levels.build_next_level(x, -np.inf, block_size=16, min_ess=1)
    k = int(np.floor(256*(1-np.exp(-1))))
    assert result["threshold"] == k
    assert result["ratio"] == (255-k)/256
    tied = np.repeat(np.arange(16.0), 16).reshape(128, 2)
    result = dns_levels.build_next_level(tied, -np.inf, block_size=16, min_ess=1)
    assert result["ratio"] == np.mean(tied > result["threshold"])
    assert dns_levels.build_next_level(np.ones((128, 2)), 0, block_size=16)["status"] != "ok"
    with pytest.raises(ValueError):
        dns_levels.build_next_level(x, 10, block_size=16)


def test_correlation_diagnostic_rejects_raw_sample_count_argument():
    rng = np.random.default_rng(199)
    independent = rng.normal(size=(4096, 4))
    correlated = np.repeat(rng.normal(size=(64, 4)), 64, axis=0)
    a = dns_levels.tail_diagnostics(independent, 0, 64)
    b = dns_levels.tail_diagnostics(correlated, 0, 64)
    assert a["sample_count"] == b["sample_count"]
    assert b["ess"] < a["ess"]/10
    assert b["standard_error"] > 4*b["naive_iid_se"]
    stuck = np.tile([-1., 1.], (4096, 1))
    assert dns_levels.tail_diagnostics(stuck, 0)["ess"] < 10


def test_real_slow_slice_kernel_stops_for_insufficient_information():
    _, prior, likelihood, _ = problem("gaussian")
    kernel = dns_kernels.build_constrained_slice_kernel(prior, likelihood,
        generate_slice_direction_fn=lambda key, position: jnp.array(0.0001), max_steps=1)
    starts = jnp.linspace(-0.9, 0.9, 8)
    cfg = dns_levels.ConstructionConfig(construction_draws=512, calibration_draws=512,
        burn_in=0, thinning=1, block_size=32, min_ess=100)
    levels, report = dns_levels.construct_levels(16, starts, starts[::-1],
        prior, likelihood, kernel, config=cfg)
    assert len(levels.log_mass) == 1
    assert report["status"] == "insufficient_tail_information"
    assert report["rows"][0]["selection"]["ess"] < 100


def test_independent_draw_order_statistic_oracle():
    # |theta| is uniform: radius at selected threshold has an exact Beta law.
    n, repetitions = 256, 128
    samples = np.asarray(jax.random.uniform(jax.random.key(816), (repetitions, n)))
    k = int(np.floor(n*(1-np.exp(-1))))
    rank = n-k
    true_ratios = []
    for row in samples:
        selected = dns_levels.build_next_level((-row**2/2).reshape(128, 2),
            -np.inf, block_size=16, min_ess=1)
        true_ratios.append(np.sqrt(-2*selected["threshold"]))
    expectation = rank/(n+1)
    variance = rank*(n+1-rank)/((n+1)**2*(n+2))
    assert abs(np.mean(true_ratios)-expectation) < 4*np.sqrt(variance/repetitions)


def check_production(result):
    # Six estimated-standard-error envelopes, not seed-selected tolerances.
    # A small discretization/MC floor protects near-deterministic diagnostics.
    for row in result["levels"]:
        occ = row["occupancy"]
        assert abs(occ["estimate"]-occ["expected"]) < 6*occ["se"] + 0.004
        for cdf in row["cdf"]:
            assert abs(cdf["estimate"]-cdf["q"]) < 6*cdf["se"] + 0.004
        for name, expected in (("cdf_uniform_mean", 0.5), ("cdf_uniform_second", 1/3)):
            stat = row[name]
            assert abs(stat["estimate"]-expected) < 6*stat["se"]+0.004
        for name in ("position_mean", "position_second"):
            stat = row[name]
            assert abs(stat["estimate"]-stat["expected"]) < 6*stat["se"]+1e-6*max(1, stat["expected"])
    for edge in result["edges"]:
        for name in ("eligibility", "up_acceptance", "down_acceptance"):
            stat = edge[name]
            assert abs(stat["estimate"]-stat["expected"]) < 6*stat["se"] + 0.006
    assert min(result["round_trips"]) > 10
    assert result["parameter_failures"] == 0


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_slice_constructed_analytic_ladder_and_frozen_distribution(seed):
    levels, meta = build("gaussian", seed)
    assert meta["status"] == "max_levels"
    assert len(levels.log_mass) == 5
    assert np.all(np.diff(levels.loglikelihood) > 0)
    error_variance = 0
    for row in meta["rows"]:
        select, cal = row["selection"], row["calibration"]
        actual = row["true_compression"]
        assert abs(actual-np.exp(-1)) < 6*select["standard_error"]+0.005
        assert abs(actual-cal["ratio"]) < 6*cal["standard_error"]+0.005
        error_variance += (cal["standard_error"]/cal["ratio"])**2
        error = meta["delta_log_mass"][row["level"]]
        assert abs(error) < 6*np.sqrt(error_variance)+0.01
    # Calibration exceedance fraction is not the forced order-statistic fraction.
    assert any(r["selection"]["ratio"] != r["calibration"]["ratio"] for r in meta["rows"])
    check_production(production("gaussian", seed+1000, levels, sweeps=6000))


def test_refinement_reduces_mass_error_across_independent_runs():
    before, after = [], []
    _, prior, likelihood, kernel = problem("gaussian")
    for seed in (71, 72, 73, 74):
        cfg = dns_levels.ConstructionConfig(calibration_draws=256, block_size=32, min_ess=30)
        levels, _ = build("gaussian", seed, config=cfg)
        starts = jax.random.uniform(jax.random.key(seed+500), (8,), minval=-1., maxval=1.)
        refined, _ = dns_levels.calibrate_level_masses(
            seed+1000, levels, starts, prior, likelihood, kernel)
        truth = np.log(true_mass("gaussian", levels.loglikelihood))
        before.extend((np.asarray(levels.log_mass)[1:]-truth[1:])**2)
        after.extend((np.asarray(refined.log_mass)[1:]-truth[1:])**2)
        np.testing.assert_array_equal(refined.loglikelihood, levels.loglikelihood)
    assert np.mean(after) < np.mean(before)


def test_builder_reproducibility_and_stopping_controls():
    cfg = dns_levels.ConstructionConfig(num_levels=3, construction_draws=128,
        calibration_draws=128, block_size=16, min_ess=10)
    a, am = build("gaussian", 50, walkers=4, config=cfg)
    b, bm = build("gaussian", 50, walkers=4, config=cfg)
    np.testing.assert_array_equal(a.loglikelihood, b.loglikelihood)
    assert am == bm
    cfg = dns_levels.ConstructionConfig(max_parameter_steps=1)
    levels, meta = build("gaussian", 1, config=cfg)
    assert len(levels.log_mass) == 1 and meta["status"] == "max_parameter_steps"
    cfg = dns_levels.ConstructionConfig(terminal_loglikelihood=-0.2)
    levels, meta = build("gaussian", 1, config=cfg)
    assert len(levels.log_mass) == 2 and meta["status"] == "terminal_loglikelihood"
    cfg = dns_levels.ConstructionConfig(min_ess=1e10)
    levels, meta = build("gaussian", 1, config=cfg)
    assert len(levels.log_mass) == 1 and meta["status"] == "insufficient_tail_information"


@pytest.mark.parametrize("kwargs", [dict(num_levels=0), dict(thinning=0), dict(burn_in=-1),
    dict(target_compression=1), dict(min_ess=0), dict(construction_draws=4)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        dns_levels.ConstructionConfig(**kwargs)
