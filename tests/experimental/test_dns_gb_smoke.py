"""Cheap smoke-report checks; never load or sample the LISA model."""
import json
import numpy as np
import pytest
from examples.lisa_dns_stage4.run_smoke import diagonal_scales, json_safe, level_summary, summarize


def test_diagonal_scales_and_invalid_history():
    samples = np.arange(4.)[:, None] * np.arange(1., 55.)[None, :]
    np.testing.assert_allclose(diagonal_scales(samples), np.std(samples, axis=0, ddof=1))
    for bad in [np.zeros((4, 54)), np.ones((1, 54)), np.ones((4, 53)), samples * np.nan]:
        with pytest.raises(ValueError):
            diagonal_scales(bad)


def test_round_trips_require_both_endpoints():
    result = level_summary([0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2], 3)
    assert result['round_trips'] == 2
    assert sum(result['occupancy']) == 11
    assert np.sum(result['transitions']) == 11
    assert level_summary([0, 1, 0, 1], 3)['round_trips'] == 0
    assert level_summary([2, 1, 2, 1, 2], 3)['round_trips'] == 0


def test_json_nonfinite_threshold_is_explicit():
    result = json_safe({'thresholds': np.array([-np.inf, -91100.]), 'count': np.int64(3)})
    assert json.loads(json.dumps(result, allow_nan=False)) == {'thresholds': ['-inf', -91100.], 'count': 3}


def test_component_accounting():
    records = [dict(component=c, labels=l, slice_accepted=a, cost=cost, loglikelihood=ll)
               for c,l,a,cost,ll in [(0,[-1,-1],True,3,-5.), (1,[4,-1],False,5,-5.), (2,[7,2],True,4,-4.)]]
    result = summarize(records)
    assert result['component_counts'] == {'0':1, '1':1, '2':1}
    assert result['single_label_counts'] == {'4':1}
    assert result['unordered_pair_counts'] == {'2,7':1}
    assert result['slice_failures'] == 1
    assert result['likelihood_evaluation_proxy'] == 12


def test_prior_banks_use_distinct_streams_and_selection_cannot_read_calibration():
    import jax
    from examples.lisa_dns_stage4.prior_bootstrap import draw_prior_banks, select_iid_threshold, calibrate_iid
    calls = []
    def sampler(key, n):
        calls.append(np.asarray(jax.random.key_data(key)))
        return jax.random.normal(key, (n, 3))
    banks, metadata = draw_prior_banks(sampler, 123, 456, 512)
    assert len(calls) == 2 and not np.array_equal(*calls)
    assert metadata['shared_rows'] == 0
    assert metadata['selection_sha256'] != metadata['calibration_sha256']
    repeated, _ = draw_prior_banks(sampler, 123, 456, 512)
    for a,b in zip(banks,repeated):
        np.testing.assert_array_equal(a,b)
    selection_values = banks[0][:,0]
    selected = select_iid_threshold(selection_values)
    threshold = selected['threshold']
    assert threshold == np.sort(selection_values)[int(np.floor(512*(1-np.exp(-1))))]
    low = calibrate_iid(np.full(512, threshold-1), threshold)
    high = calibrate_iid(np.full(512, threshold+1), threshold)
    assert low['ratio'] == 0 and high['ratio'] == 1
    assert not low['accepted'] and not high['accepted']
    assert select_iid_threshold(selection_values) == selected
    assert 'ess' not in high
    with pytest.raises(ValueError):
        draw_prior_banks(sampler, 123, 123, 512)


def test_iid_strict_ties_and_binomial_interval():
    from examples.lisa_dns_stage4.prior_bootstrap import select_iid_threshold, calibrate_iid
    selection = select_iid_threshold(np.array([0., 1., 1., 1., 2.]), .5)
    assert selection['threshold'] == 1 and selection['ratio'] == .2
    result = calibrate_iid(np.r_[np.zeros(256), np.ones(256)], .5)
    assert result['accepted'] and result['ratio'] == .5
    np.testing.assert_allclose(result['standard_error'], np.sqrt(.25/512))
    assert result['interval'][0] < .5 < result['interval'][1]
    assert result['log_ratio'] == np.log(.5)


def test_survivors_remain_separate_and_have_no_duplicates():
    import jax
    from examples.lisa_dns_stage4.prior_bootstrap import survivor_indices
    values = np.arange(100.)
    indices = survivor_indices(values, 50., jax.random.key(10), walkers=8)
    assert len(np.unique(indices)) == 8
    assert np.all(values[indices] > 50)
    with pytest.raises(ValueError):
        survivor_indices(values, 95., jax.random.key(10), walkers=8)


def test_between_walker_disagreement_is_visible():
    from examples.lisa_dns_stage4.prior_bootstrap import correlation_breakdown
    values = np.tile([-1., -1., 1., 1.], (256, 1))
    result = correlation_breakdown(values, 0.)
    assert result['dominant_term'] == 'between_walker'
    assert result['standard_error_terms']['between_walker'] > result['standard_error_terms']['iid']
