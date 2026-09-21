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
