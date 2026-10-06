"""Cheap exact-measure termination and restart boundary failure checks."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from blackjax.ns import dns_reconstruction as r
from blackjax.ns.dns_termination import (
    TerminationConfig, stopping_decision, remaining_evidence_fraction,
)


def records():
    return r.ReconstructionRecords(
        np.array([-np.inf,0.,np.log(2.)]), np.log([1.,.5,.01]),
        np.array([[0.],[1.],[2.]]),np.zeros(3),np.log([.5,1.5,3.]),
        np.arange(3),np.zeros(3,dtype=int),np.zeros(3,dtype=int),np.array(['a','b','c']),
        dict(sampling_target=r.TARGET,mass_status=['prior','calibrated','calibrated'],
             levels=[dict(level=j,threshold=t,stream_id=str(j)) for j,t in enumerate(['-inf',0.,np.log(2.)])]))


def config(epsilon=.1):
    return TerminationConfig(epsilon,'Finite discrete likelihood has global maximum three')


def test_cap_layer_cake_and_known_early_failure():
    result=stopping_decision(records(),np.log(3.),config())
    assert result['stop']
    assert result['remaining_evidence']==pytest.approx(.03)
    assert np.exp(result['log_evidence_lower'])==pytest.approx(.51)
    assert result['remaining_fraction']==pytest.approx(.03/.51)
    # Large unseen peak: an observed max of 3 cannot justify stopping.
    assert not stopping_decision(records(),np.log(100.),config())['stop']
    assert not stopping_decision(records().through(1),np.log(3.),config())['stop']


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,np.log(2.),-3.])
def test_missing_or_violated_upper_bound_and_negative_likelihood_scale(value):
    with pytest.raises(ValueError,match='upper bound'):
        stopping_decision(records(),value,config())


@pytest.mark.parametrize('masses',[[0.,-.5,np.nan],[0.,-.5,.1],[0.,0.,-1.],[0.,-.5]])
def test_invalid_X(masses):
    with pytest.raises(ValueError):
        stopping_decision(replace(records(),log_masses=np.array(masses)),np.log(3.),config())


def test_missing_deepest_samples():
    data=records()
    with pytest.raises(ValueError,match='requires reconstruction samples'):
        stopping_decision(replace(data,origin_level=np.array([0,1,1]),walker=np.array([0,0,1])),np.log(3.),config())


def test_nonfinite_metric():
    data=replace(records(),thresholds=np.array([-np.inf,-1000.,-999.]),
        metadata={**records().metadata,'levels':[dict(level=j,threshold=t,stream_id=str(j)) for j,t in enumerate(['-inf',-1000.,-999.])]})
    with np.errstate(over='ignore'),pytest.raises(ValueError,match='Nonfinite stopping metric'):
        stopping_decision(data,np.log(3.),config())


@pytest.mark.parametrize('fraction',[np.nan,np.inf,-.1,0.,1.])
def test_invalid_config(fraction):
    with pytest.raises(ValueError):config(fraction)


def test_immutable_configuration_and_required_justification():
    with pytest.raises(FrozenInstanceError):config().remaining_fraction=.2
    with pytest.raises(ValueError,match='justification'):TerminationConfig()


def test_mass_uncertainty_prevents_early_stop_and_encloses_plugin():
    data=records()
    assert stopping_decision(data,np.log(3.),config())['stop']
    assert not stopping_decision(data,np.log(3.),config(),
        log_mass_lower=np.log([1.,.4,.005]),log_mass_upper=np.log([1.,.6,.02]))['stop']
    with pytest.raises(ValueError,match='enclose'):
        stopping_decision(data,np.log(3.),config(),log_mass_upper=np.log([1.,.4,.005]))


def test_checkpoint_before_stop_and_exact_boundary(tmp_path):
    data=records(); before=data.through(1)
    r.save_records(tmp_path/'before',before)
    assert not stopping_decision(r.load_records(tmp_path/'before'),np.log(3.),config())['stop']
    metric=stopping_decision(data,np.log(3.),config())['remaining_fraction']
    boundary=config(metric)
    r.save_records(tmp_path/'boundary',data)
    expected=stopping_decision(data,np.log(3.),boundary)
    assert expected['stop']
    assert stopping_decision(r.load_records(tmp_path/'boundary'),np.log(3.),boundary)==expected
    assert not stopping_decision(data,np.log(3.),config(np.nextafter(metric,0.)))['stop']


def test_level_zero_does_not_stop():
    result=stopping_decision(records().through(0),np.log(3.),config())
    assert not result['stop'] and result['remaining_fraction'] is None


@pytest.mark.parametrize('remainder', [-1., np.nan, np.inf])
def test_negative_or_nonfinite_remainder_estimate(remainder):
    with pytest.raises(ValueError, match='negative remainder|Nonfinite'):
        remaining_evidence_fraction(remainder, 1.)


@pytest.mark.parametrize('denominator', [0., -1., np.nan, np.inf])
def test_invalid_evidence_normalization(denominator):
    with pytest.raises(ValueError, match='evidence lower'):
        remaining_evidence_fraction(.1, denominator)


def test_driver_rejects_records_from_a_different_checkpoint():
    from types import SimpleNamespace
    from examples.dns_automatic_termination import decision
    data = records()
    checkpoint = SimpleNamespace(thresholds=data.thresholds,
                                 log_masses=data.log_masses - .1)
    with pytest.raises(ValueError, match='current frozen ladder'):
        decision(checkpoint, data, None, {})
