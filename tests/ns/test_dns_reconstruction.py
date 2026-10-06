"""Exact finite-measure tests and reconstruction-integrity failure cases."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from blackjax.ns import dns_reconstruction as r


def fixture():
    # Prior is uniform on {-1,+1}, likelihood {1,3}, X1=1/2.
    # Exact deterministic-mixture allocation: two prior draws, one level-1 draw.
    return r.ReconstructionRecords(np.array([-np.inf,np.log(1.5)]),np.log([1.,.5]),
        np.array([[-1.],[1.],[1.]]),np.full(3,-np.log(2)),np.log([1.,3.,3.]),
        np.array([0,0,1]),np.array([0,1,0]),np.array([0,0,0]),np.array(['a','b','c']),
        dict(sampling_target=r.TARGET,mass_status=['prior','calibrated'],
             levels=[dict(level=0,threshold='-inf',stream_id='s0'),dict(level=1,threshold=np.log(1.5),stream_id='s1')]))


def test_exact_discrete_evidence_posterior_overlap_and_shell():
    data=fixture(); result=r.reconstruct_evidence(data)
    assert result['logZ']==pytest.approx(np.log(2),abs=1e-14)
    np.testing.assert_allclose(result['weights'],[.25,.375,.375],rtol=0,atol=1e-14)
    assert r.posterior_expectation(data.position[:,0],result['weights'])==pytest.approx(.5)
    assert r.shell_diagnostic(data)['logZ']==pytest.approx(np.log(2))
    assert result['raw_samples']==3 and result['per_level_counts']==[2,1]
    # Equal coordinates in separate draws are legal; same retained event is not.
    assert data.position[1].tobytes()==data.position[2].tobytes()


def test_level_zero_reduces_to_prior_monte_carlo():
    data=fixture().through(0); estimate=r.reconstruct_evidence(data)
    assert estimate['logZ']==pytest.approx(np.log(np.exp(data.loglikelihood).mean()))
    np.testing.assert_allclose(estimate['weights'],[.25,.75])


@pytest.mark.parametrize('masses',[[0.,.1],[0.,0.],[-.1,-1.],[0.,np.nan]])
def test_invalid_X_and_negative_shell_mass(masses):
    with pytest.raises(ValueError,match='X_0|shell|mass'):
        r.reconstruct_evidence(replace(fixture(),log_masses=np.array(masses)))


@pytest.mark.parametrize('field,value',[
    ('thresholds',np.array([-np.inf,np.nan])),
    ('log_masses',np.array([0.])),
    ('loglikelihood',np.array([0.,np.inf,1.])),
    ('loglikelihood',np.array([0.,-np.inf,1.])),
    ('origin_level',np.array([0,0,2])),
    ('origin_level',np.array([0.,0.,1.])),
    ('origin_level',np.array([1,0,1])),
    ('event_id',np.array(['a','b','b']))])
def test_corrupted_records_fail(field,value):
    with pytest.raises(ValueError):r.reconstruct_evidence(replace(fixture(),**{field:value}))


def test_missing_metadata_uncalibrated_candidate_and_wrong_measure():
    data=fixture()
    for update in [dict(levels=[]),dict(mass_status=['prior','candidate']),dict(sampling_target='selection_construction')]:
        with pytest.raises(ValueError):r.reconstruct_evidence(replace(data,metadata={**data.metadata,**update}))


def test_double_counting_relabelled_event_is_rejected():
    data=fixture()
    fields={name:np.concatenate([getattr(data,name),getattr(data,name)[:1]]) for name in
            ['position','logprior','loglikelihood','origin_level','walker','draw','event_id']}
    fields['event_id'][-1]='z'
    with pytest.raises(ValueError,match='Duplicate'):
        r.reconstruct_evidence(replace(data,**fields))


@pytest.mark.parametrize('weights',[[.2,.2,.2],[-.1,.5,.6],[np.nan,.5,.5],[np.inf,0.,0.]])
def test_non_normalized_or_invalid_weights_rejected(weights):
    with pytest.raises(ValueError,match='normalize'):
        r.posterior_expectation(np.arange(3),weights)


def test_stable_weights_quantiles_and_cdf_ties():
    w=r.posterior_weights(np.array([1000.,1000.,999.]))
    assert np.isfinite(w).all() and abs(w.sum()-1)<1e-12
    np.testing.assert_array_equal(r.posterior_quantiles([0.,0.,1.],[.2,.3,.5],[.05,.5,.95]),[0,0,1])
    d=r.weighted_cdf_discrepancy([0.,0.,1.],[.2,.3,.5],lambda x:(x+1)/3)
    assert d==pytest.approx(1/3)


def test_archive_roundtrip_and_checksum(tmp_path):
    data=fixture(); r.save_records(tmp_path/'record',data); restored=r.load_records(tmp_path/'record')
    np.testing.assert_array_equal(r.reconstruct_evidence(restored)['weights'],r.reconstruct_evidence(data)['weights'])
    p=tmp_path/'record/samples.npz';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):r.load_records(tmp_path/'record')


def test_cache_callback_detects_corruption():
    from blackjax.ns.dns_automatic import NumericalContract
    data=fixture()
    evaluate=lambda x:(np.full(len(x),-np.log(2)),np.where(x[:,0]<0,0.,np.log(3)))
    assert r.validate_caches(data,evaluate,NumericalContract())['passed']
    with pytest.raises(ValueError,match='Likelihood cache'):
        r.validate_caches(replace(data,loglikelihood=data.loglikelihood+1e-4),evaluate,NumericalContract())


def test_mass_sensitivity_does_not_mutate_records():
    data=fixture(); old=data.log_masses.copy()
    r.reconstruct_evidence(data,log_masses=np.log([1.,.4]))
    np.testing.assert_array_equal(data.log_masses,old)
