"""Cheap rejection bookkeeping/independence checks; no LISA model loading."""
import jax
import numpy as np
import pytest
from examples.lisa_dns_stage4.run_rejection import condition_bank, independent_banks, calibrate_level2


def test_conditioned_samples_strict_threshold_and_bookkeeping():
    points=np.arange(20.).reshape(10,2)
    likes=np.array([-3.,0.,1.,2.,0.,4.,5.,6.,7.,8.])
    indices,info=condition_bank(points,np.zeros(10),likes,0.,4)
    np.testing.assert_array_equal(indices,[2,3,5,6])
    assert np.all(likes[indices]>0)
    assert info['proposed']==10 and info['accepted']==7
    assert info['retained']==4 and info['excess_accepted']==3
    assert info['empirical_prior_acceptance']==.7
    assert info['acceptance_uncertainty']['sample_count']==10
    assert info['acceptance_uncertainty']['exceeds']==7


def test_insufficient_accepts_does_not_retry_or_duplicate():
    points=np.arange(10.).reshape(5,2)
    indices,info=condition_bank(points,np.zeros(5),np.arange(5.),3.,4)
    assert not info['success'] and info['retained']==1 and info['proposed']==5
    np.testing.assert_array_equal(indices,[4])


def test_rejection_streams_are_separate_and_reproducible():
    calls=[]
    def sampler(key,n):
        calls.append(np.asarray(jax.random.key_data(key)))
        return jax.random.normal(key,(n,2))
    banks,meta=independent_banks(sampler,[4424,4425],128)
    assert not np.array_equal(calls[0],calls[1]) and meta['shared_rows']==0
    kept=[]
    for bank in banks:
        idx,info=condition_bank(bank,np.zeros(128),bank[:,0],0.,20)
        assert info['success']
        kept.append(bank[idx])
    assert not {r.tobytes() for r in kept[0]} & {r.tobytes() for r in kept[1]}
    repeated,_=independent_banks(sampler,[4424,4425],128)
    for a,b in zip(banks,repeated): np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError): independent_banks(sampler,[1,1],128)


def test_threshold_never_uses_calibration_bank():
    selection=np.arange(512.)
    first=calibrate_level2(selection,np.linspace(0,511,512),.355)
    second=calibrate_level2(selection,np.linspace(200,711,512),.355)
    assert first['selection']==second['selection']
    assert first['calibration']['ratio']!=second['calibration']['ratio']
    assert first['selection']['threshold']==selection[int(np.floor(512*(1-np.exp(-1))))]
    assert first['X2']==.355*first['calibration']['ratio']
    assert 'ess' not in first['calibration']


def test_nonfinite_rejection_bank_fails():
    with pytest.raises(ValueError):
        condition_bank(np.zeros((2,2)),np.zeros(2),np.array([0.,np.nan]),-1.,1)
