"""Cheap extension bookkeeping only; no LISA model or GPU construction."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.run_ladder_extension import iid_starts, promote, assess, uncertainty


def test_initialization_excludes_old_retained_and_obeys_strict_contour():
    bank=dict(positions=np.arange(30*54).reshape(30,54),loglikelihood=np.arange(30),
              accepted_mask=np.ones(30,bool),retained_indices=np.arange(10))
    positions,indices=iid_starts(bank,10,5100)
    assert len(set(indices))==8 and np.all(indices>10)
    np.testing.assert_array_equal(positions,bank['positions'][indices])
    bank['retained_indices']=np.arange(25)
    with pytest.raises(ValueError): iid_starts(bank,10,5100)


def test_promotion_preserves_walker_and_requires_strict_survivors():
    x=np.arange(3*4*2).reshape(3,4,2)
    ll=np.tile(np.arange(4),(3,1))
    starts,indices=promote(x,ll,2,5400)
    assert indices==[3,3,3]
    np.testing.assert_array_equal(starts,x[:,3])
    ll[1]=2
    with pytest.raises(ValueError): promote(x,ll,2,5400)


def test_calibration_failure_does_not_accept_selection_mass():
    rng=np.random.default_rng(1)
    selection=rng.uniform(0,1,(8,256))
    calibration=np.repeat(np.array([.1]*4+[.99]*4)[:,None],256,axis=1)
    result=assess(selection,calibration,-1,-2)
    assert result['selection']['status']=='ok'
    assert result['calibration']['ess']<20 and not result['accepted']
    assert result['estimated_log_mass']==pytest.approx(-2+np.log(.5))
    terms=uncertainty(calibration,result['threshold'])
    assert terms['within_walker_block_se']==0
    assert terms['between_walker_se']>terms['block_means_se']


def test_independent_mass_and_successful_extension():
    rng=np.random.default_rng(2)
    result=assess(rng.uniform(size=(8,256)),rng.uniform(size=(8,256)),-1,-2)
    assert result['accepted']
    assert result['estimated_log_mass']==pytest.approx(-2+np.log(result['calibration']['ratio']))
