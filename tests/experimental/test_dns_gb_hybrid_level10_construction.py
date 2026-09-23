"""Cheap one-level construction control tests; no LISA likelihood calls."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.hybrid_level10_construction import (
    verify_prefix,recover_starts,retained_bank,decision_ladder,freeze_level10,SEEDS)
from examples.lisa_dns_stage4.run_ladder_batch import check_banks,load_checkpoint


def fixture():
    thresholds=np.r_[-np.inf,np.arange(1.,10.)];masses=-np.arange(10,dtype=float)
    traces={}
    for b,name in enumerate(['selection','calibration']):
        traces[name]=dict(position=np.arange(8*256*54,dtype=float).reshape(8,256,54)+b*1e7,
            logprior=np.full((8,256),-100.),loglikelihood=np.full((8,256),30.))
    indices={name:list(range(8)) for name in traces}
    states={name:{k:v[np.arange(8),indices[name]].copy() for k,v in a.items()} for name,a in traces.items()}
    return thresholds,masses,traces,indices,states


def row(ess=20.,selection_status='ok'):
    accepted=ess>=20 and selection_status=='ok'
    return dict(level=10,accepted=accepted,threshold=20.,selection={'status':selection_status},
        calibration={'ratio':.5,'ess':ess},estimated_log_mass=-9.+np.log(.5))


def test_exact_frozen_prefix_and_rejection_of_tampering():
    t,m,*_=fixture();verify_prefix(t,m,t,m)
    bad=m.copy();bad[5]=np.nextafter(bad[5],np.inf)
    with pytest.raises(ValueError,match='prefix changed'):verify_prefix(t,bad,t,m)
    with pytest.raises(ValueError,match='level-0..9'):verify_prefix(t,m,t[:-1],m[:-1])


def test_exact_recovery_of_both_historical_banks_and_caches():
    _,_,traces,indices,states=fixture();recovered=recover_starts(states,traces,indices)
    for name in states:
        for k in states[name]:assert recovered[name][k].tobytes()==states[name][k].tobytes()
    states['calibration']['logprior'][3]=np.nextafter(-100.,0.)
    with pytest.raises(ValueError,match='exact historical'):recover_starts(states,traces,indices)


def test_independent_banks_and_streams():
    t,_,_,_,states=fixture();check_banks(states,t[-1])
    assert len(set(SEEDS['selection']+SEEDS['calibration']))==16
    assert not (set(SEEDS['selection']) & set(SEEDS['calibration']))
    states['calibration']=states['selection']
    with pytest.raises(ValueError,match='separate'):check_banks(states,t[-1])


def test_exact_burn_in_and_retained_bookkeeping():
    raw=dict(walker=np.repeat(np.arange(8),384),iteration=np.tile(np.arange(384),8),
        position=np.arange(8*384*54).reshape(-1,54),logprior=np.arange(8*384),loglikelihood=np.arange(8*384))
    retained=retained_bank(raw)
    assert retained['position'].shape==(8,256,54)
    for k in retained:np.testing.assert_array_equal(retained[k],raw[k].reshape((8,384)+raw[k].shape[1:])[:,128:])
    raw['iteration'][0]=1
    with pytest.raises(AssertionError):retained_bank(raw)


@pytest.mark.parametrize('ess,accepted',[(19.999,False),(20.,True)])
def test_unchanged_calibration_gate_boundary(ess,accepted):
    t,m,*_=fixture();newt,newm=decision_ladder(t,m,row(ess))
    assert len(newt)==10+int(accepted)
    assert newt[:10].tobytes()==t.tobytes() and newm[:10].tobytes()==m.tobytes()
    if accepted:assert newm[-1]==m[-1]+np.log(.5)


def test_failed_calibration_creates_no_checkpoint(tmp_path):
    t,m,traces,_,_=fixture()
    nt,nm,path=freeze_level10(tmp_path,t,m,row(19.),traces,{'diagnostics':[]},{})
    assert path is None and not list(tmp_path.iterdir())
    np.testing.assert_array_equal(nt,t);np.testing.assert_array_equal(nm,m)
    # Selection gate is also retained, not bypassed by calibration passing.
    nt,nm=decision_ladder(t,m,row(100.,'insufficient_tail_information'))
    assert len(nt)==10


def test_success_freezes_exactly_one_level_and_no_level11_path(tmp_path):
    t,m,traces,_,_=fixture()
    nt,nm,path=freeze_level10(tmp_path,t,m,row(),traces,{'diagnostics':[]},{})
    assert sorted(p.name for p in tmp_path.iterdir())==['checkpoint_level10.json']
    rt,rm,_,_=load_checkpoint(path,(t,m))
    np.testing.assert_array_equal(rt,nt);np.testing.assert_array_equal(rm,nm)
    assert len(rt)==11 and rt[-1]==20.
    with pytest.raises(ValueError,match='level 10 only'):
        freeze_level10(tmp_path,nt,nm,row(),traces,{'diagnostics':[]},{})
    assert not (tmp_path/'checkpoint_level11.json').exists()
