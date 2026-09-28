"""Stage-4V controls use synthetic states only; never evaluate LISA."""
import inspect
import numpy as np
import pytest
from examples.lisa_dns_stage4 import population_level10_construction as v
from examples.lisa_dns_stage4 import frequency_population_benchmark as u
from examples.lisa_dns_stage4.run_ladder_batch import check_banks, load_checkpoint


def fixture():
    t=np.r_[-np.inf,np.arange(1.,10.)];m=-np.arange(10,dtype=float)
    traces={n:dict(position=np.arange(8*256*54,dtype=float).reshape(8,256,54)+b*1e7,
        logprior=np.full((8,256),-100.),loglikelihood=np.full((8,256),30.))
        for b,n in enumerate(['selection','calibration'])}
    ix={n:list(range(8)) for n in traces}
    starts={n:{k:a[np.arange(8),ix[n]].copy() for k,a in r.items()} for n,r in traces.items()}
    return t,m,traces,ix,starts


def row(ess=20.):
    return dict(level=10,accepted=ess>=20,threshold=20.,selection={'status':'ok'},
        calibration={'ratio':.5,'ess':ess},estimated_log_mass=-9.+np.log(.5))


def test_exact_bank_recovery_rejects_cache_and_position_corruption():
    _,_,traces,indices,states=fixture()
    for key in ['position','logprior','loglikelihood']:
        recovered=v.recover_starts(states,traces,indices)
        for n in states:assert recovered[n][key].tobytes()==states[n][key].tobytes()
        bad={n:{k:a.copy() for k,a in r.items()} for n,r in states.items()}
        bad['selection'][key].flat[0]+=1
        with pytest.raises(ValueError,match='exact historical'):v.recover_starts(bad,traces,indices)


def test_frozen_prefix():
    t,m,*_=fixture();v.verify_prefix(t,m,t,m)
    bad=m.copy();bad[2]=np.nextafter(bad[2],0)
    with pytest.raises(ValueError):v.verify_prefix(t,bad,t,m)


def test_independent_population_banks_and_streams():
    t,_,_,_,states=fixture();check_banks(states,t[-1])
    seeds=sum(v.SEEDS.values(),[])+sum(v.EXCHANGE_SEEDS.values(),[])
    assert len(seeds)==len(set(seeds))==20
    states['calibration']=states['selection']
    with pytest.raises(ValueError):check_banks(states,t[-1])


def test_population_bookkeeping():
    raw=dict(position=np.zeros((8,384,54)),logprior=np.zeros((8,384)),
        loglikelihood=np.broadcast_to(np.arange(384),(8,384)))
    bank=v.retained_bank(raw)
    assert bank['position'].shape==(8,256,54)
    np.testing.assert_array_equal(bank['loglikelihood'][0],np.arange(128,384))
    raw['position']=raw['position'][:,:383]
    with pytest.raises(ValueError):v.retained_bank(raw)


def test_selection_has_no_calibration_input():
    assert list(inspect.signature(v.select_candidate).parameters)==['selection']
    selected=v.select_candidate({'loglikelihood':np.linspace(u.ELL9+1,u.ELL9+1000,2048).reshape(8,256)})
    assert u.ELL9<selected['threshold']<u.ELL9+1000
    source=inspect.getsource(v.main)
    assert source.index("if name=='selection'")<source.index('row=assess')


@pytest.mark.parametrize('ess,number',[(19.999,10),(20.,11)])
def test_exact_min_ess_gate(ess,number):
    t,m,*_=fixture();nt,nm=v.decision_ladder(t,m,row(ess))
    assert len(nt)==len(nm)==number
    v.verify_prefix(nt,nm,t,m)


def test_failed_gate_freezes_nothing(tmp_path):
    t,m,traces,_,_=fixture()
    _,_,path=v.freeze_level10(tmp_path,t,m,row(19.),traces,{'diagnostics':[]},{})
    assert path is None and not list(tmp_path.iterdir())


def test_success_freezes_one_and_impossible_level11(tmp_path):
    t,m,traces,_,_=fixture()
    nt,nm,path=v.freeze_level10(tmp_path,t,m,row(),traces,{'diagnostics':[]},{})
    rt,rm,_,_=load_checkpoint(path,(t,m))
    np.testing.assert_array_equal(rt,nt);np.testing.assert_array_equal(rm,nm)
    assert len(nt)==11 and [p.name for p in tmp_path.iterdir()]==['checkpoint_level10.json']
    with pytest.raises(ValueError,match='level 10 only'):
        v.freeze_level10(tmp_path,nt,nm,row(),traces,{'diagnostics':[]},{})


def test_transition_body_is_frozen_stage4u():
    expected=inspect.getsource(u.run).replace('def run():',
        'def run_bank(output, slice_seeds):\n    OUT = Path(output)\n    SLICE_SEEDS = slice_seeds')
    expected=expected.replace("design['sweeps']==512","design['sweeps']==384").replace('range(512)','range(384)').replace('/512;','/384;')
    assert inspect.getsource(v.run_bank)==expected
    for name in ['joint_decision','fixed_exchange','log_probabilities','round_robin','commit_pair','exchange_tags']:
        assert getattr(v,name) is getattr(u,name)
