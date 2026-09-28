"""Stage-4Z single-level controls; synthetic arrays only, no LISA evaluation."""
import inspect
import numpy as np
import pytest
from examples.lisa_dns_stage4 import population_level12_construction as x
from examples.lisa_dns_stage4 import level11_population_validation as w
from examples.lisa_dns_stage4 import analyze_population_level12 as analysis
from examples.lisa_dns_stage4 import analyze_level11_population as wa
from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint


def fixture():
    t=np.r_[-np.inf,np.linspace(x.ELL11-1000,x.ELL11,11)]
    m=np.linspace(0,x.LOGX11,12)
    raw={n:dict(position=np.arange(8*384*54,dtype=float).reshape(8,384,54)+b*1e7,
        logprior=np.full((8,384),-100.-b),loglikelihood=np.full((8,384),x.ELL11+100.))
        for b,n in enumerate(x.NAMES)}
    return t,m,raw


def row(ess=20.,selection_status='ok'):
    return dict(level=12,accepted=ess>=20 and selection_status=='ok',threshold=x.ELL11+50.,
        selection={'status':selection_status},calibration={'ratio':.5,'ess':ess},
        estimated_log_mass=x.LOGX11+np.log(.5))


def test_exact_frozen_level0_through11_prefix():
    t,m,_=fixture();x.verify_prefix(t,m,t,m)
    for i in [0,10,11]:
        bad=m.copy();bad[i]=np.nextafter(bad[i],np.inf)
        with pytest.raises(ValueError,match='prefix changed'):x.verify_prefix(t,bad,t,m)
    with pytest.raises(ValueError):x.verify_prefix(t,m,t[:-1],m[:-1])


def test_deterministic_own_bank_last_survivor_and_independent_cache_storage():
    _,_,raw=fixture()
    raw['selection']['loglikelihood'][6,200]=x.ELL11+1000
    raw['selection']['loglikelihood'][6,300:]=x.ELL11
    raw['calibration']['loglikelihood'][6,377:]=x.ELL11
    starts,proof=x.recover_banks(raw)
    assert proof['selection']['absolute_trace_indices'][6]==299
    assert proof['calibration']['absolute_trace_indices'][6]==376
    for n in x.NAMES:
        for k in starts[n]:
            expected=raw[n][k][np.arange(8),proof[n]['absolute_trace_indices']]
            assert expected.tobytes()==starts[n][k].tobytes()
            assert not np.shares_memory(starts[n][k],raw[n][k])
    for k in starts['selection']:assert not np.shares_memory(starts['selection'][k],starts['calibration'][k])


def test_missing_survivor_stops_and_duplicate_bank_rejected():
    _,_,raw=fixture();raw['selection']['loglikelihood'][0,128:]=x.ELL11
    with pytest.raises(ValueError,match='STOP'):x.recover_banks(raw)
    _,_,raw=fixture();raw['calibration']=raw['selection']
    with pytest.raises(ValueError,match='separate'):x.recover_banks(raw)


def test_independent_random_streams():
    seeds=sum(x.SEEDS.values(),[])+sum(x.EXCHANGE_SEEDS.values(),[])
    assert len(seeds)==len(set(seeds))==20


def test_strict_ell11_exchange_and_unchanged_population_body():
    assert x.joint_decision is w.joint_decision
    lp=np.zeros((2,9))
    assert not x.joint_decision([0,0],[0,0],lp,lp,[0,0],[x.ELL11+1,x.ELL11],.1)['accepted']
    expected=inspect.getsource(w.run).replace('def run():',
        'def run_bank(output, slice_seeds):\n    OUT = Path(output)\n    SLICE_SEEDS = slice_seeds')
    expected=expected.replace("design['sweeps']==512","design['sweeps']==384").replace('range(512)','range(384)').replace('/512;','/384;')
    assert inspect.getsource(x.run_bank)==expected
    for name in ['round_robin','log_probabilities','slice_blocks','commit_pair','exchange_tags','fixed_exchange']:
        assert getattr(x,name) is getattr(w,name)


def test_candidate_selection_only_and_exact_retained_budget():
    assert list(inspect.signature(x.select_candidate).parameters)==['selection']
    _,_,raw=fixture();raw['selection']['loglikelihood']=np.linspace(x.ELL11+1,x.ELL11+100,3072).reshape(8,384)
    bank=x.retained_bank(raw['selection']);assert bank['position'].shape==(8,256,54)
    candidate=x.select_candidate(bank)
    assert candidate['threshold']==np.sort(bank['loglikelihood'].ravel())[int(np.floor(2048*(1-np.exp(-1))))]
    raw['calibration']['loglikelihood'][:]=1e6
    assert x.select_candidate(bank)==candidate
    source=inspect.getsource(x.main)
    assert source.index("if name=='selection'")<source.index('row=assess')
    assert 'ell_12_diag' not in source


@pytest.mark.parametrize('ess,count',[(19.999,12),(20.,13)])
def test_unchanged_ess20_boundary(ess,count):
    t,m,_=fixture();nt,nm=x.decision_ladder(t,m,row(ess))
    assert len(nt)==len(nm)==count
    x.verify_prefix(nt,nm,t,m)


def test_failed_gate_freezes_nothing(tmp_path):
    t,m,raw=fixture();traces={n:x.retained_bank(a) for n,a in raw.items()}
    nt,nm,path=x.freeze_level12(tmp_path,t,m,row(19.),traces,{'diagnostics':[]},{})
    assert path is None and not list(tmp_path.iterdir())
    np.testing.assert_array_equal(nt,t);np.testing.assert_array_equal(nm,m)
    nt,nm=x.decision_ladder(t,m,row(100.,'insufficient_tail_information'))
    assert len(nt)==12


def test_success_freezes_only_level12_and_no_level13_path(tmp_path):
    t,m,raw=fixture();traces={n:x.retained_bank(a) for n,a in raw.items()}
    nt,nm,path=x.freeze_level12(tmp_path,t,m,row(),traces,{'diagnostics':[]},{})
    rt,rm,_,_=load_checkpoint(path,(t,m))
    assert len(rt)==13 and rt[-1]==x.ELL11+50 and rm[-1]==x.LOGX11+np.log(.5)
    assert [p.name for p in tmp_path.iterdir()]==['checkpoint_level12.json']
    with pytest.raises(ValueError,match='level 12 only'):x.freeze_level12(tmp_path,nt,nm,row(),traces,{'diagnostics':[]},{})
    wrong=row();wrong['level']=13
    with pytest.raises(ValueError,match='level 12 only'):x.decision_ladder(t,m,wrong)
    assert not (tmp_path/'checkpoint_level13.json').exists()


def test_mass_must_come_from_independent_calibration():
    t,m,_=fixture();bad=row();bad['estimated_log_mass']=x.LOGX11-1
    with pytest.raises(ValueError,match='independent calibration'):x.decision_ladder(t,m,bad)


def test_reuses_exact_ell11_provenance_replay():
    # Full accepted/rejected/tamper fixtures remain in Stage-4W tests.
    assert analysis.replay is wa.replay
    assert analysis.communication is wa.communication
    source=inspect.getsource(x.main)
    assert source.index('audit_bank(OUT/name)')<source.index('nt,nm,path=freeze_level12')


def test_candidate_is_persisted_before_calibration_sampling(tmp_path,monkeypatch):
    import json
    t,m,raw=fixture();starts,recovery=x.recover_banks(raw)
    out=tmp_path/'run';out.mkdir();source=tmp_path/'source';source.mkdir()
    for folder in [out,source]:(folder/'checkpoint_level11.json').write_text('unchanged checkpoint sentinel')
    proof=dict(file_sha256={},frozen_thresholds=t,frozen_log_masses=m)
    monkeypatch.setattr(x,'OUT',out);monkeypatch.setattr(x,'X',source)
    monkeypatch.setattr(x,'preflight',lambda:(t,m,starts,{'provenance':{},'diagnostics':[]},proof))
    seen=[]
    def sample(folder,seeds):
        name=folder.name
        if name=='calibration':
            candidate=json.loads((out/'selection_candidate.json').read_text())
            assert candidate==x.serial(x.select_candidate(x.retained_bank(raw['selection'])))
            assert seen==['selection']
        folder.mkdir();np.savez(folder/'trace.npz',**raw[name]);seen.append(name)
    monkeypatch.setattr(x,'run_bank',sample)
    monkeypatch.setattr(analysis,'audit_bank',lambda folder:{'synthetic_replay':True})
    x.main()
    assert seen==['selection','calibration']
    assert not (out/'checkpoint_level12.json').exists()
