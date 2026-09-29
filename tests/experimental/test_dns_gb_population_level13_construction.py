"""Stage-5B single-level controls; synthetic arrays only, no LISA evaluation."""
import inspect
import numpy as np
import pytest
from examples.lisa_dns_stage4 import population_level13_construction as x
from examples.lisa_dns_stage4 import level12_population_validation as w
from examples.lisa_dns_stage4 import analyze_population_level13 as analysis
from examples.lisa_dns_stage4 import analyze_level12_population as wa
from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint


def fixture():
    # Exact Stage-4Z prefix, independent of external files during pytest.
    t=np.array([-np.inf, -113725.65889538352, -113448.99632193986, -113172.84824147604, -112838.36560726895, -112388.10901787043, -111915.6904902109, -111344.96574156903, -110809.82135735315, -110252.99476697217, -109780.28875081123, -108868.60077896297, -108187.99712765554])
    m=np.array([0.0, -1.0343179379627125, -1.9996430044384734, -3.118477930469643, -4.172213954289457, -5.0146408844420645, -5.840204008542667, -6.561075737117467, -7.156607159127204, -7.67544001580533, -8.394305725653265, -9.643899891984567, -10.418392711998465])
    raw={n:dict(position=np.arange(8*384*54,dtype=float).reshape(8,384,54)+b*1e7,
        logprior=np.full((8,384),-100.-b),loglikelihood=np.full((8,384),x.ELL12+100.))
        for b,n in enumerate(x.NAMES)}
    return t,m,raw


def row(ess=20.,selection_status='ok'):
    return dict(level=13,accepted=ess>=20 and selection_status=='ok',threshold=x.ELL12+50.,
        selection={'status':selection_status},calibration={'ratio':.5,'ess':ess},
        estimated_log_mass=x.LOGX12+np.log(.5))


def test_exact_frozen_level0_through12_prefix():
    t,m,_=fixture();x.verify_prefix(t,m,t,m)
    assert (t[-1],m[-1])==(-108187.99712765554,-10.418392711998465)
    for key in ['threshold','mass']:
        for i in range(13):
            bt=t.copy();bm=m.copy();bad=bt if key=='threshold' else bm
            bad[i]=np.nextafter(bad[i],np.inf)
            with pytest.raises(ValueError,match='prefix changed'):x.verify_prefix(bt,bm,t,m)
    with pytest.raises(ValueError):x.verify_prefix(t,m,t[:-1],m[:-1])


def test_deterministic_own_bank_last_survivor_and_independent_cache_storage():
    _,_,raw=fixture()
    raw['selection']['loglikelihood'][6,200]=x.ELL12+1000
    raw['selection']['loglikelihood'][6,300:]=x.ELL12
    raw['calibration']['loglikelihood'][6,377:]=x.ELL12
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
    _,_,raw=fixture();raw['selection']['loglikelihood'][0,128:]=x.ELL12
    with pytest.raises(ValueError,match='STOP'):x.recover_banks(raw)
    _,_,raw=fixture();raw['calibration']=raw['selection']
    with pytest.raises(ValueError,match='separate'):x.recover_banks(raw)


def test_independent_random_streams():
    seeds=sum(x.SEEDS.values(),[])+sum(x.EXCHANGE_SEEDS.values(),[])
    assert len(seeds)==len(set(seeds))==20


def test_strict_ell12_exchange_and_unchanged_population_body():
    assert x.joint_decision is w.joint_decision
    lp=np.zeros((2,9))
    assert not x.joint_decision([0,0],[0,0],lp,lp,[0,0],[x.ELL12+1,x.ELL12],.1)['accepted']
    expected=inspect.getsource(w.run).replace('def run():',
        'def run_bank(output, slice_seeds):\n    OUT = Path(output)\n    SLICE_SEEDS = slice_seeds')
    expected=expected.replace("design['sweeps']==512","design['sweeps']==384").replace('range(512)','range(384)').replace('/512;','/384;')
    assert inspect.getsource(x.run_bank)==expected
    for name in ['round_robin','log_probabilities','slice_blocks','commit_pair','exchange_tags','fixed_exchange']:
        assert getattr(x,name) is getattr(w,name)


def test_candidate_selection_only_and_exact_retained_budget():
    assert list(inspect.signature(x.select_candidate).parameters)==['selection']
    _,_,raw=fixture();raw['selection']['loglikelihood']=np.linspace(x.ELL12+1,x.ELL12+100,3072).reshape(8,384)
    bank=x.retained_bank(raw['selection']);assert bank['position'].shape==(8,256,54)
    candidate=x.select_candidate(bank)
    assert candidate['threshold']==np.sort(bank['loglikelihood'].ravel())[int(np.floor(2048*(1-np.exp(-1))))]
    raw['calibration']['loglikelihood'][:]=1e6
    assert x.select_candidate(bank)==candidate
    source=inspect.getsource(x.main)
    assert source.index("if name=='selection'")<source.index('row=assess')
    assert 'ell_13_diag' not in source


@pytest.mark.parametrize('ess,count',[(19.999,13),(20.,14)])
def test_unchanged_ess20_boundary(ess,count):
    t,m,_=fixture();nt,nm=x.decision_ladder(t,m,row(ess))
    assert len(nt)==len(nm)==count
    x.verify_prefix(nt,nm,t,m)


def test_failed_gate_freezes_nothing(tmp_path):
    t,m,raw=fixture();traces={n:x.retained_bank(a) for n,a in raw.items()}
    nt,nm,path=x.freeze_level13(tmp_path,t,m,row(19.),traces,{'diagnostics':[]},{})
    assert path is None and not list(tmp_path.iterdir())
    np.testing.assert_array_equal(nt,t);np.testing.assert_array_equal(nm,m)
    nt,nm=x.decision_ladder(t,m,row(100.,'insufficient_tail_information'))
    assert len(nt)==13


def test_success_freezes_only_level13_and_no_level14_path(tmp_path):
    t,m,raw=fixture();traces={n:x.retained_bank(a) for n,a in raw.items()}
    nt,nm,path=x.freeze_level13(tmp_path,t,m,row(),traces,{'diagnostics':[]},{})
    rt,rm,_,_=load_checkpoint(path,(t,m))
    assert len(rt)==14 and rt[-1]==x.ELL12+50 and rm[-1]==x.LOGX12+np.log(.5)
    assert [p.name for p in tmp_path.iterdir()]==['checkpoint_level13.json']
    with pytest.raises(ValueError,match='level 13 only'):x.freeze_level13(tmp_path,nt,nm,row(),traces,{'diagnostics':[]},{})
    wrong=row();wrong['level']=14
    with pytest.raises(ValueError,match='level 13 only'):x.decision_ladder(t,m,wrong)
    assert not (tmp_path/'checkpoint_level14.json').exists()


def test_mass_must_come_from_independent_calibration():
    t,m,_=fixture();bad=row();bad['estimated_log_mass']=x.LOGX12-1
    with pytest.raises(ValueError,match='independent calibration'):x.decision_ladder(t,m,bad)


def test_reuses_exact_ell12_provenance_replay():
    # Full accepted/rejected/tamper fixtures remain in Stage-4W tests.
    assert analysis.replay is wa.replay
    assert analysis.communication is wa.communication
    source=inspect.getsource(x.main)
    assert source.index('audit_bank(OUT/name)')<source.index('nt,nm,path=freeze_level13')


@pytest.mark.parametrize('tamper',[False,True])
def test_candidate_is_persisted_before_calibration_and_hash_unchanged(tmp_path,monkeypatch,tamper):
    import json
    t,m,raw=fixture();starts,recovery=x.recover_banks(raw)
    out=tmp_path/'run';out.mkdir();source=tmp_path/'source';source.mkdir()
    for folder in [out,source]:(folder/'checkpoint_level12.json').write_text('unchanged checkpoint sentinel')
    proof=dict(file_sha256={},frozen_thresholds=t,frozen_log_masses=m)
    monkeypatch.setattr(x,'OUT',out);monkeypatch.setattr(x,'Z',source)
    monkeypatch.setattr(x,'preflight',lambda:(t,m,starts,{'provenance':{},'diagnostics':[]},proof))
    seen=[]
    def sample(folder,seeds):
        name=folder.name
        if name=='calibration':
            candidate=json.loads((out/'selection_candidate.json').read_text())
            assert candidate==x.serial(x.select_candidate(x.retained_bank(raw['selection'])))
            assert seen==['selection']
            if tamper:
                (out/'selection_candidate.json').write_text('tampered during calibration')
        folder.mkdir();np.savez(folder/'trace.npz',**raw[name]);seen.append(name)
    monkeypatch.setattr(x,'run_bank',sample)
    monkeypatch.setattr(analysis,'audit_bank',lambda folder:{'synthetic_replay':True})
    if tamper:
        with pytest.raises(AssertionError):x.main()
        assert json.loads((out/'report.json').read_text())['status']=='STOP_failure'
        assert not (out/'checkpoint_level13.json').exists()
        return
    x.main()
    assert seen==['selection','calibration']
    assert not (out/'checkpoint_level13.json').exists()
