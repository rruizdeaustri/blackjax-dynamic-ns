"""Cheap Stage-4W controls and replay tests; no model imports or LISA trajectory."""
import inspect
from dataclasses import FrozenInstanceError,asdict
import numpy as np
import pytest
from examples.lisa_dns_stage4 import level10_population_validation as w
from examples.lisa_dns_stage4 import frequency_population_benchmark as u
from examples.lisa_dns_stage4 import analyze_level10_population as a


def trace():
    return dict(position=np.arange(8*384*54,dtype=float).reshape(8,384,54),
        logprior=np.full((8,384),-100.),loglikelihood=np.full((8,384),w.ELL10+1))


def test_exact_frozen_checkpoint():
    rt=np.r_[-np.inf,np.arange(1.,10.)];rm=-np.arange(10,dtype=float)
    t=np.r_[rt,w.ELL10];m=np.r_[rm,w.LOGX10]
    w.verify_ladder(t,m,rt,rm)
    for changed,index in [(t,5),(t,10),(m,10)]:
        old=changed[index];changed[index]=np.nextafter(old,np.inf)
        with pytest.raises(ValueError):w.verify_ladder(t,m,rt,rm)
        changed[index]=old
    with pytest.raises(ValueError):w.verify_ladder(np.r_[t,0.],np.r_[m,0.],rt,rm)


def test_last_valid_fallback_not_highest_and_preserves_caches():
    r=trace();r['loglikelihood'][6,200]=w.ELL10+1000
    r['loglikelihood'][6,301:]=w.ELL10
    r['loglikelihood'][2,-1]=w.ELL10-1
    starts,proof=w.recover_starts(r)
    assert proof['fallback_walkers']==[2,6]
    assert proof['absolute_trace_indices']==[383,383,382,383,383,383,300,383]
    for k in starts:np.testing.assert_array_equal(starts[k],r[k][np.arange(8),proof['absolute_trace_indices']])


def test_no_retained_survivor_stops_even_with_valid_burnin():
    r=trace();r['loglikelihood'][6,128:]=w.ELL10
    with pytest.raises(ValueError,match='STOP'):w.recover_starts(r)


@pytest.mark.parametrize('logL,accept',[(w.ELL10,False),(w.ELL10-1,False),(w.ELL10+1,True)])
def test_both_walkers_must_strictly_pass_level10(logL,accept):
    lp=np.zeros((2,9))
    d=w.joint_decision([0.,0.],[0.,0.],lp,lp,(0,0),[w.ELL10+2,logL],.5)
    assert d['accepted']==accept and d['joint_contour']==accept
    if not accept:
        # The old level-9 rule would accept these values: catch wrong contour reuse.
        assert u.joint_decision([0.,0.],[0.,0.],lp,lp,(0,0),[w.ELL10+2,logL],.5)['accepted']


def test_fixed_transition_source_equivalence():
    assert inspect.getsource(w.joint_decision)==inspect.getsource(u.joint_decision).replace('ELL9','ELL10')
    expected=inspect.getsource(u.run).replace('ELL9','ELL10').replace('LOGX9','LOGX10').replace('ell9','ell10').replace('logX9','logX10')
    expected=expected.replace("Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')","(V/'checkpoint_level10.json')")
    assert inspect.getsource(w.run)==expected
    for name in ['round_robin','log_probabilities','fixed_exchange','commit_pair','exchange_tags','slice_blocks']:
        assert getattr(w,name) is getattr(u,name)
    assert w.SWEEPS==512


def test_diagnostic_only_no_level_schema_or_freezing_path():
    values=np.linspace(w.ELL10+1,w.ELL10+100,4096).reshape(8,512)
    diag=a.diagnostic_threshold(values)
    assert diag.ell_11_diag==np.sort(values.ravel())[int(np.floor(4096*(1-np.exp(-1))))]
    assert diag.diagnostic_only and 'threshold' not in asdict(diag) and 'log_mass' not in asdict(diag)
    with pytest.raises(FrozenInstanceError):diag.ell_11_diag=0
    for module in [w,a]:
        source=inspect.getsource(module)
        for forbidden in ['build_next_level(', 'construct_levels(', 'freeze_level10(', 'save_checkpoint(']:
            assert forbidden not in source
    with pytest.raises(ValueError):a.diagnostic_threshold(values[:,:256])
    with pytest.raises(ValueError):a.exceedance_summary(values,diag.ell_11_diag)


def test_walker6_predeclared_iqr_inclusive_and_sweep_numbering():
    initial=np.arange(8,dtype=float);initial[6]=-100
    ll=np.tile(initial[:,None],(1,4));ll[6]=[-100,2,5,100]
    r=a.walker6_recovery(initial,ll)
    # Other seven values are 0,1,2,3,4,5,7: IQR [1.5,4.5].
    assert r['first_entry_sweep']==2 and r['fraction_sweeps_in_range']==.25
    ll[6]=[1.5,4.5,1.49,4.51]
    assert a.walker6_recovery(initial,ll)['fraction_sweeps_in_range']==.5
    ll[6]=-100
    assert a.walker6_recovery(initial,ll)['first_entry_sweep'] is None
    initial[6]=3
    assert a.walker6_recovery(initial,ll)['first_entry_sweep']==0


def replay_fixture(accept):
    sx=np.linspace(-.2,.2,8*54).reshape(8,54);sp=np.full(8,-100.);sl=np.full(8,w.ELL10+10)
    starts=dict(position=sx.copy(),logprior=sp.copy(),loglikelihood=sl.copy())
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy()
    births=[dict(token=i,birth_walker=i//9,birth_sweep=-1,birth_label=i%9,parent=-1,lineage=i) for i in range(72)]
    for walker in range(8):
        token=len(births);births.append(dict(token=token,birth_walker=walker,birth_sweep=0,birth_label=0,parent=int(tokens[walker,0]),lineage=int(lineage[walker,0])))
        tokens[walker,0]=token
    st=tokens.copy();lin=lineage.copy();x=sx.copy();lp=sp.copy();ll=sl.copy();bounds=[.001840,.001842]
    rng=dict(gumbels=np.zeros((1,4,2,9)),uniforms=np.full((1,4),.5));records=[]
    for slot,(aa,bb) in enumerate(w.round_robin()[0]):
        pre=sx[[aa,bb]];fw,f=w.log_probabilities(pre,bounds)
        i,j=[w.draw_label(fw[z],rng['gumbels'][0,slot,z]) for z in range(2)]
        prop=np.array(w.fixed_exchange(pre[0],pre[1],i,j));rv,_=w.log_probabilities(prop,bounds)
        # Force an accepted joint case with U=0 or a contour rejection.
        rng['uniforms'][0,slot]=0.
        pl=np.full(2,w.ELL10+10 if accept else w.ELL10)
        d=w.joint_decision(sp[[aa,bb]],sp[[aa,bb]],fw,rv,[i,j],pl,0.)
        ranks=np.argsort(np.argsort(-fw,axis=-1,kind='stable'),axis=-1,kind='stable')+1
        selected=f[np.arange(2),[i,j]]
        records.append(dict(sweep=0,slot=slot,pair=[aa,bb],labels=[i,j],pre_tokens=tokens[[aa,bb],[i,j]],pre_lineages=lineage[[aa,bb],[i,j]],
            post_slice_position=pre,proposed_position=prop,forward_log_probabilities=fw,reverse_log_probabilities=rv,
            selected_log_probabilities=fw[np.arange(2),[i,j]],selected_ranks=ranks[np.arange(2),[i,j]],selected_frequency=selected,
            inside_count=int(((selected>=.0018407250)&(selected<=.0018412366)).sum()),
            proposed_logprior=sp[[aa,bb]],proposed_loglikelihood=pl,**d))
        if d['accepted']:x[[aa,bb]]=prop;ll[[aa,bb]]=pl
        tokens=w.exchange_tags(tokens,aa,bb,i,j,d['accepted']);lineage=w.exchange_tags(lineage,aa,bb,i,j,d['accepted'])
    raw={k:np.asarray(v)[:,None] for k,v in dict(position=x,logprior=lp,loglikelihood=ll,
        post_slice_position=sx,post_slice_logprior=sp,post_slice_loglikelihood=sl,
        slice_component=np.ones(8,int),slice_labels=np.zeros((8,2),int),
        tokens=tokens,lineage=lineage,post_slice_tokens=st,post_slice_lineage=lin).items()}
    ex={k:np.asarray([r[k] for r in records]) for k in records[0]}
    return raw,ex,starts,rng,births,bounds


@pytest.mark.parametrize('accept',[False,True])
def test_exact_provenance_and_joint_state_replay(accept):
    args=replay_fixture(accept);r=a.replay(*args)
    assert r['exact_replay_passed'] and len(r['arrivals'])==8*accept
    comm=a.communication(r)
    assert comm['distinct_donors']==[int(accept)]*8
    args[0]['position'][0,0,0]+=1
    with pytest.raises(AssertionError):a.replay(*args)
