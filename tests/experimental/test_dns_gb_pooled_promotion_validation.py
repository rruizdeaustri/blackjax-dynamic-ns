"""Cheap synthetic final-promotion checks; never evaluates a LISA model."""
import ast
import inspect
from pathlib import Path
import numpy as np
import pytest
from examples.lisa_dns_stage4 import pooled_promotion_validation as w
from examples.lisa_dns_stage4 import analyze_pooled_promotion as a


def histories():
    result={}
    for b,name in enumerate(w.NAMES):
        x=np.arange(8*256*54,dtype=float).reshape(8,256,54)+b*1e6
        lp=np.arange(2048,dtype=float).reshape(8,256)+b*3000
        ll=np.full((8,256),w.ELL_TEST+10.)
        ll[6]=w.ELL_TEST
        result[name]=dict(position=x,logprior=lp,loglikelihood=ll)
    return result


def test_independent_pools_strict_caches_rng_and_no_same_walker_rule():
    raw=histories();states,proof=w.promote_banks(raw)
    for name in w.NAMES:
        p=proof[name];h=raw[name]
        assert p['eligible']==1792 and p['donors']==7 and 6 not in p['donor_ids']
        s2,p2=w.pool_promote(h,w.PROMOTION_SEEDS[name])
        np.testing.assert_array_equal(p['selected_flat_indices'],p2['selected_flat_indices'])
        for k in states[name]:
            np.testing.assert_array_equal(states[name][k],h[k][p['donor_ids'],p['retained_indices']])
            np.testing.assert_array_equal(s2[k],states[name][k])
            assert not np.shares_memory(states[name][k],h[k])
    assert not np.shares_memory(states['selection']['position'],states['calibration']['position'])
    assert max(states['selection']['position'].ravel())<min(states['calibration']['position'].ravel())


def test_duplicates_allowed_and_no_cross_bank_rescue():
    raw=histories()
    h=raw['selection'];h['loglikelihood'][:]=w.ELL_TEST
    h['loglikelihood'][3,17]=w.ELL_TEST+1
    s,p=w.pool_promote(h,11)
    assert p['duplicates']==7 and p['unique_promoted_states']==1
    np.testing.assert_array_equal(s['position'],np.tile(h['position'][3,17],(8,1)))
    h['loglikelihood'][:]=w.ELL_TEST
    with pytest.raises(ValueError,match='Entire bank empty'):w.promote_banks(raw)


def test_uniform_replacement_exact_generic_sampler():
    src=inspect.getsource(w.pool_promote)
    assert '_survivors(key,particle,ELL_TEST,8)' in src
    assert 'replace=True' in src
    assert 'calibration' not in src


def test_immutable_ladder_and_no_freeze_path():
    t=np.r_[-np.inf,np.linspace(-120000,w.ELL12,12)];m=np.linspace(0,w.LOGX12,13)
    w.verify_ladder(t,m,t.copy(),m.copy())
    with pytest.raises(ValueError):w.verify_ladder(np.r_[t,w.ELL_TEST],np.r_[m,-11],t,m)
    changed=t.copy();changed[3]+=1
    with pytest.raises(ValueError):w.verify_ladder(changed,m,t,m)
    source=inspect.getsource(w)
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
    assert not any(any(x in c for x in ['save_checkpoint','build_next_level','freeze_level','assess']) for c in calls)
    assert 'logX13' not in source and 'log_X_13' not in source
    prep=inspect.getsource(w.prepare)
    assert prep.index("OUT/'preflight.json'")<prep.index('starts,proof=promote_banks')


def test_frozen_transition_source_equivalence():
    path=Path('examples/lisa_dns_stage4/population_level13_construction.py')
    text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_bank')
    expected='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    expected=expected.replace('ELL12','ELL_TEST').replace("design['ell12']==ELL_TEST and design['logX12']==LOGX12","design['ell_test']==ELL_TEST")
    expected=expected.replace('ell12=ELL_TEST,logX12=LOGX12,','ell_test=ELL_TEST,')
    expected=expected.replace('np.testing.assert_allclose(ap,ps,rtol=0,atol=1e-9)','np.testing.assert_array_equal(ap,ps)').replace('np.testing.assert_allclose(al,ls,rtol=0,atol=1e-7)','np.testing.assert_array_equal(al,ls)')
    actual='\n'.join(line for line in inspect.getsource(w.run_bank).splitlines() if "sampling_started" not in line)+'\n'
    assert actual==expected
    from examples.lisa_dns_stage4 import level12_population_validation as frozen
    assert inspect.getsource(w.joint_decision)==inspect.getsource(frozen.joint_decision).replace('ELL12','ELL_TEST')
    from examples.lisa_dns_stage4 import analyze_level12_population as old
    assert inspect.getsource(a.replay)==inspect.getsource(old.replay).replace('ELL12','ELL_TEST')


@pytest.mark.parametrize('ll,accepted',[(w.ELL_TEST,False),(w.ELL_TEST-1,False),(w.ELL_TEST+1,True)])
def test_strict_fixed_contour_joint_decision(ll,accepted):
    p=np.full((2,9),-np.log(9))
    assert w.joint_decision([0,0],[0,0],p,p,[0,0],[ll,ll],0.)['accepted']==accepted


def replay_fixture(accept):
    sx=np.linspace(-.2,.2,8*54).reshape(8,54);sp=np.full(8,-100.);sl=np.full(8,w.ELL_TEST+10)
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
        pl=np.full(2,w.ELL_TEST+10 if accept else w.ELL_TEST)
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
def test_exact_provenance_replay_and_tamper(accept):
    args=replay_fixture(accept)
    r=a.replay(*args)
    assert r['exact_replay_passed'] and len(r['arrivals'])==8*accept
    args[0]['tokens'][0,0,0]+=1
    with pytest.raises(AssertionError):a.replay(*args)
