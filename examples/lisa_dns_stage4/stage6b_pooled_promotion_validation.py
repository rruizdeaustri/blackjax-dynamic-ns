"""Stage 6B: one CPU/x64 fixed-contour validation; persistent no-retry lock."""
import os
import sys
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    serial,load,digest,round_robin,log_probabilities,draw_label,commit_pair,
    exchange_tags,slice_blocks,fixed_exchange,INTERVAL,fixed_frequency_region,
)

from examples.lisa_dns_stage4.numerical_cache_contract import CONTRACT, validate_cache

OUT=Path('/tmp/lisa_dns_stage6b_pooled_promotion_validation')
B=Path('/tmp/lisa_dns_stage5b_population_level13')
Z=Path('/tmp/lisa_dns_stage4z_population_level12')
A=Path('/tmp/lisa_dns_stage5a_level12_validation')
ELL_TEST=-107123.66036615883
ELL12=-108187.99712765554
LOGX12=-10.418392711998465
NAMES=('selection','calibration')
SEEDS=dict(selection=list(range(970100,970108)),calibration=list(range(970200,970208)))
PROMOTION_SEEDS=dict(selection=2026093011,calibration=2026093012)
EXCHANGE_SEEDS=dict(selection=[2026093013,2026093014],calibration=[2026093015,2026093016])


def verify_ladder(t,m,rt,rm):
    arrays=[np.asarray(a,dtype=np.float64) for a in (t,m,rt,rm)]
    if any(a.shape!=(13,) for a in arrays):
        raise ValueError('Only immutable levels 0..12 permitted')
    t,m,rt,rm=arrays
    if t.tobytes()!=rt.tobytes() or m.tobytes()!=rm.tobytes():
        raise ValueError('Frozen prefix changed')
    if (t[-1],m[-1])!=(ELL12,LOGX12):
        raise ValueError('Wrong frozen endpoint')


def pool_promote(history, seed):
    """Canonical time-major Stage-3 pool; copy full states, duplicates allowed."""
    require_cpu_x64()
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    from blackjax.ns.dns_levels import _survivors
    pool=eligible_indices(history)
    if not len(pool):raise ValueError('Entire bank empty; no other-bank rescue')
    particle=DNSParticleState(*[jnp.asarray(history[k].swapaxes(0,1)) for k in ['position','logprior','loglikelihood']])
    key=jax.random.key(seed)
    index=np.asarray(jax.random.choice(key,jnp.asarray(pool),shape=(8,),replace=True))
    donor=index%8;retained=index//8
    promoted=_survivors(key,particle,ELL_TEST,8)
    starts={k:np.array(v,copy=True) for k,v in zip(['position','logprior','loglikelihood'],promoted)}
    for k,v in starts.items():
        assert v.tobytes()==history[k][donor,retained].tobytes()
        assert not np.shares_memory(v,history[k])
    assert np.all(starts['loglikelihood']>ELL_TEST)
    hashes=[hashlib.sha256(b''.join(np.asarray(starts[k][w]).tobytes() for k in ['position','logprior','loglikelihood'])).hexdigest() for w in range(8)]
    unique=len({row.tobytes() for row in starts['position']})
    return starts,dict(total_retained=2048,eligible=len(pool),eligible_fraction=len(pool)/2048,
        donors=int(len(np.unique(pool%8))),per_donor_counts=np.bincount(pool%8,minlength=8),
        pool_order='time-major, flat=retained_index*8+donor',selected_flat_indices=index,
        retained_indices=retained,absolute_trace_indices=retained+128,donor_ids=donor,
        selected_eligible_pool_indices=np.searchsorted(pool,index),
        duplicate_source_hashes={h:hashes.count(h) for h in set(hashes) if hashes.count(h)>1},
        source_state_sha256=hashes,donor_multiplicities=np.bincount(donor,minlength=8),
        unique_promoted_states=unique,duplicates=8-unique,initial_logL=starts['loglikelihood'])


def promote_banks(histories):
    if set(histories)!=set(NAMES):raise ValueError('Two independent named banks required')
    starts={};proof={}
    for name in NAMES:starts[name],proof[name]=pool_promote(histories[name],PROMOTION_SEEDS[name])
    for k in starts['selection']:
        assert not np.shares_memory(starts['selection'][k],starts['calibration'][k])
    # Independence is bank provenance and separate streams/storage; duplicates within a bank are valid.
    return starts,proof


def prepare():
    assert OUT.is_dir() and (OUT/'run_started').is_file()
    require_cpu_x64()
    checkpoint=json.loads((Z/'checkpoint_level12.json').read_text())
    payload=checkpoint['payload']
    assert hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()==checkpoint['sha256']
    t=np.array([float(v) for v in payload['thresholds']]);m=np.array(payload['log_masses'])
    levels=load(Z/'levels.npz');verify_ladder(t,m,levels['thresholds'],levels['log_mass'])
    assert (B/'checkpoint_level12.json').read_bytes()==(Z/'checkpoint_level12.json').read_bytes()
    candidate=json.loads((B/'selection_candidate.json').read_text())
    assert candidate['threshold']==ELL_TEST
    validation=json.loads((B/'validation.json').read_text())
    assert not validation['freeze_completed'] and validation['exact_exchange_and_provenance_replay']
    assert not any(B.rglob('checkpoint_level13*'))
    sources=[Path(__file__),Path('examples/lisa_dns_stage4/frequency_population_benchmark.py'),
        Path('examples/lisa_dns_stage4/level12_population_validation.py'),
        Path('examples/lisa_dns_stage4/proposals.py'),Path('examples/lisa_dns_stage4/audit.py'),
        Path('examples/lisa_dns_stage4/analyze_frequency_exchange_design.py'),
        Path('blackjax/ns/dns_kernels.py'),Path('blackjax/ns/dns_levels.py'),
        Z/'checkpoint_level12.json',Z/'levels.npz',A/'comparison.json',A/'report.json']
    sources+=sorted(p for p in B.rglob('*') if p.is_file())
    stage6a=json.loads(Path('/tmp/lisa_dns_stage6a_numerical_audit/results.json').read_text())
    assert stage6a['decision']=='PASS' and stage6a['contract']==CONTRACT
    for path,h in stage6a['source_sha256'].items():assert digest(path)==h,path
    sources += [Path('examples/lisa_dns_stage4/numerical_cache_contract.py'),
                Path('examples/lisa_dns_stage4/analyze_stage6b_pooled_promotion.py'),
                Path('/tmp/lisa_dns_stage6a_numerical_audit/results.json')]
    protected={**stage6a['source_sha256'],**{str(p):digest(p) for p in sources}}
    allseeds=sum(SEEDS.values(),[])+list(PROMOTION_SEEDS.values())+sum(EXCHANGE_SEEDS.values(),[])
    assert len(set(allseeds))==22
    design=json.loads((A/'design.json').read_text())
    assert (design['f_star'],design['sigma_f'])==fixed_frequency_region()
    np.testing.assert_array_equal(design['round_robin'],round_robin())
    # Persist all seeds/settings/hashes BEFORE ANY random promotion or model call.
    preflight=dict(purpose='one fixed-contour promotion validation, no ladder construction',ell_test=ELL_TEST,
        frozen_thresholds=t,frozen_log_masses=m,promotion_seeds=PROMOTION_SEEDS,slice_seeds=SEEDS,
        exchange_seeds=EXCHANGE_SEEDS,walkers=8,burn_in=128,retained=256,sweeps=384,
        file_sha256=protected,no_retry=True,level_construction=False,contract=CONTRACT,
        decision_policy='All integrity checks and full budgets required; assess persistent cross-walker separation in both halves and qualitative cross-bank compatibility; no tuning or repeat.')
    (OUT/'preflight.json').write_text(json.dumps(serial(preflight),indent=2,allow_nan=False))
    histories={name:{k:v[:,128:].copy() for k,v in load(B/name/'trace.npz').items()
                    if k in ['position','logprior','loglikelihood']} for name in NAMES}
    pools={name:pool_summary(histories[name]) for name in NAMES}
    (OUT/'eligible_pools.json').write_text(json.dumps(serial(pools),indent=2))
    print('ELIGIBLE POOLS BEFORE PROMOTION',serial(pools),flush=True)
    assert pools['selection']['per_donor_counts'][6]==0
    starts,proof=promote_banks(histories)
    assert proof['selection']['per_donor_counts'][6]==0
    (OUT/'promotion.json').write_text(json.dumps(serial(proof),indent=2,allow_nan=False))
    for name in NAMES:
        folder=OUT/name;folder.mkdir()
        np.savez_compressed(folder/'initial_states.npz',**starts[name])
        ls,ms=EXCHANGE_SEEDS[name]
        np.savez_compressed(folder/'random_schedule.npz',
            gumbels=np.random.Generator(np.random.PCG64(ls)).gumbel(size=(384,4,2,9)),
            uniforms=np.random.Generator(np.random.PCG64(ms)).random((384,4)))
        d=dict(bank=name,ell_test=ELL_TEST,sweeps=384,burn_in=128,retained=256,
            f_star=design['f_star'],sigma_f=design['sigma_f'],round_robin=round_robin(),
            slice=design['slice'],slice_seeds=SEEDS[name],label_seed=ls,mh_seed=ms,
            promotion=proof[name],file_sha256={**protected,**{str(p):digest(p) for p in
                [OUT/'preflight.json',OUT/'promotion.json',folder/'initial_states.npz',folder/'random_schedule.npz']}})
        (folder/'design.json').write_text(json.dumps(serial(d),indent=2,allow_nan=False))
        print('PROMOTED',name,serial(proof[name]),flush=True)


def joint_decision(old_prior,new_prior,forward_logp,reverse_logp,labels,new_logL,uniform):
    assert 0<=uniform<1
    i,j=labels
    prior_delta=float(np.sum(new_prior)-np.sum(old_prior))
    qf=float(forward_logp[0,i]+forward_logp[1,j])
    qr=float(reverse_logp[0,i]+reverse_logp[1,j])
    selection=qr-qf;ratio=prior_delta+selection
    finite=bool(np.isfinite(new_prior).all() and np.isfinite(new_logL).all() and np.isfinite(ratio))
    contour=bool(np.isfinite(new_logL).all() and np.all(np.asarray(new_logL)>ELL_TEST))
    alpha=float(np.exp(min(0.,ratio))) if finite and contour else 0.
    logu=float(np.log(uniform)) if uniform>0 else -np.inf
    accepted=bool(finite and contour and logu<min(0.,ratio))
    return dict(delta_logprior=prior_delta,log_q_forward=qf,log_q_reverse=qr,
                log_r_selection=selection,log_r=ratio,alpha=alpha,uniform=uniform,
                joint_contour=contour,accepted=accepted,nonfinite_prior=int((~np.isfinite(new_prior)).sum()),
                nonfinite_likelihood=int((~np.isfinite(new_logL)).sum()))

def run_bank(output, slice_seeds, context):
    OUT = Path(output)
    SLICE_SEEDS = slice_seeds
    with (OUT/'sampling_started').open('x') as lock:lock.write('No retry, even after interruption.\n')
    require_cpu_x64()
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    assert json.loads((OUT.parent/'integrity_gate.json').read_text())['passed']
    assert not (OUT/'report.json').exists(),'Never rerun or extend this benchmark'
    design=json.loads((OUT/'design.json').read_text())
    for path,h in design['file_sha256'].items():assert digest(path)==h,path
    assert design['sweeps']==384 and design['ell_test']==ELL_TEST
    assert (design['f_star'],design['sigma_f'])==fixed_frequency_region()
    step,evaluate,bounds,compilation=context
    starts=load(OUT/'initial_states.npz');rng=load(OUT/'random_schedule.npz')
    x,lp,ll=[starts[k].copy() for k in ['position','logprior','loglikelihood']]
    keys=[jax.random.key(seed) for seed in SLICE_SEEDS]
    records=[];exchanges=[];births=[dict(token=w*9+k,birth_walker=w,birth_sweep=-1,birth_label=k,parent=-1,lineage=w*9+k) for w in range(8) for k in range(9)]
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy()
    report=dict(status='running',ell_test=ELL_TEST,completed_sweeps=0,
        design_sha256=digest(OUT/'design.json'),file_sha256=design['file_sha256'],devices=str(jax.devices()),
        compilation_seconds=compilation,source_gain_calls=0,structural_failures=0,
        max_cache_prior_error=0.,max_cache_logL_error=0.,min_selector_log_probability=0.,
        cost=dict(slice_likelihood_proxy=0,slice_prior_proxy=0,exchange_likelihood=0,exchange_prior=0,
                  direct_cache_likelihood=0,direct_cache_prior=0),nonfinite_proposed_priors=0,nonfinite_proposed_likelihoods=0)
    def save():
        (OUT/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
        if records:
            # Canonical walker-major traces compatible with saved baseline tools.
            np.savez_compressed(OUT/'trace.npz',**{k:np.swapaxes(np.asarray([r[k] for r in records]),0,1) for k in records[0]})
            np.savez_compressed(OUT/'exchanges.npz',**{k:np.asarray([r[k] for r in exchanges]) for k in exchanges[0]})
        (OUT/'provenance.json').write_text(json.dumps(births,separators=(',',':')))
    def validate(xs,ps,ls):
        assert xs.shape==(8,54) and ps.shape==ls.shape==(8,)
        assert np.isfinite(xs).all() and np.isfinite(ps).all() and np.isfinite(ls).all() and np.all(ls>ELL_TEST)
        ap,al=map(np.asarray,evaluate(jnp.asarray(xs)))
        report['cost']['direct_cache_prior']+=8;report['cost']['direct_cache_likelihood']+=8
        report['max_cache_prior_error']=max(report['max_cache_prior_error'],float(abs(ap-ps).max()))
        report['max_cache_logL_error']=max(report['max_cache_logL_error'],float(abs(al-ls).max()))
        validate_cache(xs,xs.copy(),ps,ls,ap,al,ELL_TEST,
                       backend=jax.default_backend(),x64=bool(jax.config.x64_enabled))
        return ap.copy(),al.copy()
    save();loop_start=time.perf_counter()
    try:
        validate(x,lp,ll)
        for sweep in range(384):
            subkeys=[]
            for w in range(8):keys[w],sub=jax.random.split(keys[w]);subkeys.append(sub)
            values,info=step(jnp.stack(subkeys),DNSParticleState(jnp.asarray(x),jnp.asarray(lp),jnp.asarray(ll)))
            sx,sp,sl=map(np.asarray,values)
            assert np.asarray(info.slice.is_accepted).all(),'Slice failure: STOP'
            sap,sal=validate(sx,sp,sl)
            ns=np.asarray(info.slice.num_steps);nh=np.asarray(info.slice.num_shrink)
            cost=ns+nh;report['cost']['slice_likelihood_proxy']+=int(cost.sum());report['cost']['slice_prior_proxy']+=int(cost.sum())
            for w in range(8):
                selected=slice_blocks(int(info.component[w]),np.asarray(info.labels[w]))
                keep=np.setdiff1d(np.arange(9),selected)
                assert sx[w].reshape(9,6)[keep].tobytes()==x[w].reshape(9,6)[keep].tobytes()
                for label in selected:
                    token=len(births)
                    births.append(dict(token=token,birth_walker=w,birth_sweep=sweep,birth_label=int(label),parent=int(tokens[w,label]),lineage=int(lineage[w,label])))
                    tokens[w,label]=token
            slice_tokens=tokens.copy();slice_lineage=lineage.copy()
            forward,frequencies=log_probabilities(sx,bounds)
            pairs=round_robin()[sweep%7]
            proposed=sx.copy();labels=[]
            for slot,(a,b) in enumerate(pairs):
                i=draw_label(forward[a],rng['gumbels'][sweep,slot,0]);j=draw_label(forward[b],rng['gumbels'][sweep,slot,1])
                labels.append((i,j))
                proposed[a],proposed[b]=fixed_exchange(sx[a],sx[b],i,j)
                xx,yy=fixed_exchange(proposed[a],proposed[b],i,j)
                assert xx.tobytes()==sx[a].tobytes() and yy.tobytes()==sx[b].tobytes()
                for recipient,donor,rb,db in [(a,b,i,j),(b,a,j,i)]:
                    keep=np.arange(9)!=rb
                    assert proposed[recipient].reshape(9,6)[keep].tobytes()==sx[recipient].reshape(9,6)[keep].tobytes()
                    assert proposed[recipient].reshape(9,6)[rb].tobytes()==sx[donor].reshape(9,6)[db].tobytes()
            pp,pl=map(np.asarray,evaluate(jnp.asarray(proposed)))
            report['cost']['exchange_prior']+=8;report['cost']['exchange_likelihood']+=8
            reverse,_=log_probabilities(proposed,bounds)
            report['min_selector_log_probability']=min(report['min_selector_log_probability'],float(forward.min()),float(reverse.min()))
            x,lp,ll=sx.copy(),sp.copy(),sl.copy()
            for slot,((a,b),(i,j)) in enumerate(zip(pairs,labels)):
                decision=joint_decision(sp[[a,b]],pp[[a,b]],forward[[a,b]],reverse[[a,b]],(i,j),pl[[a,b]],float(rng['uniforms'][sweep,slot]))
                report['nonfinite_proposed_priors']+=decision['nonfinite_prior'];report['nonfinite_proposed_likelihoods']+=decision['nonfinite_likelihood']
                old=(sx[[a,b]],sp[[a,b]],sl[[a,b]]);new=(proposed[[a,b]],pp[[a,b]],pl[[a,b]])
                result=commit_pair(old,new,decision['accepted'])
                x[[a,b]],lp[[a,b]],ll[[a,b]]=result
                for actual,expected in zip(result,new if decision['accepted'] else old):assert actual.tobytes()==expected.tobytes()
                tokens=exchange_tags(tokens,a,b,i,j,decision['accepted'])
                lineage=exchange_tags(lineage,a,b,i,j,decision['accepted'])
                rank=np.argsort(np.argsort(-forward[[a,b]],axis=-1,kind='stable'),axis=-1,kind='stable')+1
                selected_f=frequencies[[a,b],[i,j]]
                exchanges.append(dict(sweep=sweep,round=sweep%7,slot=slot,pair=np.array([a,b]),labels=np.array([i,j]),
                    post_slice_position=sx[[a,b]],proposed_position=proposed[[a,b]],
                    old_logprior=sp[[a,b]],old_loglikelihood=sl[[a,b]],proposed_logprior=pp[[a,b]],proposed_loglikelihood=pl[[a,b]],
                    forward_log_probabilities=forward[[a,b]],reverse_log_probabilities=reverse[[a,b]],
                    selected_log_probabilities=forward[[a,b],[i,j]],selected_ranks=rank[np.arange(2),[i,j]],
                    selected_frequency=selected_f,inside_count=int(np.sum((selected_f>=float(INTERVAL[0]))&(selected_f<=float(INTERVAL[1])))),
                    pre_tokens=slice_tokens[[a,b]][np.arange(2),[i,j]],pre_lineages=slice_lineage[[a,b]][np.arange(2),[i,j]],**decision))
            assert len(np.unique(tokens))==72
            np.testing.assert_array_equal(np.sort(lineage.ravel()),np.arange(72))
            rap,ral=validate(x,lp,ll)
            records.append(dict(position=x.copy(),logprior=lp.copy(),loglikelihood=ll.copy(),
                post_slice_position=sx.copy(),post_slice_logprior=sp.copy(),post_slice_loglikelihood=sl.copy(),
                post_slice_recomputed_prior=sap,post_slice_recomputed_likelihood=sal,
                recomputed_prior=rap,recomputed_likelihood=ral,
                num_steps=ns,num_shrink=nh,slice_component=np.asarray(info.component),slice_labels=np.asarray(info.labels),
                tokens=tokens.copy(),lineage=lineage.copy(),post_slice_tokens=slice_tokens,post_slice_lineage=slice_lineage,
                likelihood_total_proxy=cost+3,proposal_likelihood_proxy=cost+1))
            report['completed_sweeps']=sweep+1;report['loop_wall_seconds']=time.perf_counter()-loop_start
            if (sweep+1)%16==0:
                save();print('sweeps',sweep+1,'/384; joint accepted',sum(r['accepted'] for r in exchanges),'/',len(exchanges),flush=True)
        for path,h in design['file_sha256'].items():assert digest(path)==h,path
        assert digest(OUT/'design.json')==report['design_sha256']
        report.update(status='completed_one_fixed_contour_benchmark',protected_hashes_unchanged=True,
                      exchange_proposals=len(exchanges),joint_accepted=sum(r['accepted'] for r in exchanges))
        report['cost']['total_likelihood_proxy']=sum(report['cost'][k] for k in ['slice_likelihood_proxy','exchange_likelihood','direct_cache_likelihood'])
        report['cost']['total_prior_proxy']=sum(report['cost'][k] for k in ['slice_prior_proxy','exchange_prior','direct_cache_prior'])
    except Exception as exc:
        report.update(status='STOP_failure',structural_failures=int(isinstance(exc,AssertionError)),error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


def require_cpu_x64():
    """Reject environment drift; never silently repair a conflicting setting."""
    if os.environ.get('JAX_PLATFORMS')!='cpu' or os.environ.get('JAX_ENABLE_X64')!='1':
        raise ValueError('Stage 6B requires JAX_PLATFORMS=cpu JAX_ENABLE_X64=1')
    import jax
    if jax.default_backend()!='cpu' or not jax.config.x64_enabled:
        raise ValueError('Stage 6B requires actual CPU/x64 execution')


def eligible_indices(history):
    for key,shape in [('position',(8,256,54)),('logprior',(8,256)),('loglikelihood',(8,256))]:
        value=np.asarray(history[key])
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Require complete finite float64 retained bank')
    return np.flatnonzero(history['loglikelihood'].T.ravel()>ELL_TEST)


def pool_summary(history):
    pool=eligible_indices(history)
    counts=np.bincount(pool%8,minlength=8)
    return dict(total_retained=2048,total_eligible=len(pool),eligible_fraction=len(pool)/2048,
                contributing_donor_walkers=np.flatnonzero(counts),per_donor_counts=counts)


def verify_source_copy(starts,history,proof):
    donor=np.asarray(proof['donor_ids']);retained=np.asarray(proof['retained_indices'])
    pool=eligible_indices(history);indices=retained*8+donor
    if not np.isin(indices,pool).all():raise ValueError('Wrong eligible pool membership')
    if not np.array_equal(indices,proof['selected_flat_indices']):raise ValueError('Wrong source indices')
    if not np.array_equal(np.searchsorted(pool,indices),proof['selected_eligible_pool_indices']):
        raise ValueError('Wrong eligible pool indices')
    for key in ('position','logprior','loglikelihood'):
        value=np.asarray(starts[key]);source=history[key][donor,retained]
        if value.dtype!=np.float64 or value.shape!=source.shape or value.tobytes()!=source.tobytes():
            raise ValueError('Source byte-exact copy failed: '+key)
        if not np.isfinite(value).all():raise ValueError('Nonfinite source copy')
    if not np.all(starts['loglikelihood']>ELL_TEST):raise ValueError('Strict contour failed')
    hashes=[hashlib.sha256(b''.join(np.asarray(starts[k][w]).tobytes()
            for k in ('position','logprior','loglikelihood'))).hexdigest() for w in range(8)]
    if hashes!=proof['source_state_sha256']:raise ValueError('Source hash mismatch')


def compile_model():
    """Called once inside the locked validation; never from pytest."""
    require_cpu_x64()
    from examples.lisa_dns_stage4 import audit,proposals
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    problem,cfg,config_path=audit.load_problem()
    payload=json.loads((Z/'checkpoint_level12.json').read_text())['payload']
    assert cfg==payload['provenance']['config']
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    scalar_step=proposals.build_parameter_step(prior,like,np.full(54,np.pi/np.sqrt(3)),max_steps=10,max_shrinkage=100)
    step=jax.jit(lambda keys,states:jax.lax.map(lambda data:scalar_step(data[0],data[1],jnp.asarray(ELL_TEST)),(keys,states)))
    evaluate=jax.jit(lambda xs:jax.lax.map(lambda x:(prior(x),like(x)),xs))
    starts=load(OUT/'selection/initial_states.npz')
    x,lp,ll=[jnp.asarray(starts[k]) for k in ('position','logprior','loglikelihood')]
    timer=time.perf_counter()
    step=step.lower(jnp.stack([jax.random.key(seed) for seed in SEEDS['selection']]),DNSParticleState(x,lp,ll)).compile()
    evaluate=evaluate.lower(x).compile()
    elapsed=time.perf_counter()-timer
    environment=dict(backend=jax.default_backend(),devices=str(jax.devices()),x64=bool(jax.config.x64_enabled),
        dtype=str(x.dtype),jax_version=jax.__version__,jaxlib_version=__import__('jaxlib').__version__,
        numpy_version=np.__version__,executable=sys.executable,config_path=str(config_path),
        config_sha256=digest(config_path),compilation_seconds=elapsed,
        environment={k:os.environ.get(k) for k in ['JAX_PLATFORMS','JAX_ENABLE_X64']})
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    return step,evaluate,(problem.f_min_cfg,problem.f_max_cfg),elapsed


def pre_sampling_gate(context):
    """Validate BOTH banks before permitting the first constituent transition."""
    require_cpu_x64()
    import jax
    import jax.numpy as jnp
    evaluate=context[1]
    promotion=json.loads((OUT/'promotion.json').read_text())
    starts={name:load(OUT/name/'initial_states.npz') for name in NAMES}
    for key in starts['selection']:
        assert not np.shares_memory(starts['selection'][key],starts['calibration'][key])
    seeds=sum(SEEDS.values(),[])+list(PROMOTION_SEEDS.values())+sum(EXCHANGE_SEEDS.values(),[])
    assert len(seeds)==len(set(seeds))==22
    hashes={};checks={}
    for name in NAMES:
        folder=OUT/name
        design=json.loads((folder/'design.json').read_text())
        for path,h in design['file_sha256'].items():assert digest(path)==h,path
        history={k:v[:,128:].copy() for k,v in load(B/name/'trace.npz').items()
                 if k in ('position','logprior','loglikelihood')}
        verify_source_copy(starts[name],history,promotion[name])
        x,lp,ll=[starts[name][k] for k in ('position','logprior','loglikelihood')]
        ap,al=map(np.asarray,evaluate(jnp.asarray(x)))
        np.savez_compressed(folder/'initial_recomputation.npz',logprior=ap,loglikelihood=al)
        validate_cache(x,history['position'][promotion[name]['donor_ids'],promotion[name]['retained_indices']],
                       lp,ll,ap,al,ELL_TEST,backend=jax.default_backend(),x64=bool(jax.config.x64_enabled))
        checks[name]=dict(passed=True,states=8,source_bank=name,
            max_prior_error=float(np.max(abs(ap-lp))),max_likelihood_error=float(np.max(abs(al-ll))),
            copied_coordinates_byte_exact=True,copied_scalar_caches_byte_exact=True,
            strict_cached_and_recomputed_contour=True)
        for filename in ['initial_states.npz','random_schedule.npz','initial_recomputation.npz','design.json']:
            path=folder/filename;hashes[str(path)]=digest(path)
    schedules={name:load(OUT/name/'random_schedule.npz') for name in NAMES}
    for key in ('gumbels','uniforms'):
        assert not np.array_equal(schedules['selection'][key],schedules['calibration'][key])
    streams={name:dict(
        slice_rng='jax.random default key implementation',
        slice_key_sha256=[hashlib.sha256(np.asarray(jax.random.key_data(jax.random.key(seed))).tobytes()).hexdigest()
                          for seed in SEEDS[name]],
        label_rng='NumPy PCG64 Gumbel',mh_rng='NumPy PCG64 uniform',
        label_schedule_sha256=hashlib.sha256(schedules[name]['gumbels'].tobytes()).hexdigest(),
        mh_schedule_sha256=hashlib.sha256(schedules[name]['uniforms'].tobytes()).hexdigest()) for name in NAMES}
    assert len({h for bank in streams.values() for h in bank['slice_key_sha256']})==16
    gate=dict(passed=True,contract=CONTRACT,banks=checks,independent_arrays=True,
        distinct_slice_label_mh_streams=True,promotion_seeds=PROMOTION_SEEDS,
        slice_seeds=SEEDS,exchange_seeds=EXCHANGE_SEEDS,file_sha256=hashes,streams=streams,
        direct_prior_evaluations=16,direct_likelihood_evaluations=16)
    (OUT/'integrity_gate.json').write_text(json.dumps(serial(gate),indent=2,allow_nan=False))
    print('BOTH BANKS PASSED PRE-SAMPLING GATE',serial(checks),flush=True)


def main():
    # Directory creation is the one-shot lock, before promotion/model execution.
    OUT.mkdir(exist_ok=False)
    (OUT/'run_started').write_text('Exactly one Stage-6B attempt; no retry after any failure or interruption.\n')
    try:
        prepare()
        context=compile_model()
        pre_sampling_gate(context)
        for name in NAMES:run_bank(OUT/name,SEEDS[name],context)
        preflight=json.loads((OUT/'preflight.json').read_text())
        gate=json.loads((OUT/'integrity_gate.json').read_text())
        for path,h in {**preflight['file_sha256'],**gate['file_sha256']}.items():assert digest(path)==h,path
        assert not any(OUT.rglob('checkpoint*')) and not any(B.rglob('checkpoint_level13*'))
        (OUT/'sampling_completed.json').write_text(json.dumps(dict(selection_sweeps=384,calibration_sweeps=384,no_retry=True)))
    except BaseException as exc:
        (OUT/'validation.json').write_text(json.dumps(dict(decision='FAIL',error=f'{type(exc).__name__}: {exc}',
            no_retry=True,frozen_through=12,manual_methodology_ended=True),indent=2))
        raise


if __name__=='__main__':main()
