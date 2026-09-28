"""Stage 4W: one immutable level-10 population validation, no construction.

The Stage-4U run body and joint decision are copied with only the fixed contour,
checkpoint source and metadata names changed. Source-equivalence tests protect
that contract. Selector and all other transition primitives are imported.
"""
import json
from pathlib import Path
import time
import numpy as np
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    serial, load, digest, round_robin, log_probabilities, draw_label,
    commit_pair, exchange_tags, slice_blocks, fixed_exchange, INTERVAL,
    fixed_frequency_region,
)

OUT=Path('/tmp/lisa_dns_stage4w_level10_validation')
V=Path('/tmp/lisa_dns_stage4v_population_level10')
U=Path('/tmp/lisa_dns_stage4u_population')
G=Path('/tmp/lisa_dns_stage4_ladder_batch')
ELL10=-109780.28875081123
LOGX10=-8.394305725653265
SLICE_SEEDS=list(range(900100,900108))
LABEL_SEED=2026092805
MH_SEED=2026092806
SWEEPS=512
RECOVERY_CRITERION=('Inclusive contemporaneous 25th-75th percentiles of the other '
    'seven walkers, numpy linear quantiles; sweep 0 is the recovered ensemble; '
    'sweeps 1..512 are post-exchange states; no temporal smoothing or retuning.')


def verify_ladder(thresholds,masses,reference_thresholds,reference_masses):
    t,m,rt,rm=[np.asarray(a,dtype=np.float64) for a in
               [thresholds,masses,reference_thresholds,reference_masses]]
    if t.shape!=(11,) or m.shape!=(11,) or rt.shape!=(10,) or rm.shape!=(10,):
        raise ValueError('Exactly frozen levels 0..10 and Stage-4G levels 0..9 required')
    if t[:10].tobytes()!=rt.tobytes() or m[:10].tobytes()!=rm.tobytes():
        raise ValueError('Stage-4G prefix changed')
    if t[-1]!=ELL10 or m[-1]!=LOGX10:
        raise ValueError('Frozen level 10 changed')


def recover_starts(raw):
    """Final state if valid, otherwise LAST valid retained state; never argmax."""
    for k,shape in [('position',(8,384,54)),('logprior',(8,384)),('loglikelihood',(8,384))]:
        if np.asarray(raw[k]).shape!=shape or not np.isfinite(raw[k]).all():
            raise ValueError('Invalid Stage-4V calibration trace')
    indices=[];fallback=[]
    for w in range(8):
        valid=np.flatnonzero(raw['loglikelihood'][w,128:]>ELL10)
        if not len(valid):raise ValueError(f'No retained valid start for walker {w}; STOP')
        ix=int(valid[-1]+128);indices.append(ix)
        if ix!=383:fallback.append(w)
    starts={k:np.array(raw[k][np.arange(8),indices],copy=True)
            for k in ['position','logprior','loglikelihood']}
    assert np.all(starts['loglikelihood']>ELL10)
    return starts,dict(fallback_walkers=fallback,absolute_trace_indices=indices,
        retained_trace_indices=[i-128 for i in indices],rule='last retained strict ell10 survivor')


def prepare():
    from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint
    assert not OUT.exists(),'Single run only: never retry or extend'
    t,m,_,meta=load_checkpoint(V/'checkpoint_level10.json')
    rt,rm,_,_=load_checkpoint(G/'checkpoint_level9.json')
    verify_ladder(t,m,rt,rm)
    validation=json.loads((V/'validation.json').read_text())
    assert validation['status']=='passed' and validation['frozen_through']==10
    for path,h in validation['output_sha256'].items():assert digest(path)==h,path
    vr=json.loads((V/'report.json').read_text());ud=json.loads((U/'design.json').read_text())
    assert vr['status']=='frozen_level10'
    for path,h in vr['file_sha256'].items():assert digest(path)==h,path
    starts,recovery=recover_starts(load(V/'calibration/trace.npz'))
    assert (ud['f_star'],ud['sigma_f'])==fixed_frequency_region()
    np.testing.assert_array_equal(ud['round_robin'],round_robin())
    paths=[V/'checkpoint_level10.json',V/'calibration/trace.npz',V/'validation.json',
        U/'design.json',U/'comparison.json',U/'trace.npz',U/'report.json',
        Path(__file__),Path('examples/lisa_dns_stage4/analyze_level10_population.py')]
    hashes={**vr['file_sha256'],**{str(p):digest(p) for p in paths}}
    OUT.mkdir()
    (OUT/'checkpoint_level10.json').write_bytes((V/'checkpoint_level10.json').read_bytes())
    np.savez_compressed(OUT/'initial_states.npz',**starts)
    np.savez_compressed(OUT/'random_schedule.npz',
        gumbels=np.random.Generator(np.random.PCG64(LABEL_SEED)).gumbel(size=(512,4,2,9)),
        uniforms=np.random.Generator(np.random.PCG64(MH_SEED)).random((512,4)))
    hashes.update({str(OUT/p):digest(OUT/p) for p in
                  ['checkpoint_level10.json','initial_states.npz','random_schedule.npz']})
    design=dict(stage='4W',ell10=ELL10,logX10=LOGX10,sweeps=512,walkers=8,burn_in=0,
        frozen_thresholds=t,frozen_log_masses=m,recovery=recovery,
        initial_loglikelihood=starts['loglikelihood'],walker6_criterion=RECOVERY_CRITERION,
        diagnostic_threshold='After completion only: sorted pooled logL[floor(N*(1-exp(-1)))]; strict upper-tail exceedance; never a level or calibrated mass',
        windows=['first256','second256','all512'],block_size=32,
        frequency_interval=INTERVAL,f_star=ud['f_star'],sigma_f=ud['sigma_f'],
        round_robin=round_robin(),slice_seeds=SLICE_SEEDS,label_seed=LABEL_SEED,mh_seed=MH_SEED,
        slice=ud['slice'],ordering=ud['ordering'],provenance=ud['provenance'],
        geometry=ud['geometry'],cache_checks=ud['cache_checks'],
        classification='A good communication / B usable residual heterogeneity / C persistent serious separation / D structural failure; prioritize both halves, all-walker logL and upper-tail agreement, pairwise KS, walker6 recovery, provenance and permutation-invariant geometry. No acceptance-only classification; only A or convincing B may justify a later separate construction.',
        level_construction=False,independent_calibration=False,source_gain_calls=0,
        file_sha256=hashes)
    (OUT/'design.json').write_text(json.dumps(serial(design),indent=2,allow_nan=False))
    print('Frozen starts:',recovery,'logL:',starts['loglikelihood'],flush=True)


def joint_decision(old_prior,new_prior,forward_logp,reverse_logp,labels,new_logL,uniform):
    assert 0<=uniform<1
    i,j=labels
    prior_delta=float(np.sum(new_prior)-np.sum(old_prior))
    qf=float(forward_logp[0,i]+forward_logp[1,j])
    qr=float(reverse_logp[0,i]+reverse_logp[1,j])
    selection=qr-qf;ratio=prior_delta+selection
    finite=bool(np.isfinite(new_prior).all() and np.isfinite(new_logL).all() and np.isfinite(ratio))
    contour=bool(np.isfinite(new_logL).all() and np.all(np.asarray(new_logL)>ELL10))
    alpha=float(np.exp(min(0.,ratio))) if finite and contour else 0.
    logu=float(np.log(uniform)) if uniform>0 else -np.inf
    accepted=bool(finite and contour and logu<min(0.,ratio))
    return dict(delta_logprior=prior_delta,log_q_forward=qf,log_q_reverse=qr,
                log_r_selection=selection,log_r=ratio,alpha=alpha,uniform=uniform,
                joint_contour=contour,accepted=accepted,nonfinite_prior=int((~np.isfinite(new_prior)).sum()),
                nonfinite_likelihood=int((~np.isfinite(new_logL)).sum()))


def run():
    # This is the only entry point importing/evaluating the real LISA model.
    from examples.lisa_dns_stage4 import audit,proposals
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    assert not (OUT/'report.json').exists(),'Never rerun or extend this benchmark'
    design=json.loads((OUT/'design.json').read_text())
    for path,h in design['file_sha256'].items():assert digest(path)==h,path
    assert design['sweeps']==512 and design['ell10']==ELL10 and design['logX10']==LOGX10
    assert (design['f_star'],design['sigma_f'])==fixed_frequency_region()
    problem,cfg,config_path=audit.load_problem()
    payload=json.loads((V/'checkpoint_level10.json').read_text())['payload']
    assert cfg==payload['provenance']['config']
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    scalar_step=proposals.build_parameter_step(prior,like,np.full(54,np.pi/np.sqrt(3)),max_steps=10,max_shrinkage=100)
    step=jax.jit(lambda keys,states:jax.lax.map(lambda data:scalar_step(data[0],data[1],jnp.asarray(ELL10)),(keys,states)))
    evaluate=jax.jit(lambda xs:jax.lax.map(lambda x:(prior(x),like(x)),xs))
    starts=load(OUT/'initial_states.npz');rng=load(OUT/'random_schedule.npz')
    x,lp,ll=[starts[k].copy() for k in ['position','logprior','loglikelihood']]
    keys=[jax.random.key(seed) for seed in SLICE_SEEDS]
    timer=time.perf_counter()
    step=step.lower(jnp.stack(keys),DNSParticleState(jnp.asarray(x),jnp.asarray(lp),jnp.asarray(ll))).compile()
    evaluate=evaluate.lower(jnp.asarray(x)).compile()
    compilation=time.perf_counter()-timer
    bounds=(problem.f_min_cfg,problem.f_max_cfg)
    records=[];exchanges=[];births=[dict(token=w*9+k,birth_walker=w,birth_sweep=-1,birth_label=k,parent=-1,lineage=w*9+k) for w in range(8) for k in range(9)]
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy()
    report=dict(status='running',ell10=ELL10,logX10=LOGX10,completed_sweeps=0,
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
        assert np.isfinite(xs).all() and np.isfinite(ps).all() and np.isfinite(ls).all() and np.all(ls>ELL10)
        ap,al=map(np.asarray,evaluate(jnp.asarray(xs)))
        report['cost']['direct_cache_prior']+=8;report['cost']['direct_cache_likelihood']+=8
        report['max_cache_prior_error']=max(report['max_cache_prior_error'],float(abs(ap-ps).max()))
        report['max_cache_logL_error']=max(report['max_cache_logL_error'],float(abs(al-ls).max()))
        np.testing.assert_allclose(ap,ps,rtol=0,atol=1e-9)
        np.testing.assert_allclose(al,ls,rtol=0,atol=1e-7)
    save();loop_start=time.perf_counter()
    try:
        validate(x,lp,ll)
        for sweep in range(512):
            subkeys=[]
            for w in range(8):keys[w],sub=jax.random.split(keys[w]);subkeys.append(sub)
            values,info=step(jnp.stack(subkeys),DNSParticleState(jnp.asarray(x),jnp.asarray(lp),jnp.asarray(ll)))
            sx,sp,sl=map(np.asarray,values)
            assert np.asarray(info.slice.is_accepted).all(),'Slice failure: STOP'
            validate(sx,sp,sl)
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
            validate(x,lp,ll)
            records.append(dict(position=x.copy(),logprior=lp.copy(),loglikelihood=ll.copy(),
                post_slice_position=sx.copy(),post_slice_logprior=sp.copy(),post_slice_loglikelihood=sl.copy(),
                num_steps=ns,num_shrink=nh,slice_component=np.asarray(info.component),slice_labels=np.asarray(info.labels),
                tokens=tokens.copy(),lineage=lineage.copy(),post_slice_tokens=slice_tokens,post_slice_lineage=slice_lineage,
                likelihood_total_proxy=cost+3,proposal_likelihood_proxy=cost+1))
            report['completed_sweeps']=sweep+1;report['loop_wall_seconds']=time.perf_counter()-loop_start
            if (sweep+1)%16==0:
                save();print('sweeps',sweep+1,'/512; joint accepted',sum(r['accepted'] for r in exchanges),'/',len(exchanges),flush=True)
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


if __name__=='__main__':
    prepare()
    run()
