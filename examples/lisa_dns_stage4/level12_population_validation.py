"""Stage 5A: one fixed frozen level-12 validation; no level creation or mass update."""
import json
from pathlib import Path
import time
import numpy as np
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    serial,load,digest,round_robin,log_probabilities,draw_label,commit_pair,
    exchange_tags,slice_blocks,fixed_exchange,INTERVAL,fixed_frequency_region,
)

OUT=Path('/tmp/lisa_dns_stage5a_level12_validation')
X=Path('/tmp/lisa_dns_stage4x_population_level11')
Z=Path('/tmp/lisa_dns_stage4z_population_level12')
Y=Path('/tmp/lisa_dns_stage4y_level11_validation')
W=Path('/tmp/lisa_dns_stage4w_level10_validation')
V=Path('/tmp/lisa_dns_stage4v_population_level10')
G=Path('/tmp/lisa_dns_stage4_ladder_batch')
ELL12=-108187.99712765554
ELL11=-108868.60077896297
LOGX12=-10.418392711998465
LOGX11=-9.643899891984567
ELL10=-109780.28875081123
LOGX10=-8.394305725653265
SWEEPS=512
SLICE_SEEDS=list(range(940100,940108))
LABEL_SEED=2026092917
MH_SEED=2026092918
RECOVERY_CRITERION=('For each of all eight walkers separately: inclusive contemporaneous '
    '25th-75th percentiles of the OTHER seven walkers, numpy linear quantiles; '
    'sweep 0 is the recovered ensemble; sweeps 1..512 are post-exchange states; '
    'no smoothing or criterion retuning.')


def verify_ladder(thresholds,masses,reference_thresholds,reference_masses):
    t,m,rt,rm=[np.asarray(a,dtype=np.float64) for a in
               [thresholds,masses,reference_thresholds,reference_masses]]
    if t.shape!=(13,) or m.shape!=(13,) or rt.shape!=(12,) or rm.shape!=(12,):
        raise ValueError('Exactly frozen levels 0..12 and Stage-4X levels 0..11 required')
    if t[:12].tobytes()!=rt.tobytes() or m[:12].tobytes()!=rm.tobytes():
        raise ValueError('Frozen lower prefix changed')
    if (t[-2],m[-2],t[-1],m[-1])!=(ELL11,LOGX11,ELL12,LOGX12):
        raise ValueError('Frozen level 11 or 12 changed')


def recover_starts(raw):
    """Final state if valid, otherwise LAST valid retained state; never argmax."""
    for k,shape in [('position',(8,384,54)),('logprior',(8,384)),('loglikelihood',(8,384))]:
        if np.asarray(raw[k]).shape!=shape or not np.isfinite(raw[k]).all():
            raise ValueError('Invalid Stage-4Z calibration trace')
    indices=[];fallback=[]
    for w in range(8):
        valid=np.flatnonzero(raw['loglikelihood'][w,128:]>ELL12)
        if not len(valid):raise ValueError(f'No retained valid start for walker {w}; STOP')
        ix=int(valid[-1]+128);indices.append(ix)
        if ix!=383:fallback.append(w)
    starts={k:np.array(raw[k][np.arange(8),indices],copy=True)
            for k in ['position','logprior','loglikelihood']}
    assert np.all(starts['loglikelihood']>ELL12)
    return starts,dict(fallback_walkers=fallback,absolute_trace_indices=indices,
        retained_trace_indices=[i-128 for i in indices],rule='last retained strict ell12 survivor')


def prepare():
    from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint
    from examples.lisa_dns_stage4.level10_population_validation import verify_ladder as verify_level10
    assert not OUT.exists(),'Exactly one validation; never retry or extend'
    from examples.lisa_dns_stage4.level11_population_validation import verify_ladder as verify_level11
    t,m,_,metadata=load_checkpoint(Z/'checkpoint_level12.json')
    rt,rm,_,_=load_checkpoint(X/'checkpoint_level11.json')
    vt,vm,_,_=load_checkpoint(V/'checkpoint_level10.json')
    gt,gm,_,_=load_checkpoint(G/'checkpoint_level9.json')
    verify_level10(vt,vm,gt,gm);verify_level11(rt,rm,vt,vm);verify_ladder(t,m,rt,rm)
    xr=json.loads((Z/'report.json').read_text());wd=json.loads((Y/'design.json').read_text())
    assert xr['status']=='frozen_level12' and xr['structural_failures']==0
    assert xr['attempted_levels']==[12]
    levels=load(Z/'levels.npz')
    assert levels['thresholds'].tobytes()==t.tobytes() and levels['log_mass'].tobytes()==m.tobytes()
    for name in ['selection','calibration']:
        assert xr['population'][name]['exact_replay_passed']
    for path,h in xr['file_sha256'].items():assert digest(path)==h,path
    for path,h in wd['file_sha256'].items():assert digest(path)==h,path
    starts,recovery=recover_starts(load(Z/'calibration/trace.npz'))
    assert (wd['f_star'],wd['sigma_f'])==fixed_frequency_region()
    np.testing.assert_array_equal(wd['round_robin'],round_robin())
    paths=[Z/'checkpoint_level12.json',Z/'calibration/trace.npz',Z/'report.json',Z/'comparison.json',
        Z/'levels.npz',X/'checkpoint_level11.json',Y/'design.json',Y/'comparison.json',Y/'report.json',
        W/'design.json',W/'comparison.json',W/'report.json',W/'provenance_transfers.json',
        Path(__file__),Path('examples/lisa_dns_stage4/analyze_level12_population.py')]
    hashes={**xr['file_sha256'],**{str(p):digest(p) for p in paths}}
    OUT.mkdir()
    (OUT/'checkpoint_level12.json').write_bytes((Z/'checkpoint_level12.json').read_bytes())
    np.savez_compressed(OUT/'initial_states.npz',**starts)
    np.savez_compressed(OUT/'random_schedule.npz',
        gumbels=np.random.Generator(np.random.PCG64(LABEL_SEED)).gumbel(size=(512,4,2,9)),
        uniforms=np.random.Generator(np.random.PCG64(MH_SEED)).random((512,4)))
    hashes.update({str(OUT/p):digest(OUT/p) for p in
        ['checkpoint_level12.json','initial_states.npz','random_schedule.npz']})
    design=dict(stage='5A',ell12=ELL12,logX12=LOGX12,sweeps=512,walkers=8,burn_in=0,
        frozen_thresholds=t,frozen_log_masses=m,recovery=recovery,
        initial_loglikelihood=starts['loglikelihood'],diagnostic_walkers=list(range(8)),recovery_criterion=RECOVERY_CRITERION,
        diagnostic_threshold='After all 512 sweeps only: sorted pooled logL[floor(N*(1-exp(-1)))]; strict upper tail; never a level or calibrated mass',
        windows=['first256','second256','all512'],block_size=32,
        frequency_interval=INTERVAL,f_star=wd['f_star'],sigma_f=wd['sigma_f'],round_robin=round_robin(),
        slice_seeds=SLICE_SEEDS,label_seed=LABEL_SEED,mh_seed=MH_SEED,slice=wd['slice'],
        ordering=wd['ordering'],provenance=wd['provenance'],geometry=wd['geometry'],cache_checks=wd['cache_checks'],
        classification='A good communication / B usable residual heterogeneity / C persistent serious separation / D structural or statistical failure; prioritize half-to-half logL and diagnostic upper-tail agreement, pairwise KS, symmetric all-walker recovery, provenance and permutation-invariant geometry. Only A or convincing B may justify one later construction; never classify from acceptance alone.',
        half_to_half='Describe whether low-exceedance behavior persists, migrates or largely disappears, without a new gate.',
        level_construction=False,independent_calibration=False,source_gain_calls=0,file_sha256=hashes)
    (OUT/'design.json').write_text(json.dumps(serial(design),indent=2,allow_nan=False))
    print('Frozen starts',recovery,'logL',starts['loglikelihood'],flush=True)


def joint_decision(old_prior,new_prior,forward_logp,reverse_logp,labels,new_logL,uniform):
    assert 0<=uniform<1
    i,j=labels
    prior_delta=float(np.sum(new_prior)-np.sum(old_prior))
    qf=float(forward_logp[0,i]+forward_logp[1,j])
    qr=float(reverse_logp[0,i]+reverse_logp[1,j])
    selection=qr-qf;ratio=prior_delta+selection
    finite=bool(np.isfinite(new_prior).all() and np.isfinite(new_logL).all() and np.isfinite(ratio))
    contour=bool(np.isfinite(new_logL).all() and np.all(np.asarray(new_logL)>ELL12))
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
    assert design['sweeps']==512 and design['ell12']==ELL12 and design['logX12']==LOGX12
    assert (design['f_star'],design['sigma_f'])==fixed_frequency_region()
    problem,cfg,config_path=audit.load_problem()
    payload=json.loads((Z/'checkpoint_level12.json').read_text())['payload']
    assert cfg==payload['provenance']['config']
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    scalar_step=proposals.build_parameter_step(prior,like,np.full(54,np.pi/np.sqrt(3)),max_steps=10,max_shrinkage=100)
    step=jax.jit(lambda keys,states:jax.lax.map(lambda data:scalar_step(data[0],data[1],jnp.asarray(ELL12)),(keys,states)))
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
    report=dict(status='running',ell12=ELL12,logX12=LOGX12,completed_sweeps=0,
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
        assert np.isfinite(xs).all() and np.isfinite(ps).all() and np.isfinite(ls).all() and np.all(ls>ELL12)
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
