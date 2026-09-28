"""Stage 4X: exactly one independent level-11 construction attempt; no level 12.

The run body is frozen Stage 4W with output, slice streams and sweep count
substituted. Both banks target ell10 throughout, including calibration.
"""
import json
from pathlib import Path
import time
import numpy as np
from blackjax.ns.dns_levels import build_next_level
from examples.lisa_dns_stage4 import level10_population_validation as frozen
from examples.lisa_dns_stage4.level10_population_validation import (
    ELL10,LOGX10,V,G,serial,load,digest,round_robin,fixed_frequency_region,
    joint_decision,slice_blocks,log_probabilities,draw_label,fixed_exchange,
    commit_pair,exchange_tags,INTERVAL,
)
from examples.lisa_dns_stage4.population_level10_construction import retained_bank
from examples.lisa_dns_stage4.run_ladder_batch import (
    NAMES,load_checkpoint,save_checkpoint,promoted_states,check_banks,
)
from examples.lisa_dns_stage4.run_ladder_extension import SETTINGS,assess

OUT=Path('/tmp/lisa_dns_stage4x_population_level11')
W=Path('/tmp/lisa_dns_stage4w_level10_validation')
SEEDS=dict(selection=list(range(910100,910108)),calibration=list(range(910200,910208)))
EXCHANGE_SEEDS=dict(selection=[2026092807,2026092808],calibration=[2026092809,2026092810])


def verify_prefix(thresholds,masses,reference_thresholds,reference_masses):
    t,m,rt,rm=[np.asarray(a,dtype=np.float64) for a in
               [thresholds,masses,reference_thresholds,reference_masses]]
    if rt.shape!=(11,) or rm.shape!=(11,) or t.shape not in [(11,),(12,)] or m.shape!=t.shape:
        raise ValueError('Requires exact level-0..10 reference and at most one new level')
    if t[:11].tobytes()!=rt.tobytes() or m[:11].tobytes()!=rm.tobytes():
        raise ValueError('Frozen level-0..10 prefix changed')


def recover_banks(histories):
    """Only each named bank's own retained history; exact W last-survivor rule."""
    if set(histories)!=set(NAMES):raise ValueError('Two independent named histories required')
    starts={};proof={}
    for name in NAMES:
        starts[name],proof[name]=frozen.recover_starts(histories[name])
        indices=proof[name]['absolute_trace_indices']
        for key in starts[name]:
            expected=np.asarray(histories[name][key])[np.arange(8),indices]
            if starts[name][key].tobytes()!=expected.tobytes():
                raise ValueError('Start/cache recovery mismatch')
            assert not np.shares_memory(starts[name][key],histories[name][key])
        proof[name]['starting_logL']=starts[name]['loglikelihood'].copy()
    check_banks(starts,ELL10)
    for key in starts['selection']:
        if np.shares_memory(starts['selection'][key],starts['calibration'][key]):
            raise ValueError('Shared bank state/cache storage')
    return starts,proof


def select_candidate(selection):
    """No calibration argument and no Stage-4W pseudo-threshold input."""
    return build_next_level(selection['loglikelihood'].T,ELL10,
        target_compression=SETTINGS['target_compression'],
        block_size=SETTINGS['block_size'],min_ess=SETTINGS['min_ess'])


def decision_ladder(thresholds,masses,row):
    if len(thresholds)!=11 or len(masses)!=11 or row.get('level')!=11:
        raise ValueError('This entry point attempts level 11 only; no level-12 path')
    t=np.asarray(thresholds,dtype=np.float64).copy();m=np.asarray(masses,dtype=np.float64).copy()
    eligible=(row['selection']['status']=='ok' and 0<row['calibration']['ratio']<1
              and row['calibration']['ess']>=SETTINGS['min_ess'])
    if bool(row['accepted'])!=bool(eligible):raise ValueError('Inconsistent existing calibration gate')
    if eligible:
        if not np.isfinite(row['threshold']) or row['threshold']<=t[-1]:
            raise ValueError('Non-increasing threshold')
        mass=float(m[-1]+np.log(row['calibration']['ratio']))
        if mass!=row['estimated_log_mass']:raise ValueError('Mass must use independent calibration only')
        t=np.append(t,row['threshold']);m=np.append(m,mass)
    verify_prefix(t,m,thresholds,masses)
    return t,m


def freeze_level11(output,thresholds,masses,row,traces,metadata,provenance):
    t,m=decision_ladder(thresholds,masses,row)
    if len(t)==11:return t,m,None
    states,indices=promoted_states(traces,row['threshold'],11)
    path=Path(output)/'checkpoint_level11.json'
    save_checkpoint(path,t,m,states,list(metadata['diagnostics'])+[row],
        {**provenance,'promotion_indices':indices,
         'terminal_stage':'Stage-4X stops after level 11; no continuation executed'})
    rt,rm,_,_=load_checkpoint(path,(thresholds,masses))
    assert rt.tobytes()==t.tobytes() and rm.tobytes()==m.tobytes()
    return t,m,path


def preflight():
    assert not OUT.exists(),'Exactly one attempt; no retry or extension'
    t,m,_,metadata=load_checkpoint(V/'checkpoint_level10.json')
    rt,rm,_,_=load_checkpoint(G/'checkpoint_level9.json')
    frozen.verify_ladder(t,m,rt,rm);verify_prefix(t,m,t,m)
    validation=json.loads((V/'validation.json').read_text())
    assert validation['status']=='passed' and validation['frozen_through']==10
    for path,h in validation['output_sha256'].items():assert digest(path)==h,path
    wvalidation=json.loads((W/'validation.json').read_text())
    assert wvalidation['status']=='passed' and wvalidation['classification']=='B'
    for path,h in wvalidation['output_sha256'].items():assert digest(path)==h,path
    vr=json.loads((V/'report.json').read_text());wd=json.loads((W/'design.json').read_text())
    for path,h in vr['file_sha256'].items():assert digest(path)==h,path
    for path,h in wd['file_sha256'].items():assert digest(path)==h,path
    histories={name:load(V/name/'trace.npz') for name in NAMES}
    starts,recovery=recover_banks(histories)
    assert len(set(sum(SEEDS.values(),[])+sum(EXCHANGE_SEEDS.values(),[])))==20
    assert (wd['f_star'],wd['sigma_f'])==fixed_frequency_region()
    np.testing.assert_array_equal(wd['round_robin'],round_robin())
    paths=[V/'checkpoint_level10.json',V/'comparison.json',V/'validation.json',W/'validation.json',
        W/'decision.json',W/'design.json',Path(__file__),Path(frozen.__file__),
        Path('examples/lisa_dns_stage4/analyze_level10_population.py'),
        Path('examples/lisa_dns_stage4/analyze_population_level11.py'),
        *[V/name/'trace.npz' for name in NAMES]]
    hashes={**vr['file_sha256'],**wd['file_sha256'],**{str(p):digest(p) for p in paths}}
    OUT.mkdir()
    (OUT/'checkpoint_level10.json').write_bytes((V/'checkpoint_level10.json').read_bytes())
    hashes[str(OUT/'checkpoint_level10.json')]=digest(OUT/'checkpoint_level10.json')
    proof=dict(stage='4X',frozen_thresholds=t,frozen_log_masses=m,settings=SETTINGS,
        seeds=SEEDS,exchange_seeds=EXCHANGE_SEEDS,recovery=recovery,
        bank_sources={n:str(V/n/'trace.npz') for n in NAMES},
        exact_starts_and_caches=True,independent_banks=True,attempted_levels=[11],
        starting_rule='LAST own-walker retained Stage-4V strict ell10 survivor',
        stage4w_use='Validation and frozen kernel only; no W states or pseudo-threshold used',
        file_sha256=hashes)
    (OUT/'preflight.json').write_text(json.dumps(serial(proof),indent=2,allow_nan=False))
    for name in NAMES:
        bank=OUT/name;bank.mkdir()
        np.savez_compressed(bank/'initial_states.npz',**starts[name])
        ls,ms=EXCHANGE_SEEDS[name]
        np.savez_compressed(bank/'random_schedule.npz',
            gumbels=np.random.Generator(np.random.PCG64(ls)).gumbel(size=(384,4,2,9)),
            uniforms=np.random.Generator(np.random.PCG64(ms)).random((384,4)))
        design=dict(stage='4X',bank=name,ell10=ELL10,logX10=LOGX10,
            sweeps=384,burn_in=128,retained=256,walkers=8,settings=SETTINGS,
            frequency_interval=INTERVAL,f_star=wd['f_star'],sigma_f=wd['sigma_f'],
            round_robin=round_robin(),slice_seeds=SEEDS[name],label_seed=ls,mh_seed=ms,
            slice=wd['slice'],ordering=wd['ordering'],provenance=wd['provenance'],
            cache_checks=wd['cache_checks'],source_gain_calls=0,
            recovery=recovery[name],source_history=str(V/name/'trace.npz'),
            file_sha256={**hashes,**{str(bank/p):digest(bank/p) for p in ['initial_states.npz','random_schedule.npz']}})
        (bank/'design.json').write_text(json.dumps(serial(design),indent=2,allow_nan=False))
        print('Recovered',name,recovery[name],flush=True)
    return t,m,starts,metadata,proof


def run_bank(output, slice_seeds):
    OUT = Path(output)
    SLICE_SEEDS = slice_seeds
    # This is the only entry point importing/evaluating the real LISA model.
    from examples.lisa_dns_stage4 import audit,proposals
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    assert not (OUT/'report.json').exists(),'Never rerun or extend this benchmark'
    design=json.loads((OUT/'design.json').read_text())
    for path,h in design['file_sha256'].items():assert digest(path)==h,path
    assert design['sweeps']==384 and design['ell10']==ELL10 and design['logX10']==LOGX10
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
        for sweep in range(384):
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


def main():
    t,m,starts,metadata,proof=preflight()
    traces={};report=dict(status='running_single_level11_attempt',attempted_levels=[11],
        settings=SETTINGS,seeds=SEEDS,exchange_seeds=EXCHANGE_SEEDS,
        structural_failures=0,file_sha256=proof['file_sha256'])
    def save():
        (OUT/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
    save()
    try:
        for name in NAMES:
            run_bank(OUT/name,SEEDS[name])
            traces[name]=retained_bank(load(OUT/name/'trace.npz'))
            verify_prefix(t,m,proof['frozen_thresholds'],proof['frozen_log_masses'])
            if name=='selection':
                candidate=select_candidate(traces[name])
                (OUT/'selection_candidate.json').write_text(json.dumps(serial(candidate),indent=2,allow_nan=False))
                report['candidate_selected_before_calibration']=candidate;save()
                print('SELECTION',candidate,flush=True)
        check_banks(starts,ELL10)
        assert not ({x.tobytes() for x in traces['selection']['position'].reshape(-1,54)} &
                    {x.tobytes() for x in traces['calibration']['position'].reshape(-1,54)})
        row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],ELL10,LOGX10)
        assert serial(row['selection'])==serial(candidate)
        row['level']=11;report['decision']=row
        for path,h in proof['file_sha256'].items():assert digest(path)==h,path
        assert (OUT/'checkpoint_level10.json').read_bytes()==(V/'checkpoint_level10.json').read_bytes()
        from examples.lisa_dns_stage4.analyze_population_level11 import audit_bank
        report['population']={name:audit_bank(OUT/name) for name in NAMES}
        nt,nm,path=freeze_level11(OUT,t,m,row,traces,metadata,
            {**metadata['provenance'],'population_construction':str(OUT),
             'population_seeds':SEEDS,'exchange_seeds':EXCHANGE_SEEDS,
             'file_sha256':proof['file_sha256']})
        report.update(status='frozen_level11' if path else 'stopped_failed_level11_gate',
            frozen_thresholds=nt,frozen_log_masses=nm,
            checkpoint_level11=str(path) if path else None,
            frozen_prefix_unchanged=True,independent_banks_verified=True)
        np.savez_compressed(OUT/'levels.npz',thresholds=nt,log_mass=nm)
    except Exception as exc:
        report.update(status='STOP_failure',structural_failures=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()
    print('DECISION',report['status'],'calibration ESS',row['calibration']['ess'],flush=True)


if __name__=='__main__':main()
