"""Stage-4P: exactly one Hybrid-P level-10 construction attempt; no level loop."""
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from blackjax.ns.dns_levels import build_next_level
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.hybrid_logistic_refresh import build_frozen_slice, hybrid_sweep, sweep_cost
from examples.lisa_dns_stage4.sequential_logistic_refresh import State,transition,random_schedule,validate
from examples.lisa_dns_stage4.logistic_mh_refresh import ELL9,safe
from examples.lisa_dns_stage4.run_ladder_batch import (
    NAMES,load_checkpoint,save_checkpoint,check_banks,promoted_states,schedule)
from examples.lisa_dns_stage4.run_ladder_extension import SETTINGS,assess
from examples.lisa_dns_stage4.compare_sequential_refresh import load

SOURCE=Path('/tmp/lisa_dns_stage4_ladder_batch')
OUT=Path('/tmp/lisa_dns_stage4p_hybrid_level10')
SEEDS=dict(selection=list(range(880100,880108)),calibration=list(range(880200,880208)))


def verify_prefix(thresholds,masses,reference_thresholds,reference_masses):
    """Require all ten frozen lower levels, exactly, without rounded constants."""
    if len(reference_thresholds)!=10 or len(reference_masses)!=10:
        raise ValueError('Requires a frozen level-0..9 reference')
    for actual,ref in [(thresholds,reference_thresholds),(masses,reference_masses)]:
        a=np.asarray(actual,dtype=np.float64);b=np.asarray(ref,dtype=np.float64)
        if len(a)<10 or a[:10].tobytes()!=b.tobytes():
            raise ValueError('Frozen lower prefix changed')


def recover_starts(states,retained,indices):
    """Recover each original level-10 start from its own level-9 retained trace."""
    recovered={}
    for name in NAMES:
        ix=np.asarray(indices[name],dtype=int)
        if ix.shape!=(8,) or np.any((ix<0)|(ix>=256)):
            raise ValueError('Invalid historical promotion indices')
        recovered[name]={}
        for k in ['position','logprior','loglikelihood']:
            a=np.asarray(retained[name][k])[np.arange(8),ix]
            b=np.asarray(states[name][k],dtype=np.float64)
            if a.shape!=b.shape or a.tobytes()!=b.tobytes():
                raise ValueError('Cannot recover exact historical starts and caches')
            recovered[name][k]=a.copy()
    return recovered


def retained_bank(raw):
    """Frozen Stage-4G bookkeeping: 8 x (128 discarded + 256 retained)."""
    np.testing.assert_array_equal(raw['walker'],np.repeat(np.arange(8),384))
    np.testing.assert_array_equal(raw['iteration'],np.tile(np.arange(384),8))
    return {k:raw[k].reshape((8,384)+raw[k].shape[1:])[:,128:]
            for k in ['position','logprior','loglikelihood']}


def decision_ladder(thresholds,masses,row):
    """One and only one possible append. No level-11 continuation is exposed."""
    if len(thresholds)!=10 or len(masses)!=10:
        raise ValueError('This entry point attempts level 10 only')
    t=np.asarray(thresholds,dtype=np.float64).copy();m=np.asarray(masses,dtype=np.float64).copy()
    eligible=(row['selection']['status']=='ok' and 0<row['calibration']['ratio']<1
              and row['calibration']['ess']>=SETTINGS['min_ess'])
    if bool(row['accepted'])!=bool(eligible):raise ValueError('Inconsistent existing calibration gate')
    if eligible:
        if not np.isfinite(row['threshold']) or row['threshold']<=t[-1]:raise ValueError('Non-increasing threshold')
        mass=float(m[-1]+np.log(row['calibration']['ratio']))
        if mass!=row['estimated_log_mass']:raise ValueError('Mass must use independent calibration only')
        t=np.append(t,row['threshold']);m=np.append(m,mass)
    verify_prefix(t,m,thresholds,masses)
    return t,m


def freeze_level10(output,thresholds,masses,row,traces,metadata,provenance):
    """Failure creates no new checkpoint; success freezes exactly one level."""
    t,m=decision_ladder(thresholds,masses,row)
    if len(t)==10:return t,m,None
    states,indices=promoted_states(traces,row['threshold'],10)
    path=Path(output)/'checkpoint_level10.json'
    save_checkpoint(path,t,m,states,list(metadata['diagnostics'])+[row],
        {**provenance,'promotion_indices':indices,'terminal_stage':'Stage-4P stops after level 10; no continuation executed'})
    rt,rm,_,_=load_checkpoint(path,(thresholds,masses))
    assert rt.tobytes()==t.tobytes() and rm.tobytes()==m.tobytes()
    return t,m,path


def preflight():
    OUT.mkdir(exist_ok=False)
    checkpoint=SOURCE/'checkpoint_level9.json'
    t,m,states,metadata=load_checkpoint(checkpoint)
    verify_prefix(t,m,t,m);assert len(t)==10 and t[-1]==ELL9
    old=json.loads((SOURCE/'report.json').read_text())
    runner=Path('examples/lisa_dns_stage4/run_ladder_batch.py')
    assert hashlib.sha256(runner.read_bytes()).hexdigest()==old['runner_sha256']
    assert old['rows'][-1]['level']==10 and not old['rows'][-1]['accepted']
    assert old['rows'][-1]['seeds']==schedule(10)
    assert metadata['prng']['next_level_seeds']==schedule(10)
    verify_prefix([float(x) for x in old['frozen_thresholds']],old['frozen_log_mass'],t,m)
    old_retained={name:retained_bank(load(SOURCE/f'level9_{name}_trace.npz')) for name in NAMES}
    regenerated,indices=promoted_states(old_retained,t[-1],9)
    assert indices==metadata['provenance']['promotion_indices']
    recovered=recover_starts(states,old_retained,indices)
    for name in NAMES:
        for k in recovered[name]:assert recovered[name][k].tobytes()==regenerated[name][k].tobytes()
    check_banks(recovered,t[-1])
    assert len(set(sum(SEEDS.values(),[])))==16
    (OUT/'checkpoint_level9.json').write_bytes(checkpoint.read_bytes())
    assert (OUT/'checkpoint_level9.json').read_bytes()==checkpoint.read_bytes()
    for name in NAMES:np.savez_compressed(OUT/f'{name}_initial_states.npz',**recovered[name])
    # Calibration genealogies persist from these starts to Stage-4H starts. No relabelling.
    hstart=load('/tmp/lisa_dns_stage4h_continuation/initial_states.npz')
    oldcal=load(SOURCE/'level10_calibration_trace.npz')
    for k in hstart:assert hstart[k].tobytes()==oldcal[k].reshape((8,384)+oldcal[k].shape[1:])[:,-1].tobytes()
    o_audit=Path('/tmp/lisa_dns_stage4o_hybrid/bottleneck_audit.json')
    sticky=json.loads(o_audit.read_text())
    mapping=dict(selection=[None]*8,calibration=[r['label'] for r in sticky['rows']])
    paths=[checkpoint,SOURCE/'report.json',runner,o_audit,
        Path('/tmp/lisa_dns_stage4h_continuation/initial_states.npz'),
        *[SOURCE/f'level9_{n}_trace.npz' for n in NAMES],
        *[SOURCE/f'level10_{n}_trace.npz' for n in NAMES]]
    proof=dict(status='exact_historical_starts_recovered',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        promotion_indices=indices,exact_states_and_caches=True,distinct_banks=True,
        historical_runner_source_verified=True,settings=SETTINGS,seeds=SEEDS,
        frozen_thresholds=t,frozen_log_masses=m,sticky_mapping=mapping,
        sticky_mapping_notice='Calibration: exact labelled genealogy to failed isotropic endpoints used as Stage-4H/4N starts. Selection: independent bank, no established Stage-4N/4O source mapping; all nine blocks are diagnosed without inventing identities.',
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'preflight.json').write_text(json.dumps(safe(proof),indent=2,allow_nan=False))
    print('Exact starts recovered for both historical banks; log X9 =',m[-1],flush=True)
    return t,m,recovered,metadata,proof


def main():
    from examples.lisa_dns_stage4 import audit
    t,m,starts,metadata,proof=preflight()
    problem,cfg,config_path=audit.load_problem();assert cfg==metadata['provenance']['config']
    model=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
    for p in [model,config_path,Path(proposals.__file__)]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==metadata['provenance']['file_sha256'][str(p)]
    paths=[model,config_path,Path(proposals.__file__),Path(__file__),
        Path('examples/lisa_dns_stage4/hybrid_logistic_refresh.py'),
        Path('examples/lisa_dns_stage4/sequential_logistic_refresh.py'),
        Path('examples/lisa_dns_stage4/run_ladder_extension.py'),Path('blackjax/mcmc/ss.py'),
        *Path('blackjax/ns').glob('dns*.py')]
    hashes={**proof['file_sha256'],**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    step=jax.jit(build_frozen_slice(prior,like))
    b=starts['selection'];p0=DNSParticleState(jnp.asarray(b['position'][0]),jnp.asarray(b['logprior'][0]),jnp.asarray(b['loglikelihood'][0]))
    step=step.lower(jax.random.key(0),p0,jnp.asarray(t[-1])).compile()
    report=dict(status='running_single_level10_attempt',settings=SETTINGS,seeds=SEEDS,
        frozen_thresholds=t,frozen_log_masses=m,ell9=float(t[-1]),logX9=float(m[-1]),
        file_sha256=hashes,failures=0,banks={},sticky_mapping=proof['sticky_mapping'],
        devices=str(jax.devices()),attempted_levels=[10],production=False,evidence=False,
        existing_gate='selection.status == ok and 0 < calibration.ratio < 1 and calibration.ess >= 20',
        notice=audit.NOTICE)
    def save(): (OUT/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    save();traces={};candidate=None
    try:
        for name in NAMES:
            records=[];costs=dict(initial_prior=0,initial_likelihood=0);errors=np.zeros(2);times=[]
            for w in range(8):
                bank=starts[name]
                current=State(bank['position'][w].copy(),float(bank['logprior'][w]),float(bank['loglikelihood'][w]))
                errors=np.maximum(errors,validate(current,prior,like));costs['initial_prior']+=1;costs['initial_likelihood']+=1
                slice_key,refresh_key=jax.random.split(jax.random.key(SEEDS[name][w]))
                categories,labels,draws,uniforms=random_schedule(refresh_key,2,count=384)
                timer=time.perf_counter()
                for i in range(384):
                    slice_key,subkey=jax.random.split(slice_key)
                    before=current;before_bytes=before.position.tobytes()
                    def do_slice(state):
                        particle=DNSParticleState(jnp.asarray(state.position),jnp.asarray(state.logprior),jnp.asarray(state.loglikelihood))
                        new,info=step(subkey,particle,jnp.asarray(t[-1]))
                        assert bool(info.slice.is_accepted), 'Slice failure: STOP'
                        return State(np.asarray(new.position),float(new.logdensity),float(new.loglikelihood)),info
                    def do_refresh(state):return transition(state,labels[i],draws[i],float(uniforms[i]),prior,like)
                    def check(state):
                        nonlocal errors
                        errors=np.maximum(errors,validate(state,prior,like))
                    current,post,sinfo,rinfo=hybrid_sweep(current,do_slice,do_refresh,check)
                    assert before.position.tobytes()==before_bytes
                    if not rinfo['accepted']:assert current is post
                    else:
                        assert current.position.tobytes()==rinfo['proposed_position'].tobytes()
                        assert current.logprior==rinfo['proposed_logprior'] and current.loglikelihood==rinfo['proposed_loglikelihood']
                    cost=sweep_cost(int(sinfo.slice.num_steps),int(sinfo.slice.num_shrink),not rinfo['nonfinite_draw'])
                    for k,v in cost.items():costs[k]=costs.get(k,0)+v
                    records.append(dict(walker=w,iteration=i,retained=i>=128,category=int(categories[i]),
                        position=current.position.copy(),logprior=current.logprior,loglikelihood=current.loglikelihood,
                        post_slice_position=post.position.copy(),post_slice_logprior=post.logprior,
                        post_slice_loglikelihood=post.loglikelihood,slice_component=int(sinfo.component),
                        slice_labels=np.asarray(sinfo.labels),num_steps=int(sinfo.slice.num_steps),num_shrink=int(sinfo.slice.num_shrink),
                        **rinfo,**cost))
                times.append(time.perf_counter()-timer)
                raw={k:np.asarray([r[k] for r in records]) for k in records[0]}
                np.savez_compressed(OUT/f'level10_{name}_trace.npz',**raw)
                report['banks'][name]=dict(completed_walkers=w+1,cost=costs,max_cache_absolute_error=errors,walker_seconds=times)
                verify_prefix(t,m,proof['frozen_thresholds'],proof['frozen_log_masses'])
                save();print('Hybrid-P level10',name,'walker',w,'completed 128+256 sweeps',flush=True)
            traces[name]=retained_bank(raw)
            costs['including_initial_likelihood_total_proxy']=costs['likelihood_total_proxy']+8
            costs['including_initial_prior_total_proxy']=costs['prior_total_proxy']+8
            if name=='selection':
                candidate=build_next_level(traces[name]['loglikelihood'].T,t[-1],
                    target_compression=SETTINGS['target_compression'],block_size=SETTINGS['block_size'],min_ess=SETTINGS['min_ess'])
                (OUT/'selection_candidate.json').write_text(json.dumps(safe(candidate),indent=2,allow_nan=False))
                report['candidate_selected_before_calibration']=candidate;save()
                print('Selection-only candidate',candidate['threshold'],'selection ESS',candidate['ess'],flush=True)
        # No overlap of retained states across distinct genealogies/streams.
        assert not ({z.tobytes() for z in traces['selection']['position'].reshape(-1,54)} &
                    {z.tobytes() for z in traces['calibration']['position'].reshape(-1,54)})
        row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],t[-1],m[-1])
        assert safe(row['selection'])==safe(candidate)
        row['level']=10;report['decision']=row
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        assert (OUT/'checkpoint_level9.json').read_bytes()==(SOURCE/'checkpoint_level9.json').read_bytes()
        nt,nm,path=freeze_level10(OUT,t,m,row,traces,metadata,
            {**metadata['provenance'],'parent_checkpoint':str(SOURCE/'checkpoint_level9.json'),
             'hybrid_pair_construction':str(OUT),'hybrid_pair_seeds':SEEDS,'file_sha256':hashes})
        report.update(status='frozen_level10' if path else 'stopped_failed_level10_gate',
            frozen_thresholds=nt,frozen_log_masses=nm,checkpoint_level10=str(path) if path else None,
            frozen_prefix_unchanged=True,independent_banks_verified=True,protected_hashes_unchanged=True)
        np.savez_compressed(OUT/'levels.npz',thresholds=nt,log_mass=nm)
    except Exception as exc:
        report.update(status='STOP_structural_or_execution_failure',failures=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()
    print('DECISION',report['status'],'calibration ESS',row['calibration']['ess'],flush=True)


if __name__=='__main__':main()
