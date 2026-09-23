"""Stage-4O deterministic slice-then-refresh composition, benchmark only.

Both kernels preserve P = pi_impl * I(logL > ell9). For bounded f,
P K_slice K_refresh f = P K_refresh f = P f. In function-composition
notation this is K_refresh o K_slice. Reversibility of the product is neither
asserted nor needed. The constituent implementations are imported unchanged.
"""
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales
from examples.lisa_dns_stage4.sequential_logistic_refresh import State, transition, random_schedule, validate
from examples.lisa_dns_stage4.logistic_mh_refresh import ELL9, ELL10, MATERIAL, safe
from examples.lisa_dns_stage4.prior_refresh_audit import MODEL
from examples.lisa_dns_stage4.compare_sequential_refresh import load

OUT=Path('/tmp/lisa_dns_stage4o_hybrid')


def build_frozen_slice(prior,likelihood):
    scales=frozen_scales('isotropic')
    return proposals.build_parameter_step(prior,likelihood,scales,max_steps=10,max_shrinkage=100)


def hybrid_sweep(current,slice_step,refresh_step,check):
    """Exact ordering; check both intermediate and final states."""
    post_slice,slice_info=slice_step(current)
    check(post_slice)
    final,refresh_info=refresh_step(post_slice)
    check(final)
    return final,post_slice,slice_info,refresh_info


def sweep_cost(num_steps,num_shrink,refresh_evaluated=True):
    """Baseline-compatible likelihood proxy, plus counted refresh/audit calls.

    Each slice loop body calls prior and likelihood once. The saved loop counts
    are retained as proxies, consistent with earlier stages; no timing claims.
    Initial state checks are accounted for separately (one per walker).
    """
    if num_steps<0 or num_shrink<0: raise ValueError('Nonnegative counts required')
    return dict(slice_likelihood_proxy=int(num_steps+num_shrink),
        slice_prior_proxy=int(num_steps+num_shrink),refresh_likelihood=int(refresh_evaluated),
        refresh_prior=int(refresh_evaluated),cache_likelihood=2,cache_prior=2,
        likelihood_total_proxy=int(num_steps+num_shrink)+int(refresh_evaluated)+2,
        prior_total_proxy=int(num_steps+num_shrink)+int(refresh_evaluated)+2)


def cost_matched_prefix(costs,budget,initial_cost=0):
    """Largest common time prefix across walkers, chosen only by pooled cost."""
    c=np.asarray(costs)
    if c.ndim!=2 or not np.isfinite(c).all() or np.any(c<0) or budget<0 or initial_cost<0:
        raise ValueError('Finite nonnegative walker/time costs and budget required')
    cumulative=initial_cost+np.cumsum(c.sum(axis=0))
    n=int(np.sum(cumulative<=budget))
    return dict(sweeps_per_walker=n,cost=float(initial_cost if n==0 else cumulative[n-1]),
                next_cost=float(cumulative[n]) if n<c.shape[1] else None,budget=float(budget))


def main():
    from examples.lisa_dns_stage4 import audit
    audit_path=OUT/'bottleneck_audit.json'
    bottleneck=json.loads(audit_path.read_text())
    assert bottleneck['status']=='saved_trace_audit_completed_before_sampling'
    assert not (OUT/'report.json').exists(), 'Never rerun or extend a completed benchmark'
    h=Path('/tmp/lisa_dns_stage4h_continuation');n=Path('/tmp/lisa_dns_stage4n_sequential')
    starts=load(h/'initial_states.npz')
    for path in [n/'initial_states.npz',n/'single/initial_states.npz',n/'pair/initial_states.npz']:
        saved=load(path)
        for k in starts: assert starts[k].tobytes()==saved[k].tobytes()
    np.savez_compressed(OUT/'initial_states.npz',**starts)
    copied=load(OUT/'initial_states.npz')
    for k in starts: assert copied[k].tobytes()==starts[k].tobytes()
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    payload=json.loads(checkpoint.read_text())['payload'];assert payload['thresholds'][-1]==ELL9
    problem,cfg,config_path=audit.load_problem();assert cfg==payload['provenance']['config']
    for p in [MODEL,config_path]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['provenance']['file_sha256'][str(p)]
    protected=[MODEL,config_path,checkpoint,h/'initial_states.npz',h/'trace.npz',h/'report.json',
        n/'report.json',n/'single/trace.npz',n/'pair/trace.npz',audit_path,Path(__file__),
        Path(proposals.__file__),Path('examples/lisa_dns_stage4/sequential_logistic_refresh.py'),
        Path('examples/lisa_dns_stage4/logistic_mh_refresh.py'),Path('blackjax/mcmc/ss.py'),
        *Path('blackjax/ns').glob('dns*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    slice_step=jax.jit(build_frozen_slice(prior,like))
    p0=DNSParticleState(jnp.asarray(starts['position'][0]),jnp.asarray(starts['logprior'][0]),jnp.asarray(starts['loglikelihood'][0]))
    timer=time.perf_counter()
    slice_step=slice_step.lower(jax.random.key(0),p0,jnp.asarray(ELL9)).compile()
    compile_seconds=time.perf_counter()-timer
    report=dict(status='running',ell9=ELL9,ell10=ELL10,level10_remains_rejected=True,
        starts_exact_byte_match=True,walkers=8,sweeps_per_walker=512,burn_in_removed=0,
        ordering='isotropic_slice_then_exact_refresh',slice_scale=float(np.pi/np.sqrt(3)),
        slice_mixture=[.2,.4,.4],max_steps=10,max_shrinkage=100,devices=str(jax.devices()),
        compilation_seconds=compile_seconds,file_sha256=hashes,failures=0,kernels={},
        material_correction_threshold=MATERIAL,
        cost_prefix_rule='Largest common sweep prefix across all eight walkers with pooled cumulative slice proxy + actual refresh + both direct cache checks, plus eight initial checks, <= baseline 27342+4104. Also report proposal-only prefix <=27342.',
        classification_policy='A/B/C/D relative mainly to Stage-4H cross-walker likelihood-mean spread, ell10 exceedance spread and pairwise KS; assess concordance, halves and cost-matched prefix. No probability tuning or ESS-only decision.')
    def save(): (OUT/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    save()
    try:
        for size,name,seed in [(1,'hybrid_single',870100),(2,'hybrid_pair',870200)]:
            folder=OUT/name;folder.mkdir(exist_ok=False);np.savez_compressed(folder/'initial_states.npz',**starts)
            rows=[];timings=[];error=np.zeros(2)
            costs=dict(initial_likelihood=0,initial_prior=0)
            for w in range(8):
                current=State(starts['position'][w].copy(),float(starts['logprior'][w]),float(starts['loglikelihood'][w]))
                error=np.maximum(error,validate(current,prior,like));costs['initial_likelihood']+=1;costs['initial_prior']+=1
                slice_key,refresh_key=jax.random.split(jax.random.key(seed+w))
                categories,labels,draws,uniforms=random_schedule(refresh_key,size)
                timer=time.perf_counter()
                for i in range(512):
                    slice_key,subkey=jax.random.split(slice_key)
                    before=current;before_bytes=before.position.tobytes()
                    def do_slice(state):
                        p=DNSParticleState(jnp.asarray(state.position),jnp.asarray(state.logprior),jnp.asarray(state.loglikelihood))
                        value,info=slice_step(subkey,p,jnp.asarray(ELL9))
                        assert bool(info.slice.is_accepted), 'Slice transition failure: STOP'
                        return State(np.asarray(value.position),float(value.logdensity),float(value.loglikelihood)),info
                    def do_refresh(state):
                        return transition(state,labels[i],draws[i],float(uniforms[i]),prior,like)
                    def check(state):
                        nonlocal error
                        error=np.maximum(error,validate(state,prior,like))
                    current,post_slice,sinfo,rinfo=hybrid_sweep(current,do_slice,do_refresh,check)
                    assert before.position.tobytes()==before_bytes
                    if not rinfo['accepted']: assert current is post_slice
                    else:
                        assert current.position.tobytes()==rinfo['proposed_position'].tobytes()
                        assert current.logprior==rinfo['proposed_logprior'] and current.loglikelihood==rinfo['proposed_loglikelihood']
                    cost=sweep_cost(int(sinfo.slice.num_steps),int(sinfo.slice.num_shrink),not rinfo['nonfinite_draw'])
                    for k,v in cost.items():costs[k]=costs.get(k,0)+v
                    rows.append(dict(walker=w,iteration=i,category=int(categories[i]),position=current.position.copy(),
                        logprior=current.logprior,loglikelihood=current.loglikelihood,
                        post_slice_position=post_slice.position.copy(),post_slice_logprior=post_slice.logprior,
                        post_slice_loglikelihood=post_slice.loglikelihood,slice_component=int(sinfo.component),
                        slice_labels=np.asarray(sinfo.labels),num_steps=int(sinfo.slice.num_steps),
                        num_shrink=int(sinfo.slice.num_shrink),**rinfo,**cost))
                timings.append(time.perf_counter()-timer)
                np.savez_compressed(folder/'trace.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
                report['kernels'][name]=dict(completed_walkers=w+1,seeds=list(range(seed,seed+8)),cost=costs,
                    walker_seconds=timings,max_cache_absolute_error=error)
                save();print(name,'walker',w,'completed 512 sweeps',flush=True)
            costs['including_initial_likelihood_total_proxy']=costs['likelihood_total_proxy']+costs['initial_likelihood']
            costs['including_initial_prior_total_proxy']=costs['prior_total_proxy']+costs['initial_prior']
        for p,digest in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
        report.update(status='completed',protected_hashes_unchanged=True)
    except Exception as exc:
        report.update(status='STOP_structural_or_execution_failure',failures=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()
    print('Completed both frozen hybrids:',OUT,flush=True)


if __name__=='__main__':main()
