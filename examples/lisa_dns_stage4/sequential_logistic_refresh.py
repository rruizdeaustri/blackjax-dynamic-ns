"""Stage-4N standalone sequential refresh benchmark, never a DNS kernel."""
import hashlib
import json
import time
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from examples.lisa_dns_stage4.logistic_mh_refresh import (
    ELL9, ELL10, MATERIAL, design, fixed_proposal, logq, mh, safe)
from examples.lisa_dns_stage4.prior_refresh_audit import MODEL, block_indices


class State(NamedTuple):
    position: np.ndarray
    logprior: float
    loglikelihood: float


def random_schedule(key, size, count=512):
    """Separate selection/draw and MH-uniform streams; no state dependency."""
    proposal_key, uniform_key = jax.random.split(key)
    category, labels, draws = design(proposal_key, size, count)
    uniforms = np.asarray(jax.random.uniform(uniform_key, (count,), dtype=jnp.float64))
    return category, labels, draws, uniforms


def transition(current, labels, draw, uniform, prior, likelihood):
    """Propose from CURRENT state; return it unchanged on rejection."""
    if not 0 <= uniform < 1:
        raise ValueError('MH uniform must lie in [0,1)')
    if not (np.isfinite(current.logprior) and np.isfinite(current.loglikelihood)
            and current.loglikelihood > ELL9 and np.isfinite(current.position).all()):
        raise ValueError('Invalid current state')
    proposed = fixed_proposal(current.position, labels, draw)
    indices = block_indices(labels)
    keep = np.setdiff1d(np.arange(54), indices)
    assert proposed[keep].tobytes() == current.position[keep].tobytes()
    np.testing.assert_array_equal(proposed[indices], np.asarray(draw).ravel())
    finite = bool(np.isfinite(proposed).all())
    lp = float(prior(proposed)) if finite else np.nan
    ll = float(likelihood(proposed)) if finite else np.nan
    qold = logq(current.position[indices]); qnew = logq(proposed[indices])
    ratio, alpha = mh(current.logprior, lp, qold, qnew, ll, finite)
    accepted = bool(uniform < alpha)
    result = State(proposed, lp, ll) if accepted else current
    if not accepted:
        assert result is current
    info = dict(labels=np.asarray(labels),draw=np.asarray(draw),uniform=uniform,
        proposed_position=proposed,proposed_logprior=lp,proposed_loglikelihood=ll,
        old_logprior=current.logprior,old_loglikelihood=current.loglikelihood,
        logq_old=qold,logq_new=qnew,log_mh_ratio=ratio,alpha=alpha,accepted=accepted,
        passes_ell9=bool(finite and np.isfinite(ll) and ll>ELL9),
        nonfinite_draw=not finite,nonfinite_draw_coordinates=int(np.sum(~np.isfinite(draw))))
    return result, info


def validate(current, prior, likelihood):
    """Direct cache checks on every retained current state, including rejections."""
    assert current.position.shape == (54,) and np.isfinite(current.position).all()
    assert np.isfinite(current.logprior) and np.isfinite(current.loglikelihood)
    assert current.loglikelihood > ELL9
    lp, ll = float(prior(current.position)), float(likelihood(current.position))
    np.testing.assert_allclose(lp,current.logprior,rtol=0,atol=1e-9)
    np.testing.assert_allclose(ll,current.loglikelihood,rtol=0,atol=1e-7)
    return abs(lp-current.logprior), abs(ll-current.loglikelihood)


def main():
    from examples.lisa_dns_stage4 import audit
    from examples.lisa_dns_stage4.diagnose_ladder_failure import load_trace
    out=Path('/tmp/lisa_dns_stage4n_sequential');out.mkdir(exist_ok=False)
    baseline=Path('/tmp/lisa_dns_stage4h_continuation')
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    predecessor=Path('/tmp/lisa_dns_stage4_ladder_batch/level10_calibration_trace.npz')
    payload=json.loads(checkpoint.read_text())['payload'];assert payload['thresholds'][-1]==ELL9
    baseline_report=json.loads((baseline/'report.json').read_text())
    assert baseline_report['ell9']==ELL9 and baseline_report['ell10']==ELL10
    with np.load(baseline/'initial_states.npz') as stored:
        starts={k:stored[k].copy() for k in stored.files}
    previous=load_trace(predecessor)
    for k in ['position','logprior','loglikelihood']:
        assert starts[k].tobytes()==previous[k][:,-1].tobytes()
    np.savez_compressed(out/'initial_states.npz',**starts)
    with np.load(out/'initial_states.npz') as copied:
        for k in starts: assert copied[k].tobytes()==starts[k].tobytes()
    problem,cfg,config_path=audit.load_problem()
    assert cfg==payload['provenance']['config']
    for p in [MODEL,config_path]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['provenance']['file_sha256'][str(p)]
    paths=[MODEL,config_path,checkpoint,predecessor,baseline/'initial_states.npz',
           baseline/'trace.npz',baseline/'report.json',Path(__file__),
           Path('examples/lisa_dns_stage4/proposals.py'),
           Path('examples/lisa_dns_stage4/logistic_mh_refresh.py'),*Path('blackjax/ns').glob('dns*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    report=dict(status='running',ell9=ELL9,ell10=ELL10,level10_remains_rejected=True,
        walkers=8,transitions_per_walker=512,burn_in_removed=0,baseline_rerun=False,
        starts_exact_byte_match=True,devices=str(jax.devices()),jax_version=jax.__version__,
        file_sha256=hashes,failures=0,kernels={},material_correction_threshold=MATERIAL,
        classification_policy='Qualitative A/B/C/D assessment mainly from cross-walker likelihood and exceedance agreement, autocorrelation, source-set exploration and cost. Acceptance alone does not determine classification.',
        nonfinite_draw_policy='Record and reject without resampling or model calls.',
        timing_notice='CPU elapsed measurements include Python and integrity-check overhead. Baseline used GPU and lacks comparable timing; no wall-time speedup claim.')
    def save(): (out/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    save()
    try:
        for size,name,seed in [(1,'single',860100),(2,'pair',860200)]:
            folder=out/name;folder.mkdir()
            np.savez_compressed(folder/'initial_states.npz',**starts)
            rows=[]
            cost=dict(proposal_prior=0,proposal_likelihood=0,initial_prior=0,initial_likelihood=0,
                      cache_prior=0,cache_likelihood=0)
            timing=[];max_error=np.zeros(2)
            for w in range(8):
                start=State(starts['position'][w].copy(),float(starts['logprior'][w]),float(starts['loglikelihood'][w]))
                cost['initial_prior']+=1;cost['initial_likelihood']+=1
                max_error=np.maximum(max_error,validate(start,prior,like))
                category,labels,draws,uniforms=random_schedule(jax.random.key(seed+w),size)
                current=start;timer=time.perf_counter()
                for i in range(512):
                    old=current;old_bytes=old.position.tobytes()
                    current,info=transition(old,labels[i],draws[i],float(uniforms[i]),prior,like)
                    evaluated=int(not info['nonfinite_draw'])
                    cost['proposal_prior']+=evaluated;cost['proposal_likelihood']+=evaluated
                    assert old.position.tobytes()==old_bytes
                    if not info['accepted']:
                        assert current is old
                    else:
                        assert current.position.tobytes()==info['proposed_position'].tobytes()
                        assert current.logprior==info['proposed_logprior']
                        assert current.loglikelihood==info['proposed_loglikelihood']
                    cost['cache_prior']+=1;cost['cache_likelihood']+=1
                    max_error=np.maximum(max_error,validate(current,prior,like))
                    rows.append(dict(walker=w,iteration=i,category=int(category[i]),
                        position=current.position.copy(),logprior=current.logprior,
                        loglikelihood=current.loglikelihood,**info))
                timing.append(time.perf_counter()-timer)
                np.savez_compressed(folder/'trace.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
                report['kernels'][name]=dict(completed_walkers=w+1,seeds=list(range(seed,seed+8)),
                    cost=cost,walker_seconds=timing,max_cache_absolute_error=max_error)
                save();print(name,'walker',w,'completed 512 transitions',flush=True)
            cost['total_prior']=sum(cost[k] for k in ['proposal_prior','initial_prior','cache_prior'])
            cost['total_likelihood']=sum(cost[k] for k in ['proposal_likelihood','initial_likelihood','cache_likelihood'])
        for p,digest in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
        report.update(status='completed',protected_hashes_unchanged=True)
    except Exception as exc:
        report.update(status='STOP_structural_or_execution_failure',failures=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print('Completed standalone refresh benchmarks:',out,flush=True)


if __name__=='__main__': main()
