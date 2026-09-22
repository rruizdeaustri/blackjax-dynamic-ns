"""Stage-4J frozen differential orientation benchmark; no construction."""
import json
import hashlib
from pathlib import Path
from examples.lisa_dns_stage4 import audit,proposals
from examples.lisa_dns_stage4 import differential_directions as differential
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales
from examples.lisa_dns_stage4.diagnose_ladder_failure import (
    load_trace,block_diagnostics,early_late,write_blocks,serial)


def main():
    source=Path('/tmp/lisa_dns_stage4_ladder_batch')
    output=Path('/tmp/lisa_dns_stage4j_differential');output.mkdir(exist_ok=False)
    baseline=Path('/tmp/lisa_dns_stage4h_continuation')
    old=json.loads((source/'report.json').read_text())
    ell9=old['frozen_thresholds'][-1];ell10=old['rows'][-1]['threshold']
    assert old['rows'][-1]['level']==10 and not old['rows'][-1]['accepted']
    trace=load_trace(source/'level10_calibration_trace.npz')
    report=dict(status='initializing',ell9=ell9,ell10=ell10,level10_remains_rejected=True,
        steps_per_walker=512,walkers=8,discarded_steps=0,halves=[256,256],seeds=list(range(810100,810108)),
        scale=float(np.pi/np.sqrt(3)),mixture=[.2,.4,.4],block_size=32,
        failures={k:0 for k in ['contour','cache','nonfinite','parameter_contract','slice']},
        cost=dict(likelihood_proxy=0,direct_cache_checks=0,initialization_likelihood_evaluations=0),
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'report.json',source/'level10_calibration_trace.npz',Path(__file__),Path(proposals.__file__)]})
    reference_trace=load_trace(source/'level9_selection_trace.npz')
    candidate=reference_trace['position'][:,128:].reshape(-1,54)
    candidate_ll=reference_trace['loglikelihood'][:,128:].ravel()
    retained_indices=np.flatnonzero(candidate_ll>ell9)
    with np.load(baseline/'initial_states.npz') as initial:
        for k in trace:
            np.testing.assert_array_equal(initial[k],trace[k][:,-1])
            assert initial[k].tobytes()==trace[k][:,-1].tobytes()
    reference=differential.freeze_reference(candidate[retained_indices],trace['position'][:,-1])
    original_reference=reference.copy()
    report.update(reference_count=len(reference),reference_retained_indices=retained_indices,
        reference_source='Stage-4G level9 selection retained survivors strictly above ell9; no calibration states',
        starts_byte_equal=True,baseline_rerun=False,common_random_numbers=True,
        fallback_count=0,max_direction_norm_error=0.,max_steps=10,max_shrinkage=100)
    for p in [source/'level9_selection_trace.npz',baseline/'report.json',baseline/'trace.npz',baseline/'initial_states.npz',Path(differential.__file__)]:
        report['file_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    baseline_report=json.loads((baseline/'report.json').read_text())
    assert baseline_report['status']=='completed_fixed_contour_diagnostic'
    assert not any(baseline_report['failures'].values())
    assert baseline_report['seeds']==report['seeds'] and baseline_report['ell9']==ell9 and baseline_report['ell10']==ell10
    np.savez(output/'reference_bank.npz',position=reference,retained_indices=retained_indices)
    records=[]
    def save():
        (output/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
        if records:
            np.savez_compressed(output/'trace.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]})
    try:
        provenance=json.loads((source/'checkpoint_level9.json').read_text())['payload']['provenance']
        problem,cfg,config_path=audit.load_problem();assert cfg==provenance['config']
        model=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        for p in [config_path,model,Path(proposals.__file__)]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['file_sha256'][str(p)]
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        step=jax.jit(differential.build_parameter_step(prior,like,reference,trace['position'][:,-1]))
        report['devices']=str(jax.devices())
        def validate(p):
            x=np.asarray(p.position);lp=np.asarray(p.logdensity);ll=np.asarray(p.loglikelihood)
            finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(lp)) and np.all(np.isfinite(ll)))
            contour=bool(np.all(ll>ell9));contract=x.shape==(54,) and lp.shape==() and ll.shape==() and finite and contour
            for name,ok in [('nonfinite',finite),('contour',contour),('parameter_contract',contract)]:
                report['failures'][name]+=int(not ok)
            if not contract: raise ValueError('Invalid constrained parameter state')
            a=prior(p.position);b=like(p.position);report['cost']['direct_cache_checks']+=1
            if not (np.isclose(lp,a,rtol=0,atol=1e-9) and np.isclose(ll,b,rtol=0,atol=1e-7)):
                report['failures']['cache']+=1;raise ValueError('Direct cache failure')
        np.savez(output/'initial_states.npz',**{k:v[:,-1] for k,v in trace.items()})
        save()
        for w in range(8):
            p=DNSParticleState(jnp.asarray(trace['position'][w,-1]),jnp.asarray(trace['logprior'][w,-1]),jnp.asarray(trace['loglikelihood'][w,-1]))
            validate(p);key=jax.random.key(report['seeds'][w])
            for i in range(512):
                key,subkey=jax.random.split(key);p,info=step(subkey,p,jnp.asarray(ell9))
                report['cost']['likelihood_proxy']+=int(info.slice.num_steps+info.slice.num_shrink)
                validate(p)
                if not bool(info.slice.is_accepted):
                    report['failures']['slice']+=1;raise ValueError('Slice failure')
                report['fallback_count']+=int(info.fallback)
                norm_error=abs(float(info.direction_norm)-differential.NORM)
                report['max_direction_norm_error']=max(report['max_direction_norm_error'],norm_error)
                if norm_error>1e-10:
                    report['failures']['parameter_contract']+=1;raise ValueError('Direction norm mismatch')
                records.append(dict(walker=w,iteration=i,position=np.asarray(p.position),logprior=float(p.logdensity),
                    loglikelihood=float(p.loglikelihood),num_steps=int(info.slice.num_steps),num_shrink=int(info.slice.num_shrink),
                    component=int(info.component),labels=np.asarray(info.labels),reference_pair=np.asarray(info.reference_pair),fallback=bool(info.fallback),direction_norm=float(info.direction_norm)))
            report['completed_walkers']=w+1;save();print('Diagnostic walker',w,'completed 512 steps',flush=True)
        ll=np.array([r['loglikelihood'] for r in records]).reshape(8,512)
        f=np.array([r['position'] for r in records]).reshape(8,512,54)[:,:,::6]
        bounds=(problem.f_min_cfg,problem.f_max_cfg)
        report['blocks']=block_diagnostics(ll,f,ell10,physical_bounds=bounds)
        report['comparison']=early_late(ll,f,ell10,burn_in=0,physical_bounds=bounds)
        write_blocks(output/'blocks.csv',report['blocks'],burn_in=0)
        np.testing.assert_array_equal(reference,original_reference)
        with np.load(baseline/'trace.npz') as previous:
            for k in ['component','labels']:
                np.testing.assert_array_equal(previous[k],np.asarray([r[k] for r in records]))
        report['identical_component_masks_verified']=True
        report['status']='completed_frozen_differential_benchmark'
    except Exception as exc:
        report.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        save();print('Report:',output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
