"""Exactly 512 fixed-contour diagnostic steps; never builds or accepts a level."""
import json
import hashlib
from pathlib import Path
from examples.lisa_dns_stage4 import audit,proposals
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales
from examples.lisa_dns_stage4.diagnose_ladder_failure import (
    load_trace,block_diagnostics,early_late,write_blocks,serial)


def main():
    source=Path('/tmp/lisa_dns_stage4_ladder_batch')
    output=Path('/tmp/lisa_dns_stage4h_continuation');output.mkdir(exist_ok=False)
    decision=json.loads(Path('/tmp/lisa_dns_stage4h_diagnosis/decision_before_continuation.json').read_text())
    assert decision['classification'].startswith('C:')
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
        step=jax.jit(proposals.build_parameter_step(prior,like,frozen_scales('isotropic')))
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
                records.append(dict(walker=w,iteration=i,position=np.asarray(p.position),logprior=float(p.logdensity),
                    loglikelihood=float(p.loglikelihood),num_steps=int(info.slice.num_steps),num_shrink=int(info.slice.num_shrink),
                    component=int(info.component),labels=np.asarray(info.labels)))
            report['completed_walkers']=w+1;save();print('Diagnostic walker',w,'completed 512 steps',flush=True)
        ll=np.array([r['loglikelihood'] for r in records]).reshape(8,512)
        f=np.array([r['position'] for r in records]).reshape(8,512,54)[:,:,::6]
        bounds=(problem.f_min_cfg,problem.f_max_cfg)
        report['blocks']=block_diagnostics(ll,f,ell10,physical_bounds=bounds)
        report['comparison']=early_late(ll,f,ell10,burn_in=0,physical_bounds=bounds)
        write_blocks(output/'blocks.csv',report['blocks'],burn_in=0)
        report['status']='completed_fixed_contour_diagnostic'
    except Exception as exc:
        report.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        save();print('Report:',output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
