"""Resumable fixed-budget Stage-4G construction, with no production path."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from examples.lisa_dns_stage4 import audit, proposals
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4.run_ladder_extension import SETTINGS, assess, promote
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales, autocorrelation_information
from examples.lisa_dns_stage4.run_smoke import json_safe

BASE_THRESHOLDS = (-np.inf, -113725.65889538352, -113448.99632193986,
                   -113172.84824147604, -112838.36560726895)
BASE_MASSES = (0., -1.0343179379627125, -1.9996430044384734,
               -3.118477930469643, -4.172213954289457)
NAMES = ('selection', 'calibration')


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def schedule(level):
    return [[700000+level*100+b*10+w for w in range(8)] for b in range(2)]


def check_banks(states, threshold):
    if set(states)!=set(NAMES):
        raise ValueError('Two named banks required')
    rows=[]
    for name in NAMES:
        bank=states[name]
        for key,shape in [('position',(8,54)),('logprior',(8,)),('loglikelihood',(8,))]:
            a=np.asarray(bank[key])
            if a.shape!=shape or not np.all(np.isfinite(a)):
                raise ValueError('Invalid checkpoint state')
        if np.any(np.asarray(bank['loglikelihood'])<=threshold):
            raise ValueError('Checkpoint outside strict contour')
        rows.append({r.tobytes() for r in np.asarray(bank['position'],dtype=np.float64)})
    if rows[0]&rows[1] or any(len(r)!=8 for r in rows):
        raise ValueError('Banks must be separate and walkers distinct')


def save_checkpoint(path, thresholds, masses, states, diagnostics, provenance):
    """Atomic, append-only checkpoint; integrity digest covers states and metadata."""
    path=Path(path)
    if path.exists():
        raise ValueError('Accepted checkpoint cannot be overwritten')
    check_banks(states,thresholds[-1])
    payload=dict(version=1,thresholds=list(thresholds),log_masses=list(masses),settings=SETTINGS,
                 diagnostics=diagnostics,provenance=provenance,
                 prng=dict(implementation='jax threefry2x32',next_level_seeds=schedule(len(thresholds)),
                           promotion_rule='NumPy default_rng, seed 800000 + accepted_level*10 + bank; one own-walker survivor'),
                 states={n:{k:np.asarray(v).tolist() for k,v in states[n].items()} for n in NAMES})
    wrapper=dict(payload=json_safe(payload),sha256=digest(payload))
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(wrapper,indent=2,allow_nan=False))
    temp.replace(path)


def load_checkpoint(path, expected_prefix=None):
    wrapper=json.loads(Path(path).read_text());p=wrapper['payload']
    if digest(p)!=wrapper['sha256'] or p['version']!=1 or p['settings']!=SETTINGS:
        raise ValueError('Checkpoint integrity, version, or settings mismatch')
    thresholds=np.array([float(x) for x in p['thresholds']]);masses=np.array(p['log_masses'])
    if len(thresholds)!=len(masses) or np.any(np.diff(thresholds)<=0) or np.any(np.diff(masses)>=0):
        raise ValueError('Invalid frozen ladder')
    if expected_prefix is not None:
        for a,b in zip((thresholds,masses),expected_prefix):
            if not np.array_equal(a[:len(b)],b): raise ValueError('Accepted lower levels changed')
    if p['prng']['next_level_seeds']!=schedule(len(thresholds)):
        raise ValueError('Unexpected PRNG schedule')
    states={n:{k:np.array(v,dtype=float) for k,v in p['states'][n].items()} for n in NAMES}
    check_banks(states,thresholds[-1])
    for a in [thresholds,masses,*[v for b in states.values() for v in b.values()]]:
        a.setflags(write=False)
    return thresholds,masses,states,p


def promoted_states(traces, threshold, level):
    states={};indices={}
    for b,name in enumerate(NAMES):
        trace=traces[name]
        position,ix=promote(trace['position'],trace['loglikelihood'],threshold,800000+level*10+b)
        indices[name]=ix
        states[name]=dict(position=position,
            logprior=trace['logprior'][np.arange(8),ix],
            loglikelihood=trace['loglikelihood'][np.arange(8),ix])
    check_banks(states,threshold)
    return states,indices


def import_stage4f(path, output):
    report=json.loads((path/'report.json').read_text())
    assert report['status']=='frozen_level4' and not any(report['failures'].values())
    np.testing.assert_array_equal([float(x) for x in report['frozen_thresholds']],BASE_THRESHOLDS)
    np.testing.assert_array_equal(report['frozen_log_mass'],BASE_MASSES)
    assert report['settings']==SETTINGS
    traces={}
    for name in NAMES:
        with np.load(path/f'level4_{name}_trace.npz') as t:
            np.testing.assert_array_equal(t['walker'],np.repeat(np.arange(8),384))
            traces[name]={k:t[k].reshape((8,384)+t[k].shape[1:])[:,128:] for k in ['position','logprior','loglikelihood']}
    states,indices=promoted_states(traces,BASE_THRESHOLDS[-1],4)
    files=[path/'report.json',*[path/f'level4_{n}_trace.npz' for n in NAMES]]
    provenance=dict(imported_stage4f=str(path),promoted_indices=indices,config=report['config'],
                    file_sha256=report['file_sha256'],
                    parent_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    save_checkpoint(output,BASE_THRESHOLDS,BASE_MASSES,states,report['rows'],provenance)


def bounded_levels(first,last,attempt):
    """No attempt after the first failed selection/calibration."""
    for level in range(first,last+1):
        row=attempt(level)
        yield row
        if not row['accepted']: break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage4f',type=Path,default=Path('/tmp/lisa_dns_stage4_ladder_extension'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/tmp/lisa_dns_stage4_ladder_batch'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='initializing',settings=SETTINGS,rows=[],notice=audit.NOTICE,
                failures={k:0 for k in ['contour','cache','nonfinite','parameter_contract','slice']},
                cost={k:0 for k in ['selection_likelihood_cost_proxy','calibration_likelihood_cost_proxy',
                                  'direct_cache_check_evaluations','initialization_evaluations']})
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    try:
        checkpoint=args.resume
        if checkpoint is None:
            checkpoint=args.output/'checkpoint_level4.json'
            import_stage4f(args.stage4f,checkpoint)
        thresholds,masses,states,metadata=load_checkpoint(checkpoint,(BASE_THRESHOLDS,BASE_MASSES))
        if not 5<=len(thresholds)<=10: raise ValueError('Resume must precede levels 5 through 10')
        initial_thresholds=thresholds.copy();initial_masses=masses.copy()
        thresholds=list(thresholds);masses=list(masses)
        diagnostics=list(metadata['diagnostics']);provenance=metadata['provenance']
        problem,cfg,config_path=audit.load_problem()
        assert cfg==provenance['config']
        model=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        for p in [config_path,model,Path(proposals.__file__)]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['file_sha256'][str(p)]
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        scales=frozen_scales('isotropic');step=jax.jit(proposals.build_parameter_step(prior,like,scales))
        report.update(resumed_checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      frozen_thresholds=thresholds,frozen_log_mass=masses,devices=str(jax.devices()),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      initialization='Persisted separate Stage-4F banks; own-walker strict survivors; no lower levels rebuilt')
        def validate(p,current):
            x=np.asarray(p.position);lp=np.asarray(p.logdensity);ll=np.asarray(p.loglikelihood)
            finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(lp)) and np.all(np.isfinite(ll)))
            contour=bool(np.all(ll>current))
            contract=x.shape==(54,) and lp.shape==() and ll.shape==() and finite and contour
            for name,ok in [('nonfinite',finite),('contour',contour),('parameter_contract',contract)]:
                report['failures'][name]+=int(not ok)
            if not contract: raise ValueError('Invalid constrained state')
            a=prior(p.position);b=like(p.position)
            report['cost']['direct_cache_check_evaluations']+=1
            if not (np.isclose(lp,a,rtol=0,atol=1e-9) and np.isclose(ll,b,rtol=0,atol=1e-7)):
                report['failures']['cache']+=1;raise ValueError('Cache inconsistency')
        def attempt(level):
            nonlocal states,diagnostics,provenance
            start_time=time.monotonic();before=report['cost'].copy();current=thresholds[-1]
            traces={};mixing={}
            for b,name in enumerate(NAMES):
                records=[]
                for w in range(8):
                    bank=states[name]
                    # Use saved scalar caches; direct revalidation is a cache-check cost.
                    p=DNSParticleState(jnp.asarray(bank['position'][w]),jnp.asarray(bank['logprior'][w]),jnp.asarray(bank['loglikelihood'][w]))
                    validate(p,current)
                    key=jax.random.key(schedule(level)[b][w])
                    for i in range(384):
                        key,subkey=jax.random.split(key);p,info=step(subkey,p,jnp.asarray(current))
                        report['cost'][f'{name}_likelihood_cost_proxy']+=int(info.slice.num_steps+info.slice.num_shrink)
                        validate(p,current)
                        if not bool(info.slice.is_accepted):
                            report['failures']['slice']+=1;raise ValueError('Slice failure')
                        records.append(dict(walker=w,iteration=i,position=np.asarray(p.position),logprior=float(p.logdensity),
                            loglikelihood=float(p.loglikelihood),num_steps=int(info.slice.num_steps),num_shrink=int(info.slice.num_shrink),
                            component=int(info.component),labels=np.asarray(info.labels)))
                    np.savez_compressed(args.output/f'level{level}_{name}_trace.npz',
                        **{k:np.asarray([r[k] for r in records]) for k in records[0]})
                    print('level',level,name,'walker',w,'completed',flush=True);save()
                traces[name]={k:np.array([r[k] for r in records]).reshape((8,384)+np.asarray(records[0][k]).shape)[:,128:]
                              for k in ['position','logprior','loglikelihood']}
                f0=traces[name]['position'][:,:,::6]
                ess=np.array([[autocorrelation_information(v)['ess'] for v in walker.T] for walker in f0])
                means=f0.mean(1);llmeans=traces[name]['loglikelihood'].mean(1)
                mixing[name]=dict(walker_logL_mean_range=[float(llmeans.min()),float(llmeans.max())],
                    f0_ess_median=float(np.median(ess)),f0_ess_minimum=float(ess.min()),
                    f0_walker_means=means.tolist(),f0_between_walker_mean_sd=means.std(0,ddof=1).tolist(),
                    f0_between_walker_mean_range=np.ptp(means,axis=0).tolist())
            row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],current,masses[-1])
            u=row['calibration_uncertainty']
            row.update(level=level,threshold_increment=row['threshold']-current,mixing=mixing,
                calibration_fraction_range=[min(row['calibration']['walker_fractions']),max(row['calibration']['walker_fractions'])],
                between_walker_dominates=u['between_walker_se']>=max(u['block_means_se'],row['calibration']['naive_iid_se']),
                cost={k:report['cost'][k]-before[k] for k in before},elapsed_seconds=time.monotonic()-start_time,
                seeds=schedule(level))
            report['rows'].append(row)
            if row['accepted']:
                states,indices=promoted_states(traces,row['threshold'],level)
                thresholds.append(row['threshold']);masses.append(row['estimated_log_mass']);diagnostics.append(row)
                np.testing.assert_array_equal(thresholds[:len(initial_thresholds)],initial_thresholds)
                np.testing.assert_array_equal(masses[:len(initial_masses)],initial_masses)
                provenance={**provenance,'parent_checkpoint':str(checkpoint),'promotion_indices':indices,
                            'runner_sha256':report['runner_sha256'],'cumulative_batch_cost':report['cost'].copy()}
                save_checkpoint(args.output/f'checkpoint_level{level}.json',thresholds,masses,states,diagnostics,provenance)
                report['status']=f'frozen_level{level}'
            else:
                report['status']='stopped_first_failed_calibration_or_selection'
            save();print('RESULT',level,row['accepted'],'calibration ESS',row['calibration']['ess'],flush=True)
            return row
        # Provenance changes are local to attempt; retain original source hashes throughout.
        for row in bounded_levels(len(thresholds),10,attempt): pass
    except Exception as exc:
        report.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save();print('Report:',args.output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
