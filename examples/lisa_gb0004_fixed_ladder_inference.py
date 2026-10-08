"""Stage 10A: bounded fixed-stratum LISA reconstruction; no construction/termination.

The protected population run and MIS routines are called unchanged. Observation
wrappers persist proposal/occupation provenance without creating transitions.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DESIGN = REPO/'docs/examples/lisa_gb0004_fixed_ladder_inference_design.json'
CHECKSUM = DESIGN.with_suffix('.sha256')
OUT = Path('/tmp/lisa-gb0004-stage10a-fixed-ladder')
START_UUID = str(uuid.uuid4())
START_TIME = time.time()
NAMES = ('f0', 'fdot', 'iota', 'psi', 'lam', 'beta')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def frozen():
    expected = CHECKSUM.read_text().split()[0]
    if digest(DESIGN) != expected: raise ValueError('Frozen design checksum mismatch')
    d = json.loads(DESIGN.read_text())
    for p,h in d['protected_sha256'].items():
        if digest(p) != h: raise ValueError('Protected methodology changed: '+p)
    if digest(__file__) != d['driver_sha256']: raise ValueError('Frozen driver changed')
    return d


def arrays(d):
    return np.array([float(v) for v in d['thresholds']],np.float64),np.array(d['log_masses'],np.float64)


def provenance():
    return {'pid':os.getpid(),'ppid':os.getppid(),'process_start_uuid':START_UUID,
            'process_start_timestamp':START_TIME,'timestamp':time.time(),
            'python':sys.executable,'command':[sys.executable,*sys.argv],
            'environment':{k:os.environ[k] for k in ('JAX_PLATFORMS','JAX_ENABLE_X64')}}


def model_kernel(d):
    import jax
    # Paths only: do not call the historical loader that reads posterior samples.
    root=Path('/r5/home/rruiz/projects/sbi')
    sys.path[:0]=[str(root/'BayesLISAx/src'),str(root/'lisa/gbgpu-main'),
                 str(root/'lisa/jaxlisa-main'),str(root/'lisa/likelihood-master')]
    os.environ['JAX_SAMPLERS_CONFIG']=d['config_path']
    from jax_samplers.problems.lisa_gb_transdim_problem import make
    from examples.lisa_dns_stage4.automatic_ladder_integration import make_frozen_kernel
    from blackjax.ns.dns_automatic import NumericalContract
    problem=make()
    assert problem.dim==54 and problem.Kmax==9 and problem.marg_Aphi
    assert not problem.use_gates and not problem.order_f0
    contract=NumericalContract()
    environment=contract.environment()
    assert jax.default_backend()=='cpu' and jax.config.jax_enable_x64
    prior=lambda x:problem.logprior(x)
    likelihood=lambda x:problem.loglikelihood(x)
    kernel=make_frozen_kernel(prior,likelihood,d['physical_bounds']['f0'],model_id=d['model_id'])
    return problem,kernel,contract,environment


def fields(history):
    return {'position':np.asarray(history.position),'logprior':np.asarray(history.logdensity),
            'loglikelihood':np.asarray(history.loglikelihood)}


def load_history(path):
    from blackjax.ns.dns import DNSParticleState
    with np.load(path,allow_pickle=False) as f:
        return DNSParticleState(f['position'].copy(),f['logprior'].copy(),f['loglikelihood'].copy())


def make_records(root, set_name, last, d):
    from blackjax.ns import dns_reconstruction as r
    t,m=arrays(d); columns={k:[] for k in ('position','logprior','loglikelihood','origin_level','walker','draw','event_id')}
    metadata=[]
    for j in range(last+1):
        folder=root/set_name/f'level_{j:02d}'
        h=load_history(folder/'retained.npz')
        for name,x in fields(h).items():columns[name].append(x.reshape((-1,54) if name=='position' else (-1,)))
        columns['origin_level'].append(np.full(8192,j,dtype=np.int64))
        columns['walker'].append(np.tile(np.arange(8),1024))
        columns['draw'].append(np.repeat(np.arange(1024),8))
        columns['event_id'].append(np.array([f'{set_name}/L{j}/{s}/{w}' for s in range(1024) for w in range(8)]))
        pre=json.loads((folder/'preflight.json').read_text())
        metadata.append({'level':j,'threshold':d['thresholds'][j],
                         'stream_id':f'{set_name}/L{j}','streams':pre['streams'],
                         'source_bank':'fresh-inference','sampling_target':r.TARGET,
                         'promotion':pre['promotion']})
    return r.ReconstructionRecords(t[:last+1],m[:last+1],**{k:np.concatenate(v) for k,v in columns.items()},
            metadata={'sampling_target':r.TARGET,'mass_status':['prior']+['calibrated']*last,
                      'levels':metadata,'reconstruction_set':set_name,'burn_in':512,'retained':1024,
                      'contract':d['numerical_contract'],'model_id':d['model_id'],
                      'construction_samples_included':False}).validate()


def pool_records(a,b):
    """Unique walker namespace preserves the unchanged Stage-7B record schema."""
    from blackjax.ns.dns_reconstruction import ReconstructionRecords
    kwargs={k:np.concatenate([getattr(a,k),getattr(b,k)]) for k in
            ('position','logprior','loglikelihood','origin_level','draw','event_id')}
    kwargs['walker']=np.concatenate([a.walker,b.walker+8])
    meta=dict(a.metadata,reconstruction_set='pooled_R1_R2',
              walker_namespace={'R1':list(range(8)),'R2':list(range(8,16))},
              levels=[dict(x,stream_id=f'pooled/L{j}',streams_by_set={'R1':x['streams'],'R2':b.metadata['levels'][j]['streams']})
                      for j,x in enumerate(a.metadata['levels'])])
    return ReconstructionRecords(a.thresholds.copy(),a.log_masses.copy(),**kwargs,metadata=meta).validate()


def observe_kernel(base, folder, d):
    from blackjax.ns.dns_population import FrozenPopulationKernel
    from examples.lisa_dns_stage4.frequency_population_benchmark import slice_blocks,exchange_tags
    import jax
    context={'info':None,'completed':0,'tags':np.arange(72).reshape(8,9)}
    saved={}
    specs={'post_slice_position':((1536,8,54),np.float64),'post_slice_logprior':((1536,8),np.float64),
           'post_slice_loglikelihood':((1536,8),np.float64),'position':((1536,8,54),np.float64),
           'logprior':((1536,8),np.float64),'loglikelihood':((1536,8),np.float64),
           'slice_component':((1536,8),np.int64),'slice_labels':((1536,8,2),np.int64),
           'slice_num_steps':((1536,8),np.int64),'slice_num_shrink':((1536,8),np.int64),
           'slice_accepted':((1536,8),np.bool_),'slice_keys':((1536,8,2),np.uint32),
           'exchange_gumbels':((1536,4,2,9),np.float64),'exchange_uniforms':((1536,4),np.float64),
           'lineage_tokens':((1536,8,9),np.int64)}
    for name,(shape,dtype) in specs.items():
        saved[name]=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
    stream=(folder/'exchange_decisions.jsonl').open('x')
    timer=time.time()
    def step(keys,state,contour):
        result,info=base.slice_step(keys,state,contour)
        context['info']=info
        s=context['completed']
        saved['slice_keys'][s]=np.asarray(jax.random.key_data(keys))
        for name,x in fields(result).items():saved['post_slice_'+name][s]=x
        for name,x in [('slice_component',info.component),('slice_labels',info.labels),
                       ('slice_num_steps',info.slice.num_steps),('slice_num_shrink',info.slice.num_shrink),
                       ('slice_accepted',info.slice.is_accepted)]:saved[name][s]=np.asarray(x)
        return result,info
    class Observed(FrozenPopulationKernel):
        def exchange_sweep(self,sliced,threshold,sweep,gumbels,uniforms,contract):
            result,decisions=super().exchange_sweep(sliced,threshold,sweep,gumbels,uniforms,contract)
            for name,x in fields(result).items():saved[name][sweep]=x
            saved['exchange_gumbels'][sweep]=gumbels
            saved['exchange_uniforms'][sweep]=uniforms
            tags=context['tags'].copy();info=context['info']
            for w in range(8):
                selected=slice_blocks(int(info.component[w]),np.asarray(info.labels[w]))
                tags[w,selected]=(sweep+1)*72+w*9+selected
            for dec in decisions:
                a,b=dec['pair'];i,j=dec['labels'];tags=exchange_tags(tags,a,b,i,j,dec['accepted'])
            context['tags']=tags;saved['lineage_tokens'][sweep]=tags
            stream.write(json.dumps(clean({'sweep':sweep,'decisions':decisions}),allow_nan=False)+'\n')
            context['completed']=sweep+1
            if (sweep+1)%32==0:
                for x in saved.values():x.flush()
                stream.flush()
                write(folder/'progress.json',{'completed_sweeps':sweep+1,'elapsed_seconds':time.time()-timer})
                print(folder.parent.name,folder.name,'sweep',sweep+1,'seconds',round(time.time()-timer,1),flush=True)
            return result,decisions
    kernel=Observed(step,base.evaluator,base.selector,base.exchange,base.slice_blocks,base.identity)
    return kernel,context,saved,stream


def run_level(root,set_name,j,problem,base,contract,d,previous):
    import jax
    from blackjax.ns.dns import DNSParticleState
    from blackjax.ns.dns_automatic import promote,LadderConfig
    t,_=arrays(d);folder=root/set_name/f'level_{j:02d}';folder.mkdir(parents=True,exist_ok=False)
    rng=d['streams'][set_name]['levels'][str(j)]
    if j==0:
        seed=d['streams'][set_name]['prior_initialization_seed']
        x=np.asarray(problem.sample_prior(jax.random.key(seed,impl='threefry2x32'),8))
        lp,ll=base.evaluate(x);starts=DNSParticleState(x,lp,ll)
        promotion={'kind':'fresh-prior','seed':seed,'historical_samples_used':False}
    else:
        starts,promotion=promote(previous,t[j],rng['promotion_seed'],contract)
    contract.validate(starts,t[j],base.evaluate(starts.position))
    np.savez_compressed(folder/'starts.npz',**fields(starts))
    write(folder/'preflight.json',{'level':j,'reconstruction_set':set_name,'threshold':d['thresholds'][j],
                                 'streams':rng,'promotion':promotion,'process':provenance()})
    kernel,context,saved,decisions=observe_kernel(base,folder,d)
    try:
        history,info=kernel.run(starts,t[j],rng,LadderConfig(burn_in=512,retained=1024,block_size=64),contract)
        # All model transitions above are in the original protected run method.
        for name,x in fields(history).items():
            assert x.tobytes()==np.asarray(saved[name][512:]).tobytes(),name
        np.savez_compressed(folder/'retained.npz',**fields(history))
        # Deterministic 128-state saved-file audit, including first and last retained sweeps.
        sel=np.linspace(0,1023,16,dtype=int)
        loaded=load_history(folder/'retained.npz')
        audit=DNSParticleState(*(np.asarray(v)[sel].reshape((-1,54) if k==0 else (-1,)) for k,v in enumerate(loaded)))
        source=DNSParticleState(*(np.asarray(v)[sel].reshape((-1,54) if k==0 else (-1,)) for k,v in enumerate(history)))
        assert all(a.tobytes()==b.tobytes() for a,b in zip(audit,source))
        recomputed=base.evaluate(audit.position);contract.validate(audit,t[j],recomputed)
        np.savez_compressed(folder/'deterministic_cache_audit.npz',indices=sel,
            position=audit.position,logprior=audit.logdensity,loglikelihood=audit.loglikelihood,
            recomputed_logprior=recomputed[0],recomputed_loglikelihood=recomputed[1])
        # Exact terminal RNG-state receipt. Schedules are fully allocated before the first sweep.
        def final_key(seed):
            key=jax.random.key(seed,impl='threefry2x32')
            return jax.lax.fori_loop(0,1536,lambda i,k:jax.random.split(k)[0],key)
        terminal=np.stack([np.asarray(jax.random.key_data(final_key(seed))) for seed in rng['slice_seeds']])
        labels=np.random.Generator(np.random.PCG64(rng['label_seed']));g=labels.gumbel(size=(1536,4,2,9))
        mh=np.random.Generator(np.random.PCG64(rng['mh_seed']));u=mh.random((1536,4))
        assert g.tobytes()==saved['exchange_gumbels'].tobytes() and u.tobytes()==saved['exchange_uniforms'].tobytes()
        np.savez_compressed(folder/'terminal_rng.npz',walker_key_data=terminal,terminal_position=history.position[-1],
                            terminal_logprior=history.logdensity[-1],terminal_loglikelihood=history.loglikelihood[-1])
        write(folder/'terminal_rng.json',{'label_rng_state':labels.bit_generator.state,'mh_rng_state':mh.bit_generator.state,
              'level_complete':True,'next_level':j+1,'next_level_keys_reset_by_frozen_recipe':True})
        write(folder/'completed.json',dict(info,cache_audit_passed=True,strict_contour_passed=True,
             completed_at=time.time(),cache_max_errors=[float(np.max(np.abs(recomputed[0]-audit.logdensity))),
             float(np.max(np.abs(recomputed[1]-audit.loglikelihood)))]))
        return history
    finally:
        for x in saved.values():x.flush()
        decisions.close()
        write(folder/'last_progress.json',{'completed_sweeps':context['completed'],'slice_attempts_recorded':
                                          context['completed']+int(context['info'] is not None)})


def checkpoint(root,d):
    from blackjax.ns import dns_reconstruction as r
    p=root/'R1_L6_checkpoint';p.mkdir(exist_ok=False)
    records=make_records(root,'R1',6,d);r.save_records(p/'completed_records',records)
    source=root/'R1/level_06'
    import shutil
    for name in ('retained.npz','terminal_rng.npz','terminal_rng.json'):
        shutil.copyfile(source/name,p/name)
    write(p/'manifest.json',{'level_complete':6,'next_level':7,'set':'R1','design_sha256':digest(DESIGN),
            'thresholds':d['thresholds'],'log_masses':d['log_masses'],'streams':d['streams']['R1'],
            'contract':d['numerical_contract'],'parent_process':provenance(),
            'files':{str(f.relative_to(p)):digest(f) for f in p.rglob('*') if f.is_file()}})


def execute(child=False):
    from blackjax.ns import dns_reconstruction as r
    d=frozen()
    if child:
        OUT.joinpath('restart_started').open('x').close()
        root=OUT/'restarted';root.mkdir(exist_ok=False)
        check=OUT/'R1_L6_checkpoint';manifest=json.loads((check/'manifest.json').read_text())
        assert manifest['design_sha256']==digest(DESIGN)
        for f,h in manifest['files'].items():assert digest(check/f)==h,f
        old=r.load_records(check/'completed_records')
        assert old.log_masses.tobytes()==arrays(d)[1][:7].tobytes()
        prior=load_history(check/'retained.npz')
        write(root/'process.json',provenance())
    else:
        OUT.mkdir(exist_ok=False);OUT.joinpath('sampling_started').open('x').close();root=OUT
        write(root/'process.json',provenance());prior=None
    problem,base,contract,env=model_kernel(d)
    write(root/'environment.json',env)
    work=[('R1',range(7,13))] if child else [('R1',range(13)),('R2',range(13))]
    for set_name,levels in work:
        if set_name=='R2':prior=None
        for j in levels:
            prior=run_level(root,set_name,j,problem,base,contract,d,prior)
            if not child and set_name=='R1' and j==6:checkpoint(root,d)
        if not child:r.save_records(root/set_name/'records',make_records(root,set_name,12,d))
    if child:
        write(root/'sampling_complete.json',{'process':provenance(),'replay_only':True})
        return
    # R1 uninterrupted and R2 authoritative completed. One genuine fresh-process replay.
    command=[sys.executable,str(Path(__file__).resolve()),'restart-child']
    with (OUT/'restart.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=REPO,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'restart_launch.json',{'parent':provenance(),'child_pid':proc.pid,
                                       'command':command,'launch_timestamp':time.time()})
        code=proc.wait()
    if code:raise RuntimeError('Fresh-process restart failed: exit '+str(code))
    checks=[]
    for j in range(7,13):
        a=OUT/'R1'/f'level_{j:02d}';b=OUT/'restarted/R1'/f'level_{j:02d}'
        for p in a.glob('*.npy'):
            x=np.load(p);y=np.load(b/p.name)
            assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),str(p)
            checks.append(f'L{j}/{p.name}')
        for name in ('starts.npz','retained.npz','terminal_rng.npz','deterministic_cache_audit.npz'):
            with np.load(a/name) as f,np.load(b/name) as g:
                for k in f.files:
                    assert f[k].dtype==g[k].dtype and f[k].shape==g[k].shape and f[k].tobytes()==g[k].tobytes(),(j,name,k)
                    checks.append(f'L{j}/{name}/{k}')
        assert (a/'exchange_decisions.jsonl').read_bytes()==(b/'exchange_decisions.jsonl').read_bytes()
        assert json.loads((a/'terminal_rng.json').read_text())==json.loads((b/'terminal_rng.json').read_text())
        aa=json.loads((a/'preflight.json').read_text());bb=json.loads((b/'preflight.json').read_text())
        assert aa['promotion']==bb['promotion'] and aa['streams']==bb['streams']
    # Assemble replay records with the immutable completed prefix, never resample it.
    import shutil
    for j in range(7):shutil.copytree(OUT/'R1'/f'level_{j:02d}',OUT/'restarted/R1'/f'level_{j:02d}')
    ref=r.load_records(OUT/'R1/records');repeat=make_records(OUT/'restarted','R1',12,d)
    for k in ('thresholds','log_masses','position','logprior','loglikelihood','origin_level','walker','draw','event_id'):
        assert getattr(ref,k).tobytes()==getattr(repeat,k).tobytes(),k
    assert ref.metadata==repeat.metadata
    er,ec=r.reconstruct_evidence(ref),r.reconstruct_evidence(repeat)
    assert er['logZ']==ec['logZ'] and er['weights'].tobytes()==ec['weights'].tobytes()
    parent=json.loads((OUT/'process.json').read_text());kid=json.loads((OUT/'restarted/process.json').read_text())
    assert parent['pid']!=kid['pid'] and kid['ppid']==parent['pid'] and parent['process_start_uuid']!=kid['process_start_uuid']
    write(OUT/'restart_comparison.json',{'pass':True,'array_checks':checks,'records_exact':True,
            'mis_outputs_exact':True,'parent':parent,'child':kid,'replay_events_counted':False})
    write(OUT/'sampling_complete.json',{'finished_at':time.time(),'design_sha256':digest(DESIGN)})
    frozen()
    analyze(problem,d)


def reference_catalogues(d):
    audit=json.loads(Path(d['matching']['reference_path']).read_text())
    return [np.stack([np.array(row['decoded'][k])[0] for k in NAMES],axis=-1) for row in audit['families']]


def source_view(records,d):
    from scipy.special import expit
    from examples.lisa_dns_stage4.diagnose_source_permutations import source_distances
    bounds=d['physical_bounds'];lo=np.array([bounds[k][0] for k in NAMES]);hi=np.array([bounds[k][1] for k in NAMES])
    phys=lo+(hi-lo)*expit(records.position.reshape(-1,9,6))
    refs=reference_catalogues(d);widths=hi-lo;widths[3]=np.pi;widths[4]=2*np.pi
    matched=np.empty_like(phys);permutations=np.empty((len(phys),9),np.int64)
    distances=np.empty((len(phys),len(refs)))
    for i,x in enumerate(phys):
        result=source_distances(refs[0],x,widths,{3:np.pi,4:2*np.pi})
        perm=result['permutation'];matched[i]=x[perm];permutations[i]=perm;distances[i,0]=result['assignment']
        for q in range(1,len(refs)):distances[i,q]=source_distances(refs[q],x,widths,{3:np.pi,4:2*np.pi})['assignment']
    # Unwrap circular quantities around the fixed reference, never fit a cut from results.
    for k,period in [(3,np.pi),(4,2*np.pi)]:
        matched[:,:,k]=refs[0][:,k]+(matched[:,:,k]-refs[0][:,k]+period/2)%period-period/2
    return matched,permutations,distances


def summaries(records,matched,distances,masses=None):
    from blackjax.ns import dns_reconstruction as r
    from scipy.special import logsumexp
    est=r.reconstruct_evidence(records,log_masses=masses);w=est['weights'];lr=est['log_importance']
    raw_fraction=np.bincount(records.origin_level,weights=w,minlength=len(records.thresholds))
    high=np.sort(w)[::-1]
    multiplicity=(records.loglikelihood[:,None]>records.thresholds).sum(1)
    q={name:[r.posterior_quantiles(matched[:,s,k],w,[.05,.16,.5,.84,.95]).tolist() for s in range(9)] for k,name in enumerate(NAMES)}
    families=np.bincount(np.argmin(distances,axis=1),weights=w,minlength=3)
    terminal=float(w[records.loglikelihood>records.thresholds[-1]].sum())
    return {'logZ':est['logZ'],'posterior_weight_ess':est['weight_ess'],'maximum_weight':est['maximum_weight'],
            'weight_entropy':est['entropy'],'per_level_raw_evidence_fraction':raw_fraction.tolist(),
            'per_level_posterior_weight':raw_fraction.tolist(),
            'per_level_log_evidence':[float(logsumexp(lr[records.origin_level==j])-np.log(len(w))) for j in range(len(records.thresholds))],
            'weighted_eligible_multiplicity':np.bincount(multiplicity,weights=w,minlength=len(records.thresholds)+1).tolist(),
            'cumulative_origin_evidence_fraction':np.cumsum(raw_fraction).tolist(),
            'cumulative_origin_log_evidence':[float(logsumexp(lr[records.origin_level<=j])-np.log(len(w))) for j in range(len(records.thresholds))],
            'origin_L12_fraction':float(raw_fraction[12]) if len(raw_fraction)==13 else None,
            'target_terminal_fraction':terminal,'highest_weight_fraction':
            {str(p):float(high[:int(np.ceil(p*len(w)))].sum()) for p in (.01,.05,.1)},
            'eligible_level_multiplicity_counts':np.bincount(multiplicity,minlength=len(records.thresholds)+1).tolist(),
            'source_quantiles':q,
            'source_means':{name:r.posterior_expectation(matched[:,:,k],w).tolist() for k,name in enumerate(NAMES)},
            'nearest_historical_catalogue_weight':families.tolist()},est


def analyze(problem,d):
    from blackjax.ns import dns_reconstruction as r
    from scipy.special import expit,logit
    a,b=[r.load_records(OUT/name/'records') for name in ('R1','R2')]
    pooled=pool_records(a,b);r.save_records(OUT/'pooled/records',pooled)
    ratios=np.exp(np.diff(arrays(d)[1]));se=np.array(d['mass_perturbations']['calibration_standard_errors'])
    perturb=expit(logit(ratios)+np.random.default_rng(10600403).normal(size=(128,12))*se/(ratios*(1-ratios)))
    masses=np.column_stack([np.zeros(128),np.cumsum(np.log(perturb),axis=1)])
    np.savez_compressed(OUT/'mass_perturbations.npz',log_masses=masses)
    report={'stage':'10A','design_sha256':digest(DESIGN),'results':{},'evidence_convention':d['evidence_convention']}
    for name,rec in [('R1',a),('R2',b),('pooled',pooled)]:
        print('Analyzing',name,flush=True)
        folder=OUT/name;folder.mkdir(exist_ok=True)
        matched,perms,distances=source_view(rec,d)
        np.savez_compressed(folder/'matched_sources.npz',physical=matched,permutations=perms,reference_distances=distances)
        primary,est=summaries(rec,matched,distances)
        np.savez_compressed(folder/'mis_outputs.npz',posterior_weights=est['weights'],log_importance=est['log_importance'])
        primary['correlation_diagnostics']={}
        # Pooled walker namespaces have 16 walkers; same estimator, descriptive ESS only.
        scaled=np.exp(est['log_importance']-est['log_importance'].max())
        for observable,values in [('loglikelihood',rec.loglikelihood),('logprior',rec.logprior),('evidence',scaled)]:
            primary['correlation_diagnostics'][observable]=r.correlation_diagnostics(rec,values,64)
        for k,coordinate in enumerate(NAMES):
            for s in range(9):
                value=matched[:,s,k]
                primary['correlation_diagnostics'][f'{coordinate}_{s}']=r.correlation_diagnostics(rec,value,64)
                residual=scaled*(value-r.posterior_expectation(value,est['weights']))
                primary['correlation_diagnostics'][f'posterior_{coordinate}_{s}']=r.correlation_diagnostics(rec,residual,64)
        primary['between_walker_and_temporal']=[]
        from scipy.stats import ks_2samp
        nw=16 if name=='pooled' else 8
        for j in range(13):
            mask=rec.origin_level==j
            ll=np.empty((1024,nw));ll[rec.draw[mask],rec.walker[mask]]=rec.loglikelihood[mask]
            geom=np.empty((1024,nw,9,6));geom[rec.draw[mask],rec.walker[mask]]=matched[mask]
            primary['between_walker_and_temporal'].append({'level':j,
                'logL_walker_means':ll.mean(0).tolist(),
                'logL_half_means':[ll[:512].mean(0).tolist(),ll[512:].mean(0).tolist()],
                'matched_coordinate_walker_means':geom.mean(0).tolist(),
                'matched_coordinate_half_means':[geom[:512].mean(0).tolist(),geom[512:].mean(0).tolist()],
                'logL_pairwise_ks':[float(ks_2samp(ll[:,a],ll[:,b]).statistic) for a in range(nw) for b in range(a)]})
        primary['mixture_sensitivity']={}
        for j in range(8,13):
            keep=rec.origin_level<=j
            primary['mixture_sensitivity'][str(j)]=summaries(rec.through(j),matched[keep],distances[keep])[0]
        responses=[summaries(rec,matched,distances,m)[0] for m in masses]
        write(folder/'mass_sensitivity.json',responses)
        primary['mass_sensitivity']={'logZ_q05_median_q95':np.quantile([x['logZ'] for x in responses],[.05,.5,.95]).tolist(),
                'source_quantile_ranges':{c:np.quantile(np.array([x['source_quantiles'][c] for x in responses]),[.05,.5,.95],axis=0).tolist() for c in NAMES},
                'reference_family_weights_range':np.quantile([x['nearest_historical_catalogue_weight'] for x in responses],[.05,.5,.95],axis=0).tolist(),
                'responses_path':str(folder/'mass_sensitivity.json'),'interpretation':'Sensitivity only, not a confidence interval'}
        report['results'][name]=primary
    # Residual/catalogue diagnostics on the predeclared deterministic subset only.
    residual=[]
    for name in ('R1','R2'):
        for j in range(13):
            h=load_history(OUT/name/f'level_{j:02d}'/'retained.npz')
            for sweep in (0,511,1023):
                x=np.asarray(h.position[sweep,0]);diag=np.asarray(problem._marg_diag(x));recon=problem._marg_recon(x)
                residual.append({'set':name,'level':j,'draw':sweep,'walker':0,'residual_chi2':float(diag[2]),
                    'logdet':float(diag[3]),'amplitude_fitted':np.abs(np.asarray(recon[5])).tolist(),
                    'phase_fitted':np.angle(np.asarray(recon[5])).tolist(),'source_gain':np.asarray(recon[6]).tolist()})
    write(OUT/'residual_catalogue_diagnostics.json',residual)
    report['residual_diagnostics_path']=str(OUT/'residual_catalogue_diagnostics.json')
    report['classification']='PENDING_FROZEN_QUALITATIVE_SCIENTIFIC_REVIEW'
    write(OUT/'analysis.json',report)


def fail(exc):
    OUT.mkdir(exist_ok=True)
    status={'case':'D','label':'RECONSTRUCTION / NUMERICAL FAILURE','numerical_status':('RECONSTRUCTION-INCOMPLETE' if 'no eligible states' in str(exc) else 'NUMERICAL-FAIL'),
            'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'process':provenance(),'no_retry':True,'design_sha256':digest(DESIGN) if DESIGN.exists() else None}
    write(OUT/('restart_failure.json' if len(sys.argv)>1 and sys.argv[1]=='restart-child' else 'failure.json'),status)
    print(json.dumps(status),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','restart-child'])
    args=parser.parse_args()
    try:execute(args.phase=='restart-child')
    except Exception as exc:fail(exc);raise

if __name__=='__main__':main()
