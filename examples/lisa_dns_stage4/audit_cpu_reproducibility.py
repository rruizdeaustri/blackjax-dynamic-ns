"""Stage 6A: direct saved-state CPU evaluations only; never samples or promotes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from examples.lisa_dns_stage4.numerical_cache_contract import CONTRACT, validate_cache

OUT=Path('/tmp/lisa_dns_stage6a_numerical_audit')
B=Path('/tmp/lisa_dns_stage5b_population_level13')
Z=Path('/tmp/lisa_dns_stage4z_population_level12')
CLOSED=Path('/tmp/lisa_dns_pooled_promotion_validation')
ELL12=-108187.99712765554
ELL_TEST=-107123.66036615883
STAGES=[('4V','stage4v_population_level10','population_level10_construction'),
        ('4W','stage4w_level10_validation','level10_population_validation'),
        ('4X','stage4x_population_level11','population_level11_construction'),
        ('4Y','stage4y_level11_validation','level11_population_validation'),
        ('4Z','stage4z_population_level12','population_level12_construction'),
        ('5A','stage5a_level12_validation','level12_population_validation'),
        ('5B','stage5b_population_level13','population_level13_construction')]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    OUT.mkdir(exist_ok=False)
    # Freeze numerical criterion before any new model evaluations.
    write(OUT/'contract.json',CONTRACT)
    history=[]
    for stage,folder,module in STAGES:
        source=Path('examples/lisa_dns_stage4')/(module+'.py')
        text=source.read_text()
        assert 'np.testing.assert_allclose(ap,ps,rtol=0,atol=1e-9)' in text
        assert 'np.testing.assert_allclose(al,ls,rtol=0,atol=1e-7)' in text
        root=Path('/tmp/lisa_dns_'+folder)
        paths=[root/'selection/report.json',root/'calibration/report.json'] if (root/'selection').exists() else [root/'report.json']
        for path in paths:
            r=json.loads(path.read_text())
            assert r['devices']=='[CpuDevice(id=0)]'
            history.append(dict(stage=stage,report=str(path),source=str(source),source_sha256=digest(source),
                backend=r['devices'],prior_max=r['max_cache_prior_error'],likelihood_max=r['max_cache_logL_error']))
    previous=json.loads((B/'selection/report.json').read_text())
    # Includes historical BlackJAX, BayesLISAx model, and exact parameter-config hashes.
    for path,h in previous['file_sha256'].items():assert digest(path)==h,path
    protected={**previous['file_sha256']}
    for root in [B,CLOSED]:
        protected.update({str(p):digest(p) for p in root.rglob('*') if p.is_file()})
    for path in [Z/'checkpoint_level12.json',Z/'levels.npz',Path(__file__),
                 Path('examples/lisa_dns_stage4/numerical_cache_contract.py'),OUT/'contract.json']:
        protected[str(path)]=digest(path)
    assert not (B/'checkpoint_level13.json').exists()
    assert (B/'checkpoint_level12.json').read_bytes()==(Z/'checkpoint_level12.json').read_bytes()
    arrays={k:[] for k in ['position','logprior','loglikelihood']};manifest=[]
    coverage={}
    for bank in ['selection','calibration']:
        starts=load(B/bank/'initial_states.npz');raw=load(B/bank/'trace.npz')
        assert raw['position'].shape==(8,384,54)
        for w in range(8):
            for k in arrays:arrays[k].append(starts[k][w])
            manifest.append(dict(bank=bank,walker=w,kind='stage5b_start',absolute_index=None))
        coverage[bank]=[]
        for w in range(8):
            ll=raw['loglikelihood'][w,128:]
            survivors=np.flatnonzero(ll>ELL_TEST)
            regular=np.linspace(0,255,9,dtype=int)
            chosen=np.unique(np.r_[survivors,regular,np.argmin(ll),np.argmax(ll)])
            coverage[bank].append(dict(walker=w,survivors=len(survivors),selected=len(chosen)))
            for i in chosen:
                for k in arrays:arrays[k].append(raw[k][w,128+i])
                manifest.append(dict(bank=bank,walker=w,kind='retained',retained_index=int(i),
                    absolute_index=int(i+128),candidate_survivor=bool(ll[i]>ELL_TEST)))
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in arrays.items()}
    assert np.all(arrays['loglikelihood']>ELL12)
    np.savez_compressed(OUT/'saved_states.npz',**arrays)
    protected[str(OUT/'saved_states.npz')]=digest(OUT/'saved_states.npz')
    write(OUT/'design.json',dict(contract=CONTRACT,history=history,coverage=coverage,state_manifest=manifest,
        sample_count=len(manifest),selection='all sixteen starts; all retained strict candidate survivors; nine evenly spaced retained indices and min/max per walker; deduplicate indices within walker only',
        repetitions=dict(process_one=2,fresh_process=1),batch_size=8,contour=ELL12,
        file_sha256=protected,sampling=False,promotion=False))
    print('Declared CPU contract; saved states',len(manifest),flush=True)


def evaluate_worker(name, repetitions):
    # Environment asserted before importing JAX or the model; no GPU initialization.
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert os.environ.get('JAX_ENABLE_X64')=='1'
    import jax
    import jax.numpy as jnp
    from examples.lisa_dns_stage4 import audit
    assert jax.default_backend()=='cpu' and jax.config.x64_enabled
    d=json.loads((OUT/'design.json').read_text())
    for path,h in d['file_sha256'].items():assert digest(path)==h,path
    problem,cfg,config_path=audit.load_problem()
    expected=json.loads((Z/'checkpoint_level12.json').read_text())['payload']['provenance']['config']
    assert cfg==expected
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    evaluate=jax.jit(lambda xs:jax.lax.map(lambda x:(prior(x),like(x)),xs))
    data=load(OUT/'saved_states.npz');x=data['position'];outputs=[]
    for repeat in range(repetitions):
        priors=[];likes=[]
        for begin in range(0,len(x),8):
            chunk=x[begin:begin+8];n=len(chunk)
            chunk=np.concatenate([chunk,np.repeat(chunk[-1:],8-n,axis=0)])
            ap,al=map(np.asarray,evaluate(jnp.asarray(chunk)))
            priors.extend(ap[:n]);likes.extend(al[:n])
        outputs.append(dict(logprior=np.array(priors),loglikelihood=np.array(likes)))
        print(name,'direct evaluation pass',repeat+1,'states',len(x),flush=True)
    for repeat,values in enumerate(outputs):
        np.savez_compressed(OUT/f'{name}_{repeat}.npz',**values)
    write(OUT/f'{name}_environment.json',dict(pid=os.getpid(),backend=jax.default_backend(),devices=str(jax.devices()),
        x64=bool(jax.config.x64_enabled),jax_version=jax.__version__,
        jaxlib_version=__import__('jaxlib').__version__,numpy_version=np.__version__,
        executable=sys.executable,config_path=str(config_path),config_sha256=digest(config_path),
        evaluation_passes=repetitions,unique_input_states=len(x),
        scalar_evaluations_per_quantity=repetitions*((len(x)+7)//8)*8))


def differences(actual,expected,atol):
    delta=np.abs(actual-expected)
    relative=np.divide(delta,np.abs(expected),out=np.zeros_like(delta),where=expected!=0)
    return dict(max_absolute=float(delta.max()),max_relative=float(relative.max()),
        median_absolute=float(np.median(delta)),q95_absolute=float(np.quantile(delta,.95)),
        failures=int(np.sum(~np.isfinite(actual)|(delta>atol))),
        bitwise_equal=actual.tobytes()==expected.tobytes())


def summarize():
    d=json.loads((OUT/'design.json').read_text());data=load(OUT/'saved_states.npz')
    outputs=[load(OUT/f'{p}.npz') for p in ['process_one_0','process_one_1','fresh_process_0']]
    errors=[];stats={}
    for label,out in zip(['process_one_0','process_one_1','fresh_process_0'],outputs):
        try:
            validate_cache(data['position'],data['position'].copy(),data['logprior'],data['loglikelihood'],
                out['logprior'],out['loglikelihood'],ELL12,backend='cpu',x64=True)
            survivor=data['loglikelihood']>ELL_TEST
            validate_cache(data['position'][survivor],data['position'][survivor].copy(),
                data['logprior'][survivor],data['loglikelihood'][survivor],
                out['logprior'][survivor],out['loglikelihood'][survivor],ELL_TEST,backend='cpu',x64=True)
        except ValueError as exc:errors.append(label+': '+str(exc))
        stats[label]={k:differences(out[k],data[k],CONTRACT['prior_atol' if k=='logprior' else 'likelihood_atol'])
                      for k in ['logprior','loglikelihood']}
    repeats={}
    for label,out in zip(['same_process','fresh_process'],outputs[1:]):
        repeats[label]={k:differences(out[k],outputs[0][k],CONTRACT['prior_atol' if k=='logprior' else 'likelihood_atol'])
                        for k in ['logprior','loglikelihood']}
    for path,h in d['file_sha256'].items():assert digest(path)==h,path
    result=dict(decision='FAIL' if errors else 'PASS',errors=errors,contract=CONTRACT,
        sample_count=d['sample_count'],coverage=d['coverage'],history=d['history'],
        vs_saved=stats,repeatability=repeats,
        environment=[json.loads((OUT/f'{p}_environment.json').read_text()) for p in ['process_one','fresh_process']],
        protected_hashes_unchanged=True,protected_files=len(d['file_sha256']),sampling_calls=0,promotions=0,
        frozen_through=12,stage5b_unchanged=True,aborted_validation_unchanged=True,
        source_sha256=d['file_sha256'])
    write(OUT/'results.json',result)
    # Small immutable real-data regression fixture: all 16 original starting states.
    ix=np.array([i for i,m in enumerate(d['state_manifest']) if m['kind']=='stage5b_start'])
    path=Path('tests/experimental/data/dns_lisa_cpu_cache_audit.npz');path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,**{k:data[k][ix] for k in data},
        recomputed_prior=outputs[0]['logprior'][ix],recomputed_likelihood=outputs[0]['loglikelihood'][ix])
    write(path.with_suffix('.json'),dict(source=str(OUT/'saved_states.npz'),source_sha256=digest(OUT/'saved_states.npz'),
        states=[d['state_manifest'][i] for i in ix],fixture_sha256=digest(path),contract=CONTRACT))
    print(json.dumps({k:result[k] for k in ['decision','sample_count','vs_saved','repeatability']},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['process_one','fresh_process'])
    args=parser.parse_args()
    if args.worker:
        evaluate_worker(args.worker,2 if args.worker=='process_one' else 1);return
    prepare()
    env={**os.environ,'JAX_PLATFORMS':'cpu','JAX_ENABLE_X64':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in ['process_one','fresh_process']:
        subprocess.run([sys.executable,__file__,'--worker',name],env=env,check=True)
    summarize()


if __name__=='__main__':main()
