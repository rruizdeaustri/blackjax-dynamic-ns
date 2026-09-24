"""Stage-4U single fixed-contour benchmark; no ladder or generic-kernel changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logsumexp

from examples.lisa_dns_stage4.analyze_frequency_exchange_design import (
    INTERVAL, fixed_exchange, fixed_frequency_region,
)

OUT = Path('/tmp/lisa_dns_stage4u_population')
H = Path('/tmp/lisa_dns_stage4h_continuation')
O = Path('/tmp/lisa_dns_stage4o_hybrid')
ELL9 = -110252.99476697217
LOGX9 = -7.67544001580533
SWEEPS = 512
SLICE_SEEDS = list(range(810100,810108))
LABEL_SEED = 2026092401
MH_SEED = 2026092402


def serial(x):
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def round_robin():
    ring=list(range(8));rounds=[]
    for _ in range(7):
        rounds.append(sorted(tuple(sorted((ring[k],ring[-1-k]))) for k in range(4)))
        ring=[ring[0],ring[-1],*ring[1:-1]]
    return np.asarray(rounds,dtype=int)


def log_probabilities(position,frequency_bounds):
    x=np.asarray(position)
    assert x.shape[-1]==54 and np.isfinite(x).all()
    low,high=frequency_bounds
    f=low+(high-low)*expit(x[...,::6])
    center,sigma=fixed_frequency_region()
    logw=-.5*((f-center)/sigma)**2
    logp=logw-logsumexp(logw,axis=-1,keepdims=True)
    assert np.isfinite(logp).all()
    # Extended precision audit only; label sampling and MH never exponentiate these.
    p=np.exp(logp.astype(np.longdouble))
    assert np.isfinite(p).all() and np.all(p>0)
    assert np.max(abs(p.sum(axis=-1)-1))<2e-13
    return logp,f


def draw_label(logp,gumbels):
    """Log-space Gumbel-max; one preallocated independent variate per label."""
    logp=np.asarray(logp);gumbels=np.asarray(gumbels)
    assert logp.shape==gumbels.shape==(9,)
    assert np.isfinite(logp).all() and np.isfinite(gumbels).all()
    return int(np.argmax(logp+gumbels))


def joint_decision(old_prior,new_prior,forward_logp,reverse_logp,labels,new_logL,uniform):
    assert 0<=uniform<1
    i,j=labels
    prior_delta=float(np.sum(new_prior)-np.sum(old_prior))
    qf=float(forward_logp[0,i]+forward_logp[1,j])
    qr=float(reverse_logp[0,i]+reverse_logp[1,j])
    selection=qr-qf;ratio=prior_delta+selection
    finite=bool(np.isfinite(new_prior).all() and np.isfinite(new_logL).all() and np.isfinite(ratio))
    contour=bool(np.isfinite(new_logL).all() and np.all(np.asarray(new_logL)>ELL9))
    alpha=float(np.exp(min(0.,ratio))) if finite and contour else 0.
    logu=float(np.log(uniform)) if uniform>0 else -np.inf
    accepted=bool(finite and contour and logu<min(0.,ratio))
    return dict(delta_logprior=prior_delta,log_q_forward=qf,log_q_reverse=qr,
                log_r_selection=selection,log_r=ratio,alpha=alpha,uniform=uniform,
                joint_contour=contour,accepted=accepted,nonfinite_prior=int((~np.isfinite(new_prior)).sum()),
                nonfinite_likelihood=int((~np.isfinite(new_logL)).sum()))


def commit_pair(old,new,accepted):
    """Each tuple is (two positions, two cached priors, two cached likelihoods)."""
    return tuple(np.array(v,copy=True) for v in (new if accepted else old))


def exchange_tags(tags,a,b,i,j,accepted):
    result=tags.copy()
    if accepted:result[a,i],result[b,j]=tags[b,j],tags[a,i]
    return result


def slice_blocks(component,labels):
    return np.arange(9) if component==0 else np.asarray(labels[:1] if component==1 else labels)


def prefix_from_cost(costs,budget,initial=8):
    c=np.asarray(costs)
    if c.ndim!=1 or not np.isfinite(c).all() or np.any(c<0) or budget<0:
        raise ValueError('Nonnegative per-population-sweep costs required')
    cumulative=initial+np.cumsum(c)
    n=int(np.sum(cumulative<=budget))
    return dict(sweeps=n,cost=float(cumulative[n-1] if n else initial),
                next_cost=float(cumulative[n]) if n<len(c) else None,budget=float(budget))


def prepare():
    assert not OUT.exists(),'Do not recreate or rerun the one benchmark'
    starts=load(H/'initial_states.npz')
    original=load('/tmp/lisa_dns_stage4_ladder_batch/level10_calibration_trace.npz')
    for k in starts:
        assert starts[k].tobytes()==original[k].reshape((8,384)+original[k].shape[1:])[:,-1].tobytes()
    for p in [O/'initial_states.npz',Path('/tmp/lisa_dns_stage4n_sequential/initial_states.npz')]:
        other=load(p)
        for k in starts:assert starts[k].tobytes()==other[k].tobytes()
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    payload=json.loads(checkpoint.read_text())['payload']
    assert payload['thresholds'][-1]==ELL9 and payload['log_masses'][-1]==LOGX9
    previous=json.loads(Path('/tmp/lisa_dns_stage4t_frequency_design/analysis.json').read_text())
    for path,h in previous['input_sha256'].items():assert digest(path)==h,path
    hreport=json.loads((H/'report.json').read_text())
    for path,h in hreport['file_sha256'].items():assert digest(path)==h,path
    assert hreport['scale']==float(np.pi/np.sqrt(3)) and hreport['mixture']==[.2,.4,.4]
    paths=[checkpoint,H/'initial_states.npz',H/'trace.npz',H/'report.json',
           O/'initial_states.npz',O/'hybrid_pair/trace.npz',O/'report.json',
           Path('/tmp/lisa_dns_stage4_prior_smoke/report.json'),
           Path('/tmp/lisa_dns_stage4t_frequency_design/design.json'),Path(__file__),
           Path('examples/lisa_dns_stage4/proposals.py'),Path('blackjax/mcmc/ss.py'),
           *Path('blackjax/ns').glob('dns*.py')]
    # Frozen model/configuration are protected without evaluating either.
    paths += [Path(p) for p in payload['provenance']['file_sha256'] if 'lisa_gb_transdim_problem.py' in p or p.endswith('k9.json')]
    design=dict(ell9=ELL9,logX9=LOGX9,sweeps=SWEEPS,walkers=8,burn_in=0,
        frequency_interval=INTERVAL,f_star=fixed_frequency_region()[0],sigma_f=fixed_frequency_region()[1],
        round_robin=round_robin(),slice_seeds=SLICE_SEEDS,label_rng='NumPy PCG64 Gumbel-max in log space',
        label_seed=LABEL_SEED,mh_rng='NumPy PCG64 one explicit joint uniform per pair',mh_seed=MH_SEED,
        slice='Unchanged proposals.build_parameter_step; scales pi/sqrt(3), mixture .2/.4/.4; max_steps=10 max_shrinkage=100',
        common_random_numbers='Same per-walker slice keys as Stage H, split once per transition; exchange randomness independent',
        ordering='8 independent slice transitions, then 4 disjoint joint exchanges; round=sweep mod 7',
        cache_checks='8 initial; each of 8 post-slice states and each of 8 post-exchange states every sweep, prior and likelihood directly recomputed',
        cost_rule='Largest complete population-sweep prefix with cumulative slice num_steps+num_shrink +8 exchange likelihood +16 direct cache likelihood per sweep +8 initial checks <= Stage-H 27342+4104. Supplement proposal-only prefix <=27342.',
        classification='A strong/B modest/C no meaningful/D worse vs H; prioritize concordant logL mean SD/MAD/IQR, old failed ell10 exceedance SD, median pairwise logL KS and permutation-invariant communication. Examine both halves, all512 and cost prefix. No classification from acceptance/labelled ESS alone. O Hybrid P descriptive only.',
        geometry='Stage I frequency-set RMS all cross-time pairs; six-coordinate Hungarian RMS at every 32nd window state; final window last256, with last32 descriptive. Per-walker start/end, max-from-start and within-window source-set distances.',
        provenance='Initial lineage IDs 0..71 follow slots through slice descendants and exchange. Exact realization token is newly minted at every slice-targeted block (conservative invalidation even if roundoff leaves it unchanged); untouched tokens persist. Tokens swapped only on joint acceptance. Birth walker, parent token, sweep and source slot logged independently of value matching.',
        file_sha256={str(p):digest(p) for p in paths},starts_byte_identical=True,
        level10_remains_rejected=True,source_gain_calls=0)
    OUT.mkdir()
    np.savez_compressed(OUT/'initial_states.npz',**starts)
    np.savez_compressed(OUT/'random_schedule.npz',
        gumbels=np.random.Generator(np.random.PCG64(LABEL_SEED)).gumbel(size=(512,4,2,9)),
        uniforms=np.random.Generator(np.random.PCG64(MH_SEED)).random((512,4)))
    design['file_sha256'].update({str(OUT/p):digest(OUT/p) for p in ['initial_states.npz','random_schedule.npz']})
    (OUT/'design.json').write_text(json.dumps(serial(design),indent=2,allow_nan=False))
    print('Frozen design and starts:',OUT,flush=True)


def run():
    # This is the only entry point importing/evaluating the real LISA model.
    from examples.lisa_dns_stage4 import audit,proposals
    import jax
    import jax.numpy as jnp
    from blackjax.ns.dns import DNSParticleState
    assert not (OUT/'report.json').exists(),'Never rerun or extend this benchmark'
    design=json.loads((OUT/'design.json').read_text())
    for path,h in design['file_sha256'].items():assert digest(path)==h,path
    assert design['sweeps']==512 and design['ell9']==ELL9 and design['logX9']==LOGX9
    assert (design['f_star'],design['sigma_f'])==fixed_frequency_region()
    problem,cfg,config_path=audit.load_problem()
    payload=json.loads(Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json').read_text())['payload']
    assert cfg==payload['provenance']['config']
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    scalar_step=proposals.build_parameter_step(prior,like,np.full(54,np.pi/np.sqrt(3)),max_steps=10,max_shrinkage=100)
    step=jax.jit(lambda keys,states:jax.lax.map(lambda data:scalar_step(data[0],data[1],jnp.asarray(ELL9)),(keys,states)))
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
    report=dict(status='running',ell9=ELL9,logX9=LOGX9,completed_sweeps=0,
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
        assert np.isfinite(xs).all() and np.isfinite(ps).all() and np.isfinite(ls).all() and np.all(ls>ELL9)
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
