"""Stage-4M fixed-base feasibility only: preserve the implemented scalar target.

For a state-independent labelled block B, w(B) cancels in reverse/forward
proposal probabilities. The unchanged coordinates are point masses in both
directions. Thus log r = lp(new)-lp(old)+log q(old_B)-log q(new_B).
No factorization or cancellation of the implemented prior is assumed.
Nonfinite generator output is recorded, never redrawn, and rejected without
model evaluation. Negative-infinite proposed prior is rejected without repair.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expit
from examples.lisa_dns_stage4.latent_logistic_gate import direct_blocks
from examples.lisa_dns_stage4.prior_refresh_audit import MODEL, block_indices, label_from_index

ELL9 = -110252.99476697217
ELL10 = -109401.11425363769
MATERIAL = 1e-8  # log-density units, fixed before probes
HIGH_ALPHA = .5  # descriptive geometry subset, not a viability threshold


def logq(x):
    """Stable analytic Logistic proposal density, never a model Jacobian."""
    x = np.asarray(x, dtype=np.float64)
    return float(-np.sum(np.logaddexp(0., x) + np.logaddexp(0., -x)))


def mh(lp_old, lp_new, q_old, q_new, ll_new, finite_draw=True):
    if not np.isfinite(lp_old):
        raise ValueError('Base prior must be finite')
    if not finite_draw:
        return np.nan, 0.
    if np.isnan(lp_new) or np.isposinf(lp_new):
        raise ValueError('Invalid implemented prior evaluation')
    ratio = lp_new - lp_old + q_old - q_new
    if np.isneginf(lp_new) or not np.isfinite(ll_new) or ll_new <= ELL9:
        return ratio, 0.
    return ratio, float(np.exp(min(0., ratio)))


def design(key, size, number=4096):
    """No state argument: independent label/draw streams and uniform bijection."""
    label_key, draw_key = jax.random.split(key)
    category = np.asarray(jax.random.randint(label_key, (number,), 0, 9 if size == 1 else 36))
    labels = np.array([label_from_index(int(i), size) for i in category])
    return category, labels, np.asarray(direct_blocks(draw_key, number, size))


def fixed_proposal(base, labels, draw):
    result = np.array(base, copy=True)
    result[block_indices(labels)] = np.asarray(draw).reshape(-1)
    return result


def quantiles(x):
    x = np.asarray(x)
    if not len(x):
        return dict(n=0, min=None, p05=None, median=None, p95=None, max=None)
    # Order-statistic quantiles avoid interpolation involving infinities.
    return dict(n=len(x), **dict(zip(['min','p05','median','p95','max'],
                                    np.quantile(x, [0,.05,.5,.95,1], method='nearest').tolist())))


def summary(a, mask):
    n = int(np.sum(mask)); s = int(np.sum(a['passes_ell9'][mask]))
    if not n:
        return dict(n=0, survivors=0, raw_survival=None, wilson95=None, mean_alpha=None)
    p = s/n; z = 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n, survivors=s, raw_survival=p, wilson95=[center-half,center+half],
                mean_alpha=float(a['alpha'][mask].mean()))


def safe(x):
    if isinstance(x, dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return str(x)
    return x


def main():
    from examples.lisa_dns_stage4 import audit
    from examples.lisa_dns_stage4.diagnose_ladder_failure import load_trace
    from examples.lisa_dns_stage4.diagnose_source_permutations import source_distances, NAMES, DF
    out = Path('/tmp/lisa_dns_stage4m_logistic_mh'); out.mkdir(exist_ok=False)
    checkpoint = Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    tracepath = Path('/tmp/lisa_dns_stage4h_continuation/trace.npz')
    boundpath = Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    payload = json.loads(checkpoint.read_text())['payload']
    assert payload['thresholds'][-1] == ELL9
    provenance = payload['provenance']
    problem, cfg, config_path = audit.load_problem()
    assert cfg == provenance['config']
    for p in [MODEL, config_path]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == provenance['file_sha256'][str(p)]
    protected = [MODEL, config_path, checkpoint, tracepath, boundpath,
                 Path(__file__), Path('examples/lisa_dns_stage4/proposals.py'), *Path('blackjax/ns').glob('dns*.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    t = load_trace(tracepath)
    indices = np.rint(np.linspace(0, t['position'].shape[1]-1, 16)).astype(int)
    assert len(set(indices)) == 16
    bases = t['position'][:,indices].reshape(128,54).copy()
    original = bases.copy()
    prior, like = map(jax.jit, audit.scalar_functions(problem))
    prior_map = jax.jit(lambda x: jax.lax.map(prior,x))
    like_map = jax.jit(lambda x: jax.lax.map(like,x))
    lp = np.asarray(prior_map(bases)); ll = np.asarray(like_map(bases))
    assert np.all(np.isfinite(bases)) and np.all(np.isfinite(lp)) and np.all(np.isfinite(ll))
    assert np.all(ll > ELL9)
    np.testing.assert_allclose(ll,t['loglikelihood'][:,indices].ravel(),rtol=0,atol=1e-7)
    np.testing.assert_allclose(lp,t['logprior'][:,indices].ravel(),rtol=0,atol=1e-9)
    np.savez_compressed(out/'bases.npz',position=bases,logprior=lp,loglikelihood=ll,
                        walker=np.repeat(np.arange(8),16),trace_index=np.tile(indices,8))
    bounds = json.loads(boundpath.read_text())['prior_audit'][0]['physical_bounds']
    lo = np.array([bounds[n][0] for n in NAMES]); hi = np.array([bounds[n][1] for n in NAMES])
    widths = hi-lo; widths[3]=np.pi; widths[4]=2*np.pi
    def physical(u): return lo+(hi-lo)*expit(np.asarray(u).reshape(9,6))
    def geometry(old,new,labels):
        ix = block_indices(labels); a=physical(old); b=physical(new)
        return dict(latent_block_l2=float(np.linalg.norm(new[ix]-old[ix])),
                    physical_f0_delta_hz=(b[labels,0]-a[labels,0]),
                    physical_f0_rms_bins=float(np.sqrt(np.mean((b[labels,0]-a[labels,0])**2))/DF),
                    source_set_frequency_rms_bins=source_distances(a[:,0,None],b[:,0,None],[DF])['assignment'],
                    source_set_full_rms=source_distances(a,b,widths,{3:np.pi,4:2*np.pi})['assignment'])
    report = dict(status='running',ell9=ELL9,ell10=ELL10,level10_remains_rejected=True,
        indices=indices,base_source=str(tracepath),base_count=128,proposals_per_base=32,
        seeds=dict(single=8501,pair=8502),material_log_density_tolerance=MATERIAL,
        high_alpha_geometry_cut=HIGH_ALPHA,devices=str(jax.devices()),jax_version=jax.__version__,
        uncertainty='Mean alpha is the sample mean of bounded probabilities; no binomial interval. Wilson intervals describe raw survival conditional on these fixed bases, are approximate for nonidentical probabilities, and do not quantify target coverage.',
        classification_policy='Qualitative fixed-budget assessment: consider exact mean alpha, all eight walkers, and movement surviving permutation matching relative to ordinary isotropic steps. No numerical acceptance cutoff; no calibration claim.',
        nonfinite_draw_policy='Reject without resampling or model calls; unevaluated values NaN, alpha zero; counted separately.',
        base_checks=dict(finite_prior=True,finite_likelihood=True,strict_contour=True,cache_consistent=True),
        file_sha256=hashes,results={})
    def save(): (out/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    save(); print('128 base checks passed',flush=True)
    for size, name in [(1,'single'),(2,'pair')]:
        categories, labels, draws = design(jax.random.key(8500+size),size)
        rows=[]
        for base_id, base in enumerate(bases):
            start=base_id*32; stop=start+32
            new=np.array([fixed_proposal(base,labels[j],draws[j]) for j in range(start,stop)])
            for k,j in enumerate(range(start,stop)):
                ix=block_indices(labels[j]); keep=np.setdiff1d(np.arange(54),ix)
                assert new[k,keep].tobytes()==base[keep].tobytes()
                np.testing.assert_array_equal(new[k,ix],draws[j].ravel())
                assert base.tobytes()==original[base_id].tobytes()
            finite=np.all(np.isfinite(new),axis=1)
            pnew=np.full(32,np.nan); lnew=np.full(32,np.nan)
            if finite.any():
                pnew[finite]=np.asarray(prior_map(new[finite]))
                lnew[finite]=np.asarray(like_map(new[finite]))
            for k,j in enumerate(range(start,stop)):
                ix=block_indices(labels[j]); qo=logq(base[ix]); qn=logq(new[k,ix])
                ratio, alpha=mh(lp[base_id],pnew[k],qo,qn,lnew[k],bool(finite[k]))
                row=dict(walker=base_id//16,base_id=base_id,base_state_index=int(indices[base_id%16]),
                    category=int(categories[j]),labels=labels[j],draw=draws[j],position=new[k],
                    current_logprior=lp[base_id],proposed_logprior=pnew[k],logq_old=qo,logq_new=qn,
                    log_mh_ratio=ratio,delta_correction=ratio,alpha=alpha,proposed_logL=lnew[k],
                    passes_ell9=bool(lnew[k]>ELL9),passes_ell10=bool(lnew[k]>ELL10),
                    delta_logL=lnew[k]-ll[base_id],nonfinite_draw=not bool(finite[k]),
                    analytic_logprior=logq(new[k]),prior_minus_analytic=pnew[k]-logq(new[k]))
                if finite[k]: row.update(geometry(base,new[k],labels[j]))
                else: row.update({key:np.full(size,np.nan) if key=='physical_f0_delta_hz' else np.nan for key in
                    ['latent_block_l2','physical_f0_delta_hz','physical_f0_rms_bins','source_set_frequency_rms_bins','source_set_full_rms']})
                rows.append(row)
            if (base_id+1)%16==0: print(name,'walker',base_id//16,'complete',flush=True)
        assert bases.tobytes()==original.tobytes()
        a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(out/f'{name}_proposals.npz',**a)
        survivor=a['passes_ell9']; valid=~a['nonfinite_draw']
        result=dict(overall=summary(a,np.ones(4096,dtype=bool)),
            by_walker=[summary(a,a['walker']==w) for w in range(8)],
            by_category=[dict(labels=label_from_index(i,size),**summary(a,a['category']==i)) for i in range(9 if size==1 else 36)],
            contour_survivor_alpha=quantiles(a['alpha'][survivor]),
            contour_survivor_log_correction=quantiles(a['delta_correction'][survivor]),
            delta_correction=quantiles(a['delta_correction'][valid]),
            ell10_raw_fraction=float(a['passes_ell10'].mean()),
            ell10_mh_weighted=float(np.mean(a['alpha']*a['passes_ell10'])),
            nonfinite_draw_count=int(a['nonfinite_draw'].sum()),
            nonfinite_draw_coordinate_count=int(np.sum(~np.isfinite(a['draw']))),
            negative_infinite_prior_count=int(np.isneginf(a['proposed_logprior']).sum()),
            nonfinite_likelihood_count=int(np.sum(valid & ~np.isfinite(a['proposed_logL']))),
            material_mismatch_count=int(np.sum(valid & (np.abs(a['prior_minus_analytic'])>MATERIAL))),
            finite_material_mismatch_count=int(np.sum(np.isfinite(a['proposed_logprior']) & (np.abs(a['prior_minus_analytic'])>MATERIAL))),
            prior_minus_analytic=quantiles(a['prior_minus_analytic'][valid]),geometry={})
        for group,mask in [('positive_alpha',a['alpha']>0),('high_alpha',a['alpha']>=HIGH_ALPHA)]:
            result['geometry'][group]={k:quantiles(a[k][mask]) for k in
                ['latent_block_l2','physical_f0_rms_bins','source_set_frequency_rms_bins','source_set_full_rms']}
        report['results'][name]=result;save()
    # Ordinary Stage-4H adjacent moves: same geometry and no new likelihood calls.
    with np.load('/tmp/lisa_dns_stage4h_continuation/trace.npz') as raw:
        components=raw['component'].reshape(8,-1)
    ordinary={}
    for size,name in [(1,'single'),(2,'pair')]:
        samples=[]
        for w in range(8):
            for i in range(1,t['position'].shape[1]):
                if components[w,i]!=size: continue
                old=t['position'][w,i-1];new=t['position'][w,i]
                changed=np.flatnonzero(np.any(old.reshape(9,6)!=new.reshape(9,6),axis=1))
                if len(changed)!=size: continue
                samples.append(geometry(old,new,changed))
        ordinary[name]={k:quantiles([s[k] for s in samples]) for k in
            ['latent_block_l2','physical_f0_rms_bins','source_set_frequency_rms_bins','source_set_full_rms']}
    report.update(ordinary_stage4h_geometry=ordinary,status='completed',structural_integrity_failures=0,
                  likelihood_calls=128+sum(4096-r['nonfinite_draw_count'] for r in report['results'].values()),
                  base_prior_minus_analytic=quantiles(lp-np.array([logq(b) for b in bases])))
    for path,digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    report['protected_hashes_unchanged']=True;save()
    print('Completed:',out,flush=True)


if __name__=='__main__': main()
