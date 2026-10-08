"""Stage 9F saved-data diagnostics only. Never import or invoke a sampler."""
import os
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')
import hashlib
import json
from pathlib import Path
import numpy as np
import arviz as az
from scipy import stats
from scipy.spatial.distance import cdist

REPORT = Path('docs/examples/dns_conditional_reliability_gate.json')
BURN, K, J = 1024, 2048, 4


def diagnostics(y):
    """ArviZ standard rank/folded split Rhat and bulk/tail ESS; chains x draws."""
    if np.ptp(y) == 0:
        return {'constant': True}
    n = y.shape[1] // 2
    split = np.concatenate([y[:, :n], y[:, -n:]])
    within = np.var(split, axis=1, ddof=1).mean()
    between = n * np.var(split.mean(axis=1), ddof=1)
    return dict(rhat=float(az.rhat(y, method='rank')),
                bulk_ess=float(az.ess(y, method='bulk')),
                tail_ess=float(az.ess(y, method='tail')),
                between_within_ratio=float(between / within))


def holm(pvalues, alpha=.05):
    order = sorted(pvalues, key=pvalues.get)
    rejected = []
    for i, name in enumerate(order):
        if pvalues[name] > alpha / (len(order)-i):
            break
        rejected.append(name)
    return rejected


def energy(x, y):
    """Standard biased empirical energy distance, supports arbitrary theta dimension."""
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def compression_ess(ll, threshold):
    # Original frozen Bernoulli / block64 / between-walker maximum-SE convention.
    z = (ll > threshold).astype(float)
    p = z.mean()
    variance = p*(1-p)
    blocks = z.reshape(8, K//64, 64).mean(axis=2)
    se2 = max(variance/z.size, blocks.var(ddof=1)/blocks.size,
              z.mean(axis=1).var(ddof=1)/8)
    return float(min(z.size, variance/se2)) if se2 else 0.


def evaluate(path, threshold, model):
    with np.load(path) as f:
        theta = f['position'].copy()
        j = f['assigned_level'].copy()
        if 'loglikelihood' in f:
            ll = f['loglikelihood'].copy()
        else:
            # Evaluate the supplied model likelihood, without any analytic reference.
            centers, widths = np.array(model['centers']), np.array(model['widths'])
            ll = (-.5*((theta[...,0,None]-centers)/widths)**2).max(axis=-1)
            ll -= np.log(widths.sum()*np.sqrt(2*np.pi))
        lp = f['logdensity'].copy() if 'logdensity' in f else np.full(j.shape,-np.log(16))
    indices = [np.flatnonzero(j[BURN:,w]==J)+BURN for w in range(8)]
    quota = all(len(i)>=K for i in indices)
    assert quota, 'Missing quota must fail; never pad saved data'
    retained = np.stack([i[:K] for i in indices])
    horizon = int(retained[:,-1].max()+1)
    mid = (BURN+horizon)//2
    d = theta.shape[-1]
    obs = {f'theta_{i}':theta[...,i] for i in range(d)}
    obs['log_likelihood'] = ll
    if np.ptp(lp): obs['log_prior'] = lp
    projections = {'projection_sum': theta.sum(axis=-1)/np.sqrt(d)}
    if d > 1:
        projections['projection_alternating'] = (theta * (-1.)**np.arange(d)).sum(axis=-1)/np.sqrt(d)
    obs.update(projections)
    gate_names = list(projections)+['log_likelihood']+(['log_prior'] if np.ptp(lp) else [])
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  quota=quota, horizon=horizon, physical_split=mid, observables={}, triggers=[])
    pvals = {}
    for name, values in obs.items():
        visits = np.stack([values[retained[w],w] for w in range(8)])
        halves = [[values[a:b,w][j[a:b,w]==J] for w in range(8)]
                  for a,b in [(BURN,mid),(mid,horizon)]]
        delta = np.array([halves[1][w].mean()-halves[0][w].mean() for w in range(8)])
        pvals[name] = float(stats.ttest_1samp(delta,0).pvalue) if np.ptp(delta) else (1. if delta[0]==0 else 0.)
        n = min(len(v) for half in halves for v in half)
        physical = np.stack([v[:n] for half in halves for v in half])
        second = all(len(i)>=2*K for i in indices)
        row = dict(retained=diagnostics(visits), physical_balanced=diagnostics(physical),
                   physical_counts=[[len(v) for v in half] for half in halves],
                   physical_paired_t_p=pvals[name], physical_mean_differences=delta.tolist(),
                   retained_half_ks=float(stats.ks_2samp(visits[:,:K//2].ravel(),visits[:,K//2:].ravel()).statistic),
                   physical_half_ks=float(stats.ks_2samp(np.concatenate(halves[0]),np.concatenate(halves[1])).statistic),
                   walker_pair_ks_max=max(float(stats.ks_2samp(visits[a],visits[b]).statistic) for a in range(8) for b in range(a)),
                   first_second_K_ks=float(stats.ks_2samp(visits.ravel(),np.concatenate([values[i[K:2*K],w] for w,i in enumerate(indices)])).statistic) if second else None)
        result['observables'][name]=row
        if name in gate_names:
            metrics=row['retained']
            if 'constant' not in metrics:
                for metric, bad in [('rhat',metrics['rhat']>=1.01),('bulk_ess',metrics['bulk_ess']<800),('tail_ess',metrics['tail_ess']<800)]:
                    if bad: result['triggers'].append(name+':'+metric)
    result['holm_rejections']=holm(pvals)
    result['triggers'] += ['physical_Holm:'+name for name in result['holm_rejections']]
    result['assigned_level'] = {'conditioned_top': {'constant':True,'value':J},
                                'uncensored_physical':diagnostics(j[BURN:horizon].T)}
    result['log_prior_constant'] = not bool(np.ptp(lp))
    # Fixed computational cap: deterministic evenly spaced subindices; distance only.
    x=np.concatenate([theta[BURN:mid,w][j[BURN:mid,w]==J] for w in range(8)])
    y=np.concatenate([theta[mid:horizon,w][j[mid:horizon,w]==J] for w in range(8)])
    result['physical_energy_distance']=energy(x[np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)],y[np.linspace(0,len(y)-1,min(512,len(y)),dtype=int)])
    v0=np.concatenate([theta[i[:K//2],w] for w,i in enumerate(indices)])
    v1=np.concatenate([theta[i[K//2:K],w] for w,i in enumerate(indices)])
    result['visit_energy_distance']=energy(v0[::max(1,len(v0)//512)],v1[::max(1,len(v1)//512)])
    top_ll=np.stack([ll[retained[w],w] for w in range(8)])
    result['compression_ess']=compression_ess(top_ll,threshold)
    if result['compression_ess']<20: result['triggers'].append('compression_ess')
    result['pass']=quota and not result['triggers']
    return result


def main():
    design=json.loads(REPORT.read_text())
    frozen={k:v for k,v in design.items() if k not in ('results','summary','implementation')}
    frozen_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    spec=json.loads(Path('docs/examples/dns_top_level_stationarity_design.json').read_text())
    paths=[('stage9d_failed_calibration',Path('/tmp/dns-stage9d-heldout/original/attempt_000004/calibration_trace.npz')),
           ('stage9d_selection',Path('/tmp/dns-stage9d-heldout/original/attempt_000004/selection_trace.npz'))]
    paths += [(p.stem,p) for p in sorted(Path('/tmp/dns-stage9e-stationarity').glob('*.npz'))]
    rows={name:evaluate(p,spec['candidate_threshold'],spec['model']) for name,p in paths}
    stationary=[r for name,r in rows.items() if 'exact_joint_stationary' in name]
    promoted=[r for name,r in rows.items() if 'saved_promoted_start' in name]
    count=sum(not r['pass'] for r in stationary)
    failure=not rows['stage9d_failed_calibration']['pass']
    case='B' if count>=4 else ('C' if not failure else 'E')
    design.update(results=rows,implementation={'arviz_version':az.__version__,'frozen_design_sha256':frozen_hash,
                  'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  summary={'historical_failure_detected':failure,'stationary_rejected':count,'stationary_total':8,
                           'false_rejection_fraction':count/8,'promoted_rejected':sum(not r['pass'] for r in promoted),
                           'case':case,'heldout_validation_justified':False,'development_only':True})
    REPORT.write_text(json.dumps(design,indent=2,allow_nan=False)+'\n')
    print(json.dumps(design['summary']))
    for name,r in rows.items(): print(name,r['pass'],r['triggers'])

if __name__=='__main__': main()
