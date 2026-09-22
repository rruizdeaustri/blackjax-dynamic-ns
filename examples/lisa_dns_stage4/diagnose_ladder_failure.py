"""Stage-4H read-only trace diagnostics; no likelihood or MCMC calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp


def block_diagnostics(logL, f0_u, threshold, block_size=32, physical_bounds=None):
    """Keep walker and block axes separate; use strict exceedances, ddof=1."""
    ll=np.asarray(logL,dtype=float);f=np.asarray(f0_u,dtype=float)
    if ll.ndim!=2 or f.shape!=ll.shape+(9,) or not np.all(np.isfinite(ll)) or not np.all(np.isfinite(f)):
        raise ValueError('Expected finite walker/time and walker/time/9 arrays')
    if block_size<2 or ll.shape[1]%block_size:
        raise ValueError('Require complete blocks of at least two observations')
    b=ll.reshape(ll.shape[0],-1,block_size);u=f.reshape(f.shape[0],-1,block_size,9)
    result=dict(exceedance=(b>threshold).mean(2),logL_mean=b.mean(2),logL_median=np.median(b,axis=2),
                logL_min=b.min(2),logL_max=b.max(2),f0_u_mean=u.mean(2),f0_u_variance=u.var(2,ddof=1))
    if physical_bounds is not None:
        lo,hi=physical_bounds
        physical=lo+(hi-lo)*expit(u)
        result.update(f0_hz_mean=physical.mean(2),f0_hz_variance=physical.var(2,ddof=1))
    return result


def window_summary(ll,f,threshold,bounds):
    physical=bounds[0]+(bounds[1]-bounds[0])*expit(f)
    return dict(exceedance=(ll>threshold).mean(1),logL_mean=ll.mean(1),logL_median=np.median(ll,axis=1),
                logL_min=ll.min(1),logL_max=ll.max(1),f0_u_mean=f.mean(1),f0_u_variance=f.var(1,ddof=1),
                f0_hz_mean=physical.mean(1),f0_hz_variance=physical.var(1,ddof=1))


def pairwise(summary,ll):
    def diff(a): return a[:,None]-a[None,:]
    f=summary['f0_u_mean'];p=summary['f0_hz_mean']
    return dict(mean_logL_difference=diff(summary['logL_mean']),exceedance_difference=diff(summary['exceedance']),
        sourcewise_f0_u_difference=diff(f),sourcewise_f0_hz_difference=diff(p),
        mean_f0_u_euclidean_distance=np.linalg.norm(diff(f),axis=2),
        logL_KS_distance=np.array([[ks_2samp(a,b).statistic for b in ll] for a in ll]))


def early_late(logL,f0_u,threshold,burn_in=128,physical_bounds=(0.,1.)):
    ll=np.asarray(logL);f=np.asarray(f0_u)
    if burn_in<0 or ll.shape[1]<=burn_in or (ll.shape[1]-burn_in)%2:
        raise ValueError('Need two equal nonempty retained halves')
    mid=burn_in+(ll.shape[1]-burn_in)//2
    early=window_summary(ll[:,burn_in:mid],f[:,burn_in:mid],threshold,physical_bounds)
    late=window_summary(ll[:,mid:],f[:,mid:],threshold,physical_bounds)
    def spread(a):
        return dict(exceedance_sd=float(a['exceedance'].std(ddof=1)),
                    logL_mean_sd=float(a['logL_mean'].std(ddof=1)),
                    f0_mean_sd=a['f0_u_mean'].std(0,ddof=1))
    return dict(early=early,late=late,late_minus_early={k:late[k]-early[k] for k in early},
                early_spread=spread(early),late_spread=spread(late),
                early_late_logL_KS=np.array([ks_2samp(a,b).statistic for a,b in zip(ll[:,burn_in:mid],ll[:,mid:])]),
                late_pairwise=pairwise(late,ll[:,mid:]),
                f0_u_first_to_last=f[:,-1]-f[:,0])


def classify_evidence(*,coherent_convergence,persistent_separation,temporal_stability):
    """Qualitative evidence rubric, not a statistical test or ESS acceptance gate.

    Inputs must be justified by the full logL/frequency/time diagnostics.
    Mixed temporal behavior is explicitly ambiguous rather than forced into B.
    """
    if coherent_convergence and not persistent_separation:
        return 'A: likely insufficient burn-in'
    if persistent_separation and temporal_stability and not coherent_convergence:
        return 'B: likely persistent constrained mixing'
    return 'C: ambiguous'


def serial(value):
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def write_blocks(path,blocks,burn_in=128):
    rows=[]
    for w in range(blocks['exceedance'].shape[0]):
        for b in range(blocks['exceedance'].shape[1]):
            row=dict(walker=w,block=b+1,first_step=b*32+1,last_step=(b+1)*32,phase='burn_in' if b*32<burn_in else 'retained')
            for k,v in blocks.items():
                if v.ndim==2: row[k]=v[w,b]
                else:
                    for label in range(9): row[f'{k}_label{label}']=v[w,b,label]
            rows.append(row)
    with path.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def load_trace(path):
    with np.load(path) as data:
        n=len(data['loglikelihood'])//8
        np.testing.assert_array_equal(data['walker'],np.repeat(np.arange(8),n))
        np.testing.assert_array_equal(data['iteration'],np.tile(np.arange(n),8))
        return {k:data[k].reshape((8,n)+data[k].shape[1:]) for k in ['position','loglikelihood','logprior']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('/tmp/lisa_dns_stage4_ladder_batch'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/lisa_dns_stage4h_diagnosis'))
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    report=json.loads((args.input/'report.json').read_text());failed=report['rows'][-1]
    ell10=failed['threshold'];ell9=report['frozen_thresholds'][-1]
    assert failed['level']==10 and not failed['accepted'] and len(report['frozen_thresholds'])==10
    # Bounds are from the already audited real prior, not nominal decimal band endpoints.
    bounds=(0.001830003805175038,0.0018500126839167937)
    result=dict(ell9=ell9,ell10=ell10,physical_f0_bounds=bounds,banks={},across_levels=[],
                new_mcmc_steps=0,notice='Evidence unvalidated/out of scope. Descriptive MCMC diagnostics; no family classification.',
                file_sha256={str(args.input/'report.json'):hashlib.sha256((args.input/'report.json').read_bytes()).hexdigest()})
    for level in [9,10]:
        for name in ['selection','calibration']:
            path=args.input/f'level{level}_{name}_trace.npz';trace=load_trace(path)
            result['file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            ll=trace['loglikelihood'];f=trace['position'][:,:,::6]
            contour=report['frozen_thresholds'][level-1]
            assert np.all(ll>contour) and np.all(np.isfinite(trace['logprior']))
            blocks=block_diagnostics(ll,f,ell10,physical_bounds=bounds)
            key=f'level{level}_{name}'
            result['banks'][key]=dict(generating_contour=contour,blocks=blocks,
                comparison=early_late(ll,f,ell10,physical_bounds=bounds))
            write_blocks(args.output/f'{key}_blocks.csv',blocks)
    for row in report['rows'][:5]:
        level=row['level'];s=row['selection'];c=row['calibration']
        delta=c['ratio']-s['ratio'];se=np.hypot(s['standard_error'],c['standard_error'])
        entry=dict(level=level,selection_ratio=s['ratio'],calibration_ratio=c['ratio'],difference=delta,
            selection_se=s['standard_error'],calibration_se=c['standard_error'],
            approximate_combined_se=se,difference_over_combined_se=delta/se,
            calibration_interval=c['interval'],selection_interval=s['interval'],
            calibration_between_walker_se=row['calibration_uncertainty']['between_walker_se'],
            calibration_walker_fraction_sd=np.std(c['walker_fractions'],ddof=1),banks={})
        arrays=[]
        for name in ['selection','calibration']:
            path=args.input/f'level{level}_{name}_trace.npz';trace=load_trace(path)
            result['file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            ll=trace['loglikelihood'][:,128:];arrays.append(ll)
            entry['banks'][name]=dict(pooled_logL_mean=float(ll.mean()),pooled_logL_median=float(np.median(ll)),
                walker_logL_means=ll.mean(1),early_late_logL_mean_change=ll[:,128:].mean(1)-ll[:,:128].mean(1),
                walker_logL_mean_sd=float(ll.mean(1).std(ddof=1)))
        entry['pooled_logL_KS_descriptive']=ks_2samp(arrays[0].ravel(),arrays[1].ravel()).statistic
        result['across_levels'].append(entry)
    result['uncertainty_notice']='Combined SE is a descriptive compatibility scale: selection threshold is data-dependent and levels share ancestry. It is not an independent z-test or a global bias test. Core intervals are approximate Student intervals.'
    (args.output/'existing_trace_analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for w in range(8):
        b=result['banks']['level10_calibration']['blocks']
        axes[0].plot(np.arange(1,13),b['exceedance'][w],marker='o',label=f'walker {w}')
        axes[1].plot(np.arange(1,13),b['logL_mean'][w],marker='o')
    axes[0].set_ylabel('Strict exceedance fraction');axes[1].set_ylabel('Mean log likelihood')
    for ax in axes: ax.axvline(4.5,color='k',ls='--');ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[1].set_xlabel('32-step block (1–4 original burn-in; 5–12 retained)')
    fig.tight_layout();fig.savefig(args.output/'existing_calibration_blocks.png',dpi=160);plt.close(fig)
    print('Saved read-only diagnostics:',args.output,flush=True)


if __name__=='__main__': main()
