"""Stage-4I: analysis-only source-set matching of saved states; no model calls."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from scipy.stats import spearmanr
from examples.lisa_dns_stage4.diagnose_ladder_failure import load_trace,serial

NAMES=('f0','fdot','iota','psi','lam','beta')
DF=1/31536000.


def circular_difference(a,b,period):
    if period<=0: raise ValueError('Positive period required')
    return (np.asarray(a)-np.asarray(b)+period/2)%period-period/2


def source_distances(a,b,widths,periods=None):
    """RMS normalized coordinate distance, with labelled/sorted/optimal pairings."""
    a=np.asarray(a);b=np.asarray(b);widths=np.asarray(widths)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=len(widths) or np.any(widths<=0):
        raise ValueError('Equal source catalogues and positive coordinate widths required')
    delta=a[:,None,:]-b[None,:,:]
    for k,period in (periods or {}).items():
        delta[:,:,k]=circular_difference(a[:,None,k],b[None,:,k],period)
    cost=np.mean((delta/widths)**2,axis=2)
    rows,cols=linear_sum_assignment(cost)
    ai=np.argsort(a[:,0]);bi=np.argsort(b[:,0])
    labelled=float(np.sqrt(np.mean(np.diag(cost))))
    assignment=float(np.sqrt(np.mean(cost[rows,cols])))
    return dict(labelled=labelled,sorted_f0=float(np.sqrt(np.mean(cost[ai,bi]))),
                assignment=assignment,assignment_over_labelled=assignment/labelled if labelled else None,
                permutation=cols.tolist())


def sorted_frequency_summary(f):
    """Sort each actual catalogue before computing source-rank statistics."""
    f=np.asarray(f)
    if f.ndim!=2: raise ValueError('Expected time/source array')
    s=np.sort(f,axis=-1)
    return dict(mean=s.mean(0),variance=s.var(0,ddof=1))


def cross_frequency_distance(a,b):
    """Exact RMS over all cross-time pairs in frequency-bin units.

    One-dimensional quadratic optimal assignment equals sorting each catalogue.
    Compute labelled and sorted distances from first and second moments.
    """
    a=np.asarray(a)/DF;b=np.asarray(b)/DF
    def squared(x,y):
        # Center near zero to avoid cancellation at millihertz / narrow bin scale.
        origin=(x.mean()+y.mean())/2;x=x-origin;y=y-origin
        return max(0.,float(np.mean(x*x)+np.mean(y*y)-2*np.mean(x.mean(0)*y.mean(0))))
    label=np.sqrt(squared(a,b));sa=np.sort(a,axis=1);sb=np.sort(b,axis=1)
    matched=np.sqrt(squared(sa,sb))
    return dict(labelled=float(label),sorted_f0=float(matched),assignment=float(matched),
                assignment_over_labelled=float(matched/label) if label else None,
                sorted_mean_distance=float(np.sqrt(np.mean((sa.mean(0)-sb.mean(0))**2))))


def pair_window(f,physical,ll,ell,widths):
    """Frequency: all cross-time pairs. Full coordinates: every 32nd endpoint."""
    fractions=(ll>ell).mean(1);means=ll.mean(1);pairs=[]
    periods={3:np.pi,4:2*np.pi}
    for i in range(8):
        for j in range(i+1,8):
            freq=cross_frequency_distance(f[i],f[j])
            samples=[source_distances(a,b,widths,periods) for a in physical[i,31::32] for b in physical[j,31::32]]
            full={k:float(np.sqrt(np.mean([s[k]**2 for s in samples]))) for k in ['labelled','sorted_f0','assignment']}
            full['assignment_over_labelled']=full['assignment']/full['labelled']
            pairs.append(dict(walker_i=i,walker_j=j,**freq,full_coordinate=full,
                logL_mean_difference=float(means[i]-means[j]),exceedance_difference=float(fractions[i]-fractions[j])))
    return pairs


def main():
    out=Path('/tmp/lisa_dns_stage4i_permutations');out.mkdir(exist_ok=False)
    oldpath=Path('/tmp/lisa_dns_stage4_ladder_batch/level10_calibration_trace.npz')
    contpath=Path('/tmp/lisa_dns_stage4h_continuation/trace.npz')
    priorpath=Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    runpath=Path('/tmp/lisa_dns_stage4h_continuation/report.json')
    prior=json.loads(priorpath.read_text());bounds=prior['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES])
    widths=hi-lo
    # Full metric uses one period for circular coordinates, prior widths otherwise.
    widths[3]=np.pi;widths[4]=2*np.pi
    run=json.loads(runpath.read_text());ell=run['ell10']
    result=dict(ell9=run['ell9'],ell10=ell,coordinate_names=NAMES,physical_bounds=bounds,
        frequency_unit_hz=DF,full_coordinate_widths=widths,angular_periods=dict(psi=np.pi,lam=2*np.pi),
        methodology='Frequency RMS in Fourier bins, all 256x256 cross-time catalogue pairs per walker pair; sorted and Hungarian frequency-only optima coincide. Supplementary full-coordinate RMS uses equal normalized coordinate weights, 8 block-end states per half, all 8x8 cross-state pairs. No fitted weights.',
        likelihood_calls=0,mcmc_steps=0,windows={},temporal=[],
        residual_notice='Saved traces have no residual_chi2/source_gain. These require model evaluation and were skipped; no new likelihood calls.',
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [oldpath,contpath,priorpath,runpath,Path(__file__)]})
    arrays={}
    for name,path in [('failed',oldpath),('continuation',contpath)]:
        t=load_trace(path);assert np.all(t['loglikelihood']>run['ell9'])
        physical=lo+(hi-lo)*expit(t['position'].reshape(8,-1,9,6))
        arrays[name]=(physical,t['loglikelihood'])
        # Persist all decoded physical samples; no waveform or amplitude/phase calls.
        np.savez_compressed(out/f'{name}_decoded.npz',physical=physical,names=NAMES,loglikelihood=t['loglikelihood'])
        rows=[]
        for w in range(8):
            for block in range(physical.shape[1]//32):
                x=physical[w,block*32:(block+1)*32];s=np.sort(x[:,:,0],axis=1)
                row=dict(walker=w,block=block+1,exceedance=float((t['loglikelihood'][w,block*32:(block+1)*32]>ell).mean()))
                for k in range(9):
                    row[f'sorted_f0_mean_{k}']=s[:,k].mean();row[f'sorted_f0_variance_{k}']=s[:,k].var(ddof=1)
                rows.append(row)
        with (out/f'{name}_sorted_blocks.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for name,source,part in [('failed_retained','failed',slice(128,384)),('continuation_first','continuation',slice(0,256)),('continuation_second','continuation',slice(256,512))]:
        physical,ll=arrays[source];physical=physical[:,part];ll=ll[:,part];f=physical[:,:,:,0]
        pairs=pair_window(f,physical,ll,ell,widths)
        result['windows'][name]=dict(pairs=pairs,walker_exceedance=(ll>ell).mean(1),walker_logL_mean=ll.mean(1),
            sorted_frequency_means=np.sort(f,axis=2).mean(1),
            sorted_frequency_variances=np.sort(f,axis=2).var(1,ddof=1),
            within_walker_frequency_pair_rms=[cross_frequency_distance(x,x)['assignment'] for x in f],
            frequency_ratio_median=float(np.median([p['assignment_over_labelled'] for p in pairs])),
            frequency_ratio_range=[min(p['assignment_over_labelled'] for p in pairs),max(p['assignment_over_labelled'] for p in pairs)],
            full_ratio_median=float(np.median([p['full_coordinate']['assignment_over_labelled'] for p in pairs])),
            distance_vs_abs_logL_spearman=float(spearmanr([p['assignment'] for p in pairs],[abs(p['logL_mean_difference']) for p in pairs]).statistic),
            distance_vs_abs_exceedance_spearman=float(spearmanr([p['assignment'] for p in pairs],[abs(p['exceedance_difference']) for p in pairs]).statistic))
    physical,ll=arrays['continuation'];f=physical[:,:,:,0]
    for w in range(8):
        a=f[w,:256];b=f[w,256:];sa=np.sort(a,axis=1);sb=np.sort(b,axis=1)
        # Continuation transitions are assessed on catalogue sets, not source identities.
        result['temporal'].append(dict(walker=w,**cross_frequency_distance(a,b),
            rank_mean_shift_bins=(sb.mean(0)-sa.mean(0))/DF,
            within_first_rms=cross_frequency_distance(a,a)['assignment'],
            within_second_rms=cross_frequency_distance(b,b)['assignment'],
            adjacent_block_sorted_mean_distances_bins=[float(np.sqrt(np.mean((np.sort(f[w,i:i+32],axis=1).mean(0)-np.sort(f[w,i-32:i],axis=1).mean(0))**2))/DF) for i in range(32,512,32)],
            instantaneous_sorted_step_rms_bins_quantiles=np.quantile(np.sqrt(np.mean(np.diff(np.sort(f[w],axis=1),axis=0)**2,axis=1))/DF,[.5,.9,1.])))
    endpoints=[]
    for i in range(8):
        for j in range(i+1,8):
            endpoints.append(dict(walker_i=i,walker_j=j,frequency=source_distances(f[i,-1,:,None],f[j,-1,:,None],[DF]),
                full_coordinate=source_distances(physical[i,-1],physical[j,-1],widths,{3:np.pi,4:2*np.pi})))
    result['final_state_pairs']=endpoints
    # Independently check Hungarian implementation against sorted optimum for actual states.
    for p in endpoints:
        np.testing.assert_allclose(p['frequency']['assignment'],p['frequency']['sorted_f0'],rtol=1e-12,atol=1e-10)
    (out/'report.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    for name,window in result['windows'].items():
        rows=[]
        for p in window['pairs']:
            row={k:v for k,v in p.items() if k!='full_coordinate'}
            row.update({f'full_{k}':v for k,v in p['full_coordinate'].items()});rows.append(row)
        with (out/f'{name}_pairs.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print('Saved analysis-only source-set diagnostics:',out)


if __name__=='__main__':main()
