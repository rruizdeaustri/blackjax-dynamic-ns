"""Read-only comparison of completed Stage-4H and Stage-4J traces."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from examples.lisa_dns_stage4.diagnose_ladder_failure import load_trace,serial
from examples.lisa_dns_stage4.diagnose_source_permutations import cross_frequency_distance,DF
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information


def summarize(path,starts,ell):
    trace=load_trace(path/'trace.npz');ll=trace['loglikelihood'];u=trace['position'][:,:,::6]
    lower=.001830003805175038;upper=.0018500126839167937
    f=lower+(upper-lower)*expit(u)
    initial=lower+(upper-lower)*expit(starts[:,::6])
    metrics={}
    with np.load(path/'trace.npz') as raw:
        steps=raw['num_steps'].reshape(8,512);shrink=raw['num_shrink'].reshape(8,512)
        for name,window in [('first256',slice(0,256)),('second256',slice(256,512)),('all512',slice(0,512))]:
            values=ll[:,window];coords=u[:,window];freq=f[:,window]
            fractions=(values>ell).mean(1);means=values.mean(1)
            ess=np.array([[autocorrelation_information(v)['ess'] for v in x.T] for x in coords])
            pairs=[dict(walker_i=i,walker_j=j,**cross_frequency_distance(freq[i],freq[j])) for i in range(8) for j in range(i+1,8)]
            ns=steps[:,window];nh=shrink[:,window]
            metrics[name]=dict(walker_exceedance=fractions,exceedance_sd=float(fractions.std(ddof=1)),
                walker_logL_mean=means,logL_mean_sd=float(means.std(ddof=1)),
                f0_u_ess_median=float(np.median(ess)),f0_u_ess_minimum=float(ess.min()),f0_u_ess=ess,
                pairwise_frequency_distances=pairs,
                pairwise_frequency_rms_median=float(np.median([p['assignment'] for p in pairs])),
                pairwise_frequency_mean_distance_median=float(np.median([p['sorted_mean_distance'] for p in pairs])),
                likelihood_proxy=int((ns+nh).sum()),num_steps_mean=float(ns.mean()),num_shrink_mean=float(nh.mean()),
                num_steps_quantiles=np.quantile(ns,[.5,.9,1.]),num_shrink_quantiles=np.quantile(nh,[.5,.9,1.]))
    first=metrics['first256'];second=metrics['second256']
    metrics['early_to_late_logL_change']=second['walker_logL_mean']-first['walker_logL_mean']
    metrics['early_to_late_exceedance_change']=second['walker_exceedance']-first['walker_exceedance']
    metrics['source_set_movement']=[dict(walker=w,**cross_frequency_distance(f[w,:256],f[w,256:]),
        start_to_end_sorted_rms_bins=float(np.sqrt(np.mean((np.sort(initial[w])-np.sort(f[w,-1]))**2))/DF),
        first_within_rms=cross_frequency_distance(f[w,:256],f[w,:256])['assignment'],
        second_within_rms=cross_frequency_distance(f[w,256:],f[w,256:])['assignment']) for w in range(8)]
    metrics['final_state_pairwise_frequency_distances']=[dict(walker_i=i,walker_j=j,**cross_frequency_distance(f[i,-1:],f[j,-1:])) for i in range(8) for j in range(i+1,8)]
    return metrics


def main():
    output=Path('/tmp/lisa_dns_stage4j_differential')
    paths=dict(isotropic=Path('/tmp/lisa_dns_stage4h_continuation'),differential=output)
    reports={n:json.loads((p/'report.json').read_text()) for n,p in paths.items()}
    assert reports['differential']['status']=='completed_frozen_differential_benchmark'
    with np.load(paths['isotropic']/'initial_states.npz') as a,np.load(output/'initial_states.npz') as b:
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        starts=a['position']
    result={n:summarize(p,starts,reports[n]['ell10']) for n,p in paths.items()}
    result['audits']={n:{k:r[k] for k in ['cost','failures']} for n,r in reports.items()}
    result['fallback_count']=reports['differential']['fallback_count']
    result['max_direction_norm_error']=reports['differential']['max_direction_norm_error']
    result['baseline_rerun']=False
    (output/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    for n in paths:
        for window in ['first256','second256','all512']:
            m=result[n][window]
            print(n,window,'p SD',m['exceedance_sd'],'logL SD',m['logL_mean_sd'],'f0ESS',m['f0_u_ess_median'],m['f0_u_ess_minimum'],'cost',m['likelihood_proxy'])


if __name__=='__main__':main()
