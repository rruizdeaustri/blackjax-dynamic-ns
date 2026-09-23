"""Read-only Stage-4N comparison; no model evaluation or chain extension."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4.logistic_mh_refresh import ELL9, ELL10, MATERIAL, quantiles, safe, logq, mh
from examples.lisa_dns_stage4.prior_refresh_audit import label_from_index, block_indices
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    source_distances,cross_frequency_distance,NAMES,DF)
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac

OUT=Path('/tmp/lisa_dns_stage4n_sequential')


def load(path):
    with np.load(path) as data: return {k:data[k].copy() for k in data.files}


def agreement(ll,u):
    fractions=np.array([(x>ELL10).mean() for x in ll]);means=np.array([x.mean() for x in ll])
    ess=np.array([[ac(v)['ess'] for v in x.T] for x in u])
    llac=[ac(x) for x in ll]
    ks=[dict(walker_i=i,walker_j=j,distance=float(ks_2samp(ll[i],ll[j],method='asymp').statistic))
        for i in range(8) for j in range(i+1,8)]
    return dict(walker_exceedance=fractions,exceedance_sd=float(fractions.std(ddof=1)),
        walker_logL_mean=means,logL_mean_sd=float(means.std(ddof=1)),
        walker_logL_median=[np.median(x) for x in ll],
        walker_logL_MAD=[np.median(np.abs(x-np.median(x))) for x in ll],
        walker_logL_IQR=[np.quantile(x,.75)-np.quantile(x,.25) for x in ll],
        logL_KS_pairs=ks,logL_KS_distribution=quantiles([x['distance'] for x in ks]),
        f0_u_ess=ess,f0_u_ess_median=float(np.median(ess)),f0_u_ess_minimum=float(ess.min()),
        logL_autocorrelation=llac,logL_ess_median=float(np.median([x['ess'] for x in llac])),
        logL_ess_minimum=float(min(x['ess'] for x in llac)))


def acceptance(a,mask):
    n=int(mask.sum()); survivors=int(np.sum(a['passes_ell9'][mask]));accepted=int(np.sum(a['accepted'][mask]))
    return dict(n=n,accepted=accepted,acceptance=accepted/n if n else None,
        mean_alpha=float(a['alpha'][mask].mean()) if n else None,contour_survivors=survivors,
        contour_failure_fraction=float(np.mean(~a['passes_ell9'][mask])) if n else None,
        mh_rejections_among_survivors=int(np.sum(a['passes_ell9'][mask]&~a['accepted'][mask])),
        mh_rejection_fraction_among_survivors=(survivors-accepted)/survivors if survivors else None)


def main():
    run=json.loads((OUT/'report.json').read_text());assert run['status']=='completed'
    starts=load(OUT/'initial_states.npz')
    base=Path('/tmp/lisa_dns_stage4h_continuation')
    baseline_report=json.loads((base/'report.json').read_text())
    bounds_path=Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    bounds=json.loads(bounds_path.read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    paths=dict(isotropic=base/'trace.npz',single=OUT/'single/trace.npz',pair=OUT/'pair/trace.npz')
    result=dict(ell9=ELL9,ell10=ELL10,level10_remains_rejected=True,baseline_rerun=False,
        metrics={},file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths.values(),bounds_path,Path(__file__)]},
        ess_notice='Short-chain initial-positive monotone autocorrelation estimate, not a convergence test. Constants have ESS zero; labelled-coordinate ESS is not physical convergence.',
        physical_notice='Exact all-cross-time sorted-frequency RMS and sorted mean distances; full-coordinate Hungarian distances on every 32nd window endpoint (8 per half), same periods/normalizations as Stage-4I. Broad within-walker variation can increase cross-time RMS despite better distributional agreement.',
        cost_notice='Refresh model-call counts are instrumented actual calls. Baseline slice cost is the saved num_steps+num_shrink proxy; true slice likelihood/prior totals and comparable wall timing were not saved. Cache checks are reported separately.')
    raws={}
    for name,path in paths.items():
        a=load(path);raws[name]=a
        np.testing.assert_array_equal(a['walker'],np.repeat(np.arange(8),512))
        np.testing.assert_array_equal(a['iteration'],np.tile(np.arange(512),8))
        x=a['position'].reshape(8,512,54);ll=a['loglikelihood'].reshape(8,512)
        assert np.isfinite(x).all() and np.isfinite(ll).all() and np.all(ll>ELL9)
        phys=lo+(hi-lo)*expit(x.reshape(8,512,9,6));f=phys[:,:,:,0]
        m={}
        for window,sl in [('first256',slice(0,256)),('second256',slice(256,512)),('all512',slice(0,512))]:
            v=ll[:,sl];u=x[:,sl,::6];freq=f[:,sl];physical=phys[:,sl]
            s=agreement(v,u)
            s['unique_state_fraction']=[len(np.unique(z,axis=0))/len(z) for z in x[:,sl]]
            pairs=[]
            for i in range(8):
                for j in range(i+1,8):
                    full=[source_distances(p,q,widths,{3:np.pi,4:2*np.pi})['assignment']
                          for p in physical[i,31::32] for q in physical[j,31::32]]
                    pairs.append(dict(walker_i=i,walker_j=j,**cross_frequency_distance(freq[i],freq[j]),
                        full_assignment_rms=float(np.sqrt(np.mean(np.square(full))))))
            s['source_set_pairs']=pairs
            for key in ['assignment','sorted_mean_distance','full_assignment_rms']:
                s[f'pairwise_{key}']=quantiles([p[key] for p in pairs])
            s['within_walker_frequency_rms_bins']=[cross_frequency_distance(z,z)['assignment'] for z in freq]
            s['within_walker_full_rms']=[float(np.sqrt(np.mean([
                source_distances(p,q,widths,{3:np.pi,4:2*np.pi})['assignment']**2
                for p in z[31::32] for q in z[31::32]]))) for z in physical]
            if name!='isotropic':
                accepted=a['accepted'].reshape(8,512)[:,sl]
                s['acceptance_autocorrelation']=[ac(z) for z in accepted]
                s['proposal_likelihood_calls']=int(np.sum(~a['nonfinite_draw'].reshape(8,512)[:,sl]))
            else:
                s['slice_likelihood_proxy']=int((a['num_steps']+a['num_shrink']).reshape(8,512)[:,sl].sum())
            m[window]=s
        m['early_late_logL_KS']=[float(ks_2samp(z[:256],z[256:],method='asymp').statistic) for z in ll]
        initial=lo+(hi-lo)*expit(starts['position'].reshape(8,9,6))
        m['trajectory_source_set_movement']=[dict(walker=w,
            between_halves=cross_frequency_distance(f[w,:256],f[w,256:]),
            start_to_end_frequency=source_distances(initial[w,:,0,None],f[w,-1,:,None],[DF]),
            start_to_end_full=source_distances(initial[w],phys[w,-1],widths,{3:np.pi,4:2*np.pi}),
            max_frequency_distance_from_start=max(source_distances(initial[w,:,0,None],z[:,0,None],[DF])['assignment'] for z in phys[w]))
            for w in range(8)]
        geometry=[]
        for w in range(8):
            old=starts['position'][w]
            for i in range(512):
                new=x[w,i];index=w*512+i
                accepted=bool(a['accepted'][index]) if name!='isotropic' else not np.array_equal(old,new)
                if accepted:
                    p=lo+(hi-lo)*expit(old.reshape(9,6));q=phys[w,i]
                    changed=np.flatnonzero(np.any(old.reshape(9,6)!=new.reshape(9,6),axis=1))
                    geometry.append(dict(latent_l2=float(np.linalg.norm(new-old)),
                        physical_f0_rms_bins=float(np.sqrt(np.mean((q[changed,0]-p[changed,0])**2))/DF),
                        source_set_frequency_rms_bins=source_distances(p[:,0,None],q[:,0,None],[DF])['assignment'],
                        source_set_full_rms=source_distances(p,q,widths,{3:np.pi,4:2*np.pi})['assignment'],
                        component=int(a['component'][index]) if name=='isotropic' else (1 if name=='single' else 2)))
                old=new
        keys=['latent_l2','physical_f0_rms_bins','source_set_frequency_rms_bins','source_set_full_rms']
        m['accepted_move_geometry']={k:quantiles([z[k] for z in geometry]) for k in keys}
        if name=='isotropic':
            m['accepted_move_geometry_by_component']={str(c):{k:quantiles([z[k] for z in geometry if z['component']==c]) for k in keys} for c in range(3)}
            m['cost']=dict(slice_likelihood_proxy=baseline_report['cost']['likelihood_proxy'],
                direct_prior_cache_checks=baseline_report['cost']['direct_cache_checks'],
                direct_likelihood_cache_checks=baseline_report['cost']['direct_cache_checks'],
                actual_total_prior=None,actual_total_likelihood=None,wall_seconds=None)
        else:
            size=1 if name=='single' else 2
            m['acceptance']=dict(overall=acceptance(a,np.ones(4096,dtype=bool)),
                by_walker=[acceptance(a,a['walker']==w) for w in range(8)],
                by_category=[dict(labels=label_from_index(c,size),**acceptance(a,a['category']==c)) for c in range(9 if size==1 else 36)])
            m['tail_counts']=dict(negative_infinite_prior=int(np.isneginf(a['proposed_logprior']).sum()),
                nonfinite_draws=int(a['nonfinite_draw'].sum()),nonfinite_draw_coordinates=int(a['nonfinite_draw_coordinates'].sum()),
                nonfinite_proposed_likelihood=int(np.sum(~a['nonfinite_draw']&~np.isfinite(a['proposed_loglikelihood']))),
                material_correction=int(np.sum(np.abs(a['log_mh_ratio'])>MATERIAL)))
            m['log_correction']=quantiles(a['log_mh_ratio'][~a['nonfinite_draw']])
            m['contour_survivor_alpha']=quantiles(a['alpha'][a['passes_ell9']])
            attempted=np.zeros((8,9),dtype=int);accepted_labels=np.zeros((8,9),dtype=int)
            for w,labels,ok in zip(a['walker'],a['labels'],a['accepted']):
                attempted[w,labels]+=1
                if ok: accepted_labels[w,labels]+=1
            m['source_updates_by_walker']=dict(attempted=attempted,accepted=accepted_labels,
                never_refreshed_labels=[np.flatnonzero(z==0).tolist() for z in accepted_labels])
            m['cost']=run['kernels'][name]['cost']
            m['wall_seconds_excluding_compilation_starts_io']=sum(run['kernels'][name]['walker_seconds'])
            # Independent saved-array audit of sequential progression, caches and MH.
            for w in range(8):
                old=starts['position'][w];lp=starts['logprior'][w];vl=starts['loglikelihood'][w]
                for i in range(512):
                    j=w*512+i;ix=block_indices(a['labels'][j]);keep=np.setdiff1d(np.arange(54),ix)
                    assert a['old_logprior'][j]==lp and a['old_loglikelihood'][j]==vl
                    assert a['proposed_position'][j,keep].tobytes()==old[keep].tobytes()
                    np.testing.assert_array_equal(a['proposed_position'][j,ix],a['draw'][j].ravel())
                    qold=logq(old[ix]);qnew=logq(a['proposed_position'][j,ix])
                    assert qold==a['logq_old'][j] and qnew==a['logq_new'][j]
                    ratio,alpha=mh(lp,a['proposed_logprior'][j],qold,qnew,
                                   a['proposed_loglikelihood'][j],not a['nonfinite_draw'][j])
                    np.testing.assert_equal(ratio,a['log_mh_ratio'][j])
                    assert alpha==a['alpha'][j]
                    assert bool(a['uniform'][j]<a['alpha'][j])==bool(a['accepted'][j])
                    expected=a['proposed_position'][j] if a['accepted'][j] else old
                    assert x[w,i].tobytes()==expected.tobytes()
                    expected_lp=a['proposed_logprior'][j] if a['accepted'][j] else lp
                    expected_ll=a['proposed_loglikelihood'][j] if a['accepted'][j] else vl
                    assert a['logprior'][j]==expected_lp and ll[w,i]==expected_ll
                    old=x[w,i];lp=a['logprior'][j];vl=ll[w,i]
        # Descriptive per-walker ESS per 1000 proposal likelihood costs; separate metrics, no combined score.
        for window in ['first256','second256','all512']:
            s=m[window];cost=s.get('proposal_likelihood_calls',s.get('slice_likelihood_proxy'))/8
            s['median_f0_ess_per_1000_proposal_likelihood_cost']=s['f0_u_ess_median']/cost*1000
            s['median_logL_ess_per_1000_proposal_likelihood_cost']=s['logL_ess_median']/cost*1000
        result['metrics'][name]=m
    # Largest common baseline prefix costing at most one refresh run's 4096 proposal calls.
    b=raws['isotropic'];c=(b['num_steps']+b['num_shrink']).reshape(8,512)
    n=int(np.sum(np.cumsum(c.sum(0))<=4096));assert n>=4
    result['baseline_at_refresh_proposal_cost']=dict(transitions_per_walker=n,
        slice_likelihood_proxy=int(c[:,:n].sum()),
        **agreement(b['loglikelihood'].reshape(8,512)[:,:n],b['position'].reshape(8,512,54)[:,:n,::6]))
    for name in ['single','pair']:
        m=result['metrics'][name]
        m['physical_distance_change_vs_baseline']={window:{key:
            m[window][key]['median']-result['metrics']['isotropic'][window][key]['median']
            for key in ['pairwise_assignment','pairwise_sorted_mean_distance','pairwise_full_assignment_rms']}
            for window in ['first256','second256','all512']}
    result['saved_array_sequential_integrity_passed']=True
    (OUT/'comparison.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False))
    for name,m in result['metrics'].items():
        for win in ['first256','second256','all512']:
            s=m[win];print(name,win,{k:s[k] for k in ['exceedance_sd','logL_mean_sd','f0_u_ess_median','logL_ess_median']},'KS',s['logL_KS_distribution']['median'],flush=True)


if __name__=='__main__':main()
