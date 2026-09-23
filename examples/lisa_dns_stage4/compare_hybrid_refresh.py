"""Stage-4O saved-trace comparison and sticky-block audit, no model calls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4.compare_sequential_refresh import load, agreement, acceptance
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    NAMES,DF,source_distances,cross_frequency_distance)
from examples.lisa_dns_stage4.logistic_mh_refresh import (
    ELL9,ELL10,MATERIAL,logq,mh,quantiles,safe)
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices,label_from_index
from examples.lisa_dns_stage4.hybrid_logistic_refresh import OUT,cost_matched_prefix,sweep_cost
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac


def main():
    run=json.loads((OUT/'report.json').read_text());assert run['status']=='completed'
    bottle=json.loads((OUT/'bottleneck_audit.json').read_text())
    npath=Path('/tmp/lisa_dns_stage4n_sequential/comparison.json')
    previous=json.loads(npath.read_text())
    for path,digest in previous['file_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    starts=load(OUT/'initial_states.npz');start=starts['position']
    bounds_path=Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    bounds=json.loads(bounds_path.read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    def physical(x):return lo+(hi-lo)*expit(np.asarray(x).reshape(np.asarray(x).shape[:-1]+(9,6)))
    initial=physical(start)
    def geometry(a,b,labels):
        p=physical(a);q=physical(b);ix=block_indices(labels) if len(labels)<=2 else np.arange(54)
        return dict(latent_l2=float(np.linalg.norm(b[ix]-a[ix])),
            physical_f0_rms_bins=float(np.sqrt(np.mean((q[labels,0]-p[labels,0])**2))/DF),
            source_set_frequency_rms_bins=source_distances(p[:,0,None],q[:,0,None],[DF])['assignment'],
            source_set_full_rms=source_distances(p,q,widths,{3:np.pi,4:2*np.pi})['assignment'])
    def summarize(x,ll):
        s=agreement(ll,x[:,:,::6]);p=physical(x);f=p[:,:,:,0]
        s['unique_state_fraction']=[len(np.unique(z,axis=0))/len(z) for z in x]
        pairs=[]
        for i in range(8):
            for j in range(i+1,8):
                full=[source_distances(a,b,widths,{3:np.pi,4:2*np.pi})['assignment']
                      for a in p[i,31::32] for b in p[j,31::32]]
                pairs.append(dict(walker_i=i,walker_j=j,**cross_frequency_distance(f[i],f[j]),
                    full_assignment_rms=float(np.sqrt(np.mean(np.square(full)))) if full else None))
        s['source_set_pairs']=pairs
        for key in ['assignment','sorted_mean_distance','full_assignment_rms']:
            s[f'pairwise_{key}']=quantiles([v[key] for v in pairs if v[key] is not None])
        s['within_walker_frequency_rms_bins']=[cross_frequency_distance(z,z)['assignment'] for z in f]
        s['within_walker_full_rms']=[float(np.sqrt(np.mean([
            source_distances(a,b,widths,{3:np.pi,4:2*np.pi})['assignment']**2
            for a in z[31::32] for b in z[31::32]]))) for z in p]
        return s
    paths=[OUT/name/'trace.npz' for name in ['hybrid_single','hybrid_pair']]
    result=dict(ell9=ELL9,ell10=ELL10,level10_remains_rejected=True,baseline_rerun=False,
        bottleneck_audit=str(OUT/'bottleneck_audit.json'),metrics=previous['metrics'],
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,npath,Path(__file__),bounds_path]},
        cost_prefix_rule=run['cost_prefix_rule'],
        physical_notice=previous['physical_notice'],ess_notice=previous['ess_notice'])
    baseline=previous['metrics']['isotropic']
    baseline_cost=baseline['cost']['slice_likelihood_proxy']+baseline['cost']['direct_likelihood_cache_checks']
    for size,name,path in zip([1,2],['hybrid_single','hybrid_pair'],paths):
        a=load(path);x=a['position'].reshape(8,512,54);ll=a['loglikelihood'].reshape(8,512)
        post=a['post_slice_position'].reshape(8,512,54)
        np.testing.assert_array_equal(a['walker'],np.repeat(np.arange(8),512))
        np.testing.assert_array_equal(a['iteration'],np.tile(np.arange(512),8))
        assert np.isfinite(x).all() and np.isfinite(post).all()
        for lpkey,llkey in [('logprior','loglikelihood'),('post_slice_logprior','post_slice_loglikelihood')]:
            assert np.isfinite(a[lpkey]).all() and np.isfinite(a[llkey]).all() and np.all(a[llkey]>ELL9)
        m={}
        for window,sl in [('first256',slice(0,256)),('second256',slice(256,512)),('all512',slice(0,512))]:
            s=summarize(x[:,sl],ll[:,sl]);s['acceptance_autocorrelation']=[ac(z) for z in a['accepted'].reshape(8,512)[:,sl]]
            m[window]=s
        p=physical(x);f=p[:,:,:,0]
        m['early_late_logL_KS']=[float(ks_2samp(z[:256],z[256:],method='asymp').statistic) for z in ll]
        m['trajectory_source_set_movement']=[dict(walker=w,
            between_halves=cross_frequency_distance(f[w,:256],f[w,256:]),
            start_to_end_frequency=source_distances(initial[w,:,0,None],f[w,-1,:,None],[DF]),
            start_to_end_full=source_distances(initial[w],p[w,-1],widths,{3:np.pi,4:2*np.pi}),
            max_frequency_distance_from_start=max(source_distances(initial[w,:,0,None],z[:,0,None],[DF])['assignment'] for z in p[w]))
            for w in range(8)]
        m['acceptance']=dict(overall=acceptance(a,np.ones(4096,dtype=bool)),
            by_walker=[acceptance(a,a['walker']==w) for w in range(8)],
            by_category=[dict(labels=label_from_index(c,size),**acceptance(a,a['category']==c)) for c in range(9 if size==1 else 36)])
        m['tail_counts']=dict(negative_infinite_prior=int(np.isneginf(a['proposed_logprior']).sum()),
            nonfinite_draws=int(a['nonfinite_draw'].sum()),nonfinite_draw_coordinates=int(a['nonfinite_draw_coordinates'].sum()),
            nonfinite_proposed_likelihood=int(np.sum(~a['nonfinite_draw']&~np.isfinite(a['proposed_loglikelihood']))),
            material_correction=int(np.sum(np.abs(a['log_mh_ratio'])>MATERIAL)))
        m['log_correction']=quantiles(a['log_mh_ratio'][~a['nonfinite_draw']])
        m['contour_survivor_alpha']=quantiles(a['alpha'][a['passes_ell9']])
        m['cost']=run['kernels'][name]['cost']
        m['loop_wall_seconds']=sum(run['kernels'][name]['walker_seconds'])
        m['cost_matched']={}
        for label,cost,budget,init in [
            ('including_audits',a['likelihood_total_proxy'],baseline_cost,8),
            ('proposal_only',a['slice_likelihood_proxy']+a['refresh_likelihood'],baseline['cost']['slice_likelihood_proxy'],0)]:
            prefix=cost_matched_prefix(cost.reshape(8,512),budget,init);length=prefix['sweeps_per_walker']
            assert length>=32
            m['cost_matched'][label]=dict(prefix=prefix,metrics=summarize(x[:,:length],ll[:,:length]))
        m['physical_distance_change_vs_baseline']={window:{key:
            m[window][key]['median']-baseline[window][key]['median']
            for key in ['pairwise_assignment','pairwise_sorted_mean_distance','pairwise_full_assignment_rms']}
            for window in ['first256','second256','all512']}
        movements=dict(slice=[],accepted_refresh=[],whole_sweep=[]);sticky=[]
        for w in range(8):
            label=bottle['rows'][w]['label'];ix=block_indices([label])
            old=start[w];oldlp=starts['logprior'][w];oldll=starts['loglikelihood'][w]
            attempts=[];accepts=[];slice_changes=[];accepted_after_earlier_slice=[]
            latent_path=0.;slice_path=0.;refresh_path=0.;f0_path=0.;sticky_values=[old[ix].copy()]
            for i in range(512):
                j=w*512+i;mid=post[w,i];new=x[w,i]
                assert a['old_logprior'][j]==a['post_slice_logprior'][j]
                assert a['old_loglikelihood'][j]==a['post_slice_loglikelihood'][j]
                # Refresh proposal and rejection/acceptance contract from the post-slice state.
                selected=block_indices(a['labels'][j]);keep=np.setdiff1d(np.arange(54),selected)
                assert a['proposed_position'][j,keep].tobytes()==mid[keep].tobytes()
                np.testing.assert_array_equal(a['proposed_position'][j,selected],a['draw'][j].ravel())
                qo=logq(mid[selected]);qn=logq(a['proposed_position'][j,selected])
                assert qo==a['logq_old'][j] and qn==a['logq_new'][j]
                ratio,alpha=mh(a['post_slice_logprior'][j],a['proposed_logprior'][j],qo,qn,
                    a['proposed_loglikelihood'][j],not a['nonfinite_draw'][j])
                np.testing.assert_equal(ratio,a['log_mh_ratio'][j]);assert alpha==a['alpha'][j]
                assert bool(a['uniform'][j]<alpha)==bool(a['accepted'][j])
                if a['accepted'][j]:
                    assert new.tobytes()==a['proposed_position'][j].tobytes()
                    assert a['logprior'][j]==a['proposed_logprior'][j] and ll[w,i]==a['proposed_loglikelihood'][j]
                else:
                    assert new.tobytes()==mid.tobytes()
                    assert a['logprior'][j]==a['post_slice_logprior'][j] and ll[w,i]==a['post_slice_loglikelihood'][j]
                # Labelled slice masks use exactly the original 20/40/40 implementation.
                component=int(a['slice_component'][j]);slabs=a['slice_labels'][j]
                slice_labels=np.arange(9) if component==0 else slabs[slabs>=0]
                if component!=0:
                    keep_slice=np.setdiff1d(np.arange(54),block_indices(slice_labels))
                    assert mid[keep_slice].tobytes()==old[keep_slice].tobytes()
                changed=not np.array_equal(mid[ix],old[ix])
                if changed:slice_changes.append(i)
                if label in a['labels'][j]:
                    attempts.append(i)
                    if a['accepted'][j]:
                        accepts.append(i)
                        if any(t<i for t in slice_changes):accepted_after_earlier_slice.append(i)
                ds=float(np.linalg.norm(mid[ix]-old[ix]));dr=float(np.linalg.norm(new[ix]-mid[ix]))
                slice_path+=ds;refresh_path+=dr;latent_path+=ds+dr
                f0=lo[0]+(hi[0]-lo[0])*expit([old[ix[0]],mid[ix[0]],new[ix[0]]])
                f0_path+=float(np.abs(np.diff(f0)).sum()/DF)
                sticky_values.extend([mid[ix].copy(),new[ix].copy()])
                movements['slice'].append(geometry(old,mid,slice_labels))
                movements['whole_sweep'].append(geometry(old,new,np.arange(9)))
                if a['accepted'][j]:movements['accepted_refresh'].append(geometry(mid,new,a['labels'][j]))
                expected_cost=sweep_cost(int(a['num_steps'][j]),int(a['num_shrink'][j]),not a['nonfinite_draw'][j])
                for k,v in expected_cost.items():assert a[k][j]==v
                old=new;oldlp=a['logprior'][j];oldll=ll[w,i]
            trajectory=np.array(sticky_values);freq=lo[0]+(hi[0]-lo[0])*expit(trajectory[:,0])
            sticky.append(dict(walker=w,label=label,slice_change_count=len(slice_changes),
                refresh_attempts=len(attempts),refresh_accepts=len(accepts),
                accepted_refreshes_after_earlier_slice_change=len(accepted_after_earlier_slice),
                first_slice_change_sweep=slice_changes[0] if slice_changes else None,
                accepted_refresh_sweeps=accepts,attempted_refresh_sweeps=attempts,
                latent_block_net_l2=float(np.linalg.norm(trajectory[-1]-trajectory[0])),
                latent_block_path_l2=latent_path,slice_latent_path_l2=slice_path,refresh_latent_path_l2=refresh_path,
                latent_block_max_from_start_l2=float(np.linalg.norm(trajectory-trajectory[0],axis=1).max()),
                initial_f0_hz=float(freq[0]),final_f0_hz=float(freq[-1]),f0_net_bins=float((freq[-1]-freq[0])/DF),
                f0_range_bins=float(np.ptp(freq)/DF),f0_path_bins=f0_path))
        m['sticky_blocks']=sticky
        m['geometry']={phase:{k:quantiles([z[k] for z in values]) for k in values[0]} for phase,values in movements.items()}
        for k in ['slice_likelihood_proxy','slice_prior_proxy','refresh_likelihood','refresh_prior',
                  'cache_likelihood','cache_prior','likelihood_total_proxy','prior_total_proxy']:
            assert int(a[k].sum())==m['cost'][k]
        result['metrics'][name]=m
    # Robust between-walker summaries complement SD; they must not hide an outlier.
    for m in result['metrics'].values():
        summaries=[m[w] for w in ['first256','second256','all512']]
        summaries += [v['metrics'] for v in m.get('cost_matched',{}).values()]
        for s in summaries:
            means=np.asarray(s['walker_logL_mean']);median=np.median(means)
            s['between_walker_logL_mean_MAD']=float(np.median(np.abs(means-median)))
            s['between_walker_logL_mean_IQR']=float(np.diff(np.quantile(means,[.25,.75]))[0])
    result['independent_saved_trace_integrity_passed']=True
    for path,digest in run['file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (OUT/'comparison.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False))
    for name in ['hybrid_single','hybrid_pair']:
        m=result['metrics'][name]
        for win in ['first256','second256','all512']:
            s=m[win];print(name,win,'logL SD',s['logL_mean_sd'],'exceed SD',s['exceedance_sd'],
                'KS',s['logL_KS_distribution']['median'],'f0 ESS',s['f0_u_ess_median'],flush=True)
        for label,v in m['cost_matched'].items():
            s=v['metrics'];print(name,label,v['prefix'],'logL SD',s['logL_mean_sd'],
                'exceed SD',s['exceedance_sd'],'KS',s['logL_KS_distribution']['median'],flush=True)


if __name__=='__main__':main()
