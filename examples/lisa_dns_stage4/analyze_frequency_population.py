"""Stage-4U saved-trace integrity and comparison; never runs a chain."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    OUT,H,O,ELL9,load,serial,digest,round_robin,log_probabilities,draw_label,
    joint_decision,fixed_exchange,exchange_tags,slice_blocks,prefix_from_cost,
)
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    NAMES,DF,source_distances,cross_frequency_distance,
)
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac


def distribution(a):
    a=np.asarray(a,dtype=float);finite=a[np.isfinite(a)]
    result=dict(n=len(a),nonfinite=int(len(a)-len(finite)))
    if len(finite):
        result.update(dict(zip(['min','q25','median','q75','max'],np.quantile(finite,[0,.25,.5,.75,1]).tolist())))
    return result



def rate(ex,mask):
    n=int(mask.sum());accepted=int(np.sum(ex['accepted'][mask]));survives=int(np.sum(ex['joint_contour'][mask]))
    return dict(n=n,accepted=accepted,fraction=accepted/n if n else None,contour_survivors=survives,
        contour_fraction=survives/n if n else None,mh_rejected_survivors=survives-accepted,
        mh_rejection_fraction=(survives-accepted)/survives if survives else None)


def main():
    run=json.loads((OUT/'report.json').read_text());assert run['status']=='completed_one_fixed_contour_benchmark'
    for path,h in run['file_sha256'].items():assert digest(path)==h,path
    assert digest(OUT/'design.json')==run['design_sha256']
    raw=load(OUT/'trace.npz');ex=load(OUT/'exchanges.npz');starts=load(OUT/'initial_states.npz')
    rng=load(OUT/'random_schedule.npz');births=json.loads((OUT/'provenance.json').read_text())
    assert raw['position'].shape==(8,512,54) and ex['pair'].shape==(2048,2)
    for k in starts:assert starts[k].tobytes()==load(H/'initial_states.npz')[k].tobytes()
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    def physical(x):return lo+(hi-lo)*expit(x.reshape(x.shape[:-1]+(9,6)))
    def geom(x,y):return source_distances(x,y,widths,{3:np.pi,4:2*np.pi})['assignment']
    # Independent operation-based provenance replay and exact state/cache reconstruction.
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy();next_token=72
    old=starts['position'];arrivals=[];lineage_visits={k:{k//9} for k in range(72)};token_visits={k:{k//9} for k in range(72)}
    for s in range(512):
        sx=raw['post_slice_position'][:,s];sp=raw['post_slice_logprior'][:,s];sl=raw['post_slice_loglikelihood'][:,s]
        assert np.isfinite(sp).all() and np.isfinite(sl).all() and np.all(sl>ELL9)
        for w in range(8):
            selected=slice_blocks(int(raw['slice_component'][w,s]),raw['slice_labels'][w,s])
            keep=np.setdiff1d(np.arange(9),selected)
            assert sx[w].reshape(9,6)[keep].tobytes()==old[w].reshape(9,6)[keep].tobytes()
            for k in selected:
                birth=births[next_token]
                assert birth==dict(token=next_token,birth_walker=w,birth_sweep=s,birth_label=int(k),parent=int(tokens[w,k]),lineage=int(lineage[w,k]))
                tokens[w,k]=next_token;token_visits[next_token]={w};next_token+=1
        np.testing.assert_array_equal(tokens,raw['post_slice_tokens'][:,s])
        np.testing.assert_array_equal(lineage,raw['post_slice_lineage'][:,s])
        expected=sx.copy();expectedp=sp.copy();expectedl=sl.copy()
        for slot,(a,b) in enumerate(round_robin()[s%7]):
            n=4*s+slot;i,j=ex['labels'][n]
            assert ex['sweep'][n]==s and ex['slot'][n]==slot
            np.testing.assert_array_equal(ex['pair'][n],[a,b])
            np.testing.assert_array_equal(ex['pre_tokens'][n],tokens[[a,b],[i,j]])
            np.testing.assert_array_equal(ex['pre_lineages'][n],lineage[[a,b],[i,j]])
            pre=sx[[a,b]];proposed=np.array(fixed_exchange(pre[0],pre[1],i,j))
            assert pre.tobytes()==ex['post_slice_position'][n].tobytes()
            assert proposed.tobytes()==ex['proposed_position'][n].tobytes()
            fw,f=log_probabilities(pre,bounds['f0']);rv,_=log_probabilities(proposed,bounds['f0'])
            np.testing.assert_array_equal(fw,ex['forward_log_probabilities'][n])
            np.testing.assert_array_equal(rv,ex['reverse_log_probabilities'][n])
            np.testing.assert_array_equal(ex['selected_log_probabilities'][n],fw[np.arange(2),[i,j]])
            ranks=np.argsort(np.argsort(-fw,axis=-1,kind='stable'),axis=-1,kind='stable')+1
            np.testing.assert_array_equal(ex['selected_ranks'][n],ranks[np.arange(2),[i,j]])
            np.testing.assert_array_equal(ex['selected_frequency'][n],f[np.arange(2),[i,j]])
            inside=((f[np.arange(2),[i,j]]>=.0018407250)&(f[np.arange(2),[i,j]]<=.0018412366)).sum()
            assert ex['inside_count'][n]==inside
            assert (draw_label(fw[0],rng['gumbels'][s,slot,0]),draw_label(fw[1],rng['gumbels'][s,slot,1]))==(i,j)
            decision=joint_decision(sp[[a,b]],ex['proposed_logprior'][n],fw,rv,[i,j],ex['proposed_loglikelihood'][n],float(rng['uniforms'][s,slot]))
            for k,v in decision.items():np.testing.assert_equal(ex[k][n],v)
            if decision['accepted']:
                expected[[a,b]]=proposed;expectedp[[a,b]]=ex['proposed_logprior'][n];expectedl[[a,b]]=ex['proposed_loglikelihood'][n]
                for origin,destination,token,lin in [(a,b,int(tokens[a,i]),int(lineage[a,i])),(b,a,int(tokens[b,j]),int(lineage[b,j]))]:
                    birth=births[token]
                    arrivals.append(dict(sweep=s,from_walker=int(origin),to_walker=int(destination),token=token,lineage=lin,
                        birth_walker=birth['birth_walker'],initial_exact_token=token<72,
                        foreign_birth=birth['birth_walker']!=destination,first_token_visit=destination not in token_visits[token]))
                    token_visits[token].add(int(destination));lineage_visits[lin].add(int(destination))
            tokens=exchange_tags(tokens,a,b,i,j,decision['accepted']);lineage=exchange_tags(lineage,a,b,i,j,decision['accepted'])
        assert expected.tobytes()==raw['position'][:,s].tobytes()
        assert expectedp.tobytes()==raw['logprior'][:,s].tobytes()
        assert expectedl.tobytes()==raw['loglikelihood'][:,s].tobytes()
        assert np.isfinite(expectedp).all() and np.isfinite(expectedl).all() and np.all(expectedl>ELL9)
        np.testing.assert_array_equal(tokens,raw['tokens'][:,s]);np.testing.assert_array_equal(lineage,raw['lineage'][:,s])
        old=expected
    assert next_token==len(births)
    ell_old=json.loads((H/'report.json').read_text())['ell10']
    ell_p=json.loads(Path('/tmp/lisa_dns_stage4p_hybrid_level10/report.json').read_text())['decision']['threshold']
    def summarize(x,ll):
        means=ll.mean(1);p=physical(x);f=p[...,0]
        ess=np.array([[ac(z)['ess'] for z in w[:,::6].T] for w in x])
        ll_ess=[ac(z)['ess'] for z in ll]
        pairs=[]
        for a in range(8):
            for b in range(a+1,8):
                full=[geom(y,z) for y in p[a,31::32] for z in p[b,31::32]]
                pairs.append(dict(a=a,b=b,**cross_frequency_distance(f[a],f[b]),
                    full_assignment_rms=float(np.sqrt(np.mean(np.square(full))))))
        thresholds={}
        for name,ell in [('old_isotropic',ell_old),('stage4p',ell_p)]:
            frac=(ll>ell).mean(1)
            thresholds[name]=dict(threshold=ell,walker_exceedance=frac,exceedance_sd=float(frac.std(ddof=1)))
        return dict(n=x.shape[1],walker_logL_mean=means,logL_mean_sd=float(means.std(ddof=1)),
            logL_mean_MAD=float(np.median(abs(means-np.median(means)))),logL_mean_IQR=float(np.ptp(np.quantile(means,[.25,.75]))),
            thresholds=thresholds,median_pairwise_logL_KS=float(np.median([ks_2samp(ll[a],ll[b],method='asymp').statistic for a in range(8) for b in range(a+1,8)])),
            f0_u_ess=ess,f0_u_ess_median=float(np.median(ess)),f0_u_ess_minimum=float(ess.min()),
            logL_ess=ll_ess,logL_ess_median=float(np.median(ll_ess)),unique_state_fractions=[len(np.unique(z,axis=0))/len(z) for z in x],
            source_set_pairs=pairs,median_frequency_set_rms=float(np.median([z['assignment'] for z in pairs])),
            median_frequency_set_mean_distance=float(np.median([z['sorted_mean_distance'] for z in pairs])),
            median_full_source_set_rms=float(np.median([z['full_assignment_rms'] for z in pairs])),
            within_walker_frequency_rms=[cross_frequency_distance(z,z)['assignment'] for z in f],
            within_walker_full_rms=[float(np.sqrt(np.mean([geom(a,b)**2 for a in z[31::32] for b in z[31::32]]))) for z in p])
    datasets={'H_isotropic':load(H/'trace.npz'),'O_hybrid_P':load(O/'hybrid_pair/trace.npz'),'U_population':raw}
    metrics={};initial=physical(starts['position'])
    for name,a in datasets.items():
        x=a['position'].reshape(8,512,54);ll=a['loglikelihood'].reshape(8,512)
        metrics[name]={window:summarize(x[:,sl],ll[:,sl]) for window,sl in [('first256',slice(0,256)),('second256',slice(256,512)),('all512',slice(0,512)),('last32',slice(480,512))]}
        p=physical(x)
        metrics[name]['movement']=[dict(walker=w,start_to_end_frequency=source_distances(initial[w,:,0,None],p[w,-1,:,0,None],[DF])['assignment'],
            start_to_end_full=geom(initial[w],p[w,-1]),max_frequency_from_start=max(source_distances(initial[w,:,0,None],z[:,0,None],[DF])['assignment'] for z in p[w]),
            max_full_from_start=max(geom(initial[w],z) for z in p[w]),
            between_halves_frequency=cross_frequency_distance(p[w,:256,:,0],p[w,256:,:,0])) for w in range(8)]
    hr=json.loads((H/'report.json').read_text());budget=hr['cost']['likelihood_proxy']+hr['cost']['direct_cache_checks']
    costs=raw['likelihood_total_proxy'].sum(0);proposal_costs=raw['proposal_likelihood_proxy'].sum(0)
    assert int(costs.sum()+8)==run['cost']['total_likelihood_proxy']
    prefixes={}
    for name,cs,bud,init in [('including_audits',costs,budget,8),('proposal_only',proposal_costs,hr['cost']['likelihood_proxy'],0)]:
        prefix=prefix_from_cost(cs,bud,init);n=prefix['sweeps'];assert n>=32
        prefixes[name]=dict(prefix=prefix,metrics=summarize(raw['position'][:,:n],raw['loglikelihood'][:,:n]))
    pairs=[dict(a=a,b=b,**rate(ex,(ex['pair'][:,0]==a)&(ex['pair'][:,1]==b))) for a in range(8) for b in range(a+1,8)]
    strata={}
    for k in [1,2]:
        relevant=ex['selected_ranks']<=k;counts=relevant.sum(1)
        strata[str(k)]={name:rate(ex,mask) for name,mask in [('both',counts==2),('at_least_one',counts>=1),('neither',counts==0),('exactly_one',counts==1)]}
    result=dict(status='completed_saved_trace_analysis',ell9=ELL9,structural_failures=0,metrics=metrics,cost_matched=prefixes,
        exchange=dict(overall=rate(ex,np.ones(2048,bool)),by_pair=pairs,rank_strata=strata,
            interval_strata={str(k):rate(ex,ex['inside_count']==k) for k in range(3)},
            selection_log_ratio=distribution(ex['log_r_selection']),joint_prior_ratio=distribution(ex['delta_logprior']),
            selected_probability_A=distribution(np.exp(ex['selected_log_probabilities'][:,0])),
            selected_probability_B=distribution(np.exp(ex['selected_log_probabilities'][:,1])),
            selected_rank_A=distribution(ex['selected_ranks'][:,0]),selected_rank_B=distribution(ex['selected_ranks'][:,1]),
            nonfinite_prior=int(ex['nonfinite_prior'].sum()),nonfinite_likelihood=int(ex['nonfinite_likelihood'].sum())),
        provenance=dict(exact_token_count=len(births),accepted_directional_transfers=len(arrivals),
            transfers_of_exact_initial_tokens=sum(z['initial_exact_token'] for z in arrivals),
            foreign_birth_arrivals=sum(z['foreign_birth'] for z in arrivals),first_token_visits=sum(z['first_token_visit'] for z in arrivals),
            distinct_realizations_visiting_other_walker=sum(len(v)>1 for v in token_visits.values()),
            initial_lineages_visiting_other_walker=sum(len(v)>1 for v in lineage_visits.values()),
            initial_lineage_walker_visits={str(k):sorted(v) for k,v in lineage_visits.items()},
            arrivals=arrivals),
        run_cost=run['cost'],baseline_cost=hr['cost'],baseline_rerun=False,source_gain_calls=0,
        max_cache_prior_error=run['max_cache_prior_error'],max_cache_logL_error=run['max_cache_logL_error'],
        file_sha256={str(p):digest(p) for p in [OUT/'report.json',OUT/'trace.npz',OUT/'exchanges.npz',OUT/'provenance.json',Path(__file__)]})
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    for name in metrics:
        print(name,{w:{k:metrics[name][w][k] for k in ['logL_mean_sd','logL_mean_MAD','logL_mean_IQR','median_pairwise_logL_KS','median_frequency_set_mean_distance','median_full_source_set_rms']} for w in ['first256','second256','all512']})
    print('exchange',result['exchange']['overall'],'prefixes',{k:v['prefix'] for k,v in prefixes.items()})


if __name__=='__main__':main()
