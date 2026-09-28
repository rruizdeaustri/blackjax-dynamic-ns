"""Read-only Stage-4Y diagnostics; no construction, calibration or mass update."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from blackjax.ns.dns_levels import tail_diagnostics
from examples.lisa_dns_stage4.level11_population_validation import (
    OUT,W,X,V,G,ELL11,LOGX11,RECOVERY_CRITERION,load,serial,digest,round_robin,
    log_probabilities,draw_label,joint_decision,fixed_exchange,exchange_tags,slice_blocks,
    verify_ladder,recover_starts,
)
from examples.lisa_dns_stage4.analyze_frequency_population import distribution,rate
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    NAMES,DF,source_distances,cross_frequency_distance,
)
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac


@dataclass(frozen=True)
class DiagnosticThreshold:
    ell_12_diag: float
    order_index: int
    sample_count: int
    diagnostic_only: bool = True


def diagnostic_threshold(logL):
    """Strict exp(-1) upper-tail order statistic; intentionally no level schema."""
    values=np.asarray(logL)
    if values.shape!=(8,512) or not np.isfinite(values).all() or np.any(values<=ELL11):
        raise ValueError('Diagnostic requires the completed strict level-11 trace')
    ordered=np.sort(values.ravel());index=int(np.floor(len(ordered)*(1-np.exp(-1))))
    return DiagnosticThreshold(float(ordered[index]),index,len(ordered))


def exceedance_summary(logL,diagnostic):
    if not isinstance(diagnostic,DiagnosticThreshold) or not diagnostic.diagnostic_only:
        raise ValueError('A diagnostic-only threshold is required')
    ll=np.asarray(logL);indicator=(ll>diagnostic.ell_12_diag).astype(float)
    fractions=indicator.mean(1);blocks=indicator.reshape(8,-1,32).mean(2)
    return dict(tail_diagnostics(ll.T,diagnostic.ell_12_diag,block_size=32),
        between_walker_se=float(fractions.std(ddof=1)/np.sqrt(8)),
        block_se=float(blocks.std(ddof=1)/np.sqrt(blocks.size)),
        fraction_sd=float(fractions.std(ddof=1)),fraction_range=[float(fractions.min()),float(fractions.max())])


def walker_recovery(initial_logL,logL,walker):
    """Predeclared leave-one-walker-out contemporaneous inclusive IQR criterion."""
    if walker not in (2,7):raise ValueError('Predeclared focal walkers are 2 and 7')
    values=np.asarray(logL);initial=np.asarray(initial_logL)
    if values.ndim!=2 or values.shape[0]!=8 or initial.shape!=(8,):
        raise ValueError('Eight walker histories required')
    if not np.isfinite(values).all() or not np.isfinite(initial).all():
        raise ValueError('Finite likelihoods required')
    all_values=np.column_stack([initial,values]);others=np.delete(all_values,walker,axis=0)
    lo,hi=np.quantile(others,[.25,.75],axis=0,method='linear')
    inside=(all_values[walker]>=lo)&(all_values[walker]<=hi)
    first=np.flatnonzero(inside);first_sample=np.flatnonzero(inside[1:])+1
    return dict(criterion=RECOVERY_CRITERION,initial_logL=float(initial[walker]),
        initial_in_range=bool(inside[0]),first_entry_sweep=int(first[0]) if len(first) else None,
        first_post_transition_entry_sweep=int(first_sample[0]) if len(first_sample) else None,
        fraction_sweeps_in_range=float(inside[1:].mean()),
        per_sweep_in_range=inside[1:],lower_IQR=lo,upper_IQR=hi)


def replay(raw,ex,starts,rng,births,frequency_bounds):
    # Independent operation-based provenance replay and exact state/cache reconstruction.
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy();next_token=72
    old=starts['position'];arrivals=[];lineage_visits={k:{k//9} for k in range(72)};token_visits={k:{k//9} for k in range(72)}
    for s in range(raw['position'].shape[1]):
        sx=raw['post_slice_position'][:,s];sp=raw['post_slice_logprior'][:,s];sl=raw['post_slice_loglikelihood'][:,s]
        assert np.isfinite(sp).all() and np.isfinite(sl).all() and np.all(sl>ELL11)
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
            fw,f=log_probabilities(pre,frequency_bounds);rv,_=log_probabilities(proposed,frequency_bounds)
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
        assert np.isfinite(expectedp).all() and np.isfinite(expectedl).all() and np.all(expectedl>ELL11)
        np.testing.assert_array_equal(tokens,raw['tokens'][:,s]);np.testing.assert_array_equal(lineage,raw['lineage'][:,s])
        old=expected
    assert next_token==len(births)
    return dict(arrivals=arrivals,token_visits=token_visits,lineage_visits=lineage_visits,
                exact_replay_passed=True,exact_token_count=len(births))


def communication(replayed):
    arrivals=replayed['arrivals'];donors=[set() for _ in range(8)]
    counts=np.zeros((8,8),int);migration={}
    for z in arrivals:
        a,b=z['from_walker'],z['to_walker'];donors[b].add(a);counts[a,b]+=1
        token=z['token'];migration[token]=migration.get(token,0)+1
    return dict(accepted_directional_transfers=len(arrivals),
        per_walker_received=counts.sum(0),distinct_donors=[len(d) for d in donors],
        donor_ids=[sorted(d) for d in donors],migration_matrix=counts,
        unique_received_realizations=[len({z['token'] for z in arrivals if z['to_walker']==w}) for w in range(8)],
        exact_token_count=replayed['exact_token_count'],
        source_realization_migration_counts=migration,
        distinct_migrated_realizations=len(migration),
        foreign_birth_arrivals=sum(z['foreign_birth'] for z in arrivals),
        first_token_visits=sum(z['first_token_visit'] for z in arrivals),
        exact_initial_token_transfers=sum(z['initial_exact_token'] for z in arrivals),
        initial_lineages_visiting_other_walker=sum(len(v)>1 for v in replayed['lineage_visits'].values()))


def main():
    run=json.loads((OUT/'report.json').read_text())
    assert run['status']=='completed_one_fixed_contour_benchmark' and run['completed_sweeps']==512
    for path,h in run['file_sha256'].items():assert digest(path)==h,path
    assert digest(OUT/'design.json')==run['design_sha256']
    raw=load(OUT/'trace.npz');ex=load(OUT/'exchanges.npz');starts=load(OUT/'initial_states.npz')
    rng=load(OUT/'random_schedule.npz');births=json.loads((OUT/'provenance.json').read_text())
    assert raw['position'].shape==(8,512,54) and ex['pair'].shape==(2048,2)
    recovered,recovery=recover_starts(load(X/'calibration/trace.npz'))
    for k in starts:assert starts[k].tobytes()==recovered[k].tobytes()
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    replayed=replay(raw,ex,starts,rng,births,bounds['f0']);comm=communication(replayed)
    diagnostic=diagnostic_threshold(raw['loglikelihood'])
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    def physical(x):return lo+(hi-lo)*expit(x.reshape(x.shape[:-1]+(9,6)))
    def geom(x,y):return source_distances(x,y,widths,{3:np.pi,4:2*np.pi})['assignment']
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
        ks=[dict(a=a,b=b,ks=float(ks_2samp(ll[a],ll[b],method='asymp').statistic)) for a in range(8) for b in range(a+1,8)]
        return dict(pairwise_logL_KS=ks,focal_KS={str(w):[z for z in ks if w in (z['a'],z['b'])] for w in (2,7)},
            maximum_pairwise_logL_KS=max(z['ks'] for z in ks),
            diagnostic_exceedance=exceedance_summary(ll,diagnostic),n=x.shape[1],walker_logL_mean=means,logL_mean_sd=float(means.std(ddof=1)),
            logL_mean_MAD=float(np.median(abs(means-np.median(means)))),logL_mean_IQR=float(np.ptp(np.quantile(means,[.25,.75]))),
            median_pairwise_logL_KS=float(np.median([ks_2samp(ll[a],ll[b],method='asymp').statistic for a in range(8) for b in range(a+1,8)])),
            f0_u_ess=ess,f0_u_ess_median=float(np.median(ess)),f0_u_ess_minimum=float(ess.min()),
            logL_ess=ll_ess,logL_ess_median=float(np.median(ll_ess)),unique_state_fractions=[len(np.unique(z,axis=0))/len(z) for z in x],
            source_set_pairs=pairs,median_frequency_set_rms=float(np.median([z['assignment'] for z in pairs])),
            median_frequency_set_mean_distance=float(np.median([z['sorted_mean_distance'] for z in pairs])),
            median_full_source_set_rms=float(np.median([z['full_assignment_rms'] for z in pairs])),
            within_walker_frequency_rms=[cross_frequency_distance(z,z)['assignment'] for z in f],
            within_walker_full_rms=[float(np.sqrt(np.mean([geom(a,b)**2 for a in z[31::32] for b in z[31::32]]))) for z in p])
    windows=[('first256',slice(0,256)),('second256',slice(256,512)),('all512',slice(0,512))]
    metrics={w:summarize(raw['position'][:,sl],raw['loglikelihood'][:,sl]) for w,sl in windows}
    initial=physical(starts['position']);p=physical(raw['position'])
    movement=[dict(walker=w,
        start_to_end_frequency=source_distances(initial[w,:,0,None],p[w,-1,:,0,None],[DF])['assignment'],
        start_to_end_full=geom(initial[w],p[w,-1]),
        max_frequency_from_start=max(source_distances(initial[w,:,0,None],z[:,0,None],[DF])['assignment'] for z in p[w]),
        max_full_from_start=max(geom(initial[w],z) for z in p[w]),
        between_halves_frequency=cross_frequency_distance(p[w,:256,:,0],p[w,256:,:,0])) for w in range(8)]
    historical=json.loads((W/'comparison.json').read_text())
    oldmetrics=historical['metrics']
    oldarrivals=json.loads((W/'provenance_transfers.json').read_text())
    olddonors=[sorted({z['from_walker'] for z in oldarrivals if z['to_walker']==w}) for w in range(8)]
    oldunique=[len({z['token'] for z in oldarrivals if z['to_walker']==w}) for w in range(8)]
    wrs={}
    for w in (2,7):
        wr=walker_recovery(starts['loglikelihood'],raw['loglikelihood'],w)
        wr.update(received_transfers=int(comm['per_walker_received'][w]),
            unique_received_realizations=comm['unique_received_realizations'][w],
            distinct_donors=comm['distinct_donors'][w],donor_ids=comm['donor_ids'][w],
            accepted_exchanges=int(np.sum(ex['accepted']&np.any(ex['pair']==w,axis=1))),
            movement=movement[w],
            window_fraction_in_range={win:float(np.asarray(wr['per_sweep_in_range'])[sl].mean()) for win,sl in windows},
            window_exceedance={win:metrics[win]['diagnostic_exceedance']['walker_fractions'][w] for win,sl in windows},
            logL_ess={win:metrics[win]['logL_ess'][w] for win,sl in windows})
        wrs[str(w)]=wr
    costs=run['cost'];assert costs['slice_likelihood_proxy']==int((raw['num_steps']+raw['num_shrink']).sum())
    assert costs['exchange_likelihood']==costs['exchange_prior']==4096
    assert costs['direct_cache_likelihood']==costs['direct_cache_prior']==8200
    assert costs['total_likelihood_proxy']==costs['slice_likelihood_proxy']+4096+8200
    assert (OUT/'checkpoint_level11.json').read_bytes()==(X/'checkpoint_level11.json').read_bytes()
    assert not (OUT/'checkpoint_level12.json').exists()
    result=dict(status='completed_saved_trace_validation',ell11=ELL11,logX11=LOGX11,
        structural_failures=0,exact_replay_passed=True,ladder_unchanged=True,level_construction=False,
        starting_recovery=recovery,diagnostic_threshold=asdict(diagnostic),metrics=metrics,
        movement=movement,focal_walkers=wrs,provenance=comm,
        exchange=dict(overall=rate(ex,np.ones(2048,bool)),
            by_pair=[dict(a=a,b=b,**rate(ex,np.all(ex['pair']==[a,b],axis=1))) for a in range(8) for b in range(a+1,8)],
            selection_log_ratio=distribution(ex['log_r_selection']),joint_prior_ratio=distribution(ex['delta_logprior']),
            nonfinite_prior=int(ex['nonfinite_prior'].sum()),nonfinite_likelihood=int(ex['nonfinite_likelihood'].sum()),
            nonfinite_ratio=int((~np.isfinite(ex['log_r'])).sum())),
        cost=costs,level10_comparison=dict(metrics=oldmetrics,exchange=historical['exchange'],
            provenance={k:v for k,v in historical['provenance'].items() if k!='arrivals'},
            direct_donor_ids=olddonors,distinct_donors=[len(d) for d in olddonors],unique_received_realizations=oldunique,cost=historical['cost']),
        file_sha256={str(p):digest(p) for p in [OUT/'report.json',OUT/'trace.npz',OUT/'exchanges.npz',OUT/'provenance.json',Path(__file__)]})
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    (OUT/'provenance_transfers.json').write_text(json.dumps(serial(replayed['arrivals']),separators=(',',':')))
    for w in metrics:print(w,{k:metrics[w][k] for k in ['logL_mean_sd','median_pairwise_logL_KS','maximum_pairwise_logL_KS']},flush=True)
    for w,wr in wrs.items():
        print('Walker',w,{k:v for k,v in wr.items() if k not in ['per_sweep_in_range','lower_IQR','upper_IQR']},flush=True)


if __name__=='__main__':main()
