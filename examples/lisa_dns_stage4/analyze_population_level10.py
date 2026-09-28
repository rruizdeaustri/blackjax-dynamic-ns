"""Read-only Stage-4V three-attempt comparison and exact provenance replay."""
import json
from pathlib import Path
import numpy as np
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    ELL9, LOGX9, load, serial, digest, round_robin, log_probabilities, draw_label,
    joint_decision, fixed_exchange, exchange_tags, slice_blocks,
)
from examples.lisa_dns_stage4.analyze_frequency_population import distribution, rate
from examples.lisa_dns_stage4.compare_hybrid_level10 import bank_summary
from examples.lisa_dns_stage4 import hybrid_level10_construction as historical
from examples.lisa_dns_stage4.run_ladder_extension import assess

OUT=Path('/tmp/lisa_dns_stage4v_population_level10')
NAMES=['selection','calibration']


def audit_bank(folder):
    folder=Path(folder)
    run=json.loads((folder/'report.json').read_text())
    assert run['status']=='completed_one_fixed_contour_benchmark'
    assert run['completed_sweeps']==384 and run['structural_failures']==0
    raw=load(folder/'trace.npz');ex=load(folder/'exchanges.npz')
    starts=load(folder/'initial_states.npz');rng=load(folder/'random_schedule.npz')
    births=json.loads((folder/'provenance.json').read_text())
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    # Independent operation-based provenance replay and exact state/cache reconstruction.
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy();next_token=72
    old=starts['position'];arrivals=[];lineage_visits={k:{k//9} for k in range(72)};token_visits={k:{k//9} for k in range(72)}
    for s in range(384):
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
    donors=[set() for _ in range(8)];birth_donors=[set() for _ in range(8)]
    migration={};matrix=np.zeros((8,8),dtype=int)
    for arrival in arrivals:
        a,b=arrival['from_walker'],arrival['to_walker']
        donors[b].add(a);birth_donors[b].add(arrival['birth_walker']);matrix[a,b]+=1
        token=arrival['token'];migration[token]=migration.get(token,0)+1
    assert len(ex['accepted'])==1536
    cost=run['cost']
    assert cost['slice_likelihood_proxy']==int((raw['num_steps']+raw['num_shrink']).sum())
    assert cost['exchange_likelihood']==cost['exchange_prior']==3072
    assert cost['direct_cache_likelihood']==cost['direct_cache_prior']==6152
    result=dict(exchange=rate(ex,np.ones(1536,dtype=bool)),
        by_pair={f'{a},{b}':rate(ex,np.all(ex['pair']==[a,b],axis=1)) for a in range(8) for b in range(a+1,8)},
        selection_log_ratio=distribution(ex['log_r_selection']),
        joint_prior_log_ratio=distribution(ex['delta_logprior']),
        nonfinite_prior=int(ex['nonfinite_prior'].sum()),nonfinite_likelihood=int(ex['nonfinite_likelihood'].sum()),
        nonfinite_mh_ratio=int((~np.isfinite(ex['log_r'])).sum()),
        cost=cost,exact_replay_passed=True,
        provenance=dict(accepted_directional_transfers=len(arrivals),
            distinct_direct_donor_walkers=[len(d) for d in donors],direct_donor_walkers=[sorted(d) for d in donors],
            distinct_birth_donor_walkers=[len(d-{w}) for w,d in enumerate(birth_donors)],
            migration_matrix=matrix,source_realization_migration_counts=migration,
            distinct_migrated_realizations=len(migration),
            exact_realizations_created=len(births),unmigrated_realizations=len(births)-len(migration),
            migration_count_histogram={str(k):sum(migration.get(b['token'],0)==k for b in births)
                for k in range(max(migration.values(),default=0)+1)},
            foreign_birth_transfers=sum(a['foreign_birth'] for a in arrivals),
            first_recipient_visits=sum(a['first_token_visit'] for a in arrivals),
            initial_exact_token_transfers=sum(a['initial_exact_token'] for a in arrivals),
            initial_lineages_visiting_other_walkers=sum(len(v)>1 for v in lineage_visits.values())))
    (folder/'provenance_transfers.json').write_text(json.dumps(serial(arrivals),separators=(',',':')))
    return result


def main():
    from examples.lisa_dns_stage4.population_level10_construction import retained_bank
    report=json.loads((OUT/'report.json').read_text())
    assert report['status'] in ['frozen_level10','stopped_failed_level10_gate']
    result=dict(status=report['status'],attempts={},population=report['population'],
        frozen_thresholds=report['frozen_thresholds'],frozen_log_masses=report['frozen_log_masses'])
    for label,path,pop in [('original isotropic',historical.SOURCE,False),('Hybrid P',historical.OUT,False),('Stage-4U population',OUT,True)]:
        raw={n:load(path/n/'trace.npz' if pop else path/f'level10_{n}_trace.npz') for n in NAMES}
        traces={n:(retained_bank(r) if pop else historical.retained_bank(r)) for n,r in raw.items()}
        row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],ELL9,LOGX9)
        stored=json.loads((path/'report.json').read_text())
        saved=stored['rows'][-1] if label=='original isotropic' else stored['decision']
        for k in ['threshold','selection','calibration','accepted']:assert serial(row[k])==saved[k]
        banks={}
        for n in NAMES:
            b=bank_summary(traces[n],row[n],row[n+'_uncertainty'])
            means=traces[n]['loglikelihood'].mean(1)
            b.update(walker_logL_mean_mad=float(np.median(abs(means-np.median(means)))),
                     walker_logL_mean_iqr=float(np.subtract(*np.percentile(means,[75,25]))))
            if pop:b['cost']=report['population'][n]['cost']
            elif label=='Hybrid P':b['cost']=stored['banks'][n]['cost']
            else:
                proxy=int((raw[n]['num_steps']+raw[n]['num_shrink']).sum())
                b['cost']=dict(slice_likelihood_proxy=proxy,slice_prior_proxy=proxy,
                    exchange_likelihood=0,exchange_prior=0,direct_cache_likelihood=3080,direct_cache_prior=3080)
            banks[n]=b
        result['attempts'][label]=dict(threshold=row['threshold'],accepted=row['accepted'],banks=banks)
    for path,h in report['file_sha256'].items():assert digest(path)==h,path
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('Three-attempt comparison and saved-operation audit complete',flush=True)


if __name__=='__main__':main()
