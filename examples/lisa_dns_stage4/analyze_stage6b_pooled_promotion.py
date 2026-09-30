"""Saved-operation replay and read-only diagnostics for the Stage-6B CPU promotion validation."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4.stage6b_pooled_promotion_validation import (
    OUT,B,Z,A,ELL_TEST,ELL12,LOGX12,NAMES,load,serial,digest,round_robin,
    slice_blocks,fixed_exchange,log_probabilities,draw_label,joint_decision,exchange_tags,
    CONTRACT,validate_cache,verify_source_copy,
)
from examples.lisa_dns_stage4.analyze_level12_population import communication
from examples.lisa_dns_stage4.analyze_frequency_population import distribution,rate
from examples.lisa_dns_stage4.diagnose_population_level13 import pair_geometry,distribution_metrics
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac

def replay(raw,ex,starts,rng,births,frequency_bounds):
    # Independent operation-based provenance replay and exact state/cache reconstruction.
    tokens=np.arange(72).reshape(8,9);lineage=tokens.copy();next_token=72
    old=starts['position'];arrivals=[];lineage_visits={k:{k//9} for k in range(72)};token_visits={k:{k//9} for k in range(72)}
    for s in range(raw['position'].shape[1]):
        sx=raw['post_slice_position'][:,s];sp=raw['post_slice_logprior'][:,s];sl=raw['post_slice_loglikelihood'][:,s]
        assert np.isfinite(sp).all() and np.isfinite(sl).all() and np.all(sl>ELL_TEST)
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
        assert np.isfinite(expectedp).all() and np.isfinite(expectedl).all() and np.all(expectedl>ELL_TEST)
        np.testing.assert_array_equal(tokens,raw['tokens'][:,s]);np.testing.assert_array_equal(lineage,raw['lineage'][:,s])
        old=expected
    assert next_token==len(births)
    return dict(arrivals=arrivals,token_visits=token_visits,lineage_visits=lineage_visits,
                exact_replay_passed=True,exact_token_count=len(births))


def audit_bank(folder, bounds):
    run=json.loads((folder/'report.json').read_text())
    assert run['status']=='completed_one_fixed_contour_benchmark'
    assert run['completed_sweeps']==384 and run['structural_failures']==0
    assert run['ell_test']==ELL_TEST
    assert run['max_cache_prior_error']<=CONTRACT['prior_atol']
    assert run['max_cache_logL_error']<=CONTRACT['likelihood_atol']
    raw=load(folder/'trace.npz');ex=load(folder/'exchanges.npz')
    starts=load(folder/'initial_states.npz');rng=load(folder/'random_schedule.npz')
    births=json.loads((folder/'provenance.json').read_text())
    assert raw['position'].shape==(8,384,54) and ex['pair'].shape==(1536,2)
    for prefix in ('','post_slice_'):
        x=raw[prefix+'position'].reshape(-1,54)
        validate_cache(x,x.copy(),raw[prefix+'logprior'].ravel(),raw[prefix+'loglikelihood'].ravel(),
            raw[prefix+'recomputed_prior'].ravel(),raw[prefix+'recomputed_likelihood'].ravel(),
            ELL_TEST,backend='cpu',x64=True)
    initial=load(folder/'initial_recomputation.npz')
    validate_cache(starts['position'],starts['position'].copy(),starts['logprior'],starts['loglikelihood'],
                   initial['logprior'],initial['loglikelihood'],ELL_TEST,backend='cpu',x64=True)
    r=replay(raw,ex,starts,rng,births,bounds['f0'])
    comm=communication(r)
    (folder/'provenance_transfers.json').write_text(json.dumps(serial(r['arrivals']),separators=(',',':')))
    cost=run['cost']
    assert cost['exchange_likelihood']==cost['exchange_prior']==3072
    assert cost['direct_cache_likelihood']==cost['direct_cache_prior']==6152
    for path,h in run['file_sha256'].items():assert digest(path)==h,path
    assert digest(folder/'design.json')==run['design_sha256']
    result=dict(exchange=rate(ex,np.ones(1536,bool)),
        by_pair=[dict(a=a,b=b,**rate(ex,np.all(ex['pair']==[a,b],axis=1))) for a in range(8) for b in range(a+1,8)],
        selection_log_ratio=distribution(ex['log_r_selection']),joint_prior_log_ratio=distribution(ex['delta_logprior']),
        nonfinite={k:int((~np.isfinite(v)).sum()) for k,v in ex.items() if np.issubdtype(v.dtype,np.number)},
        per_walker_accepted_exchanges=[int(np.sum(ex['accepted']&np.any(ex['pair']==w,axis=1))) for w in range(8)],
        cost=cost,provenance=comm,exact_operation_and_provenance_replay=True,
        cache_checks_frozen_contract=True,
        max_cache_prior_error=run['max_cache_prior_error'],max_cache_logL_error=run['max_cache_logL_error'])
    return raw,starts,result


def summarize(x,ll,settings):
    result=distribution_metrics(x,ll,settings)
    result.pop('fractions')  # ell_test is the target; no new diagnostic threshold.
    result.update(logL_ESS=[ac(row)['ess'] for row in ll],
        unique_state_fractions=[len(np.unique(row,axis=0))/len(row) for row in x],
        pooled_mean_logL=float(ll.mean()),pooled_quantiles=dict(zip(
            ['min','q10','q25','q50','q75','q90','max'],np.quantile(ll,[0,.1,.25,.5,.75,.9,1]))))
    result['median_logL_ESS']=float(np.median(result['logL_ESS']))
    return result


def snapshot(x,settings):
    pairs=[dict(a=a,b=b,**pair_geometry(x[a,None],x[b,None],settings)) for a in range(8) for b in range(a+1,8)]
    return dict(pairs=pairs,unique_states=len(np.unique(x,axis=0)),
        distances={k:distribution([p[k] for p in pairs]) for k in ['assignment','sorted_mean_distance','full_assignment_rms']})


def analyze():
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    from examples.lisa_dns_stage4.diagnose_source_permutations import NAMES as COORDS
    lo=np.array([bounds[n][0] for n in COORDS]);hi=np.array([bounds[n][1] for n in COORDS]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi;settings=(lo,hi,widths)
    promotion=json.loads((OUT/'promotion.json').read_text())
    histories={};banks={}
    for name in NAMES:
        raw,starts,audit=audit_bank(OUT/name,bounds)
        old=load(B/name/'trace.npz');proof=promotion[name]
        donor=np.array(proof['donor_ids']);ix=np.array(proof['absolute_trace_indices'])
        for k,v in starts.items():assert v.tobytes()==old[k][donor,ix].tobytes()
        verify_source_copy(starts,{k:old[k][:,128:] for k in starts},proof)
        windows={label:summarize(raw['position'][:,sl],raw['loglikelihood'][:,sl],settings)
                 for label,sl in [('first128',slice(128,256)),('second128',slice(256,384)),('full256',slice(128,384))]}
        duplicates=[dict(a=a,b=b,still_identical_after_burn=bool(np.array_equal(raw['position'][a,127],raw['position'][b,127])))
            for a in range(8) for b in range(a+1,8) if np.array_equal(starts['position'][a],starts['position'][b])]
        stages=dict(promoted=snapshot(starts['position'],settings),postburn=snapshot(raw['position'][:,127],settings))
        for label in ['first128','second128','full256']:
            stages[label]=dict(distances={k:distribution([p[k] for p in windows[label]['pairs']])
                for k in ['assignment','sorted_mean_distance','full_assignment_rms']})
        banks[name]=dict(windows=windows,promotion_stages=stages,duplicate_evolution=duplicates,**audit)
        histories[name]=raw
    cross={}
    for label,sl in [('first128',slice(128,256)),('second128',slice(256,384)),('full256',slice(128,384))]:
        ll=[histories[n]['loglikelihood'][:,sl] for n in NAMES]
        pairvalues=[dict(selection_walker=a,calibration_walker=b,**pair_geometry(
            histories['selection']['position'][a,sl],histories['calibration']['position'][b,sl],settings))
            for a in range(8) for b in range(8)]
        cross[label]=dict(pooled_KS=float(ks_2samp(ll[0].ravel(),ll[1].ravel()).statistic),
            mean_logL=[float(v.mean()) for v in ll],
            quantiles=[np.quantile(v,[.1,.25,.5,.75,.9]) for v in ll],
            geometry={k:distribution([p[k] for p in pairvalues]) for k in ['assignment','sorted_mean_distance','full_assignment_rms']},pairs=pairvalues)
    historical=json.loads(Path('/tmp/lisa_dns_stage5c_readonly_diagnosis/diagnostics.json').read_text())
    gate=json.loads((OUT/'integrity_gate.json').read_text())
    assert gate['passed'] and gate['contract']==CONTRACT
    for path,h in gate['file_sha256'].items():assert digest(path)==h,path
    environment=json.loads((OUT/'environment.json').read_text())
    assert environment['backend']=='cpu' and environment['x64'] and environment['dtype']=='float64'
    preflight=json.loads((OUT/'preflight.json').read_text())
    for path,h in preflight['file_sha256'].items():assert digest(path)==h,path
    assert not any(OUT.rglob('checkpoint*'))
    assert not any(B.rglob('checkpoint_level13*'))
    result=dict(ell_test=ELL_TEST,promotion=promotion,banks=banks,cross_bank=cross,
        environment=environment,contract=CONTRACT,preflight=preflight,pre_sampling_integrity=gate,
        cross_bank_quantile_order=['q10','q25','q50','q75','q90'],
        historical={k:historical[k] for k in ['windows','exchanges','provenance']},
        integrity=dict(protected_hashes_unchanged=True,frozen_levels=list(range(13)),no_level13=True,
            no_mass_update=True,exact_operation_and_provenance_replay=True,zero_structural_failures=True,
            source_hashes={str(p):digest(p) for p in [Path(__file__),Path('/tmp/lisa_dns_stage5c_readonly_diagnosis/diagnostics.json')]}))
    (OUT/'diagnostics.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print(json.dumps(serial({n:{label:{k:v for k,v in window.items() if k not in ['pairs','unique_state_fractions','pooled_quantiles']}
                                  for label,window in banks[n]['windows'].items()} for n in NAMES}),indent=2))
    return result




if __name__=='__main__':analyze()
