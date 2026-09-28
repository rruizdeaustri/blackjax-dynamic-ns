"""Stage-4X exact saved-operation audit and descriptive construction comparison."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from examples.lisa_dns_stage4.level10_population_validation import ELL10,LOGX10,V,load,serial,digest
from examples.lisa_dns_stage4.analyze_level10_population import replay,communication
from examples.lisa_dns_stage4.analyze_frequency_population import distribution,rate
from examples.lisa_dns_stage4.compare_hybrid_level10 import bank_summary
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    NAMES as COORDINATES,source_distances,cross_frequency_distance,
)
from examples.lisa_dns_stage4.run_ladder_extension import assess

OUT=Path('/tmp/lisa_dns_stage4x_population_level11')
NAMES=['selection','calibration']


def audit_bank(folder):
    """No model evaluation: replay exact ell10 exchange/state/cache/token contracts."""
    folder=Path(folder)
    run=json.loads((folder/'report.json').read_text())
    assert run['status']=='completed_one_fixed_contour_benchmark'
    assert run['completed_sweeps']==384 and run['structural_failures']==0
    raw=load(folder/'trace.npz');ex=load(folder/'exchanges.npz')
    starts=load(folder/'initial_states.npz');rng=load(folder/'random_schedule.npz')
    births=json.loads((folder/'provenance.json').read_text())
    assert raw['position'].shape==(8,384,54) and ex['pair'].shape==(1536,2)
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    replayed=replay(raw,ex,starts,rng,births,bounds['f0']);comm=communication(replayed)
    cost=run['cost']
    assert cost['slice_likelihood_proxy']==int((raw['num_steps']+raw['num_shrink']).sum())
    assert cost['exchange_likelihood']==cost['exchange_prior']==3072
    assert cost['direct_cache_likelihood']==cost['direct_cache_prior']==6152
    assert cost['total_likelihood_proxy']==cost['slice_likelihood_proxy']+3072+6152
    for path,h in run['file_sha256'].items():assert digest(path)==h,path
    assert digest(folder/'design.json')==run['design_sha256']
    (folder/'provenance_transfers.json').write_text(json.dumps(serial(replayed['arrivals']),separators=(',',':')))
    result=dict(exchange=rate(ex,np.ones(1536,bool)),
        by_pair=[dict(a=a,b=b,**rate(ex,np.all(ex['pair']==[a,b],axis=1))) for a in range(8) for b in range(a+1,8)],
        selection_log_ratio=distribution(ex['log_r_selection']),joint_prior_log_ratio=distribution(ex['delta_logprior']),
        nonfinite_prior=int(ex['nonfinite_prior'].sum()),nonfinite_likelihood=int(ex['nonfinite_likelihood'].sum()),
        nonfinite_ratio=int((~np.isfinite(ex['log_r'])).sum()),
        per_walker_accepted_exchanges=[int(np.sum(ex['accepted']&np.any(ex['pair']==w,axis=1))) for w in range(8)],
        cost=cost,provenance=comm,exact_replay_passed=True)
    return result


def geometry(position):
    """Final retained 256 window; identical Stage-4I/U/W metrics and subsampling."""
    assert position.shape==(8,256,54)
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in COORDINATES]);hi=np.array([bounds[n][1] for n in COORDINATES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    p=lo+(hi-lo)*expit(position.reshape(8,256,9,6))
    pairs=[]
    for a in range(8):
        for b in range(a+1,8):
            full=[source_distances(y,z,widths,{3:np.pi,4:2*np.pi})['assignment'] for y in p[a,31::32] for z in p[b,31::32]]
            pairs.append(dict(a=a,b=b,**cross_frequency_distance(p[a,...,0],p[b,...,0]),
                full_assignment_rms=float(np.sqrt(np.mean(np.square(full))))))
    return dict(window='retained256 = absolute sweeps 129..384 (1-based)',pairs=pairs,
        median_frequency_set_rms=float(np.median([z['assignment'] for z in pairs])),
        median_sorted_frequency_mean_distance=float(np.median([z['sorted_mean_distance'] for z in pairs])),
        median_full_source_set_rms=float(np.median([z['full_assignment_rms'] for z in pairs])))


def main():
    from examples.lisa_dns_stage4.population_level11_construction import retained_bank,recover_banks,verify_prefix
    report=json.loads((OUT/'report.json').read_text())
    assert report['status'] in ['frozen_level11','stopped_failed_level11_gate']
    traces={n:retained_bank(load(OUT/n/'trace.npz')) for n in NAMES}
    row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],ELL10,LOGX10)
    for k,v in row.items():assert serial(v)==report['decision'][k]
    assert serial(row['selection'])==json.loads((OUT/'selection_candidate.json').read_text())
    starts,recovery=recover_banks({n:load(V/n/'trace.npz') for n in NAMES})
    banks={};physical={};walker6={}
    for n in NAMES:
        saved=load(OUT/n/'initial_states.npz')
        for k in saved:assert saved[k].tobytes()==starts[n][k].tobytes()
        b=bank_summary(traces[n],row[n],row[n+'_uncertainty'])
        means=traces[n]['loglikelihood'].mean(1)
        b.update(walker_logL_mean_mad=float(np.median(abs(means-np.median(means)))),
                 walker_logL_mean_iqr=float(np.subtract(*np.percentile(means,[75,25]))),
                 cost=report['population'][n]['cost'])
        banks[n]=b;physical[n]=geometry(traces[n]['position'])
        pop=report['population'][n];comm=pop['provenance'];other=np.arange(8)!=6
        fractions=np.asarray(b['walker_fractions'])
        walker6[n]=dict(starting_logL=float(starts[n]['loglikelihood'][6]),
            retained_mean_logL=float(means[6]),exceedance_fraction=float(fractions[6]),
            other_mean_logL_range=[float(means[other].min()),float(means[other].max())],
            other_fraction_range=[float(fractions[other].min()),float(fractions[other].max())],
            received_transfers=comm['per_walker_received'][6],distinct_donors=comm['distinct_donors'][6],
            donor_ids=comm['donor_ids'][6],accepted_exchanges=pop['per_walker_accepted_exchanges'][6])
    vcomparison=json.loads((V/'comparison.json').read_text());vr=json.loads((V/'report.json').read_text())
    result=dict(status=report['status'],candidate_threshold=row['threshold'],accepted=row['accepted'],
        stage4x=dict(threshold=row['threshold'],banks=banks,population=report['population'],geometry=physical,walker6=walker6),
        stage4v=dict(vcomparison['attempts']['Stage-4U population'],population=vr['population']),
        historical_failed_level10={k:vcomparison['attempts'][k] for k in ['original isotropic','Hybrid P']},
        recovery=recovery,frozen_thresholds=report['frozen_thresholds'],frozen_log_masses=report['frozen_log_masses'],
        caveat='Walker 6 denotes ensemble lineage, not a permanently pathological physical mode. Physical geometry is diagnostic only and is not an additional gate.',
        file_sha256={str(p):digest(p) for p in [OUT/'report.json',OUT/'selection_candidate.json',Path(__file__)]})
    for path,h in report['file_sha256'].items():assert digest(path)==h,path
    assert (OUT/'checkpoint_level10.json').read_bytes()==(V/'checkpoint_level10.json').read_bytes()
    assert not (OUT/'checkpoint_level12.json').exists()
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('Comparison completed',report['status'],row['threshold'],row['calibration']['ess'],flush=True)


if __name__=='__main__':main()
