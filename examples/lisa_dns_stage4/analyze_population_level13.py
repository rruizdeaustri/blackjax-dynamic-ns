"""Stage-5B exact saved-operation audit and descriptive construction comparison."""
import json
from pathlib import Path
import numpy as np
from examples.lisa_dns_stage4.level12_population_validation import ELL12,LOGX12,X,Z,V,load,serial,digest
from examples.lisa_dns_stage4.analyze_level12_population import replay,communication
from examples.lisa_dns_stage4.analyze_frequency_population import distribution,rate
from examples.lisa_dns_stage4.compare_hybrid_level10 import bank_summary
from examples.lisa_dns_stage4.analyze_population_level12 import geometry
from examples.lisa_dns_stage4.run_ladder_extension import assess

OUT=Path('/tmp/lisa_dns_stage5b_population_level13')
NAMES=['selection','calibration']


def audit_bank(folder):
    """No model evaluation: replay exact ell12 exchange/state/cache/token contracts."""
    folder=Path(folder)
    run=json.loads((folder/'report.json').read_text())
    assert run['status']=='completed_one_fixed_contour_benchmark'
    assert run['completed_sweeps']==384 and run['structural_failures']==0
    # Exact cache equality is required before freeze, beyond the historical run tolerances.
    assert run['max_cache_prior_error']==run['max_cache_logL_error']==0.
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


def main():
    from examples.lisa_dns_stage4.population_level13_construction import retained_bank,recover_banks
    report=json.loads((OUT/'report.json').read_text())
    assert report['status'] in ['frozen_level13','stopped_failed_level13_gate']
    traces={n:retained_bank(load(OUT/n/'trace.npz')) for n in NAMES}
    row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],ELL12,LOGX12)
    for k,v in row.items():assert serial(v)==report['decision'][k]
    assert serial(row['selection'])==json.loads((OUT/'selection_candidate.json').read_text())
    starts,recovery=recover_banks({n:load(Z/n/'trace.npz') for n in NAMES})
    banks={};physical={};halves={}
    for n in NAMES:
        saved=load(OUT/n/'initial_states.npz')
        for k in saved:assert saved[k].tobytes()==starts[n][k].tobytes()
        b=bank_summary(traces[n],row[n],row[n+'_uncertainty'])
        means=traces[n]['loglikelihood'].mean(1)
        b.update(walker_logL_mean_mad=float(np.median(abs(means-np.median(means)))),
                 walker_logL_mean_iqr=float(np.subtract(*np.percentile(means,[75,25]))),
                 cost=report['population'][n]['cost'])
        banks[n]=b;physical[n]=geometry(traces[n]['position'])
        u=row[n+'_uncertainty']
        halves[n]=dict(window='First/second 128 of the retained 256; absolute sweeps 129..256 / 257..384 (1-based)',
            first_exceedance=u['walker_first_half_fractions'],second_exceedance=u['walker_second_half_fractions'],
            first_mean_logL=u['walker_logL_first_half_means'],second_mean_logL=u['walker_logL_second_half_means'])
    extrema={}
    for n in NAMES:
        fractions=np.asarray(banks[n]['walker_fractions'])
        extrema[n]=dict(lowest_fraction=float(fractions.min()),highest_fraction=float(fractions.max()),
            lowest_walkers=np.flatnonzero(fractions==fractions.min()),highest_walkers=np.flatnonzero(fractions==fractions.max()),
            notice='Post-run descriptive identification only; no walker-specific probabilities')
    previous=json.loads((Z/'comparison.json').read_text())
    result=dict(status=report['status'],candidate_threshold=row['threshold'],accepted=row['accepted'],
        attempts=dict(**previous['attempts'],
            stage5b=dict(threshold=row['threshold'],banks=banks,population=report['population'],geometry=physical)),
        half_to_half=halves,bank_extrema=extrema,recovery=recovery,
        frozen_thresholds=report['frozen_thresholds'],frozen_log_masses=report['frozen_log_masses'],
        caveat='All eight lineages use identical transition settings. Half-to-half and physical diagnostics are not additional gates.',
        file_sha256={str(p):digest(p) for p in [OUT/'report.json',OUT/'selection_candidate.json',Path(__file__)]})
    for path,h in report['file_sha256'].items():assert digest(path)==h,path
    assert (OUT/'checkpoint_level12.json').read_bytes()==(Z/'checkpoint_level12.json').read_bytes()
    assert not (OUT/'checkpoint_level14.json').exists()
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('Comparison completed',report['status'],row['threshold'],row['calibration']['ess'],flush=True)


if __name__=='__main__':main()
