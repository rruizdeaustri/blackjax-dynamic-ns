"""Read-only diagnostics for a stopped Stage-5B attempt; no sampling or freeze path."""
import json
from pathlib import Path
import numpy as np
from examples.lisa_dns_stage4.analyze_population_level13 import (
    OUT, NAMES, Z, ELL12, LOGX12, load, serial, digest, bank_summary, geometry, assess,
)
from examples.lisa_dns_stage4.population_level13_construction import retained_bank, recover_banks
from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint


def main():
    report=json.loads((OUT/'report.json').read_text())
    assert report['status']=='STOP_failure'
    assert report['error']=='ValueError: A walker has no survivor; stop without cloning other walkers'
    assert not (OUT/'checkpoint_level13.json').exists()
    assert not (OUT/'checkpoint_level14.json').exists()
    t,m,_,_=load_checkpoint(Z/'checkpoint_level12.json')
    traces={n:retained_bank(load(OUT/n/'trace.npz')) for n in NAMES}
    row=assess(traces['selection']['loglikelihood'],traces['calibration']['loglikelihood'],ELL12,LOGX12)
    for k,v in row.items():assert serial(v)==report['decision'][k]
    assert digest(OUT/'selection_candidate.json')==report['candidate_sha256']
    starts,recovery=recover_banks({n:load(Z/n/'trace.npz') for n in NAMES})
    banks={};physical={};halves={};extrema={};survivors={}
    for n in NAMES:
        saved=load(OUT/n/'initial_states.npz')
        for k in saved:assert saved[k].tobytes()==starts[n][k].tobytes()
        assert report['population'][n]['exact_replay_passed']
        b=bank_summary(traces[n],row[n],row[n+'_uncertainty'])
        means=traces[n]['loglikelihood'].mean(1)
        b.update(walker_logL_mean_mad=float(np.median(abs(means-np.median(means)))),
                 walker_logL_mean_iqr=float(np.subtract(*np.percentile(means,[75,25]))),
                 cost=report['population'][n]['cost'])
        banks[n]=b;physical[n]=geometry(traces[n]['position'])
        u=row[n+'_uncertainty']
        halves[n]=dict(window='First/second 128 retained sweeps; absolute sweeps 129..256 / 257..384 (1-based)',
            first_exceedance=u['walker_first_half_fractions'],second_exceedance=u['walker_second_half_fractions'],
            first_mean_logL=u['walker_logL_first_half_means'],second_mean_logL=u['walker_logL_second_half_means'])
        fractions=np.asarray(b['walker_fractions'])
        extrema[n]=dict(lowest_fraction=float(fractions.min()),highest_fraction=float(fractions.max()),
            lowest_walkers=np.flatnonzero(fractions==fractions.min()),highest_walkers=np.flatnonzero(fractions==fractions.max()))
        survivors[n]=(traces[n]['loglikelihood']>row['threshold']).sum(1)
    assert survivors['selection'][6]==0
    previous=json.loads((Z/'comparison.json').read_text())
    result=dict(status=report['status'],candidate_threshold=row['threshold'],accepted=False,
        statistical_gate_passed=bool(row['accepted']),freeze_completed=False,
        failure=report['error'],candidate_survivors=survivors,
        attempts=dict(**previous['attempts'],stage5b=dict(threshold=row['threshold'],banks=banks,
            population=report['population'],geometry=physical)),
        half_to_half=halves,bank_extrema=extrema,recovery=recovery,
        frozen_thresholds=t,frozen_log_masses=m,
        caveat='Calibration ESS passes, but the existing per-walker survivor/promotion contract fails. No alternative promotion, retry or freeze is performed.',
        file_sha256={str(p):digest(p) for p in [OUT/'report.json',OUT/'selection_candidate.json',Path(__file__)]})
    for path,h in report['file_sha256'].items():assert digest(path)==h,path
    assert (OUT/'checkpoint_level12.json').read_bytes()==(Z/'checkpoint_level12.json').read_bytes()
    (OUT/'comparison.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('Read-only failure diagnostics complete; level 12 remains frozen',flush=True)


if __name__=='__main__':main()
