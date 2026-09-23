"""Stage-4O saved-trace bottleneck audit; no model calls or sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from examples.lisa_dns_stage4.compare_sequential_refresh import load
from examples.lisa_dns_stage4.diagnose_source_permutations import NAMES, DF, source_distances
from examples.lisa_dns_stage4.logistic_mh_refresh import safe

OUT=Path('/tmp/lisa_dns_stage4o_hybrid')


def main():
    OUT.mkdir(exist_ok=False)
    h=Path('/tmp/lisa_dns_stage4h_continuation');n=Path('/tmp/lisa_dns_stage4n_sequential')
    paths=[h/'initial_states.npz',h/'trace.npz',n/'initial_states.npz',
           n/'single/trace.npz',n/'pair/trace.npz',
           Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')]
    start=load(paths[0]);baseline=load(paths[1]);other=load(paths[2])
    for k in start: assert start[k].tobytes()==other[k].tobytes()
    bounds=json.loads(paths[-1].read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[k][0] for k in NAMES]);hi=np.array([bounds[k][1] for k in NAMES])
    widths=hi-lo;widths[3]=np.pi;widths[4]=2*np.pi
    initial=lo+(hi-lo)*expit(start['position'].reshape(8,9,6))
    traces={name:load(n/name/'trace.npz') for name in ['single','pair']}
    report=dict(status='saved_trace_audit_completed_before_sampling',model_calls=0,
        fixed_hybrid_definition='512 sweeps per walker; unchanged isotropic slice followed by one single/pair exact refresh. No source-dependent tuning.',
        rows=[],pairwise_initial_catalogue_matching=[],
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__)]})
    sticky=[]
    for w in range(8):
        counts={}
        for name,a in traces.items():
            mask=a['walker']==w;attempted=np.zeros(9,int);accepted=np.zeros(9,int)
            for labels,ok in zip(a['labels'][mask],a['accepted'][mask]):
                attempted[labels]+=1
                if ok:accepted[labels]+=1
            never=np.flatnonzero(accepted==0)
            counts[name]=dict(attempted=attempted,accepted=accepted,never_refreshed=never)
        np.testing.assert_array_equal(counts['single']['never_refreshed'],counts['pair']['never_refreshed'])
        assert len(counts['single']['never_refreshed'])==1
        label=int(counts['single']['never_refreshed'][0]);sticky.append(label)
        ix=np.arange(6*label,6*label+6)
        x=np.concatenate([start['position'][w:w+1],baseline['position'][baseline['walker']==w]])[:,ix]
        f=lo[0]+(hi[0]-lo[0])*expit(x[:,0])
        for name,a in traces.items():
            block=a['position'][a['walker']==w][:,ix]
            assert np.all(block==start['position'][w,ix])
        report['rows'].append(dict(walker=w,label=label,refresh_counts=counts,
            initial_and_stage4n_current_f0_hz=float(f[0]),
            stage4h_f0_final_hz=float(f[-1]),stage4h_f0_min_hz=float(f.min()),stage4h_f0_max_hz=float(f.max()),
            stage4h_f0_net_bins=float((f[-1]-f[0])/DF),stage4h_f0_range_bins=float(np.ptp(f)/DF),
            stage4h_f0_path_bins=float(np.abs(np.diff(f)).sum()/DF),
            stage4h_block_net_l2=float(np.linalg.norm(x[-1]-x[0])),
            stage4h_block_path_l2=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),
            stage4h_block_max_from_start_l2=float(np.linalg.norm(x-x[0],axis=1).max()),
            stage4h_block_change_count=int(np.any(np.diff(x,axis=0)!=0,axis=1).sum())))
    for i in range(8):
        for j in range(i+1,8):
            full=source_distances(initial[i],initial[j],widths,{3:np.pi,4:2*np.pi})
            freq=source_distances(initial[i,:,0,None],initial[j,:,0,None],[DF])
            report['pairwise_initial_catalogue_matching'].append(dict(walker_i=i,walker_j=j,
                sticky_frequency_difference_bins=float(abs(initial[i,sticky[i],0]-initial[j,sticky[j],0])/DF),
                full_assignment=full,frequency_assignment=freq,
                full_matches_sticky_to_sticky=full['permutation'][sticky[i]]==sticky[j],
                frequency_matches_sticky_to_sticky=freq['permutation'][sticky[i]]==sticky[j]))
    report['matching_notice']='Assignments compare complete physical catalogues. Close frequencies or matched labels do not establish equal likelihood importance; no source-gain or named-family inference is made.'
    (OUT/'bottleneck_audit.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    for row in report['rows']:
        print('walker',row['walker'],'sticky',row['label'],'attempts',*[int(row['refresh_counts'][k]['attempted'][row['label']]) for k in ['single','pair']],
              'f0',row['initial_and_stage4n_current_f0_hz'],'slice f0 range bins',row['stage4h_f0_range_bins'],flush=True)


if __name__=='__main__':main()
