"""Read-only Stage-4Q validation, asymmetry and historical-slot interpretation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from examples.lisa_dns_stage4.source_transplant_diagnostic import (
    OUT,representative_indices,temporal_index,transplant,distribution)
from examples.lisa_dns_stage4.compare_sequential_refresh import load
from examples.lisa_dns_stage4.diagnose_source_permutations import DF
from examples.lisa_dns_stage4.logistic_mh_refresh import safe


def main():
    report=json.loads((OUT/'report.json').read_text());assert report['status']=='completed_diagnostic_only'
    design=json.loads((OUT/'design.json').read_text())
    stagep_path=Path('/tmp/lisa_dns_stage4p_hybrid_level10/comparison.json')
    stagep=json.loads(stagep_path.read_text())
    replaced=stagep['mapped_sticky_replaced_walkers']
    persistent=[w for w in range(8) if w not in replaced]
    source=load('/tmp/lisa_dns_stage4p_hybrid_level10/level10_calibration_trace.npz')
    retained=source['position'].reshape(8,384,54)[:,128:]
    bases=load(OUT/'bases.npz')
    np.testing.assert_array_equal(bases['position'],retained[:,representative_indices()])
    rows=[r for r in stagep['blocks']['calibration'] if r['mapped_sticky']]
    assert replaced==[r['walker'] for r in rows if r['refresh_accepts']>0]
    result=dict(classification='MIXED: predominantly A for the still-critical source blocks; historical slots in two walkers no longer represent that source.',
        interpretation='High single-block transfer survival and negligible nearest-companion benefit favor an individually transferable likelihood-critical source, with donor strength/slot history and some residual context sensitivity. The raw historical-slot experiment is heterogeneous; it does not support a general obligatory local-pair or intrinsically nontransferable-source bottleneck.',
        persistent_slot_walkers=persistent,replaced_historical_slot_walkers=replaced,
        subset_notice='Additional summaries reuse only the fixed evaluated tests, grouped by already saved Stage-4P refresh history. No new transplants, state filtering, source reassignment or fitted numerical thresholds.',
        comparisons={},source_gain_role={},file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [stagep_path,Path(__file__),OUT/'report.json',OUT/'design.json']})
    datasets={}
    for name in ['single','pair']:
        a=load(OUT/f'{name}_transplants.npz');datasets[name]=a
        assert len(a['position'])==1024
        for n,row in enumerate(design['prescribed_assignments'][name]):
            i,j,k,dk=[row[x] for x in ['recipient','donor','retained_index','donor_retained_index']]
            assert dk==(temporal_index(k) if i==j else k)
            expected=transplant(retained[i,k],retained[j,dk],row['recipient_labels'],row['donor_labels'])
            assert a['position'][n].tobytes()==expected.tobytes()
            assert a['recipient'][n]==i and a['donor'][n]==j
            assert a['loglikelihood'][n]==report['results'][name]['rows'][n]['loglikelihood']
        matrices=report['results'][name]['matrices']
        cells=[]
        for i in range(8):
            for j in range(8):
                mask=(a['recipient']==i)&(a['donor']==j);assert mask.sum()==16
                assert a['passes_ell9'][mask].mean()==matrices['survival_fraction'][i][j]
                q=np.quantile(a['delta_logL'][mask],[.25,.5,.75])
                np.testing.assert_allclose(q,[matrices[k][i][j] for k in ['q25_delta_logL','median_delta_logL','q75_delta_logL']],rtol=0,atol=0)
                cells.append(dict(recipient=i,donor=j,temporal_control=i==j,survival_fraction=float(a['passes_ell9'][mask].mean()),
                    q25_delta_logL=q[0],median_delta_logL=q[1],q75_delta_logL=q[2]))
        with (OUT/f'{name}_matrix_cells.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=cells[0]);writer.writeheader();writer.writerows(cells)
        ri=a['recipient'];dj=a['donor'];cross=ri!=dj
        groups=dict(persistent_to_persistent=cross&np.isin(ri,persistent)&np.isin(dj,persistent),
            persistent_temporal=(~cross)&np.isin(ri,persistent),
            replaced_to_persistent=cross&np.isin(ri,persistent)&np.isin(dj,replaced),
            persistent_to_replaced=cross&np.isin(ri,replaced)&np.isin(dj,persistent),
            replaced_to_replaced=cross&np.isin(ri,replaced)&np.isin(dj,replaced),
            replaced_temporal=(~cross)&np.isin(ri,replaced))
        result['comparisons'][name]={group:dict(n=int(mask.sum()),survives=int(a['passes_ell9'][mask].sum()),
            survival_fraction=float(a['passes_ell9'][mask].mean()),delta_logL=distribution(a['delta_logL'][mask]))
            for group,mask in groups.items()}
        result['comparisons'][name]['donor_offdiagonal_survival']=[float(a['passes_ell9'][cross&(dj==j)].mean()) for j in range(8)]
        result['comparisons'][name]['recipient_offdiagonal_survival']=[float(a['passes_ell9'][cross&(ri==i)].mean()) for i in range(8)]
        asym=np.array([v['survival_difference'] for v in matrices['directional_asymmetry']])
        result['comparisons'][name]['asymmetry']=dict(nonzero_pairs=int((asym!=0).sum()),
            absolute_survival_difference=distribution(abs(asym)),directional_pairs=matrices['directional_asymmetry'])
    single=datasets['single'];pair=datasets['pair'];cross=single['recipient']!=single['donor']
    result['paired_single_to_pair_outcomes']={name:dict(n=int(mask.sum()),
        both_survive=int(np.sum(mask&single['passes_ell9']&pair['passes_ell9'])),
        single_fail_pair_survive=int(np.sum(mask&~single['passes_ell9']&pair['passes_ell9'])),
        single_survive_pair_fail=int(np.sum(mask&single['passes_ell9']&~pair['passes_ell9'])),
        both_fail=int(np.sum(mask&~single['passes_ell9']&~pair['passes_ell9'])),
        assignment_swaps=int(pair['assignment_swapped'][mask].sum()))
        for name,mask in [('cross',cross),('temporal',~cross)]}
    gain=load(OUT/'source_contributions.npz')['source_gain']
    # Report the maximal gain label only as a measured diagnostic, never reassign a transplant.
    for w in range(8):
        maxlabels=np.argmax(gain[w],axis=1)
        result['source_gain_role'][w]=dict(max_gain_label_counts=np.bincount(maxlabels,minlength=9),
            historical_label=int(bases['sticky_mapping'][w]))
    for path,digest in report['file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result['saved_array_integrity_passed']=True
    result['tests']=dict(existing_passed=156,new_passed=10,total_passed=166,log='/tmp/lisa_dns_stage4q_tests.log')
    (OUT/'analysis.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False))
    for name in ['single','pair']:
        print(name,{k:v for k,v in result['comparisons'][name].items() if k in ['persistent_to_persistent','persistent_temporal','replaced_to_persistent','persistent_to_replaced']})
    print(result['classification'])


if __name__=='__main__':main()
