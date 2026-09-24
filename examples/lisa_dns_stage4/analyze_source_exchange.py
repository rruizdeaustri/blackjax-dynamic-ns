"""Read-only independent verification and summaries of Stage-4R saved exchanges."""
import hashlib
import json
from pathlib import Path
import numpy as np
from examples.lisa_dns_stage4.dominant_source_exchange import OUT, uniform_labels, pair_cases, correlation
from examples.lisa_dns_stage4.logistic_mh_refresh import safe


def main():
    report=json.loads((OUT/'report.json').read_text())
    assert report['status']=='completed_diagnostic_only'
    for path,digest in report['file_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    bases=np.load(OUT/'bases.npz');x=bases['position'];original=x.tobytes()
    q=np.load('/tmp/lisa_dns_stage4q_transplantations/bases.npz')
    assert x.tobytes()==q['position'].tobytes()
    validation=dict(structural_failures=0,verified_catalogues=0,source_states=128)
    for mode,result in report['results'].items():
        saved=np.load(OUT/f'{mode}_swaps.npz')
        for n,row in enumerate(result['joint_rows']):
            i,j,t,b,c=[row[k] for k in ['i','j','t','label_i','label_j']]
            for side,(r,d,rb,db) in enumerate([(i,j,b,c),(j,i,c,b)]):
                expected=x[r,t].copy().reshape(9,6)
                expected[rb]=x[d,t].reshape(9,6)[db]
                actual=saved['position'][n,side].reshape(9,6)
                assert actual.tobytes()==expected.tobytes()
                keep=np.arange(9)!=rb
                assert actual[keep].tobytes()==x[r,t].reshape(9,6)[keep].tobytes()
                assert actual[rb].tobytes()==x[d,t].reshape(9,6)[db].tobytes()
                validation['verified_catalogues']+=1
            np.testing.assert_array_equal(row['new_logprior'],saved['logprior'][n])
            np.testing.assert_array_equal(row['new_logL'],saved['loglikelihood'][n])
            assert row['joint_survives']==all(saved['loglikelihood'][n]>report['ell9'])
        m=np.asarray(result['joint_matrix'],float)
        np.testing.assert_array_equal(m,m.T)
        assert np.isnan(m.diagonal()).all()
    rows=report['results']['uniform']['joint_rows']
    np.testing.assert_array_equal([[z['label_i'],z['label_j']] for z in rows],uniform_labels())
    assert [(z['i'],z['j'],z['t']) for z in rows]==pair_cases()
    assert x.tobytes()==original
    dominant=report['results']['dominant'];s=np.asarray(dominant['transplant_survival_matrix'],float)
    validation['directional_asymmetries']=[dict(i=i,j=j,j_to_i=s[i,j],i_to_j=s[j,i],difference=s[i,j]-s[j,i])
        for i in range(8) for j in range(i+1,8)]
    persistent={0,2,3,4,5,7}
    validation['matched_persistent_comparison']={}
    for mode,result in report['results'].items():
        rows=[r for r in result['joint_rows'] if r['i'] in persistent and r['j'] in persistent]
        validation['matched_persistent_comparison'][mode]=dict(n=len(rows),survives=sum(r['joint_survives'] for r in rows))
    validation['uniform_selected_dominant_count']={}
    for count in range(3):
        rows=[r for r in report['results']['uniform']['joint_rows']
              if int(r['label_i']==bases['dominant'][r['i'],r['t']])
                 +int(r['label_j']==bases['dominant'][r['j'],r['t']])==count]
        validation['uniform_selected_dominant_count'][str(count)]=dict(n=len(rows),survives=sum(r['joint_survives'] for r in rows))
    validation['joint_gain_balance_correlations']={}
    for mode,result in report['results'].items():
        balance=[];outcome=[]
        for r in result['joint_rows']:
            a=bases['source_gain'][r['i'],r['t'],r['label_i']]
            b=bases['source_gain'][r['j'],r['t'],r['label_j']]
            balance.append(min(a,b)/max(a,b) if max(a,b)>0 else np.nan)
            outcome.append(r['joint_survives'])
        validation['joint_gain_balance_correlations'][mode]=correlation(balance,outcome)
    validation['total_calls_including_reporting_interruption']={
        key:report[key]+report.get('resumed_reporting_failure',{}).get(key,0)
        for key in ['prior_calls','likelihood_calls','source_gain_calls']}
    validation['max_abs_joint_prior_discrepancy']=max(report['deterministic_actual_prior_algebra']['max_abs'],
        *[r['max_abs_joint_prior_ratio'] for r in report['results'].values()])
    validation['nonfinite_priors']=sum(r['nonfinite_prior'] for r in report['results'].values())
    validation['nonfinite_likelihoods']=sum(r['nonfinite_likelihood'] for r in report['results'].values())
    (OUT/'validation.json').write_text(json.dumps(safe(validation),indent=2,allow_nan=False))
    print(json.dumps(safe(validation),indent=2))


if __name__=='__main__':main()
