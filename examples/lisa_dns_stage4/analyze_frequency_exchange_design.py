"""Stage 4T: saved-array-only soft-frequency proposal algebra; no sampler."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from examples.lisa_dns_stage4.analyze_uniform_source_relevance import (
    ELL9, INPUT, distribution, ranks_from_saved_gain, verify_saved_inputs,
)

OUTPUT = Path('/tmp/lisa_dns_stage4t_frequency_design')
INTERVAL = ('0.0018407250', '0.0018412366')


def fixed_frequency_region():
    low, high = map(Decimal, INTERVAL)
    return float((low+high)/2), float((high-low)/2)


def frequency_probabilities(frequency):
    """Unmodified Gaussian, log-space normalization; longdouble avoids underflow."""
    f = np.asarray(frequency, dtype=float)
    if f.shape[-1] != 9 or not np.isfinite(f).all():
        raise ValueError('Nine finite physical frequencies required')
    center, sigma = fixed_frequency_region()
    logw = -0.5*((f-center)/sigma)**2
    logz = logsumexp(logw, axis=-1)
    logp = logw-logz[...,None]
    p = np.exp(logp.astype(np.longdouble))
    if not np.isfinite(p).all() or not (p>0).all():
        raise ValueError('Platform precision cannot represent strictly positive weights')
    np.testing.assert_allclose(p.sum(axis=-1),1,rtol=0,atol=2e-13)
    return p, logp, logz


def fixed_exchange(x, y, i, j):
    """Original labelled slots i,j remain i,j in the reverse exchange."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape!=(54,) or y.shape!=(54,) or not (0<=i<9 and 0<=j<9):
        raise ValueError('54-coordinate catalogues and labels 0..8 required')
    xx, yy = x.copy().reshape(9,6), y.copy().reshape(9,6)
    xx[i] = y.reshape(9,6)[j]
    yy[j] = x.reshape(9,6)[i]
    return xx.reshape(54), yy.reshape(54)


def selection_log_ratio(logp_x, logp_y, logp_x_new, logp_y_new, i, j):
    return float(logp_x_new[i]+logp_y_new[j]-logp_x[i]-logp_y[j])


def corrected_alpha(log_selection_ratio, both_survive, joint_prior_delta=0.):
    """Expected MH probability only; no random draw or state update."""
    ratio = np.asarray(log_selection_ratio)+np.asarray(joint_prior_delta)
    if not np.isfinite(ratio).all():
        raise ValueError('Finite log MH ratios required')
    return np.where(both_survive, np.exp(np.minimum(0.,ratio)), 0.)


def decimal_probabilities(p):
    """JSON decimals as strings preserve sub-float64 probabilities without clipping."""
    p = np.asarray(p)
    if p.ndim == 1:
        return [np.format_float_scientific(v,precision=18) for v in p]
    return [decimal_probabilities(v) for v in p]


def replay_saved(base, swaps, frequency_bounds):
    """No model functions: source identities and logL/prior come only from arrays."""
    ranks = ranks_from_saved_gain(base['source_gain'])
    np.testing.assert_array_equal(ranks,base['ranks'])
    dominant = np.argmin(ranks,axis=-1)
    np.testing.assert_array_equal(dominant,base['dominant'])
    frequency = base['physical'][...,0]
    p, logp, logz = frequency_probabilities(frequency)
    f_low, f_high = frequency_bounds
    from_positions = f_low+(f_high-f_low)*expit(base['position'].reshape(base['position'].shape[:-1]+(9,6))[...,0])
    np.testing.assert_array_equal(frequency,from_positions)
    proposed_frequency = f_low+(f_high-f_low)*expit(swaps['position'].reshape(-1,2,9,6)[...,0])
    post_p, post_logp, post_logz = frequency_probabilities(proposed_frequency)
    rows = []
    for n in range(len(swaps['i'])):
        a,b,t,i,j = [int(swaps[k][n]) for k in ('i','j','t','label_i','label_j')]
        assert (i,j)==(int(dominant[a,t]),int(dominant[b,t]))
        x,y = base['position'][a,t],base['position'][b,t]
        xx,yy = fixed_exchange(x,y,i,j)
        assert xx.tobytes()==swaps['position'][n,0].tobytes()
        assert yy.tobytes()==swaps['position'][n,1].tobytes()
        xr,yr = fixed_exchange(swaps['position'][n,0],swaps['position'][n,1],i,j)
        assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()
        for side,(recipient,donor,rb,db) in enumerate(((a,b,i,j),(b,a,j,i))):
            expected_frequency = frequency[recipient,t].copy()
            expected_frequency[rb] = frequency[donor,t,db]
            np.testing.assert_array_equal(expected_frequency,proposed_frequency[n,side])
        ratio = selection_log_ratio(logp[a,t],logp[b,t],post_logp[n,0],post_logp[n,1],i,j)
        simplified = float(logz[a,t]+logz[b,t]-post_logz[n,0]-post_logz[n,1])
        assert abs(ratio-simplified)<2e-13
        reverse = selection_log_ratio(post_logp[n,0],post_logp[n,1],logp[a,t],logp[b,t],i,j)
        assert abs(ratio+reverse)<2e-13
        ll = swaps['loglikelihood'][n]
        assert np.isfinite(ll).all()
        survives = bool(np.all(ll>ELL9))
        assert survives==bool(swaps['joint_survives'][n])
        prior_delta = float(swaps['logprior'][n].sum()-base['logprior'][a,t]-base['logprior'][b,t])
        # Match Stage-4R summation grouping when preserving the tiny numerical residual.
        prior_delta = float(swaps['logprior'][n].sum()-(base['logprior'][a,t]+base['logprior'][b,t]))
        assert abs(prior_delta-float(swaps['joint_log_prior_ratio'][n]))<2e-13
        rows.append(dict(case=n,A=a,B=b,t=t,index=int(base['indices'][t]),label_A=i,label_B=j,
            reverse_label_A=i,reverse_label_B=j,
            p_A=float(p[a,t,i]),p_B=float(p[b,t,j]),P_both=float(p[a,t,i]*p[b,t,j]),
            reverse_p_A=float(post_p[n,0,i]),reverse_p_B=float(post_p[n,1,j]),
            log_q_forward=float(logp[a,t,i]+logp[b,t,j]),
            log_q_reverse=float(post_logp[n,0,i]+post_logp[n,1,j]),
            log_r_selection=ratio,normalizer_log_ratio=simplified,joint_prior_delta=prior_delta,
            joint_survives=survives,
            alpha_selection_corrected=float(corrected_alpha(ratio,survives)),
            alpha_with_saved_prior_residual=float(corrected_alpha(ratio,survives,prior_delta))))
    selection_order = np.argsort(-logp,axis=-1,kind='stable')
    nearest = np.argmin(abs(frequency-fixed_frequency_region()[0]),axis=-1)
    np.testing.assert_array_equal(nearest,selection_order[...,0])
    states=[]
    for walker in range(frequency.shape[0]):
        for t in range(frequency.shape[1]):
            dom = int(dominant[walker,t])
            states.append(dict(walker=walker,t=t,index=int(base['indices'][t]),
                maximum_p_label=int(selection_order[walker,t,0]),source_gain_dominant=dom,
                nearest_label=int(nearest[walker,t]),
                top2_contains_dominant=bool(dom in selection_order[walker,t,:2]),
                top3_contains_dominant=bool(dom in selection_order[walker,t,:3]),
                p_dominant=float(p[walker,t,dom]),probabilities_decimal=decimal_probabilities(p[walker,t]),
                log_probabilities=logp[walker,t].tolist()))
    groups={}
    for name,walkers in [('persistent_six',[0,2,3,4,5,7]),('replaced_1_6',[1,6]),
                         ('walker_1',[1]),('walker_6',[6]),('all',list(range(8)))]:
        group=[s for s in states if s['walker'] in walkers]
        if not group:
            continue
        groups[name]=dict(n=len(group),agreement=sum(s['maximum_p_label']==s['source_gain_dominant'] for s in group),
            top2=sum(s['top2_contains_dominant'] for s in group),top3=sum(s['top3_contains_dominant'] for s in group),
            p_dominant=distribution([s['p_dominant'] for s in group]),
            p_dominant_mean=float(np.mean([s['p_dominant'] for s in group])))
    survivors=[r for r in rows if r['joint_survives']]
    return dict(states=states,groups=groups,pairs=rows,
        P_both=distribution([r['P_both'] for r in rows]),uniform_P_both=1/81,
        replay=dict(n=len(rows),raw_joint_survivors=len(survivors),raw_joint_fraction=len(survivors)/len(rows),
            mean_alpha_selection_corrected=float(np.mean([r['alpha_selection_corrected'] for r in rows])),
            mean_alpha_with_saved_prior_residual=float(np.mean([r['alpha_with_saved_prior_residual'] for r in rows])),
            alpha_among_survivors=distribution([r['alpha_selection_corrected'] for r in survivors]),
            mean_alpha_among_survivors=float(np.mean([r['alpha_selection_corrected'] for r in survivors])),
            log_r_selection_among_survivors=distribution([r['log_r_selection'] for r in survivors]),
            max_prior_residual_effect=max(abs(r['alpha_selection_corrected']-r['alpha_with_saved_prior_residual']) for r in rows),
            max_normalizer_identity_error=max(abs(r['log_r_selection']-r['normalizer_log_ratio']) for r in rows)),
        post_probabilities_decimal=decimal_probabilities(post_p),post_log_probabilities=post_logp.tolist(),
        integrity=dict(probabilities_finite=True,probabilities_positive=True,
            min_probability_decimal=np.format_float_scientific(min(p.min(),post_p.min()),precision=18),
            min_log_probability=float(min(logp.min(),post_logp.min())),
            max_sum_error=float(max(abs(p.sum(axis=-1)-1).max(),abs(post_p.sum(axis=-1)-1).max())),
            float64_underflow_entries=int((p<np.nextafter(0.,1.)).sum()+(post_p<np.nextafter(0.,1.)).sum()),
            longdouble_smallest_subnormal=str(np.nextafter(np.longdouble(0),np.longdouble(1))),
            structural_failures=0,saved_positions_exact=True,reverse_labels_unchanged=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    # A design file must already exist: freeze constants before outcome analysis.
    design_path=args.output/'design.json'
    design=json.loads(design_path.read_text())
    assert design['frequency_interval_hz']==list(INTERVAL)
    assert float(design['f_star_hz'])==fixed_frequency_region()[0]
    assert float(design['Delta_f_hz'])==fixed_frequency_region()[1]
    assert not (args.output/'analysis.json').exists()
    stage_s=Path('/tmp/lisa_dns_stage4s_uniform_relevance/analysis.json')
    saved_s=json.loads(stage_s.read_text())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path,expected in saved_s['input_sha256'].items():
        assert digest(path)==expected,path
    base,swaps,_,hashes,verified=verify_saved_inputs(INPUT)
    hashes.update({str(stage_s):digest(stage_s),str(design_path):digest(design_path),
                   str(Path(__file__).resolve()):digest(__file__)})
    bounds_path=Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    bounds=json.loads(bounds_path.read_text())['prior_audit'][0]['physical_bounds']['f0']
    result=replay_saved(base,swaps['dominant'],bounds)
    assert len(result['states'])==128 and len(result['pairs'])==448
    assert result['replay']['raw_joint_survivors']==265
    result.update(stage='4T',status='completed_saved_array_design_audit',ell9=ELL9,design=design,
                  input_sha256=hashes,existing_provenance_hashes_verified=verified,
                  stage_s_input_hashes_verified=len(saved_s['input_sha256']),
                  likelihood_calls=0,prior_calls=0,source_gain_calls=0,mcmc_steps=0,
                  replay_notice='Unweighted replay of the 448 prescribed dominant-label proposals. This is not overall acceptance or efficiency of a soft-selector chain.')
    for path,expected in hashes.items():
        assert digest(path)==expected,path
    result['input_hashes_unchanged']=True
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['groups','P_both','replay','integrity']},indent=2))


if __name__=='__main__':
    main()
