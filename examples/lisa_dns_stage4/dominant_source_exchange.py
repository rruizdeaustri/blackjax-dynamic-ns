"""Stage-4R fixed-state diagnostics only; this is NOT a population kernel."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from examples.lisa_dns_stage4.source_transplant_diagnostic import (
    representative_indices, transplant, verify_saved_mapping, distribution)
from examples.lisa_dns_stage4.compare_sequential_refresh import load
from examples.lisa_dns_stage4.diagnose_source_permutations import NAMES, DF, source_distances
from examples.lisa_dns_stage4.logistic_mh_refresh import safe

OUT = Path('/tmp/lisa_dns_stage4r_source_exchange')
Q = Path('/tmp/lisa_dns_stage4q_transplantations')
SEED = 20260923


def source_ranks(gain):
    """Ranks 1..9; descending finite gain, lowest source label breaks ties."""
    gain = np.asarray(gain)
    if gain.shape[-1] != 9 or not np.isfinite(gain).all():
        raise ValueError('Nine finite source gains required')
    order = np.argsort(-gain, axis=-1, kind='stable')
    return np.argsort(order, axis=-1, kind='stable') + 1


def dominant_labels(gain):
    return np.argmin(source_ranks(gain), axis=-1)


def fixed_swap(x, y, b, c):
    """Swap labelled blocks, copying both ORIGINAL inputs; unit Jacobian."""
    return transplant(x, y, [b], [c]), transplant(y, x, [c], [b])


def joint_log_ratio(px, py, px_new, py_new):
    return (np.asarray(px_new) + np.asarray(py_new)) - (np.asarray(px) + np.asarray(py))


def pair_cases():
    return [(i, j, t) for i in range(8) for j in range(i+1, 8) for t in range(16)]


def uniform_labels(seed=SEED):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, 9, size=(448, 2))


def symmetric_matrix(rows):
    matrix = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(i + 1, 8):
            cell = [r for r in rows if (r['i'], r['j']) == (i, j)]
            if cell:
                assert len(cell) == 16
                matrix[i, j] = matrix[j, i] = np.mean([r['joint_survives'] for r in cell])
    return matrix


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), rho=None, reason='constant variable or insufficient finite data')
    return dict(n=len(x), rho=float(spearmanr(x, y).statistic),
                notice='Descriptive only; dependent repeated states; no inferential p-value')


def main():
    from examples.lisa_dns_stage4 import audit
    import jax
    import jax.numpy as jnp
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-reporting-failure', action='store_true')
    args = parser.parse_args()
    previous = None
    if args.resume_reporting_failure:
        previous = json.loads((OUT/'interrupted_report.json').read_text())
        assert previous['error'].startswith('TypeError: numpy boolean subtract')
        assert previous['results'] == {}
        assert not (OUT/'dominant_swaps.npz').exists()
    else:
        OUT.mkdir(exist_ok=False)
    qreport = json.loads((Q/'report.json').read_text())
    for path, digest in qreport['file_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    mapping, stagep = verify_saved_mapping()
    checkpoint = Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    payload = json.loads(checkpoint.read_text())['payload']
    ell9, logX9 = payload['thresholds'][-1], payload['log_masses'][-1]
    assert ell9 == -110252.99476697217 and logX9 == -7.67544001580533
    tracepath = Path('/tmp/lisa_dns_stage4p_hybrid_level10/level10_calibration_trace.npz')
    raw = load(tracepath); indices = representative_indices()
    bases = raw['position'].reshape(8,384,54)[:,128:][:,indices].copy()
    saved = load(Q/'bases.npz')
    assert bases.tobytes() == saved['position'].tobytes()
    np.testing.assert_array_equal(mapping, saved['sticky_mapping'])
    original_bytes = bases.tobytes()
    bases.setflags(write=False)
    np.testing.assert_array_equal(raw['walker'], np.repeat(np.arange(8),384))
    np.testing.assert_array_equal(raw['iteration'], np.tile(np.arange(384),8))
    comparison = json.loads(Path('/tmp/lisa_dns_stage4p_hybrid_level10/comparison.json').read_text())
    persistent = [w for w in range(8) if w not in comparison['mapped_sticky_replaced_walkers']]
    cases = pair_cases(); uniform = uniform_labels()
    design = dict(indices=indices, dominant_rule='argmax existing source_gain; lowest label breaks ties',
        rank_rule='descending source_gain, stable lowest-label ties; rank 1 is greatest',
        uniform_rng='NumPy Generator PCG64', uniform_seed=SEED, uniform_labels=uniform,
        pair_cases=cases, persistent_oracle_walkers=persistent, historical_mapping=mapping,
        counts=dict(bases=128, directed_transplants=896, dominant_joint_swaps=448, uniform_joint_swaps=448, oracle_joint_swaps=240),
        correlations='Spearman donor gain, donor rank and donor/recipient gain ratio vs directional survival, joint survival and delta logL; additionally joint min gain and mean rank. Dominant rank is constant and uninformative. No classifier.',
        diagonals='Not evaluated: NaN (serialized as string nan), never an invented success or temporal control',
        notice='Physical feasibility only. State-dependent argmax selection is NOT claimed to define an exact kernel.')
    (OUT/'design.json').write_text(json.dumps(safe(design),indent=2))
    problem,cfg,config_path = audit.load_problem()
    assert cfg == payload['provenance']['config']
    paths = [Path(p) for p in qreport['file_sha256']] + [Q/'report.json',Q/'bases.npz',Q/'source_contributions.npz',
        Path('/tmp/lisa_dns_stage4p_hybrid_level10/comparison.json'),Path(__file__),OUT/'design.json']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(status='running',ell9=ell9,logX9=logX9,file_sha256=hashes,structural_failures=0,
        prior_calls=0,likelihood_calls=0,source_gain_calls=0,new_levels=0,mcmc_steps=0,results={})
    def save(): (OUT/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    prior,like = map(jax.jit,audit.scalar_functions(problem))
    pmap = jax.jit(lambda xs:jax.lax.map(prior,xs))
    lmap = jax.jit(lambda xs:jax.lax.map(like,xs))
    def evaluate(xs, likelihood=True):
        xs=np.asarray(xs); ps=[]; ls=[]
        for start in range(0,len(xs),32):
            chunk=xs[start:start+32];ps.extend(np.asarray(pmap(chunk)).tolist());report['prior_calls']+=len(chunk)
            if likelihood:ls.extend(np.asarray(lmap(chunk)).tolist());report['likelihood_calls']+=len(chunk)
        return np.asarray(ps), np.asarray(ls)
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    def physical(x):return lo+(hi-lo)*expit(np.asarray(x).reshape(np.asarray(x).shape[:-1]+(9,6)))
    phys=physical(bases)
    try:
        if previous is None:
            bp,bl=evaluate(bases.reshape(-1,54));bp=bp.reshape(8,16);bl=bl.reshape(8,16)
        else:
            verified = load(OUT/'bases.npz')
            assert verified['position'].tobytes() == original_bytes
            bp, bl = verified['logprior'], verified['loglikelihood']
            report['resumed_reporting_failure'] = dict(
                prior_calls=previous['prior_calls'], likelihood_calls=previous['likelihood_calls'],
                source_gain_calls=previous['source_gain_calls'], structural_failures=0,
                explanation='Boolean correlation TypeError, not structural failure; interrupted report incorrectly classified all exceptions as structural. Verified bases and prior probes reused; 896 unsaved dominant evaluations repeated.')
        assert np.isfinite(bp).all() and np.isfinite(bl).all() and np.all(bl>ell9)
        cachedp=raw['logprior'].reshape(8,384)[:,128:][:,indices]
        cachedl=raw['loglikelihood'].reshape(8,384)[:,128:][:,indices]
        np.testing.assert_allclose(bp,cachedp,rtol=0,atol=1e-9)
        np.testing.assert_allclose(bl,cachedl,rtol=0,atol=1e-7)
        report['base_integrity']=dict(prior_max_error=float(abs(bp-cachedp).max()),likelihood_max_error=float(abs(bl-cachedl).max()),finite=True,strict_contour=True)
        if previous is None:
            gain=np.asarray(jax.jit(lambda xs:jax.lax.map(lambda u:problem._marg_recon(u)[6],xs))(jnp.asarray(bases.reshape(-1,54)))).reshape(8,16,9)
            report['source_gain_calls']=128
        else:
            gain = verified['source_gain']
        np.testing.assert_allclose(gain,load(Q/'source_contributions.npz')['source_gain'],rtol=1e-10,atol=1e-7)
        ranks=source_ranks(gain);dominant=dominant_labels(gain)
        np.savez_compressed(OUT/'bases.npz',position=bases,logprior=bp,loglikelihood=bl,source_gain=gain,ranks=ranks,dominant=dominant,physical=phys,historical=mapping,indices=indices)
        report['source_audit']=[dict(walker=i,retained_index=int(indices[t]),source_gain=gain[i,t],ranks=ranks[i,t],
            dominant_label=int(dominant[i,t]),dominant_f0=float(phys[i,t,dominant[i,t],0]),historical_label=int(mapping[i]),
            historical_rank=int(ranks[i,t,mapping[i]]),historical_f0=float(phys[i,t,mapping[i],0])) for i in range(8) for t in range(16)]
        report['walker_source_summary']=[dict(walker=i,dominant_label_counts=np.bincount(dominant[i],minlength=9),
            historical_top_counts={str(k):int(np.sum(ranks[i,:,mapping[i]]<=k)) for k in [1,2,3]},
            dominant_frequency_hz=distribution(phys[i,np.arange(16),dominant[i],0]),historical_frequency_hz=distribution(phys[i,:,mapping[i],0])) for i in range(8)]
        matching=[]
        for i in comparison['mapped_sticky_replaced_walkers']:
            for j in persistent:
                for t in range(16):
                    full=source_distances(phys[i,t],phys[j,t],widths,{3:np.pi,4:2*np.pi})
                    freq=source_distances(phys[i,t,:,0,None],phys[j,t,:,0,None],[DF])
                    b,c=int(dominant[i,t]),int(dominant[j,t])
                    matching.append(dict(walker=i,reference=j,index=int(indices[t]),dominant_f0_difference_bins=float(abs(phys[i,t,b,0]-phys[j,t,c,0])/DF),
                        full_matched_to_reference_dominant=full['permutation'][b]==c,frequency_matched_to_reference_dominant=freq['permutation'][b]==c,
                        full_source_set_distance=full['assignment'],frequency_set_distance_bins=freq['assignment']))
        report['posthoc_matching']=matching
        # Deterministic finite examples exercise the actual prior, including its +36 tail.
        if previous is None:
            examples=[]
            for n in range(18):
                x=np.linspace(-5,5,54)+n/100;y=np.cos(np.arange(54)+n)*4
                if n%3==0:x[(n%9)*6]=36.
                b=n%9;c=(n*5+2)%9;xx,yy=fixed_swap(x,y,b,c)
                xr,yr=fixed_swap(xx,yy,b,c);assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()
                examples.extend([x,y,xx,yy])
            ep,_=evaluate(examples,False);ep=ep.reshape(-1,4);assert np.isfinite(ep).all()
            er=joint_log_ratio(*ep.T)
            report['deterministic_actual_prior_algebra']=dict(examples=18,log_ratio=distribution(er),max_abs=float(abs(er).max()),all_finite=True)
            assert abs(er).max() < 1e-10
        else:
            report['deterministic_actual_prior_algebra'] = previous['deterministic_actual_prior_algebra']
        print('128 bases and current source gains verified; deterministic actual-prior swap audit completed',flush=True)
        # Both outcomes of dominant swaps are precisely the 896 requested directed transplants.
        for mode in ['dominant','uniform','oracle']:
            plan=[];positions=[]
            for n,(i,j,t) in enumerate(cases):
                if mode=='oracle' and (i not in persistent or j not in persistent):continue
                b,c=(dominant[i,t],dominant[j,t]) if mode=='dominant' else (uniform[n] if mode=='uniform' else (mapping[i],mapping[j]))
                b,c=int(b),int(c);x,y=bases[i,t],bases[j,t];xx,yy=fixed_swap(x,y,b,c)
                xr,yr=fixed_swap(xx,yy,b,c)
                assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()
                positions.extend([xx,yy]);plan.append(dict(i=i,j=j,t=t,index=int(indices[t]),label_i=b,label_j=c))
            newp,newl=evaluate(positions);newp=newp.reshape(-1,2);newl=newl.reshape(-1,2)
            np.savez_compressed(OUT/f'{mode}_evaluations.npz', position=np.asarray(positions).reshape(-1,2,54), logprior=newp, loglikelihood=newl)
            joint=[];directed=[];matrix=np.full((8,8),np.nan)
            for n,row in enumerate(plan):
                i,j,t,b,c=[row[k] for k in ['i','j','t','label_i','label_j']]
                passed=np.isfinite(newl[n])&(newl[n]>ell9)
                ratio=float(joint_log_ratio(bp[i,t],bp[j,t],*newp[n]))
                joint.append(dict(**row,new_logprior=newp[n],new_logL=newl[n],passes=passed,joint_survives=bool(passed.all()),joint_log_prior_ratio=ratio,
                    min_source_gain=float(min(gain[i,t,b],gain[j,t,c])),mean_source_rank=float((ranks[i,t,b]+ranks[j,t,c])/2)))
                for side,(recipient,donor,rb,db) in enumerate([(i,j,b,c),(j,i,c,b)]):
                    newphys=physical(positions[2*n+side]);oldphys=phys[recipient,t]
                    directed.append(dict(recipient=recipient,donor=donor,t=t,index=int(indices[t]),recipient_label=rb,donor_label=db,
                        logprior=float(newp[n,side]),logL=float(newl[n,side]),delta_logL=float(newl[n,side]-bl[recipient,t]),passes_ell9=bool(passed[side]),
                        passes_old_ell10=bool(np.isfinite(newl[n,side]) and newl[n,side]>qreport['old_isotropic_ell10']),
                        passes_stage4p_ell10=bool(np.isfinite(newl[n,side]) and newl[n,side]>qreport['stage4p_ell10']),
                        donor_gain=float(gain[donor,t,db]),donor_rank=int(ranks[donor,t,db]),
                        gain_ratio=float(gain[donor,t,db]/gain[recipient,t,rb]) if gain[recipient,t,rb] != 0 else np.nan,
                        joint_survives=bool(passed.all()),
                        f0_displacement_hz=float(newphys[rb,0]-oldphys[rb,0]),
                        frequency_set_displacement_bins=source_distances(oldphys[:,0,None],newphys[:,0,None],[DF])['assignment'],
                        six_coordinate_set_displacement=source_distances(oldphys,newphys,widths,{3:np.pi,4:2*np.pi})['assignment']))
            matrix=symmetric_matrix(joint)
            survival=np.array([z['joint_survives'] for z in joint]);ds=np.array([z['passes_ell9'] for z in directed])
            ratios=np.array([z['joint_log_prior_ratio'] for z in joint])
            assert np.isfinite(ratios).all() and abs(ratios).max() < 1e-10
            result=dict(n_joint=len(joint),joint_survives=int(survival.sum()),joint_fraction=float(survival.mean()),joint_matrix=matrix,
                directed_survives=int(ds.sum()),directed_n=len(directed),directed_fraction=float(ds.mean()),joint_rows=joint,directed_rows=directed,
                nonfinite_prior=int((~np.isfinite(newp)).sum()),nonfinite_likelihood=int((~np.isfinite(newl)).sum()),joint_prior_ratio=distribution(ratios),
                max_abs_joint_prior_ratio=float(np.nanmax(abs(ratios))),
                correlations=dict(donor_gain_to_transplant=correlation([z['donor_gain'] for z in directed],ds),
                    donor_rank_to_transplant=correlation([z['donor_rank'] for z in directed],ds),
                    min_gain_to_joint=correlation([z['min_source_gain'] for z in joint],survival),
                    mean_rank_to_joint=correlation([z['mean_source_rank'] for z in joint],survival)),
                geometry={str(ok):{k:distribution([z[k] for z in directed if z['passes_ell9']==ok]) for k in ['f0_displacement_hz','frequency_set_displacement_bins','six_coordinate_set_displacement']} for ok in [False,True]})
            result['predictive_correlations'] = {
                feature: {outcome: correlation([z[feature] for z in directed], [z[outcome] for z in directed])
                          for outcome in ['passes_ell9', 'joint_survives', 'delta_logL']}
                for feature in ['donor_gain', 'donor_rank', 'gain_ratio']}
            assert bases.tobytes() == original_bytes
            if mode=='dominant':
                sm=np.full((8,8),np.nan);dm=sm.copy()
                for i in range(8):
                    for j in range(8):
                        if i==j:continue
                        cell=[z for z in directed if z['recipient']==i and z['donor']==j];assert len(cell)==16
                        sm[i,j]=np.mean([z['passes_ell9'] for z in cell]);dm[i,j]=np.median([z['delta_logL'] for z in cell])
                result['transplant_survival_matrix']=sm;result['transplant_median_delta_logL']=dm
            np.savez_compressed(OUT/f'{mode}_swaps.npz',position=np.asarray(positions).reshape(-1,2,54),logprior=newp,loglikelihood=newl,
                **{k:np.array([z[k] for z in joint]) for k in joint[0] if k not in ['new_logprior','new_logL']})
            report['results'][mode]=result;save()
            print(mode,'joint',result['joint_survives'],'/',len(joint),'directed',int(ds.sum()),'/',len(ds),flush=True)
        assert bases.tobytes()==original_bytes
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        report.update(status='completed_diagnostic_only',protected_hashes_unchanged=True,
            exactness_notice='Fixed-label block exchange is an involutive coordinate permutation. Uniform state-independent selection has symmetric probability. Dynamic argmax selection is state dependent; no exact dynamic kernel has been constructed or claimed.',
            required_fixed_label_log_MH='log pi_impl(x_new)+log pi_impl(y_new)-log pi_impl(x)-log pi_impl(y), with zero acceptance if either contour fails; preserve any computed residual.',
            source_gain_notice=qreport['source_contribution_notice'])
    except Exception as exc:
        report.update(status='STOP_structural_or_execution_failure',structural_failures=int(isinstance(exc, AssertionError)),error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


if __name__=='__main__':main()
