"""Stage-4Q fixed-base source compatibility measurements; no sampling kernel."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from examples.lisa_dns_stage4.compare_sequential_refresh import load
from examples.lisa_dns_stage4.logistic_mh_refresh import safe
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices, MODEL
from examples.lisa_dns_stage4.diagnose_source_permutations import NAMES,DF,source_distances

OUT=Path('/tmp/lisa_dns_stage4q_transplantations')


def representative_indices():
    return np.rint(np.linspace(0,255,16)).astype(int)


def temporal_index(index):
    if not 0<=index<256:raise ValueError('Retained index required')
    return (int(index)+128)%256


def nearest_companion(frequency,sticky):
    f=np.asarray(frequency,dtype=float)
    if f.shape!=(9,) or not np.isfinite(f).all() or not 0<=sticky<9:
        raise ValueError('Nine finite frequencies and a source label required')
    distances=np.abs(f-f[sticky]);distances[sticky]=np.inf
    return int(np.argmin(distances))  # lowest label on ties


def pair_assignment(recipient_f,donor_f,recipient_sticky,donor_sticky):
    """Squared frequency cost; sticky-to-sticky tie rule; no likelihood input."""
    slots=np.array([recipient_sticky,nearest_companion(recipient_f,recipient_sticky)])
    sources=np.array([donor_sticky,nearest_companion(donor_f,donor_sticky)])
    rf=np.asarray(recipient_f)[slots];df=np.asarray(donor_f)[sources]
    straight=float(np.sum(((rf-df)/DF)**2));swapped=float(np.sum(((rf-df[::-1])/DF)**2))
    swap=swapped<straight
    return slots,sources[::-1].copy() if swap else sources,swap,straight,swapped


def transplant(base,donor,recipient_labels,donor_labels):
    """Every call copies its original recipient; donor blocks copied bit exactly."""
    base=np.asarray(base);donor=np.asarray(donor)
    ri=block_indices(recipient_labels);di=block_indices(donor_labels)
    if base.shape!=(54,) or donor.shape!=(54,) or len(ri)!=len(di):
        raise ValueError('54-coordinate states and equal-sized block sets required')
    result=base.copy();result[ri]=donor[di]
    keep=np.setdiff1d(np.arange(54),ri)
    assert result[keep].tobytes()==base[keep].tobytes()
    assert result[ri].tobytes()==donor[di].tobytes()
    return result


def experiment_plan():
    """128 diagonal controls and 896 off-diagonal tests, index selection only."""
    return [dict(recipient=i,donor=j,retained_index=int(k),
                 donor_retained_index=temporal_index(k) if i==j else int(k),temporal_control=i==j)
            for i in range(8) for j in range(8) for k in representative_indices()]


def distribution(values):
    values=np.asarray(values,dtype=float);finite=values[np.isfinite(values)]
    keys=['min','p05','q25','median','q75','p95','max']
    result=dict(n=len(values),nonfinite=int(len(values)-len(finite)))
    result.update(dict(zip(keys,np.quantile(finite,[0,.05,.25,.5,.75,.95,1]).tolist())) if len(finite)
                  else dict.fromkeys(keys,None))
    return result


def compatibility_matrices(rows):
    result={key:np.zeros((8,8)) for key in ['count','survives','survival_fraction','median_delta_logL','q25_delta_logL','q75_delta_logL']}
    for i in range(8):
        for j in range(8):
            cell=[r for r in rows if r['recipient']==i and r['donor']==j]
            if len(cell)!=16:raise ValueError('Exactly 16 tests required in every cell')
            delta=np.array([r['delta_logL'] for r in cell])
            finite=delta[np.isfinite(delta)]
            result['count'][i,j]=len(cell);result['survives'][i,j]=sum(r['passes_ell9'] for r in cell)
            result['survival_fraction'][i,j]=result['survives'][i,j]/16
            for key,q in [('median_delta_logL',.5),('q25_delta_logL',.25),('q75_delta_logL',.75)]:
                result[key][i,j]=np.quantile(finite,q) if len(finite) else np.nan
    result['directional_asymmetry']=[dict(walker_i=i,walker_j=j,
        j_to_i_survival=result['survival_fraction'][i,j],i_to_j_survival=result['survival_fraction'][j,i],
        survival_difference=result['survival_fraction'][i,j]-result['survival_fraction'][j,i],
        median_delta_difference=result['median_delta_logL'][i,j]-result['median_delta_logL'][j,i])
        for i in range(8) for j in range(i+1,8)]
    return result


def verify_saved_mapping():
    p=Path('/tmp/lisa_dns_stage4p_hybrid_level10')
    report=json.loads((p/'report.json').read_text());proof=json.loads((p/'preflight.json').read_text())
    assert report['status']=='stopped_failed_level10_gate' and not report['decision']['accepted']
    for path,digest in report['file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    audit=json.loads(Path('/tmp/lisa_dns_stage4o_hybrid/bottleneck_audit.json').read_text())
    mapping=proof['sticky_mapping']['calibration']
    assert mapping==report['sticky_mapping']['calibration']==[r['label'] for r in audit['rows']]
    for kind in ['single','pair']:
        trace=load(f'/tmp/lisa_dns_stage4n_sequential/{kind}/trace.npz')
        for w,label in enumerate(mapping):
            mask=trace['walker']==w;counts=np.zeros(9,int)
            for labels,ok in zip(trace['labels'][mask],trace['accepted'][mask]):
                if ok:counts[labels]+=1
            np.testing.assert_array_equal(np.flatnonzero(counts==0),[label])
    return np.asarray(mapping),report


def main():
    from examples.lisa_dns_stage4 import audit
    import jax
    import jax.numpy as jnp
    OUT.mkdir(exist_ok=False)
    mapping,previous=verify_saved_mapping()
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    payload=json.loads(checkpoint.read_text())['payload']
    ell9=float(payload['thresholds'][-1]);logX9=float(payload['log_masses'][-1])
    assert ell9==-110252.99476697217 and logX9==-7.67544001580533
    old=json.loads(Path('/tmp/lisa_dns_stage4_ladder_batch/report.json').read_text())['rows'][-1]
    assert not old['accepted'];thresholds=[ell9,float(old['threshold']),float(previous['decision']['threshold'])]
    tracepath=Path('/tmp/lisa_dns_stage4p_hybrid_level10/level10_calibration_trace.npz')
    raw=load(tracepath)
    np.testing.assert_array_equal(raw['walker'],np.repeat(np.arange(8),384))
    np.testing.assert_array_equal(raw['iteration'],np.tile(np.arange(384),8))
    retained={k:raw[k].reshape((8,384)+raw[k].shape[1:])[:,128:] for k in ['position','logprior','loglikelihood']}
    assert np.isfinite(retained['position']).all() and np.isfinite(retained['logprior']).all()
    assert np.isfinite(retained['loglikelihood']).all() and np.all(retained['loglikelihood']>ell9)
    indices=representative_indices();bases=retained['position'][:,indices].copy();base_bytes=bases.tobytes()
    bounds_path=Path('/tmp/lisa_dns_stage4_prior_smoke/report.json')
    bounds=json.loads(bounds_path.read_text())['prior_audit'][0]['physical_bounds']
    lo=np.array([bounds[n][0] for n in NAMES]);hi=np.array([bounds[n][1] for n in NAMES]);widths=hi-lo
    widths[3]=np.pi;widths[4]=2*np.pi
    def physical(x):return lo+(hi-lo)*expit(np.asarray(x).reshape(np.asarray(x).shape[:-1]+(9,6)))
    catalogue=physical(retained['position'])
    plan=experiment_plan()
    design=dict(ell9=ell9,logX9=logX9,diagnostic_only_ell10=thresholds[1:],sticky_mapping=mapping,
        retained_indices=indices,within_walker_offset=128,plan=plan,
        assignment_rule='Nearest f0 by absolute difference; lowest label tie. Compare two complete donor-to-recipient block assignments by sum of squared frequency mismatch in Fourier bins; sticky-to-sticky on exact tie. No likelihood used.',
        donor_identity_notice='Historical labelled slots from verified genealogy; slots replaced during Stage-4P are not silently reassigned to a new likelihood-critical source.',
        classification_policy='Descriptive A/B/C/D or MIXED from raw within/cross survival and single/pair contrasts. No numerical classification cutoff chosen after results.',
        counts=dict(bases=128,single_cross=896,single_temporal=128,pair_cross=896,pair_temporal=128),
        mh_steps=0,sequential_updates=0,new_levels=0)
    # Persist all prescribed assignments BEFORE loading/evaluating the likelihood.
    plans={}
    for size,name in [(1,'single'),(2,'pair')]:
        rows=[]
        for row in plan:
            i,j,k,dk=[row[n] for n in ['recipient','donor','retained_index','donor_retained_index']]
            if size==1:
                slots=np.array([mapping[i]]);sources=np.array([mapping[j]]);swap=False;c0=c1=None
            else:slots,sources,swap,c0,c1=pair_assignment(catalogue[i,k,:,0],catalogue[j,dk,:,0],mapping[i],mapping[j])
            rows.append(dict(**row,recipient_labels=slots,donor_labels=sources,assignment_swapped=swap,
                straight_frequency_cost=c0,swapped_frequency_cost=c1))
        plans[name]=rows
    design['prescribed_assignments']=plans
    (OUT/'design.json').write_text(json.dumps(safe(design),indent=2,allow_nan=False))
    problem,cfg,config_path=audit.load_problem();assert cfg==payload['provenance']['config']
    for p in [MODEL,config_path]:assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['provenance']['file_sha256'][str(p)]
    paths=[checkpoint,tracepath,bounds_path,MODEL,config_path,OUT/'design.json',Path(__file__),
           Path('/tmp/lisa_dns_stage4p_hybrid_level10/report.json'),
           Path('/tmp/lisa_dns_stage4p_hybrid_level10/preflight.json'),
           Path('/tmp/lisa_dns_stage4o_hybrid/bottleneck_audit.json'),
           Path('/tmp/lisa_dns_stage4n_sequential/single/trace.npz'),Path('/tmp/lisa_dns_stage4n_sequential/pair/trace.npz'),
           Path('examples/lisa_dns_stage4/proposals.py'),*Path('blackjax/ns').glob('dns*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prior,like=map(jax.jit,audit.scalar_functions(problem))
    prior_map=jax.jit(lambda xs:jax.lax.map(prior,xs));like_map=jax.jit(lambda xs:jax.lax.map(like,xs))
    report=dict(status='running',design_file=str(OUT/'design.json'),ell9=ell9,logX9=logX9,
        old_isotropic_ell10=thresholds[1],stage4p_ell10=thresholds[2],file_sha256=hashes,
        structural_failures=0,likelihood_calls=0,prior_calls=0,results={},devices=str(jax.devices()),
        source_contribution_notice='Existing _marg_recon source_gain = |x_k|^2/(M_jitter^-1)_kk: conditional regularized profiled quadratic gain, with all other amplitudes reprofiled. It is nonadditive, excludes integrated logdet, and is NOT an additive per-source integrated log likelihood.')
    def save():(OUT/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    save()
    try:
        flat=bases.reshape(128,54)
        lp=np.asarray(prior_map(flat)).reshape(8,16);ll=np.asarray(like_map(flat)).reshape(8,16)
        report['prior_calls']+=128;report['likelihood_calls']+=128
        assert np.isfinite(lp).all() and np.isfinite(ll).all() and np.all(ll>ell9)
        np.testing.assert_allclose(lp,retained['logprior'][:,indices],rtol=0,atol=1e-9)
        np.testing.assert_allclose(ll,retained['loglikelihood'][:,indices],rtol=0,atol=1e-7)
        report['base_integrity']=dict(finite=True,strict_contour=True,cached_prior_max_error=float(abs(lp-retained['logprior'][:,indices]).max()),cached_likelihood_max_error=float(abs(ll-retained['loglikelihood'][:,indices]).max()))
        np.savez_compressed(OUT/'bases.npz',position=bases,logprior=lp,loglikelihood=ll,indices=indices,sticky_mapping=mapping)
        print('128 fixed bases verified; design and all assignments frozen before likelihood evaluations',flush=True)
        for name in ['single','pair']:
            rows=[];positions=[]
            for predeclared in plans[name]:
                i,j,k,dk=[predeclared[n] for n in ['recipient','donor','retained_index','donor_retained_index']]
                base=retained['position'][i,k];donor=retained['position'][j,dk]
                new=transplant(base,donor,predeclared['recipient_labels'],predeclared['donor_labels'])
                assert base.tobytes()==bases[i,np.flatnonzero(indices==k)[0]].tobytes()
                positions.append(new)
            positions=np.asarray(positions);newlp=[];newll=[]
            for start in range(0,1024,32):
                newlp.extend(np.asarray(prior_map(positions[start:start+32])).tolist())
                newll.extend(np.asarray(like_map(positions[start:start+32])).tolist())
                report['prior_calls']+=32;report['likelihood_calls']+=32
            for row,new,pvalue,lvalue in zip(plans[name],positions,newlp,newll):
                i=row['recipient'];k=row['retained_index'];bi=int(np.flatnonzero(indices==k)[0])
                oldphys=catalogue[i,k];newphys=physical(new)
                geometry=dict(frequency_set_displacement_bins=source_distances(oldphys[:,0,None],newphys[:,0,None],[DF])['assignment'],
                    six_coordinate_set_displacement=source_distances(oldphys,newphys,widths,{3:np.pi,4:2*np.pi})['assignment'])
                rows.append(dict(**row,implemented_logprior=pvalue,loglikelihood=lvalue,
                    delta_logL=lvalue-ll[i,bi],base_loglikelihood=ll[i,bi],
                    passes_ell9=bool(np.isfinite(lvalue) and lvalue>thresholds[0]),
                    passes_old_ell10=bool(np.isfinite(lvalue) and lvalue>thresholds[1]),
                    passes_stage4p_ell10=bool(np.isfinite(lvalue) and lvalue>thresholds[2]),
                    finite_prior=bool(np.isfinite(pvalue)),finite_likelihood=bool(np.isfinite(lvalue)),**geometry))
            assert bases.tobytes()==base_bytes
            # NPZ includes exact proposed coordinates; JSON keeps every cell/row diagnostic.
            np.savez_compressed(OUT/f'{name}_transplants.npz',position=positions,
                **{k:np.asarray([r[k] for r in rows]) for k in rows[0] if k not in ['straight_frequency_cost','swapped_frequency_cost']})
            result=dict(matrices=compatibility_matrices(rows),rows=rows,
                nonfinite_proposed_prior=sum(not r['finite_prior'] for r in rows),
                nonfinite_proposed_likelihood=sum(not r['finite_likelihood'] for r in rows),summaries={})
            for group,control in [('cross',False),('temporal_control',True)]:
                selected=[r for r in rows if r['temporal_control']==control]
                summary=dict(n=len(selected),survives=sum(r['passes_ell9'] for r in selected),
                    survival_fraction=float(np.mean([r['passes_ell9'] for r in selected])),delta_logL=distribution([r['delta_logL'] for r in selected]),
                    old_ell10_fraction=float(np.mean([r['passes_old_ell10'] for r in selected])),
                    stage4p_ell10_fraction=float(np.mean([r['passes_stage4p_ell10'] for r in selected])),geometry={})
                for category,ok in [('survives',True),('fails',False)]:
                    subset=[r for r in selected if r['passes_ell9']==ok]
                    summary['geometry'][category]={key:distribution([r[key] for r in subset]) for key in geometry}
                result['summaries'][group]=summary
            report['results'][name]=result;save()
            print(name,'cross survivors',result['summaries']['cross']['survives'],'/896; temporal',result['summaries']['temporal_control']['survives'],'/128',flush=True)
        # Existing reconstruction shares active likelihood basis, alive mask and jitter.
        # Do NOT use _marg_diag, which has a different numerical mask/jitter.
        @jax.jit
        def contributions(u):
            recon=problem._marg_recon(u)
            return recon[6],jnp.sum(recon[4])
        gains,residuals=jax.jit(lambda xs:jax.lax.map(contributions,xs))(flat)
        gains=np.asarray(gains).reshape(8,16,9);residuals=np.asarray(residuals).reshape(8,16)
        if not (np.isfinite(gains).all() and np.isfinite(residuals).all()):
            report['source_contribution_status']='skipped_nonfinite_existing_reconstruction'
        else:
            np.savez_compressed(OUT/'source_contributions.npz',source_gain=gains,residual_power_sum=residuals,indices=indices)
            info=[]
            for w in range(8):
                label=mapping[w];companions=[nearest_companion(catalogue[w,k,:,0],label) for k in indices]
                sticky_gains=gains[w,:,label];companion_gains=gains[w,np.arange(16),companions]
                info.append(dict(walker=w,sticky_label=int(label),sticky_gain=distribution(sticky_gains),
                    companion_gain=distribution(companion_gains),max_gain=distribution(gains[w].max(1)),
                    sticky_largest_gain_count=int(np.sum(np.argmax(gains[w],axis=1)==label)),
                    residual_power_sum=distribution(residuals[w]),
                    sticky_f0_hz=distribution(catalogue[w,indices,label,0]),
                    nearest_gap_bins=distribution([abs(catalogue[w,k,label,0]-catalogue[w,k,c,0])/DF for k,c in zip(indices,companions)]),
                    companions=companions))
            report['source_contribution_status']='existing_conditional_profile_gain_only';report['source_contributions']=info
        report['source_reconstruction_calls']=128
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        report.update(status='completed_diagnostic_only',protected_hashes_unchanged=True,new_levels=0,mcmc_steps=0)
    except Exception as exc:
        report.update(status='STOP_structural_or_execution_failure',structural_failures=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()
    print('Completed fixed diagnostic; ladder unchanged through level 9:',OUT,flush=True)


if __name__=='__main__':main()
