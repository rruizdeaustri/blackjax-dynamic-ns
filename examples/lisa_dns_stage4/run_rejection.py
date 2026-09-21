"""IID level-1 rejection diagnostic; never runs MCMC or DNS production."""
import argparse
import hashlib
import json
from pathlib import Path

from examples.lisa_dns_stage4 import audit
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4 import prior_bootstrap as bootstrap
from examples.lisa_dns_stage4.run_smoke import json_safe


def condition_bank(positions, logprior, loglikelihood, threshold, target):
    """Filter a fixed-budget prior bank, retaining its first target successes.

    Count all successes, including excess ones, in the acceptance denominator.
    Fixed proposal budgets make acceptance a binomial observation, not a
    stopping-time fraction. Taking the first target successes preserves IID
    conditional draws; insufficient successes is a failure, never a retry.
    """
    positions, logprior, loglikelihood = map(np.asarray, (positions, logprior, loglikelihood))
    n = len(positions)
    if positions.ndim != 2 or logprior.shape != (n,) or loglikelihood.shape != (n,):
        raise ValueError('Inconsistent bank shapes')
    if not all(np.all(np.isfinite(x)) for x in (positions, logprior, loglikelihood)):
        raise ValueError('Nonfinite bank values')
    if target < 1:
        raise ValueError('target must be positive')
    accepted_indices = np.flatnonzero(loglikelihood > threshold)
    retained = accepted_indices[:target]
    summary = dict(proposed=n, accepted=len(accepted_indices), retained=len(retained),
                   excess_accepted=max(0, len(accepted_indices)-target),
                   empirical_prior_acceptance=len(accepted_indices)/n,
                   target=target, success=len(retained)==target,
                   acceptance_uncertainty=bootstrap.calibrate_iid(loglikelihood, threshold))
    return retained, summary


def independent_banks(sample_prior, seeds, proposed):
    return bootstrap.draw_prior_banks(sample_prior, seeds[0], seeds[1], proposed)


def calibrate_level2(selection_logL, calibration_logL, x1):
    """Only selection determines the threshold; only calibration estimates r1."""
    selection = bootstrap.select_iid_threshold(selection_logL)
    if selection['status'] != 'ok':
        return dict(selection=selection, accepted=False)
    calibration = bootstrap.calibrate_iid(calibration_logL, selection['threshold'])
    row = dict(selection=selection, calibration=calibration, accepted=calibration['accepted'])
    if row['accepted']:
        row.update(X2=x1*calibration['ratio'], log_X2=float(np.log(x1)+calibration['log_ratio']),
                   X2_interval_conditional_on_Xhat1=(x1*np.array(calibration['interval'])).tolist(),
                   uncertainty_notice='The scaled r1 interval treats Xhat1 as fixed; it is not a joint confidence interval for X2.')
    return row


def describe(values):
    values = np.asarray(values)
    return dict(mean=values.mean(axis=0).tolist(), variance=values.var(axis=0, ddof=1).tolist(),
                quantiles=np.quantile(values, [.05,.25,.5,.75,.95], axis=0).tolist())


def compare_saved_walkers(trace, reference, reference_logL, previous, physical_f0):
    """Descriptive comparisons only: do not assign IID p-values to MCMC rows."""
    ref_mean = reference.mean(0)
    ref_var = reference.var(0, ddof=1)
    f0_columns = np.arange(0, 54, 6)
    ref_f0 = physical_f0(reference[:, f0_columns])
    result = dict(reference_count=len(reference), reference_logL=describe(reference_logL),
                  reference_latent=describe(reference), reference_physical_f0=describe(ref_f0), walkers=[],
                  notice='KS distances are descriptive; MCMC dependence precludes naive IID p-values. Source labels are fixed/exchangeable, not identified physical modes.')
    for bank in ['selection','calibration']:
        starts = np.asarray(next(b['initial_positions'] for b in previous['walker_initialization'] if b['bank']==bank))
        for w in range(8):
            all_rows = (trace['phase']==bank) & (trace['walker']==w)
            retained = all_rows & (trace['iteration']>=128)
            x, ll = trace['position'][retained], trace['loglikelihood'][retained]
            if x.shape != (256,54):
                raise ValueError('Unexpected saved walker history')
            end = trace['position'][all_rows][-1]
            f0 = physical_f0(x[:,f0_columns])
            f0_delta = end[f0_columns]-starts[w,f0_columns]
            result['walkers'].append(dict(bank=bank, walker=w, retained=len(x), logL=describe(ll),
                logL_KS_distance=float(ks_2samp(ll,reference_logL).statistic),
                latent=describe(x), latent_mean_shift_in_reference_sd=((x.mean(0)-ref_mean)/np.sqrt(ref_var)).tolist(),
                latent_variance_ratio_to_reference=(x.var(0,ddof=1)/ref_var).tolist(),
                f0_latent_end_minus_start=f0_delta.tolist(), f0_latent_mean_absolute_movement=float(np.abs(f0_delta).mean()),
                physical_f0=describe(f0),
                physical_f0_KS_distances=[float(ks_2samp(f0[:,k],ref_f0[:,k]).statistic) for k in range(9)]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('/tmp/lisa_dns_stage4_prior_smoke'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/lisa_dns_stage4_rejection'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    previous=json.loads((args.previous/'report.json').read_text())
    level1=previous['ladder'][0]
    assert level1['accepted']
    ell1=level1['selection']['threshold']
    x1=level1['calibration']['ratio']
    seeds=[4424,4425]
    report=dict(notice=audit.NOTICE,status='initializing',ell1=ell1,Xhat1=x1,
                settings=dict(proposed_per_bank=2048,retained_per_bank=512,chunk_size=32,seeds=seeds),
                fixed_budget_notice='2048 proposals per bank, then retain first 512 successes. No retry or budget extension.',
                production='not_run_by_design',new_MCMC_steps=0,proposal_changes=False)
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    try:
        problem,cfg,config_path=audit.load_problem()
        assert cfg==previous['config']
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        source=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        # Stop if the audited model/config changed since the accepted bootstrap.
        for p in [source,config_path]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==previous['file_sha256'][str(p)]
        report.update(config=cfg,config_path=str(config_path),devices=str(jax.devices()),jax_version=jax.__version__,
            file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [source,config_path,Path(__file__),Path(bootstrap.__file__),args.previous/'report.json',args.previous/'trace.npz']})
        banks,metadata=independent_banks(problem.sample_prior,seeds,2048)
        report['independence']=metadata
        evaluate=jax.jit(lambda xs:jax.lax.map(lambda x:(prior(x),like(x)),xs))
        retained_banks, retained_likes = [], []
        report['banks']=[]
        for b,name in enumerate(['selection','calibration']):
            lp,ll=[],[]
            assert banks[b].shape==(2048,54) and banks[b].dtype==np.float64
            for i in range(0,2048,32):
                p,l=evaluate(jnp.asarray(banks[b][i:i+32]))
                lp.extend(np.asarray(p));ll.extend(np.asarray(l))
                if (i+32)%256==0:
                    print(name,'evaluated',i+32,flush=True)
            lp,ll=np.asarray(lp),np.asarray(ll)
            indices,summary=condition_bank(banks[b],lp,ll,ell1,512)
            summary.update(name=name,seed=seeds[b],retained_sha256=bootstrap.array_sha256(banks[b][indices]))
            old_se=level1['calibration']['standard_error']
            summary['acceptance_difference_from_previous_Xhat1']=summary['empirical_prior_acceptance']-x1
            summary['acceptance_difference_in_combined_SE']=float((summary['empirical_prior_acceptance']-x1)/np.sqrt(
                old_se**2+summary['acceptance_uncertainty']['standard_error']**2))
            report['banks'].append(summary)
            np.savez_compressed(args.output/f'{name}_bank.npz',positions=banks[b],logprior=lp,loglikelihood=ll,
                                accepted_mask=ll>ell1,retained_indices=indices)
            save()
            if not summary['success']:
                report['status']='stopped_insufficient_rejection_samples'
                return
            retained_banks.append(banks[b][indices]);retained_likes.append(ll[indices])
        row=calibrate_level2(*retained_likes,x1)
        report['IID_level2']=row
        old=previous['ladder'][1]
        report['previous_MCMC_level2']=dict(threshold=old['selection']['threshold'],selection=old['selection'],calibration=old['calibration'])
        # The old MCMC threshold is fixed independently of these new banks.
        report['IID_at_previous_MCMC_threshold']=bootstrap.calibrate_iid(np.concatenate(retained_likes),old['selection']['threshold'])
        if not row['accepted']:
            report['status']='stopped_IID_calibration'
            return
        report['IID_level2']['target_exp_minus_one_in_calibration_interval']=bool(
            row['calibration']['interval'][0] <= np.exp(-1) <= row['calibration']['interval'][1])
        # A simple labelled-coordinate decode uses the existing model transform.
        from jax_samplers.problems.lisa_gb_transdim_problem import u_to_f0_unordered
        physical=lambda u:np.asarray(u_to_f0_unordered(jnp.asarray(u),problem.f_min_cfg,problem.f_max_cfg))
        trace=np.load(args.previous/'trace.npz')
        comparison=compare_saved_walkers(trace,np.concatenate(retained_banks),np.concatenate(retained_likes),previous,physical)
        (args.output/'walker_comparison.json').write_text(json.dumps(json_safe(comparison),indent=2,allow_nan=False))
        fcols=np.arange(0,54,6)
        report['walker_comparison_summary']=dict(
            logL_KS_range=[min(w['logL_KS_distance'] for w in comparison['walkers']),max(w['logL_KS_distance'] for w in comparison['walkers'])],
            f0_variance_ratio_median=float(np.median([np.array(w['latent_variance_ratio_to_reference'])[fcols] for w in comparison['walkers']])),
            f0_mean_absolute_movement_range=[min(w['f0_latent_mean_absolute_movement'] for w in comparison['walkers']),max(w['f0_latent_mean_absolute_movement'] for w in comparison['walkers'])],
            reference_logL=comparison['reference_logL'],
            walker_mean_logL=[dict(bank=w['bank'],walker=w['walker'],mean_logL=w['logL']['mean']) for w in comparison['walkers']])
        report.update(status='completed_IID_reference',likelihood_evaluations=4096,
            conclusion='IID constrained calibration succeeds; current blocker is short constrained-kernel mixing, not inability to calibrate this shallow contour. Labelled regions differ; distinct physical modes are not established.',
            optional_level2_rejection='Not performed; the level-1 IID experiment resolves the requested distinction.')
        print('IID level2',json.dumps(row),flush=True)
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
        print('Report:',args.output/'report.json',report['status'],flush=True)


if __name__=='__main__':
    main()
