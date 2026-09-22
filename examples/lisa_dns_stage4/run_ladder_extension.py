"""Bounded isotropic LISA ladder extension from fixed IID level 2; no production."""
import argparse
import hashlib
import json
from pathlib import Path

from examples.lisa_dns_stage4 import audit
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns import DNSParticleState
from blackjax.ns.dns_levels import build_next_level, tail_diagnostics
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales, autocorrelation_information
from examples.lisa_dns_stage4.run_smoke import json_safe

ELL2 = -113448.99632193986
LOGX2 = -1.9996430044384734
SETTINGS = dict(walkers=8, burn_in=128, retained=256, block_size=32, min_ess=20,
                target_compression=float(np.exp(-1)), mixture=[.2,.4,.4], max_steps=10, max_shrinkage=100)


def iid_starts(bank, threshold, seed):
    """Use excess IID draws, excluding the 512 used for previous calibration."""
    eligible = np.flatnonzero((bank['loglikelihood'] > threshold) & bank['accepted_mask'])
    eligible = np.setdiff1d(eligible, bank['retained_indices'])
    if len(eligible) < 8:
        raise ValueError('Insufficient unused IID survivors; no automatic sampling')
    indices = np.random.default_rng(seed).choice(eligible, 8, replace=False)
    return bank['positions'][indices], indices


def promote(position, logL, threshold, seed):
    """One survivor per original walker preserves separate walker genealogies."""
    rng = np.random.default_rng(seed)
    indices = []
    for values in logL:
        eligible = np.flatnonzero(values > threshold)
        if not len(eligible):
            raise ValueError('A walker has no survivor; stop without cloning other walkers')
        indices.append(int(rng.choice(eligible)))
    return position[np.arange(len(position)), indices], indices


def uncertainty(values, threshold):
    """Input walker/time; block means never cross walker boundaries."""
    indicator = (values > threshold).astype(float)
    means = indicator.reshape(len(values), -1, 32).mean(2)
    within = np.sqrt(np.sum(means.var(1, ddof=1) / means.shape[1]) / len(values)**2)
    return dict(within_walker_block_se=float(within),
                between_walker_se=float(indicator.mean(1).std(ddof=1)/np.sqrt(len(values))),
                block_means_se=float(means.std(ddof=1)/np.sqrt(means.size)),
                walker_fractions=indicator.mean(1).tolist(),
                walker_logL_means=values.mean(1).tolist(),
                walker_logL_mean_sd=float(values.mean(1).std(ddof=1)),
                walker_logL_first_half_means=values[:,:128].mean(1).tolist(),
                walker_logL_second_half_means=values[:,128:].mean(1).tolist(),
                walker_first_half_fractions=indicator[:,:128].mean(1).tolist(),
                walker_second_half_fractions=indicator[:,128:].mean(1).tolist(),
                walker_logL_ess=[autocorrelation_information(v)['ess'] for v in values])


def assess(selection, calibration, current, log_mass):
    proposal = build_next_level(selection.T, current, block_size=32, min_ess=20)
    threshold = proposal['threshold']
    check = tail_diagnostics(calibration.T, threshold, block_size=32)
    accepted = proposal['status']=='ok' and 0 < check['ratio'] < 1 and check['ess'] >= 20
    return dict(threshold=threshold, selection=proposal, calibration=check, accepted=accepted,
                estimated_log_mass=float(log_mass+np.log(check['ratio'])) if check['ratio']>0 else None,
                selection_uncertainty=uncertainty(selection, threshold),
                calibration_uncertainty=uncertainty(calibration, threshold),
                gap_to_seed44=float(-90964.02933083786-threshold))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('/tmp/lisa_dns_stage4_rejection'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/lisa_dns_stage4_ladder_extension'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='initializing',settings=SETTINGS,notice=audit.NOTICE,rows=[],
                failures=dict(contour=0,cache=0,nonfinite=0,parameter_contract=0,slice=0),
                cost=dict(selection_likelihood_evaluations=0,calibration_likelihood_evaluations=0,
                          direct_cache_check_evaluations=0,initialization_likelihood_evaluations=0),
                seeds=dict(initial=[5100,5101],level3=[list(range(5200,5208)),list(range(5300,5308))],
                           promotion=[5400,5401],level4=[list(range(5500,5508)),list(range(5600,5608))]),
                frozen_thresholds=[float('-inf'),-113725.65889538352,ELL2],
                frozen_log_mass=[0.,-1.0343179379627125,LOGX2])
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
        np.savez(args.output/'levels.npz',thresholds=report['frozen_thresholds'],log_mass=report['frozen_log_mass'])
    def validate(p, threshold):
        x=np.asarray(p.position); lp=np.asarray(p.logdensity); ll=np.asarray(p.loglikelihood)
        finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(lp)) and np.all(np.isfinite(ll)))
        contour=bool(np.all(ll>threshold))
        contract=x.shape==(54,) and lp.shape==() and ll.shape==() and finite and contour
        report['failures']['nonfinite']+=int(not finite)
        report['failures']['contour']+=int(not contour)
        report['failures']['parameter_contract']+=int(not contract)
        if not contract: raise ValueError('Constrained parameter contract failed')
        direct_lp=prior(p.position);direct_ll=like(p.position)
        report['cost']['direct_cache_check_evaluations']+=1
        if not (np.isclose(lp,direct_lp,rtol=0,atol=1e-9) and np.isclose(ll,direct_ll,rtol=0,atol=1e-7)):
            report['failures']['cache']+=1
            raise ValueError('Direct cache consistency failed')
    try:
        accepted=json.loads((args.reference/'report.json').read_text())
        assert accepted['IID_level2']['accepted']
        assert accepted['IID_level2']['selection']['threshold']==ELL2
        assert accepted['IID_level2']['log_X2']==LOGX2
        assert accepted['ell1']==report['frozen_thresholds'][1]
        assert np.log(accepted['Xhat1'])==report['frozen_log_mass'][1]
        problem,cfg,config_path=audit.load_problem()
        assert cfg==accepted['config']
        model=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        for path in [model,config_path]:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted['file_sha256'][str(path)]
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        scales=frozen_scales('isotropic')
        step=jax.jit(proposals.build_parameter_step(prior,like,scales))
        paths=[args.reference/f'{name}_bank.npz' for name in ['selection','calibration']]
        banks=[np.load(path) for path in paths]
        starts=[];start_indices=[];references=[]
        for b,bank in enumerate(banks):
            x,indices=iid_starts(bank,ELL2,5100+b)
            starts.append(x);start_indices.append(indices.tolist())
            retained=bank['retained_indices']
            references.append(bank['positions'][retained[bank['loglikelihood'][retained]>ELL2]])
        assert not ({r.tobytes() for r in banks[0]['positions']} & {r.tobytes() for r in banks[1]['positions']})
        report.update(config=cfg,devices=str(jax.devices()),initial_indices=start_indices,
            initialization='Eight distinct unused IID survivors from each respective bank; no historical positions',
            scale=float(scales[0]),file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [*paths,args.reference/'report.json',model,config_path,Path(__file__),Path(proposals.__file__)]},
            cost_notice='num_steps + num_shrink counts slice_fn likelihood calls in the unchanged scalar kernel; cache checks and initialization separate. No previous production costs included.')
        np.savez(args.output/'frozen_inputs.npz',starts=starts,scales=scales)
        save()
        for level in [3,4]:
            current=report['frozen_thresholds'][-1]
            histories=[];bank_reports=[]
            for b,name in enumerate(['selection','calibration']):
                records=[]
                seeds=report['seeds'][f'level{level}'][b]
                for w in range(8):
                    p=DNSParticleState(jnp.asarray(starts[b][w]),prior(starts[b][w]),like(starts[b][w]))
                    report['cost']['initialization_likelihood_evaluations']+=1
                    validate(p,current)
                    key=jax.random.key(seeds[w])
                    for i in range(384):
                        key,subkey=jax.random.split(key)
                        p,info=step(subkey,p,jnp.asarray(current))
                        report['cost'][f'{name}_likelihood_evaluations']+=int(info.slice.num_steps+info.slice.num_shrink)
                        validate(p,current)
                        if not bool(info.slice.is_accepted):
                            report['failures']['slice']+=1
                            raise ValueError('Slice failure')
                        records.append(dict(walker=w,iteration=i,position=np.asarray(p.position),
                            logprior=float(p.logdensity),loglikelihood=float(p.loglikelihood),
                            num_steps=int(info.slice.num_steps),num_shrink=int(info.slice.num_shrink),
                            component=int(info.component),labels=np.asarray(info.labels),slice_accepted=True))
                    print('level',level,name,'walker',w,'completed',flush=True)
                    np.savez_compressed(args.output/f'level{level}_{name}_trace.npz',
                        **{k:np.asarray([r[k] for r in records]) for k in records[0]})
                    save()
                x=np.array([r['position'] for r in records]).reshape(8,384,54)[:,128:]
                ll=np.array([r['loglikelihood'] for r in records]).reshape(8,384)[:,128:]
                histories.append((x,ll))
                f0=x[:,:,::6];reference=references[1-b][:,::6]
                ratios=f0.var(1,ddof=1)/reference.var(0,ddof=1)
                bank_reports.append(dict(f0_variance_reference='Opposite-bank fixed IID level2 survivors; level4 comparison is to lower contour, not a level3 coverage estimate',
                    f0_within_variance_ratios=ratios.tolist(),f0_within_variance_ratio_median=float(np.median(ratios)),
                    f0_within_variances=f0.var(1,ddof=1).tolist(),
                    f0_between_walker_mean_sd=f0.mean(1).std(0,ddof=1).tolist(),
                    f0_ess_median=float(np.median([[autocorrelation_information(v)['ess'] for v in walker.T] for walker in f0])),
                    likelihood_evaluations=int(sum(r['num_steps']+r['num_shrink'] for r in records))))
            row=assess(histories[0][1],histories[1][1],current,report['frozen_log_mass'][-1])
            row.update(level=level,bank_diagnostics=dict(zip(['selection','calibration'],bank_reports)))
            report['rows'].append(row)
            if not row['accepted']:
                report['status']='stopped_failed_calibration_or_selection';save();break
            report['frozen_thresholds'].append(row['threshold'])
            report['frozen_log_mass'].append(row['estimated_log_mass'])
            report['status']=f'frozen_level{level}';save()
            print('FROZEN',level,row['threshold'],row['calibration']['ess'],flush=True)
            if level==3:
                promoted=[promote(x,ll,row['threshold'],5400+b) for b,(x,ll) in enumerate(histories)]
                starts=[p[0] for p in promoted]
                report['promotion_indices']=[p[1] for p in promoted]
                np.savez(args.output/'level4_starts.npz',starts=starts)
        np.testing.assert_array_equal(scales,np.full(54,np.pi/np.sqrt(3)))
    except Exception as exc:
        report.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save();print('Report:',args.output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
