"""Fixed ell_1 scale benchmark only: no level building or level transitions."""
import argparse
import hashlib
import json
from pathlib import Path

from examples.lisa_dns_stage4 import audit
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.prior_bootstrap import array_sha256
from examples.lisa_dns_stage4.run_smoke import diagonal_scales, json_safe

COMMON_SCALE = float(np.pi / np.sqrt(3))


def frozen_scales(kind, historical=None, tuning=None):
    if kind in ('historical', 'historical_rms_normalized'):
        scales = np.array(historical, dtype=float, copy=True)
    elif kind == 'isotropic':
        scales = np.full(54, COMMON_SCALE)
    elif kind == 'iid_relative':
        scales = diagonal_scales(tuning)
        scales *= COMMON_SCALE / np.sqrt(np.mean(scales**2))
    else:
        raise ValueError('Unknown scale strategy')
    if scales.shape != (54,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Invalid scales')
    if kind == 'historical_rms_normalized':
        scales *= COMMON_SCALE / np.sqrt(np.mean(scales**2))
    scales.setflags(write=False)
    return scales


def separate_banks(tuning, evaluation, starts):
    arrays = [np.asarray(a) for a in (tuning, evaluation, starts)]
    if any(a.ndim != 2 or a.shape[1] != 54 or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Expected finite banks with 54 coordinates')
    rows = [{r.tobytes() for r in a} for a in arrays]
    if any(rows[i] & rows[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('Tuning, evaluation, and starts must not share rows')
    if len(rows[2]) != len(starts):
        raise ValueError('Walker starts must be distinct')
    return dict(tuning_sha256=array_sha256(arrays[0]), evaluation_sha256=array_sha256(arrays[1]),
                starts_sha256=array_sha256(arrays[2]), shared_rows=0)


def autocorrelation_information(values):
    """Biased FFT autocorrelation and initial-positive monotone paired sums.

    Descriptive short-chain estimate, not a convergence or stationarity test.
    Constants have ESS zero. ESS is capped at n; no antithetic bonus is used.
    """
    x=np.asarray(values,dtype=float)
    n=len(x)
    if n < 4 or not np.all(np.isfinite(x)):
        raise ValueError('Expected at least four finite observations')
    centered=x-x.mean()
    if np.dot(centered,centered) <= np.finfo(float).tiny:
        return dict(lag1=1.,tau=None,ess=0.,positive_pairs=0)
    transformed=np.fft.rfft(centered,n=2*n)
    cov=np.fft.irfft(transformed*np.conj(transformed),n=2*n)[:n]
    rho=cov/cov[0]
    pairs=[]
    for i in range(1,n-1,2):
        pair=float(rho[i]+rho[i+1])
        if pair <= 0: break
        pairs.append(min(pair,pairs[-1]) if pairs else pair)
    tau=max(1.,1+2*sum(pairs))
    return dict(lag1=float(rho[1]),tau=tau,ess=float(n/tau),positive_pairs=len(pairs))


def describe(x):
    return dict(mean=x.mean(axis=0).tolist(),variance=x.var(axis=0,ddof=1).tolist(),
                quantiles=np.quantile(x,[.05,.25,.5,.75,.95],axis=0).tolist())


def diagnostics(x, ll, starts, reference, reference_ll, physical):
    # Shapes: walker, retained time, coordinate; do not flatten for within-chain diagnostics.
    fcols=np.arange(0,54,6)
    ref_var=reference.var(0,ddof=1)
    within_var=x.var(1,ddof=1)
    ratios=within_var/ref_var
    between_sd=x.mean(1).std(0,ddof=1)/np.sqrt(ref_var)
    f0=physical(x[:,:,fcols]); ref_f0=physical(reference[:,fcols])
    f0_ref_range=np.quantile(ref_f0,.95,axis=0)-np.quantile(ref_f0,.05,axis=0)
    f0_spans=np.ptp(f0,axis=1)/f0_ref_range
    f0_ess=[[autocorrelation_information(x[w,:,k]) for k in fcols] for w in range(len(x))]
    ll_ess=[autocorrelation_information(a) for a in ll]
    ks=[float(ks_2samp(a,reference_ll).statistic) for a in ll]
    walkers=[]
    for w in range(len(x)):
        walkers.append(dict(walker=w,latent=describe(x[w]),variance_ratio_to_IID=ratios[w].tolist(),
            logL=describe(ll[w]),logL_KS=ks[w],physical_f0=describe(f0[w]),
            f0_latent_end_minus_start=(x[w,-1,fcols]-starts[w,fcols]).tolist(),
            f0_latent_span=np.ptp(x[w,:,fcols],axis=1).tolist(),
            f0_physical_span_over_IID_90percent_width=f0_spans[w].tolist(),
            parameter_displacement_norm=float(np.linalg.norm(x[w,-1]-starts[w])),
            retained_mean_step_norm=float(np.linalg.norm(np.diff(x[w],axis=0),axis=1).mean()),
            logL_autocorrelation=ll_ess[w],f0_autocorrelation=f0_ess[w]))
    return dict(reference_latent=describe(reference),reference_logL=describe(reference_ll),
        reference_physical_f0=describe(ref_f0),pooled_latent=describe(x.reshape(-1,54)),
        pooled_variance_ratio_to_IID=(x.reshape(-1,54).var(0,ddof=1)/ref_var).tolist(),
        pooled_logL_KS=float(ks_2samp(ll.ravel(),reference_ll).statistic),walkers=walkers,
        between_walker_coordinate_mean_sd_over_IID_sd=between_sd.tolist(),
        summary=dict(f0_within_variance_ratio_median=float(np.median(ratios[:,fcols])),
            f0_within_variance_ratio_range=[float(ratios[:,fcols].min()),float(ratios[:,fcols].max())],
            logL_KS_range=[min(ks),max(ks)],logL_KS_median=float(np.median(ks)),
            between_walker_logL_mean_sd=float(ll.mean(1).std(ddof=1)),
            between_walker_f0_mean_sd_over_IID_sd_median=float(np.median(between_sd[fcols])),
            f0_physical_span_over_IID_90percent_width_median=float(np.median(f0_spans)),
            f0_ESS_median=float(np.median([[a['ess'] for a in row] for row in f0_ess])),
            f0_lag1_median=float(np.median([[a['lag1'] for a in row] for row in f0_ess])),
            logL_ESS_median=float(np.median([a['ess'] for a in ll_ess])),
            logL_lag1_median=float(np.median([a['lag1'] for a in ll_ess]))),
        notice='KS distances are descriptive; no IID p-values for MCMC. Short-chain ESS/autocorrelation cannot guarantee global mixing. Within-chain variance is primary; pooling distinct starts can mask poor mixing.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('/tmp/lisa_dns_stage4_rejection'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/lisa_dns_stage4_kernel_benchmark'))
    parser.add_argument('--strategy',choices=['original_three','historical_rms_normalized'],default='original_three')
    parser.add_argument('--previous-benchmark',type=Path,default=Path('/tmp/lisa_dns_stage4_kernel_benchmark'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    previous = None
    if args.strategy == 'historical_rms_normalized':
        if args.output.resolve() == args.previous_benchmark.resolve():
            parser.error('Use a separate output directory to preserve the completed benchmark')
        previous=json.loads((args.previous_benchmark/'report.json').read_text())
    accepted=json.loads((args.reference/'report.json').read_text())
    ell1=accepted['ell1']
    report=dict(notice=audit.NOTICE,status='initializing',threshold=ell1,
        settings=dict(walkers=8,burn_in=128,retained=256,mixture=[.2,.4,.4],max_steps=10,max_shrinkage=100),
        direction_scale='scales*masked_normal/||masked_normal||; isotropic norm=c for every component. C has RMS scale c, hence equal global expected squared norm to B.',
        common_scale=COMMON_SCALE,strategies={},no_level_construction=True,no_level_transitions=True)
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    try:
        problem,cfg,config_path=audit.load_problem()
        assert cfg==accepted['config']
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        source=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        for p in [source,config_path]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==accepted['file_sha256'][str(p)]
        tuning_file=np.load(args.reference/'selection_bank.npz')
        eval_file=np.load(args.reference/'calibration_bank.npz')
        tuning=tuning_file['positions'][tuning_file['retained_indices']]
        reference=eval_file['positions'][eval_file['retained_indices']]
        reference_ll=eval_file['loglikelihood'][eval_file['retained_indices']]
        unused=np.setdiff1d(np.flatnonzero(tuning_file['accepted_mask']),tuning_file['retained_indices'])
        start_indices=unused[:8];starts=tuning_file['positions'][start_indices]
        assert starts.shape==(8,54)
        report['banks']=dict(**separate_banks(tuning,reference,starts),
            tuning='selection_bank retained 512',evaluation='independent calibration_bank retained 512',
            starts='first 8 excess accepted selection_bank samples; excluded from scale estimation and evaluation',
            start_indices=start_indices.tolist())
        assert np.all(reference_ll>ell1) and np.all(tuning_file['loglikelihood'][start_indices]>ell1)
        historical_path=audit.HIST/'seed44/posterior.npz'
        historical=diagonal_scales(np.load(historical_path)['samples_u'])
        names = (['historical_rms_normalized'] if previous is not None
                 else ['historical','isotropic','iid_relative'])
        # Freeze requested geometries before any benchmark chain starts.
        all_scales={name:frozen_scales(name,historical,tuning) for name in names}
        if previous is not None:
            assert previous['status'] == 'completed_fixed_contour_benchmark'
            assert previous['threshold'] == ell1 and previous['settings'] == report['settings']
            saved=np.load(args.previous_benchmark/'frozen_inputs.npz')
            np.testing.assert_array_equal(starts,saved['starts'])
            np.testing.assert_array_equal(historical,saved['historical'])
            for field in ['tuning_sha256','evaluation_sha256','starts_sha256']:
                assert report['banks'][field] == previous['banks'][field]
            for p in [source,config_path,historical_path,Path(proposals.__file__)]:
                assert hashlib.sha256(p.read_bytes()).hexdigest() == previous['file_sha256'][str(p)]
            report['comparison_provenance']=dict(previous_report=str(args.previous_benchmark/'report.json'),
                sha256=hashlib.sha256((args.previous_benchmark/'report.json').read_bytes()).hexdigest(),
                old_strategies_rerun=False,identical_inputs_and_settings_verified=True)
        files=[source,config_path,historical_path,Path(__file__),Path(proposals.__file__),
               args.reference/'selection_bank.npz',args.reference/'calibration_bank.npz']
        report.update(config=cfg,devices=str(jax.devices()),jax_version=jax.__version__,
            file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            scale_definitions=dict(historical='69 seed44 samples_u coordinate std(ddof=1), unchanged',
                historical_rms_normalized='original historical std * (pi/sqrt(3)) / RMS(original historical std)',
                isotropic='all 54 scales pi/sqrt(3)',iid_relative='512 tuning samples coordinate std(ddof=1), divided by RMS std and multiplied by pi/sqrt(3)'),
            scales={k:v.tolist() for k,v in all_scales.items()})
        np.savez(args.output/'frozen_inputs.npz',starts=starts,**all_scales)
        from jax_samplers.problems.lisa_gb_transdim_problem import u_to_f0_unordered
        physical=lambda x:np.asarray(u_to_f0_unordered(jnp.asarray(x),problem.f_min_cfg,problem.f_max_cfg))
        save()
        for name,scales in all_scales.items():
            strategy_index={'historical':0,'isotropic':1,'iid_relative':2,'historical_rms_normalized':3}[name]
            original_scales=scales.copy()
            step=jax.jit(proposals.build_parameter_step(prior,like,scales))
            positions=np.empty((8,384,54));logL=np.empty((8,384));lp=np.empty((8,384))
            nsteps=np.empty((8,384),int);nshrink=np.empty((8,384),int);accept=np.empty((8,384),bool)
            component=np.empty((8,384),int);labels=np.empty((8,384,2),int)
            row=dict(seeds=[4500+100*strategy_index+w for w in range(8)],
                     contour_failures=0,cache_failures=0,nonfinite_failures=0,completed_walkers=0)
            report['strategies'][name]=row
            for w in range(8):
                p=DNSParticleState(jnp.asarray(starts[w]),prior(starts[w]),like(starts[w]))
                key=jax.random.key(row['seeds'][w])
                for i in range(384):
                    key,subkey=jax.random.split(key)
                    p,info=step(subkey,p,jnp.asarray(ell1))
                    x=np.asarray(p.position);a=float(p.logdensity);b=float(p.loglikelihood)
                    if not np.all(np.isfinite(x)) or not np.isfinite(a) or not np.isfinite(b):
                        row['nonfinite_failures']+=1;raise ValueError('Nonfinite state')
                    if b<=ell1:
                        row['contour_failures']+=1;raise ValueError('Contour violation')
                    try:
                        np.testing.assert_allclose(a,prior(p.position),rtol=0,atol=1e-9)
                        np.testing.assert_allclose(b,like(p.position),rtol=0,atol=1e-7)
                    except AssertionError:
                        row['cache_failures']+=1;raise
                    positions[w,i]=x;lp[w,i]=a;logL[w,i]=b
                    nsteps[w,i]=int(info.slice.num_steps);nshrink[w,i]=int(info.slice.num_shrink)
                    accept[w,i]=bool(info.slice.is_accepted);component[w,i]=int(info.component);labels[w,i]=info.labels
                    if (i+1)%128==0: print(name,'walker',w,'steps',i+1,flush=True)
                row['completed_walkers']=w+1
                save()
            np.testing.assert_array_equal(scales,original_scales)
            np.savez_compressed(args.output/f'{name}_trace.npz',position=positions,logprior=lp,loglikelihood=logL,
                num_steps=nsteps,num_shrink=nshrink,slice_accepted=accept,component=component,labels=labels)
            result=diagnostics(positions[:,128:],logL[:,128:],starts,reference,reference_ll,physical)
            (args.output/f'{name}_diagnostics.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))
            row.update(result['summary'],parameter_steps=3072,slice_failures=int(np.sum(~accept)),
                likelihood_evaluation_proxy=int(np.sum(nsteps+nshrink)),direct_cache_likelihood_evaluations=3072,
                initialization_likelihood_evaluations=8,num_steps_mean=float(nsteps.mean()),
                num_shrink_mean=float(nshrink.mean()),num_steps_quantiles=np.quantile(nsteps,[.5,.9,1]).tolist(),
                num_shrink_quantiles=np.quantile(nshrink,[.5,.9,1]).tolist(),
                component_counts=np.bincount(component.ravel(),minlength=3).tolist(),scales_unchanged=True)
            # Sparse physical checkpoint decode only, without waveform reconstruction.
            checkpoint_positions=np.stack([starts,positions[:,127],positions[:,255],positions[:,-1]])
            decoded=problem.decode_batch(checkpoint_positions.reshape(-1,54),sort_by=None)
            np.savez_compressed(args.output/f'{name}_catalogue_checkpoints.npz',
                position=checkpoint_positions,iterations=np.array([-1,127,255,383]),
                **{k:np.asarray(v) for k,v in decoded.items()})
            print(name,'SUMMARY',json.dumps(row),flush=True);save()
        report['status']='completed_fixed_contour_benchmark'
        if previous is not None:
            report['comparison']={name:previous['strategies'][name] for name in ['historical','isotropic','iid_relative']}
            report['comparison']['historical_rms_normalized']=report['strategies']['historical_rms_normalized']
            report['comparison_RMS']={name:float(np.sqrt(np.mean(np.asarray(s)**2)))
                for name,s in {**previous['scales'],**report['scales']}.items()}
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save();print('Report:',args.output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
