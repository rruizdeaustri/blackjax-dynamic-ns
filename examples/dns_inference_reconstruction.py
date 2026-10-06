"""Stage-7B analytic truths, automatic ladders, and independent inference streams.

This module never loads a LISA problem. Run with CPU/x64 and a saved design.
"""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
from jax.scipy.special import logsumexp as jlogsumexp
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm

from blackjax.ns import dns_automatic as a, dns_reconstruction as r
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4.automatic_ladder_integration import make_frozen_kernel

QUANTILES = np.array([.05, .16, .5, .84, .95])


class AnalyticModel:
    def __init__(self, name, parameters):
        self.name, self.parameters = name, parameters
        self.sigma = parameters['sigma_prior']
        if name == 'gaussian':
            self.coefficients = np.array([1.])
            self.centers = np.array([parameters['y']])
            self.widths = np.array([parameters['sigma_likelihood']])
        else:
            self.coefficients = np.array(parameters['component_weights'])
            self.centers = np.array(parameters['centers'])
            self.widths = np.array(parameters['widths'])
        # Closed-form normal-product identities, independent of reconstruction.
        joint = np.log(self.coefficients)+norm.logpdf(self.centers, scale=np.sqrt(self.sigma**2+self.widths**2))
        self.logZ = float(logsumexp(joint))
        self.probabilities = np.exp(joint-self.logZ)
        self.means = self.centers*self.sigma**2/(self.sigma**2+self.widths**2)
        self.variances = self.sigma**2*self.widths**2/(self.sigma**2+self.widths**2)

    def cdf(self, theta):
        return np.sum(self.probabilities*norm.cdf(np.asarray(theta)[..., None],
            loc=self.means, scale=np.sqrt(self.variances)), axis=-1)

    def reference(self):
        mean = float(self.probabilities@self.means)
        variance = float(self.probabilities@(self.variances+self.means**2)-mean**2)
        lower = float(self.means.min()-12*np.sqrt(self.variances.max()))
        upper = float(self.means.max()+12*np.sqrt(self.variances.max()))
        quantiles = ([float(norm.ppf(p, self.means[0], np.sqrt(self.variances[0]))) for p in QUANTILES]
                     if self.name == 'gaussian' else [brentq(lambda x: self.cdf(x)-p, lower, upper, xtol=1e-13) for p in QUANTILES])
        return dict(logZ=self.logZ, Z=float(np.exp(self.logZ)), mean=mean, variance=variance,
            standard_deviation=float(np.sqrt(variance)), quantiles=quantiles,
            quantile_probabilities=QUANTILES.tolist(), component_probabilities=self.probabilities.tolist(),
            component_means=self.means.tolist(), component_variances=self.variances.tolist(),
            region_probabilities=[float(self.cdf(0.)), float(1-self.cdf(0.))])

    def log_likelihood(self, theta):
        return logsumexp(np.log(self.coefficients)+norm.logpdf(np.asarray(theta)[..., None],
            loc=self.centers, scale=self.widths), axis=-1)

    def responsibilities(self, theta):
        components = np.log(self.coefficients)+norm.logpdf(np.asarray(theta)[..., None], loc=self.centers, scale=self.widths)
        return np.exp(components-logsumexp(components, axis=-1, keepdims=True))

    def contour_mass(self, threshold):
        if np.isneginf(threshold):
            return 1.
        if self.name == 'gaussian':
            radius = self.widths[0]*np.sqrt(-2*(threshold+np.log(self.widths[0]*np.sqrt(2*np.pi))))
            return float(norm.cdf((self.centers[0]+radius)/self.sigma)-norm.cdf((self.centers[0]-radius)/self.sigma))
        # Additional numerical contour-root calculation, not the analytic evidence reference.
        grid = np.linspace(-20*self.sigma, 20*self.sigma, 20001)
        values = self.log_likelihood(grid)-threshold
        brackets = np.flatnonzero(values[:-1]*values[1:] < 0)
        roots = [brentq(lambda x: float(self.log_likelihood(x)-threshold), grid[i], grid[i+1], xtol=1e-13) for i in brackets]
        bounds = [-np.inf]+roots+[np.inf]
        mass = 0.
        for low, high in zip(bounds[:-1], bounds[1:]):
            point = ((low+high)/2 if np.isfinite(low) and np.isfinite(high)
                     else high-10*self.sigma if np.isneginf(low) else low+10*self.sigma)
            if self.log_likelihood(point)>threshold:
                mass += norm.cdf(high/self.sigma)-norm.cdf(low/self.sigma)
        return float(mass)

    def kernel(self):
        sigma = self.sigma
        log_coeff = jnp.asarray(np.log(self.coefficients)); centers = jnp.asarray(self.centers)
        widths = jnp.asarray(self.widths)
        def prior(u):
            return -.5*jnp.sum(u*u)-27*jnp.log(2*jnp.pi)
        def likelihood(u):
            theta = sigma*u[0]
            return jlogsumexp(log_coeff-jnp.log(widths*jnp.sqrt(2*jnp.pi))-.5*((theta-centers)/widths)**2)
        return make_frozen_kernel(prior, likelihood, (.00183, .00185),
            model_id='analytic-'+self.name+'-'+a.digest(a.encoded(self.parameters)))


def initial_state(model, kernel, seed, config):
    banks = {}
    for name, child in zip(a.BANKS, np.random.SeedSequence(seed).spawn(2)):
        x = np.random.Generator(np.random.PCG64(child)).normal(size=(16, 8, 54))
        lp, ll = kernel.evaluate(x)
        banks[name] = DNSParticleState(x, lp, ll)
    return a.LadderState((-np.inf,), (0.,), **banks, seed=seed, kernel_metadata=a.encoded(kernel.metadata), config=config)


def inference_records(state, kernel, checkpoint_folders, seed, config, output, model_name):
    """Fresh inference at fixed levels; starts come only from saved calibration histories."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    fields = {k: [] for k in ['position','logprior','loglikelihood','origin_level','walker','draw','event_id']}
    levels = []; source_hashes = {}
    for j, threshold in enumerate(state.thresholds):
        source_path = checkpoint_folders[min(j, len(checkpoint_folders)-1)]
        source = a.load_checkpoint(source_path, kernel)
        streams = a.streams(seed, j, 'calibration', 8)
        starts, promotion = a.promote(source.calibration, threshold, streams['promotion_seed'], state.contract)
        a.write_json(output/f'level_{j}_preflight.json', dict(level=j, threshold=threshold,
            log_mass=state.log_masses[j], streams=streams, promotion=promotion,
            source_history_hash=a.particle_hash(source.calibration), sampling_target=r.TARGET,
            config=a.asdict(config), contract=a.asdict(state.contract)))
        history, diagnostic = kernel.run(starts, threshold, streams, config, state.contract)
        state.contract.validate(history, threshold, kernel.evaluate(history.position))
        n = history.loglikelihood.size
        for name, value in zip(['position','logprior','loglikelihood'], history):
            fields[name].append(value.reshape((n, 54) if name == 'position' else (n,)))
        draw = np.repeat(np.arange(config.retained), 8); walker = np.tile(np.arange(8), config.retained)
        fields['origin_level'].append(np.full(n, j, dtype=int)); fields['walker'].append(walker); fields['draw'].append(draw)
        fields['event_id'].append(np.array([f'{model_name}/{seed}/{j}/{t}/{w}' for t, w in zip(draw, walker)]))
        levels.append(dict(level=j, threshold=str(threshold) if not np.isfinite(threshold) else threshold,
            stream_id=f'{model_name}/{seed}/{j}', streams=streams, diagnostics=diagnostic,
            originating_threshold=threshold if np.isfinite(threshold) else '-inf', source_bank='fresh-inference',
            source_history_hash=a.particle_hash(source.calibration)))
        source_hashes[str(j)] = a.digest((source_path/'banks.npz').read_bytes())
        print(model_name, 'inference level', j, 'completed', flush=True)
    metadata = dict(sampling_target=r.TARGET, mass_status=['prior']+['calibrated']*(len(levels)-1),
        levels=levels, kernel=kernel.metadata, contract=a.asdict(state.contract), environment=state.contract.environment(),
        seed=seed, checkpoint_history_hashes=source_hashes, burn_in=config.burn_in, retained=config.retained)
    records = r.ReconstructionRecords(np.array(state.thresholds), np.array(state.log_masses),
        **{k: np.concatenate(v) for k, v in fields.items()}, metadata=metadata)
    records.validate()
    r.save_records(output/'records', records)
    return records


def summarize(model, records, *, masses=None):
    estimate = r.reconstruct_evidence(records, log_masses=masses)
    theta = model.sigma*records.position[:, 0]; weights = estimate['weights']
    mean = float(r.posterior_expectation(theta, weights))
    variance = float(r.posterior_expectation((theta-mean)**2, weights))
    quantiles = r.posterior_quantiles(theta, weights, QUANTILES)
    components = r.posterior_expectation(model.responsibilities(theta), weights)
    region = float(r.posterior_expectation(theta<0, weights))
    truth = model.reference()
    summary = {k:v for k,v in estimate.items() if k not in ['weights','log_importance']}
    summary.update(mean=mean, variance=variance, standard_deviation=float(np.sqrt(variance)), quantiles=quantiles,
        component_probabilities=components, region_probabilities=[region, 1-region],
        cdf_discrepancy=r.weighted_cdf_discrepancy(theta, weights, model.cdf),
        errors=dict(logZ_absolute=abs(estimate['logZ']-truth['logZ']), Z_relative=abs(np.expm1(estimate['logZ']-truth['logZ'])),
            mean_absolute=abs(mean-truth['mean']), variance_absolute=abs(variance-truth['variance']),
            standard_deviation_absolute=abs(np.sqrt(variance)-truth['standard_deviation']),
            quantile_absolute=abs(quantiles-truth['quantiles']), component_probability_absolute=abs(components-truth['component_probabilities']),
            region_probability_absolute=abs(np.array([region, 1-region])-truth['region_probabilities'])),
        weights_finite=bool(np.isfinite(weights).all()), weights_nonnegative=bool(np.all(weights>=0)), weight_sum=float(weights.sum()))
    return a.json_value(summary), estimate


def accuracy_checks(name, summary, tolerance):
    e = summary['errors']
    checks = dict(logZ=e['logZ_absolute']<=tolerance['logZ_absolute'], relative_Z=e['Z_relative']<=tolerance['Z_relative'],
        mean=e['mean_absolute']<=tolerance[name+'_mean_absolute'], variance=e['variance_absolute']<=tolerance[name+'_variance_absolute'],
        quantiles=max(e['quantile_absolute'])<=tolerance[name+'_quantile_absolute'], cdf=summary['cdf_discrepancy']<=tolerance[name+'_cdf'],
        weights=summary['weights_finite'] and summary['weights_nonnegative'] and abs(summary['weight_sum']-1)<=tolerance['weight_sum_absolute'])
    if name == 'mixture':
        checks.update(components=max(e['component_probability_absolute'])<=tolerance['mixture_component_probability_absolute'],
                      regions=max(e['region_probability_absolute'])<=tolerance['mixture_region_probability_absolute'])
    return checks


def run_model(name, design, root):
    output = root/name; output.mkdir(exist_ok=False)
    model = AnalyticModel(name, design['models'][name]); kernel = model.kernel()
    a.write_json(output/'exact_truth.json', model.reference())
    b = design['builder']
    config = a.LadderConfig(burn_in=b['burn_in'], retained=b['retained'], block_size=b['block_size'])
    initial = initial_state(model, kernel, design['seeds'][name], config)
    full = a.run_ladder(initial, kernel, output/'continuous', max_new_levels=b['new_levels'])
    if full.status != 'ready' or len(full.thresholds) != b['new_levels']+1:
        result = dict(decision='FAIL', reason='Automatic construction gate failed; no scientific retuning', records=[json.loads(v) for v in full.records])
        a.write_json(output/'results.json', result)
        return result
    print(name, 'ladder completed', list(full.thresholds), flush=True)
    checkpoints = [output/'continuous'/f'attempt_{j:06d}'/'checkpoint' for j in range(full.iteration)]
    # Restart after two accepted levels, rebuilding only the remaining fixed attempt.
    resumed_start = a.load_checkpoint(checkpoints[1], model.kernel())
    restarted = a.run_ladder(resumed_start, model.kernel(), output/'restarted', max_new_levels=1)
    assert restarted.thresholds == full.thresholds and restarted.log_masses == full.log_masses
    assert restarted.records == full.records
    for bank in a.BANKS:
        assert a.particle_hash(getattr(restarted, bank)) == a.particle_hash(getattr(full, bank))
    f = design['reconstruction']
    inference_config = a.LadderConfig(burn_in=f['burn_in'], retained=f['retained'], block_size=f['block_size'])
    seed = design['seeds']['reconstruction_'+name]
    records = inference_records(full, kernel, checkpoints, seed, inference_config, output/'inference', name)
    restart_checkpoints = checkpoints[:2]+[output/'restarted/attempt_000002/checkpoint']
    repeated = inference_records(restarted, model.kernel(), restart_checkpoints, seed, inference_config, output/'inference_restarted', name)
    for field in ['thresholds','log_masses','position','logprior','loglikelihood','origin_level','walker','draw','event_id']:
        assert getattr(records, field).tobytes() == getattr(repeated, field).tobytes(), field
    assert records.metadata == repeated.metadata
    primary, estimate = summarize(model, records); repeat_summary, repeat_estimate = summarize(model, repeated)
    assert a.encoded(primary) == a.encoded(repeat_summary)
    assert estimate['weights'].tobytes() == repeat_estimate['weights'].tobytes()
    np.savez_compressed(output/'posterior_weights.npz', weights=estimate['weights'], log_importance=estimate['log_importance'])
    prefix = [summarize(model, records.through(j))[0] for j in range(len(records.thresholds))]
    exact_masses = np.log([model.contour_mass(t) for t in records.thresholds])
    exact_mass_summary, _ = summarize(model, records, masses=exact_masses)
    calibrated = [json.loads(v)['calibration'] for v in full.records]
    ratios = np.array([v['ratio'] for v in calibrated]); se = np.array([v['standard_error'] for v in calibrated])
    rng = np.random.default_rng(design['seeds']['mass_sensitivity'])
    perturb = expit(logit(ratios)+rng.normal(size=(design['diagnostics']['mass_sensitivity_draws'],len(ratios)))*se/(ratios*(1-ratios)))
    sensitivity = []
    theta = model.sigma*records.position[:, 0]
    responsibility = model.responsibilities(theta)
    for row in perturb:
        e = r.reconstruct_evidence(records, log_masses=np.r_[0.,np.cumsum(np.log(row))])
        sensitivity.append([e['logZ'], float(r.posterior_expectation(theta,e['weights'])), *r.posterior_expectation(responsibility,e['weights'])])
    log_r = estimate['log_importance']; scaled_r = np.exp(log_r-log_r.max())
    correlations = dict(loglikelihood=r.correlation_diagnostics(records,records.loglikelihood,f['block_size']),
        evidence_integrand=r.correlation_diagnostics(records,scaled_r,f['block_size']),
        posterior_mean_integrand=r.correlation_diagnostics(records,scaled_r*(theta-primary['mean']),f['block_size']),
        mode_integrand=r.correlation_diagnostics(records,scaled_r*((theta<0)-primary['region_probabilities'][0]),f['block_size']))
    checks = accuracy_checks(name, primary, design['tolerances'])
    result = dict(decision='PASS' if all(checks.values()) else 'FAIL', model=name, parameters=model.parameters,
        exact=model.reference(), primary=primary, accuracy_checks=checks, progressive_levels=prefix,
        shell_diagnostic=r.shell_diagnostic(records), recorded_log_masses=full.log_masses,
        thresholds=full.thresholds, exact_contour_log_masses=exact_masses,
        exact_mass_diagnostic=exact_mass_summary, calibration=calibrated,
        mass_sensitivity=dict(method=design['diagnostics']['mass_sensitivity'],seed=design['seeds']['mass_sensitivity'],
            columns=['logZ','posterior_mean']+[f'component_{i}' for i in range(len(model.coefficients))],
            q05_q50_q95=np.quantile(sensitivity,[.05,.5,.95],axis=0),draws=len(sensitivity)),
        correlation=correlations, restart=dict(exact_thresholds=True,exact_masses=True,exact_records=True,
            identical_logZ=True,identical_weights=True,identical_summaries=True),
        cache_check=r.validate_caches(records,kernel.evaluate,full.contract),
        checkpoint_complete=True, accounted_prior_mass=float(np.sum(-np.diff(np.exp(full.log_masses)))+np.exp(full.log_masses[-1])),
        environment=full.contract.environment(), numerical_contract=a.asdict(full.contract), construction_records=[json.loads(v) for v in full.records])
    a.write_json(output/'results.json', result)
    print(name, result['decision'], 'logZ', primary['logZ'], 'checks', checks, flush=True)
    return a.json_value(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/dns-stage7b-validation'))
    args = parser.parse_args()
    design = json.loads((args.output/'design.json').read_text())
    for path, h in design['frozen_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h, path
    results = {name: run_model(name,design,args.output) for name in ('gaussian','mixture')}
    for path, h in design['frozen_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h, path
    result = dict(stage='7B',decision='PASS' if all(v['decision']=='PASS' for v in results.values()) else 'FAIL',
        design=design,models=results,frozen_hashes_unchanged=True,LISA_model_calls=0,production_stopping_implemented=False)
    a.write_json(args.output/'results.json',result)
    print('FINAL',result['decision'],flush=True)


if __name__ == '__main__':
    main()
