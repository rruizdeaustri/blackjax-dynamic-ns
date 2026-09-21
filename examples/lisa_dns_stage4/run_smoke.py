"""Seed-44 engineering smoke run; no evidence or posterior reconstruction.

Run with the blackjax-ns Python and PYTHONPATH=. (CUDA defaults come from audit).
Level zero uses independent direct-prior banks. Higher contours use separate
walker populations. Seed44 is used only to initialize frozen production.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from examples.lisa_dns_stage4 import audit
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns import dns, dns_levels
from examples.lisa_dns_stage4 import proposals, prior_bootstrap as bootstrap


def diagonal_scales(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 54 or len(samples) < 2:
        raise ValueError('Require at least two 54-dimensional samples')
    scales = np.std(samples, axis=0, ddof=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Historical diagonal scales must be positive and finite')
    return scales


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def summarize(records):
    components = Counter(str(r['component']) for r in records)
    singles = Counter(str(r['labels'][0]) for r in records if r['component'] == 1)
    pairs = Counter(','.join(map(str, sorted(r['labels']))) for r in records if r['component'] == 2)
    return dict(steps=len(records), component_counts=dict(components),
                single_label_counts=dict(singles), unordered_pair_counts=dict(pairs),
                slice_acceptances=sum(r['slice_accepted'] for r in records),
                slice_failures=sum(not r['slice_accepted'] for r in records),
                likelihood_evaluation_proxy=sum(r['cost'] for r in records),
                min_logL=min(r['loglikelihood'] for r in records),
                max_logL=max(r['loglikelihood'] for r in records))


def level_summary(indices, nlevels):
    indices = np.asarray(indices, dtype=int)
    transitions = np.zeros((nlevels, nlevels), dtype=int)
    for a, b in zip(indices[:-1], indices[1:]):
        transitions[a, b] += 1
    # Count complete top -> bottom -> top excursions.
    armed = bottom = False
    trips = 0
    for j in indices:
        if j == nlevels - 1:
            if armed and bottom:
                trips += 1
            armed, bottom = True, False
        elif j == 0 and armed:
            bottom = True
    return dict(occupancy=np.bincount(indices[1:], minlength=nlevels).tolist(),
                transitions=transitions.tolist(), round_trips=trips,
                min_assigned_level=int(indices.min()), max_assigned_level=int(indices.max()))


def audit_prior_bank(problem, positions, seed, logprior_values):
    """Verify the inspected production sampler, rather than infer it from logprior."""
    from jax_samplers.problems.lisa_gb_transdim_problem import box_to_u, u_to_box
    names = ['f0', 'fdot', 'iota', 'psi', 'lam', 'beta']
    assert positions.shape == (512, 54) and positions.dtype == np.float64
    assert np.all(np.isfinite(logprior_values))
    keys = jax.random.split(jax.random.key(seed), 6)
    physical = [lo + (hi-lo)*jax.random.uniform(k, (len(positions), 9), dtype=jnp.float64)
                for k, (lo, hi) in zip(keys, [problem.prior_box[name] for name in names])]
    replay = np.asarray(jnp.stack([box_to_u(x, *problem.prior_box[name])
                                  for name, x in zip(names, physical)], axis=-1)).reshape(-1, 54)
    np.testing.assert_array_equal(positions, replay)
    unit = np.asarray(jax.nn.sigmoid(jnp.asarray(positions)))
    ordered = np.sort(unit, axis=0)
    n, d = unit.shape
    ks = np.maximum(np.max(np.arange(1, n+1)[:, None]/n-ordered, axis=0),
                    np.max(ordered-np.arange(n)[:, None]/n, axis=0))
    # DKW union bound across all 54 coordinates; alpha=.005 per bank.
    bound = float(np.sqrt(np.log(2*d/.005)/(2*n)))
    assert np.max(ks) <= bound, 'Coordinate marginal audit failed simultaneous DKW check'
    roundtrip_errors = []
    for column, name in enumerate(names):
        lo, hi = problem.prior_box[name]
        decoded = np.asarray(u_to_box(jnp.asarray(positions.reshape(-1,9,6)[:,:,column]), lo, hi))
        err = float(np.max(np.abs(decoded-np.asarray(physical[column])))/(hi-lo))
        assert err < 1e-10
        roundtrip_errors.append(err)
    return dict(shape=list(positions.shape), dtype=str(positions.dtype),
                exact_sampler_replay=True, logprior_range=[float(logprior_values.min()), float(logprior_values.max())],
                names=names, physical_bounds=problem.prior_box, uniform_coordinate_mean=unit.mean(0).tolist(),
                uniform_coordinate_variance=unit.var(0).tolist(), uniform_coordinate_KS=ks.tolist(),
                simultaneous_DKW_bound=bound, max_KS=float(ks.max()),
                physical_roundtrip_relative_errors=roundtrip_errors,
                clipped_coordinates=int(np.sum((unit <= 1.00001e-9) | (unit >= 1-1.00001e-9))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-families', type=Path, default=Path('/tmp/lisa_dns_stage4_audit/initial_families.npz'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/lisa_dns_stage4_prior_smoke'))
    parser.add_argument('--seed', type=int, default=4404)
    parser.add_argument('--sweeps', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.sweeps <= 512:
        parser.error('This engineering runner permits 1 through 512 production sweeps')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(notice=audit.NOTICE, seed=args.seed, status='initializing',
                  mixture_probabilities=[.2, .4, .4],
                  previous_attempt=dict(method='seed44-initialized level-0 MCMC',
                      threshold=-98673.78800261783, calibration_ESS=13.770400887339566, accepted=False),
                  production=dict(status='not_run'),
                  concern='Level-0 calibration is IID conditional on its independent selection threshold. Higher-level block ESS cannot certify global mixing. Historical scales are local to seed44.',
                  cost_notice='Slice likelihood proxy is sum(num_steps+num_shrink); direct bank and cache-check evaluations are reported separately.')
    records = []
    thresholds, masses = [-np.inf], [0.]
    audit_evals = 0
    def save():
        report['levels'] = dict(thresholds=thresholds, log_masses=masses, log_weights=[0.]*len(masses))
        report['direct_cache_check_likelihood_evaluations'] = audit_evals
        if records:
            report['all_parameter_steps'] = summarize(records)
            np.savez_compressed(args.output/'trace.npz', **{k: np.asarray([r[k] for r in records]) for k in records[0]})
        (args.output/'report.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
        np.savez(args.output/'levels.npz', thresholds=thresholds, log_mass=masses, log_weight=np.zeros(len(masses)))
    try:
        problem, cfg, config_path = audit.load_problem()
        assert cfg['dtype'] == 'float64'
        assert not cfg.get('priors', {}).get('gaussian') and not cfg.get('prior', {}).get('gaussian_priors')
        prior, like = map(jax.jit, audit.scalar_functions(problem))
        source = audit.HIST/'seed44/posterior.npz'
        scales = diagonal_scales(np.load(source)['samples_u'])
        np.save(args.output/'scales.npy', scales)
        module_path = audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        nss_path = audit.ROOT/'BayesLISAx/src/jax_samplers/samplers/blackjax_ns.py'
        hashed_paths = [config_path, source, module_path, nss_path, Path(__file__),
                        Path(bootstrap.__file__), Path(proposals.__file__)]
        report.update(config=cfg, config_path=str(config_path), devices=str(jax.devices()),
                      jax_version=jax.__version__, scales=scales.tolist(),
                      file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed_paths},
                      scale_source=dict(path=str(source), key='samples_u', method='coordinate std, ddof=1; unchanged from first smoke'),
                      prior_sampler=dict(function='LISAGBTransdimProblem.sample_prior -> make.<locals>.sample_prior',
                        nss_call='BlackJAXNestedSampler._setup_common: problem.sample_prior(sub, n_live)',
                        transform='six independent physical uniform (n,9) blocks -> box_to_u -> stack axis=-1 -> reshape(n,54)',
                        clipping='box_to_u clips normalized coordinates to [1e-9, 1-1e-9]; unchanged existing implementation'),
                      settings=dict(prior_draws_per_bank=512, likelihood_chunk=32, waveform_parallelism=1,
                        walkers_per_bank=8, burn_in_per_walker=128, draws_per_walker=256, block_size=32, min_ess=20.,
                        iid_min_count_each_outcome=20, target_compression=float(np.exp(-1)), max_levels=3, production_sweeps=args.sweeps))
        print('Frozen scales:', scales.tolist(), flush=True)
        seeds = [args.seed+10, args.seed+11]
        banks, metadata = bootstrap.draw_prior_banks(problem.sample_prior, *seeds, 512)
        report['prior_banks'] = metadata
        # Only scalar evaluations occur inside lax.map, in chunks of 32.
        evaluate = jax.jit(lambda xs: jax.lax.map(lambda x: (prior(x), like(x)), xs))
        bank_priors, bank_likes = [], []
        for b, positions in enumerate(banks):
            lps, lls = [], []
            for start in range(0, len(positions), 32):
                lp, ll = evaluate(jnp.asarray(positions[start:start+32]))
                lps.extend(np.asarray(lp)); lls.extend(np.asarray(ll))
                assert np.all(np.isfinite(lps)) and np.all(np.isfinite(lls))
                print('prior bank', b, 'evaluated', start+32, flush=True)
            bank_priors.append(np.asarray(lps)); bank_likes.append(np.asarray(lls))
            report.setdefault('prior_audit', []).append(audit_prior_bank(problem, positions, seeds[b], bank_priors[-1]))
        report['prior_likelihood_evaluations'] = 1024
        np.savez_compressed(args.output/'prior_banks.npz', selection_positions=banks[0], calibration_positions=banks[1],
                            selection_logprior=bank_priors[0], calibration_logprior=bank_priors[1],
                            selection_logL=bank_likes[0], calibration_logL=bank_likes[1])
        selection = bootstrap.select_iid_threshold(bank_likes[0])
        row = dict(level=1, method='direct independent prior bootstrap', selection=selection, accepted=False)
        report['ladder'] = [row]
        if selection['status'] != 'ok':
            report['status'] = 'stopped_IID_selection'
            return
        calibration = bootstrap.calibrate_iid(bank_likes[1], selection['threshold'])
        row['calibration'] = calibration
        if not calibration['accepted']:
            report['status'] = 'stopped_IID_calibration'
            return
        thresholds.append(selection['threshold']); masses.append(calibration['log_ratio'])
        row.update(accepted=True, log_mass=masses[-1])
        print('IID level 1:', json.dumps(row), flush=True)
        save()

        step = jax.jit(proposals.build_parameter_step(prior, like, scales))
        report['parameter_contract_failures'] = 0
        def record(p, info, phase, threshold, walker=-1, iteration=-1, assigned=-1, eligible=-1, transition=None):
            nonlocal audit_evals
            try:
                assert np.all(np.isfinite(np.asarray(p.position)))
                assert np.isfinite(float(p.logdensity)) and np.isfinite(float(p.loglikelihood))
                assert np.isneginf(threshold) or float(p.loglikelihood) > threshold
                np.testing.assert_allclose(p.logdensity, prior(p.position), rtol=0, atol=1e-9)
                np.testing.assert_allclose(p.loglikelihood, like(p.position), rtol=0, atol=1e-7)
            except AssertionError:
                report['parameter_contract_failures'] += 1
                raise
            audit_evals += 1
            r = dict(phase=phase, walker=walker, iteration=iteration, threshold=threshold,
                     position=np.asarray(p.position), logprior=float(p.logdensity), loglikelihood=float(p.loglikelihood),
                     component=int(info.component), labels=np.asarray(info.labels).tolist(),
                     slice_accepted=bool(info.slice.is_accepted), num_steps=int(info.slice.num_steps),
                     num_shrink=int(info.slice.num_shrink), cost=int(info.slice.num_steps+info.slice.num_shrink),
                     assigned_level=assigned, highest_eligible_level=eligible,
                     level_previous=-1, level_proposed=-1, level_eligible=False, level_probability=0., level_accepted=False)
            if transition is not None:
                r.update(dict(zip(['level_previous', 'level_proposed', 'level_eligible', 'level_probability', 'level_accepted'],
                                  [np.asarray(v).item() for v in transition])))
            records.append(r)
            return r

        # Sequential walkers bound memory without collapsing independent chains.
        histories = []
        report['walker_initialization'] = []
        for b, name in enumerate(['selection', 'calibration']):
            survivor_seed = args.seed+20+b
            indices = bootstrap.survivor_indices(bank_likes[b], thresholds[1], jax.random.key(survivor_seed), 8)
            walker_seeds = [args.seed+100+100*b+w for w in range(8)]
            report['walker_initialization'].append(dict(bank=name, survivor_seed=survivor_seed,
                        survivor_indices=indices.tolist(), walker_seeds=walker_seeds,
                        initial_positions=banks[b][indices].tolist()))
            values = np.empty((256, 8))
            start_record = len(records)
            for w, index in enumerate(indices):
                p = dns.DNSParticleState(jnp.asarray(banks[b][index]), jnp.asarray(bank_priors[b][index]), jnp.asarray(bank_likes[b][index]))
                key = jax.random.key(walker_seeds[w])
                for i in range(128+256):
                    key, subkey = jax.random.split(key)
                    p, info = step(subkey, p, jnp.asarray(thresholds[1]))
                    record(p, info, name, thresholds[1], w, i)
                    if i >= 128:
                        values[i-128, w] = float(p.loglikelihood)
                    if (i+1) % 128 == 0:
                        print(name, 'walker', w, 'step', i+1, 'logL', float(p.loglikelihood), flush=True)
            histories.append(values)
            report.setdefault('constrained_costs', {})[name] = summarize(records[start_record:])
            save()
        np.savez(args.output/'constrained_histories.npz', selection=histories[0], calibration=histories[1])
        selection = dns_levels.build_next_level(histories[0], thresholds[1], block_size=32, min_ess=20.)
        row = dict(level=2, method='8 constrained walkers per bank', selection=selection, accepted=False)
        report['ladder'].append(row)
        if 'threshold' in selection:
            row['selection_correlation'] = bootstrap.correlation_breakdown(histories[0], selection['threshold'])
        if selection['status'] != 'ok':
            report.update(status='stopped_level2_selection_ESS', diagnosis='Selection tail ESS/information gate failed; no tuning or production.')
            return
        calibration = dns_levels.tail_diagnostics(histories[1], selection['threshold'], block_size=32)
        row['calibration'] = calibration
        row['calibration_correlation'] = bootstrap.correlation_breakdown(histories[1], selection['threshold'])
        if not 0 < calibration['ratio'] < 1 or calibration['ess'] < 20:
            report.update(status='stopped_level2_calibration_ESS', diagnosis='Independent constrained calibration failed. See separate within-walker/between-walker uncertainty terms; no tuning or production.')
            return
        thresholds.append(selection['threshold']); masses.append(masses[-1]+np.log(calibration['ratio']))
        row.update(accepted=True, log_mass=masses[-1])
        save()

        # Historical optimum is loaded only after both calibrations have passed.
        archive = np.load(args.initial_families)
        matches = np.flatnonzero(archive['seeds'] == 44)
        assert matches.size == 1
        initial = jnp.asarray(archive['positions'][matches[0]])
        assert initial.shape == (54,) and initial.dtype == jnp.float64
        np.testing.assert_allclose(like(initial), -90964.02933083786, rtol=0, atol=1e-6)
        levels = dns.create_levels(thresholds, masses, np.zeros(3))
        highest = int(np.sum(float(like(initial)) > np.asarray(thresholds))-1)
        state = dns.init(initial, prior, like, levels, level_index=highest)
        report['production_initialization'] = dict(family=44, assigned_level=highest,
                        logL=float(state.particle.loglikelihood), logprior=float(state.particle.logdensity),
                        archive_sha256=hashlib.sha256(args.initial_families.read_bytes()).hexdigest())
        kernel = jax.jit(dns.build_kernel(levels, step))
        key = jax.random.key(args.seed+10000)
        report['production_seed'] = args.seed+10000
        indices, production = [highest], []
        best, lowest = state.particle, state.particle
        lowest_index = highest
        report['catalogues'] = {'initial': audit.catalogue(problem, initial)}
        for i in range(args.sweeps):
            key, subkey = jax.random.split(key)
            previous = int(state.level_index)
            state, info = kernel(subkey, state)
            assert np.all(np.asarray(info.parameter_is_valid))
            pi = jax.tree.map(lambda x: x[0], info.parameter_info)
            li = jax.tree.map(lambda x: x[0], info.level_info)
            index = int(state.level_index)
            eligible = int(np.sum(float(state.particle.loglikelihood)>np.asarray(thresholds))-1)
            production.append(record(state.particle, pi, 'production', thresholds[previous], iteration=i,
                                     assigned=index, eligible=eligible, transition=li))
            indices.append(index)
            if float(state.particle.loglikelihood)>float(best.loglikelihood):
                best = state.particle
            if index <= lowest_index:
                lowest, lowest_index = state.particle, index
            if (i+1) % 128 == 0:
                print('production', i+1, 'level', index, flush=True)
        directions = {}
        for name, sign in [('up', 1), ('down', -1)]:
            rows = [r for r in production if r['level_proposed']-r['level_previous'] == sign]
            directions[name] = dict(proposals=len(rows), eligible=sum(r['level_eligible'] for r in rows),
                                    accepted=sum(r['level_accepted'] for r in rows))
        report['production'] = dict(status='completed', **summarize(production), **level_summary(indices, 3),
                    level_proposals=directions, parameter_failures=0, best_logL=float(best.loglikelihood),
                    latent_displacement=float(np.linalg.norm(np.asarray(state.particle.position-initial))))
        checkpoints = [('initial', initial), ('best', best.position), ('lowest_assigned_level', lowest.position), ('final', state.particle.position)]
        for name, position in checkpoints[1:]:
            report['catalogues'][name] = audit.catalogue(problem, position)
        np.savez(args.output/'checkpoints.npz', names=[n for n,p in checkpoints], positions=[np.asarray(p) for n,p in checkpoints])
        report['status'] = 'completed' if report['production']['round_trips'] else 'completed_without_round_trip'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
        print('Report:', args.output/'report.json', report['status'], flush=True)


if __name__ == '__main__':
    main()
