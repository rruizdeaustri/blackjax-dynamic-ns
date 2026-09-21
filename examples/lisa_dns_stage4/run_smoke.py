"""Seed-44 engineering smoke run; no evidence or posterior reconstruction.

Run with the blackjax-ns Python and PYTHONPATH=. (CUDA defaults come from audit).
Selection/calibration use separate streams, but share a nonstationary seed-44
start: their masses are experimental, not validated prior volumes.
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
from examples.lisa_dns_stage4 import proposals


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


def finish_diagnostics(report, records, problem, initial):
    """Summarize a completed or stopped attempt without taking further steps."""
    if not records:
        return
    report['all_parameter_steps'] = summarize(records)
    report['all_parameter_steps']['parameter_contract_failures'] = 0
    report['cost_notice'] = ('Likelihood proxy is sum(num_steps + num_shrink), as in Stage 3; '
                            'it is not an exact waveform call count. Direct cache checks and '
                            'sparse reconstruction calls are additional work.')
    report['reproducibility'] = dict(seed=report['seed'],
        full_run_replayed=False, stream_scheme='fixed: seed+i; builder: seed+100*level+bank; production: seed+10000')
    for row in report.get('ladder', []):
        threshold = row['selection'].get('threshold')
        if threshold is None:
            continue
        for bank in ['selection', 'calibration']:
            values = np.array([r['loglikelihood'] for r in records
                               if r['phase'] == f'level_{row["level"]}_{bank}'])[128:]
            row[bank + '_block_exceedance_fractions'] = (values.reshape(-1, 32) > threshold).mean(1).tolist()
    if report.get('production', {}).get('status') == 'not_run_builder_stopped':
        best = max(records, key=lambda r: r['loglikelihood'])
        best_position = initial if report['initial_loglikelihood'] >= best['loglikelihood'] else best['position']
        report['catalogue_scope'] = 'Engineering checks only; no assigned-level production exists.'
        report['catalogues']['best_engineering'] = audit.catalogue(problem, jnp.asarray(best_position))
        report['catalogues']['final_engineering'] = audit.catalogue(problem, jnp.asarray(records[-1]['position']))
        report['engineering_latent_displacement'] = float(np.linalg.norm(records[-1]['position']-np.asarray(initial)))
        report['lowest_assigned_level_checkpoint'] = None
    checkpoint_names = ['initial']
    checkpoint_positions = [np.asarray(initial)]
    if report.get('production', {}).get('status') == 'not_run_builder_stopped':
        checkpoint_names += ['best_engineering', 'final_engineering']
        checkpoint_positions += [np.asarray(best_position), records[-1]['position']]
    else:
        production = [r for r in records if r['phase'] == 'production']
        if production:
            best = max(production, key=lambda r: r['loglikelihood'])
            lowest = min(r['assigned_level'] for r in production)
            low = next(r for r in reversed(production) if r['assigned_level'] == lowest)
            checkpoint_names += ['best', 'lowest_assigned_level', 'final']
            checkpoint_positions += [initial if report['initial_loglikelihood'] >= best['loglikelihood']
                                     else best['position'], low['position'], production[-1]['position']]
    return dict(names=checkpoint_names, positions=checkpoint_positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-families', type=Path, default=Path('/tmp/lisa_dns_stage4_audit/initial_families.npz'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/lisa_dns_stage4_smoke'))
    parser.add_argument('--seed', type=int, default=4404)
    parser.add_argument('--sweeps', type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(notice=audit.NOTICE, seed=args.seed, status='initializing',
                  concern='Both independent RNG banks start at the same trapped family, not equilibrated prior draws. ESS does not establish equilibrium or accurate global masses.',
                  mixture_probabilities=[.2, .4, .4])
    records = []
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
        if records:
            np.savez_compressed(args.output/'trace.npz', **{k: np.asarray([r[k] for r in records]) for k in records[0]})
    try:
        problem, cfg, config_path = audit.load_problem()
        prior, like = map(jax.jit, audit.scalar_functions(problem))
        archive = np.load(args.initial_families)
        matches = np.flatnonzero(archive['seeds'] == 44)
        assert matches.size == 1
        initial = jnp.asarray(archive['positions'][matches[0]])
        assert initial.shape == (54,) and initial.dtype == jnp.float64
        assert np.isfinite(float(prior(initial)))
        np.testing.assert_allclose(like(initial), -90964.02933083786, rtol=0, atol=1e-6)
        source = audit.HIST/'seed44/posterior.npz'
        scales = diagonal_scales(np.load(source)['samples_u'])
        report.update(config=cfg, config_path=str(config_path), devices=str(jax.devices()),
                      initial_logprior=float(prior(initial)), initial_loglikelihood=float(like(initial)),
                      scales=scales.tolist(), scale_source=dict(path=str(source), key='samples_u',
                      method='unweighted coordinate sample standard deviation, ddof=1; 69 seed44 samples; no clipping',
                      sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                      initial_archive_sha256=hashlib.sha256(args.initial_families.read_bytes()).hexdigest())
        print('Frozen scales:', scales.tolist(), flush=True)
        step = jax.jit(proposals.build_parameter_step(prior, like, scales))
        particle = dns.DNSParticleState(initial, prior(initial), like(initial))
        audit_evals = 0
        def check(p, threshold):
            nonlocal audit_evals
            assert np.all(np.isfinite(np.asarray(p.position)))
            assert np.isfinite(float(p.logdensity)) and np.isfinite(float(p.loglikelihood))
            assert np.isneginf(threshold) or float(p.loglikelihood) > threshold
            np.testing.assert_allclose(p.logdensity, prior(p.position), rtol=0, atol=1e-9)
            np.testing.assert_allclose(p.loglikelihood, like(p.position), rtol=0, atol=1e-7)
            audit_evals += 1
        def record(p, info, phase, threshold, assigned=-1, eligible=-1, transition=None):
            check(p, threshold)
            row = dict(phase=phase, position=np.asarray(p.position), logprior=float(p.logdensity),
                       loglikelihood=float(p.loglikelihood), component=int(info.component),
                       labels=np.asarray(info.labels).tolist(), slice_accepted=bool(info.slice.is_accepted),
                       num_steps=int(info.slice.num_steps), num_shrink=int(info.slice.num_shrink),
                       cost=int(info.slice.num_steps + info.slice.num_shrink),
                       assigned_level=assigned, highest_eligible_level=eligible,
                       level_previous=-1, level_proposed=-1, level_eligible=False,
                       level_probability=0., level_accepted=False)
            if transition is not None:
                row.update(dict(zip(['level_previous','level_proposed','level_eligible','level_probability','level_accepted'],
                                    [np.asarray(v).item() for v in transition])))
            records.append(row)
            return row
        def chain(seed, start, threshold, draws, burn, phase):
            key = jax.random.key(seed)
            p = start
            history, rows = [], []
            for i in range(burn + draws):
                key, subkey = jax.random.split(key)
                p, info = step(subkey, p, jnp.asarray(threshold))
                rows.append(record(p, info, phase, threshold))
                if i >= burn:
                    history.append(p)
                if (i + 1) % 64 == 0:
                    print(phase, i + 1, 'logL', float(p.loglikelihood), flush=True)
            return history, summarize(rows)
        report['catalogues'] = {'initial': audit.catalogue(problem, initial)}
        report['fixed_contours'] = []
        for i, threshold in enumerate([-np.inf, -91100.]):
            _, diag = chain(args.seed+i, particle, threshold, 64, 0, f'fixed_{i}')
            report['fixed_contours'].append(dict(threshold=threshold, **diag))
            save()
        config = dns_levels.ConstructionConfig(num_levels=3, construction_draws=256,
                    calibration_draws=256, burn_in=128, thinning=1, block_size=32, min_ess=20.)
        report['construction_settings'] = vars(config)
        report['ladder'] = []
        thresholds, masses = [-np.inf], [0.]
        banks = [particle, particle]
        for level in range(1, 3):
            histories, costs = [], []
            for bank in range(2):
                h, d = chain(args.seed+100*level+bank, banks[bank], thresholds[-1], 256, 128,
                             f'level_{level}_{"selection" if bank == 0 else "calibration"}')
                histories.append(h); costs.append(d)
            selection = dns_levels.build_next_level(np.array([float(p.loglikelihood) for p in histories[0]])[:,None],
                         thresholds[-1], block_size=32, min_ess=20.)
            row = dict(level=level, accepted=False, log_mass=None, selection=selection,
                       construction_cost=costs[0], calibration_cost=costs[1])
            report['ladder'].append(row)
            if selection['status'] != 'ok':
                report.update(status='stopped_builder_ESS', diagnosis='G: selection tail information insufficient; no tuning or production attempted.')
                break
            threshold = selection['threshold']
            calibration = dns_levels.tail_diagnostics(np.array([float(p.loglikelihood) for p in histories[1]])[:,None], threshold, 32)
            row['calibration'] = calibration
            if not 0 < calibration['ratio'] < 1 or calibration['ess'] < 20:
                report.update(status='stopped_calibration', diagnosis='G/B/D: independent stream has insufficient tail information; nonstationarity/mixing prevents mass validation.')
                break
            thresholds.append(threshold); masses.append(masses[-1]+np.log(calibration['ratio']))
            row['accepted'] = True
            row['log_mass'] = masses[-1]
            # Promote each bank using only its own eligible history and separate RNG.
            banks = [h[np.random.default_rng(args.seed+1000*level+b).choice(
                       [i for i,p in enumerate(h) if float(p.loglikelihood)>threshold])]
                     for b,h in enumerate(histories)]
            save()
        levels = dns.create_levels(thresholds, masses, np.zeros(len(masses)))
        np.savez(args.output/'levels.npz', thresholds=thresholds, log_mass=masses, log_weight=levels.log_weight, scales=scales)
        report['frozen_levels'] = dict(thresholds=thresholds, log_masses=masses, log_weights=np.asarray(levels.log_weight))
        if len(thresholds) != 3:
            report['production'] = {'status': 'not_run_builder_stopped'}
            return
        kernel = jax.jit(dns.build_kernel(levels, step))
        state = dns.init(initial, prior, like, levels)
        key = jax.random.key(args.seed+10000)
        indices, production = [0], []
        best, lowest = state.particle, state.particle
        failures = 0
        for i in range(args.sweeps):
            key, subkey = jax.random.split(key)
            previous = int(state.level_index)
            state, info = kernel(subkey, state)
            failures += int(np.sum(~np.asarray(info.parameter_is_valid)))
            assert failures == 0
            pi = jax.tree.map(lambda x: x[0], info.parameter_info)
            li = jax.tree.map(lambda x: x[0], info.level_info)
            index = int(state.level_index)
            eligible = int(np.sum(float(state.particle.loglikelihood)>np.asarray(levels.loglikelihood))-1)
            production.append(record(state.particle, pi, 'production', thresholds[previous], index, eligible, li))
            indices.append(index)
            if float(state.particle.loglikelihood)>float(best.loglikelihood): best=state.particle
            if index == 0: lowest=state.particle
        report['production'] = dict(**summarize(production), **level_summary(indices, 3), parameter_failures=failures,
                best_logL=float(best.loglikelihood), latent_displacement=float(np.linalg.norm(np.asarray(state.particle.position-initial))))
        for name, p in [('best',best),('lowest_assigned_level',lowest),('final',state.particle)]:
            report['catalogues'][name] = audit.catalogue(problem,p.position)
        report['status'] = 'completed' if report['production']['round_trips'] else 'completed_without_round_trip'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['direct_cache_check_likelihood_evaluations'] = locals().get('audit_evals',0)
        if report['status'] != 'error' and records:
            checkpoints = finish_diagnostics(report, records, problem, initial)
            np.savez(args.output/'checkpoints.npz', **checkpoints)
        save()
        print('Report:', args.output/'report.json', report['status'], flush=True)


if __name__ == '__main__':
    main()
