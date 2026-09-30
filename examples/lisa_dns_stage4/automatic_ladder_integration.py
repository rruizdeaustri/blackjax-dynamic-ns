"""Frozen LISA operation bundle and saved-artifact-only integration checks.

No LISA problem loader or trajectory entry point is provided in this module.
"""
from pathlib import Path
import json
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns_automatic as automatic
from blackjax.ns import dns_population
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    log_probabilities, fixed_exchange, fixed_frequency_region, slice_blocks,
)

ROOT = Path(__file__).resolve().parents[2]


def kernel_metadata(model_id, frequency_bounds):
    paths = [Path(__file__), Path(automatic.__file__), Path(dns_population.__file__),
             Path(proposals.__file__), ROOT/'blackjax/ns/dns_kernels.py',
             ROOT/'blackjax/ns/dns_levels.py',
             ROOT/'examples/lisa_dns_stage4/frequency_population_benchmark.py',
             ROOT/'examples/lisa_dns_stage4/analyze_frequency_exchange_design.py']
    return dict(version='frozen-population-v1', model_id=model_id,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scales='pi/sqrt(3)', mixture=[.2, .4, .4], max_steps=10, max_shrinkage=100,
        f_star=fixed_frequency_region()[0], sigma_f=fixed_frequency_region()[1],
        frequency_bounds=list(frequency_bounds), pairing=dns_population.round_robin().tolist(),
        slice_updates=8, reciprocal_exchanges=4, exact_state_dependent_selector_correction=True)


def make_frozen_kernel(prior, likelihood, frequency_bounds, *, model_id):
    """Bind model functions to the unchanged scientific transition, without tuning."""
    automatic.NumericalContract().environment()
    scalar_step = proposals.build_parameter_step(jax.jit(prior), jax.jit(likelihood),
        np.full(54, np.pi/np.sqrt(3)), max_steps=10, max_shrinkage=100)
    step = jax.jit(lambda keys, states, contour: jax.lax.map(
        lambda data: scalar_step(data[0], data[1], contour), (keys, states)))
    evaluate_batch = jax.jit(lambda xs: jax.lax.map(lambda x: (prior(x), likelihood(x)), xs))

    def evaluate(position):
        shape = np.shape(position)[:-1]
        values = evaluate_batch(jnp.asarray(position).reshape(-1, 54))
        return tuple(np.asarray(v).reshape(shape) for v in values)

    return dns_population.FrozenPopulationKernel(step, evaluate,
        lambda x: log_probabilities(x, frequency_bounds)[0], fixed_exchange, slice_blocks,
        automatic.encoded(kernel_metadata(model_id, frequency_bounds)))


def retained_history(path, burn_in):
    with np.load(path, allow_pickle=False) as data:
        return DNSParticleState(*[np.swapaxes(data[key][:, burn_in:], 0, 1).copy()
                                   for key in ('position', 'logprior', 'loglikelihood')])


def historical_bookkeeping():
    """Reconstruct existing accepted rows only; never write a LISA checkpoint."""
    rows = []
    config = automatic.LadderConfig()
    for folder, level in [('stage4v_population_level10', 10),
                          ('stage4x_population_level11', 11),
                          ('stage4z_population_level12', 12)]:
        root = Path('/tmp/lisa_dns_'+folder)
        checkpoint = json.loads((root/f'checkpoint_level{level}.json').read_text())
        payload = checkpoint['payload']
        thresholds = tuple(float(v) for v in payload['thresholds'])
        masses = tuple(payload['log_masses'])
        selection, calibration = [retained_history(root/bank/'trace.npz', 128)
                                  for bank in automatic.BANKS]
        candidate = automatic.select_candidate(selection, thresholds[-2], config)
        expected_candidate = json.loads((root/'selection_candidate.json').read_text())
        assert candidate.diagnostics == expected_candidate
        cal = automatic.calibrate(candidate, calibration, config.block_size)
        t, m, accepted = automatic.freeze_candidate(thresholds[:-1], masses[:-1], candidate, cal)
        assert accepted and t == thresholds and m == masses
        # This is replay of bookkeeping, not a claim to reproduce old start selection.
        rows.append(dict(level=level, candidate=candidate.diagnostics, candidate_hash=candidate.sha256,
            calibration=cal, exact_threshold_prefix=True, exact_log_masses=True,
            checkpoint_sha256=hashlib.sha256((root/f'checkpoint_level{level}.json').read_bytes()).hexdigest(),
            note='Historical starts used the former recovery rule; current promotion is the frozen pooled rule.'))
    return rows


def lisa_dry_run(seed=170100):
    """Prepare next-iteration inputs only. No candidate selection or model calls."""
    root = Path('/tmp/lisa_dns_stage4z_population_level12')
    payload = json.loads((root/'checkpoint_level12.json').read_text())['payload']
    bounds = json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']['f0']
    banks = {n: retained_history(Path('/tmp/lisa_dns_stage5b_population_level13')/n/'trace.npz', 128)
             for n in automatic.BANKS}
    state = automatic.LadderState(tuple(float(v) for v in payload['thresholds']),
        tuple(payload['log_masses']), **banks, seed=seed,
        kernel_metadata=automatic.encoded(kernel_metadata('saved-LISA-config', bounds)))
    before = (state.thresholds, state.log_masses)
    starts, promotion, streams = automatic.prepare_iteration(state)
    repeated, repeated_promotion, repeated_streams = automatic.prepare_iteration(state)
    assert (state.thresholds, state.log_masses) == before
    assert promotion == repeated_promotion and streams == repeated_streams
    for n in automatic.BANKS:
        for a, b in zip(starts[n], repeated[n]):
            assert a.tobytes() == b.tobytes()
    return dict(kind='dry_run_only', old_contour=state.thresholds[-1], seed=seed,
        promotion=promotion, streams=streams, numerical_contract=automatic.asdict(state.contract),
        environment=state.contract.environment(), kernel=json.loads(state.kernel_metadata),
        promoted_hashes={n: automatic.particle_hash(starts[n]) for n in automatic.BANKS},
        prefix_unchanged=True, new_LISA_model_calls=0, new_LISA_sweeps=0,
        candidate_constructed=False, checkpoint_written=False,
        cache_recomputation='Not performed: copied cached values retain saved provenance; a future live run must pass recomputation before sampling.')
