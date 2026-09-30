"""Saved LISA arrays and algebra only. Never load or evaluate the LISA model."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from blackjax.ns import dns_automatic as a, dns_population as p
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4 import automatic_ladder_integration as integration
from examples.lisa_dns_stage4 import stage6b_pooled_promotion_validation as frozen


def test_frozen_population_algebra_source_and_pairing():
    expected = inspect.getsource(frozen.joint_decision).replace('uniform):', 'uniform,threshold):').replace('ELL_TEST', 'threshold')
    assert inspect.getsource(p.joint_decision) == expected
    np.testing.assert_array_equal(p.round_robin(), frozen.round_robin())
    src = inspect.getsource(integration.make_frozen_kernel)
    assert 'proposals.build_parameter_step' in src
    assert 'np.full(54, np.pi/np.sqrt(3)), max_steps=10, max_shrinkage=100' in src
    assert 'jax.lax.map' in src


def test_historical_levels_bookkeeping_saved_only():
    if not Path('/tmp/lisa_dns_stage4z_population_level12').exists():
        pytest.skip('Historical integration artifacts not installed')
    rows = integration.historical_bookkeeping()
    assert len(rows) == 3 and all(r['exact_threshold_prefix'] and r['exact_log_masses'] for r in rows)


def test_lisa_dry_run_uses_old_contour_and_no_candidate(monkeypatch):
    if not Path('/tmp/lisa_dns_stage5b_population_level13').exists():
        pytest.skip('Saved integration artifacts not installed')
    def forbidden(*args, **kwargs):
        raise AssertionError('No LISA sampling, candidate construction, or checkpoint writes')
    monkeypatch.setattr(a, 'build_next_level', forbidden)
    monkeypatch.setattr(a, 'select_candidate', forbidden)
    monkeypatch.setattr(a, 'save_checkpoint', forbidden)
    monkeypatch.setattr(integration, 'make_frozen_kernel', forbidden)
    result = integration.lisa_dry_run()
    assert result['prefix_unchanged'] and result['new_LISA_model_calls'] == result['new_LISA_sweeps'] == 0
    assert not result['candidate_constructed'] and not result['checkpoint_written']
    for bank in a.BANKS:
        assert result['promotion'][bank]['eligible'] == 2048
    assert result['numerical_contract']['likelihood_atol'] == 1e-7
    assert result['environment']['backend'] == 'cpu' and result['environment']['x64']


def test_cpu_contract_matches_recorded_stage6a_fixture():
    data = Path('tests/experimental/data/dns_lisa_cpu_cache_audit.npz')
    with np.load(data) as f:
        particles = DNSParticleState(f['position'], f['logprior'], f['loglikelihood'])
        a.NumericalContract().validate(particles, -108187.99712765554,
            (f['recomputed_prior'], f['recomputed_likelihood']))


def test_exact_frozen_exchange_replay_without_model_evaluations():
    root = Path('/tmp/lisa_dns_stage6b_pooled_promotion_validation')
    if not root.exists():
        pytest.skip('Saved Stage-6B operations not installed')
    bounds = json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']['f0']
    for bank in a.BANKS:
        raw = frozen.load(root/bank/'trace.npz'); ex = frozen.load(root/bank/'exchanges.npz')
        rng = frozen.load(root/bank/'random_schedule.npz')
        for sweep in range(384):
            known = {}
            for n in range(4*sweep, 4*sweep+4):
                for x, lp, ll in zip(ex['proposed_position'][n], ex['proposed_logprior'][n], ex['proposed_loglikelihood'][n]):
                    known[x.tobytes()] = (lp, ll)
            for x, lp, ll in zip(raw['position'][:, sweep], raw['recomputed_prior'][:, sweep], raw['recomputed_likelihood'][:, sweep]):
                known[x.tobytes()] = (lp, ll)
            def evaluate(xs):
                values = np.array([known[x.tobytes()] for x in xs])
                return values[:, 0], values[:, 1]
            def forbidden(*args):
                raise AssertionError('No slice/model execution in saved replay')
            kernel = p.FrozenPopulationKernel(forbidden, evaluate,
                lambda x: frozen.log_probabilities(x, bounds)[0], frozen.fixed_exchange,
                frozen.slice_blocks, b'{}')
            sliced = DNSParticleState(raw['post_slice_position'][:, sweep], raw['post_slice_logprior'][:, sweep],
                                      raw['post_slice_loglikelihood'][:, sweep])
            result, decisions = kernel.exchange_sweep(sliced, frozen.ELL_TEST, sweep,
                rng['gumbels'][sweep], rng['uniforms'][sweep], a.NumericalContract())
            for actual, key in zip(result, ['position', 'logprior', 'loglikelihood']):
                assert actual.tobytes() == raw[key][:, sweep].tobytes()
            for slot, decision in enumerate(decisions):
                for key, value in decision.items():
                    np.testing.assert_equal(value, ex[key][4*sweep+slot])
