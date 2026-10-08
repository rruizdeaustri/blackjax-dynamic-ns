"""Deterministic diagnostic contracts; no stochastic trajectories."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('gate', Path(__file__).parents[2] / 'examples/dns_conditional_reliability_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_holm_stepdown_stops_and_controls_family():
    assert gate.holm({'a': .001, 'b': .02, 'c': .9}) == ['a', 'b']
    assert gate.holm({'a': .02, 'b': .03, 'c': .9}) == []


def test_multivariate_energy_invariance_and_shift():
    x = np.array([[0., 1.], [2., 3.], [4., 5.]])
    assert abs(gate.energy(x, x[::-1])) < 1e-12
    assert gate.energy(x, x+10) > 0
    assert abs(gate.energy(x, x+10)-gate.energy(x+10, x)) < 1e-12


def test_split_rhat_detects_between_walker_location_disagreement():
    # Common within-chain ranks but disjoint location ranges across walkers.
    base = np.tile(np.arange(64.), 32)
    shifted = np.stack([base+i*100 for i in range(8)])
    assert gate.diagnostics(shifted)['rhat'] > 1.01
    assert gate.diagnostics(np.ones((8,2048))) == {'constant': True}


def test_compression_matches_frozen_saved_implementation():
    from blackjax.ns.dns_levels import tail_diagnostics
    y = np.stack([(np.arange(2048)*(i+1)) % 101 for i in range(8)])
    assert np.isclose(gate.compression_ess(y, 65), tail_diagnostics(y.T,65)['ess'])
