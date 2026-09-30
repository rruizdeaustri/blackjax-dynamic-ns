"""Dynamic automatic construction and restart on the multimodal toy only."""
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns_automatic as a
from blackjax.ns.dns import DNSParticleState
from examples import dns_automatic_ladder as toy
from examples.lisa_dns_stage4 import proposals


def test_frozen_slice_adapter_matches_original_on_toy():
    kernel = toy.kernel()
    x = np.zeros((8, 54)); x[:, 0] = np.tile([-1.5, 1.5], 4)
    lp, ll = kernel.evaluate(x)
    particles = DNSParticleState(*map(jnp.asarray, (x, lp, ll)))
    keys = jax.random.split(jax.random.key(773), 8)
    original = proposals.build_parameter_step(jax.jit(toy.prior), jax.jit(toy.likelihood),
        np.full(54, np.pi/np.sqrt(3)), max_steps=10, max_shrinkage=100)
    reference = jax.jit(lambda keys, states: jax.lax.map(
        lambda data: original(data[0], data[1], jnp.asarray(-.5)), (keys, states)))
    expected = reference(keys, particles)
    actual = kernel.slice_step(keys, particles, jnp.asarray(-.5))
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(x, y)


def test_multiple_toy_levels_and_fresh_process_restart(tmp_path):
    state, kernel = toy.initial_state()
    full = a.run_ladder(state, kernel, tmp_path/'continuous', max_new_levels=4)
    assert full.status == 'ready' and len(full.thresholds) == 5
    assert np.all(np.diff(full.thresholds)>0) and np.all(np.diff(full.log_masses)<0)
    for item in full.records:
        row = json.loads(item)
        assert row['accepted'] and row['calibration']['ess'] >= 20
        for bank in a.BANKS:
            assert row[bank+'_sampling']['completed_sweeps'] == 384
            assert row[bank+'_sampling']['all_constituent_checks_passed']
    # Resume in a new interpreter. Nothing about the execution schedule is inferred
    # from files other than the checksummed completed-boundary checkpoint.
    checkpoint = tmp_path/'continuous/attempt_000001/checkpoint'
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1')
    subprocess.run([sys.executable, '-m', 'examples.dns_automatic_ladder', '--resume', str(checkpoint),
        '--levels', '2', '--output', str(tmp_path/'resumed')], env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    resumed = a.load_checkpoint(tmp_path/'resumed/attempt_000003/checkpoint', kernel)
    assert full.thresholds == resumed.thresholds and full.log_masses == resumed.log_masses
    assert full.records == resumed.records
    for bank in a.BANKS:
        assert a.particle_hash(getattr(full, bank)) == a.particle_hash(getattr(resumed, bank))
    assert full.contract == resumed.contract and full.iteration == resumed.iteration
