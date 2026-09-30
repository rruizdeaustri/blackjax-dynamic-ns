"""Automatic ladder integration: synthetic banks and a multimodal toy only."""
from dataclasses import FrozenInstanceError, replace
import inspect
import json

import numpy as np
import pytest

from blackjax.ns import dns_automatic as a
from blackjax.ns.dns import DNSParticleState


def history(values, tag=1.):
    values = np.asarray(values, dtype=np.float64)
    x = np.stack([values, np.full_like(values, tag)], axis=-1)
    return DNSParticleState(x, np.zeros_like(values), values.copy())


class SyntheticKernel:
    metadata = dict(version='synthetic-v1')

    def __init__(self, folder=None, fail=False, tamper=False):
        self.calls = []
        self.folder, self.fail, self.tamper = folder, fail, tamper

    def evaluate(self, x):
        return np.zeros(x.shape[:-1]), np.asarray(x)[..., 0].copy()

    def run(self, starts, threshold, streams, config, contract):
        name = streams['bank']
        self.calls.append((name, starts.position.copy(), threshold, streams.copy(), contract))
        if self.folder:
            assert (self.folder/'candidate.json').exists() == (name == 'calibration')
        base = 0. if np.isneginf(threshold) else threshold
        values = base+np.random.default_rng(streams['label_seed']).uniform(.01, 1., (config.retained, 8))
        if name == 'calibration' and self.fail:
            values[:] = base+.001
        if name == 'calibration' and self.tamper:
            (self.folder/'candidate.json').write_text('{}')
        return history(values, starts.position[0, 1]), dict(completed_sweeps=config.burn_in+config.retained)


def initial(kernel=None, threshold=-np.inf):
    kernel = kernel or SyntheticKernel()
    t, m = ((-np.inf,), (0.,)) if np.isneginf(threshold) else ((-np.inf, threshold), (0., -1.))
    return a.LadderState(t, m, history(np.ones((4, 8)), 1), history(np.ones((4, 8)), 2),
                         seed=8, kernel_metadata=a.encoded(kernel.metadata),
                         config=a.LadderConfig(burn_in=4, retained=128, block_size=16))


@pytest.mark.parametrize('bank', a.BANKS)
def test_pooled_promotion_zero_donor_and_bank_separation(bank):
    values = np.ones((4, 8)); values[:, 3] = -1
    h = history(values, 1 if bank == 'selection' else 2)
    starts, proof = a.promote(h, 0., 19)
    assert proof['eligible'] == 28 and proof['eligible_per_donor'][3] == 0
    assert 3 not in proof['donor_ids']
    for actual, source in zip(starts, h):
        expected = source[proof['retained_indices'], proof['donor_ids']]
        assert actual.tobytes() == expected.tobytes()
        assert not np.shares_memory(actual, source)


def test_replacement_duplicates_and_entire_empty():
    h = history(np.full((4, 8), -1.))
    h.loglikelihood[2, 5] = 1.; h.position[2, 5, 0] = 1.
    starts, p = a.promote(h, 0., 19)
    assert p['duplicates'] == 7 and p['donor_multiplicities'][5] == 8
    assert len(p['duplicate_source_hashes']) == 1
    assert np.all(starts.loglikelihood > 0.)
    with pytest.raises(ValueError, match='Entire bank'):
        a.promote(h, 1., 19)


def test_promotion_is_exact_generic_survivors():
    import jax
    from blackjax.ns import dns_levels
    h = history(np.arange(32.).reshape(4, 8))
    starts, _ = a.promote(h, 10., 19)
    expected = dns_levels._survivors(jax.random.key(19, impl='threefry2x32'), h, 10., 8)
    assert a.particle_hash(starts) == a.particle_hash(expected)


def test_independent_storage_streams_and_no_rescue(tmp_path):
    state = initial(threshold=0.)
    starts, _, streams = a.prepare_iteration(state)
    assert not any(np.shares_memory(x, y) for x in starts['selection'] for y in starts['calibration'])
    assert streams['selection'] != streams['calibration']
    assert np.all(starts['selection'].position[..., 1] == 1.)
    assert np.all(starts['calibration'].position[..., 1] == 2.)
    state = replace(state, calibration=history(np.full((4, 8), -1.), 2))
    kernel = SyntheticKernel()
    stopped = a.build_next_level(state, kernel, tmp_path/'attempt')
    assert stopped.status == 'stopped' and stopped.thresholds == state.thresholds
    assert not kernel.calls
    assert 'Entire bank' in json.loads(stopped.records[-1])['error']


def test_candidate_immutable_and_saved_before_calibration(tmp_path):
    folder = tmp_path/'attempt'
    kernel = SyntheticKernel(folder=folder)
    state = initial(kernel)
    result = a.build_next_level(state, kernel, folder)
    assert result.status == 'ready'
    candidate = a.Candidate((folder/'candidate.json').read_bytes())
    with pytest.raises(FrozenInstanceError):
        candidate.record = b'{}'
    mutated = candidate.diagnostics; mutated['threshold'] = 99
    assert candidate.threshold != 99
    assert [c[0] for c in kernel.calls] == ['selection', 'calibration']
    assert all(c[2] == state.thresholds[-1] for c in kernel.calls)
    assert all(c[4] == state.contract for c in kernel.calls)
    assert np.all(kernel.calls[0][1][..., 1] == 1.)
    assert np.all(kernel.calls[1][1][..., 1] == 2.)
    assert json.loads((folder/'candidate_hash.json').read_text())['sha256'] == candidate.sha256


def test_candidate_tampering_stops_without_freeze(tmp_path):
    folder = tmp_path/'attempt'
    kernel = SyntheticKernel(folder=folder, tamper=True)
    state = initial(kernel)
    result = a.build_next_level(state, kernel, folder)
    assert result.status == 'stopped' and result.thresholds == state.thresholds
    assert not (folder/'checkpoint').exists()
    assert 'Candidate changed' in json.loads(result.records[-1])['error']


@pytest.mark.parametrize('ess, accepted', [(19.999999, False), (20., True), (20.000001, True)])
def test_ess_boundary_and_calibration_only_mass(ess, accepted):
    candidate = a.Candidate(a.encoded(dict(status='ok', threshold=2., ratio=.75)))
    t, m, passed = a.freeze_candidate((-np.inf, 1.), (0., -1.), candidate,
                                     dict(ratio=.25, ess=ess))
    assert passed == accepted
    assert len(t) == 2+accepted
    if accepted:
        assert m[-1] == -1.+np.log(.25) and m[-1] != -1.+np.log(.75)
    else:
        assert t == (-np.inf, 1.) and m == (0., -1.)


def test_failed_gate_clean_stop_and_no_retry(tmp_path):
    kernel = SyntheticKernel(fail=True); state = initial(kernel)
    result = a.run_ladder(state, kernel, tmp_path, max_new_levels=5)
    assert result.status == 'stopped' and result.iteration == 1
    assert result.thresholds == state.thresholds and result.log_masses == state.log_masses
    assert len(kernel.calls) == 2
    assert json.loads(result.records[-1])['stop_reason'] == 'calibration_gate'
    with pytest.raises(ValueError, match='Stopped'):
        a.build_next_level(result, kernel, tmp_path/'retry')
    with pytest.raises(FileExistsError):
        a.build_next_level(state, kernel, tmp_path/'attempt_000000')


def test_passing_gate_one_level_and_checkpoint(tmp_path):
    kernel = SyntheticKernel(); state = initial(kernel)
    new = a.build_next_level(state, kernel, tmp_path/'one')
    assert new.status == 'ready' and len(new.thresholds) == len(state.thresholds)+1
    row = json.loads(new.records[-1])
    assert new.log_masses[-1] == state.log_masses[-1]+np.log(row['calibration']['ratio'])
    assert all(k in row['calibration'] for k in ['naive_iid_se', 'block_means_se', 'between_walker_se', 'ess'])
    restored = a.load_checkpoint(tmp_path/'one/checkpoint', kernel)
    assert restored.thresholds == new.thresholds and restored.records == new.records
    for n in a.BANKS:
        assert a.particle_hash(getattr(restored, n)) == a.particle_hash(getattr(new, n))
    assert not restored.selection.position.flags.writeable


def test_checkpoint_checksums_and_kernel_identity(tmp_path):
    state = initial(); kernel = SyntheticKernel()
    a.save_checkpoint(tmp_path/'checkpoint', state)
    changed = SyntheticKernel(); changed.metadata = dict(version='changed')
    with pytest.raises(ValueError, match='kernel'):
        a.load_checkpoint(tmp_path/'checkpoint', changed)
    p = tmp_path/'checkpoint/banks.npz'
    p.write_bytes(p.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        a.load_checkpoint(tmp_path/'checkpoint', kernel)


def test_numerical_contract_finiteness_tolerance_and_strict_support():
    c = a.NumericalContract(); h = history(np.ones((2, 8)))
    c.validate(h, 0., (h.logdensity+5e-10, h.loglikelihood+5e-8))
    with pytest.raises(ValueError, match='Prior cache'):
        c.validate(h, 0., (h.logdensity+2e-9, h.loglikelihood))
    with pytest.raises(ValueError, match='recomputed contour'):
        c.validate(h, 1.-1e-8, (h.logdensity, h.loglikelihood-1e-8))
    with pytest.raises(ValueError, match='cached contour'):
        c.validate(h, 1.)
    with pytest.raises(ValueError, match='frozen'):
        a.NumericalContract(likelihood_atol=1e-6)
    with pytest.raises(ValueError, match='ESS'):
        a.LadderConfig(min_ess=21)
    h.position[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        c.validate(h)


def test_no_level_specific_branching_or_lisa_inputs():
    import ast
    tree = ast.parse(inspect.getsource(a))
    source = inspect.getsource(a)
    assert 'lisa' not in source.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            assert node.value > -1000
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            assert 'level ==' not in condition and 'walker ==' not in condition


def test_multi_level_loop_and_restart_with_synthetic_kernel(tmp_path):
    kernel = SyntheticKernel(); state = initial(kernel)
    full = a.run_ladder(state, kernel, tmp_path/'full', max_new_levels=4)
    assert full.status == 'ready' and len(full.thresholds) == 5
    middle = a.load_checkpoint(tmp_path/'full/attempt_000001/checkpoint', kernel)
    resumed = a.run_ladder(middle, kernel, tmp_path/'resumed', max_new_levels=2)
    assert full.thresholds == resumed.thresholds and full.log_masses == resumed.log_masses
    assert full.records == resumed.records
    assert np.all(np.diff(full.thresholds)>0) and np.all(np.diff(full.log_masses)<0)
    for bank in a.BANKS:
        assert a.particle_hash(getattr(full, bank)) == a.particle_hash(getattr(resumed, bank))
