"""Cheap block bookkeeping and qualitative evidence rubric tests."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.diagnose_ladder_failure import block_diagnostics,early_late,classify_evidence


def test_blocks_keep_walkers_separate_and_strict_threshold():
    ll=np.array([[0,1,2,3],[4,5,6,7]],float)
    f=np.repeat(ll[:,:,None],9,axis=2)
    d=block_diagnostics(ll,f,2,block_size=2,physical_bounds=(.001,.002))
    np.testing.assert_array_equal(d['exceedance'],[[0,.5],[1,1]])
    np.testing.assert_array_equal(d['logL_mean'],[[.5,2.5],[4.5,6.5]])
    np.testing.assert_array_equal(d['logL_min'],[[0,2],[4,6]])
    np.testing.assert_array_equal(d['logL_max'],[[1,3],[5,7]])
    np.testing.assert_array_equal(d['logL_median'],d['logL_mean'])
    np.testing.assert_allclose(d['f0_u_variance'],.5)
    assert np.all((d['f0_hz_mean']>.001)&(d['f0_hz_mean']<.002))
    with pytest.raises(ValueError): block_diagnostics(ll,f,2,block_size=3)


def test_early_late_excludes_only_declared_burnin():
    ll=np.array([[100,100,0,0,2,2],[100,100,2,2,0,0]],float)
    f=np.repeat(ll[:,:,None],9,axis=2)
    d=early_late(ll,f,1,burn_in=2)
    np.testing.assert_array_equal(d['early']['exceedance'],[0,1])
    np.testing.assert_array_equal(d['late']['exceedance'],[1,0])
    np.testing.assert_array_equal(d['late_minus_early']['logL_mean'],[2,-2])
    np.testing.assert_array_equal(d['late_pairwise']['mean_logL_difference'],[[0,2],[-2,0]])
    np.testing.assert_allclose(d['late_pairwise']['mean_f0_u_euclidean_distance'],[[0,6],[6,0]])
    # Continuation has no implicit or retrospectively discarded burn-in.
    d=early_late(ll[:,:4],f[:,:4],1,burn_in=0)
    np.testing.assert_array_equal(d['early']['logL_mean'],[100,100])


def test_evidence_classification_does_not_force_mixed_traces_into_a_or_b():
    assert classify_evidence(coherent_convergence=True,persistent_separation=False,temporal_stability=False).startswith('A:')
    assert classify_evidence(coherent_convergence=False,persistent_separation=True,temporal_stability=True).startswith('B:')
    assert classify_evidence(coherent_convergence=False,persistent_separation=True,temporal_stability=False).startswith('C:')
    assert classify_evidence(coherent_convergence=True,persistent_separation=True,temporal_stability=False).startswith('C:')
