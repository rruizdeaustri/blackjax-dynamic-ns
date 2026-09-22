"""Cheap direct-logistic algebra, layout, and density-stability audit tests."""
import jax
import numpy as np
import pytest
from examples.lisa_dns_stage4.latent_logistic_gate import (
    analytic_logprior,direct_blocks,cancellation_terms,load_actual_jacobian)
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices,replace_from_fixed_base,label_from_index


def test_analytic_logistic_density_and_symmetry():
    u=np.linspace(-10,10,54)
    expected=np.log(np.exp(-u)/(1+np.exp(-u))**2).sum()
    assert float(analytic_logprior(u))==pytest.approx(expected,abs=1e-12)
    assert float(analytic_logprior(u))==pytest.approx(float(analytic_logprior(-u)),abs=1e-12)


def test_full_prior_block_delta_identity():
    old=np.linspace(-5,5,54);donor=np.linspace(3,-4,54)
    for labels in [[0],[8],[2,7]]:
        new=replace_from_fixed_base(old,donor,labels);indices=block_indices(labels)
        assert float(analytic_logprior(new)-analytic_logprior(old))==pytest.approx(
            float(analytic_logprior(new[indices])-analytic_logprior(old[indices])),abs=1e-12)


def test_direct_generator_shape_dtype_finite_no_old_clip_atoms():
    for blocks in [1,2]:
        x=np.asarray(direct_blocks(jax.random.key(8400+blocks),32,blocks))
        assert x.shape==(32,blocks,6) and x.dtype==np.float64 and np.all(np.isfinite(x))
        boundary=np.log(1e-9)-np.log1p(-1e-9)
        assert not np.any((x==boundary)|(x==-boundary))
        np.testing.assert_array_equal(x,direct_blocks(jax.random.key(8400+blocks),32,blocks))


@pytest.mark.parametrize('labels',[[5],[2,8]])
def test_single_and_pair_mh_cancellation(labels):
    old=np.linspace(-3,3,54);donor=np.linspace(6,-6,54)
    new=replace_from_fixed_base(old,donor,labels)
    assert cancellation_terms(old,new,labels)['uncancelled_log_ratio']==pytest.approx(0.,abs=1e-12)
    keep=np.setdiff1d(np.arange(54),block_indices(labels))
    assert old[keep].tobytes()==new[keep].tobytes()


def test_independent_label_pair_mapping_and_no_sequential_updates():
    assert len({label_from_index(i,1) for i in range(9)})==9
    assert len({label_from_index(i,2) for i in range(36)})==36
    base=np.zeros(54);donor=np.ones(54)
    first=replace_from_fixed_base(base,donor,[1]);second=replace_from_fixed_base(base,2*donor,[8])
    assert np.all(base==0) and np.all(second[6:12]==0) and np.all(first[6:12]==1)
    assert np.all(second[48:54]==2)


def test_actual_jacobian_has_positive_tail_identity_failure():
    actual=load_actual_jacobian()
    assert float(actual(0.,0.,1.))==pytest.approx(float(analytic_logprior([0.])),abs=1e-14)
    assert np.isfinite(float(analytic_logprior([40.])))
    assert np.isneginf(float(actual(40.,0.,1.)))
    assert abs(float(actual(36.,0.,1.))-float(analytic_logprior([36.])))>.01
