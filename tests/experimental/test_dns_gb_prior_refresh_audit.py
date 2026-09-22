"""Cheap algebra/indexing checks; no LISA likelihood calls or refresh chain."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.prior_refresh_audit import (
    block_indices,replace_from_fixed_base,label_from_index,ideal_log_mh_ratio,
    load_audited_transform,transform_source)


def test_source_block_indexing():
    np.testing.assert_array_equal(block_indices([0]),np.arange(6))
    np.testing.assert_array_equal(block_indices([8]),np.arange(48,54))
    np.testing.assert_array_equal(block_indices([1,7]),np.r_[6:12,42:48])
    with pytest.raises(ValueError): block_indices([1,1])


def test_untouched_blocks_and_base_are_unchanged():
    base=np.arange(54,dtype=float);saved=base.copy();donor=-base-100
    result=replace_from_fixed_base(base,donor,[2,6]);i=block_indices([2,6])
    keep=np.setdiff1d(np.arange(54),i)
    assert result[keep].tobytes()==base[keep].tobytes()
    np.testing.assert_array_equal(result[i],donor[i]);np.testing.assert_array_equal(base,saved)


def test_refresh_does_not_depend_on_current_selected_block():
    a=np.arange(54,dtype=float);b=a.copy();b[12:18]=999
    donor=np.full(54,-2.)
    np.testing.assert_array_equal(replace_from_fixed_base(a,donor,[2]),replace_from_fixed_base(b,donor,[2]))


def test_uniform_categorical_mapping_by_construction():
    singles=[label_from_index(i,1) for i in range(9)]
    pairs=[label_from_index(i,2) for i in range(36)]
    assert len(set(singles))==9 and len(set(pairs))==36
    assert all(a<b for a,b in pairs)
    assert all(sum(label in p for p in pairs)==8 for label in range(9))


def test_exact_prior_mh_cancellation_and_strict_contour():
    rng=np.random.default_rng(4);base=rng.normal(size=54);donor=rng.normal(size=54)
    for labels in [[3],[1,8]]:
        proposed=replace_from_fixed_base(base,donor,labels)
        assert ideal_log_mh_ratio(base,proposed,labels,2,3,1)==pytest.approx(0,abs=1e-12)
        assert ideal_log_mh_ratio(base,proposed,labels,2,1,1)==-np.inf
    with pytest.raises(ValueError):ideal_log_mh_ratio(base,base,[3],1,2,1)


def test_fixed_base_proposals_do_not_accumulate_updates():
    base=np.zeros(54)
    first=replace_from_fixed_base(base,np.ones(54),[1])
    second=replace_from_fixed_base(base,np.full(54,2.),[2])
    assert np.all(first[6:12]==1) and np.all(second[6:12]==0)
    assert np.all(second[12:18]==2) and np.all(base==0)


def test_actual_transform_clipping_prevents_exact_prior_identity():
    _,eps,_=transform_source();transform=load_audited_transform()
    result=np.asarray(transform(np.array([eps/4,eps/2,eps]),0.,1.))
    assert result[0]==result[1]==result[2]
    # Positive-measure uniform interval is collapsed to a target-zero-measure point.
    ideal=np.log(np.array([eps/4,eps/2]))-np.log1p(-np.array([eps/4,eps/2]))
    assert ideal[0]!=ideal[1]
