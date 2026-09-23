"""Cheap fixed-design transplantation tests; no LISA likelihood calls."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.source_transplant_diagnostic import (
    representative_indices,temporal_index,nearest_companion,pair_assignment,
    transplant,experiment_plan,compatibility_matrices)
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices


def test_fixed_retained_indices_are_even_and_cover_endpoints():
    np.testing.assert_array_equal(representative_indices(),np.arange(16)*17)
    assert len(set(representative_indices()))==16


def test_temporal_control_fixed_offset_and_no_self_donor():
    assert temporal_index(0)==128 and temporal_index(255)==127
    for k in range(256):
        assert temporal_index(k)!=k and temporal_index(temporal_index(k))==k
    with pytest.raises(ValueError):temporal_index(256)


def test_single_transplant_copies_donor_into_distinct_label_slot():
    base=np.arange(54,dtype=float);donor=-np.arange(54,dtype=float)-1
    result=transplant(base,donor,[3],[7]);ix=block_indices([3])
    np.testing.assert_array_equal(result[ix],donor[block_indices([7])])
    keep=np.setdiff1d(np.arange(54),ix)
    assert result[keep].tobytes()==base[keep].tobytes()
    np.testing.assert_array_equal(base,np.arange(54))


def test_nearest_companion_excludes_sticky_and_breaks_ties_by_label():
    f=np.array([10.,14.,100.,200.,12.,300.,400.,500.,600.])
    assert nearest_companion(f,4)==0
    f[4]=10.
    assert nearest_companion(f,4)==0  # distinct source with same frequency


def test_assignment_chooses_frequency_optimum_without_likelihood():
    recipient=np.r_[0.,10.,np.arange(7)+100.]
    donor=np.r_[9.,1.,np.arange(7)+200.]
    slots,sources,swapped,straight,reverse=pair_assignment(recipient,donor,0,0)
    np.testing.assert_array_equal(slots,[0,1]);np.testing.assert_array_equal(sources,[1,0])
    assert swapped and reverse<straight
    # No likelihood or state-quality input can influence assignment.
    import inspect
    assert tuple(inspect.signature(pair_assignment).parameters)==('recipient_f','donor_f','recipient_sticky','donor_sticky')


def test_pair_assignment_tie_retains_sticky_to_sticky():
    slots,sources,swapped,a,b=pair_assignment(np.r_[1.,1.,np.arange(7)+100.],np.r_[0.,2.,np.arange(7)+100.],0,0)
    assert a==b and not swapped
    np.testing.assert_array_equal(sources,[0,1])


def test_two_block_transplant_keeps_each_six_coordinate_block_intact():
    base=np.arange(54,dtype=float);donor=np.arange(54,dtype=float)+1000
    result=transplant(base,donor,[1,8],[6,2]);ix=block_indices([1,8]);keep=np.setdiff1d(np.arange(54),ix)
    np.testing.assert_array_equal(result[ix],donor[block_indices([6,2])])
    assert result[keep].tobytes()==base[keep].tobytes()
    with pytest.raises(ValueError):transplant(base,donor,[1],[6,2])


def test_no_sequential_updates_or_shared_memory():
    base=np.zeros(54);donor=np.ones(54)
    first=transplant(base,donor,[1],[2]);second=transplant(base,2*donor,[7],[4])
    assert np.all(second[6:12]==0) and np.all(first[6:12]==1)
    assert np.all(second[42:48]==2) and np.all(base==0)
    first[:]=9
    assert np.all(base==0) and np.all(second[6:12]==0)


def test_complete_design_counts_and_diagonal_control_mapping():
    rows=experiment_plan();assert len(rows)==1024
    assert sum(r['temporal_control'] for r in rows)==128
    assert sum(not r['temporal_control'] for r in rows)==896
    for r in rows:
        if r['recipient']==r['donor']:assert r['donor_retained_index']==temporal_index(r['retained_index'])
        else:assert r['donor_retained_index']==r['retained_index']


def test_matrix_bookkeeping_retains_zeros_and_directionality():
    rows=[dict(**r,passes_ell9=r['recipient']==r['donor'] or r['donor']==0,
        delta_logL=10*r['recipient']-r['donor']+r['retained_index']/255) for r in experiment_plan()]
    matrices=compatibility_matrices(rows)
    assert np.all(matrices['count']==16)
    assert np.all(np.diag(matrices['survival_fraction'])==1)
    assert matrices['survival_fraction'][0,1]==0 and matrices['survival_fraction'][1,0]==1
    assert matrices['median_delta_logL'][0,1]==pytest.approx(-.5)
    assert matrices['q25_delta_logL'][0,1]==pytest.approx(-.75)
    assert matrices['q75_delta_logL'][0,1]==pytest.approx(-.25)
    assert len(matrices['directional_asymmetry'])==28
    assert matrices['directional_asymmetry'][0]['survival_difference']==-1
    with pytest.raises(ValueError,match='16 tests'):compatibility_matrices(rows[:-1])
