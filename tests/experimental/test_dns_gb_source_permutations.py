"""Cheap analysis-only source-set comparisons, no target evaluations."""
import numpy as np
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    source_distances, sorted_frequency_summary, circular_difference, cross_frequency_distance, DF)


def test_sort_catalogues_before_averaging():
    a=np.array([[1.,3.,2.],[3.,2.,1.]])
    result=sorted_frequency_summary(a)
    np.testing.assert_array_equal(result['mean'],[1,2,3])
    np.testing.assert_array_equal(result['variance'],[0,0,0])
    x=source_distances(a[:1].T,a[1:].T,[1.])
    assert x['labelled']>0 and x['sorted_f0']==0 and x['assignment']==0


def test_assignment_is_permutation_invariant_and_frequency_equals_sort():
    rng=np.random.default_rng(37)
    a=rng.normal(size=(9,6));b=rng.normal(size=(9,6))
    p=source_distances(a,b,np.ones(6))
    q=source_distances(a[rng.permutation(9)],b[rng.permutation(9)],np.ones(6))
    np.testing.assert_allclose(p['assignment'],q['assignment'])
    p=source_distances(a[:,:1],b[:,:1],[1.])
    np.testing.assert_allclose(p['assignment'],p['sorted_f0'])
    f=a[:,:1].T*DF;g=b[:,:1].T*DF
    np.testing.assert_allclose(cross_frequency_distance(f,g)['assignment'],p['assignment'])


def test_circular_coordinates_wrap_at_physical_period():
    np.testing.assert_allclose(circular_difference(.01,2*np.pi-.01,2*np.pi),.02)
    np.testing.assert_allclose(circular_difference(.2,np.pi+.2,np.pi),0,atol=1e-15)
    a=np.array([[0.,.01]]);b=np.array([[0.,2*np.pi-.01]])
    x=source_distances(a,b,[1.,2*np.pi],{1:2*np.pi})
    np.testing.assert_allclose(x['assignment'],.02/(2*np.pi*np.sqrt(2)))
