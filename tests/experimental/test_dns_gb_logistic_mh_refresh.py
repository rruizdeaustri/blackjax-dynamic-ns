"""Cheap general MH tests: no LISA likelihood evaluation."""
import jax
import numpy as np
import pytest
from examples.lisa_dns_stage4.logistic_mh_refresh import (
    logq, mh, design, fixed_proposal, ELL9)
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices, label_from_index


def test_stable_proposal_density():
    assert logq([0.]) == pytest.approx(-2*np.log(2))
    assert logq([-1000.,1000.]) == -2000.
    assert logq([36.]) == pytest.approx(-36.)


@pytest.mark.parametrize('labels', [[3], [1,8]])
def test_general_mh_identity_without_prior_cancellation(labels):
    old=np.linspace(-2,2,54)
    new=fixed_proposal(old,labels,np.full((len(labels),6),3.))
    # Deliberately non-Logistic and coupled target proves the general ratio.
    def target(x): return -.2*np.dot(x,x)-.01*x.sum()**2
    ix=block_indices(labels)
    lp0,lp1=target(old),target(new);q0,q1=logq(old[ix]),logq(new[ix])
    r,alpha=mh(lp0,lp1,q0,q1,ELL9+1)
    forward=np.exp(lp0+q1); reverse=np.exp(lp1+q0)
    assert r==pytest.approx(np.log(reverse/forward))
    assert abs(r)>1e-3
    assert alpha==pytest.approx(min(1,reverse/forward))
    _,back=mh(lp1,lp0,q1,q0,ELL9+1)
    assert forward*alpha==pytest.approx(reverse*back,rel=1e-12,abs=0)


def test_negative_infinite_prior_rejects():
    r,a=mh(-10.,-np.inf,-3.,-4.,ELL9+1)
    assert r==-np.inf and a==0


def test_contour_failure_and_equality_reject():
    for ll in [ELL9-1,ELL9,np.nan,-np.inf]:
        assert mh(-10.,-9.,-3.,-4.,ll)[1]==0


def test_nonfinite_draw_is_not_repaired():
    new=fixed_proposal(np.zeros(54),[0],np.full((1,6),np.inf))
    assert np.isposinf(new[:6]).all()
    assert mh(-10.,np.nan,-3.,-np.inf,np.nan,False)[1]==0
    with pytest.raises(ValueError): mh(-np.inf,-9.,-3.,-4.,ELL9+1)


def test_fixed_base_preservation_and_no_sequential_update():
    base=np.arange(54,dtype=float);saved=base.copy()
    first=fixed_proposal(base,[0],np.ones((1,6)))
    second=fixed_proposal(base,[2,8],np.full((2,6),-5.))
    np.testing.assert_array_equal(base,saved)
    for new,labels in [(first,[0]),(second,[2,8])]:
        keep=np.setdiff1d(np.arange(54),block_indices(labels))
        assert base[keep].tobytes()==new[keep].tobytes()
    np.testing.assert_array_equal(second[:6],base[:6])


def test_selection_construction_and_independent_streams():
    for size,n in [(1,9),(2,36)]:
        mapping=[label_from_index(i,size) for i in range(n)]
        assert len(set(mapping))==n
        assert all(tuple(sorted(s))==s and len(set(s))==size for s in mapping)
        key=jax.random.key(42);lk,dk=jax.random.split(key)
        cats,labels,draws=design(key,size,64)
        np.testing.assert_array_equal(cats,jax.random.randint(lk,(64,),0,n))
        np.testing.assert_array_equal(draws,jax.random.logistic(dk,(64,size,6),dtype='float64'))
        np.testing.assert_array_equal(labels,[mapping[c] for c in cats])
        again=design(key,size,64)
        for a,b in zip((cats,labels,draws),again): np.testing.assert_array_equal(a,b)
