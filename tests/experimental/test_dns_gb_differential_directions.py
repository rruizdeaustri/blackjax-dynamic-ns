"""Cheap frozen-orientation invariance checks; no real target sampling."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.lisa_dns_stage4.differential_directions import (
    freeze_reference,ordered_pair,orient_pair,masked_orientation,build_direction_generator,NORM)
from examples.lisa_dns_stage4.proposals import build_direction


def test_frozen_snapshot_and_separate_banks():
    bank=np.arange(5*54,dtype=float).reshape(5,54);starts=-np.ones((8,54))
    direction,draw,frozen=build_direction_generator(bank,starts)
    expected=np.asarray(direction(jax.random.key(1),jnp.zeros(54)))
    bank[:]=0
    assert np.any(frozen) and not frozen.flags.writeable
    np.testing.assert_array_equal(direction(jax.random.key(1),jnp.zeros(54)),expected)
    with pytest.raises(ValueError): frozen[0,0]=2
    with pytest.raises(ValueError,match='separate'): freeze_reference(starts,starts)


def test_uniform_ordered_pairs_distinct_and_sign_reversal():
    # Offset mapping is bijective over all n(n-1) ordered pairs, including reversals.
    n=7
    pairs=[(a,o+(o>=a)) for a in range(n) for o in range(n-1)]
    assert len(set(pairs))==n*(n-1) and all(a!=b and (b,a) in pairs for a,b in pairs)
    for a,b in pairs:
        np.testing.assert_array_equal(orient_pair(a,b,False),orient_pair(a,b,True)[::-1])
    actual=np.asarray(jax.vmap(lambda k:ordered_pair(k,n))(jax.random.split(jax.random.key(2),100)))
    assert np.all(actual[:,0]!=actual[:,1])
    a=jnp.arange(54,dtype=float);b=a[::-1]
    for component,labels in [(0,[-1,-1]),(1,[3,-1]),(2,[2,8])]:
        d,_=masked_orientation(a,b,component,jnp.array(labels),jnp.ones(54))
        opposite,_=masked_orientation(b,a,component,jnp.array(labels),jnp.ones(54))
        np.testing.assert_array_equal(d,-opposite)
        np.testing.assert_allclose(jnp.linalg.norm(d),NORM,rtol=1e-6)


def test_same_component_masks_norm_and_position_independence():
    bank=np.random.default_rng(5).normal(size=(20,54))
    direction,draw,_=build_direction_generator(bank,np.full((8,54),100.))
    _,baseline=build_direction(np.full(54,NORM))
    for k in jax.random.split(jax.random.key(3),60):
        d,c,labels,pair,fallback=draw(k)
        _,bc,bl=baseline(k)
        np.testing.assert_array_equal(c,bc);np.testing.assert_array_equal(labels,bl)
        assert not fallback and pair[0]!=pair[1]
        mask=(int(c)==0)|np.isin(np.arange(9),np.asarray(labels))
        np.testing.assert_array_equal(np.asarray(d).reshape(9,6)[~mask],0)
        assert mask.sum()==[9,1,2][int(c)]
        np.testing.assert_allclose(np.linalg.norm(d),NORM,rtol=1e-6)
        np.testing.assert_array_equal(direction(k,jnp.zeros(54)),direction(k,jnp.full(54,jnp.nan)))


def test_zero_difference_fallback_is_reproducible_baseline_direction():
    bank=np.zeros((3,54))
    _,draw,_=build_direction_generator(bank,np.ones((8,54)))
    _,baseline=build_direction(np.full(54,NORM))
    for k in jax.random.split(jax.random.key(4),10):
        d,c,labels,_,used=draw(k)
        expected,_,_=baseline(k)
        assert used
        np.testing.assert_array_equal(d,expected)
        np.testing.assert_array_equal(draw(k)[0],d)
        np.testing.assert_allclose(np.linalg.norm(d),NORM,rtol=1e-6)
