"""Toy-only checks of generic within-bank promotion; no LISA imports."""
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.ns import dns_levels
from examples import dns_pooled_promotion_validation as toy


def fixture():
    return toy.particles(np.array([[-1., 3.], [1., 4.], [.1, 5.], [-.1, 6.]]))


def test_strict_pool_and_zero_predecessor():
    h = fixture()
    np.testing.assert_array_equal(toy.eligible(h,-4.5),[0,2])
    with pytest.raises(ValueError,match='Predecessor'):
        toy.promote(jax.random.key(1),h,-4.5,'same_walker',2)
    s,idx=toy.promote(jax.random.key(1),h,-4.5,walkers=8)
    assert set(idx)<= {0,2}
    toy.check_states(s,-4.5)
    assert len(np.unique(s.position))<8  # repeated states are explicitly allowed


def test_uniform_replacement_and_determinism():
    h=fixture()
    key=jax.random.key(731)
    s,idx=toy.promote(key,h,-4.5,walkers=20000)
    expected=jax.random.choice(key,jnp.array([0,2]),(20000,),replace=True)
    np.testing.assert_array_equal(idx,expected)
    s2,idx2=toy.promote(key,h,-4.5,walkers=20000)
    np.testing.assert_array_equal(s.position,s2.position)
    np.testing.assert_array_equal(idx,idx2)
    assert abs(np.mean(idx==0)-.5)<.02


def test_bank_isolation_and_empty_bank_cannot_be_rescued():
    selection=fixture()
    calibration=toy.particles(np.array([[-.8,3.],[.8,4.]]))
    keys=jax.random.split(jax.random.key(12))
    a,_=toy.promote(keys[0],selection,-4.5)
    b,_=toy.promote(keys[1],calibration,-4.5)
    assert not np.any(np.isin(a.position,b.position))
    assert not np.shares_memory(np.asarray(a.position),np.asarray(b.position))
    empty=toy.particles(np.full((4,2),3.))
    with pytest.raises(ValueError,match='Entire bank'):
        toy.promote(keys[0],empty,-4.5)
    with pytest.raises(ValueError,match='No valid starts'):
        dns_levels._survivors(keys[0],empty,-4.5,8)


def test_missing_fixture_copies_own_saved_states():
    h=toy.particles(np.array([[-1.,1.],[3.,1.],[4.,1.]]))
    stress=toy.missing_predecessor(h,-4.5)
    assert np.all(np.asarray(stress.loglikelihood[:,0])<=-4.5)
    assert set(np.asarray(stress.position[:,0])) <= {3.,4.}
    np.testing.assert_array_equal(stress.position[:,1],h.position[:,1])
    np.testing.assert_array_equal(h.position[:,0],[-1.,3.,4.])
    toy.promote(jax.random.key(3),stress,-4.5)


def test_cache_corruption_detected():
    s=toy.particles(np.array([-1.,1.]))
    with pytest.raises(AssertionError):
        toy.check_states(s._replace(loglikelihood=jnp.ones(2)),-4.5)


def test_runner_matches_existing_stage3_sampler():
    config=dns_levels.ConstructionConfig(burn_in=4,thinning=2,construction_draws=8,
                                         calibration_draws=8,block_size=2)
    _,_,_,kernel=toy.problem('double_well')
    start=toy.particles(jnp.linspace(.2,1.8,8))
    key=jax.random.key(29)
    expected,_,failures=dns_levels._make_sampler(kernel,8,config)(key,start,-4.5)
    full=toy.make_runner(20)(key,start,-4.5)
    for a,b in zip(expected,full):
        np.testing.assert_array_equal(a,b[5::2])
    assert failures==0
    toy.check_states(full,-4.5)


def test_no_lisa_imports_or_calls():
    tree=ast.parse(inspect.getsource(toy))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert 'lisa' not in (node.module or '').lower()
        if isinstance(node,ast.Import):
            assert all('lisa' not in a.name.lower() for a in node.names)
        if isinstance(node,ast.Call):
            assert 'lisa' not in ast.unparse(node.func).lower()
