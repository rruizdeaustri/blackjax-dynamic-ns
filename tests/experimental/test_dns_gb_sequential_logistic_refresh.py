"""Cheap sequential detailed-balance and state/cache contract checks."""
import jax
import numpy as np
import pytest
from examples.lisa_dns_stage4.sequential_logistic_refresh import (
    State, transition, random_schedule, validate)
from examples.lisa_dns_stage4.logistic_mh_refresh import ELL9, logq
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices


def prior(x): return logq(x)-.002*np.sum(x)**2

def like(x): return ELL9+10+np.tanh(np.sum(x))

def state(x): return State(x,prior(x),like(x))


@pytest.mark.parametrize('labels',[[4],[1,8]])
def test_mh_logratio_antisymmetry_and_accepted_caches(labels):
    old=state(np.linspace(-1,1,54));ix=block_indices(labels)
    new,info=transition(old,labels,np.ones((len(labels),6)),0.,prior,like)
    assert info['accepted'] and new is not old
    back,reverse=transition(new,labels,old.position[ix].reshape(-1,6),0.,prior,like)
    assert info['log_mh_ratio']==pytest.approx(-reverse['log_mh_ratio'],abs=1e-12)
    assert abs(info['log_mh_ratio'])>1e-3
    assert new.logprior==prior(new.position) and new.loglikelihood==like(new.position)
    np.testing.assert_array_equal(back.position,old.position)
    keep=np.setdiff1d(np.arange(54),ix)
    assert old.position[keep].tobytes()==new.position[keep].tobytes()
    validate(new,prior,like)


def test_contour_rejection_preserves_entire_state_and_caches():
    old=state(np.zeros(54));before=old.position.tobytes()
    result,info=transition(old,[0],np.ones((1,6)),.1,prior,lambda x:ELL9)
    assert not info['accepted'] and result is old
    assert result.position.tobytes()==before
    assert result.logprior==old.logprior and result.loglikelihood==old.loglikelihood


def test_mh_rejection_inside_contour_preserves_state():
    old=state(np.zeros(54))
    result,info=transition(old,[0],np.full((1,6),20.),.9,prior,like)
    assert info['passes_ell9'] and 0<info['alpha']<.9
    assert not info['accepted'] and result is old


def test_sequential_updates_after_acceptance_not_fixed_base():
    initial=state(np.zeros(54))
    current,first=transition(initial,[0],np.ones((1,6)),0.,prior,like)
    assert first['accepted']
    rejected,second=transition(current,[1],np.full((1,6),2.),.5,prior,lambda x:ELL9-1)
    assert rejected is current and not second['accepted']
    final,third=transition(rejected,[2],np.full((1,6),3.),0.,prior,like)
    assert third['accepted']
    np.testing.assert_array_equal(final.position[:6],np.ones(6))
    np.testing.assert_array_equal(final.position[6:12],np.zeros(6))
    np.testing.assert_array_equal(final.position[12:18],np.full(6,3.))
    assert np.all(initial.position==0)
    assert final.logprior==prior(final.position)


@pytest.mark.parametrize('size',[1,2])
def test_schedule_independent_of_state_and_separate_uniform_stream(size):
    key=jax.random.key(8600+size);_,uk=jax.random.split(key)
    first=random_schedule(key,size,32);second=random_schedule(key,size,32)
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(first[-1],jax.random.uniform(uk,(32,),dtype='float64'))
    assert np.all((first[0]>=0)&(first[0]<(9 if size==1 else 36)))
    if size==2:assert np.all(first[1][:,0]<first[1][:,1])


def test_invalid_draw_and_zero_density_reject_without_repair():
    old=state(np.zeros(54))
    def forbidden(x): raise AssertionError('Nonfinite draw must not call model')
    result,info=transition(old,[0],np.full((1,6),np.inf),0.,forbidden,forbidden)
    assert result is old and info['nonfinite_draw_coordinates']==6 and info['alpha']==0
    result,info=transition(old,[0],np.ones((1,6)),0.,lambda x:-np.inf,like)
    assert result is old and info['alpha']==0
