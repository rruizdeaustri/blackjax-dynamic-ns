"""Cheap composition/invariance and accounting tests, no LISA trajectories."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.hybrid_logistic_refresh import (
    hybrid_sweep, build_frozen_slice, sweep_cost, cost_matched_prefix)
from examples.lisa_dns_stage4.sequential_logistic_refresh import State, transition, validate
from examples.lisa_dns_stage4.logistic_mh_refresh import ELL9, logq


def test_composition_preserves_target_without_product_reversibility():
    p=np.array([.2,.3,.5])
    def mh_matrix(q):
        k=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                if i!=j:k[i,j]=q[j]*min(1,p[j]*q[i]/(p[i]*q[j]))
            k[i,i]=1-k[i].sum()
        return k
    s=mh_matrix(np.array([.6,.3,.1]));r=mh_matrix(np.array([.1,.1,.8]))
    for k in [s,r]:np.testing.assert_allclose(p@k,p,rtol=0,atol=1e-15)
    product=s@r
    np.testing.assert_allclose(p@product,p,rtol=0,atol=1e-15)
    flow=p[:,None]*product
    assert not np.allclose(flow,flow.T)  # invariance does not imply reversibility


@pytest.mark.parametrize('accept',[False,True])
def test_order_state_transfer_both_caches_and_refresh_outcome(accept):
    calls=[]
    prior=logq;like=lambda x: ELL9+10+np.tanh(x.sum())
    initial=State(np.zeros(54),prior(np.zeros(54)),like(np.zeros(54)))
    post_position=np.zeros(54);post_position[:6]=2.
    post=State(post_position,prior(post_position),like(post_position))
    def slice_step(x):
        calls.append('slice');assert x is initial
        return post,'slice_info'
    def refresh_step(x):
        calls.append('refresh');assert x is post
        # Rejection proposal is outside contour, leaving the post-slice caches intact.
        return transition(x,[1],np.ones((1,6)),0.,prior,like if accept else lambda x: ELL9)
    def check(x):calls.append('check');validate(x,prior,like)
    final,intermediate,info,refresh=hybrid_sweep(initial,slice_step,refresh_step,check)
    assert calls==['slice','check','refresh','check'] and intermediate is post
    assert info=='slice_info' and refresh['accepted']==accept
    np.testing.assert_array_equal(final.position[:6],np.full(6,2.))
    assert np.all(initial.position==0)
    if accept:
        assert final is not post
        np.testing.assert_array_equal(final.position[6:12],np.ones(6))
        assert final.logprior==prior(final.position) and final.loglikelihood==like(final.position)
    else:assert final is post


def test_structural_failure_after_slice_stops_before_refresh():
    def bad_check(x):raise ValueError('cache failure')
    def forbidden(x):raise AssertionError('refresh must not execute')
    with pytest.raises(ValueError,match='cache failure'):
        hybrid_sweep('old',lambda x:('slice',None),forbidden,bad_check)


def test_frozen_slice_builder_uses_unchanged_validated_factory(monkeypatch):
    from examples.lisa_dns_stage4 import proposals
    seen={}
    def factory(prior,like,scales,**kwargs):
        seen.update(prior=prior,like=like,scales=scales,**kwargs)
        return 'unchanged_factory'
    monkeypatch.setattr(proposals,'build_parameter_step',factory)
    assert build_frozen_slice('prior','like')=='unchanged_factory'
    assert seen['prior']=='prior' and seen['like']=='like'
    assert seen['max_steps']==10 and seen['max_shrinkage']==100
    np.testing.assert_array_equal(seen['scales'],np.full(54,np.pi/np.sqrt(3)))
    assert not seen['scales'].flags.writeable
    with pytest.raises(ValueError):seen['scales'][0]=42


def test_cost_accounting_two_checks_and_optional_nonfinite_rejection():
    c=sweep_cost(5,3,True)
    assert c['slice_likelihood_proxy']==8 and c['slice_prior_proxy']==8
    assert c['refresh_likelihood']==c['refresh_prior']==1
    assert c['cache_likelihood']==c['cache_prior']==2
    assert c['likelihood_total_proxy']==c['prior_total_proxy']==11
    assert sweep_cost(5,3,False)['likelihood_total_proxy']==10
    with pytest.raises(ValueError):sweep_cost(-1,0)


def test_cost_matched_prefix_exact_boundary_and_largest_common_prefix():
    costs=np.array([[2,3,7],[3,2,6]])
    assert cost_matched_prefix(costs,12,2)==dict(sweeps_per_walker=2,cost=12.,next_cost=25.,budget=12.)
    assert cost_matched_prefix(costs,11,2)['sweeps_per_walker']==1
    assert cost_matched_prefix(costs,100,2)['sweeps_per_walker']==3
    assert cost_matched_prefix(costs,1,2)['sweeps_per_walker']==0
    with pytest.raises(ValueError):cost_matched_prefix(np.array([[np.nan]]),10)
