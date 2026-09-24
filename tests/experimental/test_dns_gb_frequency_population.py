"""Cheap Stage-4U schedule, proposal, MH and bookkeeping tests; no LISA."""
from itertools import combinations
import numpy as np
import pytest
from examples.lisa_dns_stage4.frequency_population_benchmark import (
    ELL9, SLICE_SEEDS, LABEL_SEED, MH_SEED, round_robin, log_probabilities,
    draw_label, joint_decision, commit_pair, exchange_tags, slice_blocks,
    prefix_from_cost, fixed_frequency_region, fixed_exchange,
)


def test_seven_rounds_four_disjoint_pairs_each():
    schedule=round_robin()
    assert schedule.shape==(7,4,2)
    for pairs in schedule:
        assert sorted(pairs.ravel())==list(range(8))
        assert np.all(pairs[:,0]<pairs[:,1])


def test_round_robin_cycle_covers_every_unordered_pair_once():
    pairs=[tuple(p) for r in round_robin() for p in r]
    assert len(set(pairs))==28
    assert set(pairs)==set(combinations(range(8),2))
    counts={p:0 for p in pairs}
    for t in range(512):
        for p in round_robin()[t%7]:counts[tuple(p)]+=1
    assert sorted(counts.values())==[73]*24+[74]*4


def test_frequency_log_probabilities_stable_without_clipping():
    x=np.linspace(-15,15,54)
    center,_=fixed_frequency_region()
    z=(center-.001830)/(.001850-.001830)
    x[24]=np.log(z/(1-z))
    lp,f=log_probabilities(x,(.001830,.001850))
    assert np.isfinite(lp).all() and lp.min()<-745
    p=np.exp(lp.astype(np.longdouble))
    assert np.all(p>0) and p.sum()==pytest.approx(1.)
    center,sigma=fixed_frequency_region()
    w=-.5*((f-center)/sigma)**2
    np.testing.assert_allclose(lp-lp[0],w-w[0],atol=1e-12)


def test_label_draw_uses_all_logits_and_fixed_gumbel_stream():
    lp=np.array([0.,-1,-2,-3,-4,-5,-6,-7,-900])
    g=np.zeros(9);g[8]=1000
    assert draw_label(lp,g)==8
    a=np.random.Generator(np.random.PCG64(LABEL_SEED)).gumbel(size=(20,9))
    b=np.random.Generator(np.random.PCG64(LABEL_SEED)).gumbel(size=(20,9))
    np.testing.assert_array_equal(a,b)
    assert [draw_label(lp,z) for z in a]==[draw_label(lp,z) for z in b]
    assert LABEL_SEED!=MH_SEED


def test_reciprocal_swap_and_involution_preserve_untouched():
    x=np.arange(54.);y=-x-100
    xx,yy=fixed_exchange(x,y,2,6)
    assert xx[12:18].tobytes()==y[36:42].tobytes()
    assert yy[36:42].tobytes()==x[12:18].tobytes()
    keep=np.r_[0:12,18:54]
    np.testing.assert_array_equal(xx[keep],x[keep])
    xr,yr=fixed_exchange(xx,yy,2,6)
    assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()


def test_reverse_probabilities_use_same_slot_on_proposed_catalogues():
    x=np.linspace(-2,2,54);y=np.linspace(3,-3,54)
    xx,yy=fixed_exchange(x,y,1,8)
    f,_=log_probabilities(np.array([x,y]),(.001830,.001850))
    r,_=log_probabilities(np.array([xx,yy]),(.001830,.001850))
    d=joint_decision([0,0],[0,0],f,r,[1,8],[ELL9+1,ELL9+2],.5)
    assert d['log_q_forward']==f[0,1]+f[1,8]
    assert d['log_q_reverse']==r[0,1]+r[1,8]
    rev=joint_decision([0,0],[0,0],r,f,[1,8],[ELL9+1,ELL9+2],.5)
    assert rev['log_r']==pytest.approx(-d['log_r'])


def test_full_joint_prior_and_selection_mh_ratio():
    f=np.zeros((2,9));r=np.zeros((2,9));r[0,2]=-.3;r[1,5]=-.4
    d=joint_decision([-10,-20],[-11,-19.5],f,r,[2,5],[ELL9+1,ELL9+1],.1)
    assert d['delta_logprior']==-.5
    assert d['log_r']==pytest.approx(-1.2)
    assert d['alpha']==pytest.approx(np.exp(-1.2)) and d['accepted']


def test_one_joint_uniform_and_strict_contour_semantics():
    f=np.zeros((2,9))
    def d(ll,u):return joint_decision([0,0],[-1,0],f,f,[0,0],ll,u)
    assert d([ELL9+1,ELL9+2],.1)['accepted']
    assert not d([ELL9+1,ELL9+2],.9)['accepted']
    assert not d([ELL9,ELL9+2],0.)['accepted']
    assert not d([ELL9+1,-np.inf],0.)['accepted']
    old=(np.zeros((2,54)),np.array([1.,2.]),np.array([3.,4.]))
    new=(np.ones((2,54)),np.array([5.,6.]),np.array([7.,8.]))
    for accept,wanted in [(True,new),(False,old)]:
        for result,expected in zip(commit_pair(old,new,accept),wanted):
            assert result.tobytes()==expected.tobytes()


def test_provenance_moves_only_on_joint_acceptance():
    tags=np.arange(72).reshape(8,9)
    changed=exchange_tags(tags,0,7,2,5,True)
    assert changed[0,2]==tags[7,5] and changed[7,5]==tags[0,2]
    np.testing.assert_array_equal(np.sort(changed.ravel()),np.arange(72))
    np.testing.assert_array_equal(exchange_tags(tags,0,7,2,5,False),tags)
    np.testing.assert_array_equal(exchange_tags(changed,0,7,2,5,True),tags)
    np.testing.assert_array_equal(slice_blocks(0,[-1,-1]),np.arange(9))
    np.testing.assert_array_equal(slice_blocks(1,[4,-1]),[4])
    np.testing.assert_array_equal(slice_blocks(2,[3,6]),[3,6])


def test_cost_prefix_depends_only_on_cumulative_complete_sweeps():
    p=prefix_from_cost([10,20,30],38,8)
    assert p==dict(sweeps=2,cost=38.,next_cost=68.,budget=38.)
    assert prefix_from_cost([10,20,30],37,8)['sweeps']==1
    assert prefix_from_cost([10,20],8,8)['sweeps']==0
    assert prefix_from_cost([10,20],100,8)['next_cost'] is None
    with pytest.raises(ValueError):prefix_from_cost([1,-2],100)


def test_frozen_selector_and_slice_random_streams():
    assert fixed_frequency_region()==(.0018409808,2.558e-7)
    assert SLICE_SEEDS==list(range(810100,810108))
