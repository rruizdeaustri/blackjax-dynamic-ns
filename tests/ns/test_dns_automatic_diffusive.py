"""Deterministic contract/oracle tests only: no stochastic trajectories."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.ns import dns, dns_automatic as a, dns_automatic_diffusive as d
from examples.dns_diffusive_ladder_development import Toy


def fixture_trace():
    # Top -> full prior -> other mode -> top, including repeated top events.
    indices=np.array([[2,2],[0,2],[0,2],[1,2],[2,2],[2,2]],dtype=np.int32)
    x=np.array([[-1.,1.],[-1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]])[...,None]
    return dict(position=x,logdensity=np.full(indices.shape,-np.log(16.)),loglikelihood=-x[...,0]**2,
                assigned_level=indices,previous_level=indices,proposed_level=indices,
                eligible=np.ones(indices.shape,bool),probability=np.ones(indices.shape),
                accepted=np.ones(indices.shape,bool),parameter_valid=np.ones(indices.shape,bool),
                parameter_accepted=np.ones(indices.shape,bool))


def test_collection_assigned_top_only_and_self_loops():
    trace=fixture_trace();history,times,visits=d.collect_top(trace,2,0,3)
    assert times[:,0].tolist()==[0,4,5]
    assert times[:,1].tolist()==[0,1,2]
    assert visits==[3,6]
    assert history.position[:,0,0].tolist()==[-1.,1.,1.]
    missing,times,visits=d.collect_top(trace,2,0,4)
    assert missing is None and len(times[0])==3


def test_witness_requires_lower_assigned_mode_change():
    trace=fixture_trace();witness=d.excursion_witnesses(trace,2,0,0)
    assert len(witness)==1 and witness[0]['walker']==0
    assert witness[0]['minimum_assigned_level']==0
    trace['assigned_level'][1:3,0]=2
    assert not d.excursion_witnesses(trace,2,0,0)


def test_weights_original_level_acceptance_and_conditional_cancellation():
    levels=dns.create_levels([-np.inf,-2.,-1.],[0.,-1.,-2.],[0.,0.,0.])
    eligible,p=dns.level_move_probability(levels,jnp.asarray(0.),jnp.asarray(2),jnp.asarray(1))
    assert bool(eligible) and np.isclose(float(p),np.exp(-1))
    _,p=dns.level_move_probability(levels,jnp.asarray(-1.5),jnp.asarray(1),jnp.asarray(2))
    assert float(p)==0
    prior=np.array([.2,.3,.5]);mask=np.array([0,1,1]);weight=7.;estimated_mass=.61
    joint=prior*mask*weight/estimated_mass
    assert np.allclose(joint/joint.sum(),prior*mask/(prior@mask))


def test_exact_interval_mass_cdf_moments():
    toy=Toy(dict(bound=8.,center=1.65,sigma=.38))
    prior=toy.truth(-np.inf,np.array([-8.,0.,8.]))
    assert prior['mass']==1 and np.allclose(prior['cdf'],[0,.5,1])
    assert np.isclose(prior['second_moment'],64/3)
    truth=toy.truth(-.5,np.array([-1.65,0.,1.65]))
    assert np.isclose(truth['mass'],4*.38/16)
    assert np.allclose(truth['cdf'],[.25,.5,.75])
    assert np.isclose(truth['second_moment'],1.65**2+.38**2/3)


def test_snapshot_exact_state_keys_collection_and_reject_changed_prefix(tmp_path):
    trace=fixture_trace();state=dns.DNSState(dns.DNSParticleState(*[jnp.asarray(trace[k][-1]) for k in a.ARRAYS]),jnp.array([2,2]))
    keys=jnp.array([[1,2],[3,4]],dtype=jnp.uint32)
    config=d.VisitConfig(0,10,3,5,3);prefix={'thresholds':[-np.inf,-3.,-2.],'log_masses':[0.,-1.,-2.]}
    streams={'bank':'selection','mh_seed':9};path=tmp_path/'state'
    d.save_snapshot(path,state,keys,trace,prefix,[0.,0.,0.],streams,config,2,9)
    restored,rkeys,rtrace=d.load_snapshot(path,prefix,[0.,0.,0.],streams,config,2)
    assert np.array_equal(rkeys,keys) and np.array_equal(restored.level_index,state.level_index)
    for k in trace:assert rtrace[k].tobytes()==trace[k].tobytes()
    with pytest.raises(ValueError,match='target/schedule'):
        d.load_snapshot(path,{'thresholds':[-np.inf,-3.,-1.],'log_masses':[0.,-1.,-2.]},[0.,0.,0.],streams,config,2)


def test_no_reconstruction_import_or_export():
    text=Path('blackjax/ns/dns_automatic_diffusive.py').read_text()
    assert 'dns_reconstruction' not in text
    assert 'reconstruction_samples_exported=False' in text
