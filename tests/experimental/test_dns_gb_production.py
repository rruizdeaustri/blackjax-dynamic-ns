"""Cheap production bookkeeping tests; no real LISA trajectory."""
import jax.numpy as jnp
import numpy as np
from blackjax.ns import dns
from examples.lisa_dns_stage4.run_production_smoke import frozen_inputs,highest_eligible,complete_excursions,level_bookkeeping


def test_frozen_iid_ladder_uniform_weights_and_isotropic_scales():
    calibration={'ell1':-10.,'Xhat1':.4,'IID_level2':{'accepted':True,'selection':{'threshold':-5.},'log_X2':np.log(.15)}}
    levels,scales=frozen_inputs(calibration)
    np.testing.assert_allclose(levels.log_mass,[0.,np.log(.4),np.log(.15)],rtol=1e-6)
    np.testing.assert_array_equal(levels.log_weight,[0.,0.,0.])
    np.testing.assert_allclose(scales,np.pi/np.sqrt(3))
    assert not scales.flags.writeable


def test_initial_assigned_is_api_default_not_highest_eligible():
    levels=dns.create_levels([-np.inf,-10.,-5.],[0.,-1.,-2.],[0.,0.,0.])
    state=dns.init(jnp.zeros(2),lambda x:-jnp.sum(x*x),lambda x:jnp.asarray(-1.),levels)
    assert int(state.level_index)==0
    assert highest_eligible(levels,state.particle.loglikelihood)==2
    assert highest_eligible(levels,-5.)==1  # Strict contour.
    assert highest_eligible(levels,-10.)==0


def test_complete_excursions_require_top_bottom_and_return():
    levels=[0,1,2,2,1,0,0,1,2,1,2,1,0,1,2]
    assert complete_excursions(levels)==[{'start':3,'first_bottom':5,'end':8},{'start':10,'first_bottom':12,'end':14}]
    assert complete_excursions([0,1,2,1,2])==[]


def test_assigned_highest_eligibility_and_boundary_bookkeeping():
    assigned=[0,1,2,2,1,0,1,2]
    proposed=[1,2,3,1,0,1,2]
    result=level_bookkeeping(assigned,[2]*8,proposed,[True,True,False,True,True,True,True],[True,True,False,True,True,True,True])
    assert result['assigned_occupancy']==[1,3,3]
    assert result['highest_eligible_occupancy']==[0,0,7]
    assert result['directions']['up']=={'proposals':5,'eligible':4,'accepted':4}
    assert result['directions']['down']=={'proposals':2,'eligible':2,'accepted':2}
    assert result['low_assigned_while_highest_2']==4
    assert result['level0_while_highest_2']==1
    assert result['boundary_proposals']==1
    assert result['transition_matrix']==[[0,2,0],[1,0,2],[0,1,1]]
    assert len(result['complete_excursions'])==1
