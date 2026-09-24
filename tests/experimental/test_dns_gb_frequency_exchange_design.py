"""Cheap Stage-4T design algebra; no model, likelihood, or chain."""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit

from examples.lisa_dns_stage4.analyze_frequency_exchange_design import (
    ELL9, corrected_alpha, fixed_exchange, fixed_frequency_region,
    frequency_probabilities, selection_log_ratio,
)


def test_documented_region_has_fixed_center_and_half_width():
    assert fixed_frequency_region()==(0.0018409808,2.558e-7)


def test_frequency_weights_normalize_and_preserve_gaussian_ratios():
    center,sigma=fixed_frequency_region()
    p,lp,_=frequency_probabilities(center+sigma*np.arange(-4,5))
    assert p.sum()==pytest.approx(1.)
    assert np.argmax(p)==4
    assert lp[3]-lp[4]==pytest.approx(-.5)
    np.testing.assert_allclose(p,p[::-1],rtol=2e-12)


def test_all_probabilities_positive_in_extended_precision():
    p,lp,_=frequency_probabilities(np.linspace(.001830,.001850,9))
    assert np.isfinite(p).all() and (p>0).all() and np.isfinite(lp).all()
    assert p.min()<np.nextafter(0.,1.)
    with pytest.raises(ValueError):frequency_probabilities(np.full(9,np.nan))


def test_forward_reverse_ratio_and_normalizer_algebra():
    center,sigma=fixed_frequency_region()
    f=center+sigma*np.arange(-4,5);g=center+sigma*np.linspace(-2,3,9)
    i,j=2,7;ff=f.copy();gg=g.copy();ff[i]=g[j];gg[j]=f[i]
    p,lp,z=frequency_probabilities(f);q,lq,w=frequency_probabilities(g)
    pp,lpp,zz=frequency_probabilities(ff);qq,lqq,ww=frequency_probabilities(gg)
    ratio=selection_log_ratio(lp,lq,lpp,lqq,i,j)
    assert ratio==pytest.approx(np.log(pp[i]*qq[j]/(p[i]*q[j])))
    assert ratio==pytest.approx(z+w-zz-ww)
    assert selection_log_ratio(lpp,lqq,lp,lq,i,j)==pytest.approx(-ratio)


def test_reverse_swap_uses_same_slots_and_preserves_untouched_coordinates():
    x=np.arange(54,dtype=float);y=-x-100
    xx,yy=fixed_exchange(x,y,2,7)
    np.testing.assert_array_equal(xx.reshape(9,6)[2],y.reshape(9,6)[7])
    np.testing.assert_array_equal(yy.reshape(9,6)[7],x.reshape(9,6)[2])
    keep=np.arange(9)!=2
    np.testing.assert_array_equal(xx.reshape(9,6)[keep],x.reshape(9,6)[keep])
    xr,yr=fixed_exchange(xx,yy,2,7)
    assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()


def test_exact_mh_alpha_with_contour_and_prior_correction():
    np.testing.assert_allclose(corrected_alpha([2,0,-2],[True,False,True]),[1,0,np.exp(-2)])
    assert corrected_alpha(-2,True,.25)==pytest.approx(np.exp(-1.75))
    assert corrected_alpha(1000,False)==0
    assert corrected_alpha(1000,True)==1


def test_saved_array_only_execution_with_model_imports_forbidden(tmp_path):
    low,high=.001830,.001850
    x=np.linspace(-2,2,108).reshape(2,1,54)
    physical=np.broadcast_to(expit(x.reshape(2,1,9,6)),(2,1,9,6)).copy()
    physical[...,0]=low+(high-low)*physical[...,0]
    base=dict(position=x,physical=physical,source_gain=np.broadcast_to(np.arange(9,0,-1),(2,1,9)),
              ranks=np.broadcast_to(np.arange(1,10),(2,1,9)),dominant=np.zeros((2,1),int),
              indices=np.array([0]),logprior=np.zeros((2,1)))
    xx,yy=fixed_exchange(x[0,0],x[1,0],0,0)
    swap=dict(position=np.array([[xx,yy]]),i=np.array([0]),j=np.array([1]),t=np.array([0]),
              label_i=np.array([0]),label_j=np.array([0]),loglikelihood=np.full((1,2),ELL9+10),
              joint_survives=np.array([True]),logprior=np.zeros((1,2)),joint_log_prior_ratio=np.zeros(1))
    np.savez(tmp_path/'base.npz',**base);np.savez(tmp_path/'swap.npz',**swap)
    script='''
import builtins, sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('jax','blackjax','gbjax','jax_samplers')) or name.endswith(('.audit','.dominant_source_exchange')):
        raise AssertionError('Model/kernel import forbidden: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import numpy as np
from examples.lisa_dns_stage4.analyze_frequency_exchange_design import replay_saved
with np.load(sys.argv[1]) as f:base={k:f[k] for k in f.files}
with np.load(sys.argv[2]) as f:swap={k:f[k] for k in f.files}
r=replay_saved(base,swap,[.001830,.001850])
assert r['integrity']['structural_failures']==0
assert r['replay']['n']==1
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'base.npz'),str(tmp_path/'swap.npz')],
                          env={**os.environ,'PYTHONPATH':os.getcwd()},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
