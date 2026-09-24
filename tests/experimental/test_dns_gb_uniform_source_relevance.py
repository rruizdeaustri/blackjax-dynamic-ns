"""Cheap Stage-4S bookkeeping; synthetic saved arrays, no LISA evaluations."""
import os
import subprocess
import sys

import numpy as np
import pytest

from examples.lisa_dns_stage4.analyze_uniform_source_relevance import (
    ELL9, ranks_from_saved_gain, strata, touch_category, touch_probabilities,
)


def test_dominant_touch_category_assignment():
    assert touch_category(2,2)=='N'
    assert touch_category(1,2)=='A'
    assert touch_category(2,1)=='B'
    assert touch_category(1,1)=='AB'
    with pytest.raises(ValueError):touch_category(0,1)


def test_partition_is_mutually_exclusive_and_exhaustive():
    rows=[]
    for a in range(1,10):
        for b in range(1,10):
            rows.append(dict(top1_category=touch_category(a,b),joint_survives=True,
                             passes_A=True,passes_B=True,delta_logL_A=0.,delta_logL_B=0.,
                             minimum_side_margin=1.))
    groups=strata(rows,1)
    assert [groups[k]['n'] for k in ['N','A','B','AB']]==[64,8,8,1]
    assert sum(groups[k]['n'] for k in ['0','1','2'])==81
    assert groups['>=1']['n']==17


def test_top_k_membership_bookkeeping_and_ties():
    ranks=ranks_from_saved_gain([5,5,5,4,3,2,1,0,-1])
    np.testing.assert_array_equal(ranks,np.arange(1,10))
    for k in (1,2,3):
        categories=[touch_category(a,b,k) for a in ranks for b in ranks]
        assert categories.count('AB')==k*k
        assert categories.count('A')==categories.count('B')==k*(9-k)
        assert categories.count('N')==(9-k)**2
    with pytest.raises(ValueError):touch_category(1,1,4)


def test_combinatorial_probability_calculation():
    np.testing.assert_allclose(touch_probabilities(),np.array([64,16,1])/81)
    for k in (1,2,3):
        p=touch_probabilities(k)
        assert p.sum()==pytest.approx(1)
        np.testing.assert_allclose(p,np.array([(9-k)**2,2*k*(9-k),k*k])/81)


def test_saved_array_only_path_with_model_imports_forbidden(tmp_path):
    x=np.arange(108,dtype=float).reshape(2,1,54)/100
    gain=np.broadcast_to(np.arange(9,0,-1),(2,1,9)).copy()
    base=dict(position=x,physical=x.reshape(2,1,9,6),source_gain=gain,
              ranks=ranks_from_saved_gain(gain),dominant=np.zeros((2,1),int),
              indices=np.array([0]),loglikelihood=np.full((2,1),ELL9+10))
    labels=np.array([[1,1],[0,1],[1,0],[0,0]])
    positions=[]
    for b,c in labels:
        a=x[0,0].copy().reshape(9,6);z=x[1,0].copy().reshape(9,6)
        a[b]=x[1,0].reshape(9,6)[c];z[c]=x[0,0].reshape(9,6)[b]
        positions.append([a.reshape(54),z.reshape(54)])
    swap=dict(position=positions,i=np.zeros(4,int),j=np.ones(4,int),t=np.zeros(4,int),
              label_i=labels[:,0],label_j=labels[:,1],loglikelihood=np.full((4,2),ELL9+5),
              passes=np.ones((4,2),bool),joint_survives=np.ones(4,bool))
    np.savez(tmp_path/'base.npz',**base);np.savez(tmp_path/'swap.npz',**swap)
    script='''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('jax', 'blackjax', 'gbjax', 'jax_samplers')) or name.endswith(('.audit', '.dominant_source_exchange')):
        raise AssertionError('Model/kernel import forbidden: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import numpy as np
from examples.lisa_dns_stage4.analyze_uniform_source_relevance import load_arrays, analyze_saved_arrays
base=load_arrays(sys.argv[1]);swap=load_arrays(sys.argv[2])
rows=analyze_saved_arrays(base,swap,np.ones(6))
assert [r['top1_category'] for r in rows]==['N','A','B','AB']
assert all(r['joint_survives'] for r in rows)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'base.npz'),str(tmp_path/'swap.npz')],
                          env={**os.environ,'PYTHONPATH':os.getcwd()},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
