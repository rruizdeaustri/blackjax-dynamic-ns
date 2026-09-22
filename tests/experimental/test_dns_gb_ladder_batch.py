"""Cheap checkpoint and stop semantics, without LISA evaluations."""
import json
import numpy as np
import pytest
from examples.lisa_dns_stage4.run_ladder_batch import (
    BASE_THRESHOLDS, BASE_MASSES, save_checkpoint, load_checkpoint, bounded_levels, digest)


def banks():
    return {name:dict(position=np.arange(432,dtype=float).reshape(8,54)+b*1000,
                     logprior=np.full(8,-100.),loglikelihood=np.full(8,-110000.))
            for b,name in enumerate(['selection','calibration'])}


def test_checkpoint_roundtrip(tmp_path):
    path=tmp_path/'level4.json';states=banks()
    save_checkpoint(path,BASE_THRESHOLDS,BASE_MASSES,states,[{'accepted':True}],{'source':'test'})
    t,m,s,p=load_checkpoint(path,(BASE_THRESHOLDS,BASE_MASSES))
    np.testing.assert_array_equal(t,BASE_THRESHOLDS)
    np.testing.assert_array_equal(m,BASE_MASSES)
    assert p['diagnostics']==[{'accepted':True}]
    for name in states:
        for key in states[name]: np.testing.assert_array_equal(s[name][key],states[name][key])


def test_resume_bank_separation(tmp_path):
    path=tmp_path/'level4.json';states=banks()
    states['calibration']=states['selection']
    with pytest.raises(ValueError,match='separate'):
        save_checkpoint(path,BASE_THRESHOLDS,BASE_MASSES,states,[],{})
    save_checkpoint(path,BASE_THRESHOLDS,BASE_MASSES,banks(),[],{})
    wrapper=json.loads(path.read_text())
    wrapper['payload']['states']['calibration']=wrapper['payload']['states']['selection']
    wrapper['sha256']=digest(wrapper['payload']);path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError,match='separate'): load_checkpoint(path)


def test_frozen_levels_cannot_mutate_on_resume(tmp_path):
    path=tmp_path/'level4.json'
    save_checkpoint(path,BASE_THRESHOLDS,BASE_MASSES,banks(),[],{})
    t,m,_,_=load_checkpoint(path)
    with pytest.raises(ValueError): t[2]=0
    with pytest.raises(ValueError): m[2]=0
    with pytest.raises(ValueError,match='overwritten'):
        save_checkpoint(path,BASE_THRESHOLDS,BASE_MASSES,banks(),[],{})
    wrapper=json.loads(path.read_text());wrapper['payload']['log_masses'][4]-=.1
    wrapper['sha256']=digest(wrapper['payload']);path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError,match='lower levels changed'):
        load_checkpoint(path,(BASE_THRESHOLDS,BASE_MASSES))


def test_first_failed_calibration_stops_without_retry():
    called=[]
    def attempt(level):
        called.append(level)
        return {'accepted':level<6,'calibration':{'ess':30 if level<6 else 19.9}}
    rows=list(bounded_levels(5,10,attempt))
    assert called==[5,6] and len(rows)==2
