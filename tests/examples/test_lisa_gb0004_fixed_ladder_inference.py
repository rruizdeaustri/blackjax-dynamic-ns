"""Deterministic Stage-10A orchestration checks, without model calls or sampling."""
import importlib.util
from pathlib import Path
import numpy as np
from blackjax.ns import dns_reconstruction as r
spec=importlib.util.spec_from_file_location('stage10a',Path(__file__).parents[2]/'examples/lisa_gb0004_fixed_ladder_inference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(name,shift=0.):
    n=16
    return r.ReconstructionRecords(np.array([-np.inf,-1.]),np.array([0.,-.5]),
        np.arange(n,dtype=np.float64).reshape(n,1),np.full(n,-2.),np.linspace(-.9,-.1,n)+shift,
        np.repeat([0,1],8),np.tile(np.arange(2),8),np.tile(np.repeat(np.arange(4),2),2),
        np.array([name+str(i) for i in range(n)]),
        {'sampling_target':r.TARGET,'mass_status':['prior','calibrated'],
         'levels':[{'level':j,'threshold':t,'stream_id':name+str(j),'streams':{'namespace':name}}
                   for j,t in enumerate(['-inf',-1.])]}).validate()


def test_pool_unique_namespace_preserves_unchanged_mis():
    a,b=fixture('R1'),fixture('R2',.01)
    pooled=m.pool_records(a,b)
    assert len(set(zip(pooled.origin_level,pooled.walker,pooled.draw)))==32
    ea,eb,ep=[r.reconstruct_evidence(x) for x in (a,b,pooled)]
    assert np.isclose(ep['logZ'],np.logaddexp(ea['logZ'],eb['logZ'])-np.log(2))
    assert np.allclose(ep['weights'][:16],ea['weights']*np.exp(ea['logZ'])/(np.exp(ea['logZ'])+np.exp(eb['logZ'])))
    assert pooled.metadata['levels'][0]['streams_by_set']['R2']['namespace']=='R2'


def test_driver_does_not_call_construction_or_termination():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    banned={'run_ladder','build_next_level','construct_levels','calibrate_level_masses','should_stop'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls & banned


def test_frozen_qualitative_criteria_not_replaced_by_toy_tolerances():
    source=Path(m.__file__).read_text()
    assert 'PENDING_FROZEN_QUALITATIVE_SCIENTIFIC_REVIEW' in source
    assert 'accuracy_checks' not in source
