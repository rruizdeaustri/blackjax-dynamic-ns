"""Pure saved-array diagnostics; no real or toy model calls."""
import ast
import inspect
import subprocess
import sys
import numpy as np
import pytest
from examples.lisa_dns_stage4 import diagnose_population_level13 as d


def test_survivor_counts_strict_indices_blocks_and_runs():
    x=np.zeros(64);x[[0,2,31,32,63]]=1
    z=d.survivor_summary(x,0)
    assert (z['count'],z['fraction'],z['first'],z['last'])==(5,5/64,0,63)
    assert z['occupied_blocks']==2 and z['contiguous_survivor_runs']==4
    assert z['longest_zero_interval']==30
    np.testing.assert_array_equal(z['block_counts'],[3,2])


@pytest.mark.parametrize('value,expected',[(0,0),(1,32)])
def test_zero_survivors_and_all_survivors(value,expected):
    z=d.survivor_summary(np.full(32,value),0)
    assert z['count']==expected
    assert z['first']==(0 if expected else None)
    assert z['last']==(31 if expected else None)
    assert z['longest_zero_interval']==(0 if expected else 32)


def test_primary_and_burnin_are_separate():
    x=np.r_[np.ones(128),np.zeros(256)]
    assert d.survivor_summary(x[:128],0)['count']==128
    assert d.survivor_summary(x[128:],0)['count']==0


def test_leave_one_out_and_equal_halves():
    x=np.arange(8*256).reshape(8,256).astype(float);saved=x.copy()
    z=d.sensitivity(x)
    for w,row in enumerate(z['leave_one_out']):
        a=np.sort(np.delete(x,w,axis=0).ravel());k=int(np.floor(len(a)*(1-np.exp(-1))))
        assert row['value']==a[k] and row['n']==7*256 and row['diagnostic_only']
    a,b=d.halves(x)
    np.testing.assert_array_equal(a,x[:,:128]);np.testing.assert_array_equal(b,x[:,128:])
    np.testing.assert_array_equal(x,saved)
    assert z['first_half']['value']==d.empirical_candidate(a)['value']
    with pytest.raises(ValueError):d.halves(np.zeros((8,3)))


def test_pooled_feasibility_when_one_donor_has_no_survivors():
    pos=np.arange(8*32*3).reshape(8,32,3);ll=np.ones((8,32));ll[6]=0
    z=d.pool_feasibility(pos,ll,0)
    assert not z['A']['possible'] and list(z['A']['missing_walkers'])==[6]
    for name in ['B','C','D']:
        assert z[name]['possible'] and z[name]['eligible_unique_states']==224
        assert z[name]['eligible_donor_walkers']==7
        assert not z[name]['duplicate_state_required'] and z[name]['duplicate_donor_required']


def test_replacement_feasibility_does_not_draw_or_create_starts():
    pos=np.zeros((8,32,3));ll=np.ones((8,32));ll[1:]=0
    z=d.pool_feasibility(pos,ll,0)
    assert not z['B']['possible'] and not z['C']['possible']
    assert z['D']['possible'] and z['D']['minimum_repeated_slots']==7
    assert z['D']['duplicate_state_required']
    ll[:]=0
    assert not d.pool_feasibility(pos,ll,0)['D']['possible']


def test_output_only_diagnostics_no_checkpoint(tmp_path):
    out=tmp_path/'new'
    d.write_diagnostics(out,{'likelihood_calls':0,'sampling_calls':0})
    assert [p.name for p in out.iterdir()]==['diagnostics.json']
    with pytest.raises(FileExistsError):d.write_diagnostics(out,{})
    with pytest.raises(ValueError):d.write_diagnostics(d.B/'new',{})
    with pytest.raises(ValueError):d.write_diagnostics(d.A,{})


def test_no_model_sampling_or_promotion_imports_or_calls():
    source=inspect.getsource(d);tree=ast.parse(source)
    forbidden={'run_bank','load_problem','scalar_functions','freeze_level13','save_checkpoint',
               'promote','promoted_states','construct_levels','build_next_level','choice'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,'attr','')
            assert name not in forbidden
    code='from examples.lisa_dns_stage4 import diagnose_population_level13; import sys; assert not any(x.endswith(".audit") or x.startswith("jax_samplers") or x.startswith("blackjax") for x in sys.modules)'
    subprocess.run([sys.executable,'-c',code],check=True)


def test_geometry_uses_saved_implemented_mapping_not_nominal_bounds():
    from scipy.special import expit
    block=np.zeros((4,2,9,6));block[:,:,0,0]=np.linspace(-2,2,8).reshape(4,2)
    lo,hi=.0018300038051750386,.001850012683916793
    ex={'post_slice_position':block.reshape(4,2,54),'labels':np.zeros((4,2),int),
        'selected_frequency':lo+(hi-lo)*expit(block[:,:,0,0])}
    cfg={'band':{'f_min':0.,'f_max':1.},'priors':{'uniform':{n:[-1.,1.] for n in d.NAMES[1:]}}}
    lower,upper,_=d.physical_settings({'payload':{'provenance':{'config':cfg}}},ex)
    np.testing.assert_allclose([lower[0],upper[0]],[lo,hi],rtol=0,atol=1e-18)


def test_counterfactual_pair_distances_are_analytic_without_allocating_starts():
    pos=np.arange(8*2*54,dtype=float).reshape(8,2,54)/1000
    ll=np.full((8,2),d.CANDIDATE)
    ll[:7,0]=d.CANDIDATE+1;ll[0,1]=d.CANDIDATE+1
    lo=np.array([.00183,-1.,-1.,-np.pi,-np.pi,-1.])
    hi=np.array([.00185,1.,1.,np.pi,np.pi,1.])
    widths=hi-lo;widths[3]=np.pi;widths[4]=2*np.pi
    before=pos.copy();result=d.pool_geometry(pos,ll,(lo,hi,widths))
    assert result['eligible_unique']==8 and result['pair_count']==28
    assert result['duplicate_pair_probability_D']==1/8
    assert result['distances']['B']['frequency_set_RMS_bins']['min']>0
    assert result['distances']['D']['frequency_set_RMS_bins']['min']==0
    assert 'starts' not in result and 'position' not in result
    np.testing.assert_array_equal(pos,before)
