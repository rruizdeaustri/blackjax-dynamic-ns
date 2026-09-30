"""NumPy-only cache tests against recorded CPU evaluations; no model execution."""
import ast
import inspect
from pathlib import Path
import numpy as np
import pytest
from examples.lisa_dns_stage4 import numerical_cache_contract as c

DATA=Path(__file__).parent/'data/dns_lisa_cpu_cache_audit.npz'
ELL12=-108187.99712765554


def values():
    with np.load(DATA,allow_pickle=False) as f:
        return dict(position=f['position'].copy(),reference_position=f['position'].copy(),
                    cached_prior=f['logprior'].copy(),cached_likelihood=f['loglikelihood'].copy(),
                    recomputed_prior=f['recomputed_prior'].copy(),
                    recomputed_likelihood=f['recomputed_likelihood'].copy(),
                    threshold=ELL12,backend='cpu',x64=True)


def test_recorded_historical_cpu_starts_pass_without_model_calls():
    v=values();assert v['position'].shape==(16,54)
    c.validate_cache(**v)


def test_corrupted_coordinates_fail_even_with_unchanged_caches():
    v=values();v['position'][0,0]+=1e-4
    with pytest.raises(ValueError,match='Copied coordinates'):c.validate_cache(**v)


@pytest.mark.parametrize('field,delta,match',[
    ('cached_prior',1e-6,'Prior cache'),('cached_likelihood',1e-3,'Likelihood cache')])
def test_corrupted_cache_fails(field,delta,match):
    v=values();v[field][0]+=delta
    with pytest.raises(ValueError,match=match):c.validate_cache(**v)


@pytest.mark.parametrize('field',['position','cached_prior','cached_likelihood','recomputed_prior','recomputed_likelihood'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_rejected(field,bad):
    v=values();v[field].flat[0]=bad
    with pytest.raises(ValueError,match='Nonfinite'):c.validate_cache(**v)


@pytest.mark.parametrize('field',['cached_likelihood','recomputed_likelihood'])
def test_strict_contour_equality_fails(field):
    v=values();v[field][0]=ELL12
    with pytest.raises(ValueError,match='Strict contour'):c.validate_cache(**v)


def test_numerical_not_bitwise_recomputation_policy():
    v=values()
    v['recomputed_prior']=v['cached_prior']+c.PRIOR_ATOL/2
    v['recomputed_likelihood']=v['cached_likelihood']+c.LIKELIHOOD_ATOL/2
    c.validate_cache(**v)
    assert not np.array_equal(v['recomputed_prior'],v['cached_prior'])


@pytest.mark.parametrize('backend,x64',[('gpu',True),('cuda',True),('cpu',False)])
def test_cpu_x64_only(backend,x64):
    v=values();v.update(backend=backend,x64=x64)
    with pytest.raises(ValueError,match='CPU'):c.validate_cache(**v)


def test_float32_not_permitted():
    v=values();v['cached_prior']=v['cached_prior'].astype('float32')
    with pytest.raises(ValueError,match='float64'):c.validate_cache(**v)


def test_policy_matches_all_seven_historical_runners():
    modules=['population_level10_construction','level10_population_validation',
             'population_level11_construction','level11_population_validation',
             'population_level12_construction','level12_population_validation','population_level13_construction']
    for name in modules:
        text=(Path('examples/lisa_dns_stage4')/(name+'.py')).read_text()
        assert 'np.testing.assert_allclose(ap,ps,rtol=0,atol=1e-9)' in text
        assert 'np.testing.assert_allclose(al,ls,rtol=0,atol=1e-7)' in text
    assert (c.PRIOR_ATOL,c.LIKELIHOOD_ATOL,c.RTOL)==(1e-9,1e-7,0)


def test_contract_has_no_jax_model_or_sampling_dependencies():
    tree=ast.parse(inspect.getsource(c))
    imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert imports==['numpy']
    assert not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree))
    assert set(n.name for n in tree.body if isinstance(n,ast.FunctionDef))=={'validate_cache'}
