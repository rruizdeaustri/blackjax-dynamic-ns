"""Analytic truth cross-checks and a small dynamic automatic-ladder inference test."""
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from blackjax.ns import dns_automatic as a, dns_reconstruction as r
from examples.dns_inference_reconstruction import AnalyticModel, initial_state, inference_records, summarize

PARAMETERS={
 'gaussian':dict(sigma_prior=1.5,y=1.,sigma_likelihood=.7),
 'mixture':dict(sigma_prior=2.,component_weights=[.65,.35],centers=[-2.,2.2],widths=[.45,.7])}


@pytest.mark.parametrize('name',PARAMETERS)
def test_independent_closed_form_reference_against_quadrature(name):
    m=AnalyticModel(name,PARAMETERS[name]); truth=m.reference()
    def integrand(theta):return norm.pdf(theta,scale=m.sigma)*np.exp(m.log_likelihood(theta))
    z=quad(integrand,-np.inf,np.inf,epsabs=1e-12)[0]
    mean=quad(lambda x:x*integrand(x),-np.inf,np.inf,epsabs=1e-12)[0]/z
    second=quad(lambda x:x*x*integrand(x),-np.inf,np.inf,epsabs=1e-12)[0]/z
    assert abs(np.log(z)-truth['logZ'])<1e-10
    assert abs(mean-truth['mean'])<1e-10
    assert abs(second-mean*mean-truth['variance'])<1e-10
    np.testing.assert_allclose(m.cdf(truth['quantiles']),[.05,.16,.5,.84,.95],rtol=0,atol=1e-11)
    if name=='mixture':assert abs(truth['component_probabilities'][0]-.5)>.1


def test_small_dynamic_automatic_ladder_and_complete_records(tmp_path):
    # Structural smoke test, not the predeclared scientific-accuracy run.
    model=AnalyticModel('gaussian',PARAMETERS['gaussian']); kernel=model.kernel()
    config=a.LadderConfig(burn_in=32,retained=256,block_size=32)
    state=initial_state(model,kernel,17071,config)
    completed=a.run_ladder(state,kernel,tmp_path/'build',max_new_levels=1)
    assert completed.status=='ready'
    records=inference_records(completed,kernel,[tmp_path/'build/attempt_000000/checkpoint'],
                              17072,config,tmp_path/'infer','structural-smoke')
    restored=r.load_records(tmp_path/'infer/records')
    result,_=summarize(model,records); repeat,_=summarize(model,restored)
    assert a.encoded(result)==a.encoded(repeat)
    assert result['weight_sum']==pytest.approx(1.,abs=1e-12)
    assert records.metadata['sampling_target']==r.TARGET
    assert len(records.loglikelihood)==2*256*8
    r.validate_caches(records,kernel.evaluate,completed.contract)
