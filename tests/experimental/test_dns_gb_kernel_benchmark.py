"""Cheap frozen geometry checks; no real LISA sampling."""
import jax
import numpy as np
import pytest
from examples.lisa_dns_stage4.run_kernel_benchmark import frozen_scales, separate_banks, COMMON_SCALE, autocorrelation_information
from examples.lisa_dns_stage4.proposals import build_direction


def test_isotropic_scale_has_fixed_direction_norm():
    scales=frozen_scales('isotropic')
    np.testing.assert_array_equal(scales,np.full(54,COMMON_SCALE))
    _,draw=build_direction(scales)
    directions,_,_=jax.vmap(draw)(jax.random.split(jax.random.key(4),30))
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),COMMON_SCALE,rtol=1e-6)


def test_iid_relative_scales_are_normalized_and_frozen():
    samples=np.random.default_rng(2).normal(size=(512,54))*np.arange(1,55)
    expected=np.std(samples,axis=0,ddof=1);expected*=COMMON_SCALE/np.sqrt(np.mean(expected**2))
    scales=frozen_scales('iid_relative',tuning=samples)
    np.testing.assert_allclose(scales,expected)
    samples[:]=0
    np.testing.assert_allclose(scales,expected)
    with pytest.raises(ValueError): scales[0]=5


def test_evaluation_bank_cannot_tune_scales_or_overlap_starts():
    rng=np.random.default_rng(3)
    tuning,evaluation,starts=rng.normal(size=(512,54)),rng.normal(size=(512,54)),rng.normal(size=(8,54))
    before=frozen_scales('iid_relative',tuning=tuning)
    info=separate_banks(tuning,evaluation,starts)
    assert info['shared_rows']==0 and info['tuning_sha256']!=info['evaluation_sha256']
    evaluation[:]*=100
    np.testing.assert_array_equal(before,frozen_scales('iid_relative',tuning=tuning))
    with pytest.raises(ValueError): separate_banks(tuning,tuning,starts)
    with pytest.raises(ValueError): separate_banks(tuning,evaluation,tuning[:8])


def test_no_online_scale_mutation():
    original=np.ones(54)
    scales=frozen_scales('historical',historical=original)
    _,draw=build_direction(scales)
    key=jax.random.key(6);before=np.asarray(draw(key)[0])
    original[:]=100
    for k in jax.random.split(key,4): draw(k)
    np.testing.assert_array_equal(before,draw(key)[0])
    np.testing.assert_array_equal(scales,np.ones(54))


def test_autocorrelation_distinguishes_persistence_and_constants():
    rng=np.random.default_rng(9);noise=rng.normal(size=1024)
    correlated=np.empty(1024);correlated[0]=noise[0]
    for i in range(1,1024): correlated[i]=.95*correlated[i-1]+noise[i]
    assert autocorrelation_information(correlated)['ess'] < autocorrelation_information(noise)['ess']/5
    assert autocorrelation_information(np.ones(256))['ess']==0


def test_historical_rms_normalization_preserves_all_relative_scales():
    historical=np.geomspace(.00049,2.2,54)
    frozen=frozen_scales('historical_rms_normalized',historical=historical)
    np.testing.assert_allclose(np.sqrt(np.mean(frozen**2)),COMMON_SCALE,rtol=2e-15)
    np.testing.assert_allclose(frozen[:,None]/frozen[None,:],
                               historical[:,None]/historical[None,:],rtol=2e-15)
    np.testing.assert_array_equal(historical,np.geomspace(.00049,2.2,54))
    assert not frozen.flags.writeable
