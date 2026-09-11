"""GB-safe geometry tests; evidence reconstruction is unvalidated/out of scope."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns import dns
from examples.lisa_dns_stage4.proposals import build_direction, build_parameter_step


def test_frequencies_and_fixed_label_masks():
    direction,draw=build_direction(np.ones(54))
    keys=jax.random.split(jax.random.key(2026),100000)
    d,c,l=map(np.asarray,jax.jit(jax.vmap(draw))(keys))
    counts=np.bincount(c,minlength=3)
    np.testing.assert_allclose(counts/len(c),[.2,.4,.4],atol=.006)
    singles=l[c==1,0];sc=np.bincount(singles,minlength=9)
    assert np.max(abs(sc-len(singles)/9)) < 6*np.sqrt(len(singles)*(1/9)*(8/9))
    pairs,pc=np.unique(l[c==2],axis=0,return_counts=True)
    assert len(pairs)==36 and np.all(pairs[:,0]<pairs[:,1])
    assert np.max(abs(pc-len(l[c==2])/36)) < 6*np.sqrt(len(l[c==2])*(1/36)*(35/36))
    active=np.any(d.reshape(-1,9,6)!=0,axis=2)
    np.testing.assert_array_equal(active.sum(1),np.choose(c,[9,1,2]))
    for i in range(9):
        np.testing.assert_array_equal(active[:,i],(c==0)|(l[:,0]==i)|(l[:,1]==i))
    # Reverse current frequency ordering; identical key must give identical direction.
    x=jnp.arange(54.)
    np.testing.assert_array_equal(direction(keys[0],x),direction(keys[0],x[::-1]))
    np.testing.assert_allclose(direction(keys[0],x),jax.jit(direction)(keys[0],x),atol=1e-7)


@pytest.mark.parametrize('threshold',[-np.inf,-1.,-.1])
def test_kernel_caches_and_reproducibility(threshold):
    prior=lambda x:-jnp.sum(jax.nn.softplus(x)+jax.nn.softplus(-x))
    like=lambda x:-jnp.sum(x*x)
    step=build_parameter_step(prior,like,np.ones(54))
    x=jnp.zeros(54);p=dns.DNSParticleState(x,prior(x),like(x));key=jax.random.key(4)
    eager,ei=step(key,p,threshold);compiled,ci=jax.jit(step)(key,p,threshold)
    np.testing.assert_allclose(eager.position,compiled.position,atol=1e-7)
    def body(p,key):
        q,info=step(key,p,threshold)
        return q,(q,info)
    _,(ps,infos)=jax.jit(lambda p:jax.lax.scan(body,p,jax.random.split(key,100)))(p)
    assert np.all(np.isfinite(ps.logdensity))
    assert np.all(np.asarray(ps.loglikelihood)>threshold)
    np.testing.assert_allclose(ps.logdensity,jax.vmap(prior)(ps.position),atol=2e-5)
    np.testing.assert_allclose(ps.loglikelihood,jax.vmap(like)(ps.position),atol=1e-6)


def test_scales_are_copied_and_validated():
    s=np.ones(54);_,draw=build_direction(s);s[:]=10
    a=draw(jax.random.key(1))[0]
    np.testing.assert_allclose(np.linalg.norm(a),1,atol=1e-6)
    with pytest.raises(ValueError):build_direction(np.zeros(54))
