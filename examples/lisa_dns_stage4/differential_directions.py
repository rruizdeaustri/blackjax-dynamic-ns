"""Frozen experimental empirical orientations; no change to production proposals."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.dns_kernels import build_constrained_slice_kernel
from examples.lisa_dns_stage4.proposals import build_direction

NORM=float(np.pi/np.sqrt(3))
ZERO_NORM=1e-12  # Fixed numerical degeneracy threshold, not an orientation tuning parameter.


class DifferentialInfo(NamedTuple):
    slice: object
    component: object
    labels: object
    reference_pair: object
    fallback: object
    direction_norm: object


def freeze_reference(reference,starts):
    reference=np.array(reference,dtype=np.float64,copy=True)
    starts=np.asarray(starts,dtype=np.float64)
    if reference.ndim!=2 or reference.shape[1]!=54 or len(reference)<2 or not np.all(np.isfinite(reference)):
        raise ValueError('Need at least two finite 54-coordinate reference states')
    if starts.ndim!=2 or starts.shape[1]!=54 or not np.all(np.isfinite(starts)):
        raise ValueError('Invalid evaluation starts')
    if {x.tobytes() for x in reference}&{x.tobytes() for x in starts}:
        raise ValueError('Reference and evaluation banks must be separate')
    reference.setflags(write=False)
    return reference


def ordered_pair(key,n):
    """Distinct random pair, explicitly symmetrized by an independent fair reversal.

    Ideal uniform integer draws give probability 1/[n(n-1)]. The extra fair
    reversal guarantees sign symmetry even with bounded-integer rounding bias.
    """
    ka,kb,ks=jax.random.split(key,3)
    a=jax.random.randint(ka,(),0,n)
    offset=jax.random.randint(kb,(),0,n-1)
    b=offset+(offset>=a)
    return orient_pair(a,b,jax.random.bernoulli(ks))


def orient_pair(a,b,reverse):
    return jnp.where(reverse,jnp.array([b,a]),jnp.array([a,b]))


def masked_orientation(a,b,component,labels,fallback):
    blocks=jnp.arange(9)
    mask=jnp.repeat((component==0)|(blocks==labels[0])|(blocks==labels[1]),6)
    delta=(a-b)*mask
    norm=jnp.linalg.norm(delta)
    use_fallback=norm<=ZERO_NORM
    direction=jnp.where(use_fallback,fallback,NORM*delta/jnp.maximum(norm,ZERO_NORM))
    return direction,use_fallback


def build_direction_generator(reference,starts):
    frozen=freeze_reference(reference,starts)
    bank=jnp.asarray(frozen.copy())  # Snapshot, independent of later host-array mutation.
    _,baseline_draw=build_direction(np.full(54,NORM))
    def draw(key):
        fallback,component,labels=baseline_draw(key)
        # Preserve baseline component/fallback RNG; new reference RNG has a fixed domain tag.
        pair=ordered_pair(jax.random.fold_in(key,0x4D1F),len(frozen))
        direction,used=masked_orientation(bank[pair[0]],bank[pair[1]],component,labels,fallback)
        return direction,component,labels,pair,used
    def direction(key,position):
        return draw(key)[0]  # Position deliberately never inspected.
    return direction,draw,frozen


def build_parameter_step(prior,likelihood,reference,starts):
    direction,draw,_=build_direction_generator(reference,starts)
    kernel=build_constrained_slice_kernel(prior,likelihood,generate_slice_direction_fn=direction,
                                         max_steps=10,max_shrinkage=100)
    def step(key,particle,threshold):
        result,info=kernel(key,particle,threshold)
        _,direction_key=jax.random.split(key)
        d,component,labels,pair,fallback=draw(direction_key)
        return result,DifferentialInfo(info,component,labels,pair,fallback,jnp.linalg.norm(d))
    return step
