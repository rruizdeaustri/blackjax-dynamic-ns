"""Small automatic multimodal ladder using the frozen population sweep on a toy.

Usage: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m examples.dns_automatic_ladder
No LISA likelihood, production inference, evidence, or final termination logic.
"""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns_automatic as automatic
from blackjax.ns.dns import DNSParticleState
from examples.lisa_dns_stage4.automatic_ladder_integration import make_frozen_kernel


def prior(x):
    return -.5*jnp.sum(x*x)


def likelihood(x):
    # Two separated modes in the first coordinate; remaining coordinates are prior nuisances.
    return -.5*((jnp.abs(x[0])-1.5)/.5)**2


def kernel():
    return make_frozen_kernel(prior, likelihood, (.00183, .00185),
        model_id='54-dimensional-normal-prior-symmetric-double-well-v1')


def initial_state(seed=701):
    transition = kernel()
    banks = {}
    for bank, child in zip(automatic.BANKS, np.random.SeedSequence(seed).spawn(2)):
        x = np.random.Generator(np.random.PCG64(child)).normal(size=(8, 8, 54))
        lp, ll = transition.evaluate(x)
        banks[bank] = DNSParticleState(x, lp, ll)
    return automatic.LadderState((-np.inf,), (0.,), **banks, seed=seed,
        kernel_metadata=automatic.encoded(transition.metadata)), transition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/dns-automatic-toy'))
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume:
        transition = kernel()
        state = automatic.load_checkpoint(args.resume, transition)
    else:
        state, transition = initial_state()
    result = automatic.run_ladder(state, transition, args.output, max_new_levels=args.levels)
    report = dict(status=result.status, thresholds=result.thresholds, log_masses=result.log_masses,
        iteration=result.iteration, records=[json.loads(r) for r in result.records],
        mode_fractions={n: float(np.mean(getattr(result, n).position[..., 0]>0)) for n in automatic.BANKS},
        final_history_hashes={n: automatic.particle_hash(getattr(result, n)) for n in automatic.BANKS})
    args.output.mkdir(parents=True, exist_ok=True)
    automatic.write_json(args.output/'summary.json', report)
    print(json.dumps(automatic.json_value(report), indent=2))


if __name__ == '__main__':
    main()
