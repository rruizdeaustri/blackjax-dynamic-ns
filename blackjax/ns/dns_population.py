"""Fixed reciprocal population-sweep adapter for automatic DNS construction.

The transition is supplied as a frozen slice/selector/exchange bundle. This
module contains no likelihood model, level-specific rule, or tuning machinery.
"""
from dataclasses import dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns.dns import DNSParticleState
from blackjax.ns.dns_automatic import particles_copy


def round_robin():
    ring = list(range(8))
    rounds = []
    for _ in range(7):
        rounds.append(sorted(tuple(sorted((ring[k], ring[-1-k]))) for k in range(4)))
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return np.asarray(rounds, dtype=int)


def joint_decision(old_prior,new_prior,forward_logp,reverse_logp,labels,new_logL,uniform,threshold):
    assert 0<=uniform<1
    i,j=labels
    prior_delta=float(np.sum(new_prior)-np.sum(old_prior))
    qf=float(forward_logp[0,i]+forward_logp[1,j])
    qr=float(reverse_logp[0,i]+reverse_logp[1,j])
    selection=qr-qf;ratio=prior_delta+selection
    finite=bool(np.isfinite(new_prior).all() and np.isfinite(new_logL).all() and np.isfinite(ratio))
    contour=bool(np.isfinite(new_logL).all() and np.all(np.asarray(new_logL)>threshold))
    alpha=float(np.exp(min(0.,ratio))) if finite and contour else 0.
    logu=float(np.log(uniform)) if uniform>0 else -np.inf
    accepted=bool(finite and contour and logu<min(0.,ratio))
    return dict(delta_logprior=prior_delta,log_q_forward=qf,log_q_reverse=qr,
                log_r_selection=selection,log_r=ratio,alpha=alpha,uniform=uniform,
                joint_contour=contour,accepted=accepted,nonfinite_prior=int((~np.isfinite(new_prior)).sum()),
                nonfinite_likelihood=int((~np.isfinite(new_logL)).sum()))


@dataclass(frozen=True)
class FrozenPopulationKernel:
    """No tunable settings: callers provide a versioned, frozen operation bundle.

    The model adapter owns the slice-mixture implementation and source-block
    mapping; metadata must identify those sources and the model itself.
    """
    slice_step: object
    evaluator: object
    selector: object
    exchange: object
    slice_blocks: object
    identity: bytes

    @property
    def metadata(self):
        return json.loads(self.identity)

    def evaluate(self, position):
        return self.evaluator(position)

    def exchange_sweep(self, sliced, threshold, sweep, gumbels, uniforms, contract):
        """Four disjoint proposals; exact block-copy/commit checks and cache checks."""
        sx, sp, sl = map(np.asarray, sliced)
        forward = self.selector(sx)
        if forward.shape != (8, 9) or not np.isfinite(forward).all():
            raise ValueError("Finite selector log probabilities required")
        pairs = round_robin()[sweep % 7]
        proposed = sx.copy()
        labels = []
        for slot, (a, b) in enumerate(pairs):
            i = int(np.argmax(forward[a]+gumbels[slot, 0]))
            j = int(np.argmax(forward[b]+gumbels[slot, 1]))
            labels.append((i, j))
            proposed[a], proposed[b] = self.exchange(sx[a], sx[b], i, j)
            xx, yy = self.exchange(proposed[a], proposed[b], i, j)
            if xx.tobytes() != sx[a].tobytes() or yy.tobytes() != sx[b].tobytes():
                raise ValueError("Exchange is not an exact involution")
            for recipient, donor, rb, db in [(a, b, i, j), (b, a, j, i)]:
                keep = np.arange(9) != rb
                if (proposed[recipient].reshape(9, 6)[keep].tobytes() != sx[recipient].reshape(9, 6)[keep].tobytes()
                        or proposed[recipient].reshape(9, 6)[rb].tobytes() != sx[donor].reshape(9, 6)[db].tobytes()):
                    raise ValueError("Exchange must copy exactly one six-coordinate block")
        pp, pl = self.evaluate(proposed)
        reverse = self.selector(proposed)
        if not np.isfinite(reverse).all():
            raise ValueError("Nonfinite reverse selector")
        x, lp, ll = sx.copy(), sp.copy(), sl.copy()
        decisions = []
        for slot, ((a, b), (i, j)) in enumerate(zip(pairs, labels)):
            decision = joint_decision(sp[[a, b]], pp[[a, b]], forward[[a, b]], reverse[[a, b]],
                                      (i, j), pl[[a, b]], float(uniforms[slot]), threshold)
            old = (sx[[a, b]], sp[[a, b]], sl[[a, b]])
            new = (proposed[[a, b]], pp[[a, b]], pl[[a, b]])
            result = new if decision['accepted'] else old
            x[[a, b]], lp[[a, b]], ll[[a, b]] = result
            for actual, expected in zip((x[[a, b]], lp[[a, b]], ll[[a, b]]), result):
                if actual.tobytes() != expected.tobytes():
                    raise ValueError("Exchange must commit or restore both states jointly")
            decisions.append(dict(pair=[int(a), int(b)], labels=[i, j], **decision))
        result = DNSParticleState(x, lp, ll)
        contract.validate(result, threshold, self.evaluate(x))
        return result, decisions

    def run(self, starts, threshold, streams, config, contract):
        if starts.position.shape != (8, 54):
            raise ValueError("Frozen population requires eight 54-coordinate walkers")
        contract.validate(starts, threshold, self.evaluate(starts.position))
        keys = [jax.random.key(seed, impl='threefry2x32') for seed in streams['slice_seeds']]
        sweeps = config.burn_in+config.retained
        gumbels = np.random.Generator(np.random.PCG64(streams['label_seed'])).gumbel(size=(sweeps, 4, 2, 9))
        uniforms = np.random.Generator(np.random.PCG64(streams['mh_seed'])).random((sweeps, 4))
        state = particles_copy(starts)
        history = []
        accepted = 0
        for sweep in range(sweeps):
            subkeys = []
            for w in range(8):
                keys[w], subkey = jax.random.split(keys[w])
                subkeys.append(subkey)
            values, info = self.slice_step(jnp.stack(subkeys), DNSParticleState(*map(jnp.asarray, state)),
                                           jnp.asarray(threshold))
            sliced = particles_copy(values)
            if not np.asarray(info.slice.is_accepted).all():
                raise ValueError("Slice failure; stop immediately")
            contract.validate(sliced, threshold, self.evaluate(sliced.position))
            for w in range(8):
                selected = self.slice_blocks(int(info.component[w]), np.asarray(info.labels[w]))
                keep = np.setdiff1d(np.arange(9), selected)
                if sliced.position[w].reshape(9, 6)[keep].tobytes() != state.position[w].reshape(9, 6)[keep].tobytes():
                    raise ValueError("Slice changed untouched coordinates")
            state, decisions = self.exchange_sweep(sliced, threshold, sweep, gumbels[sweep], uniforms[sweep], contract)
            accepted += sum(d['accepted'] for d in decisions)
            if sweep >= config.burn_in:
                history.append(particles_copy(state))
        result = DNSParticleState(*[np.stack([getattr(s, name) for s in history]) for name in DNSParticleState._fields])
        return result, dict(completed_sweeps=sweeps, retained=config.retained,
                            exchange_proposals=4*sweeps, accepted_exchanges=accepted,
                            all_constituent_checks_passed=True)
