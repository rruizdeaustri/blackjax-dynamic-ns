"""Isolated conditional-top-visit adapter for automatic DNS development.

All stochastic transitions come from the existing frozen DNS core/callback.
No evidence, reconstruction, stopping criterion or adaptive weights live here.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns, dns_automatic as a, dns_diagnostics as diagnostics

TRACE_FIELDS = ('position', 'logdensity', 'loglikelihood', 'assigned_level',
                'previous_level', 'proposed_level', 'eligible', 'probability',
                'accepted', 'parameter_valid', 'parameter_accepted')


@dataclass(frozen=True)
class VisitConfig:
    burn_in: int
    max_sweeps: int
    retained: int
    checkpoint_sweep: int
    checkpoint_iteration: int

    def __post_init__(self):
        if not 0 <= self.burn_in < self.checkpoint_sweep < self.max_sweeps:
            raise ValueError('Invalid frozen sweep schedule')
        if self.retained <= 0 or self.retained > self.max_sweeps-self.burn_in:
            raise ValueError('Invalid top-observation quota')


def collect_top(trace, top, burn_in, retained):
    """Equal per-walker quota, first post-burn assigned-top visits; never pad."""
    assigned = trace['assigned_level']
    indices = []
    visits = []
    for w in range(assigned.shape[1]):
        times = np.flatnonzero(assigned[burn_in:, w] == top)+burn_in
        visits.append(len(times))
        indices.append(times[:retained])
    if min(visits) < retained:
        return None, indices, visits
    times = np.stack(indices, axis=1)
    walkers = np.arange(assigned.shape[1])[None, :]
    history = dns.DNSParticleState(*[trace[k][times, walkers].copy() for k in a.ARRAYS])
    return history, times, visits


def excursion_witnesses(trace, top, connected_level, burn_in):
    """Witness assigned top -> connected lower level -> other mode -> top.

    The mode change must itself be observed at a connected lower assigned level,
    not merely at a high-eligible state. All indices are physical sweep indices.
    """
    if top <= connected_level:
        return []
    modes = np.where(trace['position'][..., 0] < 0, -1, 1)
    result = []
    for w in range(modes.shape[1]):
        origin = None
        lower = None
        changed = None
        for t in range(burn_in, len(modes)):
            j = int(trace['assigned_level'][t, w])
            if origin is None:
                if j == top:
                    origin = t
                continue
            if j <= connected_level:
                if lower is None:
                    lower = t
                if modes[t, w] != modes[origin, w]:
                    changed = t
            if j == top:
                if lower is not None and changed is not None and modes[t, w] != modes[origin, w]:
                    result.append(dict(walker=w, origin_sweep=origin, lower_sweep=lower,
                                       changed_sweep=changed, return_sweep=t,
                                       origin_mode=int(modes[origin, w]), return_mode=int(modes[t, w]),
                                       minimum_assigned_level=int(trace['assigned_level'][origin:t+1, w].min())))
                origin = t
                lower = changed = None
    return result


def trace_diagnostics(trace, top, burn_in, times, visits, connected_level):
    indices = trace['assigned_level']
    modes = np.where(trace['position'][..., 0] < 0, -1, 1)
    previous, proposed, accepted = [trace[k] for k in ('previous_level', 'proposed_level', 'accepted')]
    up, down = proposed > previous, proposed < previous
    move_attempt = up | down
    post = indices[burn_in:]
    witnesses = excursion_witnesses(trace, top, connected_level, burn_in)
    quota = len(times) if isinstance(times, np.ndarray) else min(map(len, times))
    per_walker = [int(np.sum(modes[1:, w] != modes[:-1, w])) for w in range(indices.shape[1])]
    trips = [diagnostics.round_trip_counts(post[:, w], low_level=0, high_level=top)
             if top else {'low_high_low': 0, 'high_low_high': 0} for w in range(indices.shape[1])]
    return dict(physical_sweeps=len(indices), burn_in_joint_sweeps=burn_in,
                occupancy_fraction=np.bincount(post.ravel(), minlength=top+1)/post.size,
                occupancy_fraction_per_walker=[np.bincount(post[:, w], minlength=top+1)/len(post)
                                               for w in range(post.shape[1])],
                upward_proposals=int(up.sum()), downward_proposals=int(down.sum()),
                upward_moves=int((accepted & up).sum()), downward_moves=int((accepted & down).sum()),
                level_move_acceptance=float(accepted.sum()/move_attempt.sum()) if move_attempt.any() else None,
                top_visit_count_per_walker=visits,
                top_retained_count_per_walker=[quota]*indices.shape[1] if isinstance(times, np.ndarray)
                else [len(v) for v in times],
                round_trips=trips, genuine_cross_mode_excursions=witnesses,
                physical_cross_mode_switches_per_walker=per_walker,
                post_burn_physical_cross_mode_switches=int((modes[burn_in+1:]!=modes[burn_in:-1]).sum()),
                mode_occupancy_per_walker=[[int((modes[burn_in:,w]==m).sum()) for m in (-1,1)]
                                         for w in range(modes.shape[1])],
                all_parameter_checks_passed=bool(trace['parameter_valid'].all()))


def save_snapshot(path, state, keys, trace, prefix, weights, streams, config, top, exchange_seed):
    """Exact joint state, keys, prefix trace/collection and disabled-exchange RNG."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    history, times, visits = collect_top(trace, top, config.burn_in, config.retained)
    values = dict(theta=np.asarray(state.particle.position), logdensity=np.asarray(state.particle.logdensity),
                  loglikelihood=np.asarray(state.particle.loglikelihood), assigned_level=np.asarray(state.level_index),
                  joint_keys=np.asarray(keys), collection_counts=np.minimum(visits, config.retained),
                  **{'trace_'+k:v for k,v in trace.items()})
    if history is not None:
        values.update({'collected_'+k:np.asarray(v) for k,v in zip(a.ARRAYS,history)})
        values['collected_sweeps']=times
    else:
        # Exact ragged collection is saved as per-walker arrays, never fabricated.
        for w, idx in enumerate(times):
            values['collected_sweeps_'+str(w)]=idx
            for k in a.ARRAYS:
                values['collected_'+k+'_'+str(w)]=trace[k][idx,w]
    np.savez_compressed(path/'joint.npz', **values)
    meta=dict(schema='dns-diffusive-bank-v1', completed_sweeps=len(trace['assigned_level']),
              prefix=prefix, log_weight=weights, streams=streams, schedule=a.asdict(config), top=top,
              exchange_rule='disabled identity in both arms',
              exchange_rng_state=np.random.Generator(np.random.PCG64(exchange_seed)).bit_generator.state,
              checksum=a.digest((path/'joint.npz').read_bytes()))
    a.write_json(path/'state.json',dict(payload=meta,sha256=a.digest(a.encoded(meta))))


def load_snapshot(path, prefix, weights, streams, config, top):
    path=Path(path)
    envelope=json.loads((path/'state.json').read_text()); meta=envelope['payload']
    if envelope['sha256'] != a.digest(a.encoded(meta)) or meta['checksum'] != a.digest((path/'joint.npz').read_bytes()):
        raise ValueError('Joint checkpoint checksum mismatch')
    expected=dict(prefix=prefix,log_weight=weights,streams=streams,schedule=a.asdict(config),top=top)
    if any(a.encoded(meta[k]) != a.encoded(v) for k,v in expected.items()):
        raise ValueError('Joint checkpoint target/schedule changed')
    with np.load(path/'joint.npz',allow_pickle=False) as z:
        state=dns.DNSState(dns.DNSParticleState(*[jnp.asarray(z[k]) for k in ('theta','logdensity','loglikelihood')]),
                           jnp.asarray(z['assigned_level']))
        keys=jnp.asarray(z['joint_keys'])
        trace={k:z['trace_'+k].copy() for k in TRACE_FIELDS}
    history,times,visits=collect_top(trace,top,config.burn_in,config.retained)
    with np.load(path/'joint.npz',allow_pickle=False) as z:
        if not np.array_equal(z['collection_counts'],np.minimum(visits,config.retained)):
            raise ValueError('Collection counter mismatch')
        if history is not None:
            for k,v in zip(a.ARRAYS,history):
                if z['collected_'+k].tobytes()!=v.tobytes():raise ValueError('Collection mismatch')
    exchange=np.random.Generator(np.random.PCG64())
    exchange.bit_generator.state=meta['exchange_rng_state']
    return state,keys,trace


class ConditionalKernel:
    """Same automatic gates, with prefix-bound joint or fixed-contour evolution.

    Fixed is the current scalar-contour automatic protocol using the same frozen
    scalar parameter callback. Exchanges are identity in both development arms.
    """
    def __init__(self, parameter_step, evaluator, identity, method, visits, folder,
                 thresholds, log_masses, promotion, resume_selection=None, connected_level=0):
        self.parameter_step=parameter_step; self.evaluator=evaluator; self.identity=identity
        self.method=method; self.visits=visits; self.folder=Path(folder)
        self.thresholds=thresholds; self.log_masses=log_masses; self.promotion=promotion
        self.resume_selection=resume_selection; self.connected_level=connected_level

    @property
    def metadata(self):
        return self.identity

    def evaluate(self,position):
        return self.evaluator(position)

    def run(self,starts,threshold,streams,config,contract):
        top=len(self.thresholds)-1; weights=np.zeros(top+1)
        prefix=dict(thresholds=self.thresholds,log_masses=self.log_masses)
        if config.retained != self.visits.retained or config.burn_in != self.visits.burn_in:
            raise ValueError('Automatic and joint collection schedules disagree')
        contract.validate(starts,threshold,self.evaluate(starts.position))
        levels=dns.create_levels(self.thresholds,self.log_masses,weights)
        # Reuse the validated index transition, never duplicate its MH rule.
        joint_step=dns.build_kernel(levels,self.parameter_step)
        if self.method=='diffusive':
            advance=jax.vmap(joint_step)
        elif self.method=='fixed':
            def fixed_step(key,state):
                # Identical parameter-key layout to DNS build_kernel(inner=1).
                pk,_=jax.random.split(key)
                particle,info=self.parameter_step(jax.random.split(pk,1)[0],state.particle,threshold)
                li=dns.DNSLevelInfo(state.level_index,state.level_index,jnp.array(False),jnp.array(0.),jnp.array(False))
                return dns.DNSState(particle,state.level_index),dns.DNSInfo(info,jnp.array([True]),
                    jax.tree.map(lambda v:v[None],li))
            advance=jax.vmap(fixed_step)
        else:raise ValueError('Unknown comparison method')
        state=dns.DNSState(dns.DNSParticleState(*map(jnp.asarray,starts)),jnp.full(starts.loglikelihood.shape,top,jnp.int32))
        keys=jnp.stack([jax.random.PRNGKey(s) for s in streams['slice_seeds']])
        prior_trace=None
        if self.resume_selection is not None and streams['bank']=='selection':
            state,keys,prior_trace=load_snapshot(self.resume_selection,prefix,weights,streams,self.visits,top)
        def scan_run(state,keys,length):
            def body(carry,_):
                current,keys=carry
                split=jax.vmap(lambda k:jax.random.split(k))(keys)
                current,info=advance(split[:,1],current)
                li=info.level_info
                trace=(current.particle.position,current.particle.logdensity,current.particle.loglikelihood,
                       current.level_index,li.previous_level[:,0],li.proposed_level[:,0],li.is_eligible[:,0],
                       li.acceptance_probability[:,0],li.is_accepted[:,0],jnp.all(info.parameter_is_valid,axis=1),
                       jnp.all(info.parameter_info.is_accepted.reshape((len(keys),-1)),axis=1))
                return (current,split[:,0]),trace
            return jax.jit(lambda s,k:jax.lax.scan(body,(s,k),None,length=length))(state,keys)
        start=0 if prior_trace is None else len(prior_trace['assigned_level'])
        checkpoint=(self.method=='diffusive' and streams['iteration']==self.visits.checkpoint_iteration
                    and streams['bank']=='selection' and start==0)
        segments=[self.visits.checkpoint_sweep,self.visits.max_sweeps-self.visits.checkpoint_sweep] if checkpoint else [self.visits.max_sweeps-start]
        trace=prior_trace
        for n in segments:
            (state,keys),arrays=scan_run(state,keys,n)
            piece={k:np.asarray(v) for k,v in zip(TRACE_FIELDS,arrays)}
            trace=piece if trace is None else {k:np.concatenate([trace[k],piece[k]]) for k in TRACE_FIELDS}
            if checkpoint and len(trace['assigned_level'])==self.visits.checkpoint_sweep:
                save_snapshot(self.folder/'selection_midpoint',state,keys,trace,prefix,weights,streams,
                              self.visits,top,streams['mh_seed'])
        if not trace['parameter_valid'].all() or not trace['parameter_accepted'].all():
            raise ValueError('Invalid or unsuccessful parameter callback state')
        for k in ('position','logdensity','loglikelihood'):
            if not np.isfinite(trace[k]).all():raise ValueError('Nonfinite joint state')
        assigned_threshold=np.asarray(self.thresholds)[trace['assigned_level']]
        if np.any(trace['loglikelihood']<=assigned_threshold):raise ValueError('Assigned contour violation')
        if not np.allclose(trace['logdensity'],self.evaluate(trace['position'])[0],rtol=0,atol=contract.prior_atol):
            raise ValueError('Joint prior cache mismatch')
        if not np.allclose(trace['loglikelihood'],self.evaluate(trace['position'])[1],rtol=0,atol=contract.likelihood_atol):
            raise ValueError('Joint likelihood cache mismatch')
        history,times,visits=collect_top(trace,top,self.visits.burn_in,self.visits.retained)
        diag=trace_diagnostics(trace,top,self.visits.burn_in,times,visits,self.connected_level)
        bank=streams['bank'];np.savez_compressed(self.folder/(bank+'_trace.npz'),**trace)
        save_snapshot(self.folder/(bank+'_final_state'),state,keys,trace,prefix,weights,streams,self.visits,top,streams['mh_seed'])
        a.write_json(self.folder/(bank+'_visitation.json'),diag)
        if history is None:raise ValueError('insufficient_top_level_visits: frozen maximum sweep budget exhausted; no retry')
        contract.validate(history,threshold,self.evaluate(history.position))
        event=np.array([[f'{self.method}/{streams["iteration"]}/{bank}/{int(t)}/{w}' for w,t in enumerate(row)] for row in times])
        promoted=self.promotion[bank]
        np.savez_compressed(self.folder/(bank+'_top_observations.npz'),
                            **dict(zip(a.ARRAYS,history)),sweep_index=times,walker_id=np.broadcast_to(np.arange(len(visits)),times.shape),
                            assigned_level=np.full(times.shape,top),event_id=event,
                            promoted_donor_id=np.broadcast_to(promoted['donor_ids'],times.shape),
                            promoted_retained_index=np.broadcast_to(promoted['retained_indices'],times.shape),
                            promoted_source_hash=np.broadcast_to(promoted['source_hashes'],times.shape))
        diag.update(completed_sweeps=self.visits.max_sweeps,retained=config.retained,
                    source_history_hash=promoted['source_history_hash'],exchange_rule='identity',
                    level_weights=weights,log_a=weights-np.asarray(self.log_masses))
        return history,a.json_value(diag)


def build_next_level(state,parameter_step,evaluator,identity,method,visits,output,resume_selection=None):
    """Bind the frozen prefix, delegate unchanged promotion/candidate/gates."""
    _,promotion,_=a.prepare_iteration(state)
    kernel=ConditionalKernel(parameter_step,evaluator,identity,method,visits,output,
                             state.thresholds,state.log_masses,promotion,resume_selection)
    new=a.build_next_level(state,kernel,output)
    if new.status=='ready':
        # Automatic checkpoint contains both retained banks. Companion joint
        # bundles contain exact terminal indices, keys, counters, and histories.
        files={}
        for bank in a.BANKS:
            for name in ('joint.npz','state.json'):
                rel=f'{bank}_final_state/{name}'
                files[rel]=a.digest((Path(output)/rel).read_bytes())
        a.write_json(Path(output)/'checkpoint'/'joint_manifest.json',dict(
            schema='dns-diffusive-iteration-v1',iteration=new.iteration,files=files,
            boundary_rule='Each next attempt promotes bank-local retained histories and derives fresh counter-based streams.',
            log_weight=np.zeros(len(new.thresholds)), reconstruction_samples_exported=False))
    return new
