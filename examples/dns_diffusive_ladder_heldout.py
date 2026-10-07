"""One preregistered held-out check of the committed Stage-9C construction.

This validation harness only supplies a new analytic model and fail-fast audits.
It changes no parameter/index transition, promotion, collection or ESS estimator.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from blackjax.ns import dns, dns_automatic as a, dns_automatic_diffusive as d, dns_kernels

ROOT=Path('/tmp/dns-stage9d-heldout')
DESIGN=Path('docs/examples/dns_diffusive_ladder_heldout_design.json')
REPORT=Path('docs/examples/dns_diffusive_ladder_heldout.json')
DOC=Path('docs/examples/dns_diffusive_ladder_heldout.md')
DEVELOPMENT=Path('docs/examples/dns_diffusive_ladder_design.json')
PROCESS=dict(pid=os.getpid(),parent_pid=os.getppid(),start_uuid=str(uuid.uuid4()),
             startup_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
             executable=sys.executable,worker_identifier='stage9d-heldout')
PROTECTED=['blackjax/ns/dns.py','blackjax/ns/dns_kernels.py','blackjax/ns/dns_automatic.py',
 'blackjax/ns/dns_automatic_diffusive.py','blackjax/ns/dns_levels.py','blackjax/ns/dns_population.py',
 'blackjax/ns/dns_reconstruction.py','blackjax/ns/dns_termination.py','blackjax/ns/dns_diagnostics.py',
 'examples/dns_diffusive_ladder_development.py','tests/ns/test_dns_automatic_diffusive.py',
 'docs/examples/dns_diffusive_ladder_design.json','docs/examples/dns_diffusive_ladder_design.sha256',
 'docs/examples/dns_diffusive_ladder_development.json','docs/examples/dns_diffusive_ladder_development.md',
 'docs/examples/dns_ladder_robustness_design.json','docs/examples/dns_ladder_robustness_design.sha256',
 'docs/examples/dns_automatic_backtracking_audit.md',
 'docs/examples/dns_inference_reconstruction_validation.json']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ValidationFailure(ValueError):
    def __init__(self,classification,reason):
        self.classification=classification;self.reason=reason
        super().__init__(classification+': '+reason)


class ThreeModes:
    """Weighted normal upper envelope: exact contour union and Gaussian integrals.

    w_k = sigma_k / sum sigma gives the same peak height for all modes.
    L(theta)=max_k w_k NormalPDF(theta;mu_k,sigma_k), not a Gaussian sum.
    Prior is uniform; mode regions are fixed geometric midpoint intervals.
    """
    def __init__(self,model):
        self.bound=model['prior_bound'];self.centers=np.array(model['centers'])
        self.widths=np.array(model['widths']);self.weights=np.array(model['weights'])
        self.log_peak=float(-np.log(self.widths.sum()*np.sqrt(2*np.pi)))
        self.edges=np.array([-self.bound,*model['mode_region_boundaries'],self.bound])

    def intervals(self,ell):
        if np.isneginf(ell):return [(-self.bound,self.bound)]
        if ell>=self.log_peak:return []
        radius=self.widths*np.sqrt(2*(self.log_peak-ell))
        pieces=sorted((max(-self.bound,c-r),min(self.bound,c+r)) for c,r in zip(self.centers,radius))
        merged=[]
        for lo,hi in pieces:
            if lo>=hi:continue
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
            else:merged.append((lo,hi))
        return merged

    def truth(self,ell,values=None):
        pieces=self.intervals(ell);length=sum(hi-lo for lo,hi in pieces)
        if not length:raise ValueError('Zero analytic contour mass')
        mean=sum((hi**2-lo**2)/2 for lo,hi in pieces)/length
        second=sum((hi**3-lo**3)/3 for lo,hi in pieces)/length
        mode_mass=[sum(max(0.,min(hi,right)-max(lo,left)) for lo,hi in pieces)/length
                   for left,right in zip(self.edges[:-1],self.edges[1:])]
        result=dict(intervals=pieces,mass=length/(2*self.bound),mean=mean,second_moment=second,
                    sd=float(np.sqrt(second-mean**2)),mode_probabilities=mode_mass,disconnected=len(pieces)>1)
        if values is not None:
            x=np.asarray(values);result['cdf']=sum(np.clip(x-lo,0,hi-lo) for lo,hi in pieces)/length
        return result

    def modes(self,x):return np.searchsorted(self.edges[1:-1],np.asarray(x),side='right')

    def evaluate(self,position):
        x=np.asarray(position)[...,0]
        lp=np.where(np.abs(x)<=self.bound,-np.log(2*self.bound),-np.inf)
        ll=self.log_peak-.5*np.min(((x[...,None]-self.centers)/self.widths)**2,axis=-1)
        return lp.astype(np.float64),ll.astype(np.float64)

    def parameter_step(self,settings):
        bound=self.bound;mu=jnp.asarray(self.centers);sigma=jnp.asarray(self.widths);peak=self.log_peak
        def prior(x):return jnp.where(jnp.abs(x[0])<=bound,-jnp.log(2*bound),-jnp.inf)
        def likelihood(x):return peak-.5*jnp.min(((x[0]-mu)/sigma)**2)
        def direction(key,x):
            del x
            return jnp.array([jnp.where(jax.random.bernoulli(key),settings['direction_magnitude'],-settings['direction_magnitude'])])
        return dns_kernels.build_constrained_slice_kernel(prior,likelihood,generate_slice_direction_fn=direction,
                            max_steps=settings['max_steps'],max_shrinkage=settings['max_shrinkage'])

    def posterior_truth(self):
        # Pairwise equal-height Gaussian crossings partition the dominance cells.
        points=[-self.bound,self.bound,*self.edges[1:-1]]
        for i in range(3):
            for j in range(i+1,3):
                ci,cj=self.centers[[i,j]];si,sj=self.widths[[i,j]]
                for sign in (-1,1):
                    denominator=sj-sign*si
                    if denominator:
                        x=(sj*ci-sign*si*cj)/denominator
                        if -self.bound<x<self.bound:points.append(float(x))
        points=sorted(set(points));mass=np.zeros(3);cells=[]
        for lo,hi in zip(points[:-1],points[1:]):
            midpoint=(lo+hi)/2; k=int(np.argmin(((midpoint-self.centers)/self.widths)**2))
            integral=float(self.weights[k]*(ndtr((hi-self.centers[k])/self.widths[k])-ndtr((lo-self.centers[k])/self.widths[k])))
            mass[int(self.modes(midpoint))]+=integral
            cells.append(dict(lower=lo,upper=hi,dominant_component=k,likelihood_integral=integral))
        return dict(evidence=float(mass.sum()/(2*self.bound)),mode_probabilities=(mass/mass.sum()).tolist(),
                    likelihood_integral_per_geometric_mode=mass.tolist(),dominance_cells=cells,
                    note='Independent exact Gaussian integrals; not a reconstruction trajectory or estimator.')


def committed_hashes():
    for file in PROTECTED:
        committed=subprocess.run(['git','show','HEAD:'+file],check=True,capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest()!=sha(file):raise ValueError('Protected source not equal to HEAD: '+file)
    return {file:sha(file) for file in PROTECTED}


def prepare():
    if ROOT.exists() or DESIGN.exists() or REPORT.exists():raise ValueError('Held-out study already exists: no retry')
    old=json.loads(DEVELOPMENT.read_text());protected=committed_hashes()
    model=dict(identifier='stage9d-three-asymmetric-weighted-normal-envelope-v1',prior='Uniform[-8,8]',prior_bound=8.,
               centers=[-3.8,-.7,2.8],widths=[.27,.39,.32],weights=(np.array([.27,.39,.32])/.98).tolist(),
               likelihood='max_k w_k NormalPDF(theta;mu_k,sigma_k); w_k=sigma_k/sum(sigma)',
               mode_definition='geometric nearest-center regions, fixed before sampling',mode_region_boundaries=[-2.25,1.05],
               dimension=1,novelty='Three asymmetric modes with unequal widths and weights; not Stage 9A, Stage 9C or Stage 8H.')
    toy=ThreeModes(model);seed=194071;banks={}
    for bank,child in zip(a.BANKS,np.random.SeedSequence(seed).spawn(2)):
        rng=np.random.Generator(np.random.PCG64(child));x=rng.uniform(-8,8,size=(old['initial_draws_per_walker'],old['walkers'],1))
        lp,ll=toy.evaluate(x)
        banks[bank]=dict(position=x,logdensity=lp,loglikelihood=ll,seed_sequence=child.state,
                        initialization_rng_state=rng.bit_generator.state,history_sha256=a.particle_hash(dns.DNSParticleState(x,lp,ll)))
    criteria={k:old['criteria'][k] for k in ['ESS_gate','KS_max','mode_probability_absolute',
        'mean_absolute_scaled_by_conditional_sd','second_moment_relative','calibration_compression_absolute',
        'selection_compression_absolute','calibrated_log_mass_absolute','minimum_top_observations_per_walker',
        'minimum_genuine_excursions_each_deep_bank','minimum_deep_top_mode_switches_each_bank','connected_witness_level']}
    criteria.update(require_each_analytic_mode_each_bank_each_attempt=True,
        conditional_scope='Every attempted generating contour J, including J=0; exact truth tolerances copied from Stage 9C.',
        mixing_scope='Each bank at every disconnected generating contour: >=5 saved physical level-0 cross-mode returns, >=10 retained top-mode changes, and strongly connected directed return graph over all 3 modes.',
        immediate_failure='Audit selection before calibration, then calibration before next attempt. Fail at first failed requirement. No tuning, new method, seed, budget or retry.',
        failure_mapping='Source/cache/implementation integrity -> FAILED-CONDITIONAL with explicit reason; quotas/ESS -> FAILED-VISITATION; lost mode or disconnected return graph -> FAILED-MIXING; remaining conditional truth -> FAILED-CONDITIONAL; compression truth -> FAILED-COMPRESSION; restart -> FAILED-RESTART.',
        pass_logic='All six candidates accepted; every bank passes conditional/mode/mixing/ESS/compression/visit checks; protected hashes unchanged; no reconstruction samples exported; one fresh-process restart exactly matches.',
        freeze_final_top='Final ell_6 is newly accepted, not the generating contour of construction history. Its mass is checked analytically; no extra chain is authorized there.')
    spec=dict(stage='9D',study='ONE INDEPENDENT HELD-OUT VALIDATION',model=model,posterior_truth=toy.posterior_truth(),
        constructor_seed=seed,initial_banks=banks,walkers=old['walkers'],initial_draws_per_walker=old['initial_draws_per_walker'],
        parameter=old['parameter'],visits=old['visits'],automatic_config=old['automatic_config'],
        max_new_levels=old['max_new_levels'],level_weights=dict(log_weight=[0.]*(old['max_new_levels']+1),
            rule='Use prefix zeros; log a_j = -log Xhat_j. Weights frozen before each attempt; no adaptation.'),
        exchange_rule=old['exchange_rule'],thinning=old['thinning'],criteria=criteria,
        restart=dict(completed_level=3,midpoint_iteration=3,midpoint_bank='selection',midpoint_sweep=8192,
                     required='Exactly one new child process continues from completed level 3 and selection midpoint; no retry.',
                     provenance='Parent Popen receipt, PID/PPID, independent startup UUIDs, UTC timestamps, full launch command.'),
        stage9c_design_sha256=sha(DEVELOPMENT),inherited_settings='Budgets, quotas, parameter settings, numeric tolerances and ESS copied unchanged from Stage 9C; new model was not used to choose them.',
        numerical_contract=old['numerical_contract'],environment=a.NumericalContract().environment(),
        protected_sha256=protected,validation_driver_sha256=sha(__file__),git_commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        reconstruction_policy='No imports, invocation or export of reconstruction records; only analytic model truth is integrated.',
        no_LISA=True,no_termination=True,no_production=True,no_retry=True)
    # Deterministic truth invariants checked before any Markov trajectory.
    prior=toy.truth(-np.inf,np.array([-8.,8.]));assert prior['mass']==1 and np.allclose(prior['cdf'],[0.,1.])
    assert np.isclose(sum(prior['mode_probabilities']),1)
    assert np.isclose(sum(spec['posterior_truth']['mode_probabilities']),1)
    a.write_json(DESIGN,spec);checksum=sha(DESIGN)
    DESIGN.with_suffix('.sha256').write_text(checksum+'  '+DESIGN.name+'\n')
    ROOT.mkdir();a.write_json(ROOT/'design.json',spec);(ROOT/'design.sha256').write_text(checksum+'\n')
    print('PREREGISTERED',checksum,flush=True)


def load_design():
    spec=json.loads(DESIGN.read_text())
    if sha(DESIGN)!=(ROOT/'design.sha256').read_text().strip() or sha(DESIGN)!=DESIGN.with_suffix('.sha256').read_text().split()[0]:
        raise ValidationFailure('FAILED-CONDITIONAL','Frozen design changed')
    for file,value in spec['protected_sha256'].items():
        if sha(file)!=value:raise ValidationFailure('FAILED-CONDITIONAL','Frozen source changed: '+file)
    if sha(__file__)!=spec['validation_driver_sha256']:raise ValidationFailure('FAILED-CONDITIONAL','Frozen validation harness changed')
    if a.NumericalContract().environment()!=spec['environment']:raise ValidationFailure('FAILED-CONDITIONAL','CPU/x64 environment changed')
    return spec


def identity(spec):
    return dict(version='heldout-audit-wrapper-v1',method='diffusive',model=spec['model'],parameter=spec['parameter'],
                visits=spec['visits'],exchange=spec['exchange_rule'],frozen_implementation=spec['protected_sha256']['blackjax/ns/dns_automatic_diffusive.py'],
                validation_driver_sha256=spec['validation_driver_sha256'])


def initial(spec):
    banks={name:dns.DNSParticleState(*[np.asarray(spec['initial_banks'][name][k],dtype=np.float64) for k in a.ARRAYS]) for name in a.BANKS}
    for name,history in banks.items():
        if a.particle_hash(history)!=spec['initial_banks'][name]['history_sha256']:raise ValidationFailure('FAILED-CONDITIONAL','Preregistered initial bank mismatch')
    return a.LadderState((-np.inf,),(0.,),**banks,seed=spec['constructor_seed'],kernel_metadata=a.encoded(identity(spec)),
                        config=a.LadderConfig(**spec['automatic_config']))


def graph_connected(counts):
    for start in range(3):
        seen={start};frontier=[start]
        while frontier:
            current=frontier.pop()
            for nxt in range(3):
                if counts[current,nxt]>0 and nxt not in seen:seen.add(nxt);frontier.append(nxt)
        if len(seen)!=3:return False
    return True


def mode_diagnostics(toy,trace,top,burn,top_values):
    assigned=trace['assigned_level'];modes=toy.modes(trace['position'][...,0])
    witnesses=[];edges=np.zeros((3,3),dtype=int)
    for w in range(modes.shape[1]):
        origin=lower=changed=None
        for t in range(burn,len(modes)):
            j=int(assigned[t,w])
            if origin is None:
                if j==top:origin=t
                continue
            if j==0 and lower is None:lower=t
            if (lower is not None and t>origin and assigned[t-1,w]==0 and j==0
                    and modes[t,w]!=modes[t-1,w] and modes[t,w]!=modes[origin,w]):changed=t
            if j==top:
                if (top>0 and lower is not None and changed is not None and modes[t,w]!=modes[origin,w]
                        and modes[changed,w]==modes[t,w]):
                    start_mode=int(modes[origin,w]);end_mode=int(modes[t,w]);edges[start_mode,end_mode]+=1
                    witnesses.append(dict(walker=w,origin_sweep=origin,lower_sweep=lower,physical_change_sweep=changed,
                        return_sweep=t,origin_mode=start_mode,return_mode=end_mode,
                        physical_from_mode=int(modes[changed-1,w]),physical_to_mode=int(modes[changed,w]),
                        minimum_assigned_level=int(assigned[origin:t+1,w].min())))
                origin=t;lower=changed=None
    kept_modes=toy.modes(top_values)
    physical_counts=[int((modes[burn+1:,w]!=modes[burn:-1,w]).sum()) for w in range(modes.shape[1])]
    return dict(mode_definition='fixed three geometric midpoint regions; supplement, not replacement, of frozen binary-sign diagnostics',
        physical_mode_switches_per_walker=physical_counts,physical_mode_occupancy_per_walker=[np.bincount(modes[burn:,w],minlength=3) for w in range(modes.shape[1])],
        retained_top_mode_switches=int((kept_modes[1:]!=kept_modes[:-1]).sum()),
        retained_mode_counts_per_walker=[np.bincount(kept_modes[:,w],minlength=3) for w in range(modes.shape[1])],
        directed_cross_mode_return_counts=edges,strongly_connected=graph_connected(edges),witnesses=witnesses)


def audit_bank(spec,toy,folder,bank,history,threshold,top,candidate):
    values=history.position[...,0];sorted_values=np.sort(values.ravel());n=values.size
    truth=toy.truth(threshold,sorted_values);cdf=truth.pop('cdf')
    ks=float(max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n)))
    modes=toy.modes(values);mode_prob=np.bincount(modes.ravel(),minlength=3)/n
    mean=float(values.mean());second=float(np.mean(values**2));criteria=spec['criteria']
    tail=a.dns_levels.tail_diagnostics(history.loglikelihood,candidate.threshold,spec['automatic_config']['block_size'])
    with np.load(folder/(bank+'_top_observations.npz'),allow_pickle=False) as z:
        kept={k:z[k] for k in z.files}
    with np.load(folder/(bank+'_trace.npz'),allow_pickle=False) as z:
        trace={k:z[k] for k in z.files}
    ts=kept['sweep_index'];ws=kept['walker_id']
    exact_events=all(kept[k].dtype==trace[k][ts,ws].dtype and kept[k].tobytes()==trace[k][ts,ws].tobytes() for k in a.ARRAYS)
    conditional=dict(KS=ks<=criteria['KS_max'],
        mean=abs(mean-truth['mean'])<=criteria['mean_absolute_scaled_by_conditional_sd']*truth['sd'],
        second_moment=abs(second-truth['second_moment'])/truth['second_moment']<=criteria['second_moment_relative'],
        mode_probabilities=bool(np.all(np.abs(mode_prob-truth['mode_probabilities'])<=criteria['mode_probability_absolute'])),
        strict_membership=bool(np.all(history.loglikelihood>threshold)),assigned_level=bool(np.all(kept['assigned_level']==top)),
        unique_events=len(np.unique(kept['event_id']))==n,exact_trace_events=exact_events)
    modal=mode_diagnostics(toy,trace,top,spec['visits']['burn_in'],values)
    represented=bool(np.all(np.bincount(modes.ravel(),minlength=3)>0))
    mixing=dict(every_mode_represented=represented)
    if truth['disconnected']:
        mixing.update(real_level0_cross_mode_returns=len(modal['witnesses'])>=criteria['minimum_genuine_excursions_each_deep_bank'],
                      retained_top_mode_movement=modal['retained_top_mode_switches']>=criteria['minimum_deep_top_mode_switches_each_bank'],
                      all_three_modes_reconnected=modal['strongly_connected'])
    visitation=json.loads((folder/(bank+'_visitation.json')).read_text())
    visits=min(visitation['top_visit_count_per_walker'])>=criteria['minimum_top_observations_per_walker']
    movement=top==0 or (visitation['upward_moves']>0 and visitation['downward_moves']>0)
    truth_ratio=toy.truth(candidate.threshold)['mass']/truth['mass']
    actual=tail['ratio']; log_mass_proposed=None
    compression_error=abs(actual-truth_ratio)
    report=dict(bank=bank,generating_level=top,generating_threshold=threshold,candidate_threshold=candidate.threshold,
        conditional_truth=truth,KS=ks,mean=mean,second_moment=second,observed_mode_probabilities=mode_prob,
        conditional_checks=conditional,conditional_pass=all(conditional.values()),mixing_checks=mixing,
        mixing_pass=all(mixing.values()),level_movement_pass=movement,
        visitation=visitation,analytic_mode_diagnostics=modal,top_level_ESS=tail['ess'],compression=actual,
        exact_compression=truth_ratio,compression_absolute_error=compression_error,
        survivor_counts_per_walker=(history.loglikelihood>candidate.threshold).sum(0),
        survivor_fraction_CV=float(np.std((history.loglikelihood>candidate.threshold).mean(0),ddof=1)/actual) if actual else None,
        between_walker_standard_error=float(np.std((history.loglikelihood>candidate.threshold).mean(0),ddof=1)/np.sqrt(spec['walkers'])),
        tail_diagnostics=tail,sufficient_visits=visits)
    return a.json_value(report)


class AuditKernel:
    """Read-only audit of returned samples; fail before the next bank/attempt.

    Calls the committed ConditionalKernel unchanged. This wrapper adds no
    proposal, changes no sample, and never repeats a stochastic call.
    """
    def __init__(self,base,spec,toy,state,folder):
        self.base=base;self.spec=spec;self.toy=toy;self.state=state;self.folder=Path(folder)
        self.reports={};self.failure=None

    @property
    def metadata(self):return self.base.metadata
    def evaluate(self,x):return self.base.evaluate(x)

    def fail(self,label,reason,bank):
        self.failure=dict(classification=label,reason=reason,bank=bank)
        a.write_json(self.folder/'validation_failure.json',self.failure)
        raise ValidationFailure(label,reason)

    def run(self,starts,threshold,streams,config,contract):
        bank=streams['bank']
        load_design()
        try:history,diag=self.base.run(starts,threshold,streams,config,contract)
        except Exception as exc:
            label='FAILED-VISITATION' if 'insufficient_top_level_visits' in str(exc) else 'FAILED-CONDITIONAL'
            self.fail(label,'Frozen adapter failed: '+str(exc),bank)
        load_design()
        if bank=='selection':candidate=a.select_candidate(history,threshold,config)
        else:
            raw=(self.folder/'candidate.json').read_bytes();candidate=a.Candidate(raw)
            if a.digest(raw)!=json.loads((self.folder/'candidate_hash.json').read_text())['sha256']:
                self.fail('FAILED-CONDITIONAL','Immutable candidate hash mismatch',bank)
        if candidate.threshold is None or candidate.diagnostics['status']!='ok':
            self.fail('FAILED-VISITATION','Frozen selection candidate/ESS gate failed',bank)
        report=audit_bank(self.spec,self.toy,self.folder,bank,history,threshold,len(self.state.thresholds)-1,candidate)
        self.reports[bank]=report
        if bank=='calibration':
            log_mass=float(self.state.log_masses[-1]+np.log(report['compression']))
            exact=float(np.log(self.toy.truth(candidate.threshold)['mass']))
            report.update(calibrated_log_mass=log_mass,exact_log_mass=exact,log_mass_absolute_error=abs(log_mass-exact))
        a.write_json(self.folder/(bank+'_heldout_checks.json'),report)
        if not report['sufficient_visits'] or report['top_level_ESS']<20:
            self.fail('FAILED-VISITATION','Top visitation or ESS requirement failed',bank)
        if not report['mixing_checks']['every_mode_represented']:
            self.fail('FAILED-MIXING','An analytic mode is absent from retained samples',bank)
        if not report['conditional_pass']:
            self.fail('FAILED-CONDITIONAL','Frozen exact conditional-truth checks failed: '+str(report['conditional_checks']),bank)
        tolerance=self.spec['criteria'][bank+'_compression_absolute']
        if report['compression_absolute_error']>tolerance or report.get('log_mass_absolute_error',0)>self.spec['criteria']['calibrated_log_mass_absolute']:
            self.fail('FAILED-COMPRESSION','Frozen compression/log-mass truth tolerance failed',bank)
        if not report['mixing_pass'] or not report['level_movement_pass']:
            self.fail('FAILED-MIXING','Frozen level-movement/mode-connectivity checks failed',bank)
        return history,diag


def companion_manifest(folder,state):
    # Mirrors only the frozen helper's iteration-boundary companion bookkeeping.
    files={}
    for bank in a.BANKS:
        for name in ('joint.npz','state.json'):
            rel=f'{bank}_final_state/{name}';files[rel]=sha(Path(folder)/rel)
    a.write_json(Path(folder)/'checkpoint/joint_manifest.json',dict(schema='dns-diffusive-iteration-v1',iteration=state.iteration,
        files=files,log_weight=np.zeros(len(state.thresholds)),reconstruction_samples_exported=False,
        boundary_rule='Next attempt uses unchanged bank-local promotion and counter-derived streams.'))


def branch(spec,output,checkpoint=None,midpoint=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    a.write_json(output/'process.json',PROCESS)
    toy=ThreeModes(spec['model']);step=toy.parameter_step(spec['parameter'])
    if checkpoint:
        stub=type('Identity',(),{'metadata':identity(spec)})()
        state=a.load_checkpoint(checkpoint,stub)
        manifest=json.loads((Path(checkpoint)/'joint_manifest.json').read_text())
        for rel,value in manifest['files'].items():
            if sha(Path(checkpoint).parent/rel)!=value:raise ValidationFailure('FAILED-RESTART','Joint checkpoint checksum mismatch')
    else:
        state=initial(spec);a.save_checkpoint(output/'initial_checkpoint',state)
    first=state.iteration;rows=[];failure=None
    while state.status=='ready' and state.iteration<spec['max_new_levels']:
        load_design();j=state.iteration;folder=output/f'attempt_{j:06d}'
        _,promotion,_=a.prepare_iteration(state)
        base=d.ConditionalKernel(step,toy.evaluate,identity(spec),'diffusive',d.VisitConfig(**spec['visits']),folder,
             state.thresholds,state.log_masses,promotion,midpoint if j==first else None)
        audit=AuditKernel(base,spec,toy,state,folder)
        # The complete automatic promotion/select/persist/calibrate/gate loop is
        # still the frozen implementation. Read-only wrapper gates each bank.
        state=a.build_next_level(state,audit,folder)
        record=json.loads(state.records[-1])
        row=dict(generating_level=j,accepted=record['accepted'],banks=audit.reports)
        if audit.failure:failure=audit.failure;row['failure']=failure
        elif state.status!='ready':
            failure=dict(classification='FAILED-VISITATION' if record.get('stop_reason') in ('selection_gate','calibration_gate') else 'FAILED-CONDITIONAL',
                         reason=record.get('error',record.get('stop_reason')),bank='orchestrator');row['failure']=failure
        else:
            companion_manifest(folder,state)
            row.update(threshold=state.thresholds[-1],log_mass=state.log_masses[-1],exact_mass=toy.truth(state.thresholds[-1])['mass'])
        a.write_json(folder/'heldout_attempt.json',row);rows.append(row)
        print('heldout level',j+1,'accepted',record['accepted'],'failure',failure,
              'KS',[audit.reports.get(b,{}).get('KS') for b in a.BANKS],
              'ESS',[audit.reports.get(b,{}).get('top_level_ESS') for b in a.BANKS],flush=True)
        if failure:break
    result=dict(completed_levels=len(state.thresholds)-1,completed_all=state.status=='ready' and len(state.thresholds)-1==spec['max_new_levels'],
        thresholds=state.thresholds,log_masses=state.log_masses,exact_masses=[toy.truth(float(t))['mass'] for t in state.thresholds],
        records=[json.loads(r) for r in state.records],attempts=rows,failure=failure,process=PROCESS,
        level_log_weights=[0.]*len(state.thresholds),reconstruction_samples_exported=False)
    a.write_json(output/'result.json',result)
    return a.json_value(result)


def restart(spec,original):
    checkpoint=ROOT/'original/attempt_000002/checkpoint';midpoint=ROOT/'original/attempt_000003/selection_midpoint'
    command=[sys.executable,'-m','examples.dns_diffusive_ladder_heldout','--worker',str(checkpoint),str(midpoint),str(ROOT/'restart')]
    launched=datetime.now(timezone.utc).isoformat()
    with (ROOT/'restart.log').open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        launch=dict(parent=PROCESS,child_pid=proc.pid,timestamp_utc=launched,command=command)
        a.write_json(ROOT/'restart_launch.json',launch);code=proc.wait()
    if code:return dict(passed=False,classification='FAILED-RESTART',returncode=code,launch=launch,no_retry=True)
    other=json.loads((ROOT/'restart/result.json').read_text());child=json.loads((ROOT/'restart/process.json').read_text())
    process=child['pid']==launch['child_pid'] and child['parent_pid']==PROCESS['pid'] and child['start_uuid']!=PROCESS['start_uuid'] and child['startup_utc']>=launched
    checks={k:a.encoded(original[k])==a.encoded(other[k]) for k in ('thresholds','log_masses','exact_masses','level_log_weights','records')}
    checks['heldout_checks']=a.encoded(original['attempts'][3:])==a.encoded(other['attempts'])
    for j in range(3,spec['max_new_levels']):
        x=ROOT/'original'/f'attempt_{j:06d}';y=ROOT/'restart'/f'attempt_{j:06d}'
        for bank in a.BANKS:
            for rel in [bank+'_trace.npz',bank+'_top_observations.npz',bank+'_final_state/joint.npz']:
                if not (y/rel).exists():checks[f'{j}/{rel}']=False;continue
                with np.load(x/rel,allow_pickle=False) as aa,np.load(y/rel,allow_pickle=False) as bb:
                    checks[f'{j}/{rel}']=set(aa.files)==set(bb.files) and all(aa[k].dtype==bb[k].dtype and aa[k].shape==bb[k].shape and aa[k].tobytes()==bb[k].tobytes() for k in aa.files)
            for rel in [bank+'_final_state/state.json',bank+'_heldout_checks.json']:
                checks[f'{j}/{rel}']=(y/rel).exists() and a.encoded(json.loads((x/rel).read_text()))==a.encoded(json.loads((y/rel).read_text()))
        for rel in ['candidate.json','candidate_hash.json','decision.json','checkpoint/checkpoint.json','checkpoint/joint_manifest.json']:
            checks[f'{j}/{rel}']=(y/rel).exists() and a.encoded(json.loads((x/rel).read_text()))==a.encoded(json.loads((y/rel).read_text()))
    return dict(passed=process and all(checks.values()),new_process_established=process,checks=checks,launch=launch,child=child,
                midpoint_restored=True,exact_rule='Canonical JSON with no tolerance; NPZ key sets, dtype, shape and bytes exactly equal.',no_retry=True)


def render(report):
    lines=['# Stage 9D: independent held-out diffusive-ladder validation','',
        '**'+report['classification']+'** — '+report['reason'],'',
        'Design SHA256: `'+report['design_sha256']+'`. The [preregistered design](dns_diffusive_ladder_heldout_design.json) and [checksum](dns_diffusive_ladder_heldout_design.sha256) precede sampling. This is one held-out trajectory plus, when eligible, its one exact-continuation check. No tuning or retry.','',
        '## Frozen methodology and analytic model','',
        'The committed Stage-9C ConditionalKernel, original DNS parameter/index kernels, strict membership, pooled-with-replacement promotion, compression estimator and ESS>=20 gate are unchanged. A read-only validation wrapper audits each returned bank before the frozen orchestrator can advance to calibration or the next attempt. It creates no transition and modifies no observation. Numeric budgets and tolerances are copied unchanged from Stage 9C, before evaluating this new three-mode model.','',
        'The prior is Uniform[-8,8]. Locations are [-3.8,-0.7,2.8], widths [0.27,0.39,0.32], and weights sigma/sum(sigma). Likelihood is max_k w_k NormalPDF(theta;mu_k,sigma_k), an equal-peak weighted-normal envelope, not a Gaussian sum. This model has not appeared in Stages 9A, 9C or 8H. Modes use fixed geometric regions bounded at -2.25 and 1.05.','',
        'At a log threshold ell below the common peak, A_ell is the clipped union of intervals mu_k +/- sigma_k sqrt(2(log_peak-ell)). Exact prior masses and CDFs are interval lengths; moments use polynomial integrals; mode probabilities use interval-region intersections. Exact posterior mode probabilities/evidence are computed by analytic Gaussian integrals on pairwise dominance cells, without reconstruction sampling.','',
        'For each attempt the accepted prefix 0..J has frozen log weights zero and log a_j=-log Xhat_j. In q(theta,j)=pi(theta) I[A_j]a_j/C, conditioning on assigned j=J cancels a_J/C and gives pi(theta)I[A_J]/X_J. Only those observations enter selection/calibration. Separate banks retain disjoint random namespaces, starts, promotion pools and histories; shared adaptive levels do not make finite samples IID.','',
        'The schedule remains eight walkers, six candidate levels, 32768 physical sweeps per bank, 1024 burn-in sweeps, no thinning, and the first 2048 post-burn assigned-top observations per walker. Rejection self-loops remain observations; no padding. Cross-walker exchange remains the Stage-9C identity transition.','',
        'The adapter\'s historical sign-based mode summaries remain saved. A separate read-only reduction of raw traces labels all three analytic regions and requires, at every disconnected generating contour and for both banks, at least five physical level-zero cross-mode returns, at least ten retained-top mode changes, and a strongly connected directed return graph across all three modes. Each witness records its actual theta-changing physical step with both adjacent assigned levels zero and its top-mode endpoints. Mode balance or ESS alone cannot certify this.','',
        'Inherited tolerances: KS<=0.08; each mode probability error<=0.07; mean error<=0.12 conditional SD; second-moment relative error<=0.12; selection/calibration compression error<=0.055; log-mass error<=0.30. Every analytic mode must remain represented. Failure is immediate at a bank boundary; no later trajectory is run to rescue it.','',
        '## Saved results','',
        '| Candidate | Threshold | Calibrated log X | Exact X | Bank | KS | Mode probabilities | ESS | Physical return witnesses |','|---|---:|---:|---:|---|---:|---|---:|---:|']
    result=report.get('result',{})
    for row in result.get('attempts',[]):
        for bank,diag in row['banks'].items():
            lines.append(f'| {row["generating_level"]+1} | {row.get("threshold","not accepted")} | {row.get("log_mass","not accepted")} | {row.get("exact_mass","not accepted")} | {bank} | {diag["KS"]} | {diag["observed_mode_probabilities"]} | {diag["top_level_ESS"]} | {len(diag["analytic_mode_diagnostics"]["witnesses"])} |')
    lines+=['','The [complete JSON](dns_diffusive_ladder_heldout.json) reports the complete threshold/log-mass/exact-mass table; exact posterior probabilities; every conditional check, compression estimate and truth error; survivor counts/CV/between-walker errors; per-level/per-walker occupancy, accepted up/down moves, physical mode changes, round trips and saved cross-mode witnesses. Raw states, event provenance, traces and full joint checkpoints are under `/tmp/dns-stage9d-heldout`.','',
        'Checks apply to generating levels J=0..5. The accepted final ell_6 is a new boundary; construction histories there were generated at ell_5. Its mass is checked against truth. No extra trajectory at the final boundary is authorized or silently treated as an ell_6 conditional stream.','',
        '## Restart, isolation and disposition','',
        'Restart: '+str(report.get('restart',{}).get('passed'))+'. New process established: '+str(report.get('restart',{}).get('new_process_established'))+'. The preregistered completed level-3 checkpoint and attempt-4 selection midpoint at sweep 8192 preserve theta, assigned index, caches, exact keys, weights, collection counters/ragged histories and both retained banks. Child PID/PPID, independent startup UUIDs, timestamps and parent launch receipt are recorded. Exact comparisons include candidates, ESS/check records, thresholds/masses/weights, retained arrays, full traces, RNG/collection bundles and checkpoint payloads. No retry.','',
        'Protected hashes before/after: '+str(report.get('protected_hashes_unchanged'))+'. No reconstruction import, export or invocation, LISA, termination evaluation or production occurred. Compression ESS can be high while walkers remain trapped in modes; survivor contributions do not prove ergodicity.','',
        'A VALIDATED result freezes this construction architecture for the tested analytic multimodal scope and authorizes one new independently designed end-to-end analytic validation combining diffusive construction, frozen Stage-7B reconstruction and frozen Stage-8R termination. That end-to-end run is not performed here. A FAILED-* result authorizes no termination or LISA follow-up.','']
    DOC.write_text('\n'.join(lines))


def write_report(spec,result,restart_result,classification,reason):
    unchanged=True
    try:load_design()
    except ValidationFailure as exc:unchanged=False;classification=exc.classification;reason=exc.reason
    tree=ast.parse(Path(__file__).read_text())
    leak=any(isinstance(n,ast.ImportFrom) and 'reconstruction' in (n.module or '') for n in ast.walk(tree))
    if leak:classification='FAILED-CONDITIONAL';reason='Forbidden reconstruction import detected'
    report=dict(stage='9D',classification=classification,reason=reason,design_sha256=sha(DESIGN),design=spec,result=result,
        restart=restart_result,protected_hashes_unchanged=unchanged,no_reconstruction_import=True,reconstruction_samples_exported=False,
        construction_architecture_frozen=classification=='VALIDATED',one_end_to_end_analytic_validation_authorized=classification=='VALIDATED',
        end_to_end_validation_performed=False,no_LISA=True,no_termination=True,no_production=True,no_tuning=True,no_retry=True)
    a.write_json(ROOT/'report.json',report);a.write_json(REPORT,report);render(report)
    print('FINAL',classification,reason,flush=True)


def run():
    spec=json.loads(DESIGN.read_text());result={};restart_result=dict(performed=False,passed=False)
    try:
        load_design();a.write_json(ROOT/'sampling_started.json',dict(design_sha256=sha(DESIGN),process=PROCESS))
        result=branch(spec,ROOT/'original')
        if result['failure']:
            write_report(spec,result,restart_result,result['failure']['classification'],result['failure']['reason']);return
        if not result['completed_all']:
            write_report(spec,result,restart_result,'FAILED-VISITATION','Full preregistered ladder not constructed');return
        load_design();restart_result=restart(spec,result)
        label='VALIDATED' if restart_result['passed'] else 'FAILED-RESTART'
        reason='Every frozen held-out requirement passed; construction architecture frozen.' if label=='VALIDATED' else 'Fresh-process exact continuation or process provenance failed.'
        write_report(spec,result,restart_result,label,reason)
    except ValidationFailure as exc:write_report(spec,result,restart_result,exc.classification,exc.reason)
    except Exception as exc:
        write_report(spec,result,restart_result,'FAILED-CONDITIONAL','Validation/implementation defect: '+type(exc).__name__+': '+str(exc))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',nargs=3);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:branch(load_design(),args.worker[2],args.worker[0],args.worker[1])
    else:run()
