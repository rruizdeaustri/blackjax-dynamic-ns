"""Stage 9C: one frozen architectural development comparison, never inference."""
import argparse
from dataclasses import asdict
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

from blackjax.ns import dns, dns_automatic as a, dns_automatic_diffusive as d, dns_kernels

PROCESS=dict(pid=os.getpid(),parent_pid=os.getppid(),start_marker=str(uuid.uuid4()),
             process_start_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
             executable=sys.executable,worker_identifier='stage9c')
ROOT=Path('/tmp/dns-stage9c-development')
DESIGN=Path('docs/examples/dns_diffusive_ladder_design.json')
FROZEN=['blackjax/ns/dns.py','blackjax/ns/dns_kernels.py','blackjax/ns/dns_levels.py',
        'blackjax/ns/dns_automatic.py','blackjax/ns/dns_population.py',
        'blackjax/ns/dns_reconstruction.py','blackjax/ns/dns_termination.py',
        'docs/examples/dns_inference_reconstruction.md',
        'docs/examples/dns_inference_reconstruction_validation.json',
        'docs/examples/dns_ladder_robustness_design.json','docs/examples/dns_ladder_robustness_design.sha256',
        'examples/dns_ladder_robustness_study.py','docs/examples/dns_automatic_backtracking_audit.md']
NEW=['blackjax/ns/dns_automatic_diffusive.py',__file__,'tests/ns/test_dns_automatic_diffusive.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    if ROOT.exists() or DESIGN.exists():raise ValueError('Frozen study already exists; no retry')
    spec=dict(stage='9C',study='DEVELOPMENT ONLY',model=dict(identifier='stage9c-uniform-separated-doublewell-v1',
        prior='Uniform[-8,8]',bound=8.,center=1.65,sigma=.38,
        log_likelihood='-0.5*((abs(theta)-1.65)/0.38)**2',modes=[-1.65,1.65],
        modal_prior_weights=[.5,.5],dimension=1),
        walkers=8,initial_draws_per_walker=16,construction_seed=93271,
        methods=['fixed','diffusive'],max_new_levels=6,
        visits=dict(burn_in=1024,max_sweeps=32768,retained=2048,checkpoint_sweep=8192,checkpoint_iteration=3),
        thinning='none: retain first 2048 post-burn assigned-top observations per walker, including self-loops',
        collection='full frozen physical cap for both arms; equal per-walker top quota; no padding or retry',
        log_weight_rule='zero for every already-calibrated level, frozen before attempt',
        parameter=dict(direction_magnitude=.35,max_steps=10,max_shrinkage=100),
        exchange_rule='disabled identity for all pairs and both arms; no cross-bank interactions',
        automatic_config=dict(burn_in=1024,retained=2048,block_size=64,target_compression=float(np.exp(-1)),min_ess=20.),
        restart=dict(completed_level=3,resume_midpoint='attempt_000003 selection after 8192 sweeps',
                     process_check='parent Popen launch receipt, child PID/PPID and independent process-start UUID, timestamp and command'),
        criteria=dict(ESS_gate=20.,KS_max=.08,mode_probability_absolute=.07,
            mean_absolute_scaled_by_conditional_sd=.12,second_moment_relative=.12,
            calibration_compression_absolute=.055,selection_compression_absolute=.055,calibrated_log_mass_absolute=.30,
            require_each_mode_each_bank_each_level=True,
            deep_generating_levels=[3,4,5],minimum_genuine_excursions_each_deep_bank=5,connected_witness_level=0,
            minimum_deep_top_mode_switches_each_bank=10,
            material_improvement='at every deep generating level and bank: diffusive collected-top mode switches >= max(10,2*baseline+5); each has >=5 witnessed connected-lower cross-mode returns',
            require_full_six_levels=True,minimum_top_observations_per_walker=2048,
            require_upward_and_downward_each_bank_when_J_positive=True,
            conditional_checks='pooled KS/CDF, mean, second moment, mode fractions and exact membership at every generating contour; descriptive correlated-sample tolerances, not IID p-values',
            case_precedence=['D: incomplete ladder due to visit quota or ESS failure',
                             'B: sufficient samples fail any conditional distribution or compression truth check',
                             'C: conditional/compression pass but modal movement/representation/material improvement criteria fail',
                             'A: all full-ladder, numerical, representation, movement and material improvement criteria pass'],
            restart_required_for_A=True),
        numerical_contract=asdict(a.NumericalContract()),environment=a.NumericalContract().environment(),
        protected_sha256={p:sha(p) for p in FROZEN},integration_sha256={str(Path(p)):sha(p) for p in NEW},
        no_LISA=True,no_termination=True,no_production=True,no_reconstruction=True,no_retuning=True)
    DESIGN.parent.mkdir(parents=True,exist_ok=True)
    a.write_json(DESIGN,spec)
    Path('docs/examples/dns_diffusive_ladder_design.sha256').write_text(sha(DESIGN)+'  '+DESIGN.name+'\n')
    ROOT.mkdir()
    a.write_json(ROOT/'design.json',spec)
    (ROOT/'design.sha256').write_text(sha(DESIGN)+'\n')
    print('DESIGN FROZEN',sha(DESIGN),flush=True)


def load_design():
    spec=json.loads(DESIGN.read_text())
    if sha(DESIGN)!=(ROOT/'design.sha256').read_text().strip():raise ValueError('Design changed')
    for group in ('protected_sha256','integration_sha256'):
        for p,value in spec[group].items():
            if sha(p)!=value:raise ValueError('Frozen source changed: '+p)
    if a.NumericalContract().environment()!=spec['environment']:raise ValueError('Environment changed')
    return spec


class Toy:
    def __init__(self,spec):
        self.bound=spec['bound']; self.center=spec['center'];self.sigma=spec['sigma']

    def intervals(self,threshold):
        if np.isneginf(threshold):return [(-self.bound,self.bound)]
        radius=self.sigma*np.sqrt(max(0.,-2*threshold))
        lo=max(0.,self.center-radius); hi=min(self.bound,self.center+radius)
        return [(-hi,-lo),(lo,hi)] if lo>0 else [(-hi,hi)]

    def truth(self,threshold,values=None):
        intervals=self.intervals(threshold);length=sum(b-a for a,b in intervals)
        mean=sum((b*b-a*a)/2 for a,b in intervals)/length
        second=sum((b**3-a**3)/3 for a,b in intervals)/length
        result=dict(mass=length/(2*self.bound),mean=mean,second_moment=second,sd=np.sqrt(second-mean*mean),
                    negative_probability=sum(max(0.,min(b,0.)-a) for a,b in intervals if a<0)/length,
                    intervals=intervals,disconnected=len(intervals)>1)
        if values is not None:
            vals=np.asarray(values)
            result['cdf']=sum(np.clip(vals-a,0,b-a) for a,b in intervals)/length
        return result

    def evaluate(self,position):
        x=np.asarray(position)[...,0]
        return np.where(np.abs(x)<=self.bound,-np.log(2*self.bound),-np.inf).astype(float),(-.5*((np.abs(x)-self.center)/self.sigma)**2).astype(float)

    def parameter_step(self,spec):
        bound=self.bound; center=self.center;sigma=self.sigma
        def prior(x):return jnp.where(jnp.abs(x[0])<=bound,-jnp.log(2*bound),-jnp.inf)
        def likelihood(x):return -.5*((jnp.abs(x[0])-center)/sigma)**2
        def direction(key,x):
            del x
            return jnp.array([jnp.where(jax.random.bernoulli(key),spec['direction_magnitude'],-spec['direction_magnitude'])])
        return dns_kernels.build_constrained_slice_kernel(prior,likelihood,generate_slice_direction_fn=direction,
                                  max_steps=spec['max_steps'],max_shrinkage=spec['max_shrinkage'])


def identity(spec,method):
    return dict(version='stage9c-conditional-adapter-v1',method=method,model=spec['model'],
                parameter=spec['parameter'],visits=spec['visits'],exchange=spec['exchange_rule'],
                integration_sha256=spec['integration_sha256'])


def initial(spec,toy,method):
    banks={}
    for name,child in zip(a.BANKS,np.random.SeedSequence(spec['construction_seed']).spawn(2)):
        x=np.random.Generator(np.random.PCG64(child)).uniform(-toy.bound,toy.bound,
                            size=(spec['initial_draws_per_walker'],spec['walkers'],1))
        banks[name]=dns.DNSParticleState(x,*toy.evaluate(x))
    return a.LadderState((-np.inf,),(0.,),**banks,seed=spec['construction_seed'],
        kernel_metadata=a.encoded(identity(spec,method)),config=a.LadderConfig(**spec['automatic_config']))


def summarize_bank(spec,toy,folder,bank,threshold,candidate):
    p=folder/(bank+'_top_observations.npz')
    if not p.exists():return dict(sufficient_top_visits=False,visitation=json.loads((folder/(bank+'_visitation.json')).read_text()))
    with np.load(p,allow_pickle=False) as z:
        values=z['position'][...,0]; ll=z['loglikelihood']; assigned=z['assigned_level']; events=z['event_id']
        truth=toy.truth(threshold,np.sort(values.ravel())); cdf=truth.pop('cdf'); n=values.size
        ks=float(max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n)))
        mean=float(values.mean());second=float(np.mean(values**2));mode=float(np.mean(values<0))
        tail=a.dns_levels.tail_diagnostics(ll,candidate,spec['automatic_config']['block_size'])
        per_mode=[[int((values[:,w]<0).sum()),int((values[:,w]>=0).sum())] for w in range(values.shape[1])]
        top_switches=int(((values[1:]<0)!=(values[:-1]<0)).sum())
        tests=dict(KS=ks<=spec['criteria']['KS_max'],
                   mode_probability=abs(mode-truth['negative_probability'])<=spec['criteria']['mode_probability_absolute'],
                   mean=abs(mean-truth['mean'])<=spec['criteria']['mean_absolute_scaled_by_conditional_sd']*truth['sd'],
                   second_moment=abs(second-truth['second_moment'])/truth['second_moment']<=spec['criteria']['second_moment_relative'],
                   contour_membership=bool(np.all(ll>threshold)),assigned_top=bool(np.all(assigned==assigned[0,0])),
                   unique_events=len(np.unique(events))==events.size,each_mode_represented=bool(np.any(values<0) and np.any(values>=0)))
        return dict(sufficient_top_visits=True,conditional_truth=truth,KS=ks,mean=mean,second_moment=second,
                    negative_probability=mode,checks=tests,conditional_pass=all(v for k,v in tests.items() if k!='each_mode_represented'),
                    top_level_ESS=tail['ess'],candidate_survivors_per_walker=(ll>candidate).sum(0),
                    between_walker_standard_error=float(np.std((ll>candidate).mean(0),ddof=1)/np.sqrt(values.shape[1])),
                    top_mode_occupancy_per_walker=per_mode,top_mode_switches=top_switches,
                    visitation=json.loads((folder/(bank+'_visitation.json')).read_text()))


def branch(spec,method,output,checkpoint=None,resume_selection=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    a.write_json(output/'process.json',{**PROCESS,'worker_identifier':method,'design_sha256':sha(DESIGN)})
    toy=Toy(spec['model']);step=toy.parameter_step(spec['parameter'])
    if checkpoint:
        stub=type('Identity',(),{'metadata':identity(spec,method)})()
        state=a.load_checkpoint(checkpoint,stub)
        envelope=json.loads((Path(checkpoint)/'joint_manifest.json').read_text())
        for rel,value in envelope['files'].items():
            if sha(Path(checkpoint).parent/rel)!=value:raise ValueError('Joint companion checksum mismatch')
    else:
        state=initial(spec,toy,method)
        a.save_checkpoint(output/'initial_checkpoint',state)
    start=state.iteration
    initial_hashes={n:a.particle_hash(getattr(state,n)) for n in a.BANKS}
    summaries=[]
    while state.status=='ready' and state.iteration<spec['max_new_levels']:
        j=state.iteration;folder=output/f'attempt_{j:06d}';old=state.thresholds[-1]
        state=d.build_next_level(state,step,toy.evaluate,identity(spec,method),method,d.VisitConfig(**spec['visits']),folder,
                                 resume_selection if j==start else None)
        record=json.loads(state.records[-1]);candidate=record.get('candidate',{}).get('threshold')
        row=dict(generating_level=j,accepted=record['accepted'],stop_reason=record.get('stop_reason'),generating_threshold=old)
        if candidate is not None:
            row.update(candidate=candidate,analytic_compression=toy.truth(candidate)['mass']/toy.truth(old)['mass'])
            for bank in a.BANKS:
                if (folder/(bank+'_visitation.json')).exists():row[bank]=summarize_bank(spec,toy,folder,bank,old,candidate)
                else:row[bank]=dict(not_reached=True)
            if record['accepted']:
                row.update(selection_compression=record['candidate']['ratio'],calibration_compression=record['calibration']['ratio'],
                           calibrated_log_mass=state.log_masses[-1],analytic_log_mass=float(np.log(toy.truth(candidate)['mass'])))
                c=spec['criteria'];row['compression_checks']=dict(
                    selection=abs(row['selection_compression']-row['analytic_compression'])<=c['selection_compression_absolute'],
                    calibration=abs(row['calibration_compression']-row['analytic_compression'])<=c['calibration_compression_absolute'],
                    log_mass=abs(row['calibrated_log_mass']-row['analytic_log_mass'])<=c['calibrated_log_mass_absolute'])
        summaries.append(row)
        a.write_json(folder/'analytic_checks.json',row)
        print(method,'level',j+1,'accepted',row['accepted'],
              'ESS',[row.get(b,{}).get('top_level_ESS') for b in a.BANKS],
              'top switches',[row.get(b,{}).get('top_mode_switches') for b in a.BANKS],flush=True)
    result=dict(method=method,completed_levels=len(state.thresholds)-1,completed_all=len(state.thresholds)-1==spec['max_new_levels'] and state.status=='ready',
                thresholds=state.thresholds,log_masses=state.log_masses,records=[json.loads(r) for r in state.records],summaries=summaries,
                initial_bank_hashes=initial_hashes,process=PROCESS)
    a.write_json(output/'result.json',result)
    return a.json_value(result)


def restart(spec,original):
    checkpoint=ROOT/'diffusive'/f'attempt_{spec["restart"]["completed_level"]-1:06d}'/'checkpoint'
    midpoint=ROOT/'diffusive/attempt_000003/selection_midpoint'
    output=ROOT/'restart'
    command=[sys.executable,'-m','examples.dns_diffusive_ladder_development','--worker',str(checkpoint),str(midpoint),str(output)]
    launch=datetime.now(timezone.utc).isoformat()
    with (ROOT/'restart.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        receipt=dict(parent=PROCESS,child_pid=child.pid,command=command,launch_utc=launch)
        a.write_json(ROOT/'restart_launch.json',receipt)
        code=child.wait()
    if code:return dict(passed=False,launch=receipt,returncode=code,no_retry=True)
    other=json.loads((output/'result.json').read_text()); childmeta=json.loads((output/'process.json').read_text())
    process=(childmeta['pid']==receipt['child_pid'] and childmeta['parent_pid']==PROCESS['pid']
             and childmeta['start_marker']!=PROCESS['start_marker']
             and childmeta['process_start_utc']>=launch)
    checks={k:a.encoded(original[k])==a.encoded(other[k]) for k in ('thresholds','log_masses','records')}
    checks['summaries']=a.encoded(original['summaries'][3:])==a.encoded(other['summaries'])
    arrays={}
    for j in range(3,spec['max_new_levels']):
        x=ROOT/'diffusive'/f'attempt_{j:06d}';y=output/f'attempt_{j:06d}'
        for bank in a.BANKS:
            for name in (bank+'_trace.npz',bank+'_top_observations.npz',bank+'_final_state/joint.npz'):
                with np.load(x/name,allow_pickle=False) as aa,np.load(y/name,allow_pickle=False) as bb:
                    arrays[f'{j}/{name}']=set(aa.files)==set(bb.files) and all(aa[k].dtype==bb[k].dtype and aa[k].shape==bb[k].shape and aa[k].tobytes()==bb[k].tobytes() for k in aa.files)
            checks[f'{j}/{bank}_state_metadata']=a.encoded(json.loads((x/(bank+'_final_state/state.json')).read_text()))==a.encoded(json.loads((y/(bank+'_final_state/state.json')).read_text()))
        checks[f'{j}/checkpoint']=a.encoded(json.loads((x/'checkpoint/checkpoint.json').read_text()))==a.encoded(json.loads((y/'checkpoint/checkpoint.json').read_text()))
    checks.update(arrays)
    return dict(passed=process and all(checks.values()),genuine_new_process=process,checks=checks,launch=receipt,child=childmeta,
                midpoint_restored=True,no_retry=True)


def decision(spec,fixed,diffusive,restart_result):
    comparisons=[]; rows=diffusive['summaries']
    if not diffusive['completed_all']:
        stop=diffusive['records'][-1]
        integrity=stop.get('stop_reason')=='integrity_failure' and 'insufficient_top_level_visits' not in stop.get('error','')
        return dict(case='B' if integrity else 'D',reason='Implementation integrity failed' if integrity else 'Incomplete reliable ladder: visit quota or ESS gate failed',comparison=comparisons)
    conditional=all(row[b]['conditional_pass'] for row in rows for b in a.BANKS)
    compression=all(all(row['compression_checks'].values()) for row in rows)
    if not conditional or not compression:return dict(case='B',reason='Conditional distribution or compression truth checks failed',comparison=comparisons)
    movement=True;representation=all(row[b]['checks']['each_mode_represented'] for row in rows for b in a.BANKS)
    for row in rows:
        j=row['generating_level']
        for b in a.BANKS:
            visit=row[b]['visitation']
            if j>0:movement &= visit['upward_moves']>0 and visit['downward_moves']>0
            if j in spec['criteria']['deep_generating_levels']:
                baseline=next((v for v in fixed['summaries'] if v['generating_level']==j),None)
                baseline_switches=None if baseline is None else baseline.get(b,{}).get('top_mode_switches')
                count=len(visit['genuine_cross_mode_excursions'])
                switches=row[b]['top_mode_switches']
                passed=baseline_switches is not None and switches>=max(10,2*baseline_switches+5) and count>=5 and row[b]['conditional_truth']['disconnected']
                movement &= passed
                comparisons.append(dict(generating_level=j,bank=b,fixed_top_switches=baseline_switches,diffusive_top_switches=switches,
                                        genuine_excursions=count,passed=passed))
    if not movement or not representation:return dict(case='C',reason='Conditional/compression checks pass but robust deep movement comparison fails',comparison=comparisons)
    if not restart_result.get('passed'):raise RuntimeError('Scientific criteria passed but required restart failed; no CASE A declaration')
    return dict(case='A',reason='Full ladder, ESS, analytic conditional/compression checks, representation, genuine excursions and deep comparison pass',comparison=comparisons)


def render(report):
    lines=['# Stage 9C: diffusive automatic-ladder development','',
           '**DEVELOPMENT ONLY. CASE '+report['decision']['case']+'.** '+report['decision']['reason'],'',
           'The pre-sampling [design](dns_diffusive_ladder_design.json) and [SHA256](dns_diffusive_ladder_design.sha256) freeze all model, seed, schedule and acceptance choices. No tuning or trajectory retry occurred. Stage 9A remains CASE C and Stage 9B remains classification B.','',
           '## Measure and implementation','',
           'For each attempt, freeze the already-calibrated prefix 0..J, zero log weights, and log a_j = -log Xhat_j. Reuse dns.build_kernel with the unchanged constrained-prior slice callback and its original neighboring-level MH kernel. Then q(theta,j)=pi(theta) I[A_j] a_j/C and q(j)=a_j X_j/C, so q(theta|j=J)=pi(theta) I[A_J]/X_J exactly: the a_J/C factors cancel. Calibration measures X_(J+1)/X_J using only assigned-J observations from its own bank. Weights are frozen per attempt, not adapted within chains.','',
           'The fixed arm uses the existing dns_automatic.build_next_level scalar-contour orchestration, the same parameter callback and parameter-key layout, with its assigned level held at J. The diffusive arm binds the full prefix in an isolated adapter and delegates promotion, quantile selection, independent calibration, strict comparisons and ESS>=20 gates to unchanged code. This controlled one-coordinate toy does not invoke the 54-coordinate LISA population adapter.','',
           'Cross-walker exchange is disabled as the identity in both arms. Identity preserves the product joint target for every assigned-index vector, including mismatched levels. This isolates backtracking and makes no assertion that the common-contour exchange is valid for different indices. Independent walkers share only the frozen table; banks use disjoint namespaces, histories and promotion pools. Shared/adaptive levels do not imply unconditional IID samples.','',
           '## Frozen schedule and diagnostics','',
           'Each bank runs 32768 physical sweeps, with 1024 joint-sweep burn-in, no thinning, and the first 2048 post-burn assigned-top events per walker retained. Consecutive stays and self-loops are included; lower-level states are excluded. All eight quotas must be met or the attempt fails without retry. Both arms have the same physical cap and retained count; top-level counts differ. Per-walker visit order supplies the time axis for unchanged batch-mean/exceedance ESS. These ESS estimates and correlated-data CDF tolerances are approximate diagnostics; rare inter-mode correlation can outlast a block, so ESS is separately supplemented by physical mode/level traces and witnessed excursions.','',
           'Every event archive records position, cached prior/likelihood, assigned index, walker, event ID, physical sweep and promoted donor/index/source hash. Visitation reports include per-level/per-walker occupancy, up/down proposals and accepts, acceptance, top visits/retention, round trips, physical mode occupancy/switches and explicit top->connected-lower->different-mode->top witnesses. Analytic checks add candidate-survivor counts and between-walker standard errors.','',
           'The new toy is Uniform[-8,8] with log L=-0.5*((abs(theta)-1.65)/0.38)^2. For radius r=0.38 sqrt(-2 ell), the contour is the union of intervals around +/-1.65 clipped to prior support; overlapping intervals merge. Truth is computed by exact interval lengths, polynomial integrals and clipped-linear CDFs, independently of MCMC. Mode masses are symmetric. Six candidates are requested; deep generating indices 3,4,5 must be disconnected.','',
           '## Saved development results','',
           '| Method | Candidate level | Selection ESS | Calibration ESS | Top switches selection/calibration | Conditional checks | Compression checks |','|---|---:|---:|---:|---|---|---|']
    for result in report['results']:
        for row in result['summaries']:
            s=row.get('selection',{});c=row.get('calibration',{})
            lines.append(f'| {result["method"]} | {row["generating_level"]+1} | {s.get("top_level_ESS")} | {c.get("top_level_ESS")} | {s.get("top_mode_switches")}/{c.get("top_mode_switches")} | {s.get("conditional_pass")}/{c.get("conditional_pass")} | {row.get("compression_checks")} |')
    lines+=['','Full conditional CDF/moment/mode checks, compression and log-mass estimates against truth, level visitation, witnesses and predeclared deep comparisons are in the [JSON report](dns_diffusive_ladder_development.json). Compression ESS can be high while walkers remain in separate modes; all walkers contributing survivors does not prove cross-mode ergodicity.','',
            '## Restart and isolation','',
            'The completed level-3 checkpoint preserves both banks and links exact terminal joint states, indices, keys, exchange RNG state (identity/unconsumed), weights, counters and histories. A selection-chain snapshot at attempt 4 after sweep 8192 additionally preserves the partial trace/collection and exact PRNG keys. The child loads level 3 and restores that midpoint, then continues attempts 4–6; no restart from a fresh random trajectory. Each subsequent attempt performs the prescribed bank-local promotion and independent counter-derived stream initialization.','',
            'Restart passed: **'+str(report['restart'].get('passed'))+'**. Genuinely new process established: **'+str(report['restart'].get('genuine_new_process'))+'**. Parent Popen receipt, child PID/PPID, independently generated process-start UUIDs, timestamps and full command are recorded. Exact comparisons cover thresholds, masses, construction records, analytic summaries, complete physical traces, retained events, final joint bundles and automatic checkpoint payloads.','',
            'Protected source hashes, including original parameter/index kernels, compression/promotion/gates, CPU/x64 contract and dns_reconstruction.py, were verified unchanged. No reconstruction module is imported or called by this driver; no construction observations are exported as reconstruction records. Stage-7B fresh fixed-contour MIS remains separate and unchanged. No termination criterion, LISA or production run occurred.','',
            '## Disposition','',
            'Independent held-out validation justified: **'+str(report['heldout_validation_justified'])+'**. This report authorizes no execution of that validation; CASE A supports one separately frozen independent validation, while other cases do not. No further run or tuning is part of Stage 9C.','']
    Path('docs/examples/dns_diffusive_ladder_development.md').write_text('\n'.join(lines))


def run():
    spec=load_design()
    a.write_json(ROOT/'sampling_started.json',dict(design_sha256=sha(DESIGN),process=PROCESS))
    fixed=branch(spec,'fixed',ROOT/'fixed')
    diff=branch(spec,'diffusive',ROOT/'diffusive')
    restart_result=restart(spec,diff) if diff['completed_all'] else dict(performed=False,passed=False,reason='No complete diffusive ladder; no retry')
    chosen=decision(spec,fixed,diff,restart_result)
    load_design()
    report=dict(stage='9C',design_sha256=sha(DESIGN),design=spec,results=[fixed,diff],restart=restart_result,decision=chosen,
                matched_initial_banks=fixed['initial_bank_hashes']==diff['initial_bank_hashes'],
                heldout_validation_justified=chosen['case']=='A',protected_sources_unchanged=True,
                reconstruction_samples_exported=False,no_LISA=True,no_termination=True,no_production=True,no_retry=True)
    a.write_json(ROOT/'report.json',report)
    a.write_json('docs/examples/dns_diffusive_ladder_development.json',report)
    render(report)
    print('FINAL CASE',chosen['case'],chosen['reason'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',nargs=3);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        spec=load_design();branch(spec,'diffusive',args.worker[2],args.worker[0],args.worker[1])
    else:run()
