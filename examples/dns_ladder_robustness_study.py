"""Stage 9A development-only sweep-budget experiment; no scientific core edits."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from blackjax.ns import dns_automatic as a
from blackjax.ns.dns import DNSParticleState
from blackjax.ns.dns_population import FrozenPopulationKernel
from examples.dns_inference_reconstruction import AnalyticModel, initial_state

ROOT = Path('/tmp/dns-stage9a-development')
DESIGN = Path('docs/examples/dns_ladder_robustness_design.json')
PARAMETERS = dict(sigma_prior=3., component_weights=[.45,.55], centers=[-2.4,2.5], widths=[.45,.55])
MODEL = 'stage9a-development-bimodal'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    old = json.loads(Path('/tmp/dns-termination-heldout-validation/design.json').read_text())
    assert old['builder'] == dict(burn_in=256, retained=1024, block_size=64, min_ess=20., target_compression=float(np.exp(-1)))
    assert PARAMETERS != old['model']
    model = AnalyticModel(MODEL, PARAMETERS)
    kernel = model.kernel()  # Bind functions only; no sampling.
    sources = dict(old['frozen_sha256'])
    sources[__file__] = sha(__file__)
    for p, h in sources.items():
        assert sha(p) == h, p
    histories = {str(p): sha(p) for p in Path('docs/examples').glob('dns*') if p.is_file()}
    design = dict(stage='9A', purpose='bounded DEVELOPMENT; NOT validation', model=PARAMETERS,
        identifier=MODEL, distinct_from_stage8H=True, held_out=False,
        prior='u~N(0,I_54); theta=3*u[0]', likelihood='sum_k c_k NormalPDF(theta;mu_k,width_k)',
        seeds=dict(construction=91401, reserved_reconstruction=91402), walkers=8,
        initialization='16 independent prior states per walker per bank, initial_state SeedSequence(seed).spawn(2); pooled promotion to 8 walkers',
        baseline=dict(source='/tmp/dns-termination-heldout-validation/design.json', source_sha256=sha('/tmp/dns-termination-heldout-validation/design.json'),
            active_builder=old['builder'], generic_api_defaults=asdict(a.LadderConfig()), sweeps_per_bank=1280),
        factors=[1,2,4], retained_per_walker=1024, retained_block_size=64,
        budgets={str(f):dict(burn_in_sweeps=256*f, transitions_between_retained=f, retained=1024, total_sweeps=1280*f) for f in [1,2,4]},
        compression_target=float(np.exp(-1)), ESS_gate=20., max_accepted_levels=6, restart_completed_level=1,
        highest_successful_definition='largest factor that completes all six levels with every reliability gate passed',
        promotion='unchanged pooled with replacement; SeedSequence-v1 purpose/bank/iteration namespace',
        kernel=kernel.metadata, numerical_contract=asdict(a.NumericalContract()), environment=a.NumericalContract().environment(),
        reconstruction='unchanged; no reconstruction streams generated or estimator evaluated in this mixing study',
        stopping='unchanged and not invoked; termination remains UNVALIDATED',
        rng='same construction seed and initial states; same iteration/bank/purpose seeds; per-walker repeated threefry split; PCG64 label and MH arrays have identical common sweep prefixes. Factor f retains indices f*256+f-1, then every f sweeps. Round-robin sweep count extends unchanged. Different retained pools/contours cause matched random inputs to act on different states after level 1; promotion draw seeds remain matched but eligible-pool indexing can diverge.',
        diagnostics=dict(mode_labels='argmax c_k NormalPDF(theta;mu_k,width_k), not posterior probabilities',
            survivor_balance='sample coefficient of variation across eight survivor fractions; nonzero walker count; largest survivor share',
            switches='successive retained dominant-component label changes per walker; also physical-sweep changes including burn-in',
            lineage='exact initial six-coordinate-block ancestry tokens, invalidated by slice updates and copied on accepted exchanges; immediate promotion hashes/donor IDs retained; not a proof of independent genealogical samples'),
        interpretations=dict(A='clear monotonic improvement in ESS / walker balance / mode switching and at least one larger budget completes all requested levels: current algorithm viable but baseline mixing too small',
            B='all 1x/2x/4x show severe walker/modal persistence and fail reliability: more transitions insufficient; kernel robustness unresolved',
            C='nonmonotonic or ambiguous: no production change justified'),
        classification_rule=dict(A='baseline fails, a larger factor completes six; at every commonly attempted level and bank ESS nondecreasing, survivor CV nonincreasing, total retained switches nondecreasing across factors; at least one strict ESS and balance improvement and a strict switching improvement at a multi-component generating contour',
            B='all three fail an ESS gate; each failing bank has survivor CV>=0.5 and at least four walkers with zero retained component switches while both components are observed in that bank',
            C='all other outcomes, including all budgets succeeding or a structural failure'),
        comparison_tolerances=dict(monotonic=0., restart='dtype/shape/bytes exact arrays; canonical JSON exact metadata, candidate/ESS/decision records; no numerical tolerance'),
        no_retries=True, no_tuning=True, historical_reports_sha256=histories, frozen_sources_sha256=sources)
    ROOT.mkdir(exist_ok=False)
    a.write_json(DESIGN, design)
    h = sha(DESIGN)
    DESIGN.with_suffix('.sha256').write_text(h+'  '+DESIGN.name+'\n')
    a.write_json(ROOT/'design.json', design)
    (ROOT/'design.sha256').write_text(h+'\n')
    print('FROZEN', h, flush=True)


def load_design():
    d = json.loads(DESIGN.read_text())
    assert sha(DESIGN) == DESIGN.with_suffix('.sha256').read_text().split()[0]
    assert sha(ROOT/'design.json') == sha(DESIGN)
    for p,h in {**d['frozen_sources_sha256'], **d['historical_reports_sha256']}.items():
        assert sha(p) == h, p
    assert a.NumericalContract().environment() == d['environment']
    return d


def labels(model, position):
    theta = model.sigma*np.asarray(position)[...,0]
    return np.argmax(np.log(model.coefficients)+norm.logpdf(theta[...,None],model.centers,model.widths),axis=-1)


class BudgetKernel(FrozenPopulationKernel):
    """Delegate every physical transition/check to frozen run; thin only its output.

    The only scientific control change is number of physical sweeps. Instrumentation
    records decisions and states without modifying the transition or random inputs.
    """
    def exchange_sweep(self, sliced, threshold, sweep, gumbels, uniforms, contract):
        # Slice updates replace exact initial block ancestry, even if values coincide.
        for w in range(8):
            blocks = self.slice_blocks(int(self.last_info.component[w]),np.asarray(self.last_info.labels[w]))
            self.tokens[w, blocks] = -1
        result, decisions = super().exchange_sweep(sliced,threshold,sweep,gumbels,uniforms,contract)
        old = self.tokens.copy()
        for dec in decisions:
            if dec['accepted']:
                x,y = dec['pair']; i,j = dec['labels']
                self.tokens[x,i],self.tokens[y,j] = old[y,j],old[x,i]
        self.sweep_modes.append(labels(self.model,result.position))
        self.lineage.append(self.tokens.copy())
        self.exchanges.append(decisions)
        return result, decisions

    def run(self, starts, threshold, streams, config, contract):
        self.sweep_modes=[]; self.lineage=[]; self.exchanges=[]
        self.tokens=np.arange(72).reshape(8,9)
        self.start_modes=labels(self.model,starts.position)
        extended=replace(config,burn_in=config.burn_in*self.factor,retained=config.retained*self.factor)
        history, diag=super().run(starts,threshold,streams,extended,contract)
        keep=np.arange(self.factor-1,extended.retained,self.factor)
        result=DNSParticleState(*[np.asarray(v)[keep].copy() for v in history])
        self.captures.append(dict(streams=streams,threshold=threshold,history=result,diagnostics=diag,
            sweep_modes=np.array(self.sweep_modes),initial_modes=self.start_modes,
            initial_block_ancestry=np.array(self.lineage),exchanges=self.exchanges))
        diag.update(retained=config.retained,budget_factor=self.factor,
            burn_in_sweeps=extended.burn_in,transitions_between_retained=self.factor)
        return result,diag


def make_kernel(model,factor):
    base=model.kernel()
    # Frozen scientific metadata remains identical: the control is external schedule.
    holder={}
    def observed_step(*args):
        values,info=base.slice_step(*args)
        holder['kernel'].last_info=info
        return values,info
    kernel=BudgetKernel(observed_step,base.evaluator,base.selector,base.exchange,base.slice_blocks,base.identity)
    holder['kernel']=kernel
    object.__setattr__(kernel,'factor',factor)
    object.__setattr__(kernel,'model',model)
    object.__setattr__(kernel,'captures',[])
    return kernel


def diagnostic(capture,threshold,model,config):
    h=capture['history']; mask=h.loglikelihood>threshold
    frac=mask.mean(axis=0); p=float(mask.mean()); modes=labels(model,h.position)
    occ=np.stack([(modes==k).sum(axis=0) for k in range(len(model.centers))],axis=1)
    switches=(modes[1:]!=modes[:-1]).sum(axis=0)
    full=capture['sweep_modes']; tokens=capture['initial_block_ancestry']
    tail=a.dns_levels.tail_diagnostics(h.loglikelihood,threshold,block_size=config.block_size)
    shares=mask.sum(axis=0)/max(1,mask.sum())
    return dict(candidate_log_threshold=threshold,compression_fraction=p,candidate_survivors=int(mask.sum()),
        ESS=tail['ess'],between_walker_SE=float(np.std(frac,ddof=1)/np.sqrt(8)),tail=tail,
        survivors_per_walker=mask.sum(axis=0),fractions_per_walker=frac,
        survivor_CV=float(np.std(frac,ddof=1)/p) if p else None,
        survivor_walkers=int(np.count_nonzero(frac)),max_survivor_share=float(shares.max()),
        mode_occupancy_per_walker=occ,mode_switches_per_walker=switches,
        physical_sweep_mode_switches_per_walker=(full[1:]!=full[:-1]).sum(axis=0),
        initial_modes=capture['initial_modes'],final_modes=full[-1],
        initial_component_final_persistence=(full[-1]==capture['initial_modes']),
        accepted_population_exchanges=capture['diagnostics']['accepted_exchanges'],
        exchange_proposals=capture['diagnostics']['exchange_proposals'],
        lineage=dict(initial_block_tokens_surviving_per_sweep=(tokens>=0).sum(axis=(1,2)),
            initial_theta_block_tokens_surviving_per_sweep=(tokens[:,:,0]>=0).sum(axis=1),
            final_initial_block_ancestry=tokens[-1]),sampling=capture['diagnostics'])


def branch(d,factor,output,checkpoint=None):
    output.mkdir(exist_ok=False,parents=True)
    model=AnalyticModel(MODEL,d['model']);kernel=make_kernel(model,factor)
    config=a.LadderConfig(**d['baseline']['active_builder'])
    state=a.load_checkpoint(checkpoint,kernel) if checkpoint else initial_state(model,kernel,d['seeds']['construction'],config)
    if not checkpoint: a.save_checkpoint(output/'initial_checkpoint',state)
    rows=[]
    while state.status=='ready' and len(state.thresholds)-1<d['max_accepted_levels']:
        iteration=state.iteration;kernel.captures.clear()
        state=a.run_ladder(state,kernel,output/'ladder',max_new_levels=1)
        raw=json.loads(state.records[-1]);folder=output/'ladder'/f'attempt_{iteration:06d}'
        threshold=raw.get('candidate',{}).get('candidate')
        # Candidate field name is selected directly from frozen record below.
        if threshold is None: threshold=raw.get('candidate',{}).get('threshold')
        row=dict(level=iteration+1,accepted=raw['accepted'],decision=raw,calibration_reached=len(kernel.captures)==2)
        if threshold is None: raise RuntimeError('Missing candidate threshold field: '+repr(raw.get('candidate')))
        for c in kernel.captures:
            name=c['streams']['bank']
            row[name]=diagnostic(c,threshold,model,config)
            np.savez_compressed(folder/(name+'_observed.npz'),
                **{k:v for k,v in zip(a.ARRAYS,c['history'])},sweep_modes=c['sweep_modes'],
                initial_block_ancestry=c['initial_block_ancestry'])
            a.write_json(folder/(name+'_exchanges.json'),c['exchanges'])
        a.write_json(folder/'study_diagnostics.json',row);rows.append(row)
        print(f'{factor}x level {iteration+1}: '+ ' '.join(f'{b} ESS={row[b]["ESS"]:.6f}' for b in a.BANKS if b in row)+f' accepted={raw["accepted"]}',flush=True)
    result=dict(factor=factor,completed_levels=len(state.thresholds)-1,completed_all=len(state.thresholds)-1==d['max_accepted_levels'] and state.status=='ready',
        status=state.status,thresholds=state.thresholds,log_masses=state.log_masses,records=[json.loads(r) for r in state.records],
        rows=rows,pid=os.getpid(),final_checkpoint=str(output/'ladder'/f'attempt_{state.iteration-1:06d}'/'checkpoint') if state.status=='ready' else None)
    a.write_json(output/'result.json',result)
    return a.json_value(result)


def classify(results,d):
    common=min(len(r['rows']) for r in results)
    monotonic=True;strictess=False;strictcv=False;strictswitch=False
    comparisons=[]
    for j in range(common):
        for bank in a.BANKS:
            if not all(bank in r['rows'][j] for r in results): continue
            vals=[r['rows'][j][bank] for r in results]
            ess=[v['ESS'] for v in vals];cv=[v['survivor_CV'] for v in vals];sw=[sum(v['mode_switches_per_walker']) for v in vals]
            ok=ess[0]<=ess[1]<=ess[2] and cv[0]>=cv[1]>=cv[2] and sw[0]<=sw[1]<=sw[2]
            monotonic &= ok;strictess |= ess[2]>ess[0];strictcv |= cv[2]<cv[0];strictswitch |= sw[2]>sw[0]
            comparisons.append(dict(level=j+1,bank=bank,ESS=ess,survivor_CV=cv,switches=sw,monotonic=ok))
    severe=[]
    for r in results:
        last=r['rows'][-1]; bank='selection' if last['decision'].get('stop_reason')=='selection_gate' else 'calibration'
        v=last.get(bank)
        severe.append(bool(not r['completed_all'] and v and v['ESS']<20 and v['survivor_CV']>=.5 and
            sum(x==0 for x in v['mode_switches_per_walker'])>=4 and all(np.sum(v['mode_occupancy_per_walker'],axis=0)>0)))
    if not results[0]['completed_all'] and any(r['completed_all'] for r in results[1:]) and monotonic and strictess and strictcv and strictswitch:case='A'
    elif all(severe):case='B'
    else:case='C'
    return dict(case=case,interpretation=d['interpretations'][case],monotonic=bool(monotonic),common_level_comparisons=comparisons,severe_persistence_at_each_failure=severe)


def restart_check(d,result):
    factor=result['factor'];root=ROOT/f'{factor}x';cp=root/'ladder'/f'attempt_{d["restart_completed_level"]-1:06d}'/'checkpoint'
    command=[sys.executable,__file__,'--worker',str(factor),str(cp)]
    with (ROOT/'fresh_process_restart.log').open('x') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    if proc.returncode:return dict(passed=False,returncode=proc.returncode,no_retry=True)
    other=json.loads((ROOT/'restart/result.json').read_text());checks={}
    for field in ['thresholds','log_masses','records']:
        checks[field]=a.encoded(result[field])==a.encoded(other[field])
    checks['new_process']=other['pid']!=result['pid']
    checks['study_diagnostics']=a.encoded(result['rows'][d['restart_completed_level']:])==a.encoded(other['rows'])
    for j in range(d['restart_completed_level'],d['max_accepted_levels']):
        x=root/'ladder'/f'attempt_{j:06d}'/'checkpoint';y=ROOT/'restart/ladder'/f'attempt_{j:06d}'/'checkpoint'
        checks[f'checkpoint_payload_{j+1}']=a.encoded(json.loads((x/'checkpoint.json').read_text()))==a.encoded(json.loads((y/'checkpoint.json').read_text()))
        with np.load(x/'banks.npz') as xx,np.load(y/'banks.npz') as yy:
            checks[f'populations_{j+1}']=all(xx[k].dtype==yy[k].dtype and xx[k].shape==yy[k].shape and xx[k].tobytes()==yy[k].tobytes() for k in xx.files)
    return dict(passed=all(checks.values()),checks=checks,command=command,no_retry=True)


def run():
    d=load_design();a.write_json(ROOT/'sampling_started.json',dict(design_sha256=sha(DESIGN),pid=os.getpid()))
    results=[]
    for factor in d['factors']:results.append(branch(d,factor,ROOT/f'{factor}x'))
    conclusion=classify(results,d)
    successful=[r for r in results if r['completed_all']]
    restart=restart_check(d,successful[-1]) if successful else dict(performed=False,reason='No budget completed all requested levels')
    load_design()
    report=dict(stage='9A',study='DEVELOPMENT, NOT VALIDATION',design=d,design_sha256=sha(DESIGN),results=results,
        conclusion=conclusion,restart=restart,termination='UNVALIDATED',historical_stage8='FAIL',historical_stage8H='FAIL',
        no_LISA=True,no_production=True,no_retry=True,no_core_modification=True,historical_reports_unchanged=True)
    a.write_json(ROOT/'results.json',report)
    a.write_json('docs/examples/dns_ladder_robustness_study.json',report)
    print('CASE',conclusion['case'],conclusion['interpretation'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',nargs=2);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:branch(load_design(),int(args.worker[0]),ROOT/'restart',args.worker[1])
    else:run()
