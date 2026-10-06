"""Stage 8 analytic-only validation; frozen Stage-7B inputs and kernels."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from blackjax.ns import dns_automatic as a, dns_reconstruction as r
from blackjax.ns.dns_termination import TerminationConfig, stopping_decision
from examples.dns_inference_reconstruction import AnalyticModel, summarize, accuracy_checks

SOURCE = Path('/tmp/dns-stage7b-validation')
FIELDS = ('position','logprior','loglikelihood','origin_level','walker','draw','event_id')
JUSTIFICATION = 'Sum of component weights / (sqrt(2*pi)*component widths), a global Gaussian-density bound'


def design():
    old = json.loads((SOURCE/'design.json').read_text())
    frozen = dict(old['frozen_sha256'])
    for path in ['blackjax/ns/dns_reconstruction.py','examples/dns_inference_reconstruction.py']:
        frozen[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(stage='8', models=old['models'], tolerances=old['tolerances'],
                frozen_sha256=frozen, production_candidate_fraction=.05,
                diagnostic_fractions=[.2,.005], mass_SE_multiplier=2.,
                max_level=12, deeper_additional_levels=2,
                deeper_tolerances=dict(logZ=.05,cdf=.04,mode_probability=.04,
                    mean=.08,variance=.15,quantile=.15),
                reconstruction=old['reconstruction'], seeds=old['seeds'],
                source=str(SOURCE), restart_level=3,
                remainder_validation='Absolute evidence error to truth and deeper reference <= mass-envelope cap; exact terminal integral <= cap',
                uncertainty='Products of ratio +/- 2 recorded SE, clipped to (0,1); descriptive, not simultaneous confidence bounds')


def envelopes(state,k):
    rows=[json.loads(v)['calibration'] for v in state.records]
    ratios=np.array([v['ratio'] for v in rows]); se=np.array([v['standard_error'] for v in rows])
    return (np.r_[0.,np.cumsum(np.log(np.maximum(ratios-k*se,1e-12)))],
            np.r_[0.,np.cumsum(np.log(np.minimum(ratios+k*se,1-1e-12)))])


def decision(state, records, model, d, fraction=None):
    if not np.array_equal(records.thresholds, np.array(state.thresholds)) or not np.array_equal(records.log_masses, np.array(state.log_masses)):
        raise ValueError('Termination records must match the current frozen ladder')
    lo,hi=envelopes(state,d['mass_SE_multiplier'])
    upper=float(np.log(np.sum(model.coefficients/(np.sqrt(2*np.pi)*model.widths))))
    return stopping_decision(records,upper,TerminationConfig(
        d['production_candidate_fraction'] if fraction is None else fraction,JUSTIFICATION),
        log_mass_lower=lo,log_mass_upper=hi)


def append_level(state,kernel,records,checkpoint,seed,config,output,name):
    j=len(state.thresholds)-1; threshold=state.thresholds[-1]
    streams=a.streams(seed,j,'calibration',8)
    starts,promotion=a.promote(state.calibration,threshold,streams['promotion_seed'],state.contract)
    history,diagnostic=kernel.run(starts,threshold,streams,config,state.contract)
    state.contract.validate(history,threshold,kernel.evaluate(history.position))
    n=history.loglikelihood.size; draws=np.repeat(np.arange(config.retained),8); walkers=np.tile(np.arange(8),config.retained)
    values=dict(position=history.position.reshape(n,54),logprior=history.logdensity.reshape(n),
        loglikelihood=history.loglikelihood.reshape(n),origin_level=np.full(n,j,dtype=int),
        walker=walkers,draw=draws,event_id=np.array([f'{name}/{seed}/{j}/{t}/{w}' for t,w in zip(draws,walkers)]))
    metadata=dict(records.metadata)
    metadata['levels']=records.metadata['levels']+[dict(level=j,threshold=threshold,
        stream_id=f'{name}/{seed}/{j}',streams=streams,diagnostics=diagnostic,
        originating_threshold=threshold,source_bank='fresh-inference',
        source_history_hash=a.particle_hash(state.calibration))]
    metadata['mass_status']=['prior']+['calibrated']*j
    metadata['checkpoint_history_hashes']=dict(records.metadata['checkpoint_history_hashes'],
        **{str(j):a.digest((checkpoint/'banks.npz').read_bytes())})
    combined=r.ReconstructionRecords(np.array(state.thresholds),np.array(state.log_masses),
        **{k:np.concatenate([getattr(records,k),values[k]]) for k in FIELDS},metadata=metadata).validate()
    r.save_records(output/'records',combined)
    a.write_json(output/'preflight.json',dict(streams=streams,promotion=promotion))
    return combined


def run_branch(name,d,output,*,deeper=0,start_checkpoint=None,start_records=None):
    output.mkdir(parents=True,exist_ok=False)
    model=AnalyticModel(name,d['models'][name]); kernel=model.kernel()
    cp=Path(start_checkpoint or SOURCE/name/'continuous/attempt_000002/checkpoint')
    state=a.load_checkpoint(cp,kernel)
    records=r.load_records(Path(start_records or SOURCE/name/'inference/records'))
    cfg=d['reconstruction']; config=a.LadderConfig(burn_in=cfg['burn_in'],retained=cfg['retained'],block_size=cfg['block_size'])
    seed=d['seeds']['reconstruction_'+name]; trajectory=[]; stop=None
    while True:
        metric=decision(state,records,model,d)
        summary,estimate=summarize(model,records)
        trajectory.append(dict(level=len(state.thresholds)-1,metric=metric,summary=summary,
            diagnostic_metrics=[decision(state,records,model,d,v) for v in d['diagnostic_fractions']]))
        print(name,output.name,'level',metric['level'],'fraction',metric['remaining_fraction'],flush=True)
        if metric['stop'] and stop is None:stop=metric['level']
        if stop is not None and metric['level']>=stop+deeper:break
        if metric['level']>=d['max_level']:raise RuntimeError('Predeclared maximum reached without stop')
        state=a.run_ladder(state,kernel,output/'ladder',max_new_levels=1)
        if state.status!='ready':raise RuntimeError('Frozen automatic construction gate failed')
        cp=output/'ladder'/f'attempt_{state.iteration-1:06d}'/'checkpoint'
        records=append_level(state,kernel,records,cp,seed,config,output/f'level_{len(state.thresholds)-1}',name)
    r.save_records(output/'final_records',records)
    np.savez_compressed(output/'weights.npz',weights=estimate['weights'],log_importance=estimate['log_importance'])
    result=dict(stop_level=stop,final_checkpoint=str(cp),metric=metric,summary=summary,
        thresholds=state.thresholds,log_masses=state.log_masses,trajectory=trajectory)
    a.write_json(output/'result.json',result)
    return result


def posterior_distance(model,x,y):
    ex=r.reconstruct_evidence(x);ey=r.reconstruct_evidence(y)
    grid=np.unique(np.r_[model.sigma*x.position[:,0],model.sigma*y.position[:,0]])
    def cdf(v,e):
        theta=model.sigma*v.position[:,0]; order=np.argsort(theta,kind='stable'); cs=np.r_[0.,np.cumsum(e['weights'][order])]
        return cs[np.searchsorted(theta[order],grid,side='right')]
    return float(np.max(abs(cdf(x,ex)-cdf(y,ey))))


def validate(d,root):
    results={}
    for name in d['models']:
        stopped=run_branch(name,d,root/name/'continuous')
        # New interpreter, checkpoint BEFORE stop, same reconstruction streams.
        subprocess.run([sys.executable,'-m','examples.dns_automatic_termination','--worker',name,
            '--output',str(root/name/'restarted'),'--design',str(root/'design.json')],check=True,env=os.environ.copy())
        restarted=json.loads((root/name/'restarted/result.json').read_text())
        deep=run_branch(name,d,root/name/'deeper',deeper=d['deeper_additional_levels'],
            start_checkpoint=stopped['final_checkpoint'],start_records=root/name/'continuous/final_records')
        model=AnalyticModel(name,d['models'][name]); truth=model.reference();s=stopped['summary'];v=deep['summary']; tol=d['deeper_tolerances']
        x=r.load_records(root/name/'continuous/final_records');y=r.load_records(root/name/'restarted/final_records');z=r.load_records(root/name/'deeper/final_records')
        exact=lambda p,q:np.asarray(p).dtype==np.asarray(q).dtype and np.asarray(p).shape==np.asarray(q).shape and np.asarray(p).tobytes()==np.asarray(q).tobytes()
        restart_checks={k:exact(getattr(x,k),getattr(y,k)) for k in ('thresholds','log_masses')+FIELDS}
        restart_checks.update(metadata=x.metadata==y.metadata,decision=a.encoded(stopped['metric'])==a.encoded(restarted['metric']),
            summaries=a.encoded(s)==a.encoded(restarted['summary']),stop_level=stopped['stop_level']==restarted['stop_level'])
        with np.load(root/name/'continuous/weights.npz') as w,np.load(root/name/'restarted/weights.npz') as u:
            restart_checks['weights']=all(exact(w[k],u[k]) for k in w.files)
        boundary=run_branch(name,d,root/name/'boundary',start_checkpoint=stopped['final_checkpoint'],start_records=root/name/'continuous/final_records')
        restart_checks['boundary_no_new_level']=boundary['stop_level']==stopped['stop_level'] and not (root/name/'boundary/ladder').exists()
        changes=dict(logZ=abs(v['logZ']-s['logZ']),cdf=posterior_distance(model,x,z),
            mode_probability=float(np.max(abs(np.array(v['component_probabilities'])-s['component_probabilities']))),
            mean=abs(v['mean']-s['mean']),variance=abs(v['variance']-s['variance']),
            quantile=float(np.max(abs(np.array(v['quantiles'])-s['quantiles']))))
        cap=stopped['metric']['remaining_evidence']; threshold=stopped['metric']['threshold']
        # Independent terminal contour integral; break quadrature at contour roots.
        from scipy.optimize import brentq
        grid=np.linspace(-20*model.sigma,20*model.sigma,40001); vals=model.log_likelihood(grid)-threshold
        roots=[brentq(lambda t:float(model.log_likelihood(t)-threshold),grid[i],grid[i+1]) for i in np.flatnonzero(vals[:-1]*vals[1:]<0)]
        terminal=0.
        for low,high in zip(roots[:-1],roots[1:]):
            if model.log_likelihood((low+high)/2)>threshold:
                terminal+=quad(lambda t:norm.pdf(t,scale=model.sigma)*np.exp(model.log_likelihood(t)),low,high,epsabs=1e-12)[0]
        errors=dict(to_truth=abs(np.exp(s['logZ'])-truth['Z']),to_deeper=abs(np.exp(s['logZ'])-np.exp(v['logZ'])),exact_terminal=terminal)
        checks=accuracy_checks(name,s,d['tolerances'])
        checks.update(deeper_stable=all(changes[k]<=tol[k] for k in changes),
            remainder_conservative=all(e<=cap for e in errors.values()),restart=all(restart_checks.values()),automatic_stop=True)
        results[name]=dict(decision='PASS' if all(checks.values()) else 'FAIL',checks=checks,exact=truth,
            stopped=stopped,deeper=deep,deep_changes=changes,remainder_comparison=errors,
            restart=restart_checks,mass_uncertainty=d['uncertainty'])
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in d['frozen_sha256'].items())
    result=dict(stage='8',decision='PASS' if unchanged and all(v['decision']=='PASS' for v in results.values()) else 'FAIL',
        design=d,design_sha256=a.digest((root/'design.json').read_bytes()),models=results,frozen_hashes_unchanged=unchanged,LISA_model_calls=0)
    a.write_json(root/'results.json',result)
    return result


def audit_saved(result, root):
    """Read-only detailed audit; never starts a kernel or a sampling stream."""
    from types import SimpleNamespace
    d = result['design']
    for name, row in result['models'].items():
        model = AnalyticModel(name, d['models'][name])
        branch = root/name
        continuous = row['stopped']
        restarted = json.loads((branch/'restarted/result.json').read_text())
        boundary = json.loads((branch/'boundary/result.json').read_text())
        row['restart_execution'] = dict(fresh_process=True,
            start_checkpoint=str(SOURCE/name/'continuous/attempt_000002/checkpoint'),
            start_records=str(SOURCE/name/'inference/records'),
            configuration_sha256=result['design_sha256'],
            boundary_added_levels=0,
            command=[sys.executable, '-m', 'examples.dns_automatic_termination',
                     '--worker', name, '--output', str(branch/'restarted'),
                     '--design', str(root/'design.json')])
        cp = Path(continuous['final_checkpoint'])
        rp = Path(restarted['final_checkpoint'])
        left = json.loads((cp/'checkpoint.json').read_text())
        right = json.loads((rp/'checkpoint.json').read_text())
        row['restart'].update(
            checkpoint_payload=left['payload'] == right['payload'],
            checkpoint_hash=left['sha256'] == right['sha256'],
            retained_banks=(cp/'banks.npz').read_bytes() == (rp/'banks.npz').read_bytes(),
            boundary_decision=a.encoded(continuous['metric']) == a.encoded(boundary['metric']),
            boundary_summary=a.encoded(continuous['summary']) == a.encoded(boundary['summary']))
        state = SimpleNamespace(records=[a.encoded(v) for v in left['payload']['records']])
        lower, upper = envelopes(state, d['mass_SE_multiplier'])
        records = r.load_records(branch/'continuous/final_records')
        likelihood_upper = float(np.log(np.sum(model.coefficients/(np.sqrt(2*np.pi)*model.widths))))
        nominal = stopping_decision(records, likelihood_upper,
            TerminationConfig(d['production_candidate_fraction'], JUSTIFICATION))
        row['likelihood_upper'] = dict(log_value=likelihood_upper,
            value=float(np.exp(likelihood_upper)), justification=JUSTIFICATION,
            observed_maximum_is_not_a_certificate=True)
        theta = model.sigma*records.position[:, 0]
        row['mode_representation'] = dict(
            retained_negative=int(np.sum(theta < 0)),
            retained_nonnegative=int(np.sum(theta >= 0)),
            per_origin_level=[dict(level=j,
                negative=int(np.sum((theta < 0) & (records.origin_level == j))),
                nonnegative=int(np.sum((theta >= 0) & (records.origin_level == j))))
                for j in range(len(records.thresholds))],
            component_probabilities=continuous['summary']['component_probabilities'],
            region_probabilities=continuous['summary']['region_probabilities'],
            explanation='Lower fixed-contour strata retain secondary-mode support even when the deepest contour excludes it.')
        true_mass = model.contour_mass(continuous['metric']['threshold'])
        row['mass_uncertainty'] = dict(method=d['uncertainty'],
            log_mass_lower=lower.tolist(), log_mass_upper=upper.tolist(),
            nominal_stop_metric=nominal,
            descriptive_envelope_stop_metric=continuous['metric'],
            actual_contour_mass=true_mass,
            envelope_encloses_actual_stop_mass=bool(np.exp(lower[-1]) <= true_mass <= np.exp(upper[-1])),
            actual_cap_posterior_probability=row['remainder_comparison']['exact_terminal']/row['exact']['Z'],
            terminal_bound_conservative=row['remainder_comparison']['exact_terminal'] <= continuous['metric']['remaining_evidence'],
            total_error_bound_conservative=row['checks']['remainder_conservative'],
            simultaneous_confidence_guarantee=False)
        for fraction in d['diagnostic_fractions']:
            index = d['diagnostic_fractions'].index(fraction)
            levels = [v['level'] for v in continuous['trajectory'] + row['deeper']['trajectory']
                      if v['diagnostic_metrics'][index]['stop']]
            row.setdefault('diagnostic_first_trigger', {})[str(fraction)] = min(levels) if levels else None
        row['checks']['restart'] = all(row['restart'].values())
        row['decision'] = 'PASS' if all(row['checks'].values()) else 'FAIL'
    result['decision'] = 'PASS' if result['frozen_hashes_unchanged'] and all(
        row['decision'] == 'PASS' for row in result['models'].values()) else 'FAIL'
    result['predeclared_design_unchanged'] = (
        a.digest((root/'design.json').read_bytes()) == result['design_sha256']
        and json.loads((root/'design.json').read_text()) == d)
    result['second_posterior_criterion_introduced'] = False
    result['interpretation'] = ('A terminal-contour integral bound is not a bound on '
        'calibration or finite-sample deterministic-mixture reconstruction error. '
        'Do not promote the candidate to production if any required check fails.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--design',type=Path);p.add_argument('--worker',choices=['gaussian','mixture']);p.add_argument('--predeclare',action='store_true');args=p.parse_args()
    if args.predeclare:
        args.output.mkdir(parents=True,exist_ok=False);a.write_json(args.output/'design.json',design());return
    d=json.loads((args.design or args.output/'design.json').read_text())
    for path,h in d['frozen_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    if args.worker:run_branch(args.worker,d,args.output);return
    result=audit_saved(validate(d,args.output), args.output)
    Path('docs/examples/dns_automatic_termination_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL',result['decision'],flush=True)

if __name__=='__main__':main()
