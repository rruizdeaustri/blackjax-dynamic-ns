"""Read-only final Stage-10A report from immutable run artifacts; no sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/tmp/lisa-gb0004-stage10a-fixed-ladder')
DOC=Path('docs/examples/lisa_gb0004_fixed_ladder_inference.md')
REPORT=DOC.with_suffix('.json')
DESIGN=Path('docs/examples/lisa_gb0004_fixed_ladder_inference_design.json')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def generate(case,interpretation):
    assert case in ('A','B','C','D')
    design=json.loads(DESIGN.read_text());failure=None
    if (ROOT/'failure.json').exists():failure=json.loads((ROOT/'failure.json').read_text());assert case=='D'
    result={'stage':'10A','case':case,'classification':{'A':'FIXED-LADDER POSTERIOR STABLE',
        'B':'POSTERIOR STABLE / EVIDENCE UNRESOLVED','C':'POSTERIOR UNSTABLE',
        'D':'RECONSTRUCTION / NUMERICAL FAILURE'}[case],
        'design_sha256':sha(DESIGN),'driver_sha256':design['driver_sha256'],
        'interpretation':interpretation,'failure':failure,'sets':{},'estimates':None,
        'evidence_convention':design['evidence_convention'],'no_retry':True,
        'historical_samples_used':False,'construction_or_termination_invoked':False}
    t=np.array([float(x) for x in design['thresholds']],np.float64)
    for name in ('R1','R2'):
        entries=[]
        for j in range(13):
            p=ROOT/name/f'level_{j:02d}';done=(p/'completed.json').exists()
            progress=p/'last_progress.json'
            if not progress.exists():progress=p/'progress.json'
            sweeps=json.loads(progress.read_text()).get('completed_sweeps',0) if progress.exists() else 0
            retained=max(0,min(1024,sweeps-512))
            strict=None
            if retained and (p/'loglikelihood.npy').exists():
                ll=np.load(p/'loglikelihood.npy',mmap_mode='r')[512:512+retained]
                strict=bool(np.isfinite(ll).all() and np.all(ll>t[j]))
            if (p/'retained.npz').exists():
                with np.load(p/'retained.npz') as f:strict=bool(np.isfinite(f['loglikelihood']).all() and np.all(f['loglikelihood']>t[j]))
            entries.append({'level':j,'started':p.exists(),'completed':done,'completed_sweeps':sweeps,
                'saved_completed_retained_events':retained*8,'strict_contour_for_saved_completed_events':strict,
                'completion':json.loads((p/'completed.json').read_text()) if done else None})
        result['sets'][name]={'complete_levels':sum(x['completed'] for x in entries),
                             'complete_all_13':all(x['completed'] for x in entries),'levels':entries}
    result['mass_perturbations_completed']=128 if (ROOT/'mass_perturbations.npz').exists() else 0
    result['authoritative_mis_estimates_computed']=(ROOT/'analysis.json').exists()
    result['restart']=json.loads((ROOT/'restart_comparison.json').read_text()) if (ROOT/'restart_comparison.json').exists() else {'pass':None,'status':'not reached'}
    if (ROOT/'analysis.json').exists():
        analysis=json.loads((ROOT/'analysis.json').read_text());result['estimates']=analysis['results']
        result['residual_diagnostics_path']=analysis.get('residual_diagnostics_path')
    # Read-only post-run integrity audit; a failure cannot change historical classification to success.
    mismatches=[]
    for p,h in design['protected_sha256'].items():
        if sha(Path(p))!=h:mismatches.append(p)
    assert not mismatches,mismatches
    assert sha(Path(design['driver_path']))==design['driver_sha256']
    assert sha(DESIGN)==DESIGN.with_suffix('.sha256').read_text().split()[0]
    result['protected_hashes_unchanged']=True
    result['numerical_environment']=json.loads((ROOT/'environment.json').read_text()) if (ROOT/'environment.json').exists() else None
    result['artifact_manifest']={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():result['artifact_manifest'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    runlog=Path('/tmp/lisa-gb0004-stage10a-run.log')
    if runlog.exists():result['artifact_manifest'][str(runlog)]={'sha256':sha(runlog),'bytes':runlog.stat().st_size}
    result['raw_artifact_storage_notice']='External /tmp artifacts must be retained or copied intact for long-term audit. No large arrays committed.'
    REPORT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Stage 10A: gb_0004 fixed-ladder reconstruction','',
        f"**CASE {case} — {result['classification']}.**",'',interpretation,'',
        'Exactly one experiment was launched under the [frozen design](lisa_gb0004_fixed_ladder_inference_design.json) and [checksum](lisa_gb0004_fixed_ladder_inference_design.sha256). The [JSON report](lisa_gb0004_fixed_ladder_inference.json) records all partial/completed levels, process provenance, numerical outcomes and the paths, sizes and SHA256 hashes of every external artifact. No stochastic retry occurred.','',
        '## Completion and numerical evidence','',
        '| Set | Completed frozen levels | All thirteen complete? | Saved completed retained events |','|---|---:|---|---:|']
    for n,v in result['sets'].items():
        lines.append(f"| {n} | {v['complete_levels']}/13 | {'yes' if v['complete_all_13'] else 'no'} | {sum(x['saved_completed_retained_events'] for x in v['levels'])} |")
    if failure:
        lines+=['',f"Failure: `{failure['exception_type']}: {failure['message']}`.",
            f"Recorded status: **{failure['numerical_status']}**. The terminating signal and process identity are retained; SIGBUS provided no Python exception traceback. The authorized run stopped; no alternate seed, retry, budget extension or kernel correction was performed."]
    lines+=['','The live population wrapper records every completed post-slice/post-exchange occupation state, caches, proposal component/labels/cost, actual slice input keys, exchange Gumbels/MH uniforms and reciprocal decisions. Full retained blocks are separately serialized after successful level completion. Partially allocated NPY arrays are interpreted only up to each recorded completed-sweep count; unwritten storage is never counted as samples.','',
        'The model, configuration, existing parameter/exchange kernels, estimator, CPU/x64 contract and frozen ladder remained unchanged. All design-protected hashes and the frozen driver/design hashes passed the post-run read-only audit. Historical construction and NSS posterior samples supplied no starts or estimator events.','',
        '## Reconstruction and evidence','']
    if result['estimates'] is None:
        lines+=['The required thirteen-stratum reconstruction was incomplete. No authoritative R1, R2 or pooled MIS estimate was computed from the partial experiment. Consequently log Z, normalized posterior weights, posterior ESS, deepest-level fractions, L8–L12 proposal-mixture sensitivity and the 128 posterior mass-sensitivity reductions are **unavailable**, not zero. The preregistered perturbation definition remains frozen; no replacement analysis of incomplete strata is used to rescue the run.']
    else:
        lines+=['| Estimate | log Z, implemented convention | Importance-weight ESS | L12 origin fraction | Target fraction above ell12 | Maximum weight |','|---|---:|---:|---:|---:|---:|']
        for n,v in result['estimates'].items():lines.append(f"| {n} | {v['logZ']:.12g} | {v['posterior_weight_ess']:.8g} | {v['origin_L12_fraction']:.8g} | {v['target_terminal_fraction']:.8g} | {v['maximum_weight']:.8g} |")
        lines+=['','Per-level raw evidence and posterior-weight fractions are identical for the same MIS importance ratios. The JSON distinguishes their originating-stratum allocation from the integral above the deepest threshold, and records all origin fractions, cumulative allocations/log contributions, highest-weight 1%/5%/10%, eligible-level multiplicities, matched source intervals, population-block information, walker/half-window comparisons and all fixed perturbation responses.','',
            '| Set | J | log Z_J | Importance-weight ESS |','|---|---:|---:|---:|']
        for n,v in result['estimates'].items():
            for j,row in v['mixture_sensitivity'].items():lines.append(f"| {n} | {j} | {row['logZ']:.12g} | {row['posterior_weight_ess']:.8g} |")
        lines+=['','| Set | Mass-sensitivity log Z q05 / median / q95 |','|---|---|']
        for n,v in result['estimates'].items():lines.append(f"| {n} | {v['mass_sensitivity']['logZ_q05_median_q95']} |")
    lines+=['','L0 gives full prior support: the unchanged MIS formula is not mathematically truncated at L12. Finite proposal efficiency above the deepest contour remains an empirical concern. No missing deep integral is assumed zero; no raw L12-origin fraction supplies a termination or convergence bound. The BayesLISAx amplitude/phase integration convention drops normalization constants. Any log Z is only under the implemented likelihood convention; no externally normalized physical Bayes factor is established.','',
        'Monte Carlo R1/R2 variability, recorded-mass sensitivity, deepest-stratum/proposal-mixture sensitivity and likelihood normalization are separate issues. They are not combined into a calibrated uncertainty. Neither available-prefix stability nor restart equality establishes statistical convergence.','',
        '## Posterior, restart and disposition','',
        'K remains fixed at nine. The planned six physical coordinates are f0, fdot, iota, psi, longitude and latitude, with the historical normalized six-coordinate Hungarian reference assignment and existing angular periods. Saved historical catalogues are matching references only. Marginal reconstruction amplitudes/phases are fitted quantities, never directly sampled posterior intervals.']
    if result['estimates'] is None:lines+=['','No reproducible nine-source scientific posterior conclusion or perturbation robustness can be inferred from this incomplete experiment. The required fresh-process L6 restart was not reached.']
    else:lines+=['',f"Fresh-process restart passed: {result['restart'].get('pass')}. Exact numerical equality is a reproducibility result only. Detailed source and sensitivity outputs are preserved in JSON and external matched-source/MIS archives; the classification applies the frozen qualitative scientific criteria without new numerical tolerances."]
    lines+=['','The earlier scoped Stage-6B numerical/population/promotion results and Stage-7B toy estimator validation remain intact. They do not validate this experiment\'s finite-time conditional stationarity or an end-to-end LISA inference result. No L13 was constructed; no automatic ladder construction, level diffusion, automatic termination, scientific budget tuning or likelihood-normalization change occurred.','',
        '**STOP.** No rerun after classification.']
    DOC.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'case':case,'completion':{n:v['complete_levels'] for n,v in result['sets'].items()},'report':str(REPORT)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['A','B','C','D']);p.add_argument('--interpretation',required=True)
    args=p.parse_args();generate(args.case,args.interpretation)
