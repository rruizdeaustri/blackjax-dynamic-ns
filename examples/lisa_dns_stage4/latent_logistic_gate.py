"""Stage-4L direct-latent-prior gate. No likelihood probes or MCMC path."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from examples.lisa_dns_stage4.prior_refresh_audit import MODEL,block_indices,replace_from_fixed_base


def analytic_logprior(u):
    u=jnp.asarray(u,dtype=jnp.float64)
    return -jnp.sum(jax.nn.softplus(u)+jax.nn.softplus(-u))


def direct_blocks(key,number,blocks):
    """Candidate direct generator only; does not depend on any current state."""
    if number<1 or blocks not in (1,2): raise ValueError('Positive count, one or two blocks')
    return jax.random.logistic(key,(number,blocks,6),dtype=jnp.float64)


def cancellation_terms(old,new,labels,prior=analytic_logprior):
    indices=block_indices(labels)
    target_change=float(prior(new)-prior(old))
    proposal_change=float(analytic_logprior(np.asarray(old)[indices])-analytic_logprior(np.asarray(new)[indices]))
    return dict(target_logprior_change=target_change,reverse_minus_forward_logproposal=proposal_change,
                uncancelled_log_ratio=target_change+proposal_change)


def load_actual_jacobian():
    """Cheap test hook: actual deterministic model function, without model setup."""
    tree=ast.parse(MODEL.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='box_log_jacobian')
    env=dict(jax=jax,jnp=jnp,DTYPE=jnp.float64,EPS=jnp.asarray(jnp.finfo(jnp.float64).tiny))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(MODEL),'exec'),env)
    return env['box_log_jacobian']


def main():
    from examples.lisa_dns_stage4 import audit
    from jax._src import random as random_source
    out=Path('/tmp/lisa_dns_stage4l_latent_gate');out.mkdir(exist_ok=False)
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    provenance=json.loads(checkpoint.read_text())['payload']['provenance']
    problem,cfg,config_path=audit.load_problem()
    assert cfg==provenance['config']
    for path in [MODEL,config_path]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['file_sha256'][str(path)]
    prior=jax.jit(audit.scalar_functions(problem)[0])
    # Deliberately cover ordinary values and positive-tail cancellation/saturation.
    vectors=dict(zero=np.zeros(54),ramp8=np.linspace(-8,8,54),ramp12=np.linspace(-12,12,54),
        ramp20=np.linspace(-20,20,54),ramp30=np.linspace(-30,30,54),
        positive36=np.r_[36.,np.zeros(53)],positive40=np.r_[40.,np.zeros(53)],
        negative40=np.r_[-40.,np.zeros(53)])
    rows=[]
    for name,u in vectors.items():
        actual=float(prior(u));expected=float(analytic_logprior(u))
        rows.append(dict(name=name,position=u.tolist(),actual_logprior=actual,
                         analytic_logprior=expected,difference=actual-expected))
    block_rows=[]
    for labels in [[0],[4],[8],[1,7]]:
        for value in [4.,12.,20.,30.,36.,40.]:
            old=np.zeros(54);donor=np.zeros(54)
            for label in labels: donor[label*6]=value
            new=replace_from_fixed_base(old,donor,labels)
            block_rows.append(dict(labels=labels,changed_coordinate_value=value,
                                   **cancellation_terms(old,new,labels,prior)))
    # A finite analytic density with -inf implemented prior is an unambiguous failure,
    # independent of choosing a tolerance for smaller finite discrepancies.
    finite_mismatch=[r for r in rows if np.isfinite(r['analytic_logprior']) and not np.isfinite(r['actual_logprior'])]
    report=dict(status='failed_gate1_actual_prior_identity',gate1_passed=False,
        analytical_identity='log Logistic(u;0,1) = -softplus(u)-softplus(-u)',
        explanation='Implemented box Jacobian forms sigmoid first; positive-tail cancellation and rounding to 1 disagree with stable logistic log density. Factorization holds, but direct Logistic proposal density does not cancel the numerically implemented factor globally.',
        vectors=rows,block_changes=block_rows,nonfinite_identity_mismatches=len(finite_mismatch),
        jax_version=jax.__version__,dtype='float64',generator_source=inspect.getsource(random_source._logistic),
        generator_assessment='Candidate uses uniform(minval=finfo(dtype).tiny,maxval=1), log(x)-log1p(-x). No physical sampler or 1e-9 clipping. Full fixed-budget generator audit not run because prior-identity gate failed.',
        actual_prior_calls=len(rows)+2*len(block_rows),likelihood_calls=0,mcmc_steps=0,feasibility_proposals=0,
        gate2_status='not run',level10_remains_rejected=True,
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,config_path,checkpoint,Path(__file__)]})
    assert finite_mismatch, 'Audit logic needs review: expected positive-tail prior mismatch absent'
    def safe(x):
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,list):return [safe(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return str(x)
        return x
    (out/'report.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
    print('GATE 1 FAILED; no likelihood probes. Report:',out/'report.json')


if __name__=='__main__':main()
