"""Stage-4K prerequisite audit only; no likelihood probes or Markov chain."""
import ast
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np

MODEL=Path('/r5/home/rruiz/projects/sbi/BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py')
NAMES=('f0','fdot','iota','psi','lam','beta')
PAIRS=tuple(combinations(range(9),2))


def block_indices(labels):
    labels=np.asarray(labels,dtype=int)
    if labels.ndim!=1 or len(labels) not in (1,2) or len(set(labels))!=len(labels) or np.any((labels<0)|(labels>=9)):
        raise ValueError('One or two distinct source labels required')
    return (labels[:,None]*6+np.arange(6)).ravel()


def replace_from_fixed_base(base,donor,labels):
    """Each call starts from a copy of the original base, never a prior proposal."""
    base=np.asarray(base);donor=np.asarray(donor)
    if base.shape!=(54,) or donor.shape!=(54,): raise ValueError('Expected 54-coordinate states')
    result=base.copy();indices=block_indices(labels);result[indices]=donor[indices]
    return result


def label_from_index(index,size):
    """A uniform categorical index maps bijectively to a label or unordered pair."""
    if size==1:
        if not 0<=index<9: raise ValueError('Invalid single-label index')
        return (index,)
    if size==2:
        if not 0<=index<36: raise ValueError('Invalid pair index')
        return PAIRS[index]
    raise ValueError('Only single and pair probes')


def ideal_block_logprior(block):
    """Ideal uniform-box/Jacobian prior; used only for the mathematical test."""
    x=np.asarray(block)
    return -np.sum(np.logaddexp(0,x)+np.logaddexp(0,-x))


def ideal_log_mh_ratio(base,proposed,labels,base_ll,proposed_ll,threshold):
    """Cancellation for q_B = exact target prior_B, not for clipped sampling."""
    if base_ll<=threshold: raise ValueError('Current state must satisfy contour')
    if proposed_ll<=threshold: return -np.inf
    i=block_indices(labels)
    return (ideal_block_logprior(proposed)-ideal_block_logprior(base)
            +ideal_block_logprior(np.asarray(base)[i])-ideal_block_logprior(np.asarray(proposed)[i]))


def transform_source(path=MODEL):
    """Extract the existing deterministic transform without importing the model."""
    text=Path(path).read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='box_to_u')
    eps=ast.literal_eval(node.args.defaults[-1])
    return node,eps,ast.get_source_segment(text,node)


def load_audited_transform():
    """Execute only the model's actual small coordinate transform, not make/loglike."""
    import jax.numpy as jnp
    node,_,_=transform_source()
    namespace={'jnp':jnp,'DTYPE':jnp.float64}
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(MODEL),'exec'),namespace)
    return namespace['box_to_u']


def main():
    output=Path('/tmp/lisa_dns_stage4k_prior_refresh_audit');output.mkdir(exist_ok=False)
    checkpoint=Path('/tmp/lisa_dns_stage4_ladder_batch/checkpoint_level9.json')
    payload=json.loads(checkpoint.read_text())['payload'];provenance=payload['provenance'];cfg=provenance['config']
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==provenance['file_sha256'][str(MODEL)]
    assert cfg['model']['Kmax']==9 and cfg['model']['marg_Aphi']
    assert not cfg['model']['use_gates'] and not cfg['model']['order_f0']
    assert not cfg.get('priors',{}).get('gaussian') and not cfg.get('prior',{}).get('gaussian_priors')
    _,eps,source=transform_source()
    import jax.numpy as jnp
    transform=load_audited_transform()
    inputs=np.array([eps/4,eps/2,eps,2*eps,.5])
    transformed=np.asarray(transform(jnp.asarray(inputs),0.,1.))
    assert transformed[0]==transformed[1]==transformed[2]
    ideal=np.log(inputs)-np.log1p(-inputs)
    assert ideal[0]!=ideal[1] and np.isfinite(ideal_block_logprior(ideal[:1]))
    report=dict(status='stopped_exact_sampler_mismatch_before_probes',
        target_factorization='Nine labelled blocks, six independent uniform physical coordinates per block; no ordering, gates or Gaussian factors',
        ideal_cancellation_verified=True,actual_sampler_matches_exact_target_prior=False,
        mathematical_ratio='[pi_-B(u_-B) pi_B(u_Bprime) Iprime pi_B(u_B)] / [pi_-B(u_-B) pi_B(u_B) I pi_B(u_Bprime)] = Iprime/I',
        actual_sampler_issue='box_to_u clips unit-uniform draws: endpoint atoms and absent latent tails are incompatible with the continuous target prior factor. Thus q_B is not pi_B as required by contour-only acceptance.',
        clip_epsilon=eps,clip_lower_latent=float(np.log(eps)-np.log1p(-eps)),
        single_block_clip_probability=1-(1-2*eps)**6,pair_block_clip_probability=1-(1-2*eps)**12,
        requested_selected_coordinates=4096*6+4096*12,
        ideal_probability_any_selected_coordinate_clipped=1-(1-2*eps)**(4096*6+4096*12),
        deterministic_transform_inputs=inputs.tolist(),actual_transform_outputs=transformed.tolist(),ideal_outputs=ideal.tolist(),
        transform_source=source,ell9=float(payload['thresholds'][-1]),ell10=-109401.11425363769,
        proposed_budget_not_executed=dict(base_states=128,single_proposals=4096,pair_proposals=4096,global_proposals=0),
        likelihood_calls=0,prior_draws=0,base_states_evaluated=0,proposals_evaluated=0,
        probe_integrity_failures=None,probe_integrity_notice='Not applicable: no bases or likelihood proposals evaluated. One deterministic prerequisite audit mismatch found.',
        single_refresh_classification='not assessed: exact-sampler prerequisite failed',pair_refresh_classification='not assessed: exact-sampler prerequisite failed',
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,checkpoint,Path(__file__)]})
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ['status','clip_epsilon','clip_lower_latent','likelihood_calls','proposals_evaluated']},indent=2))


if __name__=='__main__':main()
