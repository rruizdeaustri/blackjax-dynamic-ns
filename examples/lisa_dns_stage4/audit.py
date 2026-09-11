"""Read-only LISA audit. Evidence reconstruction is unvalidated and out of scope."""
import os
from pathlib import Path
import sys
ROOT = Path('/r5/home/rruiz/projects/sbi')
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(ROOT/'BayesLISAx/src'),
               str(ROOT/'lisa/gbgpu-main'), str(ROOT/'lisa/jaxlisa-main'), str(ROOT/'lisa/likelihood-master')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import json
import hashlib
import jax
import jax.numpy as jnp
import numpy as np

WORKFLOW = ROOT/'BayesLISAx/automation/snakemake_lisa_gb'
HIST = WORKFLOW/'results/baseline_nlive1000/gb_0004/k9'
NOTICE = 'Evidence reconstruction has not yet been validated and is out of scope.'


def scalar_functions(problem):
    def prior(theta):
        return jnp.asarray(problem.logprior(jnp.asarray(theta).reshape((54,)))).reshape(())
    def likelihood(theta):
        return jnp.asarray(problem.loglikelihood(jnp.asarray(theta).reshape((54,)))).reshape(())
    return prior, likelihood


def load_problem():
    stored = np.load(HIST/'seed44/posterior.npz')
    config = Path(str(stored['config_path']))
    os.environ['JAX_SAMPLERS_CONFIG'] = str(config)
    cfg = json.loads(config.read_text())
    assert cfg['model']['Kmax'] == 9 and cfg['model']['marg_Aphi']
    assert not cfg['model']['use_gates'] and not cfg['model']['order_f0']
    from jax_samplers.problems.lisa_gb_transdim_problem import make
    problem = make()
    assert problem.dim == 54
    return problem, cfg, config


def catalogue(problem, point):
    diag = np.asarray(problem._marg_diag(point))
    recon = problem._marg_recon(point)
    decoded = problem.decode_batch(np.asarray(point)[None], sort_by=None)
    return dict(loglikelihood=float(problem.loglikelihood(point)), residual_chi2=float(diag[2]),
                logdet=float(diag[3]), residual_power=np.asarray(recon[4]).tolist(),
                source_gain=np.asarray(recon[6]).tolist(), amplitude=np.abs(recon[5]).tolist(),
                phase=np.angle(recon[5]).tolist(),
                decoded={k:np.asarray(v).tolist() for k,v in decoded.items()})


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output', required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    problem,cfg,path=load_problem()
    from jax_samplers.samplers.blackjax_ns import BlackJAXNestedSampler, NSConfig
    nss_prior,nss_like=BlackJAXNestedSampler(problem,NSConfig())._make_scalar_fns()
    prior,like=scalar_functions(problem)
    report=dict(notice=NOTICE,config=cfg,config_path=str(path),devices=str(jax.devices()),families=[])
    points=[]
    for seed in (44,66,88):
        file=HIST/f'seed{seed}/posterior.npz';samples=np.load(file)['samples_u']
        assert samples.shape[1]==54
        # Sequential map bounds waveform memory for the historical collection.
        values=np.asarray(jax.jit(lambda xs:jax.lax.map(like,xs))(jnp.asarray(samples)))
        point=jnp.asarray(samples[np.argmax(values)]);points.append(np.asarray(point))
        np.testing.assert_allclose(prior(point),nss_prior(point),rtol=0,atol=1e-10)
        np.testing.assert_allclose(like(point),nss_like(point),rtol=0,atol=1e-8)
        expected=-jnp.sum(jax.nn.softplus(point)+jax.nn.softplus(-point))
        np.testing.assert_allclose(prior(point),expected,rtol=0,atol=1e-9)
        row=dict(seed=seed,latent_prior=float(prior(point)),stored_samples=len(samples),
                 file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**catalogue(problem,point))
        report['families'].append(row)
        print('family',seed,'best stored logL',row['loglikelihood'],'residual',row['residual_chi2'],flush=True)
    np.savez(out/'initial_families.npz',positions=points,seeds=[44,66,88])
    (out/'audit.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
