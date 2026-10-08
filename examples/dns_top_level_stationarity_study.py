"""Stage 9E diagnostic only: frozen candidate-5 target, no ladder construction.

Phases are separate so saved-trace analysis precedes preregistration and sampling.
Exact reference draws initialize the stationary control only; none are injected
during transitions. All dynamics call the existing DNS kernel unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('/tmp/dns-stage9d-heldout/original/attempt_000004')
ROOT = Path('/tmp/dns-stage9e-stationarity')
DESIGN = Path('docs/examples/dns_top_level_stationarity_design.json')
REPORT = Path('docs/examples/dns_top_level_stationarity_study.json')
DOC = Path('docs/examples/dns_top_level_stationarity_study.md')
T = 32768
W = 8
K = 2048
BURN = 1024
CUTS = [1024, 8192, 16384, 24576]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def inputs():
    report = json.loads(Path('docs/examples/dns_diffusive_ladder_heldout.json').read_text())
    pre = json.loads((BASE/'preflight.json').read_text())
    return report, pre


def intervals(ell, model):
    if np.isneginf(ell):
        return [(-8., 8.)]
    peak = -np.log(sum(model['widths'])*np.sqrt(2*np.pi))
    r = np.array(model['widths'])*np.sqrt(2*(peak-ell))
    out = []
    for lo, hi in sorted(zip(np.maximum(-8, np.array(model['centers'])-r),
                            np.minimum(8, np.array(model['centers'])+r))):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((float(lo), float(hi)))
    return out


def truth(pieces):
    length = sum(hi-lo for lo, hi in pieces)
    mean = sum((hi**2-lo**2)/2 for lo, hi in pieces)/length
    second = sum((hi**3-lo**3)/3 for lo, hi in pieces)/length
    edges = [-8., -2.25, 1.05, 8.]
    return dict(intervals=pieces, mass=length/16, mean=mean, second_moment=second,
                mode_probabilities=[sum(max(0, min(hi,b)-max(lo,a)) for lo,hi in pieces)/length
                                    for a,b in zip(edges[:-1],edges[1:])])


def stats(x, pieces):
    x = np.asarray(x).ravel()
    if not len(x):
        return dict(n=0)
    ordered = np.sort(x)
    length = sum(hi-lo for lo,hi in pieces)
    cdf = sum(np.clip(ordered-lo,0,hi-lo) for lo,hi in pieces)/length
    n = len(x)
    ks = max(np.max(np.arange(1,n+1)/n-cdf), np.max(cdf-np.arange(n)/n))
    return dict(n=n, KS=float(ks), mean=float(x.mean()), second_moment=float(np.mean(x*x)),
                mode_probabilities=(np.bincount(np.searchsorted([-2.25,1.05],x),minlength=3)/n).tolist())


def ess(y):
    y = np.asarray(y,float)
    n,w = y.shape
    var = y.var(ddof=1)
    b = n//256
    block = y[:b*256].reshape(b,256,w).mean(1)
    se2 = max(var/y.size, block.var(ddof=1)/block.size, y.mean(0).var(ddof=1)/w)
    z = y-y.mean(0)
    ft = np.fft.rfft(z,n=2*n,axis=0)
    ac = np.fft.irfft(ft*ft.conj(),n=2*n,axis=0)[:n].mean(1)/n
    pairs = []
    if ac[0] > 0:
        rho = ac/ac[0]
        for lag in range(1,n-1,2):
            pair = float(rho[lag]+rho[lag+1])
            if pair <= 0:
                break
            pairs.append(min(pair,pairs[-1]) if pairs else pair)
    tau = max(1.,1+2*sum(pairs))
    return dict(block_ESS=float(min(y.size,var/se2)) if se2 else None,
                spectral_ESS=float(y.size/tau),tau=tau)


def round_trips(j):
    total = 0
    for w in range(W):
        top = False
        lower = False
        for level in j[:,w]:
            if level == 4:
                if top and lower:
                    total += 1
                top = True
                lower = False
            elif top:
                lower = True
    return total


def analyze_trace(x,j,model,ell,candidate):
    pieces = intervals(ell,model)
    result = dict(physical_windows=[],cumulative=[],visit_windows=[],cutpoints={})
    for lo,hi in zip([0,*CUTS], [*CUTS,T]):
        mask = j[lo:hi] == 4
        row = stats(x[lo:hi][mask],pieces)
        row.update(start=lo,end=hi,top_occupancy=float(mask.mean()),
                   level_occupancy=(np.bincount(j[lo:hi].ravel(),minlength=5)/((hi-lo)*W)).tolist(),
                   round_trips=round_trips(j[lo:hi]))
        result['physical_windows'].append(row)
    for hi in CUTS[1:]+[T]:
        row = stats(x[BURN:hi][j[BURN:hi]==4],pieces)
        row.update(start=BURN,end=hi,top_occupancy=float((j[BURN:hi]==4).mean()),
                   round_trips=round_trips(j[BURN:hi]))
        result['cumulative'].append(row)
    for cut in CUTS:
        visits = [x[cut:,w][j[cut:,w]==4] for w in range(W)]
        row = dict(counts=[len(v) for v in visits],windows=[])
        for offset in [0,K,2*K]:
            if min(map(len,visits)) < offset+K:
                row['windows'].append(dict(offset=offset,available=False))
                continue
            values = np.stack([v[offset:offset+K] for v in visits],axis=1)
            s = stats(values,pieces)
            peak = -np.log(sum(model['widths'])*np.sqrt(2*np.pi))
            ll = peak-.5*np.min(((values[...,None]-np.array(model['centers']))/model['widths'])**2,axis=-1)
            obs = dict(theta=values,theta2=values**2,compression=(ll>candidate).astype(float))
            obs.update({f'mode{k}':(np.searchsorted([-2.25,1.05],values)==k).astype(float) for k in range(3)})
            s.update(offset=offset,available=True,ESS={name:ess(y) for name,y in obs.items()},
                     per_walker_mode_probabilities=[stats(values[:,w],pieces)['mode_probabilities'] for w in range(W)])
            # Also preserve the frozen compression ESS, which uses Bernoulli variance.
            from blackjax.ns.dns_levels import tail_diagnostics
            s['frozen_compression_diagnostics'] = tail_diagnostics(ll,candidate,block_size=64)
            row['windows'].append(s)
        all_stats = stats(np.concatenate(visits),pieces)
        equal = np.mean([stats(v,pieces)['mode_probabilities'] for v in visits],axis=0)
        row.update(all_occupation=all_stats,equal_walker_mode_probabilities=equal.tolist())
        result['cutpoints'][str(cut)] = row
    result['visit_windows'] = result['cutpoints'][str(BURN)]['windows']
    return result


def existing():
    ROOT.mkdir(exist_ok=True)
    report,pre = inputs()
    with np.load(BASE/'calibration_trace.npz') as f:
        x=f['position'][...,0];j=f['assigned_level']
    row=report['result']['attempts'][-1]['banks']['calibration']
    result=analyze_trace(x,j,report['design']['model'],float(pre['thresholds'][-1]),row['candidate_threshold'])
    write(ROOT/'existing_analysis.json',result)
    print('EXISTING TRACE ANALYZED',[(r['start'],r['end'],r['KS']) for r in result['physical_windows']],flush=True)


def prepare():
    assert (ROOT/'existing_analysis.json').exists(), 'Analyze saved trace first'
    assert not DESIGN.exists(), 'No preregistration overwrite'
    report,pre=inputs()
    protected={p:sha(p) for p in report['design']['protected_sha256']}
    for p in ['examples/dns_diffusive_ladder_heldout.py','docs/examples/dns_diffusive_ladder_heldout.json',
              'docs/examples/dns_diffusive_ladder_heldout.md','docs/examples/dns_diffusive_ladder_heldout_design.json',
              'docs/examples/dns_diffusive_ladder_heldout_design.sha256','docs/examples/dns_diffusive_heldout_failure_audit.md']:
        protected[p]=sha(p)
    with np.load('/tmp/dns-stage9d-heldout/original/attempt_000003/checkpoint/banks.npz') as f:
        source=f['calibration_position']
    promotion=json.loads((BASE/'promotion.json').read_text())['calibration']
    starts=source[np.array(promotion['retained_indices']),np.array(promotion['donor_ids']),0]
    model=report['design']['model']
    thresholds=[float(v) for v in pre['thresholds']]
    truths=[truth(intervals(v,model)) for v in thresholds]
    relative=np.array([r['mass'] for r in truths])*np.exp(-np.array(pre['log_masses']))
    spec=dict(stage='9E DEVELOPMENT ONLY',replicate_count=8,seeds=[910701,910702,910703,910704,910705,910706,910707,910708],
              arms=['saved_promoted_start','exact_joint_stationary'],paired_transition_keys=True,
              physical_sweeps=T,walkers=W,quota=K,burn_in=BURN,cutpoints=CUTS,
              model=model,thresholds=pre['thresholds'],log_masses=pre['log_masses'],log_weights=[0.]*5,
              candidate_threshold=report['result']['attempts'][-1]['banks']['calibration']['candidate_threshold'],
              parameter=report['design']['parameter'],promoted_theta=starts.tolist(),
              exact_conditional_truths=truths,exact_joint_level_probabilities=(relative/relative.sum()).tolist(),
              reference_rule='Draw j with mass a_j X_j, then theta uniformly on merged contour intervals weighted by length. Initialization only; no reference injections during dynamics.',
              window_rule='Physical disjoint [0,1024),[1024,8192),[8192,16384),[16384,24576),[24576,32768); cumulative postburn; per-walker disjoint first/second/third K visits at each cut if all quotas available; no padding.',
              ESS_rule='Frozen compression diagnostic block64; diagnostic general block256/max between-walker SE and spectral initial-positive monotone pair tau, per observable. No retrospective acceptance gate.',
              fluctuation_rule='Empirical independent replicate first-K KS>=historical exact KS; exact binomial 95% intervals; pair later windows, not independent extra replicates. No IID-N KS significance.',
              classification_rule='A only if promoted early signed modal bias consistent across replicates and attenuates late relative stationary controls; B only if stationary controls show systematic first-K/equal-walker distortion vs physical occupation; C only if controls centered, no persistent bias, historical-size failures observed compatibly; D if reliable settling absent; E deterministic issue; otherwise F. Eight replicates limit discrimination; no significance from descriptive ESS alone.',
              protected_sha256=protected,driver_sha256=sha(__file__),existing_analysis_sha256=sha(ROOT/'existing_analysis.json'),
              saved_trace_sha256=sha(BASE/'calibration_trace.npz'),no_tuning=True,no_retry=True,no_validation=True)
    write(DESIGN,spec)
    DESIGN.with_suffix('.sha256').write_text(sha(DESIGN)+'  '+DESIGN.name+'\n')
    print('PREREGISTERED',sha(DESIGN),flush=True)


def check(spec):
    assert sha(DESIGN)==DESIGN.with_suffix('.sha256').read_text().split()[0]
    assert sha(__file__)==spec['driver_sha256']
    for p,h in spec['protected_sha256'].items():
        assert sha(p)==h,p


def run():
    import jax
    import jax.numpy as jnp
    from blackjax.ns import dns
    from examples.dns_diffusive_ladder_heldout import ThreeModes
    spec=json.loads(DESIGN.read_text());check(spec)
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    toy=ThreeModes(spec['model'])
    levels=dns.create_levels([float(v) for v in spec['thresholds']],spec['log_masses'],spec['log_weights'])
    advance=jax.vmap(dns.build_kernel(levels,toy.parameter_step(spec['parameter'])))
    def body(carry,_):
        state,keys=carry
        split=jax.vmap(lambda k:jax.random.split(k))(keys)
        state,info=advance(split[:,1],state)
        return (state,split[:,0]),(state.particle.position,state.level_index,
                                 jnp.all(info.parameter_is_valid))
    scan=jax.jit(lambda s,k:jax.lax.scan(body,(s,k),None,length=T))
    for seed in spec['seeds']:
        children=np.random.SeedSequence(seed).spawn(2)
        key_seeds=children[0].generate_state(W)
        for arm in spec['arms']:
            target=ROOT/f'{seed}_{arm}.npz'
            assert not target.exists(),'No rerun'
            if arm=='saved_promoted_start':
                x=np.array(spec['promoted_theta']);j=np.full(W,4,dtype=np.int32)
            else:
                rng=np.random.default_rng(children[1])
                j=rng.choice(5,size=W,p=spec['exact_joint_level_probabilities']).astype(np.int32)
                x=np.empty(W)
                for w,level in enumerate(j):
                    pieces=spec['exact_conditional_truths'][level]['intervals']
                    lengths=np.array([hi-lo for lo,hi in pieces])
                    lo,hi=pieces[rng.choice(len(pieces),p=lengths/lengths.sum())]
                    x[w]=rng.uniform(lo,hi)
            lp,ll=toy.evaluate(x[:,None])
            state=dns.DNSState(dns.DNSParticleState(jnp.array(x[:,None]),jnp.array(lp),jnp.array(ll)),jnp.array(j))
            keys=jnp.stack([jax.random.PRNGKey(int(k)) for k in key_seeds])
            _,arrays=scan(state,keys)
            position,assigned,valid=map(np.asarray,arrays)
            assert valid.all()
            np.savez_compressed(target,position=position,assigned_level=assigned,initial_theta=x,initial_j=j,key_seeds=key_seeds)
            check(spec)
            print('COMPLETED',seed,arm,flush=True)


def summarize():
    from scipy.stats import beta
    spec=json.loads(DESIGN.read_text());check(spec)
    out=dict(stage='9E DEVELOPMENT ONLY',design_sha256=sha(DESIGN),design=spec,
             historical_classification='FAILED-CONDITIONAL',existing=json.loads((ROOT/'existing_analysis.json').read_text()),replicates=[])
    for seed in spec['seeds']:
        arms={}
        for arm in spec['arms']:
            path=ROOT/f'{seed}_{arm}.npz'
            with np.load(path) as f:
                arms[arm]=analyze_trace(f['position'][...,0],f['assigned_level'],spec['model'],float(spec['thresholds'][-1]),spec['candidate_threshold'])
            arms[arm]['trace_sha256']=sha(path)
        out['replicates'].append(dict(seed=seed,arms=arms))
    historical=out['existing']['visit_windows'][0]['KS']
    summary={}
    for arm in spec['arms']:
        first=[r['arms'][arm]['visit_windows'][0] for r in out['replicates']]
        values=np.array([r['KS'] for r in first]);hits=int((values>=historical).sum());n=len(values)
        ci=[float(beta.ppf(.025,hits,n-hits+1)) if hits else 0.,float(beta.ppf(.975,hits+1,n-hits)) if hits<n else 1.]
        modes=np.array([r['mode_probabilities'] for r in first]);truth_modes=np.array(spec['exact_conditional_truths'][4]['mode_probabilities'])
        summary[arm]=dict(first_K_KS=values.tolist(),historical_size_hits=hits,replicates=n,empirical_frequency=hits/n,binomial_95_interval=ci,
                          mean_mode_probabilities=modes.mean(0).tolist(),mode_probability_SE=(modes.std(0,ddof=1)/np.sqrt(n)).tolist(),
                          mean_mode_error=(modes.mean(0)-truth_modes).tolist())
    out['ensemble_summary']=summary
    out['classification']='PENDING REVIEW'
    out['protected_hashes_unchanged']=True
    write(REPORT,out)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['existing','prepare','run','summarize'])
    globals()[parser.parse_args().phase]()
