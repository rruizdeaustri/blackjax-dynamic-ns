"""Bounded toy-only validation of the existing Stage-3 promotion implementation.

No adaptive ladder, LISA imports, model calls, or checkpoint writes.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

from blackjax.ns import dns, dns_levels
from examples.dns_level_construction_validation import problem, constrained_cdf, ratio_error

PROTOCOL = dict(seeds=[101, 202, 303], walkers=8, old_threshold=-8.0,
                threshold=-4.5, burn=256, draws=1024, thinning=2,
                history_draws=256, block=64,
                schemes=['same_walker', 'replacement', 'unique'],
                scenarios=['ordinary', 'missing_predecessor'],
                decision='Every mode/CDF check: abs(error) <= 6*conservative SE + 0.004; zero integrity failures')


def particles(x):
    _, prior, likelihood, _ = problem('double_well')
    x = jnp.asarray(x)
    return dns.DNSParticleState(x, prior(x), likelihood(x))


def eligible(history, threshold):
    return np.flatnonzero(np.asarray(history.loglikelihood).ravel() > threshold)


def promote(key, history, threshold, scheme='replacement', walkers=8):
    """Use generic production promotion unchanged; alternatives are toy controls."""
    pool = eligible(history, threshold)
    if not len(pool):
        raise ValueError('Entire bank has no eligible states')
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), history)
    if scheme == 'replacement':
        indices = jax.random.choice(key, jnp.asarray(pool), (walkers,), replace=True)
        result = dns_levels._survivors(key, history, threshold, walkers)
    elif scheme == 'unique':
        if len(pool) < walkers:
            raise ValueError('Insufficient eligible entries for unique control')
        indices = jax.random.choice(key, jnp.asarray(pool), (walkers,), replace=False)
        result = jax.tree.map(lambda x: x[indices], flat)
    elif scheme == 'same_walker':
        indices = []
        width = history.loglikelihood.shape[1]
        for w, subkey in enumerate(jax.random.split(key, walkers)):
            own = pool[pool % width == w]
            if not len(own):
                raise ValueError('Predecessor has no eligible states')
            indices.append(jax.random.choice(subkey, jnp.asarray(own)))
        indices = jnp.asarray(indices)
        result = jax.tree.map(lambda x: x[indices], flat)
    else:
        raise ValueError(scheme)
    for actual, expected in zip(result, jax.tree.map(lambda x: x[indices], flat)):
        np.testing.assert_array_equal(actual, expected)
    check_states(result, threshold)
    return result, np.asarray(indices)


def check_states(states, threshold):
    for a in states:
        assert np.all(np.isfinite(a))
    assert np.all(np.asarray(states.loglikelihood) > threshold)
    expected = particles(states.position)
    for a, b in zip(states, expected):
        np.testing.assert_array_equal(a, b)


def missing_predecessor(history, threshold, walker=0):
    """Cycle only this predecessor's saved ineligible states, preserving caches."""
    bad = np.flatnonzero(np.asarray(history.loglikelihood[:, walker]) <= threshold)
    if not len(bad):
        raise ValueError('Stress fixture unavailable; do not tune or retry')
    indices = np.resize(bad, history.position.shape[0])
    return jax.tree.map(lambda x: x.at[:, walker].set(x[indices, walker]), history)


def make_runner(steps):
    _, _, _, kernel = problem('double_well')
    def run(key, start, threshold):
        def step(carry, _):
            key, state = carry
            key, subkey = jax.random.split(key)
            state, _ = jax.vmap(kernel, in_axes=(0, 0, None))(
                jax.random.split(subkey, state.position.shape[0]), state, threshold)
            return (key, state), state
        return jax.lax.scan(step, (key, start), None, length=steps)[1]
    return jax.jit(run)


def distances(x):
    d = np.array([abs(float(x[i])-float(x[j])) for i,j in combinations(range(len(x)),2)])
    return dict(min=float(d.min()), median=float(np.median(d)), max=float(d.max()))


def ess(x):
    """Initial-positive paired autocorrelation estimate, descriptive per walker."""
    x = np.asarray(x, float)-np.mean(x)
    if not np.any(x):
        return 0.0
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f*f.conj())[:n]
    ac /= ac[0]
    pairs = ac[1:n-1:2] + ac[2:n:2]
    end = next((i for i,v in enumerate(pairs) if v <= 0), len(pairs))
    return float(min(n, n/max(1., 1+2*sum(pairs[:end]))))


def summary(history, start, postburn):
    x, ll = np.asarray(history.position), np.asarray(history.loglikelihood)
    mode = ratio_error(x > 0, np.ones_like(x), block=PROTOCOL['block'])
    # Analytic position CDF tests both mode weights and within-mode distribution.
    u = constrained_cdf('double_well', PROTOCOL['threshold'], x)
    cdf = [dict(q=q, **ratio_error(u <= q, np.ones_like(x), block=PROTOCOL['block']))
           for q in [.1,.25,.5,.75,.9]]
    # logL=-r^2/(2 sigma^2), r uniform [0,.9]: analytic likelihood CDF.
    llcdf = [dict(q=q, **ratio_error(ll <= -4.5*(1-q)**2, np.ones_like(x), block=PROTOCOL['block']))
             for q in [.1,.25,.5,.75,.9]]
    ks = [ks_2samp(ll[:,i],ll[:,j]).statistic for i,j in combinations(range(8),2)]
    allx = np.vstack([np.asarray(postburn.position), x])
    return dict(mode=mode, walker_mode_weights=(x>0).mean(0).tolist(), cdf=cdf, likelihood_cdf=llcdf,
                passed=all(abs(v['estimate']-v['q']) <= 6*v['se']+.004 for v in cdf+llcdf),
                logL_mean=float(ll.mean()), logL_quantiles=np.quantile(ll,[.1,.5,.9]).tolist(),
                pairwise_logL_KS=dict(median=float(np.median(ks)),max=float(max(ks))),
                logL_ESS=[ess(ll[:,w]) for w in range(8)], mode_ESS=[ess(x[:,w]>0) for w in range(8)],
                mode_switches=np.sum(np.diff(allx>0,axis=0)!=0,axis=0).tolist(),
                initial_distances=distances(start.position), postburn_distances=distances(postburn.position))


def protected_hashes():
    roots = [Path('/tmp/lisa_dns_stage5b_population_level13'),
             Path('/tmp/lisa_dns_stage4z_population_level12/checkpoint_level12.json'),
             Path('examples/lisa_dns_stage4')]
    files = []
    for root in roots:
        files.extend([root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise RuntimeError('Completed results exist; no automatic rerun')
    before = protected_hashes()
    (output/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    (output/'protected_before.json').write_text(json.dumps(before,indent=2))
    p = PROTOCOL
    history_run = make_runner(p['burn']+p['history_draws']*p['thinning'])
    promoted_run = make_runner(p['burn']+p['draws']*p['thinning'])
    rows, bank_records = [], []
    for seed in p['seeds']:
        banks, streams = [], []
        for bank_key in jax.random.split(jax.random.key(seed),2):
            initkey, historykey, promotionkey, samplingkey = jax.random.split(bank_key,4)
            # Exact independent draws from old uniform constrained support [-2.2,2.2].
            outer = 1+.3*np.sqrt(-2*p['old_threshold'])
            x = jax.random.uniform(initkey,(8,),minval=-outer,maxval=outer)
            full = history_run(historykey,particles(x),p['old_threshold'])
            check_states(full,p['old_threshold'])
            history = jax.tree.map(lambda a:a[p['burn']+p['thinning']-1::p['thinning']],full)
            banks.append(history)
            streams.append((promotionkey,samplingkey))
            bank_records.append(dict(seed=seed, history_hash=hashlib.sha256(np.asarray(history.position).tobytes()).hexdigest(),
                keys=[jax.random.key_data(k).tolist() for k in [initkey,historykey,promotionkey,samplingkey]]))
        assert not np.shares_memory(np.asarray(banks[0].position),np.asarray(banks[1].position))
        assert not np.array_equal(banks[0].position,banks[1].position)
        for b, (bank, (promotionkey,samplingkey)) in enumerate(zip(banks,streams)):
            for scenario in p['scenarios']:
                h = bank if scenario=='ordinary' else missing_predecessor(bank,p['threshold'])
                pool = eligible(h,p['threshold'])
                counts = np.bincount(pool%8,minlength=8)
                for scheme in p['schemes']:
                    row = dict(seed=seed, bank=['selection','calibration'][b], scenario=scenario, scheme=scheme,
                               eligible_count=len(pool), donor_counts=counts.tolist(), donors=int(sum(counts>0)))
                    try:
                        start, indices = promote(promotionkey,h,p['threshold'],scheme)
                    except ValueError:
                        if not (scenario=='missing_predecessor' and scheme=='same_walker'):
                            raise
                        row.update(status='expected_missing_survivor_failure')
                        rows.append(row)
                        continue
                    # No selected realization may come from the other bank's eligible pool.
                    assert not np.any(np.isin(np.asarray(start.position),np.asarray(banks[1-b].position)))
                    trace = promoted_run(samplingkey,start,p['threshold'])
                    check_states(trace,p['threshold'])
                    postburn = jax.tree.map(lambda a:a[p['burn']-1],trace)
                    retained = jax.tree.map(lambda a:a[p['burn']+p['thinning']-1::p['thinning']],trace)
                    unique = len(np.unique(np.asarray(start.position)))
                    row.update(status='complete',unique_starts=unique,duplicates=8-unique,
                               selected_indices=indices.tolist(),donor_multiplicities=np.bincount(indices%8,minlength=8).tolist(),
                               **summary(retained,start,postburn))
                    rows.append(row)
                    print(seed,row['bank'],scenario,scheme,row['mode'],flush=True)
    assert protected_hashes()==before
    keys = [tuple(k) for b in bank_records for k in b['keys']]
    assert len(set(keys))==len(keys)
    pooled = [r for r in rows if r['scheme']=='replacement']
    result = dict(protocol=p,rows=rows,banks=bank_records,
                  decision='PASS' if all(r['passed'] for r in pooled) else 'FAIL',
                  protected_hashes_unchanged=True,integrity_failures=0,
                  no_LISA_calls=True, no_LISA_level13_checkpoint=not any(Path('/tmp').glob('lisa_dns_*/checkpoint_level13*')))
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/tmp/dns_pooled_promotion_toy')
    run(parser.parse_args().output)
