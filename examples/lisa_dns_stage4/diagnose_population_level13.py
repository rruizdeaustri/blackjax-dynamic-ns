"""Stage 5C: saved-state diagnostics only. No sampler, model, or promotion imports."""
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from examples.lisa_dns_stage4.diagnose_source_permutations import (
    NAMES, DF, source_distances, cross_frequency_distance,
)

B = Path('/tmp/lisa_dns_stage5b_population_level13')
A = Path('/tmp/lisa_dns_stage5a_level12_validation')
OUT = Path('/tmp/lisa_dns_stage5c_readonly_diagnosis')
ELL12 = -108187.99712765554
LOGX12 = -10.418392711998465
CANDIDATE = -107123.66036615883


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def read(path):
    return json.loads(Path(path).read_text())


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    return value


def survivor_summary(values, threshold=CANDIDATE, block_size=32):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty one-dimensional history required')
    if len(values) % block_size:
        raise ValueError('Complete fixed contiguous blocks required')
    hit = values > threshold
    indices = np.flatnonzero(hit)
    runs = np.diff(np.r_[False, hit, False].astype(int))
    zero_runs = np.diff(np.r_[False, ~hit, False].astype(int))
    lengths = np.flatnonzero(zero_runs == -1) - np.flatnonzero(zero_runs == 1)
    return dict(count=int(hit.sum()), fraction=float(hit.mean()),
                first=int(indices[0]) if len(indices) else None,
                last=int(indices[-1]) if len(indices) else None,
                occupied_blocks=int(hit.reshape(-1, block_size).any(1).sum()),
                block_counts=hit.reshape(-1, block_size).sum(1),
                contiguous_survivor_runs=int((runs == 1).sum()),
                longest_zero_interval=int(lengths.max()) if len(lengths) else 0)


def halves(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] % 2 or not values.shape[1]:
        raise ValueError('Nonempty walker/time history with equal halves required')
    return np.split(values, 2, axis=1)


def empirical_candidate(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Finite nonempty values required')
    ordered = np.sort(values.ravel())
    index = int(np.floor(len(ordered) * (1 - np.exp(-1))))
    return dict(value=float(ordered[index]), order_index=index, n=len(ordered),
                shift=float(ordered[index] - CANDIDATE), diagnostic_only=True)


def sensitivity(values):
    values = np.asarray(values)
    first, second = halves(values)
    return dict(official_recomputed=empirical_candidate(values),
                leave_one_out=[dict(omitted=w, **empirical_candidate(np.delete(values, w, axis=0)))
                               for w in range(len(values))],
                first_half=empirical_candidate(first), second_half=empirical_candidate(second))


def eligible_pool(position, logL, threshold=CANDIDATE):
    position, logL = np.asarray(position), np.asarray(logL)
    if position.shape[:2] != logL.shape or position.ndim != 3:
        raise ValueError('Walker/time states and likelihoods required')
    mask = logL > threshold
    donors, times = np.nonzero(mask)
    entries = position[mask]
    if len(entries):
        _, first = np.unique(entries, axis=0, return_index=True)
        first = np.sort(first)
    else:
        first = np.array([], dtype=int)
    return entries[first], donors[first], times[first], mask.sum(1), len(entries)


def pool_feasibility(position, logL, threshold=CANDIDATE, walkers=8):
    points, donors, _, counts, entries = eligible_pool(position, logL, threshold)
    n = len(points); nd = len(np.unique(donors)); duplicate_min = max(0, walkers-n)
    common = dict(eligible_entries=entries, eligible_unique_states=n,
                  eligible_donor_walkers=nd, eligible_donor_ids=np.unique(donors),
                  duplicate_donor_required=nd < walkers)
    return dict(A=dict(possible=bool(np.all(counts > 0)), missing_walkers=np.flatnonzero(counts == 0),
                       counts=counts),
                B=dict(**common, possible=n >= walkers, duplicate_state_required=False),
                C=dict(**common, possible=n >= walkers, maximum_donor_diversity=min(nd, walkers),
                       duplicate_state_required=False),
                D=dict(**common, possible=n > 0, duplicate_state_required=0 < n < walkers,
                       minimum_repeated_slots=duplicate_min if n else None))


def physical_settings(checkpoint, exchanges):
    cfg = checkpoint['payload']['provenance']['config']
    # The model's implemented frequency interval is Fourier-grid aligned, unlike
    # nominal config bounds. Recover its affine map from saved selector records.
    block = exchanges['post_slice_position'].reshape(-1, 2, 9, 6)
    labels = exchanges['labels']
    unit = expit(block[np.arange(len(block))[:, None], np.arange(2)[None, :], labels, 0]).ravel()
    frequency = exchanges['selected_frequency'].ravel()
    i, j = unit.argmin(), unit.argmax()
    width = (frequency[j]-frequency[i])/(unit[j]-unit[i])
    lower = frequency[i]-width*unit[i]
    np.testing.assert_allclose(lower+width*unit, frequency, rtol=0, atol=1e-18)
    bounds = {'f0': [lower, lower+width], **cfg['priors']['uniform']}
    lo = np.array([bounds[n][0] for n in NAMES]); hi = np.array([bounds[n][1] for n in NAMES])
    widths = hi-lo; widths[3] = np.pi; widths[4] = 2*np.pi
    return lo, hi, widths


def physical(position, settings):
    lo, hi, _ = settings
    return lo + (hi-lo)*expit(np.asarray(position).reshape(np.asarray(position).shape[:-1]+(9, 6)))


def pair_geometry(a, b, settings):
    # Existing Stage-4I metrics, all cross-time frequencies and every 32nd endpoint.
    p, q = physical(a, settings), physical(b, settings)
    samplep = p[31::32] if len(p) >= 32 else p
    sampleq = q[31::32] if len(q) >= 32 else q
    distances = [source_distances(x, y, settings[2], {3: np.pi, 4: 2*np.pi})['assignment']
                 for x in samplep for y in sampleq]
    return dict(**cross_frequency_distance(p[..., 0], q[..., 0]),
                full_assignment_rms=float(np.sqrt(np.mean(np.square(distances)))))


def distribution_metrics(x, ll, settings):
    means = ll.mean(1); pairs = []
    for a, b in combinations(range(len(ll)), 2):
        pairs.append(dict(a=a, b=b, ks=float(ks_2samp(ll[a], ll[b]).statistic),
                          **pair_geometry(x[a], x[b], settings)))
    return dict(n=ll.shape[1], mean_logL=means, mean_sd=float(means.std(ddof=1)),
                mean_mad=float(np.median(abs(means-np.median(means)))),
                mean_iqr=float(np.ptp(np.quantile(means, [.25, .75]))),
                median_KS=float(np.median([z['ks'] for z in pairs])), max_KS=max(z['ks'] for z in pairs),
                median_frequency=float(np.median([z['assignment'] for z in pairs])),
                median_sorted_mean=float(np.median([z['sorted_mean_distance'] for z in pairs])),
                median_full=float(np.median([z['full_assignment_rms'] for z in pairs])),
                fractions=(ll > CANDIDATE).mean(1), pairs=pairs)


def transfer_summary(arrivals, begin, end):
    selected = [z for z in arrivals if begin <= z['sweep'] < end]
    return dict(directional=len(selected), unique_realizations=len({z['token'] for z in selected}),
                per_walker=[dict(received=sum(z['to_walker'] == w for z in selected),
                    donor_ids=sorted({z['from_walker'] for z in selected if z['to_walker'] == w}))
                    for w in range(8)])


def exchange_summary(ex, begin, end):
    mask = (ex['sweep'] >= begin) & (ex['sweep'] < end)
    n = int(mask.sum()); acc = int(ex['accepted'][mask].sum())
    per_walker = []
    for w in range(8):
        rows, sides = np.where(ex['pair'] == w)
        keep = mask[rows]; rows, sides = rows[keep], sides[keep]
        ownpass = ex['proposed_loglikelihood'][rows, sides] > ELL12
        selected = ex['selected_frequency'][rows, sides]
        ranks = ex['selected_ranks'][rows, sides]
        labels = ex['labels'][rows, sides]
        per_walker.append(dict(proposals=len(rows), accepted=int(ex['accepted'][rows].sum()),
            contour_survivors=int(ex['joint_contour'][rows].sum()), own_proposal_pass=int(ownpass.sum()),
            partner_proposal_pass=int((ex['proposed_loglikelihood'][rows, 1-sides] > ELL12).sum()),
            selected_frequency_quantiles=np.quantile(selected, [0, .25, .5, .75, 1]),
            selected_inside_frozen_interval=float(np.mean((selected >= .0018407250) & (selected <= .0018412366))),
            selected_rank_quantiles=np.quantile(ranks, [0, .25, .5, .75, 1]),
            label_counts=np.bincount(labels, minlength=9)))
    return dict(proposals=n, accepted=acc, acceptance=acc/n,
                contour_survivors=int(ex['joint_contour'][mask].sum()), per_walker=per_walker)


def weighted_summary(values, weights):
    order = np.argsort(values); values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positive = weights > 0; values, weights = values[positive], weights[positive]
    weights = weights/weights.sum(); cumulative = np.cumsum(weights)
    return dict(min=float(values.min()), median=float(values[np.searchsorted(cumulative, .5)]),
                max=float(values.max()), mean=float(np.sum(values*weights)),
                rms=float(np.sqrt(np.sum(values**2*weights))))


def pool_geometry(position, ll, settings):
    """Analytic pair-distance laws over possible pools; never allocate eight starts."""
    points, donors, _, _, entries = eligible_pool(position, ll)
    n = len(points); p = physical(points, settings); nd = len(np.unique(donors))
    assert entries == n, "Saved pool must have no duplicated states for entry-uniform D weights"
    if n < 8 or nd != 7:
        raise ValueError('This descriptive calculation is specified for the saved seven-donor pool')
    counts = np.bincount(donors, minlength=8); extra = np.flatnonzero(counts >= 2)
    vals = []; balanced_weights = []
    for i, j in combinations(range(n), 2):
        frequency = float(np.sqrt(np.mean(((np.sort(p[i,:,0])-np.sort(p[j,:,0]))/DF)**2)))
        full = source_distances(p[i], p[j], settings[2], {3: np.pi, 4: 2*np.pi})['assignment']
        vals.append((frequency, full))
        a, b = donors[i], donors[j]
        if a == b:
            weight = (1/len(extra))/((counts[a]*(counts[a]-1))/2)/28
        else:
            weight = (1 + (a in extra)/len(extra) + (b in extra)/len(extra))/(counts[a]*counts[b])/28
        balanced_weights.append(weight)
    vals = np.asarray(vals); bw = np.asarray(balanced_weights)
    assert np.isclose(bw.sum(), 1)
    result = {}
    for method, values, weights in [
        ('B', vals, np.ones(len(vals))/len(vals)),
        ('C', vals, bw),
        ('D', np.vstack([vals, [0., 0.]]), np.r_[np.full(len(vals), 2/n**2), 1/n]),
    ]:
        result[method] = {name: weighted_summary(values[:, k], weights)
                          for k, name in enumerate(['frequency_set_RMS_bins', 'six_coordinate_source_set_RMS'])}
    return dict(methodology='No start ensemble drawn or stored. B: uniform eight distinct unique states. C: all seven donors once, extra donor uniform among donors with >=2 unique states, uniform distinct states within donor. D: uniform retained survivor entries with replacement. Reported laws concern an unordered pair of slots, analytically averaged over all possible allocations.',
                eligible_unique=n, eligible_entries=entries, donor_unique_counts=counts,
                extra_eligible_donors=extra, pair_count=len(vals), distances=result,
                duplicate_pair_probability_D=1/n,
                probability_any_duplicate_in_eight_D=1-float(np.prod((n-np.arange(8))/n)))


def verify_inputs():
    before = {str(p): digest(p) for root in [B, A] for p in root.rglob('*') if p.is_file()}
    bv, av = read(B/'validation.json'), read(A/'validation.json')
    for v in [bv, av]:
        for path, h in v['output_sha256'].items():
            assert before[path] == h, path
    report, checkpoint, preflight = read(B/'report.json'), read(B/'checkpoint_level12.json'), read(B/'preflight.json')
    assert report['status'] == 'STOP_failure'
    assert read(B/'selection_candidate.json')['threshold'] == CANDIDATE
    assert digest(B/'selection_candidate.json') == report['candidate_sha256']
    assert not (B/'checkpoint_level13.json').exists()
    assert len(checkpoint['payload']['thresholds']) == 13
    assert checkpoint['payload']['thresholds'] == preflight['frozen_thresholds']
    assert checkpoint['payload']['log_masses'] == preflight['frozen_log_masses']
    assert (checkpoint['payload']['thresholds'][-1], checkpoint['payload']['log_masses'][-1]) == (ELL12, LOGX12)
    assert hashlib.sha256(json.dumps(checkpoint['payload'], sort_keys=True, allow_nan=False).encode()).hexdigest() == checkpoint['sha256']
    source = '/tmp/lisa_dns_stage4z_population_level12/checkpoint_level12.json'
    assert digest(B/'checkpoint_level12.json') == preflight['file_sha256'][source]
    for name in ['selection', 'calibration']:
        br = read(B/name/'report.json')
        assert br['completed_sweeps'] == 384 and br['structural_failures'] == 0
        assert report['population'][name]['exact_replay_passed']
        assert br['max_cache_prior_error'] == br['max_cache_logL_error'] == 0
    return before, checkpoint, report


def write_diagnostics(output, result):
    output = Path(output).resolve()
    if any(output == root.resolve() or root.resolve() in output.parents for root in [A, B]):
        raise ValueError('Historical artifact directories are read-only')
    output.mkdir(exist_ok=False)
    (output/'diagnostics.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False))


def main():
    assert not OUT.exists(), 'New diagnostic output only; never overwrite historical files'
    before, checkpoint, report = verify_inputs()
    settings = physical_settings(checkpoint, load(B/'selection/exchanges.npz'))
    histories = {n: load(B/n/'trace.npz') for n in ['selection', 'calibration']}
    histories['stage5a'] = load(A/'trace.npz')
    result = dict(stage='5C', likelihood_calls=0, sampling_calls=0, promotion_calls=0,
                  official_result='Stage-5B FAILED promotion/survivor contract; level 12 remains frozen',
                  frozen_endpoint=[ELL12, LOGX12], candidate=CANDIDATE,
                  source_sha256=before, physical_bounds=dict(zip(NAMES, np.column_stack(settings[:2]))), accounting={}, windows={}, starts={}, exchanges={}, provenance={}, trajectories={})
    for name, raw in histories.items():
        root = A if name == 'stage5a' else B/name
        starts = load(root/'initial_states.npz');ex = load(root/'exchanges.npz')
        arrivals = read(root/'provenance_transfers.json')
        ll, x = raw['loglikelihood'], raw['position']
        assert x.shape == (8, 512 if name == 'stage5a' else 384, 54) and np.all(ll > ELL12)
        assert np.isfinite(x).all() and np.isfinite(ll).all()
        result['starts'][name] = dict(logL=starts['loglikelihood'], mean=float(starts['loglikelihood'].mean()),
            sd=float(starts['loglikelihood'].std(ddof=1)), geometry=distribution_metrics(starts['position'][:,None], starts['loglikelihood'][:,None], settings))
        result['trajectories'][name] = dict(block32_mean=ll.reshape(8,-1,32).mean(2),
            block32_max=ll.reshape(8,-1,32).max(2), block32_exceedance=(ll>CANDIDATE).reshape(8,-1,32).mean(2),
            min=ll.min(1), max=ll.max(1), final=ll[:,-1])
        if name == 'stage5a':
            windows = {'first_half': (0,256), 'second_half': (256,512), 'full': (0,512)}
        else:
            windows = {'burnin': (0,128), 'first_half': (128,256), 'second_half': (256,384), 'retained': (128,384), 'full': (0,384)}
            result['accounting'][name] = {part: [survivor_summary(z) for z in ll[:,begin:end]]
                                         for part, (begin,end) in {'burnin': (0,128), 'retained': (128,384)}.items()}
        result['windows'][name] = {};result['exchanges'][name] = {};result['provenance'][name] = {}
        for window, (begin,end) in windows.items():
            result['windows'][name][window] = distribution_metrics(x[:,begin:end],ll[:,begin:end],settings)
            result['exchanges'][name][window] = exchange_summary(ex,begin,end)
            result['provenance'][name][window] = transfer_summary(arrivals,begin,end)
        print('Summarized',name,flush=True)
    result['sensitivity'] = sensitivity(histories['selection']['loglikelihood'][:,128:])
    result['pool_feasibility'] = pool_feasibility(histories['selection']['position'][:,128:], histories['selection']['loglikelihood'][:,128:])
    result['pool_geometry'] = pool_geometry(histories['selection']['position'][:,128:], histories['selection']['loglikelihood'][:,128:], settings)
    # Cross-bank distributions and geometry, all 64 same/different lineage pairs.
    result['cross_bank'] = {}
    for left,right in [('selection','calibration'),('selection','stage5a'),('calibration','stage5a')]:
        pairrows=[]
        l,r=histories[left],histories[right]
        il=slice(128,384);ir=slice(None) if right=='stage5a' else slice(128,384)
        for i in range(8):
            for j in range(8):
                pairrows.append(dict(a=i,b=j,ks=float(ks_2samp(l['loglikelihood'][i,il],r['loglikelihood'][j,ir]).statistic),
                    **pair_geometry(l['position'][i,il],r['position'][j,ir],settings)))
        result['cross_bank'][left+'/'+right]=dict(pairs=pairrows,pooled_KS=float(ks_2samp(l['loglikelihood'][:,il].ravel(),r['loglikelihood'][:,ir].ravel()).statistic))
    sa=load(A/'initial_states.npz');bc=load(B/'calibration/initial_states.npz')
    result['stage5a_and_calibration_same_starts']={k:sa[k].tobytes()==bc[k].tobytes() for k in sa}
    # Verify Stage-4I coordinate mapping against the already saved Stage-5B geometry.
    old=read(B/'comparison.json')['attempts']['stage5b']['geometry']
    for n in ['selection','calibration']:
        for a,b in [('median_frequency','median_frequency_set_rms'),('median_sorted_mean','median_sorted_frequency_mean_distance'),('median_full','median_full_source_set_rms')]:
            np.testing.assert_allclose(result['windows'][n]['retained'][a],old[n][b],rtol=0,atol=1e-8)
    for path,h in before.items():assert digest(path)==h,path
    result['input_hashes_unchanged']=True
    write_diagnostics(OUT, result)
    print('Stage-5C read-only diagnostics complete; historical inputs unchanged',flush=True)


if __name__ == '__main__':
    main()
