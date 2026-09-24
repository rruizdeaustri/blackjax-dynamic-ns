"""Stage 4S: saved-array-only relevance analysis; no model or kernel imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import multinomial

from examples.lisa_dns_stage4.diagnose_source_permutations import (
    DF, NAMES, circular_difference, source_distances,
)

ELL9 = -110252.99476697217
INPUT = Path('/tmp/lisa_dns_stage4r_source_exchange')
OUTPUT = Path('/tmp/lisa_dns_stage4s_uniform_relevance')
CATEGORIES = ('N', 'A', 'B', 'AB')


def ranks_from_saved_gain(gain):
    """Exactly Stage-4R descending gain, stable lowest-label tie breaking."""
    gain = np.asarray(gain)
    if gain.shape[-1] != 9 or not np.isfinite(gain).all():
        raise ValueError('Nine finite gains required')
    return np.argsort(np.argsort(-gain, axis=-1, kind='stable'),
                      axis=-1, kind='stable') + 1


def touch_category(rank_a, rank_b, k=1):
    if k not in (1, 2, 3):
        raise ValueError('Only predeclared top-1/top-2/top-3 descriptions')
    if not (1 <= rank_a <= 9 and 1 <= rank_b <= 9):
        raise ValueError('Source ranks must be in 1..9')
    return ('AB' if rank_b <= k else 'A') if rank_a <= k else ('B' if rank_b <= k else 'N')


def touch_probabilities(k=1):
    if k not in (1, 2, 3):
        raise ValueError('Only predeclared ranks')
    p = k / 9
    return np.array([(1-p)**2, 2*p*(1-p), p*p])


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, min=None, q25=None, median=None, q75=None, max=None)
    assert np.isfinite(a).all()
    return dict(n=len(a), **dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                    np.quantile(a, [0, .25, .5, .75, 1]).tolist())))


def summary(rows):
    n = len(rows)
    result = dict(n=n, joint_survives=sum(r['joint_survives'] for r in rows))
    result['joint_fraction'] = result['joint_survives']/n if n else None
    for side in ('A', 'B'):
        count = sum(r[f'passes_{side}'] for r in rows)
        result[side] = dict(survives=count, fraction=count/n if n else None,
                            delta_logL=distribution([r[f'delta_logL_{side}'] for r in rows]))
    result['minimum_side_margin'] = distribution([r['minimum_side_margin'] for r in rows])
    return result


def strata(rows, k):
    groups = {c: [r for r in rows if r[f'top{k}_category'] == c] for c in CATEGORIES}
    assert sum(map(len, groups.values())) == len(rows)
    combined = {**groups, '0': groups['N'], '1': groups['A']+groups['B'],
                '2': groups['AB'], '>=1': groups['A']+groups['B']+groups['AB']}
    if k == 1:
        return {c: summary(g) for c, g in combined.items()}
    # Sensitivity is restricted to counts/survival, not new geometry or delta analyses.
    result = {}
    for category, group in combined.items():
        n = len(group)
        joint = sum(r['joint_survives'] for r in group)
        result[category] = dict(n=n, joint_survives=joint, joint_fraction=joint/n if n else None)
        for side in ('A', 'B'):
            count = sum(r[f'passes_{side}'] for r in group)
            result[category][side] = dict(survives=count, fraction=count/n if n else None)
    return result


def exact_selection_check(counts):
    """Exact multinomial tail ordered by Pearson discrepancy; no simulation."""
    counts = np.asarray(counts)
    n = int(counts.sum())
    probabilities = touch_probabilities()
    expected = n*probabilities
    statistic = float(np.sum((counts-expected)**2/expected))
    tail = 0.
    for a in range(n+1):
        b = np.arange(n-a+1)
        values = np.column_stack([np.full(len(b), a), b, n-a-b])
        discrepancies = np.sum((values-expected)**2/expected, axis=1)
        tail += multinomial.pmf(values, n, probabilities)[discrepancies >= statistic-1e-12].sum()
    return dict(observed=counts.tolist(), expected=expected.tolist(),
                probabilities=probabilities.tolist(), pearson_statistic=statistic,
                exact_multinomial_tail=float(tail),
                interpretation='Mildly unusual realization, not a comfortable 5% goodness-of-fit pass; frozen PCG64 labels reproduce exactly. No resampling or weight changes.')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def analyze_saved_arrays(base, swap, widths):
    """Pure array analysis. Reconstruct physical proposals from saved donor blocks."""
    ranks = ranks_from_saved_gain(base['source_gain'])
    np.testing.assert_array_equal(ranks, base['ranks'])
    np.testing.assert_array_equal(np.argmin(ranks, axis=-1), base['dominant'])
    rows = []
    for n in range(len(swap['i'])):
        i, j, t, b, c = [int(swap[key][n]) for key in ['i', 'j', 't', 'label_i', 'label_j']]
        row = dict(case=n, i=i, j=j, t=t, index=int(base['indices'][t]),
                   label_A=b, label_B=c, rank_A=int(ranks[i,t,b]), rank_B=int(ranks[j,t,c]))
        for k in (1, 2, 3):
            row[f'top{k}_category'] = touch_category(row['rank_A'], row['rank_B'], k)
        for side, (recipient, donor, rb, db) in enumerate(((i,j,b,c), (j,i,c,b))):
            name = ('A', 'B')[side]
            old = base['position'][recipient,t].reshape(9,6)
            source = base['position'][donor,t].reshape(9,6)
            expected = old.copy(); expected[rb] = source[db]
            new = swap['position'][n,side].reshape(9,6)
            assert new.tobytes() == expected.tobytes()
            assert new[rb].tobytes() == source[db].tobytes()
            keep = np.arange(9) != rb
            assert new[keep].tobytes() == old[keep].tobytes()
            oldphys = base['physical'][recipient,t]
            newphys = oldphys.copy(); newphys[rb] = base['physical'][donor,t,db]
            delta = newphys[rb]-oldphys[rb]
            for coord, period in {3:np.pi, 4:2*np.pi}.items():
                delta[coord] = circular_difference(newphys[rb,coord],oldphys[rb,coord],period)
            distance = source_distances(oldphys,newphys,widths,{3:np.pi,4:2*np.pi})
            frequency = source_distances(oldphys[:,0,None],newphys[:,0,None],[DF])
            row[f'geometry_{name}'] = dict(
                latent_block_l2=float(np.linalg.norm(new[rb]-old[rb])),
                physical_block_rms=float(np.sqrt(np.mean((delta/widths)**2))),
                labelled_catalogue_rms=distance['labelled'],
                frequency_set_bins=frequency['assignment'],
                six_coordinate_set_rms=distance['assignment'],
                f0_shift_hz=float(delta[0]), absolute_f0_shift_bins=float(abs(delta[0])/DF))
            ll = float(swap['loglikelihood'][n,side])
            row[f'logL_{name}'] = ll
            row[f'passes_{name}'] = bool(ll > ELL9)
            row[f'delta_logL_{name}'] = float(ll-base['loglikelihood'][recipient,t])
            dominant = int(base['dominant'][recipient,t])
            # No post-swap gains: verify replacement geometrically, without a distance cutoff.
            row[f'replacement_{name}'] = dict(
                dominant_touched=rb == dominant,
                selected_block_changed=not np.array_equal(new[rb],old[rb]),
                donor_copy_exact=new[rb].tobytes()==source[db].tobytes(),
                old_dominant_block_absent=not any(np.array_equal(z,old[dominant]) for z in new),
                incoming_is_donor_dominant=db == int(base['dominant'][donor,t]),
                donor_f0_hz=float(newphys[rb,0]), original_f0_hz=float(oldphys[rb,0]),
                hungarian_frequency_old_dominant_to_replaced_slot=frequency['permutation'][dominant]==rb,
                hungarian_six_coordinate_old_dominant_to_replaced_slot=distance['permutation'][dominant]==rb)
        row['minimum_side_margin'] = min(row['logL_A'],row['logL_B'])-ELL9
        row['joint_survives'] = row['passes_A'] and row['passes_B']
        assert row['joint_survives'] == bool(swap['joint_survives'][n])
        np.testing.assert_array_equal([row['passes_A'],row['passes_B']],swap['passes'][n])
        rows.append(row)
    return rows


def movement(rows):
    keys = ('latent_block_l2','physical_block_rms','labelled_catalogue_rms',
            'frequency_set_bins','six_coordinate_set_rms','f0_shift_hz','absolute_f0_shift_bins')
    return {side: {key: distribution([r[f'geometry_{side}'][key] for r in rows]) for key in keys}
            for side in ('A','B')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_saved_inputs(root):
    report = json.loads((root/'report.json').read_text())
    assert report['status']=='completed_diagnostic_only' and report['structural_failures']==0
    assert report['ell9']==ELL9
    for path, expected in report['file_sha256'].items():
        assert digest(path)==expected, path
    paths = list(root.glob('*.npz')) + [root/'report.json',root/'design.json',root/'validation.json']
    hashes = {str(p):digest(p) for p in paths}
    base = load_arrays(root/'bases.npz')
    q = load_arrays(Path('/tmp/lisa_dns_stage4q_transplantations/bases.npz'))
    assert base['position'].tobytes()==q['position'].tobytes()
    assert base['position'].shape==(8,16,54)
    np.testing.assert_array_equal(base['indices'],np.arange(0,256,17))
    assert np.isfinite(base['logprior']).all() and np.isfinite(base['loglikelihood']).all()
    assert np.all(base['loglikelihood']>ELL9)
    bounds_path = '/tmp/lisa_dns_stage4_prior_smoke/report.json'
    assert bounds_path in report['file_sha256']
    bounds = json.loads(Path(bounds_path).read_text())['prior_audit'][0]['physical_bounds']
    lo = np.array([bounds[n][0] for n in NAMES]); hi = np.array([bounds[n][1] for n in NAMES])
    physical = lo+(hi-lo)*expit(base['position'].reshape(8,16,9,6))
    np.testing.assert_array_equal(physical,base['physical'])
    widths = hi-lo; widths[3]=np.pi; widths[4]=2*np.pi
    design = json.loads((root/'design.json').read_text())
    cases = [(i,j,t) for i in range(8) for j in range(i+1,8) for t in range(16)]
    persistent = set(design['persistent_oracle_walkers'])
    swaps = {}
    for mode in ('uniform','dominant','oracle'):
        swap = load_arrays(root/f'{mode}_swaps.npz')
        raw = load_arrays(root/f'{mode}_evaluations.npz')
        for key in ('position','logprior','loglikelihood'):
            np.testing.assert_array_equal(swap[key],raw[key])
            assert np.isfinite(swap[key]).all()
        expected = [case for case in cases if mode!='oracle' or set(case[:2])<=persistent]
        assert list(zip(swap['i'],swap['j'],swap['t']))==expected
        rows = report['results'][mode]['joint_rows']
        assert len(rows)==len(expected)
        for n,row in enumerate(rows):
            for key in ('i','j','t','index','label_i','label_j','joint_survives'):
                assert swap[key][n]==row[key]
            np.testing.assert_array_equal(swap['loglikelihood'][n],row['new_logL'])
            np.testing.assert_array_equal(swap['logprior'][n],row['new_logprior'])
        if mode=='uniform':
            labels = np.random.Generator(np.random.PCG64(20260923)).integers(0,9,(448,2))
            np.testing.assert_array_equal(labels,design['uniform_labels'])
            np.testing.assert_array_equal(labels,np.column_stack([swap['label_i'],swap['label_j']]))
        else:
            for side,key in [('i','label_i'),('j','label_j')]:
                wanted = base['dominant'][swap[side],swap['t']] if mode=='dominant' else base['historical'][swap[side]]
                np.testing.assert_array_equal(swap[key],wanted)
        swaps[mode] = swap
    return base, swaps, widths, hashes, len(report['file_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    base, swaps, widths, hashes, verified_hashes = verify_saved_inputs(args.input)
    rows = {mode:analyze_saved_arrays(base,swap,widths) for mode,swap in swaps.items()}
    uniform = rows['uniform']
    primary = strata(uniform,1)
    groups = {
        'uniform_0': [r for r in uniform if r['top1_category']=='N'],
        'uniform_1': [r for r in uniform if r['top1_category'] in ('A','B')],
        'uniform_2': [r for r in uniform if r['top1_category']=='AB'],
        'uniform_relevant': [r for r in uniform if r['top1_category']!='N'],
        'dominant': rows['dominant'], 'oracle': rows['oracle'],
    }
    comparisons = {name:dict(survival=summary(group), all_proposals_movement=movement(group),
                             surviving_movement=movement([r for r in group if r['joint_survives']]))
                   for name,group in groups.items()}
    relevant_survivors = [r for r in groups['uniform_relevant'] if r['joint_survives']]
    replaced = [r for r in relevant_survivors if all(
        r[f'replacement_{s}']['selected_block_changed'] and r[f'replacement_{s}']['donor_copy_exact']
        and r[f'replacement_{s}']['old_dominant_block_absent'] and r[f'replacement_{s}']['incoming_is_donor_dominant']
        for s in ('A','B') if r[f'replacement_{s}']['dominant_touched'])]
    result = dict(stage='4S',status='completed_saved_array_analysis',ell9=ELL9,
                  likelihood_calls=0,prior_calls=0,source_gain_calls=0,structural_failures=0,
                  verified_catalogues=sum(len(z)*2 for z in rows.values()),
                  existing_provenance_hashes_verified=verified_hashes,input_sha256=hashes,
                  hash_notice='Stage-4R recorded provenance hashes verified. Its output NPZ files had no earlier recorded output digests; new input hashes plus redundant JSON/raw-array checks and exact reconstruction establish this analysis baseline.',
                  primary=primary,sensitivity={str(k):strata(uniform,k) for k in (2,3)},
                  selection_check=exact_selection_check([primary[str(k)]['n'] for k in range(3)]),
                  comparisons=comparisons, rows=rows,
                  relevant_replacement=dict(relevant_survivors=len(relevant_survivors),
                      geometrically_replaced=len(replaced),
                      fraction=len(replaced)/len(relevant_survivors) if relevant_survivors else None,
                      rows=relevant_survivors,
                      limitation='No post-swap source_gain saved or evaluated. Exact removal/copy and saved physical matching establish realization replacement, not post-swap likelihood dominance or unique astrophysical identity.'),
                  source_identity=dict(ranks=base['ranks'].tolist(),dominant=base['dominant'].tolist(),
                      top2=np.argsort(base['ranks'],axis=-1)[...,:2].tolist(),
                      top3=np.argsort(base['ranks'],axis=-1)[...,:3].tolist()))
    for path, expected in hashes.items():
        assert digest(path)==expected, path
    result['input_hashes_unchanged']=True
    args.output.mkdir(exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(primary=primary,selection=result['selection_check'],
                          replacement_count=len(replaced)),indent=2))


if __name__=='__main__':
    main()
