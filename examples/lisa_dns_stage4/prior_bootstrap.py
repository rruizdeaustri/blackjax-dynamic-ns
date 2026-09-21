"""Direct-prior bootstrap helpers, isolated from the generic DNS core."""
import hashlib

import jax
import numpy as np
from scipy.stats import norm


def array_sha256(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()


def draw_prior_banks(sample_prior, selection_seed, calibration_seed, draws):
    """Call the problem's actual sampler with separate keys; never use MCMC."""
    if selection_seed == calibration_seed:
        raise ValueError('Selection and calibration seeds must differ')
    if draws < 1:
        raise ValueError('draws must be positive')
    banks = [np.asarray(sample_prior(jax.random.key(seed), draws))
             for seed in (selection_seed, calibration_seed)]
    if banks[0].shape != banks[1].shape or banks[0].ndim != 2 or len(banks[0]) != draws:
        raise ValueError('Invalid prior bank shapes')
    if not all(np.all(np.isfinite(bank)) for bank in banks):
        raise ValueError('Nonfinite prior sample')
    overlap = len({row.tobytes() for row in banks[0]} & {row.tobytes() for row in banks[1]})
    if overlap:
        raise ValueError('Prior banks share samples; inspect their sampler/streams')
    return banks, dict(selection_seed=selection_seed, calibration_seed=calibration_seed,
                      draws_per_bank=draws, shared_rows=overlap,
                      selection_sha256=array_sha256(banks[0]),
                      calibration_sha256=array_sha256(banks[1]))


def select_iid_threshold(selection_values, target_compression=np.exp(-1)):
    """Stage-3 order statistic and strict tail, without calling MCMC diagnostics."""
    values = np.asarray(selection_values)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite nonempty likelihood vector')
    if not 0 < target_compression < 1:
        raise ValueError('Invalid compression')
    ordered = np.sort(values)
    index = int(np.floor(len(values) * (1 - target_compression)))
    threshold = float(ordered[index])
    fraction = float(np.mean(values > threshold))
    return dict(threshold=threshold, order_index=index, ratio=fraction,
                sample_count=len(values), status='ok' if 0 < fraction < 1 else 'constant_or_zero_tail',
                notice='Selection fraction is not a calibration estimate or binomial confidence interval.')


def calibrate_iid(calibration_values, threshold, min_tail_count=20):
    """Independent binomial estimate conditional on the selected threshold.

    Predeclared engineering gate: at least 20 draws in each Bernoulli outcome.
    No MCMC ESS is computed. Wilson interval is an approximate 95% interval.
    """
    values = np.asarray(calibration_values)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite nonempty likelihood vector')
    n = len(values)
    k = int(np.sum(values > threshold))
    p = k / n
    z = float(norm.ppf(.975))
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    interval = [center-half, center+half]
    accepted = min(k, n-k) >= min_tail_count
    return dict(method='IID binomial; Wilson 95% interval', sample_count=n, exceeds=k,
                ratio=p, standard_error=float(np.sqrt(p*(1-p)/n)), interval=interval,
                log_ratio=float(np.log(p)) if p else None,
                log_ratio_interval=np.log(interval).tolist() if 0 < interval[0] else None,
                min_tail_count=min_tail_count, accepted=accepted)


def survivor_indices(values, threshold, key, walkers=8):
    """Distinct survivors from this bank only; do not duplicate one start."""
    eligible = np.flatnonzero(np.asarray(values) > threshold)
    if len(eligible) < walkers:
        raise ValueError('Too few distinct survivors')
    return np.asarray(jax.random.choice(key, eligible, (walkers,), replace=False))


def correlation_breakdown(values, threshold, block_size=32):
    """Expose the separate uncertainty terms underlying Stage-3 tail ESS."""
    values = np.asarray(values)
    indicator = (values > threshold).astype(float)
    p = indicator.mean()
    blocks = len(values)//block_size
    means = indicator[:blocks*block_size].reshape(blocks, block_size, -1).mean(1)
    terms = dict(iid=float(np.sqrt(p*(1-p)/indicator.size)),
                 within_walker_batch=float(np.sqrt(np.var(means, ddof=1)/means.size)),
                 between_walker=float(np.std(indicator.mean(0), ddof=1)/np.sqrt(values.shape[1])))
    return dict(standard_error_terms=terms, dominant_term=max(terms, key=terms.get),
                block_exceedance_fractions=means.tolist(),
                walker_mean_logL=values.mean(0).tolist(),
                first_half_walker_mean_logL=values[:len(values)//2].mean(0).tolist(),
                second_half_walker_mean_logL=values[len(values)//2:].mean(0).tolist())
