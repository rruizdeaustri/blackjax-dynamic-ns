"""Inference from explicitly labelled, fixed-contour DNS reconstruction strata.

Deterministic-mixture importance sampling, with calibrated masses as a plug-in.
Construction/adaptation samples and diffusive-mixture traces are not accepted.
See docs/examples/dns_inference_reconstruction.md for the measure derivation.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

TARGET = 'independent_fixed_contour_reconstruction'


def _mass_table(thresholds, log_masses):
    t, m = np.asarray(thresholds), np.asarray(log_masses)
    if (t.ndim != 1 or not len(t) or m.shape != t.shape or not np.isneginf(t[0])
            or not np.isfinite(t[1:]).all() or not np.all(np.diff(t)>0)):
        raise ValueError('Missing or non-monotonic threshold/mass metadata')
    if not np.isfinite(m).all() or m[0] != 0 or not np.all(np.diff(m)<0):
        raise ValueError('X_0 must be one; X must strictly decrease; negative shell mass')
    masses = np.exp(m)
    shells = -np.diff(masses)
    if np.any(shells < 0) or shells.sum()+masses[-1] > 1+1e-12:
        raise ValueError('Invalid accounted prior mass')
    return t, m


@dataclass(frozen=True)
class ReconstructionRecords:
    thresholds: np.ndarray
    log_masses: np.ndarray
    position: np.ndarray
    logprior: np.ndarray
    loglikelihood: np.ndarray
    origin_level: np.ndarray
    walker: np.ndarray
    draw: np.ndarray
    event_id: np.ndarray
    metadata: dict

    def validate(self):
        thresholds, _ = _mass_table(self.thresholds, self.log_masses)
        n = len(self.loglikelihood)
        if not n or np.asarray(self.position).ndim != 2 or len(self.position) != n:
            raise ValueError('Nonempty coordinate/sample arrays required')
        for name in ('logprior', 'loglikelihood', 'origin_level', 'walker', 'draw', 'event_id'):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError('Inconsistent sample metadata: '+name)
        if any(np.asarray(v).dtype != np.float64 or not np.isfinite(v).all()
               for v in (self.position, self.logprior, self.loglikelihood)):
            raise ValueError('Finite float64 coordinates and scalar caches required')
        for name in ('origin_level', 'walker', 'draw'):
            v = np.asarray(getattr(self, name))
            if not np.issubdtype(v.dtype, np.integer) or np.any(v < 0):
                raise ValueError('Corrupted integer level/walker/draw index')
        if np.any(self.origin_level >= len(thresholds)):
            raise ValueError('Corrupted originating level index')
        if np.any(self.loglikelihood <= thresholds[self.origin_level]):
            raise ValueError('Sample assigned to an impossible originating level')
        if len(set(map(str, self.event_id))) != n:
            raise ValueError('Duplicate retained event ID: accidental double counting')
        triples = np.stack([self.origin_level, self.walker, self.draw], axis=1)
        if len(np.unique(triples, axis=0)) != n:
            raise ValueError('Duplicate level/walker/draw event')
        if self.metadata.get('sampling_target') != TARGET:
            raise ValueError('Only fresh fixed-contour reconstruction streams are supported')
        if self.metadata.get('mass_status') != ['prior']+['calibrated']*(len(thresholds)-1):
            raise ValueError('Missing calibrated X; an uncalibrated candidate is not a level')
        levels = self.metadata.get('levels')
        if not isinstance(levels, list) or len(levels) != len(thresholds):
            raise ValueError('Missing level metadata')
        for j, row in enumerate(levels):
            if row.get('level') != j or float(row.get('threshold', np.nan)) != thresholds[j] or not row.get('stream_id'):
                raise ValueError('Missing or inconsistent originating level metadata')
            if not np.any(self.origin_level == j):
                raise ValueError('Each declared proposal requires reconstruction samples')
        return self

    def through(self, level):
        self.validate()
        if not isinstance(level, (int, np.integer)) or not 0 <= level < len(self.thresholds):
            raise ValueError('Invalid included level')
        keep = self.origin_level <= level
        fields = {name: np.asarray(getattr(self, name))[keep].copy() for name in
                  ('position', 'logprior', 'loglikelihood', 'origin_level', 'walker', 'draw', 'event_id')}
        metadata = dict(self.metadata, levels=self.metadata['levels'][:level+1],
                        mass_status=self.metadata['mass_status'][:level+1])
        return replace(self, thresholds=self.thresholds[:level+1].copy(),
                       log_masses=self.log_masses[:level+1].copy(), metadata=metadata, **fields)


def validate_weights(weights, n=None):
    w = np.asarray(weights)
    if w.ndim != 1 or not len(w) or (n is not None and len(w) != n):
        raise ValueError('Weight shape mismatch')
    if not np.isfinite(w).all() or np.any(w < 0) or abs(float(w.sum())-1) > 1e-12:
        raise ValueError('Posterior weights must be finite, nonnegative, and normalize to one')
    return w


def posterior_weights(log_importance):
    values = np.asarray(log_importance, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite log importance ratios required')
    return validate_weights(np.exp(values-logsumexp(values)))


def reconstruct_evidence(records, *, log_masses=None):
    """Return evidence and posterior weights; optional masses are sensitivity only.

    With true X this estimates the known deterministic-mixture measure. Recorded
    calibrated X gives a plug-in estimate, not an exact finite-sample unbiased one.
    """
    records.validate()
    t, m = _mass_table(records.thresholds, records.log_masses if log_masses is None else log_masses)
    n = len(records.loglikelihood)
    counts = np.bincount(records.origin_level, minlength=len(t))
    log_alpha = np.log(counts/n)
    support = records.loglikelihood[:, None] > t[None, :]
    log_s = logsumexp(np.where(support, log_alpha[None, :]-m[None, :], -np.inf), axis=1)
    log_r = records.loglikelihood-log_s
    weights = posterior_weights(log_r)
    positive = weights > 0
    return dict(logZ=float(logsumexp(log_r)-np.log(n)), log_importance=log_r,
                weights=weights, weight_ess=float(1/np.sum(weights**2)),
                maximum_weight=float(weights.max()),
                entropy=float(-np.sum(weights[positive]*np.log(weights[positive]))),
                raw_samples=n, per_level_counts=counts.tolist(),
                mass_treatment='recorded-calibration-plugin' if log_masses is None else 'explicit-mass-sensitivity')


def posterior_expectation(values, weights):
    values = np.asarray(values)
    w = validate_weights(weights, len(values))
    if not np.isfinite(values).all():
        raise ValueError('Finite observable required')
    return np.tensordot(w, values, axes=(0, 0))


def posterior_quantiles(values, weights, probabilities):
    x = np.asarray(values)
    w = validate_weights(weights, len(x))
    p = np.asarray(probabilities)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid values or quantile probabilities')
    order = np.argsort(x, kind='stable')
    cdf = np.cumsum(w[order]); cdf[-1] = 1.
    return x[order][np.minimum(np.searchsorted(cdf, p, side='left'), len(x)-1)]


def weighted_cdf_discrepancy(values, weights, reference_cdf):
    x = np.asarray(values)
    w = validate_weights(weights, len(x))
    unique, inverse = np.unique(x, return_inverse=True)
    jumps = np.bincount(inverse, weights=w)
    right = np.cumsum(jumps); left = right-jumps
    truth = np.asarray(reference_cdf(unique))
    if truth.shape != unique.shape or not np.isfinite(truth).all() or np.any((truth<0)|(truth>1)):
        raise ValueError('Invalid reference CDF')
    return float(max(np.max(abs(right-truth)), np.max(abs(left-truth))))


def shell_diagnostic(records):
    """Non-authoritative empirical shell means, including the terminal cap."""
    records.validate()
    mass = np.exp(records.log_masses)
    shell_mass = np.r_[-np.diff(mass), mass[-1]]
    representatives, counts = [], []
    for j in range(len(mass)):
        keep = records.origin_level == j
        if j+1 < len(mass):
            keep &= records.loglikelihood <= records.thresholds[j+1]
        count = int(keep.sum()); counts.append(count)
        if count == 0:
            return dict(status='undefined_empty_shell', shell=j, counts=counts)
        representatives.append(float(logsumexp(records.loglikelihood[keep])-np.log(count)))
    return dict(status='ok', logZ=float(logsumexp(np.log(shell_mass)+representatives)),
                shell_masses=shell_mass.tolist(), shell_counts=counts,
                representative_log_mean_likelihood=representatives, accounted_prior_mass=float(shell_mass.sum()))


def correlation_diagnostics(records, values, block_size=64):
    """Approximate ESS from time blocks and between-walker variation.

    Population-wide time blocks preserve contemporaneous exchange dependence.
    This is a rough diagnostic, not an independence proof or confidence interval.
    """
    records.validate()
    values = np.asarray(values)
    if values.shape != records.loglikelihood.shape or not np.isfinite(values).all():
        raise ValueError('Finite scalar diagnostic per sample required')
    rows = []
    for j in range(len(records.thresholds)):
        keep = records.origin_level == j
        times, walkers = records.draw[keep], records.walker[keep]
        nt, nw = int(times.max())+1, int(walkers.max())+1
        if nt*nw != int(keep.sum()) or nt < 4*block_size or nt % block_size:
            raise ValueError('Complete rectangular trajectory with at least four blocks required')
        x = np.empty((nt, nw)); x[times, walkers] = values[keep]
        marginal_variance = float(np.var(x, ddof=1))
        blocks = x.reshape(-1, block_size, nw).mean(axis=(1, 2))
        block_se2 = float(np.var(blocks, ddof=1)/len(blocks))
        walker_se2 = float(np.var(x.mean(0), ddof=1)/nw) if nw>1 else 0.
        se2 = max(marginal_variance/x.size, block_se2, walker_se2)
        ess = min(x.size, marginal_variance/se2) if se2>0 else float(x.size)
        rows.append(dict(level=j, raw_count=x.size, approximate_ess=ess,
            block_means_se=float(np.sqrt(block_se2)), between_walker_se=float(np.sqrt(walker_se2)),
            conservative_mean_se=float(np.sqrt(se2))))
    return dict(per_level=rows, sum_approximate_ess=float(sum(r['approximate_ess'] for r in rows)),
                interpretation='Observable-specific block/ensemble ESS; not importance weight ESS or calibrated-mass uncertainty')


def save_records(path, records):
    records.validate()
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path/'samples.npz', **{name: getattr(records, name) for name in
        ('thresholds', 'log_masses', 'position', 'logprior', 'loglikelihood', 'origin_level', 'walker', 'draw', 'event_id')})
    payload = dict(schema='dns-reconstruction-v1', metadata=records.metadata,
                   samples_sha256=hashlib.sha256((path/'samples.npz').read_bytes()).hexdigest())
    data = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    (path/'records.json').write_text(json.dumps(dict(payload=payload, sha256=hashlib.sha256(data).hexdigest()), indent=2))


def load_records(path):
    path = Path(path)
    envelope = json.loads((path/'records.json').read_text()); payload = envelope['payload']
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if envelope['sha256'] != expected or payload['schema'] != 'dns-reconstruction-v1':
        raise ValueError('Reconstruction metadata checksum/schema mismatch')
    if hashlib.sha256((path/'samples.npz').read_bytes()).hexdigest() != payload['samples_sha256']:
        raise ValueError('Reconstruction sample checksum mismatch')
    with np.load(path/'samples.npz', allow_pickle=False) as f:
        records = ReconstructionRecords(**{k:f[k].copy() for k in f.files}, metadata=payload['metadata'])
    return records.validate()


def validate_caches(records, evaluator, contract):
    """Generic model callback audit; never imports or selects a model itself."""
    from blackjax.ns.dns import DNSParticleState
    records.validate()
    maxima = [0., 0.]
    for j, threshold in enumerate(records.thresholds):
        keep = records.origin_level == j
        state = DNSParticleState(records.position[keep], records.logprior[keep], records.loglikelihood[keep])
        recomputed = evaluator(state.position)
        contract.validate(state, float(threshold), recomputed)
        for k, (actual, cached) in enumerate(zip(recomputed, state[1:])):
            maxima[k] = max(maxima[k], float(np.max(abs(actual-cached))))
    return dict(max_prior_error=maxima[0], max_likelihood_error=maxima[1], passed=True)
