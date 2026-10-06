"""Terminal-cap resolution criterion for fixed-contour DNS reconstruction.

This does not bound finite-sample reconstruction error. Calibrated mass
intervals are descriptive unless the caller supplies certified envelopes.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class TerminationConfig:
    remaining_fraction: float = 0.05
    upper_bound_justification: str = ''

    def __post_init__(self):
        if not np.isfinite(self.remaining_fraction) or not 0 < self.remaining_fraction < 1:
            raise ValueError('Finite stopping fraction in (0, 1) required')
        if not self.upper_bound_justification.strip():
            raise ValueError('A global likelihood upper-bound justification is required')


def remaining_evidence_fraction(remainder, evidence_lower):
    """Validate a nonnegative remainder estimate and normalize its scale."""
    if not np.isfinite(remainder) or remainder < 0:
        raise ValueError('Nonfinite stopping metric or negative remainder estimate')
    if not np.isfinite(evidence_lower) or evidence_lower <= 0:
        raise ValueError('Positive finite evidence lower estimate required')
    fraction = float(remainder / evidence_lower)
    if not np.isfinite(fraction):
        raise ValueError('Nonfinite stopping metric')
    return fraction


def stopping_decision(records, log_likelihood_upper, config, *,
                      log_mass_lower=None, log_mass_upper=None):
    """Bound terminal cap relative to a layer-cake evidence lower estimate.

    L_upper must bound likelihood everywhere, not merely observed samples.
    True mass envelopes give a mathematical cap bound; plug-in masses or
    descriptive envelopes give only a conservative candidate diagnostic.
    """
    records.validate()
    t = np.asarray(records.thresholds, dtype=float)
    m = np.asarray(records.log_masses, dtype=float)
    lo = m if log_mass_lower is None else np.asarray(log_mass_lower, dtype=float)
    hi = m if log_mass_upper is None else np.asarray(log_mass_upper, dtype=float)
    for v in (lo, hi):
        if (v.shape != m.shape or not np.isfinite(v).all() or v[0] != 0
                or np.any(v > 0) or np.any(np.diff(v) >= 0)):
            raise ValueError('Invalid remaining prior mass envelope')
    if np.any(lo > m) or np.any(hi < m):
        raise ValueError('Mass envelopes must enclose calibrated masses')
    if not np.isfinite(log_likelihood_upper):
        raise ValueError('No finite global likelihood upper bound')
    if log_likelihood_upper < max(float(t[-1]), float(records.loglikelihood.max())):
        raise ValueError('Likelihood upper bound violated')
    level = len(t)-1
    if level == 0:
        return dict(stop=False, level=0, reason='no_positive_evidence_lower_bound',
                    remaining_fraction=None, delta_logZ=None)
    # Z >= sum_i X_i (L_i - L_{i-1}); L_0=0.
    increments = t[1:] + np.log(-np.expm1(t[:-1]-t[1:]))
    log_lower = float(logsumexp(lo[1:]+increments))
    log_cap = float(hi[-1]+log_likelihood_upper)
    # Normalize in log space first to avoid underflow of both evidence scales.
    fraction = remaining_evidence_fraction(float(np.exp(log_cap-log_lower)), 1.)
    return dict(stop=bool(fraction <= config.remaining_fraction), level=level,
                threshold=float(t[-1]), X_stop=float(np.exp(m[-1])),
                X_upper=float(np.exp(hi[-1])), log_remaining_evidence=log_cap,
                remaining_evidence=float(np.exp(log_cap)), log_evidence_lower=log_lower,
                remaining_fraction=fraction, delta_logZ=float(np.log1p(fraction)),
                upper_bound_justification=config.upper_bound_justification,
                interpretation='Terminal-cap bound; excludes Monte Carlo and mass estimation error')
