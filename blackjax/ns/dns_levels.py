"""Experimental host-side DNS level construction; never adapts production.

For samples from pi(theta | logL > ell_j), choose the upper exp(-1) tail.
Threshold-selection samples NEVER estimate the reported calibrated mass.
Independent random streams and separate walker banks are used for selection
and calibration. Finite MCMC burn-in is not a guarantee of equilibration.
"""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm, t

from blackjax.ns import dns


@dataclass(frozen=True)
class ConstructionConfig:
    num_levels: int = 5
    construction_draws: int = 1024
    calibration_draws: int = 2048
    burn_in: int = 256
    thinning: int = 2
    block_size: int = 64
    min_ess: float = 100.0
    target_compression: float = float(np.exp(-1))
    max_parameter_steps: int = 10_000_000
    terminal_loglikelihood: float | None = None

    def __post_init__(self):
        for name in ("num_levels", "construction_draws", "calibration_draws",
                     "thinning", "block_size", "max_parameter_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.burn_in, bool) or not isinstance(self.burn_in, int) or self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if not 0 < self.target_compression < 1 or not self.min_ess > 0:
            raise ValueError("Invalid target compression or min_ess")
        if min(self.construction_draws, self.calibration_draws) < 4 * self.block_size:
            raise ValueError("At least four complete blocks per walker are required")
        if self.terminal_loglikelihood is not None and not np.isfinite(self.terminal_loglikelihood):
            raise ValueError("terminal_loglikelihood must be finite")


def tail_diagnostics(loglikelihood, threshold, block_size=64):
    """Exceedance estimate and explicitly approximate correlation diagnostics.

    Input has axes (retained time, independent walker). Batch means are formed
    within each walker, never across concatenated walker boundaries. Use the
    maximum of IID, batch-mean, and between-walker standard errors. ESS is the
    Bernoulli-variance equivalent, NOT proof that all modes were explored.
    The Student interval is approximate, not an exact binomial interval.
    """
    values = np.asarray(loglikelihood)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite (time, walker) likelihoods")
    if block_size < 1 or values.shape[0] < 4 * block_size:
        raise ValueError("Insufficient complete blocks")
    indicator = (values > threshold).astype(float)
    n = indicator.size
    p = float(indicator.mean())
    blocks = values.shape[0] // block_size
    means = indicator[:blocks * block_size].reshape(blocks, block_size, -1).mean(1)
    iid_se = np.sqrt(p * (1 - p) / n)
    batch_se = np.sqrt(np.var(means, ddof=1) / means.size)
    walkers = values.shape[1]
    walker_se = (np.std(indicator.mean(0), ddof=1) / np.sqrt(walkers)
                 if walkers > 1 else 0.0)
    se = float(max(iid_se, batch_se, walker_se))
    ess = float(min(n, p * (1 - p) / se**2)) if se > 0 else 0.0
    # Conservative degrees of freedom when between-walker variation dominates.
    df = walkers - 1 if walker_se >= max(iid_se, batch_se) and walkers > 1 else means.size - 1
    half = float(t.ppf(0.975, df) * se)
    z = norm.ppf(0.975)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(ratio=p, sample_count=n, exceeds=int(indicator.sum()),
                naive_iid_se=float(iid_se), naive_iid_wilson=[center-radius, center+radius],
                standard_error=se, interval=[max(0.0, p-half), min(1.0, p+half)],
                ess=ess, blocks=int(means.size), block_size=block_size,
                walker_fractions=indicator.mean(0).tolist())


def build_next_level(loglikelihood, current_threshold, *, target_compression=np.exp(-1),
                     block_size=64, min_ess=100):
    """Empirical strict-tail convention, with no interpolation or jitter.

    Sort N observations ascending. Let k=floor(N*(1-rho)); choose x[k]
    (zero-based). Ties are all excluded by strict >; report the actual tail.
    Reject constant, non-increasing, zero-tail, or insufficient-ESS proposals.
    This selection fraction is NOT an independent mass calibration.
    """
    values = np.asarray(loglikelihood)
    if not 0 < target_compression < 1:
        raise ValueError("target_compression must be in (0,1)")
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite (time, walker) likelihoods")
    if np.isfinite(current_threshold) and np.any(values <= current_threshold):
        raise ValueError("History contains points outside its generating contour")
    ordered = np.sort(values.ravel())
    if ordered.size == 0:
        return dict(status="insufficient_samples", sample_count=ordered.size)
    index = min(ordered.size-1, int(np.floor(ordered.size*(1-target_compression))))
    threshold = float(ordered[index])
    if threshold <= current_threshold or ordered[0] == ordered[-1]:
        return dict(status="non_increasing_or_constant", threshold=threshold, sample_count=ordered.size)
    diagnostic = tail_diagnostics(values, threshold, block_size)
    status = "ok" if 0 < diagnostic["ratio"] < 1 and diagnostic["ess"] >= min_ess else "insufficient_tail_information"
    return dict(status=status, threshold=threshold, order_index=index, **diagnostic)


def _make_sampler(parameter_step, draws, config):
    """One compiled fixed-contour sampling block; threshold is a dynamic scalar."""
    advance = jax.vmap(parameter_step, in_axes=(0, 0, None))

    def run(key, particles, threshold):
        walkers = particles.loglikelihood.shape[0]

        def step(carry, _):
            key, particles, cost, failures = carry
            key, subkey = jax.random.split(key)
            particles, info = advance(jax.random.split(subkey, walkers), particles, threshold)
            valid = (jnp.isfinite(particles.logdensity) & jnp.isfinite(particles.loglikelihood)
                     & (jnp.isneginf(threshold) | (particles.loglikelihood > threshold)))
            cost += jnp.sum(info.num_steps + info.num_shrink)
            failures += jnp.sum(~valid)
            return (key, particles, cost, failures), None

        carry = (key, particles, jnp.array(0), jnp.array(0))
        carry, _ = jax.lax.scan(step, carry, None, length=config.burn_in)

        def retain(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=config.thinning)
            return carry, carry[1]

        carry, history = jax.lax.scan(retain, carry, None, length=draws)
        return history, carry[2], carry[3]

    return jax.jit(run)


def _initial_particles(positions, logprior_fn, loglikelihood_fn):
    particles = dns.DNSParticleState(positions, jax.vmap(logprior_fn)(positions),
                                     jax.vmap(loglikelihood_fn)(positions))
    if (particles.logdensity.ndim != 1 or particles.logdensity.size == 0
            or particles.loglikelihood.shape != particles.logdensity.shape):
        raise ValueError("Prior and likelihood must return scalar values for each walker")
    if not np.all(np.isfinite(np.asarray(particles.logdensity))):
        raise ValueError("Initial particles must have finite log prior")
    return particles


def _survivors(key, history, threshold, walkers):
    flattened = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), history)
    eligible = np.flatnonzero(np.asarray(flattened.loglikelihood) > threshold)
    if eligible.size == 0:
        raise ValueError("No valid starts in this bank for the next contour")
    indices = jax.random.choice(key, jnp.asarray(eligible), shape=(walkers,), replace=True)
    return jax.tree.map(lambda x: x[indices], flattened)


def construct_levels(seed, construction_positions, calibration_positions, logprior_fn,
                     loglikelihood_fn, parameter_step, *, config=ConstructionConfig(),
                     kernel_settings=None):
    """Sequential fixed-contour construction with separate calibration walkers.

    Both starting banks should be independently initialized from the prior.
    Promotion resamples each bank's own eligible states, followed by fresh burn-in.
    Shared genealogy and incomplete within-contour mixing can bias diagnostics;
    use independent complete builds and external reference tests as well.
    Stops on the FIRST uninformative proposal; no unbounded retry heuristic.
    """
    banks = [_initial_particles(p, logprior_fn, loglikelihood_fn)
             for p in (construction_positions, calibration_positions)]
    walkers = banks[0].loglikelihood.size
    if walkers < 1 or banks[1].loglikelihood.shape != banks[0].loglikelihood.shape:
        raise ValueError("Banks must have matching nonempty walker axes")
    if any(not np.all(np.isfinite(np.asarray(b.logdensity))) for b in banks):
        raise ValueError("Initial positions must be in prior support")
    samplers = [_make_sampler(parameter_step, n, config)
                for n in (config.construction_draws, config.calibration_draws)]
    key = jax.random.key(seed)
    thresholds, log_mass, rows = [-np.inf], [0.0], []
    cost = steps = 0
    status = "max_levels"
    for j in range(config.num_levels-1):
        required = walkers * (2*config.burn_in + config.thinning *
                              (config.construction_draws + config.calibration_draws))
        if steps + required > config.max_parameter_steps:
            status = "max_parameter_steps"
            break
        histories = []
        for bank, sampler in zip(banks, samplers):
            key, subkey = jax.random.split(key)
            history, evaluations, failures = sampler(subkey, bank, jnp.asarray(thresholds[-1]))
            if int(failures):
                raise ValueError("Parameter kernel violated the constrained-prior contract")
            histories.append(history)
            cost += int(evaluations)
        steps += required
        proposal = build_next_level(histories[0].loglikelihood, thresholds[-1],
                                    target_compression=config.target_compression,
                                    block_size=config.block_size, min_ess=config.min_ess)
        if proposal["status"] != "ok":
            status = proposal["status"]
            rows.append(dict(level=j+1, selection=proposal, accepted=False))
            break
        threshold = proposal["threshold"]
        calibration = tail_diagnostics(histories[1].loglikelihood, threshold, config.block_size)
        accepted = 0 < calibration["ratio"] < 1 and calibration["ess"] >= config.min_ess
        rows.append(dict(level=j+1, threshold=threshold, selection=proposal,
                         calibration=calibration, accepted=accepted))
        if not accepted:
            status = "insufficient_calibration"
            break
        thresholds.append(threshold)
        log_mass.append(log_mass[-1] + np.log(calibration["ratio"]))
        if config.terminal_loglikelihood is not None and threshold >= config.terminal_loglikelihood:
            status = "terminal_loglikelihood"
            break
        banks = []
        for history in histories:
            key, subkey = jax.random.split(key)
            banks.append(_survivors(subkey, history, threshold, walkers))
    levels = dns.create_levels(thresholds, log_mass, np.zeros(len(thresholds)))
    return levels, dict(seed=int(seed), config=asdict(config), num_walkers=walkers,
                        kernel_settings=kernel_settings, rows=rows, status=status,
                        parameter_steps=steps, likelihood_evaluations=cost,
                        initial_likelihood_evaluations=2*walkers,
                        dtype=str(levels.log_mass.dtype),
                        construction_positions=jax.tree.map(lambda x: np.asarray(x).tolist(), construction_positions),
                        calibration_positions=jax.tree.map(lambda x: np.asarray(x).tolist(), calibration_positions))


def calibrate_level_masses(seed, levels, initial_positions, logprior_fn,
                           loglikelihood_fn, parameter_step, *, config=ConstructionConfig()):
    """Independent refinement for fixed thresholds; never uses their old masses.

    At each level sample its constrained prior, estimate next-level exceedance,
    then promote eligible starts. Insufficient calibration raises, rather than
    silently returning a partial mass table for a longer threshold ladder.
    """
    particles = _initial_particles(initial_positions, logprior_fn, loglikelihood_fn)
    walkers = particles.loglikelihood.size
    required = (len(levels.log_mass)-1)*walkers*(config.burn_in+config.thinning*config.calibration_draws)
    if required > config.max_parameter_steps:
        raise ValueError("Calibration exceeds max_parameter_steps")
    sample = _make_sampler(parameter_step, config.calibration_draws, config)
    key = jax.random.key(seed)
    logs, rows, cost = [0.0], [], 0
    for j in range(len(levels.log_mass)-1):
        key, subkey = jax.random.split(key)
        history, evaluations, failures = sample(subkey, particles, levels.loglikelihood[j])
        if int(failures):
            raise ValueError("Invalid constrained state")
        row = tail_diagnostics(history.loglikelihood, float(levels.loglikelihood[j+1]), config.block_size)
        if not 0 < row["ratio"] < 1 or row["ess"] < config.min_ess:
            raise ValueError("Insufficient calibration information")
        rows.append(row)
        logs.append(logs[-1] + np.log(row["ratio"]))
        cost += int(evaluations)
        key, subkey = jax.random.split(key)
        particles = _survivors(subkey, history, levels.loglikelihood[j+1], walkers)
    return dns.create_levels(levels.loglikelihood, logs, levels.log_weight), dict(
        seed=int(seed), rows=rows, likelihood_evaluations=cost, parameter_steps=required,
        config=asdict(config), num_walkers=walkers, initial_likelihood_evaluations=walkers,
        initial_positions=jax.tree.map(lambda x: np.asarray(x).tolist(), initial_positions))
