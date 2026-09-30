"""Host-side automatic DNS construction with independent banks and durable gates.

This module constructs levels only. It does not implement inference, evidence,
production adaptation, or a scientific termination rule. Histories are time-major.
Checkpoints are committed iteration boundaries; interrupted attempts are not retried.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from blackjax.ns.dns import DNSParticleState
from blackjax.ns import dns_levels

BANKS = ("selection", "calibration")
ARRAYS = ("position", "logdensity", "loglikelihood")
SCHEMA = "dns-automatic-v1"


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def encoded(value):
    return json.dumps(json_value(value), sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    # Exclusive writes make stale-state reuse and accidental overwrite visible.
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def particles_copy(particles, *, readonly=False):
    arrays = [np.array(v, copy=True) for v in particles]
    if readonly:
        for value in arrays:
            value.setflags(write=False)
    return DNSParticleState(*arrays)


def particle_hash(particles):
    return digest(b"".join(encoded((v.dtype.str, v.shape)) + v.tobytes()
                           for v in map(np.asarray, particles)))


@dataclass(frozen=True)
class NumericalContract:
    """Frozen CPU/x64 contract; allows -inf only as the initial contour."""
    version: int = 1
    backend: str = "cpu"
    jax_enable_x64: bool = True
    dtype: str = "float64"
    rtol: float = 0.0
    prior_atol: float = 1e-9
    likelihood_atol: float = 1e-7

    def __post_init__(self):
        if tuple(asdict(self).values()) != (1, "cpu", True, "float64", 0., 1e-9, 1e-7):
            raise ValueError("The CPU/x64 numerical contract is frozen")

    def environment(self):
        if jax.default_backend() != self.backend or not jax.config.x64_enabled:
            raise ValueError("CPU/x64 execution required")
        return dict(backend=jax.default_backend(), x64=bool(jax.config.x64_enabled),
                    jax=jax.__version__, jaxlib=__import__('jaxlib').__version__,
                    numpy=np.__version__, prng="threefry2x32", schedule="PCG64")

    def validate(self, particles, threshold=None, recomputed=None):
        self.environment()
        x, lp, ll = map(np.asarray, particles)
        if x.shape[:-1] != lp.shape or lp.shape != ll.shape or not lp.size:
            raise ValueError("Expected coordinates (..., dimension) and scalar caches (...)")
        if any(v.dtype != np.dtype(self.dtype) or not np.isfinite(v).all()
               for v in (x, lp, ll)):
            raise ValueError("Finite float64 coordinates and caches required")
        if threshold is not None:
            if not (np.isfinite(threshold) or np.isneginf(threshold)):
                raise ValueError("Invalid contour")
            if np.any(ll <= threshold):
                raise ValueError("Strict cached contour violation")
        if recomputed is not None:
            ap, al = map(np.asarray, recomputed)
            if any(v.shape != lp.shape or v.dtype != np.float64 or not np.isfinite(v).all()
                   for v in (ap, al)):
                raise ValueError("Finite float64 recomputation with matching shape required")
            if threshold is not None and np.any(al <= threshold):
                raise ValueError("Strict recomputed contour violation")
            if not np.allclose(ap, lp, rtol=self.rtol, atol=self.prior_atol):
                raise ValueError("Prior cache exceeds frozen tolerance")
            if not np.allclose(al, ll, rtol=self.rtol, atol=self.likelihood_atol):
                raise ValueError("Likelihood cache exceeds frozen tolerance")


@dataclass(frozen=True)
class LadderConfig:
    burn_in: int = 128
    retained: int = 256
    block_size: int = 32
    target_compression: float = float(np.exp(-1))
    min_ess: float = 20.0

    def __post_init__(self):
        for name in ("burn_in", "retained", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "burn_in" else 1):
                raise ValueError("Invalid fixed sampling budget")
        if self.retained < 4*self.block_size or self.retained % self.block_size:
            raise ValueError("Require at least four complete retained blocks")
        if not 0 < self.target_compression < 1 or self.min_ess != 20:
            raise ValueError("Invalid compression or changed frozen ESS=20 gate")


def streams(seed, iteration, bank, walkers):
    """Versioned, counter-based stream derivation; no implicit/global RNG state."""
    bank_id = BANKS.index(bank)
    # A SeedSequence namespace distinguishes bank, iteration, and purpose.
    root = np.random.SeedSequence([seed, iteration, bank_id])
    promotion, slices, labels, mh = root.spawn(4)
    return dict(bank=bank, iteration=iteration, recipe="SeedSequence-v1",
                promotion_seed=int(promotion.generate_state(1)[0]),
                slice_seeds=slices.generate_state(walkers).astype(np.uint32).tolist(),
                label_seed=int(labels.generate_state(1)[0]),
                mh_seed=int(mh.generate_state(1)[0]))


def eligible_pool(history, threshold, contract):
    contract.validate(history)
    if history.loglikelihood.ndim != 2:
        raise ValueError("History must have (retained time, walker) axes")
    if not (np.isfinite(threshold) or np.isneginf(threshold)):
        raise ValueError("Invalid current contour")
    return np.flatnonzero(np.asarray(history.loglikelihood).ravel() > threshold)


def promote(history, threshold, seed, contract=NumericalContract()):
    """Exactly Stage-3 pooled-with-replacement promotion; bank-local only."""
    pool = eligible_pool(history, threshold, contract)
    if not pool.size:
        raise ValueError("Entire bank has no eligible states")
    walkers = history.loglikelihood.shape[1]
    key = jax.random.key(seed, impl="threefry2x32")
    index = np.asarray(jax.random.choice(key, pool, shape=(walkers,), replace=True))
    starts = particles_copy(dns_levels._survivors(key, history, threshold, walkers))
    source = DNSParticleState(*[np.asarray(v).reshape((-1,)+v.shape[2:])[index]
                                for v in history])
    for actual, expected, old in zip(starts, source, history):
        if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            raise ValueError("Promotion must copy coordinates and scalar caches byte-exactly")
        if np.shares_memory(actual, old):
            raise ValueError("Promotion must own its mutable storage")
    contract.validate(starts, threshold)
    hashes = [particle_hash(DNSParticleState(*[v[i] for v in starts])) for i in range(walkers)]
    position_unique = len({v.tobytes() for v in starts.position})
    return starts, json_value(dict(total_retained=history.loglikelihood.size,
        eligible=int(pool.size), eligible_fraction=pool.size/history.loglikelihood.size,
        eligible_per_donor=np.bincount(pool % walkers, minlength=walkers),
        donor_ids=index % walkers, retained_indices=index // walkers,
        selected_flat_indices=index, selected_pool_indices=np.searchsorted(pool, index),
        donor_multiplicities=np.bincount(index % walkers, minlength=walkers),
        unique_promoted_states=position_unique, duplicates=walkers-position_unique,
        source_hashes=hashes, source_history_hash=particle_hash(history),
        duplicate_source_hashes={h: hashes.count(h) for h in hashes if hashes.count(h)>1}))


@dataclass(frozen=True)
class Candidate:
    """Immutable serialized selection record, hashed before calibration begins."""
    record: bytes

    @property
    def diagnostics(self):
        return json.loads(self.record)

    @property
    def sha256(self):
        return digest(self.record)

    @property
    def threshold(self):
        return self.diagnostics.get("threshold")


def select_candidate(history, current_threshold, config):
    selected = dns_levels.build_next_level(history.loglikelihood, current_threshold,
        target_compression=config.target_compression, block_size=config.block_size,
        min_ess=config.min_ess)
    return Candidate(encoded(selected))


def calibrate(candidate, history, block_size):
    """Only this bank estimates compression. Candidate cannot be reselected."""
    result = dns_levels.tail_diagnostics(history.loglikelihood, candidate.threshold, block_size)
    indicator = (history.loglikelihood > candidate.threshold).astype(float)
    blocks = indicator.reshape(-1, block_size, indicator.shape[1]).mean(1)
    result["block_means_se"] = float(np.sqrt(np.var(blocks, ddof=1)/blocks.size))
    result["between_walker_se"] = (float(np.std(indicator.mean(0), ddof=1)/np.sqrt(indicator.shape[1]))
                                    if indicator.shape[1] > 1 else 0.)
    return json_value(result)


def freeze_candidate(thresholds, log_masses, candidate, calibration, min_ess=20):
    """Pure gate: append exactly one calibrated level, or append nothing."""
    if min_ess != 20:
        raise ValueError("Frozen calibration ESS gate is 20")
    t, m = tuple(thresholds), tuple(log_masses)
    validate_prefix(t, m)
    accepted = (candidate.diagnostics["status"] == "ok"
                and 0 < calibration["ratio"] < 1 and calibration["ess"] >= min_ess
                and np.isfinite(calibration["ess"]))
    if not accepted:
        return t, m, False
    threshold = candidate.threshold
    if not np.isfinite(threshold) or threshold <= t[-1]:
        raise ValueError("Candidate must strictly increase the frozen prefix")
    return t+(threshold,), m+(float(m[-1]+np.log(calibration["ratio"])),), True


def validate_prefix(thresholds, log_masses):
    t, m = np.asarray(thresholds), np.asarray(log_masses)
    if (t.ndim != 1 or not len(t) or m.shape != t.shape or not np.isneginf(t[0])
            or m[0] != 0 or not np.isfinite(t[1:]).all() or not np.isfinite(m).all()
            or not np.all(np.diff(t)>0) or not np.all(np.diff(m)<0)):
        raise ValueError("Invalid frozen ladder prefix")


@dataclass(frozen=True)
class LadderState:
    thresholds: tuple
    log_masses: tuple
    selection: DNSParticleState
    calibration: DNSParticleState
    seed: int
    kernel_metadata: bytes
    config: LadderConfig = LadderConfig()
    contract: NumericalContract = NumericalContract()
    iteration: int = 0
    status: str = "ready"
    records: tuple = ()

    def __post_init__(self):
        validate_prefix(self.thresholds, self.log_masses)
        if not isinstance(self.seed, int) or self.seed < 0 or self.iteration < 0:
            raise ValueError("Nonnegative seed and iteration required")
        if self.status not in ("ready", "stopped"):
            raise ValueError("Unknown ladder state")
        for name in BANKS:
            history = particles_copy(getattr(self, name), readonly=True)
            self.contract.validate(history)
            if history.loglikelihood.ndim != 2:
                raise ValueError("Histories must be time-major")
            object.__setattr__(self, name, history)
        if self.selection.position.shape[1:] != self.calibration.position.shape[1:]:
            raise ValueError("Independent banks require matching population dimensions")


def prepare_iteration(state):
    """Promotion-only dry run: no candidate, sampling, or checkpoint creation."""
    if state.status != "ready":
        raise ValueError("Stopped construction cannot be retried")
    walkers = state.selection.loglikelihood.shape[1]
    schedules = {name: streams(state.seed, state.iteration, name, walkers) for name in BANKS}
    seeds = [v for s in schedules.values() for v in
             [s['promotion_seed'], *s['slice_seeds'], s['label_seed'], s['mh_seed']]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("RNG stream collision; stop, never change seed")
    starts, metadata = {}, {}
    for name in BANKS:
        starts[name], metadata[name] = promote(getattr(state, name), state.thresholds[-1],
                                              schedules[name]['promotion_seed'], state.contract)
    if any(np.shares_memory(a, b) for a in starts['selection'] for b in starts['calibration']):
        raise ValueError("Cross-bank storage alias")
    return starts, metadata, schedules


def build_next_level(state, kernel, output):
    """One durable attempt: promote both, select, persist candidate, calibrate, gate.

    kernel.run(starts, contour, streams, config, contract) returns a time-major
    DNSParticleState plus diagnostics. kernel.evaluate handles (..., dimension).
    Kernel code/settings and the numerical environment are pinned by metadata.
    """
    if state.status != "ready":
        raise ValueError("Stopped construction cannot be retried")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)  # stale state cannot reuse an attempt
    row = dict(iteration=state.iteration, accepted=False, old_threshold=state.thresholds[-1])
    try:
        if encoded(kernel.metadata) != state.kernel_metadata:
            raise ValueError("Frozen kernel identity changed")
        schedules = {n: streams(state.seed, state.iteration, n, state.selection.loglikelihood.shape[1]) for n in BANKS}
        write_json(folder/'preflight.json', dict(streams=schedules, contract=asdict(state.contract),
            environment=state.contract.environment(), kernel=kernel.metadata,
            thresholds=state.thresholds, log_masses=state.log_masses,
            history_hashes={n: particle_hash(getattr(state, n)) for n in BANKS}, config=asdict(state.config)))
        starts, promotion, schedules = prepare_iteration(state)
        row.update(promotion=promotion, streams=schedules)
        write_json(folder/'promotion.json', promotion)
        for name in BANKS:
            state.contract.validate(starts[name], state.thresholds[-1], kernel.evaluate(starts[name].position))
        histories = {}
        for name in BANKS:
            # Calibration is called only after candidate.json exists and is immutable.
            result, diagnostics = kernel.run(starts[name], state.thresholds[-1], schedules[name], state.config, state.contract)
            history = particles_copy(result, readonly=True)
            expected = (state.config.retained, state.selection.loglikelihood.shape[1])
            if history.loglikelihood.shape != expected or history.position.shape[2:] != state.selection.position.shape[2:]:
                raise ValueError("Kernel did not complete the exact fixed budget")
            state.contract.validate(history, state.thresholds[-1], kernel.evaluate(history.position))
            histories[name] = history
            row[name+'_sampling'] = diagnostics
            if name == "selection":
                candidate = select_candidate(history, state.thresholds[-1], state.config)
                with (folder/'candidate.json').open('xb') as stream:
                    stream.write(candidate.record)
                write_json(folder/'candidate_hash.json', dict(sha256=candidate.sha256))
                row.update(candidate=candidate.diagnostics, candidate_sha256=candidate.sha256)
                if candidate.diagnostics['status'] != 'ok':
                    row['stop_reason'] = 'selection_gate'
                    break
            else:
                if (folder/'candidate.json').read_bytes() != candidate.record:
                    raise ValueError("Candidate changed after calibration began")
                if json.loads((folder/'candidate_hash.json').read_bytes())['sha256'] != candidate.sha256:
                    raise ValueError("Candidate hash changed")
                calibration = calibrate(candidate, history, state.config.block_size)
                row['calibration'] = calibration
                t, m, accepted = freeze_candidate(state.thresholds, state.log_masses,
                                                   candidate, calibration, state.config.min_ess)
                row['accepted'] = accepted
                if accepted:
                    new = replace(state, thresholds=t, log_masses=m, **histories,
                                  iteration=state.iteration+1, records=state.records+(encoded(row),))
                    write_json(folder/'decision.json', row)
                    save_checkpoint(folder/'checkpoint', new)
                    return new
                row['stop_reason'] = 'calibration_gate'
    except Exception as exc:
        row.update(accepted=False, stop_reason='integrity_failure', error=f'{type(exc).__name__}: {exc}')
    stopped = replace(state, iteration=state.iteration+1, status='stopped',
                      records=state.records+(encoded(row),))
    write_json(folder/'stop.json', row)
    return stopped


def run_ladder(state, kernel, output, *, max_new_levels):
    """Bounded engineering loop; exhaustion of this budget is not DNS termination."""
    if not isinstance(max_new_levels, int) or isinstance(max_new_levels, bool) or max_new_levels < 0:
        raise ValueError("Explicit nonnegative construction budget required")
    for _ in range(max_new_levels):
        if state.status == 'stopped':
            break
        state = build_next_level(state, kernel, Path(output)/f'attempt_{state.iteration:06d}')
    return state


def save_checkpoint(path, state):
    """Portable immutable bundle at a completed iteration boundary; no pickle."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path/'banks.npz', **{n+'_'+k: v for n in BANKS
        for k, v in zip(ARRAYS, getattr(state, n))})
    payload = dict(schema=SCHEMA, thresholds=state.thresholds, log_masses=state.log_masses,
        bank_history_reference='banks.npz', bank_history_sha256=digest((path/'banks.npz').read_bytes()),
        seed=state.seed, next_iteration=state.iteration, stream_recipe='SeedSequence-v1',
        config=asdict(state.config), numerical_contract=asdict(state.contract),
        kernel_metadata=json.loads(state.kernel_metadata), environment=state.contract.environment(),
        status=state.status, records=[json.loads(r) for r in state.records])
    write_json(path/'checkpoint.json', dict(payload=payload, sha256=digest(encoded(payload))))


def load_checkpoint(path, kernel):
    path = Path(path)
    envelope = json.loads((path/'checkpoint.json').read_bytes())
    payload = envelope['payload']
    if envelope['sha256'] != digest(encoded(payload)) or payload['schema'] != SCHEMA:
        raise ValueError("Checkpoint checksum or schema mismatch")
    contract = NumericalContract(**payload['numerical_contract'])
    if payload['environment'] != contract.environment() or encoded(payload['kernel_metadata']) != encoded(kernel.metadata):
        raise ValueError("Restart must preserve environment and frozen kernel")
    if payload['stream_recipe'] != 'SeedSequence-v1' or payload['bank_history_reference'] != 'banks.npz':
        raise ValueError("Unknown stream recipe or history reference")
    if digest((path/'banks.npz').read_bytes()) != payload['bank_history_sha256']:
        raise ValueError("Checkpoint history checksum mismatch")
    with np.load(path/'banks.npz', allow_pickle=False) as arrays:
        banks = {n: DNSParticleState(*[arrays[n+'_'+k].copy() for k in ARRAYS]) for n in BANKS}
    return LadderState(tuple(float(v) for v in payload['thresholds']), tuple(payload['log_masses']),
        **banks, seed=payload['seed'], kernel_metadata=encoded(kernel.metadata),
        config=LadderConfig(**payload['config']), contract=contract,
        iteration=payload['next_iteration'], status=payload['status'],
        records=tuple(encoded(r) for r in payload['records']))
