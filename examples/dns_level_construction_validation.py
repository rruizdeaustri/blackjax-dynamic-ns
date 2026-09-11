"""Reproducible Stage-3 calibration and frozen-production experiment. No evidence."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.ns import dns, dns_diagnostics as diag, dns_kernels, dns_levels


def problem(kind):
    bound = 1.0 if kind == "gaussian" else 6.0

    def prior(x):
        return jnp.where(jnp.abs(x) <= bound, -jnp.log(2*bound), -jnp.inf)

    def likelihood(x):
        return -x*x/2 if kind == "gaussian" else -0.5*((jnp.abs(x)-1)/0.3)**2

    def direction(key, position):
        return jnp.where(jax.random.bernoulli(key), 0.5, -0.5)

    kernel = dns_kernels.build_constrained_slice_kernel(
        prior, likelihood, generate_slice_direction_fn=direction, max_steps=10, max_shrinkage=100)
    return bound, prior, likelihood, kernel


def true_mass(kind, thresholds):
    thresholds = np.asarray(thresholds)
    radius = np.sqrt(-2*thresholds)
    if kind == "gaussian":
        return np.minimum(1, radius)
    radius *= 0.3
    return (np.minimum(6, 1+radius)-np.maximum(0, 1-radius))/6


def reference_thresholds(kind, count):
    masses = np.exp(-np.arange(count))
    radii = masses if kind == "gaussian" else np.where(masses >= 1/3, 6*masses-1, 3*masses)/0.3
    thresholds = -radii*radii/2
    thresholds[0] = -np.inf
    return thresholds


def constrained_cdf(kind, threshold, x):
    if kind == "gaussian":
        r = min(1, np.sqrt(-2*threshold))
        return np.clip((np.asarray(x)+r)/(2*r), 0, 1)
    r = 0.3*np.sqrt(-2*threshold)
    outer, inner = min(6, 1+r), max(0, 1-r)
    x = np.asarray(x)
    return (np.clip(x+outer, 0, outer-inner)+np.clip(x-inner, 0, outer-inner))/(2*(outer-inner))


def build(kind, seed, walkers=8, config=None):
    bound, prior, likelihood, kernel = problem(kind)
    config = config or dns_levels.ConstructionConfig()
    keys = jax.random.split(jax.random.key(seed+10000), 2)
    banks = [jax.random.uniform(key, (walkers,), minval=-bound, maxval=bound) for key in keys]
    levels, metadata = dns_levels.construct_levels(
        seed, *banks, prior, likelihood, kernel, config=config,
        kernel_settings=dict(direction_scale=0.5, max_steps=10, max_shrinkage=100))
    metadata["initialization_seed"] = seed+10000
    metadata["kind"] = kind
    masses = true_mass(kind, levels.loglikelihood)
    metadata["thresholds"] = np.asarray(levels.loglikelihood).tolist()
    metadata["reference_thresholds"] = reference_thresholds(kind, len(masses)).tolist()
    metadata["true_mass"] = masses.tolist()
    metadata["estimated_mass"] = np.exp(levels.log_mass).tolist()
    metadata["delta_log_mass"] = (np.asarray(levels.log_mass)-np.log(masses)).tolist()
    for j, row in enumerate(metadata["rows"]):
        if row["accepted"]:
            row["true_compression"] = float(masses[j+1]/masses[j])
    return levels, metadata


def ratio_error(numerator, denominator, block=256):
    """Delta-method batch uncertainty for a conditional expectation/occupancy."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    estimate = numerator.sum()/denominator.sum()
    n = numerator.shape[0]//block
    residual = (numerator-estimate*denominator)[:n*block].reshape(n, block, -1).mean(1)
    batch_se = np.std(residual, ddof=1)/np.sqrt(residual.size)/denominator.mean()
    walker_residual = (numerator-estimate*denominator).mean(0)
    walker_se = np.std(walker_residual, ddof=1)/np.sqrt(len(walker_residual))/denominator.mean()
    return dict(estimate=float(estimate), se=float(max(batch_se, walker_se)))


def ascent_switches(position, assigned, high_level, initial_position, top_level):
    """Stronger mechanism diagnostic: both endpoints must be ASSIGNED high.

    Keep the most recent high-assigned visit in the previous mode. A change
    must therefore include a parameter sign change and a return of the level
    index to high, not merely a high likelihood attained while still at level 0.
    """
    x = np.r_[initial_position, position]
    indices = np.r_[top_level, assigned]
    anchor = 0
    events = []
    for i in range(1, len(x)):
        if indices[i] < high_level:
            continue
        if np.sign(x[i]) != np.sign(x[anchor]):
            crossings = np.flatnonzero(np.sign(x[anchor:i]) != np.sign(x[anchor+1:i+1])) + anchor + 1
            crossing = int(crossings[-1])
            events.append(dict(direction="A_to_B" if x[i] > 0 else "B_to_A",
                start=anchor, crossing=crossing, end=i,
                minimum_level=int(indices[anchor:i+1].min()),
                crossing_generation_level=int(indices[crossing-1]),
                witness_positions=x[[anchor, crossing-1, crossing, i]].tolist(),
                witness_levels=indices[[anchor, crossing-1, crossing, i]].tolist()))
        anchor = i
    return events


def production(kind, seed, levels, walkers=8, sweeps=8000, burn=1024, start_sign=-1):
    _, prior, likelihood, parameter = problem(kind)
    count = len(levels.log_mass)
    high = count-2
    start = float(start_sign) if kind == "double_well" else 0.0
    state = dns.init(jnp.array(start), prior, likelihood, levels, count-1)
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (walkers,)+x.shape), state)
    step = dns.build_kernel(levels, parameter)

    def sweep(state, key):
        new, info = jax.vmap(step)(jax.random.split(key, walkers), state)
        return new, (new.particle.position, new.particle.loglikelihood, new.level_index, info)

    _, trace = jax.jit(lambda s, keys: jax.lax.scan(sweep, s, keys))(
        state, jax.random.split(jax.random.key(seed), sweeps))
    x, ll, assigned, info = jax.tree.map(np.asarray, trace)
    costs = (info.parameter_info.num_steps+info.parameter_info.num_shrink)[..., 0]
    highest = np.asarray(diag.highest_eligible_level(levels, ll))
    rows = []
    masses = true_mass(kind, levels.loglikelihood)
    expected = np.exp(np.asarray(levels.log_weight)-np.asarray(levels.log_mass))*masses
    expected /= expected.sum()
    for j in range(count):
        mask = assigned[burn:] == j
        row = dict(level=j, occupancy=ratio_error(mask, np.ones_like(mask)))
        row["occupancy"]["expected"] = float(expected[j])
        u = constrained_cdf(kind, float(levels.loglikelihood[j]), x[burn:])
        row["cdf"] = [dict(q=q, **ratio_error(mask & (u <= q), mask)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
        row["cdf_uniform_mean"] = ratio_error(mask*u, mask)
        row["cdf_uniform_second"] = ratio_error(mask*u*u, mask)
        threshold = float(levels.loglikelihood[j])
        if kind == "gaussian":
            outer, inner = min(1, np.sqrt(-2*threshold)), 0
        else:
            radius = 0.3*np.sqrt(-2*threshold)
            outer, inner = min(6, 1+radius), max(0, 1-radius)
        row["position_mean"] = dict(expected=0., **ratio_error(mask*x[burn:], mask))
        row["position_second"] = dict(expected=float((outer*outer+outer*inner+inner*inner)/3),
                                       **ratio_error(mask*x[burn:]**2, mask))
        row["modal_fraction"] = ratio_error(mask & (x[burn:] > 0), mask)
        rows.append(row)
    edge = diag.edge_statistics(jax.tree.map(lambda a: a[burn:], info.level_info), count)
    edges = []
    li = jax.tree.map(lambda a: a[burn:, :, 0], info.level_info)
    ratios = masses[1:]/masses[:-1]
    a = np.exp(np.asarray(levels.log_weight)-np.asarray(levels.log_mass))
    for j in range(count-1):
        up = (li.previous_level == j) & (li.proposed_level == j+1)
        down = (li.previous_level == j+1) & (li.proposed_level == j)
        edges.append(dict(level=j,
            eligibility=dict(expected=float(ratios[j]), **ratio_error(up & li.is_eligible, up)),
            up_acceptance=dict(expected=float(min(1, a[j+1]/a[j])), **ratio_error(up & li.is_accepted, up & li.is_eligible)),
            down_acceptance=dict(expected=float(min(1, a[j]/a[j+1])), **ratio_error(down & li.is_accepted, down))))
    switches, trips, first, depths = [0, 0], [0, 0], [], []
    mechanisms = []
    for w in range(walkers):
        # Include the known initial high-mode state when computing first passage.
        summary = diag.mode_switch_summary(
            np.r_[start_sign, np.where(x[:, w]>0, 1, -1)],
            np.r_[count-1, assigned[:, w]], np.r_[count-1, highest[:, w]], high_level=high)
        switches[0] += summary["a_to_b"]
        switches[1] += summary["b_to_a"]
        depths.extend(summary["backtracking_minima"].tolist())
        passage = summary["first_passage_index"]
        first.append(None if passage is None else int(costs[:passage, w].sum()))
        tr = diag.round_trip_counts(np.r_[count-1, assigned[:, w]], high_level=count-1)
        trips[0] += tr["low_high_low"]
        trips[1] += tr["high_low_high"]
        if kind == "double_well":
            mechanisms.extend(dict(walker=w, **event) for event in ascent_switches(
                x[:, w], assigned[:, w], high, start, count-1))
    result = dict(seed=seed, start_sign=start_sign, sweeps=sweeps, burn=burn, walkers=walkers,
        levels=rows, edges=edges, edge_counts={k:v.tolist() for k,v in edge.items()},
        switches=switches, round_trips=trips,
        first_passage_evaluations=first, backtracking_depth_counts={str(j):depths.count(j) for j in set(depths)},
        modal_fraction=ratio_error(x[burn:]>0, np.ones_like(x[burn:])),
        likelihood_evaluations=int(costs.sum()), parameter_failures=int((~info.parameter_is_valid).sum()),
        slice_failures=int((~info.parameter_info.is_accepted).sum()))
    if kind == "double_well":
        result["assigned_high_switches"] = [sum(e["direction"] == direction for e in mechanisms)
                                            for direction in ("A_to_B", "B_to_A")]
        result["assigned_high_depth_counts"] = {str(j):sum(e["minimum_level"] == j for e in mechanisms)
                                               for j in range(count)}
        result["mechanism_witnesses"] = [next(e for e in mechanisms if e["direction"] == direction)
                                          for direction in ("A_to_B", "B_to_A")
                                          if any(e["direction"] == direction for e in mechanisms)]
    if kind == "double_well":
        # Bound control run by the DNS cost: at least 3 likelihood calls per step.
        budget = result["likelihood_evaluations"]
        control_steps = int(np.max(costs.sum(0))//3+1)
        particles = state.particle

        def control(carry, key):
            new, inf = jax.vmap(parameter, in_axes=(0, 0, None))(
                jax.random.split(key, walkers), carry, levels.loglikelihood[high])
            return new, (new.position, inf.num_steps+inf.num_shrink)

        _, (cx, ccost) = jax.jit(lambda p, k: jax.lax.scan(control, p, k))(
            particles, jax.random.split(jax.random.key(seed+50000), control_steps))
        cumulative = np.cumsum(np.asarray(ccost), axis=0)
        lengths = [min(control_steps, np.searchsorted(cumulative[:, w], costs[:, w].sum())+1) for w in range(walkers)]
        spent = sum(int(cumulative[n-1,w]) for w,n in enumerate(lengths))
        crossings = sum(int(np.count_nonzero(np.diff(np.r_[start_sign, np.sign(np.asarray(cx)[:n,w])])) ) for w,n in enumerate(lengths))
        result["control"] = dict(likelihood_evaluations=spent, cost_ratio=spent/budget,
                                   switches=crossings, high_level=high, seed=seed+50000)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/dns-stage3-results.json")
    args = parser.parse_args()
    report = dict(dtype="float64" if jax.config.x64_enabled else "float32", runs=[], sensitivity=[], refinement=[])
    config = dns_levels.ConstructionConfig()
    for kind in ("gaussian", "double_well"):
        for seed in (101, 202, 303, 404):
            levels, construction = build(kind, seed, config=config)
            if construction["status"] != "max_levels":
                raise RuntimeError(f"Construction stopped early: {kind}, {seed}, {construction['status']}")
            run = dict(construction=construction, production=production(kind, seed+1000, levels,
                         start_sign=1 if seed in (202,404) else -1))
            report["runs"].append(run)
            print(kind, seed, construction["status"], flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
    for walkers, draws in ((2, 256), (8, 256), (2, 1024), (8, 1024)):
        for seed in (501, 502, 503, 504):
            cfg = dns_levels.ConstructionConfig(construction_draws=draws, calibration_draws=2*draws,
                                                block_size=32, min_ess=30)
            _, metadata = build("gaussian", seed, walkers, cfg)
            report["sensitivity"].append(metadata)
    _, prior, likelihood, kernel = problem("gaussian")
    for seed in (71, 72, 73, 74):
        cfg = dns_levels.ConstructionConfig(calibration_draws=256, block_size=32, min_ess=30)
        coarse, metadata = build("gaussian", seed, config=cfg)
        starts = jax.random.uniform(jax.random.key(seed+500), (8,), minval=-1., maxval=1.)
        refined, calibration = dns_levels.calibrate_level_masses(
            seed+1000, coarse, starts, prior, likelihood, kernel)
        truth = np.log(true_mass("gaussian", coarse.loglikelihood))
        report["refinement"].append(dict(construction=metadata, calibration=calibration,
            initial_seed=seed+500, before=(np.asarray(coarse.log_mass)-truth).tolist(),
            after=(np.asarray(refined.log_mass)-truth).tolist()))
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
