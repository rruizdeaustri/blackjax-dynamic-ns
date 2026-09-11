# Copyright 2026 The BlackJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline diagnostics for frozen-level Diffusive Nested Sampling."""

import jax.numpy as jnp
import numpy as np


def highest_eligible_level(levels, loglikelihood):
    """Return the highest nested level containing each likelihood value.

    Level zero is always eligible. Remaining levels use strict threshold
    exceedance, matching blackjax.ns.dns.is_eligible.
    """
    values = jnp.asarray(loglikelihood)
    thresholds = levels.loglikelihood[1:]
    return jnp.sum(values[..., None] > thresholds, axis=-1, dtype=jnp.int32)


def level_occupancy(level_index, num_levels):
    """Normalized assigned-level occupancy from a retained trajectory."""
    indices = np.asarray(level_index).reshape(-1)
    counts = np.bincount(indices, minlength=int(num_levels)).astype(float)
    total = counts.sum()
    return counts / total if total else counts


def _count_round_trip(values, start, middle):
    stage = 0
    count = 0
    for value in np.asarray(values).reshape(-1):
        if stage == 0:
            if value == start:
                stage = 1
        elif stage == 1:
            if value == middle:
                stage = 2
        else:
            if value == start:
                count += 1
                stage = 1
    return count


def round_trip_counts(level_index, low_level=0, high_level=None):
    """Count complete low-high-low and high-low-high assigned-level trips.

    For batched trajectories pass one walker at a time.
    """
    values = np.asarray(level_index).reshape(-1)
    if values.size == 0:
        return {"low_high_low": 0, "high_low_high": 0}
    if high_level is None:
        high_level = int(values.max())
    return {
        "low_high_low": _count_round_trip(values, int(low_level), int(high_level)),
        "high_low_high": _count_round_trip(values, int(high_level), int(low_level)),
    }


def mode_switch_summary(
    mode_label,
    assigned_level,
    highest_level,
    *,
    high_level,
):
    """Summarize switches between high-likelihood modes.

    A mode transition is registered only between observations whose highest
    eligible level is at least high_level. The minimum assigned level since
    the preceding high-likelihood visit is retained as a backtracking-depth
    diagnostic.
    """
    modes = np.asarray(mode_label).reshape(-1)
    assigned = np.asarray(assigned_level).reshape(-1)
    highest = np.asarray(highest_level).reshape(-1)
    if not (modes.size == assigned.size == highest.size):
        raise ValueError("mode_label, assigned_level, and highest_level must align")

    high_indices = np.flatnonzero(highest >= int(high_level))
    previous_mode = None
    previous_high_index = None
    a_to_b = 0
    b_to_a = 0
    first_passage = None
    backtracking_minima = []

    for index in high_indices:
        mode = int(modes[index])
        if mode == 0:
            continue
        if previous_mode is None:
            previous_mode = mode
            previous_high_index = int(index)
            continue
        if mode != previous_mode:
            minimum = int(assigned[previous_high_index : int(index) + 1].min())
            backtracking_minima.append(minimum)
            if previous_mode < mode:
                a_to_b += 1
            else:
                b_to_a += 1
            if first_passage is None:
                first_passage = int(index)
            previous_mode = mode
        previous_high_index = int(index)

    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "num_switches": a_to_b + b_to_a,
        "first_passage_index": first_passage,
        "backtracking_minima": np.asarray(backtracking_minima, dtype=int),
    }


def edge_statistics(level_info, num_levels):
    """Aggregate neighboring-level attempts, eligibility, and acceptance."""
    previous = np.asarray(level_info.previous_level).reshape(-1)
    proposed = np.asarray(level_info.proposed_level).reshape(-1)
    eligible = np.asarray(level_info.is_eligible).reshape(-1)
    accepted = np.asarray(level_info.is_accepted).reshape(-1)
    n_edges = max(int(num_levels) - 1, 0)
    result = {
        "up_attempts": np.zeros(n_edges, dtype=int),
        "down_attempts": np.zeros(n_edges, dtype=int),
        "up_eligible": np.zeros(n_edges, dtype=int),
        "down_eligible": np.zeros(n_edges, dtype=int),
        "up_accepted": np.zeros(n_edges, dtype=int),
        "down_accepted": np.zeros(n_edges, dtype=int),
    }
    for old, new, is_eligible, is_accepted in zip(
        previous, proposed, eligible, accepted
    ):
        if new == old + 1 and 0 <= old < n_edges:
            edge = int(old)
            prefix = "up"
        elif new == old - 1 and 0 <= new < n_edges:
            edge = int(new)
            prefix = "down"
        else:
            continue
        result[f"{prefix}_attempts"][edge] += 1
        result[f"{prefix}_eligible"][edge] += int(is_eligible)
        result[f"{prefix}_accepted"][edge] += int(is_accepted)
    return result


def maximum_loglikelihood(loglikelihood):
    """Return the largest finite retained log likelihood, or -inf if empty."""
    values = np.asarray(loglikelihood).reshape(-1)
    if values.size == 0:
        return -np.inf
    return float(np.nanmax(values))
