# Copyright 2020- The Blackjax Authors.
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
"""Optional diagnostics for Nested Sampling live-point clouds.

The helpers in this module are intentionally NumPy based and are not used by
sampler kernels. They are intended for post-step inspection of live points, for
example to monitor whether a multimodal run is separating into distinct live
point clusters. They do not modify nested-sampling state, replacement sampling,
or evidence calculations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np


class LivePointClusterDiagnostics(NamedTuple):
    """Summary of connected clusters in a live-point population.

    Attributes
    ----------
    num_clusters
        Number of connected components found in the live-point coordinates.
    labels
        Integer cluster label for each live point.
    cluster_sizes
        Number of live points in each cluster.
    loglikelihood_min
        Minimum log-likelihood in each cluster.
    loglikelihood_mean
        Mean log-likelihood in each cluster.
    loglikelihood_max
        Maximum log-likelihood in each cluster.
    covariance_condition_number
        Per-cluster covariance condition number when requested, otherwise
        ``None``. Clusters with fewer than two points have infinite condition
        number.
    radius
        Distance threshold used to connect live points.
    """

    num_clusters: int
    labels: np.ndarray
    cluster_sizes: np.ndarray
    loglikelihood_min: np.ndarray
    loglikelihood_mean: np.ndarray
    loglikelihood_max: np.ndarray
    covariance_condition_number: np.ndarray | None
    radius: float


def _as_live_point_matrix(position) -> np.ndarray:
    """Flatten a live-point position PyTree into an ``(n_live, n_dim)`` array."""
    leaves = jax.tree.leaves(position)
    if not leaves:
        raise ValueError("position must contain at least one array leaf")

    arrays = [np.asarray(leaf) for leaf in leaves]
    num_live = arrays[0].shape[0]
    if any(array.shape[0] != num_live for array in arrays):
        raise ValueError("all position leaves must share the same leading dimension")

    flattened = [array.reshape(num_live, -1) for array in arrays]
    return np.concatenate(flattened, axis=1)


def _default_radius(points: np.ndarray) -> float:
    """Choose a transparent nearest-neighbour radius for connected components."""
    if points.shape[0] < 2:
        return 0.0

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size == 0:
        return 0.0
    return float(3.0 * np.median(finite_nearest))


def _connected_component_labels(points: np.ndarray, radius: float) -> np.ndarray:
    """Label connected components under an Euclidean distance threshold."""
    num_points = points.shape[0]
    labels = -np.ones(num_points, dtype=int)
    if num_points == 0:
        return labels

    adjacency = (
        np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1) <= radius
    )
    cluster_id = 0
    for start in range(num_points):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = cluster_id
        while stack:
            node = stack.pop()
            neighbours = np.flatnonzero(adjacency[node] & (labels == -1))
            labels[neighbours] = cluster_id
            stack.extend(neighbours.tolist())
        cluster_id += 1
    return labels


def diagnose_live_point_clusters(
    live_points,
    loglikelihood: np.ndarray | None = None,
    *,
    radius: float | None = None,
    include_covariance_condition: bool = False,
) -> LivePointClusterDiagnostics:
    """Cluster nested-sampling live points for optional diagnostics.

    Parameters
    ----------
    live_points
        Either an NS state/particle object with ``.particles.position`` or
        ``.position`` and ``.loglikelihood`` attributes, or a raw position PyTree.
    loglikelihood
        Per-live-point log-likelihoods. Optional when ``live_points`` carries a
        ``loglikelihood`` array.
    radius
        Euclidean connection radius in flattened parameter/unit-cube space. When
        omitted, the helper uses three times the median nearest-neighbour
        distance.
    include_covariance_condition
        Whether to report per-cluster covariance condition numbers.

    Returns
    -------
    LivePointClusterDiagnostics
        Cluster labels, sizes, log-likelihood summaries, and optional covariance
        condition numbers.

    Notes
    -----
    This helper is deliberately non-invasive: it performs NumPy calculations
    outside JIT-critical sampler kernels and never changes sampler behaviour,
    replacement sampling, or evidence integration.
    """
    particles = getattr(live_points, "particles", live_points)
    position = getattr(particles, "position", particles)
    if loglikelihood is None:
        loglikelihood = getattr(particles, "loglikelihood", None)
    if loglikelihood is None:
        raise ValueError("loglikelihood must be provided for raw position inputs")

    points = _as_live_point_matrix(position)
    loglikelihood_array = np.asarray(loglikelihood)
    if loglikelihood_array.shape[0] != points.shape[0]:
        raise ValueError("loglikelihood must have one value per live point")

    if radius is None:
        radius = _default_radius(points)
    labels = _connected_component_labels(points, float(radius))
    num_clusters = int(labels.max() + 1) if labels.size else 0

    cluster_sizes = np.zeros(num_clusters, dtype=int)
    loglikelihood_min = np.zeros(num_clusters, dtype=loglikelihood_array.dtype)
    loglikelihood_mean = np.zeros(num_clusters, dtype=float)
    loglikelihood_max = np.zeros(num_clusters, dtype=loglikelihood_array.dtype)
    condition_numbers = None
    if include_covariance_condition:
        condition_numbers = np.full(num_clusters, np.inf, dtype=float)

    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        cluster_loglikelihood = loglikelihood_array[mask]
        cluster_sizes[cluster_id] = int(mask.sum())
        loglikelihood_min[cluster_id] = np.min(cluster_loglikelihood)
        loglikelihood_mean[cluster_id] = float(np.mean(cluster_loglikelihood))
        loglikelihood_max[cluster_id] = np.max(cluster_loglikelihood)
        if include_covariance_condition and cluster_sizes[cluster_id] > 1:
            covariance = np.atleast_2d(np.cov(points[mask], rowvar=False))
            condition_numbers[cluster_id] = float(np.linalg.cond(covariance))

    return LivePointClusterDiagnostics(
        num_clusters=num_clusters,
        labels=labels,
        cluster_sizes=cluster_sizes,
        loglikelihood_min=loglikelihood_min,
        loglikelihood_mean=loglikelihood_mean,
        loglikelihood_max=loglikelihood_max,
        covariance_condition_number=condition_numbers,
        radius=float(radius),
    )
