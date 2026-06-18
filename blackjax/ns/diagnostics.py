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


class ClusterLocalWhiteningDiagnostics(NamedTuple):
    """Per-cluster covariance and whitening diagnostics.

    Attributes
    ----------
    cluster_labels
        Cluster label values in the order used by all per-cluster arrays.
    cluster_sizes
        Number of live points in each cluster.
    mean
        Per-cluster mean vectors in flattened live-point coordinates.
    covariance
        Empirical per-cluster covariance matrices before regularization.
    regularized_covariance
        Covariance matrices after diagonal jitter and eigenvalue flooring.
    eigenvalues
        Eigenvalues of the regularized covariance matrices.
    condition_number
        Ratio of largest to smallest regularized covariance eigenvalue.
    whitening_matrix
        Matrices that map centered flattened coordinates to whitened
        coordinates by ``whitening_matrix @ (x - mean)``.
    unwhitening_matrix
        Inverse maps for ``whitening_matrix``.
    """

    cluster_labels: np.ndarray
    cluster_sizes: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    regularized_covariance: np.ndarray
    eigenvalues: np.ndarray
    condition_number: np.ndarray
    whitening_matrix: np.ndarray
    unwhitening_matrix: np.ndarray


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


def cluster_local_whitening_diagnostics(
    live_points,
    labels: np.ndarray,
    *,
    absolute_jitter: float = 1e-12,
    relative_jitter: float = 1e-10,
    minimum_eigenvalue: float = 1e-12,
) -> ClusterLocalWhiteningDiagnostics:
    """Compute covariance regularization and whitening summaries per cluster.

    Parameters
    ----------
    live_points
        Raw live-point position PyTree, or an object carrying ``.position`` or
        ``.particles.position``. Position leaves are flattened and concatenated
        in the same way as :func:`diagnose_live_point_clusters`.
    labels
        Integer cluster label for each live point, for example the ``labels``
        returned by :func:`diagnose_live_point_clusters`.
    absolute_jitter
        Non-negative diagonal jitter added to every cluster covariance.
    relative_jitter
        Non-negative diagonal jitter multiplier. The added amount is
        ``relative_jitter * covariance_scale``, where ``covariance_scale`` is
        the mean covariance diagonal. Degenerate zero-scale clusters therefore
        rely on ``absolute_jitter`` and ``minimum_eigenvalue``.
    minimum_eigenvalue
        Non-negative floor applied to eigenvalues after diagonal jitter.

    Returns
    -------
    ClusterLocalWhiteningDiagnostics
        Per-cluster means, empirical and regularized covariances, regularized
        eigenvalues, condition numbers, and whitening/unwhitening matrices.

    Notes
    -----
    This helper is diagnostics-only and is not called by nested-sampling
    kernels. Clusters with fewer than two points use a zero empirical
    covariance, then the requested jitter and eigenvalue floor produce a finite
    positive-definite fallback covariance. Singular and nearly singular
    covariances are handled the same way by flooring the jittered eigenvalues.
    """
    if absolute_jitter < 0.0:
        raise ValueError("absolute_jitter must be non-negative")
    if relative_jitter < 0.0:
        raise ValueError("relative_jitter must be non-negative")
    if minimum_eigenvalue < 0.0:
        raise ValueError("minimum_eigenvalue must be non-negative")

    particles = getattr(live_points, "particles", live_points)
    position = getattr(particles, "position", particles)
    points = _as_live_point_matrix(position)
    labels_array = np.asarray(labels)
    if labels_array.shape[0] != points.shape[0]:
        raise ValueError("labels must have one value per live point")

    cluster_labels = np.unique(labels_array)
    cluster_labels = cluster_labels[cluster_labels >= 0]
    num_clusters = int(cluster_labels.size)
    num_dim = int(points.shape[1])

    cluster_sizes = np.zeros(num_clusters, dtype=int)
    means = np.zeros((num_clusters, num_dim), dtype=points.dtype)
    covariances = np.zeros((num_clusters, num_dim, num_dim), dtype=float)
    regularized_covariances = np.zeros_like(covariances)
    eigenvalues = np.zeros((num_clusters, num_dim), dtype=float)
    condition_numbers = np.full(num_clusters, np.inf, dtype=float)
    whitening_matrices = np.zeros_like(covariances)
    unwhitening_matrices = np.zeros_like(covariances)

    identity = np.eye(num_dim)
    for output_id, cluster_label in enumerate(cluster_labels):
        cluster_points = points[labels_array == cluster_label]
        cluster_sizes[output_id] = int(cluster_points.shape[0])
        means[output_id] = np.mean(cluster_points, axis=0)

        if cluster_points.shape[0] > 1:
            covariance = np.atleast_2d(np.cov(cluster_points, rowvar=False))
        else:
            covariance = np.zeros((num_dim, num_dim), dtype=float)
        covariance = np.asarray(covariance, dtype=float).reshape(num_dim, num_dim)
        covariances[output_id] = covariance

        covariance_scale = float(np.trace(covariance) / num_dim) if num_dim else 0.0
        jitter = absolute_jitter + relative_jitter * max(covariance_scale, 0.0)
        jittered_covariance = covariance + jitter * identity
        raw_eigenvalues, eigenvectors = np.linalg.eigh(jittered_covariance)
        clipped_eigenvalues = np.maximum(raw_eigenvalues, minimum_eigenvalue)
        regularized_covariance = (
            eigenvectors * clipped_eigenvalues
        ) @ eigenvectors.T

        regularized_covariances[output_id] = regularized_covariance
        eigenvalues[output_id] = clipped_eigenvalues
        if clipped_eigenvalues[0] > 0.0:
            condition_numbers[output_id] = float(
                clipped_eigenvalues[-1] / clipped_eigenvalues[0]
            )
        whitening_matrices[output_id] = (
            np.diag(1.0 / np.sqrt(clipped_eigenvalues)) @ eigenvectors.T
        )
        unwhitening_matrices[output_id] = eigenvectors @ np.diag(
            np.sqrt(clipped_eigenvalues)
        )

    return ClusterLocalWhiteningDiagnostics(
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
        mean=means,
        covariance=covariances,
        regularized_covariance=regularized_covariances,
        eigenvalues=eigenvalues,
        condition_number=condition_numbers,
        whitening_matrix=whitening_matrices,
        unwhitening_matrix=unwhitening_matrices,
    )


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
