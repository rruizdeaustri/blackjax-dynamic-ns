# Nested-sampling live-point clustering diagnostics

`blackjax.ns.diagnostics.diagnose_live_point_clusters` is an optional helper for
inspecting the current nested-sampling live-point cloud outside the sampler
kernels. It clusters live points in flattened parameter or unit-cube space using
a simple Euclidean connected-component threshold and reports:

- the number of clusters;
- cluster labels and sizes;
- per-cluster minimum, mean, and maximum log-likelihood;
- optional per-cluster covariance condition numbers.

The diagnostic is intentionally NumPy based and non-invasive. It is intended to
support future mode-aware nested-sampling improvements by making live-point mode
separation easier to monitor in multimodal or narrow-mode problems. Calling it
does not alter sampler behaviour, replacement sampling, evidence integration, or
any JIT-critical nested-sampling kernel.

```python
from blackjax.ns import diagnostics

summary = diagnostics.diagnose_live_point_clusters(
    state,
    radius=0.1,
    include_covariance_condition=True,
)
print(summary.num_clusters)
print(summary.cluster_sizes)
```

## Cluster-local covariance and whitening diagnostics

`blackjax.ns.diagnostics.cluster_local_whitening_diagnostics` can be called on
live points and cluster labels to inspect the local geometry of each detected
cluster. For every cluster it reports:

- the mean vector in flattened live-point coordinates;
- the empirical covariance matrix;
- a regularized covariance matrix;
- regularized eigenvalues and the resulting condition number;
- whitening and unwhitening matrices.

The helper is still diagnostics only. It does not change the sampler kernel,
replacement sampling, evidence calculation, or default nested-sampling
behaviour. The purpose is to make cluster-local geometry visible now and to
support future cluster-aware nested-sampling proposals.

Regularization is controlled by three keyword arguments:

- `absolute_jitter`, an absolute diagonal covariance jitter;
- `relative_jitter`, a diagonal jitter scaled by the mean covariance diagonal;
- `minimum_eigenvalue`, a floor applied after diagonal jitter.

Small or degenerate clusters are handled conservatively. A one-point cluster has
zero empirical covariance, then the requested jitter and eigenvalue floor define
a finite positive-definite fallback covariance. Singular or nearly singular
covariances are treated the same way by flooring the jittered eigenvalues.

```python
whitening = diagnostics.cluster_local_whitening_diagnostics(
    state,
    summary.labels,
    absolute_jitter=1e-12,
    relative_jitter=1e-10,
    minimum_eigenvalue=1e-12,
)

cluster_id = 0
centered = point - whitening.mean[cluster_id]
whitened = whitening.whitening_matrix[cluster_id] @ centered
```

## Toy validation: global versus cluster-local whitening

`blackjax.ns.diagnostics.compare_global_and_cluster_whitening` is a small
validation helper for labelled live-point clouds. It computes one global
whitening transform from all live points, computes separate whitening transforms
inside each cluster, and reports both sets of covariance condition numbers and
per-cluster whitened covariance errors relative to the identity matrix.

These toy diagnostics are useful for separated modes. A single global covariance
contains both within-mode geometry and between-mode separation. For narrow or
elongated clusters this can make the globally whitened covariance of each mode
far from identity and can compress the distance between separated mode centres in
whitened space. Cluster-local whitening instead measures each mode around its own
mean, so it better reflects the local anisotropy that a future cluster-aware
replacement proposal would need to navigate.

The comparison helper is intentionally not wired into the sampler. It is a
validation and documentation utility only: it does not change nested-sampling
kernels, replacement sampling, evidence calculation, or sampler state.

```python
comparison = diagnostics.compare_global_and_cluster_whitening(
    state,
    summary.labels,
)
print(comparison.global_condition_number)
print(comparison.cluster_condition_number)
print(comparison.global_identity_error)
print(comparison.local_identity_error)
```
