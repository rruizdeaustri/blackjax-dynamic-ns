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
