"""Stage-6A frozen numerical contract; NumPy only, no model/sampling imports."""
import numpy as np

PRIOR_ATOL = 1e-9
LIKELIHOOD_ATOL = 1e-7
RTOL = 0.0
CONTRACT = dict(version=1, backend='cpu', jax_enable_x64=True, dtype='float64',
                prior_atol=PRIOR_ATOL, likelihood_atol=LIKELIHOOD_ATOL, rtol=RTOL,
                finite_coordinates_and_scalars=True, strict_cached_and_recomputed_contour=True,
                copied_coordinates='byte-exact float64 copy; no tolerance',
                copied_scalar_caches='byte-exact from source; recomputation uses numerical tolerances')


def validate_cache(position, reference_position, cached_prior, cached_likelihood,
                   recomputed_prior, recomputed_likelihood, threshold, *, backend, x64):
    """Validate fixed saved/copied coordinates and directly evaluated scalars.

    reference_position is the source for a copy, or the exact evaluation input
    for a newly generated state. New transition states need not equal old ones.
    The caller must establish backend/x64 from JAX, not infer them from values.
    """
    if backend != 'cpu' or x64 is not True:
        raise ValueError('CPU with JAX x64 required')
    arrays = [np.asarray(a) for a in (position, reference_position, cached_prior,
                                    cached_likelihood, recomputed_prior, recomputed_likelihood)]
    if any(a.dtype != np.dtype('float64') for a in arrays):
        raise ValueError('All coordinates and scalar caches must be float64')
    x, ref, cp, cl, rp, rl = arrays
    if x.ndim != 2 or x.shape[1] != 54 or ref.shape != x.shape:
        raise ValueError('Expected equal (N,54) coordinate arrays')
    if any(a.shape != (len(x),) for a in (cp,cl,rp,rl)):
        raise ValueError('Expected one prior/likelihood scalar per state')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite coordinates or scalar caches')
    if x.tobytes() != ref.tobytes():
        raise ValueError('Copied coordinates differ from exact evaluation/source coordinates')
    if not np.isfinite(threshold):
        raise ValueError('This LISA audit requires a finite fixed contour')
    if np.any(cl <= threshold) or np.any(rl <= threshold):
        raise ValueError('Strict contour violation')
    if not np.allclose(rp,cp,atol=PRIOR_ATOL,rtol=RTOL):
        raise ValueError('Prior cache exceeds historical tolerance')
    if not np.allclose(rl,cl,atol=LIKELIHOOD_ATOL,rtol=RTOL):
        raise ValueError('Likelihood cache exceeds historical tolerance')
