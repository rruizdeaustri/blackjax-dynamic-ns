# Seed-44 smoke adapter

From the repository root, using the environment that passed the read-only audit:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=. MPLCONFIGDIR=/tmp/lisa_dns_mpl \
/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python \
-m examples.lisa_dns_stage4.run_smoke \
--initial-families /tmp/lisa_dns_stage4_audit/initial_families.npz \
--output /tmp/lisa_dns_stage4_prior_smoke --seed 4404
```

The first smoke attempt (`/tmp/lisa_dns_stage4_smoke`) stopped correctly at
calibration ESS 13.7704 < 20, but its level-zero chains were short relaxations
from the seed44 optimum. This runner replaces that initialization with the
actual problem's direct prior sampler. Neither historical positions nor their
likelihoods enter level-one selection or calibration. Historical scales remain
frozen as requested.

## Actual prior sampler

NSS `BlackJAXNestedSampler._setup_common` calls `problem.sample_prior(sub, n_live)`.
`LISAGBTransdimProblem.sample_prior` delegates to the `sample_prior` closure in
`lisa_gb_transdim_problem.make` (not its legacy `sample_prior_flat` function).
For this configuration it splits a key into six keys, draws uniform physical
arrays of shape `(n, 9)` for `f0, fdot, iota, psi, lam, beta`, applies the existing
`box_to_u` transform, stacks along the last axis, and reshapes to `(n, 54)`.
`box_to_u` normalizes to the box, clips to `[1e-9, 1-1e-9]`, and takes the logit.
That existing finite-precision clipping is preserved and audited for hits.
No gates, ordering, Gaussian priors, amplitude, or phase coordinates participate.

The sampler uses `problem.prior_box`, including the backend's frequency-bin
aligned endpoints, rather than substituting the requested band's decimal
endpoints. For the verified configuration the actual f0 box is
`[0.001830003805175038, 0.0018500126839167937]`. This is existing physics.

Each generated bank is checked for shape, dtype, finite prior values, exact
replay of the inspected sampler's operations, and physical-coordinate round
trips. Sigmoid-transformed coordinates are compared to uniform marginals using
54 KS statistics and a simultaneous DKW bound (alpha .005 per bank). This is a
sanity check; random-stream independence follows from the sampler construction,
not from marginal testing. Input/config/source hashes are recorded.

## Bounded construction

* Level zero: 512 draws in each bank; keys use seeds `seed+10` and `seed+11`.
  Likelihoods are evaluated with scalar `lax.map` in chunks of 32, never with a
  large waveform batch. There must be no identical rows shared by the banks.
* Level one selection: the Stage-3 strict order statistic
  `sort(logL)[floor(N*(1-exp(-1)))]`, excluding all ties from the tail.
* Level one calibration: only the other bank estimates exceedance probability.
  Report binomial standard error and a Wilson 95% interval, conditional on the
  independently selected threshold. No MCMC ESS or block diagnostic is called.
  The predeclared information gate requires at least 20 exceedances and 20
  non-exceedances. The log mass is the log of the calibration fraction.
* Level two attempt: randomly choose eight distinct survivors from each bank,
  without replacement, using seeds `seed+20` and `seed+21`. Keep the banks
  separate. Each walker gets its own stream (`seed+100+100*bank+walker`), 128
  burn-in steps, and 256 retained steps. Walkers run sequentially to bound GPU
  memory; they remain eight separate histories, not one concatenated chain.
  Stage-3 diagnostics receive `(256, 8)` arrays, block size 32, and unchanged
  `min_ess=20`. Report within-walker and between-walker uncertainty terms.
* Any information-gate failure stops construction and production. No retry,
  automatic budget increase, online adaptation, or tuning sweep is performed.
* Only after both transitions pass, freeze thresholds, calibrated log masses,
  equal level weights, scales, and the 20/40/40 global/single/pair proposal.
  Load seed44, assign its highest eligible level, and run at most 512 sweeps
  with seed `seed+10000`. No seed66/88 run is supported by this adapter.

The seed44 historical archive contains no fixed covariance. Scales remain the
unweighted coordinate standard deviations (`ddof=1`) of its 69 `samples_u`
rows, with no clipping, multiplier, label sorting, or adaptation. They may be
poorly suited to prior-wide mixing; failure diagnostics must precede any change.

## Artifacts and limitations

`report.json` contains settings, hashes, sampler audit, gates, diagnostics, and
an old/new comparison. `prior_banks.npz` stores both banks and scalar values;
`constrained_histories.npz` preserves time/walker axes; `trace.npz` stores all
constrained steps including burn-in, direct cache checks, proposal labels,
slice information, and production transitions if production runs. Construction
rows have sentinel assigned levels of -1. `levels.npz` contains only accepted
levels and equal log weights; `scales.npy` contains all 54 frozen scales.
Sparse catalogue checkpoints are decoded only if production runs. JSON encodes
negative infinity as the string `"-inf"`.

All constrained steps are directly checked for finite coordinates, finite scalar
prior/likelihood, contour membership, and cache consistency. The slice cost is
an evaluation proxy (`num_steps + num_shrink`); direct prior-bank likelihood and
cache-check evaluations are counted separately. Approximate block ESS cannot
prove higher-contour equilibrium or coverage of all catalogue families. Direct
prior calibration fixes the level-zero initialization bias, not all mixing
problems at subsequent contours.

Evidence reconstruction has not yet been validated and is out of scope.

## First direct-prior result (seed 4404)

The two 512-point banks (seeds 4414 and 4415) passed sampler replay, finite-prior,
coordinate-marginal, and independence checks, with zero clipping hits. Level one
was accepted at `-113725.65889538352`: independent calibration `182/512`,
`X1=0.35546875`, binomial SE `0.02115377`, Wilson interval
`[0.31522509, 0.39786505]`, and `log X1=-1.0343179379627125`.

The eight-walker attempt proposed level two at `-113462.60568488069`. Selection
ESS was `23.65463`, but independent calibration ESS was `13.73685`, so it was
rejected at the unchanged minimum of 20. Calibration walker exceedance fractions
ranged from `0.03125` to `0.99609375`. Between-walker SE (`0.13438244`) dominated
batch-means SE (`0.05283002`) and IID SE (`0.01100579`). The batch-means term
includes variation between walkers and is not a pure autocorrelation estimate.

All 6144 constrained steps passed the cache and contour checks with zero slice
failures. Their likelihood proxy was 50067, in addition to 1024 direct-prior
likelihoods and 6144 direct cache-check likelihoods. Mean absolute latent f0
movement was about 0.008, while iota/psi movement was about 2; the historical
scale anisotropy is a plausible contributor to slow catalogue exploration.
One short attempt cannot isolate burn-in, scales, and contour geometry.
No parameter changes, production, multi-start, or full-run replay followed the
failure. The report includes read-only diagnosis derived from the saved trace.
