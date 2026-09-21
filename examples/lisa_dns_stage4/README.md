# Seed-44 smoke adapter

From the repository root, using the environment that passed the read-only audit:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=. MPLCONFIGDIR=/tmp/lisa_dns_mpl \
/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python \
-m examples.lisa_dns_stage4.run_smoke \
--initial-families /tmp/lisa_dns_stage4_audit/initial_families.npz \
--output /tmp/lisa_dns_stage4_smoke --seed 4404
```

This command loads the audited physics unchanged. The seed44 archive contains
`samples`, `samples_u`, `weights`, `diagnostics_json`, `seed`, `algo`, and
`config_path`; no fixed covariance is persisted. Scales are the unweighted
coordinate standard deviations (`ddof=1`) of its 69 `samples_u` rows, with no
clipping, multiplier, label sorting, or online adaptation. All 54 scales and
input hashes are stored in the report. These are local posterior-derived
engineering scales, not estimates of the prior's spread.

There are 64 steps per fixed contour. Ladder selection and calibration each
have 128 burn-in steps and 256 retained steps, in eight blocks of 32, with
minimum tail ESS 20. That minimum is an engineering gate, not an accuracy
claim. Independent random streams start from the same seed44 point; neither
bank is guaranteed to be equilibrated. The adapter uses Stage-3
`build_next_level` and `tail_diagnostics` on host-managed chains to retain
label diagnostics, separate costs, and direct cache checks at every step.
Failure of either information gate stops the run without production or retry.
If accepted, all three levels, masses, equal level weights, scales, and the
20/40/40 proposal mixture are frozen for 512 production sweeps.

`report.json` contains diagnostics; `trace.npz` contains the checked states,
labels, slice diagnostics, and production level transitions (sentinel indices
-1 for construction/fixed-contour steps). `levels.npz` includes only accepted
levels. `checkpoints.npz` stores sparse named latent points. JSON represents
negative infinity as the string `"-inf"`. Reconstruction of physical catalogue
parameters uses the existing audit helper; it does not reconstruct a posterior.

The first seed-4404 attempt stopped at level-1 calibration: selection ESS
22.5266 passed, calibration ESS 13.7704 failed. No level beyond zero was
accepted, and no DNS production ran. All 896 parameter steps passed cache,
finite-value, and contour checks, with zero slice failures. Do not interpret
this as validation of masses, level diffusion, or readiness for multi-start.
Full-run reproducibility has not been replayed because the ESS failure requires
a stop and diagnosis. The random-stream scheme and input hashes are recorded.

Evidence reconstruction has not yet been validated and is out of scope.
