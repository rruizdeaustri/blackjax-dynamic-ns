# Automatic DNS ladder construction

Stage 7A integrates the frozen DNS methodology into a generic automatic builder.
It constructs levels and calibrates their masses; it does not implement posterior
or evidence reconstruction, production LISA inference, or a final DNS termination
rule. No new LISA trajectory or new LISA level was constructed in this stage.

## Frozen scientific components

The following remain unchanged from the LISA validation campaign:

1. Constrained isotropic slice transition with scale `pi/sqrt(3)`.
2. Fixed 20% global / 40% labelled single-source / 40% labelled unordered
   source-pair proposal mixture, `max_steps=10`, `max_shrinkage=100`.
3. Four reciprocal frequency-guided exchanges after eight slice updates.
4. The deterministic seven-round pairing schedule, restarting at round zero
   for each new fixed-contour bank run.
5. The exact joint-prior and state-dependent forward/reverse selector MH ratio.
6. Uniform pooled-with-replacement promotion within each independent bank.
7. Independent threshold-selection and calibration banks, storage, and streams.
8. The Stage-6A CPU/x64 numerical contract.

The LISA adapter reuses the existing proposal, selector, exchange, and source-block
functions. `f_star=0.0018409808` and `sigma_f=2.558e-7` remain fixed. Its factory
accepts model functions, physical frequency bounds, and a model identity; it
exposes no proposal, exchange-frequency, selector, or pairing tuning parameters.
Physical bounds are model inputs, not selector tuning parameters. The previously
validated LISA source files and artifacts remain byte-for-byte unchanged.

The numerical contract is `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, `float64`,
`rtol=0`, prior `atol=1e-9`, and likelihood `atol=1e-7`. Promotion copies coordinates
and scalar caches byte-exactly. Finite checks and strict cached **and** recomputed
contour membership are mandatory. The generic implementation allows `-inf` only
as the initial unconstrained contour; later contours are finite. Numerical
contract and ESS-gate modifications are rejected, not silently adopted.

## Implementation and API

- [Core automatic builder](../../blackjax/ns/dns_automatic.py): bank-local
  promotion, immutable candidate records, independent calibration, freeze gate,
  bounded loop, and checksummed restart bundles.
- [Core population adapter](../../blackjax/ns/dns_population.py): the fixed
  slice-then-reciprocal-exchange population transition with a contour argument.
- [Frozen LISA integration adapter](../../examples/lisa_dns_stage4/automatic_ladder_integration.py):
  binds the existing scientific operations and provides saved-artifact checks.
  It has no LISA problem loader or LISA trajectory command.
- [Multimodal toy example](../../examples/dns_automatic_ladder.py).

`dns_automatic.build_next_level(state, kernel, output)` makes one durable attempt.
`state` carries the current frozen threshold/log-mass prefix, both retained
histories, a master seed and next-iteration counter, fixed sampling settings,
numerical contract, and frozen-kernel metadata. The output directory must be new.

`LadderConfig` accepts `burn_in`, `retained`, `block_size`, `target_compression`,
and `min_ess`. The default budgets are 128 burn-in and 256 retained sweeps,
with block size 32 and target compression `exp(-1)`. `min_ess` must equal the
frozen value **20**; the retained budget must contain at least four complete
blocks. These are fixed run inputs, not adaptive controls.

Histories use `DNSParticleState` with time-major arrays:

| Field | Shape |
|---|---|
| `position` | `(retained_time, walkers, dimension)` |
| `logdensity` (prior cache) | `(retained_time, walkers)` |
| `loglikelihood` | `(retained_time, walkers)` |

The orchestrator does not hard-code level numbers, historical thresholds,
particular walkers, or LISA rescue rules. The validated population adapter
requires eight walkers with nine labelled six-coordinate sources; those are
properties of the frozen kernel, not level-specific branching. A different
model can supply its own frozen kernel through the same interface.

The existing `dns_levels.build_next_level` remains the strict empirical quantile
and selection-diagnostic operation. The new orchestrator calls it rather than
changing its statistics. Existing APIs and historical scripts remain available;
new automatic construction uses `dns_automatic`.

## One iteration

1. Record the prefix, source-history hashes, environment, kernel identity,
   numerical contract, fixed budgets, and all stream seeds before any sampling.
2. For each bank separately, form its strict `logL > current_contour` pool in
   time-major order. Use the existing Stage-3 `_survivors` rule to draw the
   population uniformly with replacement. Empty individual donors are valid;
   only an entirely empty bank is a promotion-support failure. Invalid dtype,
   nonfinite data, or copy corruption is separately an integrity failure.
3. Copy both banks into independent storage and verify both initial populations,
   including direct prior/likelihood recomputation, before sampling either bank.
4. Sample selection at the **old** contour for the fixed burn-in and retained
   budgets. Select the strict empirical upper-tail order statistic
   `sorted_logL[floor(N * (1 - target_compression))]`; ties are excluded by `>`.
   Preserve the existing selection-information gate.
5. Write immutable `candidate.json` and its SHA-256, including selection
   diagnostics, **before calling the calibration sampler**. Calibration receives
   only its own starts and streams, and also runs at the **old** contour.
6. Verify the candidate bytes/hash remain unchanged. Evaluate candidate
   exceedance on calibration retained samples only. Reuse `tail_diagnostics`:
   the conservative standard error is the maximum of IID, block-means, and
   between-walker SE; ESS is its Bernoulli-variance equivalent. All three SEs
   and the component walker fractions are recorded explicitly.
7. If selection is valid, calibration compression is strictly between zero and
   one, calibration ESS is **at least 20**, and integrity checks pass, append
   exactly one threshold and
   `new_log_mass = old_log_mass + log(calibration_compression)`.
   The selection fraction never updates the reported mass.
8. Otherwise return a terminal `stopped` state with the frozen prefix unchanged.
   Do not retry, change a seed, extend burn-in, tune, or rescue from the other bank.

Retained histories produced at the old contour become the inputs for the next
iteration. Its promotion filters those histories at the newly frozen contour.
Promotion metadata records eligible counts per donor, sampled donor IDs and
multiplicities, retained/flat/pool indices, source hashes, and duplicate counts.
Duplicates are allowed and are not optimized away.

`run_ladder(state, kernel, output, max_new_levels=N)` repeats this operation until
STOP or the explicit engineering budget is exhausted. Budget exhaustion leaves a
`ready` state; it is **not** an implementation of scientific DNS termination.

The population adapter checks all post-slice and post-exchange states against
the numerical contract. Disjoint outcomes are checked in batches, as in Stage 6B.
Selected six-coordinate blocks must swap exactly; untouched blocks remain
unchanged; rejection restores both states; acceptance advances them jointly.
The builder also checks the returned retained shape and recomputes its caches.

## Checkpoint and restart

Each successful attempt produces one `checkpoint/` bundle:

- `banks.npz`: exact retained positions and scalar caches for the two banks;
  no pickle or model objects.
- `checkpoint.json`: a SHA-256-protected payload with schema `dns-automatic-v1`,
  frozen thresholds/log masses, history reference and SHA-256, master seed,
  next iteration, stream recipe, fixed budgets, promotion/candidate/calibration
  records, numerical-contract version, kernel version/source hashes, and
  numerical environment.

Bank streams are derived from a versioned NumPy `SeedSequence` namespace
`[master_seed, iteration, bank_id]`, with separate children for promotion, the
per-walker slice streams, source labels, and exchange MH. Promotion and slices
use explicit JAX `threefry2x32` keys; labels and MH use independent PCG64 streams.
Any seed collision is an error, not a reason to pick another seed.

Checkpoints represent **completed iteration boundaries**. The stored next
iteration and deterministic stream recipe reproduce the continuation exactly;
there is no hidden global RNG state. Loading checks the payload/history hashes,
schema, kernel identity, CPU/x64 environment, JAX/JAXlib/NumPy versions, and
contract. Histories load into separate read-only storage; promotion creates
fresh writable arrays. Incomplete checkpoint bundles are not loadable.

An existing attempt directory cannot be reused. Interrupted attempts are not
resumed or retried from their beginning, and partial trajectories are not
interpreted as completed levels. This implementation does not promise arbitrary
mid-sweep restart. Continue a completed checkpoint in its original output tree;
if that tree already contains the next attempted iteration, stop rather than
bypassing the attempt guard.

Example, **toy only**:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m examples.dns_automatic_ladder \
  --output /tmp/automatic-toy --levels 2
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m examples.dns_automatic_ladder \
  --resume /tmp/automatic-toy/attempt_000001/checkpoint \
  --output /tmp/automatic-toy --levels 2
```

The frozen population kernel is exercised on a 54-dimensional standard-normal
prior with a two-mode likelihood in its first coordinate. The remaining
coordinates are prior nuisances. No LISA model is imported or evaluated by this
example, although it reuses the frozen operation bundle.

## Saved LISA integration

The existing accepted level-10, level-11, and level-12 construction traces were
read without rerunning their trajectories. Generic candidate selection reproduces
the saved candidate records exactly, and the generic gate reproduces the complete
threshold and log-mass prefixes exactly:

| Existing level | Saved threshold | Independent calibration compression | Calibration ESS |
|---|---:|---:|---:|
| 10 | -109780.28875081123 | 0.4873046875 | 52.27700334 |
| 11 | -108868.60077896297 | 0.28662109375 | 83.04742208 |
| 12 | -108187.99712765554 | 0.4609375 | 634.1474270 |

Those historical runs used the former own-walker recovery rule. This check does
**not** claim pooled promotion reproduces their old starts or trajectories.
Current promotion is tested against the exact generic Stage-3 survivor sampler
and the frozen pooled semantics; historical bookkeeping is replayed separately.

The LISA dry run loads the frozen level-12 prefix and the independent saved
Stage-5B retained histories, which were generated at that old contour. At
`ell_12 = -108187.99712765554`, **all 2,048 states in each bank are eligible**.
This differs intentionally from Stage 6B's 753/266 pools at its tighter validation
contour: that historical candidate does not enter the automatic dry-run logic.

Using master test seed 170100, selection donor IDs are `[4, 0, 3, 4, 7, 4, 6, 4]`
and calibration donor IDs are `[5, 7, 3, 3, 2, 7, 1, 1]`; neither draw contains
duplicate states. Repeating preparation reproduces the exact arrays, indices,
hashes, and streams. Arrays are independently owned and copies are byte-exact.
Finite cached support and the CPU/x64 contract are checked. The recorded
Stage-6A fixture also passes the generic recomputation checks using **saved**
recomputed scalars. No fresh LISA recomputation was performed: the dry-run result
explicitly records that a future live run must pass its direct recomputation gate.

Every saved Stage-6B exchange was replayed through the new population adapter:
768 population sweeps / 3,072 reciprocal pair operations across both banks.
Outputs and every recorded MH decision match exactly. This test looks up saved
prior/likelihood values; it never evaluates the LISA likelihood. A separate toy
test verifies the dynamic-contour slice adapter matches the original frozen
slice operation, including returned proposal information, for the same keys.

No historical checkpoint was overwritten. No new LISA candidate, level-13
checkpoint, or log mass was constructed. Stage 5B remains officially failed;
the LISA ladder remains frozen through level 12.

## Automatic toy and restart validation

The fixed multimodal toy constructed four successive levels without manual intervention, using eight walkers, 128 burn-in sweeps and 256 retained sweeps per bank at each iteration.

| Automatically appended level | Threshold | Log mass | Calibration compression | Calibration ESS |
|---|---:|---:|---:|---:|
| 1 | -1.56526828938 | -0.559859958368 | 0.5712890625 | 61.20680263 |
| 2 | -0.30785938654 | -1.6521446523 | 0.33544921875 | 119.8748878 |
| 3 | -0.0512075880323 | -2.71009120866 | 0.34716796875 | 197.6255234 |
| 4 | -0.00958969912439 | -3.58360872588 | 0.41748046875 | 259.0843282 |

Thresholds strictly increase, log masses strictly decrease, and every promoted state passes strict current-contour support. This validates construction plumbing and the frozen transition integration; these finite toy calibration estimates are not an inference-reconstruction validation.

A fresh interpreter resumed the checkpoint after two completed levels and constructed the remaining two. Thresholds, log masses, retained coordinate/cache bytes, all promotion/candidate/calibration records, and the next RNG iteration exactly match the uninterrupted run.

A separate deliberately uninformative calibration fixture produces compression zero and ESS zero. The automatic loop returns STOP after one attempt, freezes no level, leaves the prefix unchanged, and refuses retry. Boundary tests accept ESS exactly 20 and reject 19.999999; mass-update tests use a different selection fraction to prove that only independent calibration supplies the log-mass increment.

A single-eligible-state fixture forces seven duplicated starts and succeeds. Additional tests cover an empty individual donor, an entirely empty bank with no cross-bank rescue, candidate tampering, checkpoint corruption, kernel identity mismatch, numerical-contract propagation, and the absence of level-specific generic branches.

## Validation result and scope

**Stage 7A PASS.** The full cheap DNS suite passed: **392 tests in 134.36 seconds**,
including 23 new automatic-builder, saved-integration, and dynamic toy tests.
No real LISA trajectory ran in pytest. Saved-artifact tests require the historical
`/tmp` bundles and skip when those are not installed; all ran in this validation.
All 140 protected Stage-6B source/artifact hashes still match, including the
historical checkpoints, candidate, selector constants, and frozen transition.
No level-13 checkpoint exists.

Results and hashes: [machine-readable validation](dns_automatic_ladder_validation.json).
The full four-level toy run is saved in `/tmp/dns-stage7a-toy-development/`;
the saved LISA integration result is `/tmp/dns-stage7a-integration/results.json`.
The regression log is `/tmp/dns-stage7a-tests.log`.

Automatic ladder construction is ready for the **next phase: inference
reconstruction validation**. That phase has not started. No LISA production,
manual level-13 construction, proposal/promotion/ESS/tolerance tuning,
posterior/evidence reconstruction, or final DNS termination was implemented.
The methodology remains frozen. STOP.
