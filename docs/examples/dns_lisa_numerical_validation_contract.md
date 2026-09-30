# LISA Stage 6A numerical validation contract

**Stage 6A PASS.** The authoritative result is
`/tmp/lisa_dns_stage6a_numerical_audit/results.json` (`decision = PASS`,
`errors = []`). This document resumes from that saved result. The CPU
reproducibility audit was not rerun, and no LISA likelihood was evaluated
while preparing this document.

## Historical CPU/x64 provenance and tolerances

The saved audit records eleven historical reports across Stages 4V, 4W,
4X, 4Y, 4Z, 5A, and 5B. All identify `[CpuDevice(id=0)]`. It records
SHA-256 hashes for the seven corresponding construction/validation runners,
the model/configuration, and protected inputs. Those seven runners use
`rtol=0`, prior `atol=1e-9`, and likelihood `atol=1e-7` for cache
recomputation checks. These are historical tolerances, not tolerances
introduced to accommodate the aborted GPU attempt.

The largest historical absolute prior/likelihood discrepancies were:

| Historical reports | Prior | Likelihood |
|---|---:|---:|
| 4V selection | 5.684341886080802e-14 | 2.028536982834339e-08 |
| 4V calibration | 2.842170943040401e-14 | 9.851646609604359e-09 |
| 4W through 5B, all nine reports | 0 | 0 |

The saved Stage-6A workers explicitly verified CPU execution and enabled
JAX x64. Both recorded JAX/JAXlib 0.10.0 and NumPy 2.4.4, using
`/r5/home/rruiz/software/miniconda3/envs/blackjax-ns/bin/python`.
The recorded `k9.json` configuration SHA-256 is
`bfafd9e8784bca263bd7298e10491d0010d27483222358b84bd1b4a877536670`.

## Frozen numerical contract

| Requirement | Frozen value |
|---|---|
| Backend | `cpu` (`JAX_PLATFORMS=cpu`) |
| JAX precision | `JAX_ENABLE_X64=1` |
| Coordinates and scalar dtype | `float64` |
| Relative tolerance | `rtol=0` |
| Prior recomputation tolerance | `atol=1e-9` |
| Likelihood recomputation tolerance | `atol=1e-7` |
| Copied coordinates | Byte-exact from source; no tolerance |
| Copied scalar caches | Byte-exact from source; no tolerance |
| Recomputed scalar values | Numerical agreement within the above tolerances |
| Finite checks | Required for coordinates, scalar caches, recomputed scalars, and the fixed contour |
| Contour membership | Both cached and recomputed likelihood must be strictly greater than the applicable contour |

Copy identity and recomputation agreement are separate requirements.
Bitwise-exact repeatability is an observed audit result, not a replacement
for the frozen numerical recomputation tolerances. Backend/x64 must be
checked from the execution environment, not inferred from matching scalars.
The NumPy-only [contract helper](../../examples/lisa_dns_stage4/numerical_cache_contract.py)
checks coordinates, dtype, finiteness, numerical cache agreement, and strict
membership; the caller must separately establish byte-exact scalar-cache
copy provenance from the source.

## Saved Stage-6A audit statistics and repeatability

The audit covered **1,153 unique saved states**: all 16 Stage-5B starting
states plus 803 selected retained entries from the selection bank and 334
from calibration. Retained coverage included all strict candidate
survivors, nine evenly spaced retained indices per walker, and each
walker's likelihood minimum and maximum, deduplicating indices within
each walker. Candidate survivors numbered 753 in selection and 266 in
calibration. Selection walker 6 had no candidate survivors but still
contributed ten retained audit entries.

The saved first CPU process performed two evaluation passes; a fresh CPU
process performed one. Recorded scalar evaluations per quantity were
2,320 and 1,160 respectively, including padded batches. These are costs
of the existing audit, not new evaluations during this documentation step.

| Saved comparison | Prior and likelihood result |
|---|---|
| First process, pass 1 versus saved caches | Bitwise exact |
| First process, pass 2 versus saved caches | Bitwise exact |
| Fresh process versus saved caches | Bitwise exact |
| Same-process repeatability | Bitwise exact |
| Fresh-process repeatability | Bitwise exact |

For both quantities in every comparison, maximum absolute difference,
maximum relative difference, median absolute difference, and 95th-percentile
absolute difference were **zero**, with **zero failures**. Finite and strict
contour checks passed: all audited states exceeded
`ell_12 = -108187.99712765554`; candidate survivors also exceeded
`ell_test = -107123.66036615883` under cached and recomputed likelihoods.

## Why the aborted GPU validation remains invalid

The [earlier pooled-promotion LISA attempt](dns_pooled_promotion_lisa_validation.md)
used GPU despite the historical CPU provenance and replaced numerical cache
agreement with bitwise recomputation equality. It stopped during initial
selection-cache verification, **before sweep 1**. Calibration sampling never
started, and no mixing or retained-trajectory validation was completed.

Its maximum scalar discrepancies were prior
`5.684341886080802e-14` and likelihood `3.6816345527768135e-08`.
Both were within the historical numerical tolerances. **The aborted GPU
validation nevertheless remains invalid because it used a different backend
and stopped before sweep 1.** Stage-6A PASS does not retroactively validate
that run, certify pooled-promotion LISA mixing, or change the Stage-5B failure.
The earlier attempt's saved artifacts and restart marker remain untouched.

## Preservation, cheap tests, and decision

The saved audit reports 132 protected files unchanged, zero sampling calls,
zero promotions, Stage-5B artifacts unchanged, and the ladder frozen through
level 12. Read-only verification on resumption matched all 132 saved
SHA-256 entries. The level-12 checkpoint digest remains
`0f0ac7c6294b4cd5e06de1fe28eb975f6fb590b9daf833f9ea084c6ed73ab0c4`
in Stages 4Z, 5A, and 5B. No level-13 checkpoint was found in the repository
or saved `/tmp/lisa_dns*` artifact directories.

The cheap tests use recorded arrays, synthetic/toy problems, and source
checks; they do not evaluate the LISA likelihood or advance saved LISA
walkers. The numerical contract tests specifically use a recorded 16-state
CPU fixture and exercise cache corruption, coordinate corruption,
nonfinite values, strict contour equality, backend/x64, and dtype rejection.

Resume verification: **350 tests passed in 91.65 seconds**, running
`tests/experimental/test_dns_gb_numerical_cache_contract.py` and the cheap
`tests/ns/test_dns*.py` / `tests/experimental/test_dns_gb*.py` regression
suite with `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. After the tests, all 332
saved files under `/tmp/lisa_dns*` matched the pre-test SHA-256 inventory,
with no additions or deletions; all 132 authoritative source hashes still
matched. Stage-5B artifacts and level-12 checkpoints remain unchanged,
and no level-13 checkpoint exists. No LISA sampling or walker promotion
occurred during this resumption; only the cheap synthetic/toy test cases
exercised those operations on test data.

**Stage 6A PASS.**

Exactly one Stage-6B pooled-promotion LISA validation is authorized under
the frozen CPU/x64 numerical contract. Stage 6B has not been run. This
authorization does not authorize construction of level 13. Stop here.
