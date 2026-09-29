# Final pooled-promotion LISA validation (2 of 2)

**FAIL — the validation did not complete.** Descriptive classification: **D, validation-integrity failure; mixing was not measured.** The selection run stopped during initial cache verification, before any transition. Calibration sampling never started. There was no retry, seed change, burn-in extension, contour adaptation, or level creation.

## What failed, and what did not

I launched the real-model check on GPU, although the saved Stage-5B reports identify `CpuDevice(id=0)`. I also strengthened the copied runner’s cache checks from its existing numerical tolerances to bitwise recomputation equality. These were validation-harness deviations, not changes to promotion or the population transition. I should have matched the historical numerical execution environment.

The promoted positions, prior caches, and likelihood caches were copied **exactly** from the correct eligible retained entries. All starts were finite and strictly above ell_test. The first direct GPU recomputation disagreed with three of eight saved selection prior caches: maximum absolute prior difference **5.684341886080802e-14**; maximum likelihood difference **3.6816345527768135e-08**. The prior equality assertion stopped execution. The likelihood difference was recorded before that assertion; no subsequent likelihood-equality assertion or transition executed.

Both discrepancies are within the historical runner’s tolerances (prior 1e-9; likelihood 1e-7). Their scale and the CPU/GPU difference are consistent with floating-point evaluation differences, but their cause was not tested by another model evaluation. This is **not evidence that pooling violates invariance, support, bank independence, or mixing**. It is also not a completed validation or permission to declare the rule LISA-validated. No check was relaxed after observing the failure.

## Fixed inputs and recorded random streams

The immutable ladder remains levels 0–12, with ell_12 = **-108187.99712765554** and log X_12 = **-10.418392711998465**. The unchanged Stage-5B candidate **-107123.66036615883** was used only as ell_test. Stage 5B remains officially failed. No log X_13, level-13 checkpoint, or level-14 quantity was created.

Only saved Stage-5B retained sweeps were eligible: 256 states per predecessor walker, 2,048 per bank. All indices below are zero-based; absolute trace index = retained index + 128. The pool uses generic Stage-3 time-major order, flat index = retained index × 8 + donor walker. Sampling calls the existing `dns_levels._survivors`, uniformly with replacement, copying the complete particle and caches.

| Bank | Retained | Eligible | Fraction | Contributing donors | Per-donor eligible counts (0–7) |
|---|---:|---:|---:|---:|---|
| selection | 2048 | 753 | 0.36767578125 | 7 | [134, 138, 131, 1, 122, 109, 0, 118] |
| calibration | 2048 | 266 | 0.1298828125 | 8 | [37, 46, 34, 29, 31, 19, 24, 46] |

Selection walker **6 has zero retained eligible states**, so the historical same-walker rule cannot promote all eight. This is the known input condition, not a new failure. Selection and calibration pools stayed separate; neither rescued the other.

All seeds and protected-input hashes were written to `preflight.json` before promotion. JAX promotion seeds: selection **2026093001**, calibration **2026093002**. Slice seeds: **960100–960107** and **960200–960207**. Independent PCG64 label/MH schedule seeds: selection **2026093003/2026093004**, calibration **2026093005/2026093006**. All 22 seeds are distinct. The declared budget was 128 burn-in + 256 retained sweeps per bank; the stop condition prevented execution of that budget.

| Bank | Promoted walker | Donor | Retained index | Absolute index | Starting logL |
|---|---:|---:|---:|---:|---:|
| selection | 0 | 7 | 207 | 335 | -106959.93639847376 |
| selection | 1 | 4 | 97 | 225 | -107089.42346159703 |
| selection | 2 | 1 | 144 | 272 | -104536.41922529046 |
| selection | 3 | 5 | 227 | 355 | -107030.3785782433 |
| selection | 4 | 5 | 223 | 351 | -106940.79261477459 |
| selection | 5 | 5 | 153 | 281 | -104755.56689080255 |
| selection | 6 | 5 | 229 | 357 | -105389.94605097032 |
| selection | 7 | 7 | 122 | 250 | -106929.37673758554 |
| calibration | 0 | 7 | 64 | 192 | -106807.23935693501 |
| calibration | 1 | 3 | 232 | 360 | -106335.50282945644 |
| calibration | 2 | 7 | 123 | 251 | -106834.66284096263 |
| calibration | 3 | 2 | 214 | 342 | -105344.17869913754 |
| calibration | 4 | 3 | 195 | 323 | -106935.47522564357 |
| calibration | 5 | 0 | 227 | 355 | -105244.12530518323 |
| calibration | 6 | 1 | 111 | 239 | -106490.01829988169 |
| calibration | 7 | 7 | 166 | 294 | -105650.09248267638 |

Selection donor multiplicities: **[0, 1, 0, 0, 1, 4, 0, 2]**. Unique promoted states: **8**; duplicates: **0**. No donor-diversity optimization occurred.

Calibration donor multiplicities: **[1, 1, 1, 2, 0, 0, 0, 3]**. Unique promoted states: **8**; duplicates: **0**. No donor-diversity optimization occurred.

Full SHA-256 hashes of all sixteen source states/caches, exact indices, and pairwise initial geometry are in the [numerical record](dns_pooled_promotion_lisa_validation.json). Hash input order is position bytes, prior scalar bytes, likelihood scalar bytes. Copies into the two promoted NumPy banks have separate storage; the RNG streams are independent. Promotion-donor identities are distinct from the post-promotion exchange ancestry labels, which would start at each new recipient.

## Available geometry and unavailable trajectory diagnostics

Initial distances reuse Stage-4I permutation-invariant assignment metrics: frequency-set RMS in Fourier-bin units, sorted-frequency mean distance, and normalized six-coordinate source-set RMS with angular wrapping. Each cell is minimum / median / maximum over the 28 walker pairs.

| Bank | Frequency-set RMS | Sorted-frequency mean distance | Six-coordinate source-set RMS |
|---|---|---|---|
| selection | 0.179059 / 81.301382 / 179.751251 | 0.179059 / 81.301382 / 179.751251 | 0.059511 / 0.253720 / 0.299154 |
| calibration | 38.797080 / 88.678441 / 170.380712 | 38.797080 / 88.678441 / 170.380712 | 0.151914 / 0.251214 / 0.294680 |

Neither bank had duplicate promoted starts. Duplicate decorrelation cannot be measured because burn-in did not run. Post-burn-in, retained-first-half, retained-second-half, and full-retained geometry and distance summaries are **unavailable**, as are per-walker means/spreads, KS, ESS, unique-state fractions, quantiles, and cross-bank retained agreement.

Exchange proposals, contour survivors, accepted exchanges, and directional transfers executed: **0**. Acceptance/contour fractions, MH ratio quantiles, and donor communication diagnostics are **not applicable**, not estimates of zero acceptance or failed communication. There are no saved operations to replay. Synthetic accepted/rejected/tampered-operation replay tests passed, but these do not substitute for actual trajectory replay.

Cost incurred by selection initial verification: **8 direct prior and 8 direct likelihood evaluations**. Slice/exchange costs: **0**. Calibration model evaluations: **0**. No newly sampled ell_12 states were produced.

Historical Stage-5A full-run median/max KS were approximately **0.05859 / 0.09766**, Stage-5B selection **0.1445 / 0.7852**, and Stage-5B calibration **0.1016 / 0.1523**. These saved results motivated the experiment; the stopped validation supplies no comparable new retained trajectory. Whether pooling avoids inherited selection-lineage pathology therefore remains unanswered for LISA.

## Integrity, tests, and terminal consequence

Read-only post-stop SHA-256 verification confirmed **36 protected files unchanged**, including every Stage-5B artifact, its candidate, the exact level-12 checkpoint/full prefix, generic promotion and constrained-slice sources, frequency selector/exchange sources, and the population-validation source. Both saved promoted banks were rechecked against their original Stage-5B entries without evaluating a model. No checkpoint was written in the new output directory and no level-13 checkpoint exists in Stage 5B.

The new runner has no construction, mass update, or checkpoint-writing calls. A persistent `sampling_started` marker prevents accidental restart. Source-equivalence tests verify the transition body against the frozen runner, explicitly accounting for the contour, output metadata, restart lock, and the stricter cache check that caused this stop.

Tests: **322 passed in 91.26 s**, covering `tests/ns/test_dns*.py` and `tests/experimental/test_dns_gb*.py`, including 10 new synthetic promotion/support/cache/independence/immutability/replay tests. Pytest executed no real LISA trajectory or model evaluation.

Artifacts: `/tmp/lisa_dns_pooled_promotion_validation/`; execution log `/tmp/pooled_lisa_run.log`. Implementation: [fixed-contour driver](../../examples/lisa_dns_stage4/pooled_promotion_validation.py), [read-only analysis](../../examples/lisa_dns_stage4/analyze_pooled_promotion.py), [cheap tests](../../tests/experimental/test_dns_gb_pooled_promotion_validation.py).

**Consequence:** the algorithm remains frozen at LISA level 12. The pooled rule is not certified by this LISA validation, although the toy validation remains passed. Do not modify the promotion rule in this study, retry this experiment, or construct level 13. **Manual methodology development ends here.**
