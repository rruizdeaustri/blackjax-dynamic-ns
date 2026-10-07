# Stage 9C: diffusive automatic-ladder development

**DEVELOPMENT ONLY. CASE A.** Full ladder, ESS, analytic conditional/compression checks, representation, genuine excursions and deep comparison pass

The pre-sampling [design](dns_diffusive_ladder_design.json) and [SHA256](dns_diffusive_ladder_design.sha256) freeze all model, seed, schedule and acceptance choices. No tuning or trajectory retry occurred. Stage 9A remains CASE C and Stage 9B remains classification B.

## Measure and implementation

For each attempt, freeze the already-calibrated prefix 0..J, zero log weights, and log a_j = -log Xhat_j. Reuse dns.build_kernel with the unchanged constrained-prior slice callback and its original neighboring-level MH kernel. Then q(theta,j)=pi(theta) I[A_j] a_j/C and q(j)=a_j X_j/C, so q(theta|j=J)=pi(theta) I[A_J]/X_J exactly: the a_J/C factors cancel. Calibration measures X_(J+1)/X_J using only assigned-J observations from its own bank. Weights are frozen per attempt, not adapted within chains.

The fixed arm uses the existing dns_automatic.build_next_level scalar-contour orchestration, the same parameter callback and parameter-key layout, with its assigned level held at J. The diffusive arm binds the full prefix in an isolated adapter and delegates promotion, quantile selection, independent calibration, strict comparisons and ESS>=20 gates to unchanged code. This controlled one-coordinate toy does not invoke the 54-coordinate LISA population adapter.

Cross-walker exchange is disabled as the identity in both arms. Identity preserves the product joint target for every assigned-index vector, including mismatched levels. This isolates backtracking and makes no assertion that the common-contour exchange is valid for different indices. Independent walkers share only the frozen table; banks use disjoint namespaces, histories and promotion pools. Shared/adaptive levels do not imply unconditional IID samples.

## Frozen schedule and diagnostics

Each bank runs 32768 physical sweeps, with 1024 joint-sweep burn-in, no thinning, and the first 2048 post-burn assigned-top events per walker retained. Consecutive stays and self-loops are included; lower-level states are excluded. All eight quotas must be met or the attempt fails without retry. Both arms have the same physical cap and retained count; top-level counts differ. Per-walker visit order supplies the time axis for unchanged batch-mean/exceedance ESS. These ESS estimates and correlated-data CDF tolerances are approximate diagnostics; rare inter-mode correlation can outlast a block, so ESS is separately supplemented by physical mode/level traces and witnessed excursions.

Every event archive records position, cached prior/likelihood, assigned index, walker, event ID, physical sweep and promoted donor/index/source hash. Visitation reports include per-level/per-walker occupancy, up/down proposals and accepts, acceptance, top visits/retention, round trips, physical mode occupancy/switches and explicit top->connected-lower->different-mode->top witnesses. Analytic checks add candidate-survivor counts and between-walker standard errors.

The new toy is Uniform[-8,8] with log L=-0.5*((abs(theta)-1.65)/0.38)^2. For radius r=0.38 sqrt(-2 ell), the contour is the union of intervals around +/-1.65 clipped to prior support; overlapping intervals merge. Truth is computed by exact interval lengths, polynomial integrals and clipped-linear CDFs, independently of MCMC. Mode masses are symmetric. Six candidates are requested; deep generating indices 3,4,5 must be disconnected.

## Saved development results

| Method | Candidate level | Selection ESS | Calibration ESS | Top switches selection/calibration | Conditional checks | Compression checks |
|---|---:|---:|---:|---|---|---|
| fixed | 1 | 951.2200109282717 | 1687.7775352376204 | 1249/1263 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| fixed | 2 | 7634.873984379096 | 13104.493910121797 | 0/0 | True/False | {'selection': True, 'calibration': True, 'log_mass': True} |
| fixed | 3 | 13648.870872866493 | 14941.902727330777 | 0/0 | False/False | {'selection': True, 'calibration': True, 'log_mass': True} |
| fixed | 4 | 14721.771498571528 | 15530.979784793255 | 0/0 | False/False | {'selection': True, 'calibration': True, 'log_mass': True} |
| fixed | 5 | 16383.999999999998 | 11177.745586284316 | 0/0 | False/False | {'selection': True, 'calibration': True, 'log_mass': True} |
| fixed | 6 | 16383.999999999998 | 16384.0 | 0/0 | False/False | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 1 | 951.2200109282717 | 1687.7775352376204 | 1249/1263 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 2 | 15438.207904198352 | 15051.49204406365 | 748/707 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 3 | 14659.357634112795 | 16384.0 | 466/431 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 4 | 14414.607702150664 | 15663.72754901008 | 384/351 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 5 | 16383.999999999998 | 14815.584450402148 | 274/275 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |
| diffusive | 6 | 7877.115483766294 | 16384.0 | 257/258 | True/True | {'selection': True, 'calibration': True, 'log_mass': True} |

Full conditional CDF/moment/mode checks, compression and log-mass estimates against truth, level visitation, witnesses and predeclared deep comparisons are in the [JSON report](dns_diffusive_ladder_development.json). Compression ESS can be high while walkers remain in separate modes; all walkers contributing survivors does not prove cross-mode ergodicity.

## Restart and isolation

The completed level-3 checkpoint preserves both banks and links exact terminal joint states, indices, keys, exchange RNG state (identity/unconsumed), weights, counters and histories. A selection-chain snapshot at attempt 4 after sweep 8192 additionally preserves the partial trace/collection and exact PRNG keys. The child loads level 3 and restores that midpoint, then continues attempts 4–6; no restart from a fresh random trajectory. Each subsequent attempt performs the prescribed bank-local promotion and independent counter-derived stream initialization.

Restart passed: **True**. Genuinely new process established: **True**. Parent Popen receipt, child PID/PPID, independently generated process-start UUIDs, timestamps and full command are recorded. Exact comparisons cover thresholds, masses, construction records, analytic summaries, complete physical traces, retained events, final joint bundles and automatic checkpoint payloads.

Protected source hashes, including original parameter/index kernels, compression/promotion/gates, CPU/x64 contract and dns_reconstruction.py, were verified unchanged. No reconstruction module is imported or called by this driver; no construction observations are exported as reconstruction records. Stage-7B fresh fixed-contour MIS remains separate and unchanged. No termination criterion, LISA or production run occurred.

## Disposition

Independent held-out validation justified: **True**. This report authorizes no execution of that validation; CASE A supports one separately frozen independent validation, while other cases do not. No further run or tuning is part of Stage 9C.

## Independent saved-artifact verification

Six deterministic contract tests passed before design freezing; they ran no stochastic trajectories. A subsequent saved-data audit verified all 393,216 retained events against their exact assigned-top trace indices, including repeated states, and tied every reported excursion witness to the raw assigned-index and position trace. The midpoint snapshot preserved ragged collection counts [1853, 2048, 1861, 1561, 1640, 1731, 1871, 1603] and exact joint keys/state; no missing observation was padded. All 31 fresh-process numerical comparisons passed. Parent PID 3 launched child PID 39, whose recorded parent PID is 3; independent startup UUIDs and timestamps plus the Popen receipt establish the new process.

| Diffusive check | Worst observed | Frozen limit |
|---|---:|---:|
| Conditional KS | 0.03284581 | 0.08 |
| Calibration compression absolute error | 0.00952678 | 0.055 |
| Calibrated log-mass absolute error | 0.04297876 | 0.30 |
| Exceedance ESS (minimum) | 951.220 | >=20 |
| Top visits per walker (minimum) | 4485 | >=2048 |

At deep generating levels 3, 4 and 5, collected-top mode-switch totals are respectively 384/351, 274/275 and 257/258 for selection/calibration, versus 0/0 at each level for the fixed arm. Corresponding witnessed full-prior excursions returning in the other top mode are 1123/1087, 707/734 and 548/546. Every analytic mode remains represented in both diffusive banks at every generating level.

Implementation: [isolated adapter](../../blackjax/ns/dns_automatic_diffusive.py), [development driver](../../examples/dns_diffusive_ladder_development.py), and [deterministic tests](../../tests/ns/test_dns_automatic_diffusive.py). Raw immutable design, launch receipts, logs, checkpoints and event/physical-trace arrays are saved under `/tmp/dns-stage9c-development`; the complete numerical summaries and provenance are included in the JSON report. The runtime path is local temporary storage, so long-term reproducibility requires retaining these artifacts alongside the committed reports.
