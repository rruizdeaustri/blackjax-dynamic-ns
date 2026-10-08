# Stage 10A: gb_0004 fixed-ladder reconstruction

**CASE D — RECONSTRUCTION / NUMERICAL FAILURE.**

The single preregistered experiment suffered a process-level SIGBUS crash during R1 L1 burn-in. R1 L0 completed with its strict-contour and deterministic cache audit passing; R2 and the required L6 fresh-process restart were not reached. No complete thirteen-stratum inference exists, so no posterior/evidence stability conclusion is available. The cause of the signal is unresolved; no recorded cache or contour violation preceded it. The run is CASE D under the frozen deterministic reconstruction-failure criterion and has not been retried.

Exactly one experiment was launched under the [frozen design](lisa_gb0004_fixed_ladder_inference_design.json) and [checksum](lisa_gb0004_fixed_ladder_inference_design.sha256). The [JSON report](lisa_gb0004_fixed_ladder_inference.json) records all partial/completed levels, process provenance, numerical outcomes and the paths, sizes and SHA256 hashes of every external artifact. No stochastic retry occurred.

## Completion and numerical evidence

| Set | Completed frozen levels | All thirteen complete? | Saved completed retained events |
|---|---:|---|---:|
| R1 | 1/13 | no | 8192 |
| R2 | 0/13 | no | 0 |

Failure: `ProcessSignal: Primary Python process terminated by SIGBUS (signal 7); shell exit status 135. R1 completed L0; last flushed L1 progress is 64/1536 sweeps. No stochastic retry.`.
Recorded status: **RECONSTRUCTION-FAIL; no recorded numerical/cache contract violation**. The terminating signal and process identity are retained; SIGBUS provided no Python exception traceback. The authorized run stopped; no alternate seed, retry, budget extension or kernel correction was performed.

The live population wrapper records every completed post-slice/post-exchange occupation state, caches, proposal component/labels/cost, actual slice input keys, exchange Gumbels/MH uniforms and reciprocal decisions. Full retained blocks are separately serialized after successful level completion. Partially allocated NPY arrays are interpreted only up to each recorded completed-sweep count; unwritten storage is never counted as samples.

The model, configuration, existing parameter/exchange kernels, estimator, CPU/x64 contract and frozen ladder remained unchanged. All design-protected hashes and the frozen driver/design hashes passed the post-run read-only audit. Historical construction and NSS posterior samples supplied no starts or estimator events.

## Reconstruction and evidence

The required thirteen-stratum reconstruction was incomplete. No authoritative R1, R2 or pooled MIS estimate was computed from the partial experiment. Consequently log Z, normalized posterior weights, posterior ESS, deepest-level fractions, L8–L12 proposal-mixture sensitivity and the 128 posterior mass-sensitivity reductions are **unavailable**, not zero. The preregistered perturbation definition remains frozen; no replacement analysis of incomplete strata is used to rescue the run.

L0 gives full prior support: the unchanged MIS formula is not mathematically truncated at L12. Finite proposal efficiency above the deepest contour remains an empirical concern. No missing deep integral is assumed zero; no raw L12-origin fraction supplies a termination or convergence bound. The BayesLISAx amplitude/phase integration convention drops normalization constants. Any log Z is only under the implemented likelihood convention; no externally normalized physical Bayes factor is established.

Monte Carlo R1/R2 variability, recorded-mass sensitivity, deepest-stratum/proposal-mixture sensitivity and likelihood normalization are separate issues. They are not combined into a calibrated uncertainty. Neither available-prefix stability nor restart equality establishes statistical convergence.

## Posterior, restart and disposition

K remains fixed at nine. The planned six physical coordinates are f0, fdot, iota, psi, longitude and latitude, with the historical normalized six-coordinate Hungarian reference assignment and existing angular periods. Saved historical catalogues are matching references only. Marginal reconstruction amplitudes/phases are fitted quantities, never directly sampled posterior intervals.

No reproducible nine-source scientific posterior conclusion or perturbation robustness can be inferred from this incomplete experiment. The required fresh-process L6 restart was not reached.

The earlier scoped Stage-6B numerical/population/promotion results and Stage-7B toy estimator validation remain intact. They do not validate this experiment's finite-time conditional stationarity or an end-to-end LISA inference result. No L13 was constructed; no automatic ladder construction, level diffusion, automatic termination, scientific budget tuning or likelihood-normalization change occurred.

**STOP.** No rerun after classification.
