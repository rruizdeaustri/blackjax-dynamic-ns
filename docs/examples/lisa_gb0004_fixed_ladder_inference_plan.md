# LISA gb_0004 fixed-ladder inference plan

This is a read-only plan for one bounded scientific reconstruction experiment. No LISA problem was instantiated, likelihood evaluated, prior sampled, trajectory generated, or code changed while preparing it. The automatic-DNS methodology programme remains closed. No L13 construction, automatic extension, automatic termination, stationarity-threshold tuning, or production-ready automatic-DNS claim is permitted.

## Recovered frozen state

**The historical L0–L12 ladder is recovered exactly.** The authoritative files are `/tmp/lisa_dns_stage4z_population_level12/checkpoint_level12.json` and `/tmp/lisa_dns_stage4z_population_level12/levels.npz`. Their arrays match the requested values byte-for-byte after conversion to float64. The checkpoint copies in `/tmp/lisa_dns_stage5a_level12_validation/` and `/tmp/lisa_dns_stage5b_population_level13/` have identical file bytes and verified payload checksums. The frozen arrays in `/tmp/lisa_dns_stage6b_pooled_promotion_validation/preflight.json` match as well.

| Level | Frozen log-likelihood threshold | Frozen calibrated log mass |
|---|---:|---:|
| L0 | -inf | 0 |
| L1 | -113725.65889538352 | -1.0343179379627125 |
| L2 | -113448.99632193986 | -1.9996430044384734 |
| L3 | -113172.84824147604 | -3.118477930469643 |
| L4 | -112838.36560726895 | -4.172213954289457 |
| L5 | -112388.10901787043 | -5.0146408844420645 |
| L6 | -111915.6904902109 | -5.840204008542667 |
| L7 | -111344.96574156903 | -6.561075737117467 |
| L8 | -110809.82135735315 | -7.156607159127204 |
| L9 | -110252.99476697217 | -7.67544001580533 |
| L10 | -109780.28875081123 | -8.394305725653265 |
| L11 | -108868.60077896297 | -9.643899891984567 |
| L12 | -108187.99712765554 | -10.418392711998465 |

The checkpoint file SHA256 is `0f0ac7c6294b4cd5e06de1fe28eb975f6fb590b9daf833f9ea084c6ed73ab0c4`; its historical payload SHA256 is `df01390db08805ac1adf779773ed0a83a975a350b89bd29639c3b7fb6a5dc204`. The level archive SHA256 is `8bcdad7a099832ded0cc2a2c8234c2cef874379fd938b9a0e619e3af3c26ddc5`. These are exact recovery checks, not independent checks of the true prior masses. The recorded L12 mass is approximately `2.9877863229157394e-5`.

All 132 protected Stage-6A source hashes and all 140 Stage-6B protected file hashes match their saved manifests. The configuration is `/r5/home/rruiz/projects/sbi/BayesLISAx/automation/snakemake_lisa_gb/results/model_order/gb_0004/configs/k9.json`, SHA256 `bfafd9e8784bca263bd7298e10491d0010d27483222358b84bd1b4a877536670`. The model source is `/r5/home/rruiz/projects/sbi/BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py`, SHA256 `3c5c4f429142aa4f61b5131dc52a2ad666ff58ac11ce4e627caa38e3acc1a6a3`.

The model is the existing Sangria-observation gb_0004 configuration: Kmax=9, gates disabled, unordered source labels, amplitude/phase marginalization enabled in `integrate` mode, 54 latent coordinates, TDI1, MRDv1 noise and 512 frequency bins. The nominal band is [0.00183, 0.00185] Hz. The historical prior audit records the actual physical frequency bounds [0.001830003805175038, 0.0018500126839167937] Hz; retain these model-derived bounds rather than substituting the nominal endpoints. The data path is `/r5/home/rruiz/projects/sbi/lisa/data/LDC2_sangria_training_v2.h5`; the live preflight must record the data identity as well as the frozen model/configuration identities.

## Authorized scope

The [final methodology status audit](dns_methodology_status.md), [Stage-6A contract](dns_lisa_numerical_validation_contract.md), [Stage-6B validation](dns_stage6b_pooled_promotion_lisa_validation.md), and [Stage-7B reconstruction report](dns_inference_reconstruction.md) support reuse of:

- The CPU/x64 numerical contract: CPU selected before importing JAX/model code, x64 enabled, finite float64 positions/caches, `rtol=0`, prior `atol=1e-9`, likelihood `atol=1e-7`, strict cached and recomputed contour membership, and byte-exact copied coordinates/caches.
- The frozen parameter transition: all 54 direction scales pi/sqrt(3), fixed 20% global / 40% uniformly labelled single-source / 40% uniformly labelled unordered-source-pair mixture, `max_steps=10`, `max_shrinkage=100`.
- The reciprocal source-exchange population move: eight walkers, four disjoint exchanges per sweep, seven-round deterministic pairing, complete six-coordinate source-block copies, and exact joint prior plus reverse/forward state-dependent selector correction. Keep `f_star=0.0018409808` Hz and `sigma_f=2.558e-7` Hz. Every population in a sweep has one common fixed contour.
- Pooled-with-replacement promotion within a population's own history, with full copying, support and donor/duplicate provenance checks; no diversity optimization or rescue.
- The recovered immutable L0–L12 table as the specified proposal-stratum design and calibrated-mass input. Its historical acceptance is not a new proof of exact masses or conditional convergence.
- The unchanged Stage-7B deterministic-mixture reconstruction estimator and its record, weight, archive and numerical-validation contracts.

Not authorized: automatic ladder extension, construction of any new threshold or mass, automatic termination, deployment of the Stage-9F reliability gate, diffusive LISA population integration, or claims of production-ready automatic DNS. The new fixed-ladder LISA inference application is not itself already validated merely because its numerical components are.

## One exact proposed run

Execute two separately seeded reconstruction sets, R1 and R2, each containing thirteen fixed-contour population streams, one for every L0–L12. Their purpose is a predeclared independent comparison; neither is chosen as the preferred result after inspection. Use root seeds R1=`10600401`, R2=`10600402`, and mass-sensitivity seed `10600403`, with no seed search or retry.

| Allocation | Fixed value |
|---|---:|
| Levels per set | 13, L0–L12 |
| Walkers per level | 8 |
| Burn-in population sweeps per level | 512 |
| Retained population sweeps per level | 1024 |
| Thinning | None |
| Retained events per level per set | 8192 |
| Retained events per set | 106496 |
| Total authoritative retained events | 212992 |
| Total primary population sweeps | 39936 |
| Total primary walker sweeps including burn-in | 319488 |
| Diagnostic block size | 64 retained sweeps |
| Mass-sensitivity perturbations | 128 |

The reconstruction allocation comes directly from `/tmp/dns-stage7b-validation/design.json`: eight walkers, 512 burn-in, 1024 retained sweeps, equal fixed allocation and block64 diagnostics. It is the established baseline allocation, **not a derived minimum sufficient for LISA**, a precision guarantee, or an invitation to extend a problematic stream. No existing result establishes the sample count needed for reliable gb_0004 posterior precision. Fixing this inherited count answers the budget question without tuning from posterior outcomes. The Stage-6B 128/256 schedule validated a numerical benchmark and does not replace Stage-7B's reconstruction allocation.

Freeze the table, allocations, model identity, stream derivation, diagnostic quantities and output definitions before the first model evaluation. At each level run only the existing common-contour parameter/population kernel, never an automatic builder or level-index diffusion. Retain the post-population-sweep state at every scheduled retained sweep, including legitimate repeats/self-loops.

### Initialization and independence

For each set, initialize L0 from eight fresh draws of the existing model's matching prior sampler using a separate `SeedSequence([root, 0, 2])` initialization namespace. No historical posterior family or construction particle initializes these sets.

For j=1..12, use that set's own fresh retained L(j-1) reconstruction history as the pooled source of starts satisfying the already frozen threshold ell_j. Draw eight starts uniformly with replacement using the existing pooled-promotion operation and then perform the full fixed 512-sweep burn-in at ell_j. This does not select a threshold, estimate a mass, or construct a level. The operation is the validated promotion mechanism applied to inference-only histories; it is not a claim that this LISA initialization schedule has already demonstrated stationarity. An empty eligible pool stops the incomplete experiment without rescue, historical starts, additional draws, or an extended budget.

Use the existing `SeedSequence-v1` derivation `streams(root, j, 'calibration', 8)` for promotion, slice, label and MH substreams; here the legacy bank-name string identifies a RNG namespace only, not a new calibration bank. Its independent purpose namespaces are already defined in [dns_automatic.py](../../blackjax/ns/dns_automatic.py). Record and check the complete derived seed manifest for collisions before sampling; keep Threefry2x32 slice keys and PCG64 exchange schedules. Reuse no historical construction stream. R1 and R2 share only the frozen table/model and have separate prior draws, histories, mutable storage and all random streams.

This fresh-only initialization is stricter about construction-history separation than the original Stage-7B driver, which used saved calibration histories for starts. It preserves the Stage-7B estimator and fixed-stratum invariant targets; it does not reproduce that driver's initialization byte-for-byte. Promotion creates within-set genealogy and dependence across levels, so no IID or independent-strata claim is made. Separate random populations do not prove that either population is equilibrated.

**No sample used to construct or freeze L0–L12 enters initialization or the reconstruction estimator.** No Stage-5B/6B benchmark sample, historical NSS posterior sample, promotion copy or burn-in event is counted as a new reconstruction event. Fresh lower-level reconstruction events remain legitimate estimator inputs exactly once; copies used to start higher levels are excluded, while genuinely new higher-level retained events are counted under their new event IDs. Repeated coordinates do not imply repeated event identity.

Maintain separate complete `ReconstructionRecords` for R1 and R2. Each uses the unchanged target label `independent_fixed_contour_reconstruction`, original thresholds/log masses, origin level, walker, retained draw, unique set/level/draw/walker event ID and full stream/contract/provenance metadata. Each estimate is evaluated independently. Do not concatenate sets with colliding level/walker/draw triples into the validated record schema, and do not select or discard a set based on apparent posterior quality.

## Estimator and evidence scope

For each complete set use the unchanged [Stage-7B implementation](../../blackjax/ns/dns_reconstruction.py). With n_j=8192 and N=106496, alpha_j=1/13,

```
S_hat(theta) = sum_(j=0..12) alpha_j I[log L(theta)>ell_j] / Xhat_j
r(theta) = L(theta) / S_hat(theta)
Zhat = (1/N) sum r(theta_i)
W_i = r(theta_i) / sum r(theta_k).
```

Compute in log space through `reconstruct_evidence`; use its posterior expectation/quantile routines and preserve all eligible strata in each denominator. Do not replace MIS by shell quadrature, originating-stratum-only weights or a diffusive-trajectory estimator. Do not feed reconstruction diagnostics back into the ladder or masses. Primary estimates use the recovered calibrated masses without fitting to posterior results.

Including L0 gives full prior support. Thus this estimator targets the **total integral of the implemented likelihood**, not merely evidence below ell_12. Ending the proposal-stratum list at L12 does not mathematically delete the region above L12: a stream at L12 targets the whole set {log L>ell_12}, without an upper cutoff. There is no separate terminal remainder to add to the full-support MIS estimate.

However, proposal support is not reliable finite-time exploration. The unresolved terminal target integral

```
T_12 = integral pi(theta) L(theta) I[log L(theta)>ell_12] dtheta
```

can remain large or badly sampled. A small recorded Xhat_12 does not bound T_12 without a justified global likelihood envelope and credible masses. The existing saved audit contains best-family log likelihoods from approximately -90964 to -90495, well above ell_12=-108188; these are warnings about possible unresolved deep contributions, not likelihood upper bounds or new inference samples. No negligible-terminal-mass assumption is justified here.

Report the MIS estimate of T_12 using the same r values multiplied by the indicator, its observed fraction of Zhat, and its population-block sensitivity. This is a diagnostic estimate, not a validated upper bound. Report evidence allocation by originating stream separately from target-space shell/terminal contributions: the former reflects proposal allocation, while the latter partitions the target integral. Sum shell/terminal integrands evaluated with the same full MIS denominator; do not count the terminal contribution twice. Empty sampled shells or an apparently tiny terminal estimate must not be interpreted as proof of negligible unsampled mass.

The frozen BayesLISAx source's `loglike_marg` in `integrate` mode explicitly drops constants (`ll = ll - 0.5 * logdet`, with “constants dropped”). Therefore report **log Z in the frozen implemented marginal-likelihood convention**. The omitted normalization and amplitude-prior conventions have not been certified here as an absolute physical marginal likelihood. Do not silently restore constants, change the likelihood or shift frozen thresholds. A common constant cancels in posterior weights but not evidence. Absolute physical evidence and cross-K/model Bayes factors are outside this plan.

Scientifically defensible statements are the numerical plug-in MIS estimates in that pinned convention, their independent-set agreement or disagreement, observed terminal/shell contributions, weight concentration, conditional-sampling diagnostics and sensitivity to recorded mass uncertainty. They are conditional empirical results. Claims of fully converged total evidence, negligible terminal evidence, a certified tail/error bound, correct modal weights solely from ESS, or successful automatic termination are not supported. Adding/deleting available proposal strata changes the denominator and precision; cumulative `records.through(j)` comparisons are proposal-sensitivity diagnostics, not omitted-evidence bounds.

## Posterior outputs and permutation structure

Use the same scientific outputs already present in the gb_0004 configuration and historical BayesLISAx analysis:

- For all nine sources: f0 in Hz/mHz, fdot in Hz/s, iota, psi, ecliptic longitude lam and latitude beta, using the existing physical-coordinate decoder and exact prior conventions. Report weighted q05/q16/q50/q84/q95, appropriate means/variances and the established source-parameter correlation plots. Do not reinterpret the signed inclination convention.
- Frequency-set/order-statistic distributions, joint source catalogues and permutation-invariant frequency/whole-source comparisons already used in the historical source-permutation diagnostics. Preserve catalogue alternatives and source-assignment ambiguity.
- The existing marginal reconstruction quantities: fitted complex amplitudes, their magnitudes/phases, fitted A/E signal, residual A/E spectra, residual power/chi-square, log determinant and source_gain. These are fitted/conditional reconstruction diagnostics averaged or displayed under the shape posterior. They are not automatically full amplitude/phase posterior credible intervals. This plan introduces no new conditional amplitude sampler.
- Marginalized log-likelihood distributions and the existing historical catalogue-family comparisons as references only; historical samples have zero estimator weight.

With gates disabled and K=9 fixed, K_hard/K_soft are constant and there is no posterior source-number inference. Do not report source_gain as a Bayes factor or a posterior inclusion probability. Amplitude and phase are marginalized rather than sampled parameters.

Generate the physical decoder view with `sort_by='f0'` for presentation, permuting complete source blocks and all associated fitted amplitudes/phases/gains together. Do not sort just frequencies or change the sampler/prior. Frequency rank is an order statistic, not a permanent astrophysical identity: close-source rank crossings can exchange identities. Primary catalogue-level summaries remain permutation-invariant; any source-specific association must expose ambiguity and use the existing frequency/normalized six-coordinate assignment conventions. A matched catalogue view must not erase an alternative catalogue. Raw labels and exchange lineages are provenance, not physical source identities.

Handle circular parameters psi, lam and fitted phase with circular summaries and wrapped joint plots; ordinary endpoint-spanning linear averages/intervals can mislead. Use the existing periodicity conventions rather than introducing a new physical identification. Report prior-dominated or multimodal coordinates explicitly. All weighted posterior outputs are accompanied by R1/R2 and mass-sensitivity comparisons.

## Predeclared reconstruction diagnostics

These are descriptive diagnostics for the fixed experiment, not a new automatic conditional-stationarity acceptance algorithm. No R-hat/ESS threshold is tuned, no diagnostic launches extra sweeps, and no automatic stop criterion is introduced. Existing hard numerical/record contracts remain mandatory.

| Diagnostic | Predeclared calculation and interpretation |
|---|---|
| Contour membership | Strict cached and directly recomputed logL>origin ell_j, including all retained repetitions; verify L0 and all thirteen allocations |
| Numerical/cache consistency | CPU/x64 and fixed tolerances above; every post-slice/post-exchange outcome checked by the validated operation bundle; full retained-record recomputation and finite float64 checks; hard failure stops the experiment |
| Per-level information | Stage-7B `correlation_diagnostics` at block64 for logL, log prior, physical frequency order statistics and all six source-coordinate families; additionally evidence r and posterior residual integrands r(f−Ehat[f]); report observable-specific approximate ESS/SE and trajectories |
| Between-walker consistency | Walker means, full/half-window distributions, pairwise KS distances and existing permutation-invariant catalogue geometry; within-level first/second retained-half comparisons and R1/R2 comparisons; no IID KS significance and no eight-independent-walker convergence claim |
| Exchange provenance | Persist selected labels, pairing round, forward/reverse selector terms, joint prior/MH decisions, exact reciprocal block copies/restoration and accepted donor/lineage transfers; replay with saved schedule/arrays; label migration is not proof of physical modal mixing |
| Posterior effective information | Report Stage-7B weight ESS=1/sum W², maximum weight and entropy, separately from correlation-aware observable ESS/MCSE; weight ESS alone is not independent posterior sample count |
| Evidence by level | Per-origin contributions C_j=(1/N) sum_(origin=j) r, terminal integrand contribution, target-shell integrand contributions and cumulative fixed-prefix MIS comparisons; all sums in log space |
| Mass uncertainty | Exactly 128 predeclared Stage-7B logit-normal compression perturbations based on historical calibration SEs, with fixed samples/thresholds; recompute MIS weights, evidence and posterior summaries only |

Population-wide time blocks preserve contemporaneous exchange dependence, as in Stage 7B; eight interacting walkers must not be treated as eight independent replications. R1/R2 are the independent random populations conditional on the shared frozen inputs. Per-level ESS sums and between-level variance calculations remain approximate because promotion couples levels. Neither a small set difference nor a large approximate ESS establishes unavailable truth or certifies stationarity.

For mass sensitivity, r_j=exp(log Xhat_j−log Xhat_(j−1)); perturb logit(r_j) by independent Normal(0, [s_j/(r_j(1−r_j))]^2), map back with expit, and form log X'_j=sum_(k≤j) log r'_k, keeping X'_0=1. This is the existing Stage-7B sensitivity convention, not a new estimator or calibrated confidence model. Use the same frozen perturbation matrix for both sets so their mass response is comparable. Record q05/median/q95 shifts in logZ, terminal fraction and all posterior quantities.

The exact historical uncertainty sources are L1: `/tmp/lisa_dns_stage4_prior_smoke/report.json`, first ladder calibration; L2: `/tmp/lisa_dns_stage4_rejection/report.json`, `IID_level2.calibration`; L3–L12: the recovered checkpoint's ten calibration diagnostic rows. Their standard errors, in level order, are:

```
0.021153767292169143, 0.021460606692236043,
0.026300146597240077, 0.07195861880637901,
0.06637238993630247, 0.09563886525907439,
0.07611461904903624, 0.0756025451687297,
0.09000718315160197, 0.06913128436451788,
0.0496193565736982, 0.01979454229970753
```

Use only the accepted L2 IID calibration, not its earlier rejected MCMC estimate. Freeze/hash these sources in the run manifest. Their errors are approximate and do not capture all shared adaptive-threshold, genealogy, mass-error covariance or conditional-sampling uncertainty. The perturbation quantiles therefore measure sensitivity, not uncertainty coverage. No perturbation fits the ladder, updates primary masses, estimates a new deep mass or creates L13.

## One exact fresh-process restart check

Plan one checkpoint at the completed L6 boundary of R1, before any L7 initialization. Save the checksummed Stage-7B reconstruction archive for levels 0–6 using `save_records`, the separately pinned full L0–L12 table, immutable source/kernel/configuration identities, R1 root and `SeedSequence-v1` recipe, next level j=7, and the complete L6 retained population history needed for its own pooled promotion. Record process receipts and the numerical environment. Preserve exact dtype, shape, array bytes, event IDs and stream metadata.

The uninterrupted R1 reference finishes L7–L12 once. Exactly one fresh child loads the L6 boundary and reproduces L7–L12 from the same saved pool and counter-derived streams, with no rebuilt ladder, regenerated lower-level population, or new seed. Each level starts its seven-round pairing schedule at round zero exactly as in the validated fixed-contour run. Compare primary arrays/caches, retained events, promotion and exchange provenance, stream/level records, final MIS logZ/weights and posterior summaries exactly against the reference under the same CPU/x64 environment. Archive-container bytes need not be identical if their timestamps differ; checksums validate each archive, while equality concerns numerical/metadata payloads.

The child's retained events are replay evidence only: do not count them again or call them another independent reconstruction set. R2 supplies the scientific independent comparison. The single successful replay adds six level runs, 73728 walker sweeps including burn-in, beyond the primary budget. If sampling or comparison fails, record the first failure and stop without rerunning.

This uses already established completed-boundary replay, deterministic streams and checksummed reconstruction records. It does not claim that the existing `FrozenPopulationKernel.run` can resume mid-sweep: it has no such resume interface. No mutable within-level RNG snapshot or new checkpoint semantics is required. The inference-only boundary manifest/driver integration still needs review before execution; it has not already been validated on LISA. Stage-7B's historical fresh-process check reused reconstruction archives after a construction continuation, so the new live reconstruction replay remains an experiment to be checked, not a historical pass being assumed.

Restart equality demonstrates computational reproducibility, never posterior or evidence convergence.

## Interpretation frozen before sampling

These are scientific interpretations, not an automatic classifier with new stationarity cutoffs. Preserve every complete stream and diagnostic. Report agreement/disagreement against each observable's population-block variability and the full fixed perturbation range; do not invent a numerical pass threshold after viewing results.

| Outcome | Meaning and allowed report |
|---|---|
| A. Posterior stable across independent sets and mass perturbations | R1/R2 catalogue alternatives, source summaries and residual diagnostics are compatible within the reported Monte Carlo variability, and mass perturbations do not materially change the stated scientific interpretation. Report empirical fixed-ladder posterior robustness. This is not proof of exploration completeness or converged evidence. |
| B. Posterior stable but evidence sensitive to unresolved deep contribution or masses | Normalized posterior summaries remain compatible while logZ, terminal contribution, weight concentration, available-prefix comparisons or mass response reveal evidence instability. Report posterior robustness with qualified implemented-convention evidence; do not declare small terminal remainder. Missing deep contributions cannot be bounded solely from these diagnostics. |
| C. Posterior itself unstable | Independent sets, temporal windows or perturbations produce incompatible physical catalogues/source uncertainties, or too little effective information supports a stability judgment. Report the instability/insufficient information and avoid a single definitive posterior catalogue. Label rearrangement alone is evaluated using the permutation-aware views. |
| D. Numerical/reconstruction failure | A mandatory cache/support/finite/record/weight/checksum check fails, a fresh eligible pool is empty, the fixed allocation cannot complete, or the required restart equality fails. Identify incomplete initialization separately from a numerical defect; do not publish a validated inference result or replace the failed stream. |

Hard failures take precedence; instability is not cured by discarding R1 or R2. Posterior stability and evidence sensitivity can coexist. Neither A nor B promotes the closed automatic-ladder programme to validated status. No branch authorizes more sweeps, additional sets, modified kernels, new levels, tuned diagnostics or automatic termination.

## Execution boundary

The next planned work is exactly this bounded two-set, thirteen-stratum fixed-ladder reconstruction and one R1 L6 fresh-process replay, with 128 frozen mass-sensitivity reductions. No existing example should be run blindly: `examples/dns_inference_reconstruction.py` also builds toy ladders, and the historical LISA runners perform construction or locked one-shot benchmarks. Reuse the frozen operation bundle and Stage-7B estimator through a dedicated inference-only orchestration layer, with the above manifest and boundary replay; no scientific kernel, estimator or automatic termination change is part of the plan. No turnkey gb_0004 driver for this exact schedule currently exists, and none was implemented in this planning stage.

The plan itself authorizes no execution. STOP after documenting it. L0–L12 are immutable, the estimator remains separate from construction, and total evidence convergence remains an open scientific question for this fixed-ladder experiment.
