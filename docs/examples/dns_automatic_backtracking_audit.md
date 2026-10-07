# Automatic DNS construction: level-backtracking architectural audit

**Recommendation B:** automatic construction is fixed-contour, while the validated joint DNS backtracking machinery is not being used. Integrating genuine level diffusion into construction is the most principled next development direction. This is an architectural recommendation, not a validated implementation or a production authorization.

This audit reads existing source and saved reports only. No trajectories, stochastic tests, LISA runs, or code changes were made. Stage 9A remains frozen **CASE C: increasing the fixed-contour budget from 1x to 4x does not establish deep multimodal robustness; no production change justified.**

## 1. Original frozen DNS target and transitions

The target is explicitly defined in [dns.py](../../blackjax/ns/dns.py), its `DNSLevels`, `DNSState`, `level_move_probability`, and `build_kernel`, and in [Stage 1](dns_stage1.md). Let `h(theta)=log L(theta)`, `ell_j` denote a log-likelihood threshold, `A_0` be the full prior support, and `A_j={theta:h(theta)>ell_j}` for `j>0`. Thus the question's likelihood constraint is expressed in log space by this implementation. For a normalized prior,

```text
X_j       = integral pi(theta) I[A_j](theta) dtheta       (true mass)
Xhat_j    = exp(levels.log_mass[j])                      (fixed estimate)
w_j       = exp(levels.log_weight[j])                   (positive weight)
a_j       = w_j / Xhat_j
log a_j   = levels.log_weight[j] - levels.log_mass[j]
C         = sum_(j=0)^J a_j X_j
q(theta,j)= pi(theta) I[A_j](theta) a_j / C
q(j)      = a_j X_j / C.
```

The weights need not sum to one. Estimated masses need not be exact. If `Xhat_j=X_j`, occupancy is proportional to `w_j`; otherwise it is proportional to `w_j X_j/Xhat_j`. `create_levels` requires `ell_0=-inf`, `log_mass[0]=0`, strictly increasing finite subsequent thresholds, finite nonincreasing log masses, and finite log weights. Level zero explicitly admits zero-likelihood prior points.

A walker contains `DNSState(particle, level_index)`, where the particle has position and cached log prior/log likelihood. The assigned index is distinct from the highest eligible level.

Each `build_kernel` sweep first applies parameter moves at the **currently assigned** threshold. The callback must preserve `pi(theta) I[A_j]`, with correct caches; its slice implementation uses the prior as vertical-slice density and the likelihood constraint as eligibility. The index is held fixed during these inner parameter steps. The sweep then applies neighboring-level moves, with the particle held fixed.

A level proposal chooses `j-1` or `j+1` with probability one half. Out-of-range proposals are rejected self-loops. For an eligible neighbor `k`,

```text
alpha(j -> k | theta)
 = min(1, a_k/a_j)
 = min(1, exp(log_weight[k]-log_weight[j]-log_mass[k]+log_mass[j])).
```

Downward moves are eligible by nesting, but can be rejected by Metropolis. Upward moves require `h(theta)>ell_(j+1)` as well as acceptance. Descending changes the constraint used by the **next sweep's parameter moves**. The walker can then traverse regions forbidden at the top contour, reach the other mode, and return upward once its likelihood permits. Both subkernels preserve the joint target; their ordered composition need not itself be reversible. With uniform weights and exact `exp(-1)` mass ratios, downward acceptance is `exp(-1)` and eligible upward acceptance is one.

## 2. What earlier multimodal validation established

[Stage 2](dns_stage2.md), [its validation example](../../examples/dns_multimodal_validation.py), and [the frozen tests](../../tests/ns/test_dns_multimodal.py) specifically distinguish backtracking recovery from a parameter move that jumps directly between components.

The toy has `theta~Uniform[-6,6]`, `log L=-0.5*((abs(theta)-1)/0.3)^2`, exact masses `X_j=exp(-j)` at levels 0–4, and uniform weights (`log_weight=0`). Level 1 is connected; levels 2–4 consist of two disconnected intervals. The same local slice direction of magnitude 0.5 is used in the diffusive and fixed-high arms. The fixed-high arm remains in its starting mode, while diffusion can visit levels 0 or 1 and recover high likelihood in the other mode.

The Stage-2 mechanism checks require more than 20 high-eligible mode switches per seed (21, 22, 23), more than five in each direction, round trips, and a minimum assigned level of 0 or 1 between opposite high-mode visits. They also check conditional moments, modal balance and starting-mode sensitivity. These are documented historical validation requirements, not newly executed results. No raw Stage-2 switch totals are newly asserted here. In particular, `mode_switch_summary` defines its Stage-2 high endpoints by **highest eligible level**, rather than necessarily by assigned high index.

The saved [Stage-3 report, “Mode switching and matched control”](dns_stage3.md) strengthens the evidence: its endpoints require **assigned level >=3**. The saved eight runs report 263–296 A-to-B and 266–300 B-to-A switches, all with intervening assigned level 0 or 1; matched fixed-high controls report zero switches. For example, float64 seed 101 records 283/279 directional switches, with backtracking-depth counts 387/175 at levels 0/1. These historical records demonstrate actual lower-level recovery and upward return on the controlled toy. They do not establish performance for Stage 9A or LISA.

## 3. Automatic construction uses a fixed contour

The decisive path is [dns_automatic.py](../../blackjax/ns/dns_automatic.py): `prepare_iteration` promotes each bank at `state.thresholds[-1]`, and `build_next_level` calls

```text
kernel.run(starts[name], state.thresholds[-1], schedules[name], config, contract)
```

for selection and calibration. `LadderState` stores particle histories, thresholds and masses, but **no per-walker level index**. The kernel receives a scalar contour, not the ladder prefix or a `DNSLevels` table. Returned histories must satisfy that contour and the exact retained-history shape.

[`FrozenPopulationKernel`](../../blackjax/ns/dns_population.py). Its `run` passes the same scalar `threshold` to every slice step and every reciprocal exchange, throughout burn-in and retention. `joint_decision` requires both proposed walkers to remain above that threshold. State is `DNSParticleState`, not `DNSState`; there are no level-index proposals, descending assignments, or lower-contour parameter sweeps.

The [model adapter](../../examples/lisa_dns_stage4/automatic_ladder_integration.py), `make_frozen_kernel`, binds the scalar-threshold parameter callback and population exchanges. It does not wrap `dns.build_kernel`. [Stage 9A's `BudgetKernel`](../../examples/dns_ladder_robustness_study.py) delegates all physical transitions to that `run`, scales burn-in/retained sweeps and thins output. Its instrumentation does not add level moves. The older [dns_levels.py](../../blackjax/ns/dns_levels.py) construction sampler likewise uses a scalar fixed contour; importing `dns` for particle types does not activate diffusion.

**Answer A in the walker's A/B alternatives:** every walker is permanently constrained to the current generating level J during a construction bank run. The construction loop subsequently raises the contour after acceptance; that is not within-run diffusion. The selected candidate `ell_(J+1)` is calibrated with samples generated at `ell_J`, not with samples already generated at the candidate level. A future supplied custom kernel could have different internals, but the audited adapters and Stage-9A execution do not.

## 4. Stage 9A is consistent with disconnected-contour trapping

The saved [Stage-9A report](dns_ladder_robustness_study.md) and [JSON](dns_ladder_robustness_study.json) show all budgets completing six ESS-gated levels, every walker supplying candidate survivors, and zero physical mode switching at levels 4–6 in both banks at every budget, including burn-in and the initial promoted-start transition. Level-5 selection ESS is nonmonotonic: 1575.37, 2793.75, 1705.21 for 1x, 2x, 4x.

The frozen toy likelihood depends only on `theta=3*u[0]`; the other coordinates are prior nuisances. Its separated peaks are near -2.4 and 2.5, with weights/widths .45/.45 and .55/.55. At theta=0, both normal components are exponentially suppressed relative to the peaks. The generating log thresholds for construction level 4 (`ell_3`) are about -0.95 to -0.96, increasing further at levels 5–6. Both peaks exceed these thresholds, while theta=0 is far below them. Thus the allowed set has at least two separated pieces in the likelihood-bearing coordinate; nuisance dimensions do not furnish a path around that gap.

A kernel whose permitted transitions cannot cross between those pieces is reducible on `A_J`. Repeating it more times cannot create a missing transition. If direct cross-component jumps have small but nonzero probability, more sweeps could eventually help, but the saved data establish no effective deep crossing through 4x. Disconnected geometry alone does not make **every** fixed-contour kernel reducible: a valid global proposal or nonlocal slice jump could move directly between components without accepting intermediate gap points. The Stage-9A record supports the fixed-contour trapping explanation; it does not prove an exactly zero cross-mode transition probability for this entire proposal bundle.

Compression ESS can be high even when walkers remain trapped in separate modes. Here similar within-mode tail fractions can give small indicator autocorrelation and balanced survivor fractions despite poor exploration of mode allocation. All walkers contributing survivors does not prove cross-mode ergodicity. Promotion can redistribute existing survivors between walker labels at iteration boundaries, but is not within-run movement through the missing region and cannot recover a mode absent from the pool. ESS reliability and cross-mode mixing diagnostics are different requirements.

## 5. Top-level conditional samples have the required measure

For a frozen joint target with positive `a_J` and positive true mass `X_J`,

```text
q(theta | j=J)
 = [pi(theta) I[A_J] a_J/C] / [a_J X_J/C]
 = pi(theta) I[A_J] / X_J.
```

This equality is exact. The visitation weight and estimated mass cancel even when `Xhat_J` is inaccurate. It is the constrained prior, not the posterior. For a candidate above the current threshold,

```text
P_q(h(theta)>ell_(J+1) | j=J) = X_(J+1)/X_J.
```

Consequently stationary **assigned-top-level** observations can supply selection quantiles and independent-bank compression calibration without changing the compression definition. An ergodic ratio of all top-level exceedance counts to top-level visit counts converges to this compression. Lower-level observations must not be pooled into the top-level quantile or its calibration denominator.

Retain observations at a consistent sweep phase whenever `j=J`, including consecutive stays and rejection self-loops. Taking only accepted upward arrivals or discarding repeated states generally introduces an additional selection law; the conditional derivation above does not justify that law. The stationary trace restricted to visits to J has the conditional invariant measure, but samples are correlated. Finite burn-in, promoted initialization, or stopping after a fixed number of top visits does not confer exact finite-sample stationarity or unbiasedness.

The relevant ESS is for the candidate exceedance indicator along the retained top-level trace. Long top-level residence, correlated excursions/return modes, and population coupling affect variance; physical sweeps, visits and independent effective draws are distinct counts. The present `tail_diagnostics` uses the maximum of IID, within-walker batch-mean and between-walker standard errors; it is explicitly approximate. Blocks must span relevant correlation scales. Ragged top-level histories and coupled walkers cannot simply be padded, flattened across walker boundaries, or treated as independent. The numerical **ESS >=20** gate can remain, but its input representation and correlation assessment must correctly describe the top-visit data. Round trips, backtracking depths and cross-mode returns remain separate required diagnostics.

## 6. A clean integration is possible, with explicit contracts

A mathematically clean construction iteration can freeze the existing prefix `0..J`, its calibrated mass estimates and positive weights; promote separate banks at `ell_J`; initialize assigned indices at J; run each bank's walkers with parameter moves at their assigned levels and valid index moves; retain only assigned-J observations; freeze the candidate from selection; and calibrate it with the other bank's top-level observations. Append the accepted threshold and calibrated compression only between runs, never by mutating a stationary run's target.

| Existing contract | Assessment for level diffusion |
|---|---|
| Separate selection/calibration banks | Preservable with disjoint RNG namespaces, storage and promotion pools; no inter-bank exchange. Shared/adaptively selected thresholds and previous calibrated masses already prevent a claim of unconditional statistical independence. Diffusion preserves the separation, not IID sampling. |
| Correct constrained-prior conditional | Exactly preserved by the frozen joint target and valid constituent moves. Equilibration remains an empirical requirement. |
| Compression definition | Unchanged strict-tail probability `X_(J+1)/X_J`, selected near `exp(-1)` and independently calibrated. |
| Pooled promotion | Bank-local pooled-with-replacement copying of top-level candidate survivors can remain, with donor/duplicate provenance. Promotion supplies starts, not independent equilibrium draws. |
| ESS >=20 gate | Can remain numerically unchanged; top-visit correlation, population dependence and sufficient blocks need explicit treatment. |
| Restartability | Possible, but not supplied automatically by the current checkpoint schema. |

This is not a drop-in replacement for the present scalar-contour `run`. Its interface would need the frozen prefix and weights. The existing `(retained, walker)` array contract must be reconciled with unequal visit counts. A future protocol must predeclare physical cost caps, burn-in, collection/termination rules and failure on insufficient top-level information, without retrying until a favorable gate occurs. Fixed counts per walker can support rectangular histories; their stopping behavior and cost still need validation.

For continuation within an iteration, checkpoints would need every particle and assigned index, exact prefix/weights, all RNG states or derivable counters for parameter/level/exchange moves, visit buffers/counts, sweep phase and durable candidate/calibration state. At completed iteration boundaries, fresh runs can reconstruct indices and streams by a frozen recipe, but that recipe and weight identity must be saved. Current particle-only checkpoints cannot resume an arbitrary mid-run joint state. Restartability is therefore preservable in design, not yet implemented or validated.

## 7. Available level weights

The original core accepts caller-supplied **fixed** positive weights; it does not derive adaptive visitation weights. Stage 2 uses `log_weight=zeros`. The Stage-3 protocol freezes uniform weights for its subsequent joint run. The earlier LISA frozen smoke adapters also use zero log weights; see `frozen_inputs` in [run_production_smoke.py](../../examples/lisa_dns_stage4/run_production_smoke.py). No such smoke run was executed in this audit.

During automatic construction, the accepted prefix already supplies all `ell_0..ell_J` and `log Xhat_0..log Xhat_J`. Together with the existing uniform-weight convention, it fully specifies

```text
log_weight_j = 0
log a_j      = -log Xhat_j,  j=0..J.
```

These inputs are enough for a controlled development experiment with weights frozen for each construction iteration. True occupancy would be proportional to `X_j/Xhat_j`, approximately uniform when calibration is accurate. They provide no guarantee of efficient round trips or sufficient top visits, especially with inaccurate cumulative masses or a long ladder. The newly proposed candidate has no independently calibrated mass yet; it need not be included in diffusion to estimate its compression. No adaptive production-weight rule is proposed here.

## 8. Reconstruction methodology remains separate

[Stage 7B's derivation](dns_inference_reconstruction.md), [dns_reconstruction.py](../../blackjax/ns/dns_reconstruction.py) and `inference_records` in [the reconstruction adapter](../../examples/dns_inference_reconstruction.py) exclude construction/adaptation samples from the authoritative estimator. After freezing the ladder, fresh separately seeded fixed-contour streams target

```text
p_j(theta) = pi(theta) I[A_j]/X_j
alpha_j    = n_j/N
S(theta)   = sum_j alpha_j I[A_j]/X_j
q_mix      = pi(theta) S(theta)
r(theta)   = L(theta)/S(theta).
```

For exact masses and stationary stratum marginals,

```text
E[(1/N) sum_(j,i) r(theta_ji) f(theta_ji)]
 = sum_j alpha_j integral p_j(theta) L(theta)f(theta)/S(theta) dtheta
 = integral pi(theta)L(theta)f(theta) dtheta.
```

Evidence is this estimate with `f=1`; posterior weights normalize the `r` values. All eligible levels enter the denominator. The identity depends on the fresh stratum targets and allocations, not the transition architecture that constructed the ladder. Correlation affects variance; estimated masses remain a plug-in approximation and normalized posterior estimates retain finite-sample ratio bias.

Changing construction to level diffusion can change the selected ladder, its calibrated masses and starting provenance. It does not change this MIS derivation if reconstruction still uses fresh fixed-contour streams, includes the prior stratum, and excludes construction observations. Evidence, posterior and shell-reconstruction methodology therefore remain intact. Existing finite reconstruction mixing limitations also remain: better construction alone does not guarantee fresh fixed-contour streams recover missing modes after promotion.

Feeding unfiltered joint trajectories into Stage-7B records would require a different measure/estimator and is expressly rejected by the current record contract. That is not the proposed direction. If construction kernel identities/checkpoints change, future adapters must distinguish the construction and reconstruction kernels while retaining pinned provenance; methodology preservation is not an assertion of unchanged byte-compatible APIs.

## 9. Population exchange and alternative directions

The LISA reciprocal source-exchange move swaps exactly one labeled six-coordinate source block between two walkers, evaluates both proposed states, and commits or rejects both jointly. Its acceptance includes the prior ratio and exact reverse/forward state-dependent selector correction. At present both walkers have the same scalar contour, and both proposed likelihoods must exceed it.

Level diffusion complements this move. Exchange can transfer useful source configurations between walkers; diffusion lets a walker relax its likelihood constraint, move under lower contours, and return. Neither mechanism replaces the other's purpose. To preserve the product augmented target with independently assigned indices, an exchange that leaves indices fixed must test each proposed recipient against **its own assigned threshold**; the level coefficients cancel in that theta-only acceptance ratio. Restricting exchanges to a valid same-level pairing is another possible construction, but any state-dependent pairing law needs its own balance argument. The current common-threshold exchange cannot be silently reused for walkers at different levels. Lower-level exchanges also require valid full-prior support handling at level zero. These are integration obligations, not changes made here.

| Direction | Architectural assessment |
|---|---|
| A. More fixed-contour sweeps | Repeats the same constrained transition graph. Stage 9A supplies no robust deep-mixing improvement through 4x. Cannot repair reducibility. |
| B. New global cross-mode theta proposal | Can legitimately cross disconnected components if it preserves the constrained prior and has sufficient acceptance. Needs a new proposal/invariance and efficiency validation; geometric disconnection does not rule it out. |
| C. Genuine DNS level backtracking | Uses the already-defined joint target and historically validated lower-level recovery mechanism. Directly addresses the absent architectural capability without requiring a new model-specific global jump. Needs integration and finite-mixing validation. |

The most principled single next development direction is **C in this comparison: integrate genuine DNS level diffusion into automatic construction**, with top-level conditional collection and separate banks. This supports **recommendation B in the requested final classification**. Lower-level connectivity and actual traversal are still required; level moves alone are not a guarantee of recovery in every model. Stage 9A remains CASE C, not evidence for a production budget or for successful diffusive construction.

## Final answers

1. **Does automatic construction currently backtrack in level space?** No. The audited automatic population kernels use one fixed generating contour and have no assigned level index.
2. **Can `q(theta | j=J)` provide the required constrained-prior samples?** Yes, exactly at stationarity for positive mass; use properly collected assigned-J observations and account for dependence and finite equilibration.
3. **Does Stage 9A support a disconnected-contour explanation?** Yes. Deep contours separate the toy's modes and saved runs show no crossing. It does not prove every fixed-contour proposal has zero crossing probability.
4. **Would level diffusion preserve reconstruction methodology?** Yes, if confined to construction and reconstruction continues with fresh fixed-contour strata and the existing MIS measure. Mixing and calibrated-mass uncertainty remain limitations.
5. **What is the single most principled next step?** Develop the integration of the existing joint DNS backtracking machinery into automatic construction under a separately frozen development protocol. Nothing is implemented or run by this audit.
