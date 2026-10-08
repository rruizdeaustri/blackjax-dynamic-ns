# Stage 9F: conditional-stationarity reliability gate development

**CASE B — TOO MANY FALSE REJECTIONS.** The single frozen composite detects the historical Stage-9D calibration failure but rejects **8/8 exact-joint-stationary controls (100%)**, **8/8 promoted-start controls**, and the saved successful Stage-9D selection sample. No final held-out ladder validation is justified. STOP methodological expansion; no automatic production-readiness claim.

## Scope and predeclaration

Only the two saved candidate-5 Stage-9D traces and all sixteen saved Stage-9E traces were read. No trajectories, kernel modifications, extra burn-in, replicas, LISA, termination, or production runs occurred. This model and every control are development data. Saved-trace paths and SHA256 hashes appear in the [JSON report](dns_conditional_reliability_gate.json). Their temporary storage must be retained separately for reproducibility.

Thresholds and the composite structure were written into the JSON before executing classification. The report preserves the canonical frozen-design SHA256; the [diagnostic-only driver](../../examples/dns_conditional_reliability_gate.py) records its hash and ArviZ version. No thresholds were retuned after classification. The decision priority was also predeclared: reject CASE A if half or more stationary controls fail. Only one production-candidate composite was evaluated; standalone diagnostic counts are descriptive comparisons, not alternative rules.

## Exactly one candidate composite

Require all of:

1. Original compression ESS ≥20, using the unchanged Bernoulli variance / maximum IID, block64 and between-walker standard-error convention.
2. All eight walkers supply their original first 2048 assigned-top occupation events after the unchanged 1024-sweep cutoff, including self-loops and repeated states.
3. For log likelihood, nonconstant log prior and at most two deterministic theta projections, rank-normalized folded split R-hat <1.01, bulk ESS ≥800 and tail ESS ≥800.
4. No Holm-adjusted rejection at familywise alpha .05 among paired walker physical-window mean comparisons for every coordinate, likelihood, nonconstant prior and those projections.

The two projections are sum(theta)/sqrt(d) and alternating-sign sum(theta)/sqrt(d), frozen independently of data; the identical projection at d=1 is removed. In this model the remaining projection is exactly theta. Coordinate diagnostics are also all reported; their R-hat/ESS maxima are not an extra dimension-growing gate. Optional model observables would be predeclared scalar callbacks; they extend the Holm family before evaluation and supplement the generic gate. They need no analytic target value. No model-specific or LISA-specific API logic is implemented here.

[Stan's diagnostic recommendations](https://mc-stan.org/learn-stan/diagnostics-warnings.html) motivate R-hat <1.01 and approximately 100 bulk/tail effective draws per original independent chain, hence 800 for eight walkers. Split chains do not double that floor. The [rank-normalization paper](https://arxiv.org/abs/1903.08008) supplies the standard definitions. These are precision/convergence warnings, not a hypothesis-test guarantee of stationarity. Conventional .05 familywise significance and Holm stepdown avoid independent .05 decisions for arbitrarily many coordinates. No cutoff depends on analytic mode weights or control-specific outcomes.

## Definitions and time axes

ArviZ computes the maximum of rank-normalized split and folded split R-hat, Geyer initial-positive/monotone-sequence bulk ESS on rank-normalized draws, and tail ESS as the minimum ESS for the 5% and 95% quantile indicators. Chains are independent walkers, not flattened concatenations. We also report conventional split between/within variance B/W, where B=n var(chain means) and W=mean within-chain sample variances.

Visit-order diagnostics use exactly the retained first K events per walker, splitting each into disjoint halves. First K versus second K KS is also reported from already saved later events, but later events cannot rescue the gate or retrospectively change the retained sample. Walker-to-walker KS distances assess distributional disagreement. KS is the supremum difference between empirical CDFs; no IID KS significance or ESS-substituted IID p-value is used.

The physical gate uses the common collector-completion horizon H=max(last retained sweep)+1 and compares [1024,floor((1024+H)/2)) with [floor((1024+H)/2),H). H is a stopping time determined by the quota, not an externally deterministic endpoint; the split is determined mechanically from H and is never tuned per control. All top occupation events in each physical window enter its walker mean, including events after an individual walker's Kth visit but before global completion. Counts vary; no padding or thinning occurs. The paired t statistic uses eight independent walker-level differences and 7 degrees of freedom. Holm sorts p-values and compares p_(i) with .05/(m-i+1), stopping at the first nonrejection.

This paired-t calibration is approximate: normality of walker-level averages, long-run regularity, random conditional counts, first-hit effects, and the shared stopping horizon can undermine its nominal error rate. Holm controls multiplicity only when its underlying p-values are valid; it cannot repair those limitations. These caveats prevent a finite-sample correctness claim.

For comparison, physical-window R-hat/ESS uses sixteen window-by-walker chains trimmed to their common minimum count, retaining visit order inside each window. This balanced representation loses some physical occupation information and remains descriptive only. The all-event paired means are the physical component of the candidate gate. Assigned j is identically J in retained conditional samples, so conditional index diagnostics carry no information; uncensored physical assigned-index R-hat/ESS are reported separately. Log prior is constant for every development trace and is explicitly excluded from nonconstant-observable checks.

We report the standard empirical Euclidean energy distance 2 mean||X-Y|| − mean||X-X'|| − mean||Y-Y'|| on fixed deterministic evenly spaced subsets capped at 512 points per segment. It supports multivariate theta; this development model has one coordinate, so it cannot test high-dimensional performance. No permutation significance is claimed for dependent trace points. Energy and KS are established descriptive distances, without an invented acceptance threshold.

## Retrospective performance

Every row fails the composite. The rejection triggers below are exhaustive; compression ESS and top visitation quotas pass all eighteen traces. Compression ESS matches the original saved-data calculation.

| Trace | Theta rank R-hat | Bulk ESS | Tail ESS | Balanced physical R-hat | Visit / physical KS | Rejection triggers |
|---|---:|---:|---:|---:|---|---|
| stage9d_failed_calibration | 1.01305 | 381.9 | 1459.3 | 1.02974 | 0.0596 / 0.0330 | projection_sum:rhat, projection_sum:bulk_ess |
| stage9d_selection | 1.02597 | 276.0 | 1685.3 | 1.05776 | 0.0764 / 0.0598 | projection_sum:rhat, projection_sum:bulk_ess |
| 910701_exact_joint_stationary | 1.02890 | 299.9 | 2181.7 | 1.04669 | 0.0856 / 0.0629 | projection_sum:rhat, projection_sum:bulk_ess |
| 910701_saved_promoted_start | 1.03026 | 280.7 | 1813.7 | 1.05524 | 0.1475 / 0.0985 | projection_sum:rhat, projection_sum:bulk_ess |
| 910702_exact_joint_stationary | 1.02032 | 301.4 | 1815.6 | 1.04842 | 0.0737 / 0.0355 | projection_sum:rhat, projection_sum:bulk_ess |
| 910702_saved_promoted_start | 1.02042 | 367.8 | 2111.3 | 1.04589 | 0.0691 / 0.0606 | projection_sum:rhat, projection_sum:bulk_ess |
| 910703_exact_joint_stationary | 1.01659 | 327.0 | 2189.8 | 1.04087 | 0.0483 / 0.0402 | projection_sum:rhat, projection_sum:bulk_ess, physical_Holm:log_likelihood |
| 910703_saved_promoted_start | 1.03443 | 288.1 | 2160.8 | 1.06945 | 0.0493 / 0.0620 | projection_sum:rhat, projection_sum:bulk_ess |
| 910704_exact_joint_stationary | 1.02883 | 241.3 | 1477.3 | 1.05192 | 0.0555 / 0.0450 | projection_sum:rhat, projection_sum:bulk_ess |
| 910704_saved_promoted_start | 1.03045 | 229.6 | 1508.5 | 1.07052 | 0.0510 / 0.0223 | projection_sum:rhat, projection_sum:bulk_ess |
| 910705_exact_joint_stationary | 1.01753 | 276.7 | 2186.0 | 1.04609 | 0.0604 / 0.0668 | projection_sum:rhat, projection_sum:bulk_ess |
| 910705_saved_promoted_start | 1.03792 | 288.9 | 2251.9 | 1.06794 | 0.1422 / 0.1290 | projection_sum:rhat, projection_sum:bulk_ess |
| 910706_exact_joint_stationary | 1.01824 | 308.2 | 1982.7 | 1.05391 | 0.1389 / 0.1118 | projection_sum:rhat, projection_sum:bulk_ess |
| 910706_saved_promoted_start | 1.02057 | 334.9 | 1812.9 | 1.05263 | 0.1318 / 0.0944 | projection_sum:rhat, projection_sum:bulk_ess |
| 910707_exact_joint_stationary | 1.02708 | 297.0 | 1844.2 | 1.07629 | 0.0211 / 0.0681 | projection_sum:rhat, projection_sum:bulk_ess |
| 910707_saved_promoted_start | 1.04582 | 204.0 | 1685.0 | 1.06867 | 0.1525 / 0.1491 | projection_sum:rhat, projection_sum:bulk_ess |
| 910708_exact_joint_stationary | 1.02030 | 268.6 | 1997.1 | 1.05794 | 0.0098 / 0.0401 | projection_sum:rhat, projection_sum:bulk_ess |
| 910708_saved_promoted_start | 1.02844 | 280.1 | 2194.8 | 1.06228 | 0.0625 / 0.0764 | projection_sum:rhat, projection_sum:bulk_ess |

Standalone retained theta R-hat flags the historical calibration and 8/8 stationary controls. Bulk ESS flags the same eighteen traces; tail ESS flags none. The historical theta R-hat 1.0130 is actually smaller than all eight stationary-control values (1.0166–1.0289); historical bulk ESS 381.9 is higher than every stationary-control value (241.3–327.0). Thus theta diagnostics explain the slow mixing versus the fast compression indicator, but provide **no useful discrimination of historical failure from these stationary controls**. Log-likelihood R-hat and ESS do not cause any retained-order rejection.

The physical Holm component does not flag the historical calibration or selection sample. It adds a log-likelihood rejection for stationary seed 910703 only: 1/8 stationary controls, no promoted controls. Physical balanced theta R-hat exceeds 1.01 for all traces. There is no historical failure missed in visit order and uniquely exposed by physical time; the one extra physical rejection occurs in a stationary control. This limited comparison does not prove the two time axes are interchangeable. Physical windows can reveal other features, but their significance needs dependent-data calibration beyond this toy.

The first/second-K distances, all observable metrics, physical counts, paired differences, energy distances, B/W and compression values are preserved per trace in JSON. These diagnostics are correlated views of the same trajectories; they are not additional independent replicates. The eight initialization pairs cannot be pooled into sixteen independent stationary controls.

## Interpretation after gate decisions

No exact CDF, true mean, target mode weight or mode label enters any PASS/FAIL computation. Stage-9D failure and successful selection labels identify the historical development examples, not a numerical gate input. After classification, the existing Stage-9E reference report shows historical first-K KS 0.09116 versus eight stationary first-K KS values below that. Nevertheless the generic R-hat/ESS candidate rejects all those controls and the previously successful selection sample. This illustrates the difference between a stationary generating process, a noisy realized sample and adequate effective precision; insufficient precision is the operative rejection here, not proven nonstationarity.

An 8/8 false-rejection count means rejection under the exact-joint-stationary initialization reference, following the actual first-K collector. Exact stationary physical initialization does not itself prove invariant first-hit trace initialization. This denominator is therefore the requested stationary-control denominator, not a proof of a universal false-positive rate. Eight controls and their paired arms cannot calibrate rare-event probabilities.

## High dimension and API assessment

The fixed projection budget prevents R-hat/ESS rejection probability from growing merely because additional coordinates are added. Holm provides a principled familywise multiplicity mechanism across all monitored coordinate mean tests, conditional on valid p-values. All coordinate R-hat/ESS remain visible as diagnostic information. Neither mechanism proves joint stationarity: fixed projections and coordinate means can miss changes in dependence structure, coordinate units can dominate unscaled sums, ESS is observable-specific, and high-dimensional energy-distance power was not assessed here. Production generalization is **not established**. The statistical building blocks are defensible warning diagnostics; the composite is not a defensible generic high-dimensional stationarity certificate.

A future API may accept optional model-provided scalar observables with declared names, deterministic evaluation and a family frozen before sampling. They can expose scientific features missed by generic projections and participate in adjusted physical comparisons. Generic diagnostics remain mathematically defined without these callbacks; callbacks cannot make convergence guaranteed, and analytic mode labels are never required. No LISA source-frequency implementation belongs in this development.

## Decision and disposition

**CASE B — TOO MANY FALSE REJECTIONS.** Historical failure detected: yes. Stationary controls falsely rejected: 8/8 (100%). Most informative slow-mixing diagnostics: theta projection rank R-hat and bulk ESS, but neither discriminates the historical failure from controls. Analytic mode labels required: no. High-dimensional production validity: not established. Exactly one final held-out ladder validation justified: no.

The thresholds remain unchanged as an unsuccessful development candidate. This is not validation, a production rule freeze, or an invitation to adjust cutoffs, search budgets, increase burn-in or collect replicas. STOP.
