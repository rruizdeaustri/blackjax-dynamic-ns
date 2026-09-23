# Stage-4M: exact MH-corrected Logistic block refresh

**Both single-source and pair-source Logistic independence refreshes are promising for the current implemented level-9 target.** This is a fixed-base feasibility result, not a mixing or equilibrium validation. Level 10 remains rejected.

## Target and exact rule

The target is the unchanged BayesLISAx scalar prior multiplied by the strict indicator `logL > -110252.99476697217`. For a state-independent labelled block choice, its selection probability cancels between forward and reverse proposals. Unchanged coordinates have the same point-mass proposal in both directions. Consequently, for either six or twelve refreshed coordinates:

```text
log r = log pi_impl(new) - log pi_impl(old) + log q(old_B) - log q(new_B)
log q(v) = sum(-softplus(v_k) - softplus(-v_k))
alpha = 0 outside the strict contour; otherwise exp(min(0, log r))
```

The implementation evaluates the actual full scalar prior. It does not require the prior to factorize and does not cancel it against the analytic proposal. A proposed implemented prior of negative infinity gives alpha zero without clipping or repair. Nonfinite Logistic draws are recorded and rejected without redraw or model evaluation; unevaluated quantities are NaN in the NPZ. They count separately from implemented-prior failures. No acceptance Bernoulli was drawn.

## Frozen design and integrity

Exactly 16 states per walker were selected from the 512-state Stage-4H continuation using rounded evenly spaced indices, with no likelihood or configuration filtering:

`[0, 34, 68, 102, 136, 170, 204, 238, 273, 307, 341, 375, 409, 443, 477, 511]`

There are 128 bases, with 32 single and independently 32 pair proposals each (4096 of each). JAX keys 8501 and 8502 split into separate uniform-category and direct-Logistic streams. A category maps bijectively to one of nine labels or 36 unordered pairs. Every proposal starts from its original base. The runner checks the selected values against the generated draws and untouched coordinates byte for byte.

All bases had finite implemented prior and likelihood, strict ell9 membership, and recomputed likelihood agreement with cache at absolute tolerance 1e-7 (prior 1e-9). There were zero structural integrity failures. All 8320 likelihood evaluations (128 bases plus 8192 proposals) used CPU float64, because GPU access was unavailable. Model, config, checkpoint, saved trace, isotropic proposal, and DNS code hashes were unchanged.

## Survival and exact acceptance

| Proposal | Raw ell9 survival | Wilson 95% interval | Exact mean alpha | ell10 diagnostic fraction | MH-weighted ell10 fraction |
|---|---:|---:|---:|---:|---:|
| single | 3473/4096 = 0.8479003906 | [0.836577, 0.858572] | 0.847900390624987 | 0.2570800781 | 0.257080078124996 |
| pair | 2896/4096 = 0.7070312500 | [0.692905, 0.720770] | 0.707031249999988 | 0.2319335938 | 0.231933593749995 |

Mean alpha is the sample mean of bounded acceptance probabilities, with no binomial interval attached. Wilson intervals apply only to raw survival and are approximate conditional summaries for this fixed, heterogeneous collection of bases; they do not quantify uncertainty in equilibrium coverage. The highly correlated saved histories do not become independent target samples by spacing their indices.

Among contour survivors, alpha quantiles separate the numerical correction from the contour decision:

| Proposal | Minimum | 5% | Median | 95% | Maximum |
|---|---:|---:|---:|---:|---:|
| single | 0.999999999997510 | 0.999999999999945 | 0.999999999999998 | 1.000000000000000 | 1.000000000000000 |
| pair | 0.999999999998089 | 0.999999999999936 | 1.000000000000000 | 1.000000000000000 | 1.000000000000000 |

## Walker and label summaries

| Walker | Single raw survival | Single mean alpha | Pair raw survival | Pair mean alpha |
|---|---:|---:|---:|---:|
| 0 | 442/512 = 0.863281 | 0.863281 | 336/512 = 0.656250 | 0.656250 |
| 1 | 436/512 = 0.851562 | 0.851562 | 375/512 = 0.732422 | 0.732422 |
| 2 | 446/512 = 0.871094 | 0.871094 | 384/512 = 0.750000 | 0.750000 |
| 3 | 439/512 = 0.857422 | 0.857422 | 365/512 = 0.712891 | 0.712891 |
| 4 | 432/512 = 0.843750 | 0.843750 | 358/512 = 0.699219 | 0.699219 |
| 5 | 455/512 = 0.888672 | 0.888672 | 398/512 = 0.777344 | 0.777344 |
| 6 | 428/512 = 0.835938 | 0.835937 | 360/512 = 0.703125 | 0.703125 |
| 7 | 395/512 = 0.771484 | 0.771484 | 320/512 = 0.625000 | 0.625000 |

All single categories are retained, including any zero-survival categories. Full-precision means and raw-survival Wilson intervals are in report.json.

| Label(s) | Proposals | Survivors | Raw survival | Mean alpha |
|---|---:|---:|---:|---:|
| [0] | 477 | 391 | 0.819706 | 0.819706 |
| [1] | 450 | 434 | 0.964444 | 0.964444 |
| [2] | 444 | 373 | 0.840090 | 0.840090 |
| [3] | 460 | 327 | 0.710870 | 0.710870 |
| [4] | 472 | 400 | 0.847458 | 0.847458 |
| [5] | 393 | 376 | 0.956743 | 0.956743 |
| [6] | 462 | 434 | 0.939394 | 0.939394 |
| [7] | 456 | 329 | 0.721491 | 0.721491 |
| [8] | 482 | 409 | 0.848548 | 0.848548 |

All pair categories are retained, including any zero-survival categories. Full-precision means and raw-survival Wilson intervals are in report.json.

| Label(s) | Proposals | Survivors | Raw survival | Mean alpha |
|---|---:|---:|---:|---:|
| [0, 1] | 92 | 76 | 0.826087 | 0.826087 |
| [0, 2] | 111 | 83 | 0.747748 | 0.747748 |
| [0, 3] | 104 | 55 | 0.528846 | 0.528846 |
| [0, 4] | 114 | 82 | 0.719298 | 0.719298 |
| [0, 5] | 113 | 92 | 0.814159 | 0.814159 |
| [0, 6] | 116 | 91 | 0.784483 | 0.784483 |
| [0, 7] | 127 | 64 | 0.503937 | 0.503937 |
| [0, 8] | 106 | 72 | 0.679245 | 0.679245 |
| [1, 2] | 104 | 83 | 0.798077 | 0.798077 |
| [1, 3] | 134 | 95 | 0.708955 | 0.708955 |
| [1, 4] | 89 | 73 | 0.820225 | 0.820225 |
| [1, 5] | 121 | 109 | 0.900826 | 0.900826 |
| [1, 6] | 131 | 113 | 0.862595 | 0.862595 |
| [1, 7] | 106 | 74 | 0.698113 | 0.698113 |
| [1, 8] | 112 | 93 | 0.830357 | 0.830357 |
| [2, 3] | 111 | 66 | 0.594595 | 0.594595 |
| [2, 4] | 114 | 72 | 0.631579 | 0.631579 |
| [2, 5] | 98 | 85 | 0.867347 | 0.867347 |
| [2, 6] | 135 | 106 | 0.785185 | 0.785185 |
| [2, 7] | 132 | 74 | 0.560606 | 0.560606 |
| [2, 8] | 102 | 67 | 0.656863 | 0.656863 |
| [3, 4] | 127 | 72 | 0.566929 | 0.566929 |
| [3, 5] | 110 | 77 | 0.700000 | 0.700000 |
| [3, 6] | 102 | 68 | 0.666667 | 0.666667 |
| [3, 7] | 113 | 53 | 0.469027 | 0.469027 |
| [3, 8] | 121 | 65 | 0.537190 | 0.537190 |
| [4, 5] | 123 | 98 | 0.796748 | 0.796748 |
| [4, 6] | 109 | 85 | 0.779817 | 0.779817 |
| [4, 7] | 120 | 76 | 0.633333 | 0.633333 |
| [4, 8] | 106 | 78 | 0.735849 | 0.735849 |
| [5, 6] | 108 | 102 | 0.944444 | 0.944444 |
| [5, 7] | 109 | 71 | 0.651376 | 0.651376 |
| [5, 8] | 116 | 88 | 0.758621 | 0.758621 |
| [6, 7] | 123 | 81 | 0.658537 | 0.658537 |
| [6, 8] | 130 | 101 | 0.776923 | 0.776923 |
| [7, 8] | 107 | 56 | 0.523364 | 0.523364 |

## Nonlocal geometry

All positive-alpha proposals also meet the predeclared descriptive high-alpha cut of 0.5: 3473 single and 2896 pair proposals. Their geometry distributions therefore coincide. Matching uses the existing Stage-4I Hungarian source-set metric, frequency in Fourier bins (1/31536000 Hz), and the same normalized full-coordinate metric with circular psi/lambda differences. No matching is used to construct a proposal.

| Geometry median | Single refresh | Single ordinary Stage-4H | Pair refresh | Pair ordinary Stage-4H |
|---|---:|---:|---:|---:|
| latent_block_l2 | 5.84745 | 0.923932 | 8.48901 | 0.568406 |
| physical_f0_rms_bins | 185.506 | 17.6777 | 222.453 | 11.6167 |
| source_set_frequency_rms_bins | 41.2499 | 5.62361 | 55.3985 | 5.25902 |
| source_set_full_rms | 0.114924 | 0.0264035 | 0.15169 | 0.0175625 |

The ordinary comparison uses adjacent saved Stage-4H isotropic steps of the corresponding component (1624 single and 1616 pair), excluding the first state in each walker because its preceding state is not in this trace. This is descriptive: accepted refresh probes use selected bases, while ordinary steps span the saved histories. It is not a cost-normalized mixing benchmark.

| Accepted refresh geometry | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| single latent_block_l2 | 1.11776 | 3.10159 | 5.84745 | 9.51852 | 15.0983 |
| single physical_f0_rms_bins | 0.257524 | 15.5327 | 185.506 | 485.028 | 626.216 |
| single source_set_frequency_rms_bins | 0.0858412 | 4.88283 | 41.2499 | 80.6839 | 107.716 |
| single source_set_full_rms | 0.0266822 | 0.0704489 | 0.114924 | 0.151936 | 0.183421 |
| pair latent_block_l2 | 3.00202 | 5.62989 | 8.48901 | 12.314 | 17.8347 |
| pair physical_f0_rms_bins | 8.78431 | 62.8646 | 222.453 | 412.825 | 554.164 |
| pair source_set_frequency_rms_bins | 0.927874 | 17.464 | 55.3985 | 103.221 | 164.763 |
| pair source_set_full_rms | 0.0727208 | 0.111583 | 0.15169 | 0.190197 | 0.227693 |

The large source-set displacement survives removal of label permutations. Both refresh sizes therefore produce appreciably accepted nonlocal moves in the sampled regions.

## Tail mismatch

The materiality threshold was fixed before probing at absolute log-density discrepancy greater than 1e-8. For each proposal size: **zero nonfinite generated coordinates, zero nonfinite-draw proposals, zero implemented-prior negative infinities, zero material discrepancies, and zero nonfinite likelihoods**. This finite sample does not remove the positive-tail mismatch established in Stage-4L.

The omitted term in a contour-only rule is `delta_correction = (lp_new-lp_old) - (q_new-q_old)`, identical to the recorded pre-truncation log MH ratio. Across all proposals:

| Proposal | Min | 5% | Median | 95% | Max |
|---|---:|---:|---:|---:|---:|
| single | -2.4904523e-12 | -5.5067062e-14 | -1.7763568e-15 | 5.3290705e-14 | 2.4016344e-12 |
| pair | -1.91136e-12 | -6.3948846e-14 | 0 | 6.3948846e-14 | 1.0601298e-11 |

Thus the mismatch is practically negligible in these sampled states, but the correction remains necessary for the current target. Rare far-tail proposals cannot be assessed from 8192 draws.

## Classification, validation, and artifacts

**Single: promising. Pair: promising.** Both have substantial exact acceptance in every walker and accepted source-set movement substantially larger than ordinary isotropic median movement. This qualitative classification uses the requested criteria; no desired numerical acceptance threshold was fitted after observing the data. It does not establish longer-run mixing, active-source replacement efficiency, or target coverage.

Cheap tests prove the general MH identity with a deliberately non-Logistic, coupled target for both block sizes, including reverse-flow balance. They also test stable analytic q, negative-infinite proposed prior, strict-contour rejection, nonfinite draws, untouched coordinates, fixed-base reuse, and state-independent selection construction. No real LISA likelihood appears in pytest.

Artifacts: `/tmp/lisa_dns_stage4m_logistic_mh/report.json`, `bases.npz`, `single_proposals.npz`, and `pair_proposals.npz`. The NPZ files retain every requested scalar diagnostic, labels, generated draws, full proposed positions, signed physical f0 displacements, and permutation-invariant geometry. The runner is `examples/lisa_dns_stage4/logistic_mh_refresh.py`.

The prior, isotropic kernel, and DNS implementation are unchanged. No integration, level-10 retry, level 11, production, or evidence work was performed. **Level 10 remains rejected.**

Validation: **all 125 existing tests and eight new cheap tests passed (133 total)**. A separate read-only audit recomputed every saved MH ratio and alpha, rechecked fixed-base replacements, and verified protected hashes.
