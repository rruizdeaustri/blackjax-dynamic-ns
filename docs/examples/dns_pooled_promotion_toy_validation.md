# Pooled promotion: final toy validation 1 of 2

**PASS.** The existing generic Stage-3 within-bank, uniform pooled sampling **with replacement** passed support, exact-cache, independence, deterministic stress, and analytic distribution checks. No generic DNS or LISA workflow code changed. This permits exactly **one controlled LISA validation**, which was not performed here. Stage 5B remains failed; the LISA ladder remains frozen through level 12.

## Fixed experiment

The model and transition are imported unchanged from `examples/dns_level_construction_validation.py:problem("double_well")`: uniform prior on [-6,6], logL = -0.5((|x|-1)/0.3)^2, slice direction ±0.5, max_steps=10, max_shrinkage=100. The fixed new threshold is -4.5: support is (-1.9,-0.1) ∪ (0.1,1.9), with exact mode probabilities (0.5,0.5). The gap is traversable by the existing slice kernel; this is not a test of its ability to cross arbitrarily wide gaps.

Protocol fixed before results: seeds 101/202/303; 8 walkers per bank; 256 burn-in transitions; 1,024 retained draws with thinning 2. These burn-in/thinning settings are the existing Stage-3 defaults. Predecessor histories use the same kernel at -8, with 256 burn-in and 256 retained draws, thinning 2. Initial predecessor draws are independent exact draws from that contour, [-2.2,2.2]. Each seed splits into independent selection/calibration keys, each split into initialization/history/promotion/transition keys. All 24 keys are distinct. Controls share promotion/transition keys within a bank where meaningful, never across banks.

Two predeclared scenarios: ordinary saved toy history; and a deterministic stress history in which walker 0 is replaced by cyclic copies of **its own saved ineligible states**, with caches copied. This fixture is deliberately nonstationary and is not represented as an unbiased construction sample. All other walkers remain unchanged. Each bank then uses same-walker random-survivor promotion, the actual generic `dns_levels._survivors` with replacement, or an eligible-entry without-replacement control. No parameters were tuned and no seed was retried.

Checks after every toy transition, including burn-in: finite position/prior/likelihood, strict contour membership, bitwise recomputed cache equality. Promoted states and all caches equal their indexed entries in their own retained pool. Separate bank arrays do not share storage; selected realizations do not occur in the other bank history. No shared mutable bank buffers or calibration rescue is used.

## Official with-replacement results

Here S/C denote selection/calibration, O ordinary and M missing predecessor. Every run produced 8 distinct initial positions (0 duplicates); duplicates remain permitted, and the small-pool tests explicitly require them. Distances are absolute differences in the toy position, reported as min / median / max. Donor multiplicities list walkers 0–7.

| Seed/bank/case | Eligible entries / donors | Donor multiplicities | Initial distances | Post-burn-in distances | Mode + weight ± SE |
|---|---:|---|---|---|---|
| 101/S/O | 1659 / 8 | [2, 1, 0, 2, 2, 0, 1, 0] | 0.044/0.877/2.723 | 0.060/2.379/3.388 | 0.50488 ± 0.00793 |
| 101/S/M | 1444 / 7 | [0, 1, 0, 1, 0, 0, 5, 1] | 0.057/1.272/2.814 | 0.063/1.794/3.620 | 0.50122 ± 0.00805 |
| 101/C/O | 1656 / 8 | [2, 2, 0, 0, 1, 2, 0, 1] | 0.017/2.416/3.595 | 0.179/1.224/3.459 | 0.50830 ± 0.00808 |
| 101/C/M | 1452 / 7 | [0, 1, 0, 1, 1, 2, 0, 3] | 0.088/1.163/3.366 | 0.075/1.244/3.280 | 0.51367 ± 0.00843 |
| 202/S/O | 1690 / 8 | [0, 1, 0, 1, 0, 1, 3, 2] | 0.014/1.020/3.109 | 0.111/1.470/3.209 | 0.49524 ± 0.00919 |
| 202/S/M | 1474 / 7 | [0, 2, 1, 0, 2, 0, 0, 3] | 0.027/1.609/3.333 | 0.064/1.432/3.210 | 0.50122 ± 0.00795 |
| 202/C/O | 1645 / 8 | [1, 2, 0, 1, 1, 0, 2, 1] | 0.018/1.576/3.359 | 0.014/1.234/2.949 | 0.49756 ± 0.01057 |
| 202/C/M | 1442 / 7 | [0, 1, 1, 2, 2, 1, 1, 0] | 0.154/0.942/3.054 | 0.006/1.125/2.802 | 0.49475 ± 0.00924 |
| 303/S/O | 1681 / 8 | [0, 0, 1, 0, 3, 1, 1, 2] | 0.099/1.142/3.314 | 0.155/0.972/3.153 | 0.49622 ± 0.00797 |
| 303/S/M | 1474 / 7 | [0, 0, 2, 3, 0, 0, 2, 1] | 0.022/0.928/2.702 | 0.122/0.824/3.038 | 0.49060 ± 0.00801 |
| 303/C/O | 1695 / 8 | [0, 2, 1, 0, 1, 1, 3, 0] | 0.012/1.088/2.759 | 0.042/1.078/3.035 | 0.51038 ± 0.00870 |
| 303/C/M | 1489 / 7 | [0, 0, 1, 1, 1, 0, 2, 3] | 0.032/1.181/3.274 | 0.103/1.483/3.369 | 0.51550 ± 0.01095 |

Each M history has **zero** eligible states for predecessor 0 and seven eligible donor walkers. The same-walker rule failed in all six M cases, as constructed. Pooled promotion succeeded in all six without borrowing a calibration/selection state from the other bank.

## Distribution and communication

The analytic logL mean is -1.5. Its q-quantile is -4.5(1-q)^2. Position CDF checks test both modes and within-mode uniformity; likelihood CDF checks test constrained logL independently of sign. The predeclared check at q=0.1,0.25,0.5,0.75,0.9 was |estimate-q| ≤ 6 SE + 0.004, following the existing Stage-3 validation convention. SE is the maximum of within-walker block-mean and between-walker estimates (block 64). Every check passed for all three schemes, not just the official pooled rule. These are finite-sample diagnostics, not formal simultaneous confidence guarantees.

| Scheme / scenario | Runs | Mode + range | Mean logL range | Median pairwise logL KS range | Max pairwise logL KS range | Walker logL ESS range | Walker mode ESS range | Retained mode switches per walker |
|---|---:|---|---|---|---|---|---|---|
| same_walker / ordinary | 6 | 0.4871–0.5143 | -1.5276–-1.4978 | 0.0327–0.0391 | 0.0557–0.0879 | 694.5704–1024.0000 | 318.5725–546.6699 | 267.0000–355.0000 |
| same_walker / missing_predecessor | 0 | expected promotion failure | — | — | — | — | — | — |
| replacement / ordinary | 6 | 0.4952–0.5104 | -1.5238–-1.4955 | 0.0352–0.0449 | 0.0488–0.1006 | 682.0893–1024.0000 | 323.2192–551.5724 | 269.0000–355.0000 |
| replacement / missing_predecessor | 6 | 0.4906–0.5155 | -1.5219–-1.4859 | 0.0312–0.0469 | 0.0459–0.0869 | 705.1273–1024.0000 | 304.1957–541.9837 | 279.0000–345.0000 |
| unique / ordinary | 6 | 0.4922–0.5159 | -1.5209–-1.4991 | 0.0327–0.0430 | 0.0449–0.0723 | 473.5150–1024.0000 | 312.5121–570.7371 | 267.0000–353.0000 |
| unique / missing_predecessor | 6 | 0.4928–0.5216 | -1.5345–-1.4931 | 0.0342–0.0430 | 0.0586–0.0850 | 644.4369–1024.0000 | 268.4625–535.5960 | 259.0000–349.0000 |

ESS uses initial-positive paired autocorrelations per walker; it is descriptive. Switch counts are between retained observations (plus the burn-in endpoint), so they undercount transitions undone between retained draws. Pairwise KS values are descriptive because samples are autocorrelated. Every walker in every completed run visited both modes. The without-replacement control sampled distinct eligible entries and also produced eight distinct positions in every run. It showed no clear advantage or reproducible pathology of replacement; occasional differences are compatible with Monte Carlo variability.

Uniformity is over **retained entries**, including repeated states with their multiplicities: `_survivors` uses `jax.random.choice(..., replace=True)` without weights. Tests reproduce its exact sampled indices, check empirical frequencies on a known two-entry pool, and force duplicates with eight draws from two eligible states. Lack of duplicates in the larger experimental pools is not a restriction of the rule.

## Interpretation and boundary

Promotion changes initialization, not the Markov transition. Support-valid initial states need not be stationary. The unchanged constrained slice kernel has the same invariant target; tests verify the experiment runner exactly reproduces the existing Stage-3 sampler. Pooling does not guarantee independent promoted particles, accurate empirical mode weights, or finite-time equilibration. Replacement can duplicate ancestry and cannot recover a mode absent from the bank unless subsequent transitions reach it. This experiment demonstrates relaxation and correct mode weights for this connected-by-transitions toy after the fixed burn-in; it does not establish broad LISA catalogue equilibrium.

The single rule retained for the next validation is: independently within each bank, pool strict eligible retained entries, draw the required starts uniformly with replacement, copy full states/caches, and apply the existing fixed burn-in. An entirely empty bank must stop. Never rescue selection with calibration, or vice versa.

Exactly one controlled LISA validation is authorized next. If that validation passes, freeze the algorithm. If it fails, stop at LISA level 12. There is no return to a manual level-13 → level-14 → level-15 methodology-development loop.

## Reproduction and integrity

Implementation: [toy driver](../../examples/dns_pooled_promotion_validation.py). Tests: [promotion tests](../../tests/ns/test_dns_pooled_promotion.py). [Complete numerical results](dns_pooled_promotion_toy_results.json) contain per-run donor counts, indices, mode weights, CDF tests, ESS, switches and distances, plus bank-history hashes and RNG keys. The run wrote only its own `/tmp/dns_pooled_promotion_toy/` outputs; documentation/tests were added separately.

Command (CPU, x64, repository on PYTHONPATH): `python examples/dns_pooled_promotion_validation.py --output /tmp/dns_pooled_promotion_toy`. Existing completed results are protected against accidental rerun. Protocol and protected-input hashes were saved before toy sampling. SHA-256 comparison afterward verified every Stage-5B artifact, the Stage-4Z level-12 checkpoint, and LISA workflow source files unchanged. No level-13 checkpoint exists. No LISA module/model was imported or evaluated; no checkpoint or evidence code was implemented.

Tests: `pytest -q tests/ns/test_dns*.py` — **72 passed in 72.67 s**, including 7 new toy promotion tests and all existing DNS tests. No LISA tests or sampling were run.
