# Exploratory secondary static/dynamic stress test

Five fresh evaluation seeds (901–905), separate from the primary test. Each table entry is the mean and sample SD over three training-seed means; each seed mean averages five episodes. Dynamic changes background, camera and colour together. These are secondary observations, not checkpoint-selection evidence.

| Condition | Static return | Dynamic return | Paired dynamic minus static |
|---|---:|---:|---:|
| inv | 77.92 +/- 38.01 | 70.53 +/- 37.67 | -7.39 +/- 49.89 |
| gate | 95.03 +/- 4.82 | 61.02 +/- 5.85 | -34.02 +/- 8.24 |
| adv_inv | 46.76 +/- 20.21 | 73.18 +/- 19.81 | 26.42 +/- 6.60 |
| adv_gate | 88.61 +/- 38.32 | 79.05 +/- 4.05 | -9.56 +/- 35.58 |

This test does not establish a causal explanation for the primary negative gating result. It uses one task and three trained seeds per condition; no significance or broad robustness claim is made. The manuscript reports the primary static protocol; this stress test is retained as a separate supplement.
