# Verified results tables

Completed training: 30 runs; 7.0001 measured update hours. Main evaluation: 162 episodes; size evaluation: 72 episodes.

All +/- values are sample standard deviations across three training-seed means, not standard errors or confidence intervals. Each ordinary condition uses three evaluation episodes per training seed. Returns are raw reward sums.

## Main comparison: 100,000 transitions, 1,200 update seconds per run

| Method | Familiar sources | Held-out sources |
|---|---:|---:|
| Behavioural cloning | 117.88 +/- 18.64 | 7.68 +/- 5.99 |
| DrQv2+BC | 69.11 +/- 61.58 | 1.81 +/- 0.70 |
| Stronger RGB augmentation | 139.86 +/- 67.86 | 19.63 +/- 14.91 |
| Two-view policy consistency | 96.96 +/- 15.99 | 9.98 +/- 11.78 |
| Fixed top-band mask | 30.56 +/- 23.31 | 6.26 +/- 5.77 |
| Adapted author InfoGating* | 43.30 +/- 23.02 | 59.74 +/- 57.04 |

*InfoGating uses adapted author code with batch 16 rather than default 256, different training/evaluation protocol from the paper and different architecture/objectives from DrQv2+BC. This table does not establish improvement over published InfoGating.

## Dataset-size pilot: matched 300 update seconds per run

| Method | Transitions | Familiar sources | Held-out sources |
|---|---:|---:|---:|
| Behavioural cloning | 10,000 | 79.23 +/- 104.92 | 7.55 +/- 6.66 |
| Behavioural cloning | 100,000 | 101.41 +/- 28.25 | 3.40 +/- 2.16 |
| DrQv2+BC | 10,000 | 6.14 +/- 2.21 | 7.01 +/- 9.68 |
| DrQv2+BC | 100,000 | 4.28 +/- 2.58 | 10.87 +/- 15.91 |

## Compute

| Suite | Method | Update range | Total update seconds | Peak process RSS (GiB) | Peak PyTorch CUDA allocation (MiB) |
|---|---|---:|---:|---:|---:|
| runs | Behavioural cloning | 83,635-85,974 | 3600.03 | 3.23 | 159.7 |
| runs | Two-view policy consistency | 26,383-26,440 | 3600.07 | 3.23 | 214.7 |
| runs | DrQv2+BC | 32,099-32,476 | 3600.04 | 3.29 | 214.0 |
| runs | Adapted author InfoGating* | 16,562-16,600 | 3600.02 | unmeasured | unmeasured |
| runs | Fixed top-band mask | 31,881-31,997 | 3600.06 | 3.23 | 214.4 |
| runs | Stronger RGB augmentation | 31,692-32,047 | 3600.04 | 3.23 | 213.6 |
| runs_10k | Behavioural cloning | 21,854-22,918 | 900.03 | 1.43 | 158.5 |
| runs_10k | DrQv2+BC | 9,331-9,410 | 900.04 | 1.55 | 213.6 |
| runs_100k_matched | Behavioural cloning | 22,061-22,265 | 900.02 | 3.25 | 158.4 |
| runs_100k_matched | DrQv2+BC | 8,713-8,857 | 900.07 | 3.27 | 214.5 |

Peak process RSS and allocated tensor memory are different measurements; the CUDA figure excludes driver/context and other allocations. Budgets match update time, not update count.

## Occlusion stress test: paired return change

Only evaluation seed 701 is used. The gray central band is not a verified semantic mask.

| Method | 8-pixel band | 16-pixel band | 32-pixel band |
|---|---:|---:|---:|
| Behavioural cloning | 73.98 +/- 121.96 | 53.19 +/- 85.00 | -2.50 +/- 1.80 |
| DrQv2+BC | -0.72 +/- 0.82 | -0.65 +/- 1.47 | -1.22 +/- 1.32 |
| Stronger RGB augmentation | -23.64 +/- 43.90 | -24.55 +/- 45.89 | -26.98 +/- 44.40 |
| Two-view policy consistency | -3.28 +/- 1.50 | -1.54 +/- 1.58 | -2.01 +/- 1.93 |
| Fixed top-band mask | 77.90 +/- 140.45 | 86.76 +/- 154.75 | -3.38 +/- 2.58 |
| Adapted author InfoGating* | -79.93 +/- 106.56 | -74.37 +/- 113.23 | -93.48 +/- 96.29 |
