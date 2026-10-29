# Interpretation of completed main evaluation

All 18 main training runs and 162 main evaluation episodes are complete. Aggregate validation reports no missing/duplicate main cases, non-finite returns or incomplete episodes. The separate dataset-size pilots are still running; these observations apply only to the main comparison.

| Method | Familiar-source return | Held-out-source return |
|---|---:|---:|
| BC | 117.88 +/- 18.64 | 7.68 +/- 5.99 |
| DrQv2+BC | 69.11 +/- 61.58 | 1.81 +/- 0.70 |
| Stronger RGB augmentation | 139.86 +/- 67.86 | 19.63 +/- 14.91 |
| Actor-only two-view consistency | 96.96 +/- 15.99 | 9.98 +/- 11.78 |
| Fixed top-band masking | 30.56 +/- 23.31 | 6.26 +/- 5.77 |
| Adapted author InfoGating | 43.30 +/- 23.02 | 59.74 +/- 57.04 |

Each number averages three training-seed means, with three evaluation episodes within each training seed. The +/- value is the sample standard deviation of those three means. Source: reports/aggregate.json, main suite, damage_band_pixels=0. Final paper tables should be generated from JSON rather than transcribed from this note.

Five methods have lower averages on the held-out sources. This is consistent with a generalization problem in this setup, but the small collection of evaluation seeds and source images limits the conclusion. InfoGating has the largest held-out mean and large variability; it does not show a drop between these particular condition averages. That reversal is not proof that unfamiliar images help: source-specific difficulty, seed variation and different architectures/objectives remain plausible explanations.

The ordinary RGB augmentation control has a higher mean than the DrQv2+BC baseline in both conditions. This suggests that basic augmentation is a necessary comparison before claiming a more complex new mechanism is useful. The present three-seed experiment cannot establish statistically reliable or general superiority. BC also exceeds the DrQv2+BC mean in both conditions, so reward-based policy improvement did not deliver a clear advantage at this compute budget.

The fixed mask and actor-only consistency do not establish a solution to the distraction problem. They show low held-out returns and considerable variation. The mask is a crude spatial intervention and the consistency gradients do not directly update the encoder. These implementation choices limit what the results say about semantic segmentation or learned invariant representations.

The evidence does not show that a newly proposed method improves on relevant existing research: no new adaptive method was trained, and the adapted published comparator has the highest held-out point estimate. A defensible next study should investigate the failures of basic baselines, strengthen comparator protocol matching, test moving distractions separately, and evaluate any new representation objective on additional tasks and seeds.

Evaluation uses static distractions within each episode. Performance against moving videos is untested. The raw return scale is a reward sum, not a percentage accuracy. The recorded dataset returns around703 are data-collection statistics and should not be treated as a directly matched test baseline.
