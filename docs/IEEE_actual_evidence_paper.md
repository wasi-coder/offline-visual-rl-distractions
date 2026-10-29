# Visual Offline Reinforcement Learning from Distracted Data: A Compute-Limited Cheetah Study

Empirical draft for supervisor review. Author and institutional metadata must be supplied before submission.

## Abstract

We study visual offline reinforcement learning using distracted data alone under a limited local compute budget. Six methods are trained on 100,000 cheetah-run transitions with three seeds and 1,200 measured update seconds per run. A matched 300-second pilot compares 10,000 and 100,000 transitions for BC and DrQv2+BC. The study records 7.00 update hours and 234 evaluation episodes. The highest observed held-out-source mean is infogating (59.74, seed standard deviation 57.04). Evaluation distractions are static within episodes. Results are descriptive and specific to this task and budget; the adapted InfoGating comparator is not a reproduction of published performance.

## I. Motivation and related work

Visual offline RL combines learning from a fixed action dataset with learning useful features from images. Lu et al. [1] introduced a visual offline benchmark and competitive adaptations of visual control algorithms. DrQ-v2 [2] provides a data-augmented actor-critic foundation; adding behavioural regularization is important to distinguish the offline baseline used here from the original online algorithm. Our study uses the supplied DrQv2+BC implementation rather than claiming to introduce this baseline.

Chang et al. [3] address visual distraction through domain-adversarial training. Their code and distracted dataset motivate this project. The present study examines simple controls when training only on distracted observations; it does not reproduce their complete proposed method. Avoiding separately clean data is a constraint of our experiment, not by itself evidence of novelty relative to all prior research.

SeMOPO [4] takes a model-based approach: it separates agent-related and external latent factors and estimates uncertainty on the agent-related component. This addresses the problem that moving distractors can make world-model uncertainty misleading. Our experiments use model-free policies and do not learn a world model, so they cannot claim to reproduce or outperform SeMOPO. The useful lesson is to separate information needed for control from background dynamics instead of treating all image prediction errors as equally important.

Guzel et al. [5] increase offline visual diversity through augmentation followed by diffusion-generated latent data. Our RGB-gain control changes existing observations and does not generate additional latent transition data or train a diffusion model. This paper motivates testing whether simpler augmentation already helps before introducing a more complex representation method.

ACRO [6] uses multi-step inverse prediction to learn agent-centric representations from actions and future observations. Its guarantees depend on assumptions about exogenous information, observability and data. InfoGating [7] learns to retain a small amount of task-relevant information through differentiable gating. Our published comparator follows adapted author InfoGating code, with explicit batch-size and protocol differences documented in the methods. Its measured performance must not be equated with the original paper's results. ACRO is reviewed but is not an independently trained comparator here.

MMCS [8] combines latent mask partitioning with sufficiency and conditional-independence objectives, decoupling representation learning from downstream control. This is substantially different from covering the top image rows with gray. We inspected its author implementation and accessible publisher description; the full paywalled paper was not obtained. MMCS was not trained in this study.

Together, these works establish that agent-centric representation learning, information suppression and increased visual diversity are existing research directions. The current contribution is a limited empirical comparison and reproducible local pipeline. A future method must address a demonstrated failure and distinguish its objective or mechanism from these approaches, with appropriate controlled experiments.

## II. Experimental methods


Can simple visual transformations and policy consistency improve control from a fixed, distracted dataset under a limited local training budget? The study compares practical interventions and their failure modes. It does not introduce or validate a new adaptive masking algorithm.

## Data and experimental design
The main dataset contains 100,000 transitions from 200 complete, 500-transition episodes. Twenty-five episodes were sampled from each of eight easy-distraction cheetah-run medium-expert shards. Observations are 84 by 84 RGB images, with three consecutive frames stacked for the policy. Actions have six components. No separately clean training images or online environment transitions enter optimization. The stored dataset is a subset of the downloaded collection, not the entire collection.

A nested 10,000-transition subset contains 20 complete episodes sampled across the same eight shards. Dataset-size experiments compare 10,000 and 100,000 transitions using the same 300-second update budget, algorithms and seeds. They must not be compared as a pure data-size effect against the 1,200-second main runs. The mean recorded episode returns are approximately 703.17 and 703.01 respectively; these are dataset statistics, not scores achieved by a learned agent.

The main comparison uses three training seeds per method and 1,200 seconds of measured update time per run. CUDA synchronization closes each timed update. Data loading, checkpoint writing and evaluation are outside this timer. Batch size is 16. Equal time does not imply equal numbers of gradient updates, sample presentations or convergence. Initialization, architecture and objectives also differ between the adapted published comparator and the core methods.

## Methods actually implemented

1. Behavioural cloning (BC): train the image encoder and policy to predict recorded actions by mean-squared error, with random-shift augmentation.

2. DrQv2+BC: the supplied offline baseline combining critic-based policy improvement with a behavioural cloning penalty and random shifts.

3. Stronger augmentation: multiply RGB channels by random gains in [0.6, 1.4], shared across stacked frames and current/next observations, and clip before baseline training. Colour changes are not guaranteed to preserve every task cue.

4. Two-view policy consistency: add mean-action agreement between an original view and a colour-transformed view, with a detached original target and coefficient one. This additional term updates the actor; it does not directly train the encoder to produce invariant features.

5. Fixed spatial masking: replace the top 16 image rows with a constant gray value during training and inference. This is a spatial suppression control, not segmentation, learned foreground detection, or guaranteed background removal.

6. Adapted author InfoGating code: retain the downloaded implementation's encoder, learned mask, inverse contrastive objective and BC update. Use batch 16 rather than the author's default 256, the same dataset and 1,200-second budget, and disable diagnostic plotting. The reduced batch changes the contrastive negatives. The author code's concurrent representation/policy update differs from the paper's described pretraining/frozen-feature protocol. This is an adapted author-code comparison, not a reproduction of published scores or a pure masking ablation.

ACRO and MMCS inform the literature comparison; they are not independently evaluated algorithms in this study. The earlier proposal for adaptive consistency weighting is future work, not an implemented component.

## Evaluation and statistics
Evaluate final checkpoints with fresh physical seeds 701, 702 and 703 in familiar and held-out background-source conditions. Camera and colour randomness use explicit paired seeds. Familiar sources follow the dataset generator's easy-distraction convention; held-out sources come from a disjoint DAVIS validation list. The stored training pixels have not been independently reverse-matched to the original video files, so provenance claims rely on generator conventions and manifests.

For each training seed, average the three episode returns in a condition. Report the mean and sample standard deviation across the three training-seed averages. The nine episodes are not nine independent training repetitions. Use raw returns and display individual training-seed values. Three training seeds and three evaluation seeds give limited precision; avoid significance claims or broad algorithm rankings.

An additional held-out test covers a central horizontal band of 8, 16 or 32 pixels with gray, using evaluation seed 701. Pair each corrupted result with that checkpoint's uncorrupted seed-701 result. These are descriptive closed-loop stress tests: masking may change subsequent actions and trajectories, and the bands have not been verified to cover only robot limbs or only background. They are not a direct causal explanation of representation learning.

Saved simulator videos demonstrate deployment of the learned policies, but must be labeled as physics-rendered views rather than necessarily the exact transformed observation supplied to the policy. Training exclusively offline is compatible with running simulator episodes for evaluation; no resulting transitions are used for training.

## Validation and evidence
The future-frame replay adapter matches the downloaded author sampler on 100 identically seeded synthetic minibatches, with all 700 output arrays equal. Initial algorithm updates and full evaluation episodes were validated for finite values; the evaluator checks finite, bounded actions. Final aggregation verifies expected run/condition/seed keys, uniqueness, 500-action episode lengths, finite returns, completed training budgets and checkpoint existence. Data manifests and hashes retain provenance. These checks establish pipeline properties, not algorithmic convergence or correctness of every scientific assumption.

## Limits and next research step
This study covers one task, one dataset-quality category, one distraction severity and a small fixed compute budget, without a hyperparameter search. Simple augmentation, consistency and masking already overlap published research; using only distracted data is not by itself a new contribution. A journal submission needs a justified new contribution or sufficiently broad and rigorous empirical finding, stronger matched implementations, more tasks and seeds, and longer training or demonstrated convergence.

Choose the next intervention after inspecting actual failure patterns. Candidate future work includes task-aware learned masks, action-predictive representation objectives, or confidence-controlled consistency when augmentations remove useful content. Each would need comparison against relevant existing methods and controlled removal of its components. No method should be described as better before the completed evidence supports that narrower claim.

Important evaluation scope: dynamic=False is inherited from the suite default. The background frame, colour and camera distractions are static within an episode. fixed_distraction=False controls resetting, not temporal dynamics. Familiar/held-out means familiar versus unseen video-source images under this static-distraction protocol. Do not claim robustness to moving backgrounds or temporal distraction changes. This was verified in source and saved video samples; the evaluation protocol was not changed after seeing results.

Equal-time comparisons also reach different points of the source action-noise schedule: update is called with step=2*update_count, and the standard deviation decreases from1.0to0.1 over250,000 steps. Thus methods differ in noise and warmup progress as well as update counts. BC uses mean-action targets, while DrQv2+BC and the author comparator use sampled actions during training. Deterministic evaluation uses mean actions. This further limits an isolated representation-quality interpretation; matched-update or normalized-schedule controls are future work.


## Implementation settings

The study ran on an NVIDIA GTX 1650 with 4 GiB GPU memory, Python 3.11.9 and PyTorch 2.5.1+cu121. Installed software versions are archived in environment_snapshot.json.

| Setting | Core methods |
|---|---|
| Optimizer learning rate | 0.0003 |
| Feature dimension | 50 |
| Hidden dimension | 1024 |
| Batch size | 16 |
| Discount / n-step | 0.99 / 3 |
| Target update coefficient | 0.01 |
| BC configuration weight | 2.5, source Q-normalized actor loss |
| Image history | Three84x84 RGB frames |
| Action repeat | 2 |
| Training action standard deviation | Linear1.0to0.1 over250,000 source steps |
| Action noise clip | 0.3 |
| Additional consistency coefficient | 1 |
| RGB gain range | 0.6to1.4 perchannel, temporally shared |
| Fixed mask | Top16rows replaced by gray |
| Training seeds | 0,1,2 |



## III. Actual results

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



## IV. Interpretation and submission scope

Five methods have lower averages on the held-out sources. This is consistent with a generalization problem in this setup, but the small collection of evaluation seeds and source images limits the conclusion. InfoGating has the largest held-out mean and large variability; it does not show a drop between these particular condition averages. That reversal is not proof that unfamiliar images help: source-specific difficulty, seed variation and different architectures/objectives remain plausible explanations.

The ordinary RGB augmentation control has a higher mean than the DrQv2+BC baseline in both conditions. This suggests that basic augmentation is a necessary comparison before claiming a more complex new mechanism is useful. The present three-seed experiment cannot establish statistically reliable or general superiority. BC also exceeds the DrQv2+BC mean in both conditions, so reward-based policy improvement did not deliver a clear advantage at this compute budget.

The fixed mask and actor-only consistency do not establish a solution to the distraction problem. They show low held-out returns and considerable variation. The mask is a crude spatial intervention and the consistency gradients do not directly update the encoder. These implementation choices limit what the results say about semantic segmentation or learned invariant representations.

The evidence does not show that a newly proposed method improves on relevant existing research: no new adaptive method was trained, and the adapted published comparator has the highest held-out point estimate. A defensible next study should investigate the failures of basic baselines, strengthen comparator protocol matching, test moving distractions separately, and evaluate any new representation objective on additional tasks and seeds.

Evaluation uses static distractions within each episode. Performance against moving videos is untested. The raw return scale is a reward sum, not a percentage accuracy. The recorded dataset returns around703 are data-collection statistics and should not be treated as a directly matched test baseline.


For BC, the held-out mean changed from 7.55 (seed SD 6.66) with 10,000 transitions to 3.40 (seed SD 2.16) with 100,000 transitions under the matched 300-second budget. The difference of -4.15 is descriptive; these short runs do not establish a general data-scaling benefit.

For DRQ, the held-out mean changed from 7.01 (seed SD 9.68) with 10,000 transitions to 10.87 (seed SD 15.91) with 100,000 transitions under the matched 300-second budget. The difference of 3.87 is descriptive; these short runs do not establish a general data-scaling benefit.

## Occlusion findings

For bc, the 32-pixel central band changed return by -2.50 on average (paired-change seed SD 1.80). For drq, the 32-pixel central band changed return by -1.22 on average (paired-change seed SD 1.32). For strong, the 32-pixel central band changed return by -26.98 on average (paired-change seed SD 44.40). For consistency, the 32-pixel central band changed return by -2.01 on average (paired-change seed SD 1.93). For mask, the 32-pixel central band changed return by -3.38 on average (paired-change seed SD 2.58). For infogating, the 32-pixel central band changed return by -93.48 on average (paired-change seed SD 96.29). Only one evaluation seed per checkpoint was used for this stress test. Low unoccluded returns can create a floor effect; small deterioration does not by itself indicate robust control.

These observations compare practical interventions under equal update time. They do not isolate every architectural difference or establish a generally superior method. Seed variation and short training budgets require caution. Occlusion differences are paired closed-loop diagnostics, not proof that an encoder discarded causal or noncausal features. The earlier adaptive weighting proposal was not implemented. A submission requires a justified contribution, broader task coverage, stronger comparator protocol matching and expanded uncertainty analysis.

## References

[1] C. Lu, P. J. Ball, T. G. J. Rudner, J. Parker-Holder, M. A. Osborne, and Y. W. Teh, "Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations," Transactions on Machine Learning Research, 2023. https://arxiv.org/abs/2206.04779

[2] D. Yarats, R. Fergus, A. Lazaric, and L. Pinto, "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning," International Conference on Learning Representations, 2022. https://arxiv.org/abs/2107.09645

[3] J.-Y. Chang, T. Westfechtel, T. Osa, and T. Harada, "Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training," Transactions on Machine Learning Research, 2024. https://openreview.net/forum?id=dce6ZGkJ1Z

[4] S. Wan, Z. Chen, L. Gan, S. Feng, and D.-C. Zhan, "SeMOPO: Learning High-quality Model and Policy from Low-quality Offline Visual Datasets," Proceedings of the 41st International Conference on Machine Learning, PMLR, vol. 235, pp. 49867-49889, 2024. https://proceedings.mlr.press/v235/wan24b.html

[5] A. H. Guzel, I. Bogunovic, and J. Parker-Holder, "Synthetic Data is Sufficient for Zero-Shot Visual Generalization from Offline Data," Transactions on Machine Learning Research, 2025. https://arxiv.org/abs/2508.12356 (author surname: Guzel with umlaut on u).

[6] R. Islam et al., "Principled Offline RL in the Presence of Rich Exogenous Information," Proceedings of the 40th International Conference on Machine Learning, PMLR, vol. 202, pp. 14390-14421, 2023. https://proceedings.mlr.press/v202/islam23a.html

[7] M. Tomar, R. Islam, M. E. Taylor, S. Levine, and P. Bachman, "Ignorance is Bliss: Robust Control via Information Gating," Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2303.06121

[8] Y. Zhang, K. Chen, and Y. Liu, "Causal representation learning in offline visual reinforcement learning," Knowledge-Based Systems, vol. 320, article 113565, 2025. https://doi.org/10.1016/j.knosys.2025.113565
