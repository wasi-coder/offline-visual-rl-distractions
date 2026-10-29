# Supervisor summary: actual offline RL experiments

My research goal is to learn visual control from offline data containing distractions, without requiring a separately clean training dataset. Before this study I learned RL fundamentals, implemented a Snake agent by following a tutorial, and began studying the UC Berkeley deep RL course. These learning activities are self-reported background; the experiment evidence below comes from saved machine logs.

The study completed 30 training runs and 7.00 hours of measured training updates. Loading data, evaluations and writing files took additional time. The main study used 100,000 transitions from 200 complete cheetah episodes, with three training seeds for each of six methods. A smaller pilot compared 10,000 and 100,000 transitions under the same short training budget.

The agent saw recorded images, actions and rewards during offline training. Afterwards it acted in the simulator for evaluation. Those new simulator transitions were not used to train it. This tests whether a policy learned from stored data can control the simulated cheetah. It does not test a physical robot.

The highest observed held-out-source mean in this particular main comparison was Adapted author InfoGating*: 59.74 +/- 57.04. This is a descriptive observation, not proof that this algorithm is generally best. See the table for all seed variation and comparator qualifications.

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


A higher return means more reward from running during an evaluation episode. The +/- number describes how much the average result changed between the three independently trained agents. A small mean does not prove that an algorithm cannot learn: these runs were short and hyperparameters were not tuned. A high mean with large variation can indicate unreliable learning under this setup.

The fixed mask simply hides the top of the image. It does not identify the cheetah. The consistency method encourages similar actions for colour-changed images, but does not directly enforce an invariant encoder. The adapted InfoGating comparison uses learned masks and action-related representation learning, but its batch size and protocol differ from the published paper.

These experiments support a pilot thesis study and a draft empirical paper. They do not yet establish a novel method or a submission-ready journal contribution. Next steps are to explain the observed failures, implement a justified improvement, compare against stronger faithful baselines, and evaluate more tasks, seeds and training budgets. Adaptive masking/consistency remains proposed future work.

Evidence: results_tables.md, aggregate.json, final_evaluation.jsonl, size_evaluation.jsonl, data_audit.json, sampler_validation.json, manifests, training logs and saved checkpoints. Videos show simulator rendering, which may differ from the exact transformed policy input.


## Dataset-size findings

For BC, the held-out mean changed from 7.55 (seed SD 6.66) with 10,000 transitions to 3.40 (seed SD 2.16) with 100,000 transitions under the matched 300-second budget. The difference of -4.15 is descriptive; these short runs do not establish a general data-scaling benefit.

For DRQ, the held-out mean changed from 7.01 (seed SD 9.68) with 10,000 transitions to 10.87 (seed SD 15.91) with 100,000 transitions under the matched 300-second budget. The difference of 3.87 is descriptive; these short runs do not establish a general data-scaling benefit.

## Evaluation scope

Background images, colours and camera distractions remain static within each episode (dynamic=False). These tests measure transfer to unseen source images, not robustness to moving distraction videos.

## Papers I reviewed

Visual offline RL combines learning from a fixed action dataset with learning useful features from images. Lu et al. [1] introduced a visual offline benchmark and competitive adaptations of visual control algorithms. DrQ-v2 [2] provides a data-augmented actor-critic foundation; adding behavioural regularization is important to distinguish the offline baseline used here from the original online algorithm. Our study uses the supplied DrQv2+BC implementation rather than claiming to introduce this baseline.

Chang et al. [3] address visual distraction through domain-adversarial training. Their code and distracted dataset motivate this project. The present study examines simple controls when training only on distracted observations; it does not reproduce their complete proposed method. Avoiding separately clean data is a constraint of our experiment, not by itself evidence of novelty relative to all prior research.

SeMOPO [4] takes a model-based approach: it separates agent-related and external latent factors and estimates uncertainty on the agent-related component. This addresses the problem that moving distractors can make world-model uncertainty misleading. Our experiments use model-free policies and do not learn a world model, so they cannot claim to reproduce or outperform SeMOPO. The useful lesson is to separate information needed for control from background dynamics instead of treating all image prediction errors as equally important.

Guzel et al. [5] increase offline visual diversity through augmentation followed by diffusion-generated latent data. Our RGB-gain control changes existing observations and does not generate additional latent transition data or train a diffusion model. This paper motivates testing whether simpler augmentation already helps before introducing a more complex representation method.

ACRO [6] uses multi-step inverse prediction to learn agent-centric representations from actions and future observations. Its guarantees depend on assumptions about exogenous information, observability and data. InfoGating [7] learns to retain a small amount of task-relevant information through differentiable gating. Our published comparator follows adapted author InfoGating code, with explicit batch-size and protocol differences documented in the methods. Its measured performance must not be equated with the original paper's results. ACRO is reviewed but is not an independently trained comparator here.

MMCS [8] combines latent mask partitioning with sufficiency and conditional-independence objectives, decoupling representation learning from downstream control. This is substantially different from covering the top image rows with gray. We inspected its author implementation and accessible publisher description; the full paywalled paper was not obtained. MMCS was not trained in this study.

Together, these works establish that agent-centric representation learning, information suppression and increased visual diversity are existing research directions. The current contribution is a limited empirical comparison and reproducible local pipeline. A future method must address a demonstrated failure and distinguish its objective or mechanism from these approaches, with appropriate controlled experiments.

## References

[1] C. Lu, P. J. Ball, T. G. J. Rudner, J. Parker-Holder, M. A. Osborne, and Y. W. Teh, "Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations," Transactions on Machine Learning Research, 2023. https://arxiv.org/abs/2206.04779

[2] D. Yarats, R. Fergus, A. Lazaric, and L. Pinto, "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning," International Conference on Learning Representations, 2022. https://arxiv.org/abs/2107.09645

[3] J.-Y. Chang, T. Westfechtel, T. Osa, and T. Harada, "Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training," Transactions on Machine Learning Research, 2024. https://openreview.net/forum?id=dce6ZGkJ1Z

[4] S. Wan, Z. Chen, L. Gan, S. Feng, and D.-C. Zhan, "SeMOPO: Learning High-quality Model and Policy from Low-quality Offline Visual Datasets," Proceedings of the 41st International Conference on Machine Learning, PMLR, vol. 235, pp. 49867-49889, 2024. https://proceedings.mlr.press/v235/wan24b.html

[5] A. H. Guzel, I. Bogunovic, and J. Parker-Holder, "Synthetic Data is Sufficient for Zero-Shot Visual Generalization from Offline Data," Transactions on Machine Learning Research, 2025. https://arxiv.org/abs/2508.12356 (author surname: Guzel with umlaut on u).

[6] R. Islam et al., "Principled Offline RL in the Presence of Rich Exogenous Information," Proceedings of the 40th International Conference on Machine Learning, PMLR, vol. 202, pp. 14390-14421, 2023. https://proceedings.mlr.press/v202/islam23a.html

[7] M. Tomar, R. Islam, M. E. Taylor, S. Levine, and P. Bachman, "Ignorance is Bliss: Robust Control via Information Gating," Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2303.06121

[8] Y. Zhang, K. Chen, and Y. Liu, "Causal representation learning in offline visual reinforcement learning," Knowledge-Based Systems, vol. 320, article 113565, 2025. https://doi.org/10.1016/j.knosys.2025.113565
