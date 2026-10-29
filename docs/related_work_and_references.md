# Related work and verified references for manuscript assembly

## Related work draft
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

## Source verification notes
Checked primary publisher/arXiv/author pages on 2026-10-29. Synthetic-data OpenReview page returned browser verification; arXiv supplies its authors, abstract and TMLR 2025 journal reference. MMCS metadata verified at https://www.sciencedirect.com/science/article/pii/S0950705125006112. Original source repository https://github.com/Atine/offlineRL_domain_adversarial confirms the spelling Westfechtel. Author code references: https://github.com/manantomar/agent-centric-representations and https://github.com/DMU-XMU/MMCS . Bibliography numbering here should be retained consistently or regenerated when assembling the final paper.
