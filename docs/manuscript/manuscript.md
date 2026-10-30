Abstract—Offline reinforcement learning from raw pixels is vulnerable to visual distractions: encoders trained on distracted recordings can learn background details that change during evaluation. Domain-adversarial alignment aims to reduce features that reveal the visual domain, while information gating learns to restrict the information used by a prediction objective. The two methods appear to support each other, but their objectives need not cooperate. We present a controlled two-by-two study combining learned gating with gradient-reversal alignment for offline imitation from distracted cheetah-run recordings, holding the action-prediction objective, data budget and evaluation protocol fixed across all conditions. Under matched budgets, the gated adversarial condition does not improve held-out control: its return is lower than alignment alone in every matched training seed, and frozen-feature probes show that domain identity remains linearly decodable in every condition. An exploratory moving-distraction test reverses the mean ranking, indicating that the negative finding is specific to the tested evaluation set. Under this protocol, gating does not resolve the alignment trade-off; we describe the evidence needed to support a stronger preservation claim.

## Introduction and Related Work

Offline visual control uses stored interactions to learn a policy when new training experience is unavailable. The images contain both useful information about the agent and changing visual details. A representation can therefore predict recorded behavior using correlations that do not transfer to another visual setting. Learning from a large recording is distinct from learning a representation that remains useful when background sources change [1].

Visual offline RL combines learning from a fixed action dataset with learning useful features from images. Lu et al. [1] introduced a visual offline benchmark and competitive adaptations of visual control algorithms. DrQ-v 2 [2] provides a data-augmented actor-critic foundation; adding behavioural regularization [12] is important to distinguish the offline baseline used here from the original online algorithm. The earlier pilot used the supplied DrQv2+BC implementation. The present matched study instead uses behavioural cloning without a reward-learning critic, to isolate representation learning from value learning.

Chang et al. [3] address visual distraction through domain-adversarial training. Their code and distracted dataset motivate this project. The present study examines simple controls when training only on distracted observations; it does not reproduce their complete proposed method. Avoiding separately clean data is a constraint of our experiment, not by itself evidence of novelty relative to all prior research.

SeMOPO [4] takes a model-based approach: it separates agent-related and external latent factors and estimates uncertainty on the agent-related component. This addresses the problem that moving distractors can make world-model uncertainty misleading. Our experiments use model-free policies and do not learn a world model, so they cannot claim to reproduce or outperform SeMOPO. The useful lesson is to separate information needed for control from background dynamics instead of treating all image prediction errors as equally important.

Güzel et al. [5] increase offline visual diversity through augmentation followed by diffusion-generated latent data. The earlier pilot used RGB gains rather than latent transition generation. No diffusion model is trained in the present study. This paper motivates testing whether simpler augmentation already helps before introducing a more complex representation method.

ACRO [6] uses multi-step inverse prediction to learn agent-centric representations from actions and future observations. Its guarantees depend on assumptions about external information, observability and data. InfoGating [7] learns to retain a small amount of task-relevant information through differentiable gating. The present study adapts author InfoGating components, with explicit protocol differences documented in the methods. Its measured performance must not be equated with the original paper's results. ACRO is reviewed but is not an independently trained comparator here.

MMCS [8] combines latent mask partitioning with sufficiency and conditional-independence objectives, decoupling representation learning from downstream control. This is substantially different from covering the top image rows with gray. We inspected its author implementation and accessible publisher description; the full paywalled paper was not obtained. MMCS was not trained in this study.

Domain alignment and information gating address different aspects of this problem. Alignment discourages representations that reveal visual-domain identity. Gating restricts the input while retaining a prediction objective. Their combination appears plausible, but its objectives need not cooperate: a mask can remove useful information or retain a shortcut, while a domain classifier may be fooled without eliminating all domain information. We investigate this interaction rather than assuming it is beneficial.

Our question is whether action-guided gating improves the trade-off between control-related information and visual-domain invariance when only distracted recordings are available. We use a matched two-by-two design: inverse prediction with behavioral cloning (Inv), the same model with gating (IG), adversarial alignment without gating (DA+Inv), and the complete combination (DA+IG). All conditions keep the inverse objective, policy objective, batch composition, update budget, and evaluation procedure fixed. This separates the effect of learning a gate from the effect of adding action-related supervision.

The contribution is a controlled experimental finding and a reproducible procedure, not an established superior algorithm. We combine complete-episode data separation, matched component controls, fresh simulator evaluations, and independent frozen-feature probes. The primary result is negative: the gated adversarial condition has a lower held-out control mean, a negative interaction between gating and alignment, and no demonstrated domain invariance. This finding applies to the tested setup and points to further checks of how alignment and gating interact.

## Methods

Chang et al. [3] use a domain discriminator and reversed gradients to reduce visual differences in the learned features. Their method also uses DropBlock and is evaluated on more than one locomotion task. Section 9 identifies two important limits: two datasets must be available, and the effect of their quality and differences needs further study. Appendix D already considers two distracted datasets. Therefore, using only distracted observations is not, by itself, our contribution. We study a narrower question: whether a learned gate helps alignment when the action-prediction objective and training budget are held fixed.

Gradient reversal follows the domain-adversarial approach of Ganin et al. [9]. It encourages features that make domain classification harder while retaining information needed by the main task. In our study, that main representation task is inverse action prediction. This objective does not guarantee that all useful control information is preserved; the separate probes and simulator tests are needed to check that claim.


### Fixed data protocol



We use easy and medium distracted cheetah-run medium-expert data [3], with background, camera and colour distractions following the Distracting Control Suite convention [11]. Each domain contributes 25 complete episodes from each of 8 shards. Within each source shard, 20 episodes train the representation/policy, 3 fit independent probes, and 2 evaluate probes. This yields 160,000 training, 24,000 probe-fit and 16,000 probe-test transitions. Original source observations are 84 x 84 RGB with 6 D actions; policies use 3-frame histories. No clean observations train the model. The previous easy-only study is contextual pilot evidence, not a matched baseline.

Episode partitions and SHA-256 hashes are recorded; the data-integrity checks and their limits are described in Section II-E.

### Architecture and objectives

All conditions use the supplied InfoGating encoder (feature dimension 256), inverse contrastive model and BC policy, with the same initialization seed list, batch size 16 and learning rate 0.0001. Each batch has 8 samples from each domain. Future observations remain within the sampled episode with the author adapter's horizon cap 15. The author inverse model creates 256 random negative actions; this count is independent of minibatch size.

The learned-gate condition adds the supplied UNet, mapping stacked images to retention values through relu(tanh(output)). Suppressed pixels are zero-filled. Half of current and future minibatches are replaced independently with unmasked images, following the author implementation. At inference the policy receives raw observations, following that implementation. No semantic segmentation labels are used.

The domain classifier is a 256-to 128-to 2 MLP with ReLU. We first train it using cross-entropy on detached encoder features. We then hold its parameters fixed during the representation update and reverse the feature gradient. In encoder/gate parameters the resulting effective alignment objective is to increase domain classification loss, while the classifier minimizes it. Alignment strength increases linearly to 0.1 over 10,000 updates.

Representation learning minimizes inverse contrastive loss plus optional gating penalties and the reversed domain gradient. Before 2,500 updates, masked-image branches are detached for representation learning and a negative 0.1 mask-mean term encourages retention. Afterwards the gate receives inverse/alignment gradients with 0.1 mask-mean and 0.001 squared-mask penalties. This is sparsity pressure, not a guarantee that only action-relevant pixels survive.

There is no reward-learning critic in this study. Section II-D describes the shared policy objective and which components receive updates.

### Budget and evaluation

Before training began, the time budget was revised and the target was set to 40,000 updates for each of three seeds across four conditions. The six primary runs were prioritized, followed by all six component controls; every run reached the target. The hard training cutoff is recorded in deadline_budget.json. Partial and unstarted runs are disclosed, not presented as full-budget runs.

We evaluate at 20,000 and 40,000 updates, with final 40,000 chosen in advance for the main comparison. Each condition uses 5 fresh physical seeds per familiar/held-out-source setting. Evaluation episodes do not enter replay. Distractions are static within each episode. We average test episodes within each training seed before calculating the mean and standard deviation across seeds. Matched-seed differences quantify the gating effect and its interaction with alignment.

Independent probes fit fixed-penalty linear ridge models on frozen raw-input encoder features from probe-fit episodes and test on separate probe-test episodes. The measurements are action MSE, constant-action reference error, domain accuracy and feature variance. These test linear accessibility, not universal sufficiency or causal invariance. No test-based selection of encoder checkpoint or probe penalty is allowed.



### Objectives and comparison logic

Let x denote the current three-frame history, a the recorded action, and x_k a future history from the same episode. The encoder f produces a 256-dimensional vector. The inverse contrastive model scores the action together with current and future features. Its generated negative actions are independent of the minibatch size. A behavioral policy predicts the recorded action using detached encoder features. Rewards are retained in the dataset but do not update a critic or policy in this study.

The effective representation objective can be written as follows, with nonnegative gating and alignment coefficients. The negative domain term refers to the encoder and gate update, whereas the classifier itself minimizes domain cross-entropy.

Lrep = Linv + λg Lgate − λd Ldom.  (1)

The domain classifier first learns from detached features. Its parameters are then held fixed while the reversed gradient updates the representation. Thus, the classifier and representation optimize opposing directions for the same domain signal. The ungated counterpart receives the same inverse and reversed domain objectives without a learned retention map. The comparison tests whether the extra gate improves this particular trade-off, rather than whether predicting actions is useful in general.

The policy objective is MSE between a sampled policy action and the recorded action. The encoder is detached from this objective in every condition, matching the adopted author-code convention. The four policies therefore share the same form of learning rule. Gate parameters are created for all conditions to retain the common initialization procedure, but unused gates are not updated. Domain heads are fitted for diagnostics in all conditions; only DA conditions send reversed gradients to the encoder.

Synthetic post-warmup checks confirmed the intended update paths. Encoder, inverse-model, policy, and domain-head parameters changed in both primary conditions. Gate parameters changed only for DA+IG. A separate sign check verified gradient reversal, and checkpoint restoration preserved the recorded model and optimizer states. These checks establish implementation properties; they do not demonstrate useful control or information preservation.

### Data integrity and statistical unit

Each selected episode contains 500 transitions and one reset record. Across the two domains, 320 episodes train the models, 48 episodes fit independent probes, and 32 episodes test those probes. Episode indices are partitioned within each source shard before optimization. The data audit checks complete boundaries, finite rewards and actions, observation dimensions, action bounds, and file hashes. A second audit compares entire 500-step action sequences and finds no identical sequence across the training and held-out partitions.

The hash audit is deliberately narrower than a guarantee of independent simulator states. Closely related or approximately duplicated trajectories could remain across source conditions. Similarly, the association between archived training pixels and background videos follows the supplied generator convention; the original video identity was not independently reconstructed for every frame. These limitations are recorded alongside the manifests rather than hidden by the absence of exact duplicates.

We measure variation across independently trained agents. Five test episodes are first averaged within each training seed; mean and sample SD are then calculated across three seed means. Fifteen final episodes in a condition are not fifteen independent training repetitions. The probe uses 4,000 sampled pairs per split, but these can repeat within episodes. Probe sample counts likewise must not be interpreted as independent trajectory counts.

## Results

Total measured update time: 6.376 hours. Hardware: GTX 1650 (4 GiB); batch 16, feature 256, learning rate 0.0001. Preparation, replay sampling, checkpointing and evaluation are excluded from the update timer. An environment restart interrupted probe computation after training completed.

| Method | Familiar return | Held-out return |
|---|---:|---:|
|Inverse + BC|115.06 +/- 21.78|142.36 +/- 45.05|
|Gating + inverse + BC|121.41 +/- 47.27|129.18 +/- 8.81|
|Alignment + inverse + BC|92.24 +/- 57.96|157.73 +/- 38.63|
|Alignment + gating + inverse + BC|106.27 +/- 14.97|109.61 +/- 9.51|

Fig. 1 visualises the same comparison; dots show individual training-seed means. Each mean averages five evaluation episodes per training seed, then three training-seed means. +/- is sample SD across those three means, not confidence intervals.

| Method | Action probe MSE | Domain accuracy |
|---|---:|---:|
|Inverse + BC|0.1518 +/- 0.0009|1.0000|
|Gating + inverse + BC|0.1570 +/- 0.0017|1.0000|
|Alignment + inverse + BC|0.1499 +/- 0.0020|1.0000|
|Alignment + gating + inverse + BC|0.1539 +/- 0.0007|1.0000|

Fig. 2 shows the per-encoder action probe errors underlying Table II.

### Additional comparisons

The paired comparison asks how much the return changes when gating is added to the same training seed. Figure 3 shows that all three differences under alignment are negative. The three seeds are too few for a broad claim, but this makes the observed direction clear.

FIGURE_PAIRED

The methods use the same number of updates, but gating takes more computation per update. Table III reports measured update time only; data loading, evaluation and checkpoint writing are excluded. These values describe this GPU and implementation.

| Method | Updates per run | Update minutes |
|---|---:|---:|
| Inv | 40,000 | 16.70 +/- 0.01 |
| IG | 40,000 | 46.82 +/- 0.03 |
| DA+Inv | 40,000 | 16.98 +/- 0.44 |
| DA+IG | 40,000 | 47.01 +/- 0.06 |

### Exploratory moving-distraction test

We also ran a separate stress test using five fresh evaluation seeds (901–905). Static and moving settings share these seeds, but the moving setting changes background, camera and colour together. Table IV reports means and sample SD across three trained seeds, with five episodes per seed and condition. These results were not used to select checkpoints.

| Method | Static return | Moving return |
|---|---:|---:|
| Inv | 77.92 +/- 38.01 | 70.53 +/- 37.67 |
| IG | 95.03 +/- 4.82 | 61.02 +/- 5.85 |
| DA+Inv | 46.76 +/- 20.21 | 73.18 +/- 19.81 |
| DA+IG | 88.61 +/- 38.32 | 79.05 +/- 4.05 |

The method ranking changes in this secondary test: the combined model has a higher mean than alignment alone here, unlike the primary test. The static returns also change with the fresh evaluation seeds. This shows that the primary negative finding is specific to the tested evaluation set; the results do not establish that gating is always harmful or reliably helpful. Because several distraction types change together, this test cannot identify which one caused the differences.

## Interpretation

The paired held-out gating difference under alignment is -48.11 (seed SD 29.13); the interaction is -34.93 (seed SD 29.85). The hybrid's lower held-out mean does not support a control-performance benefit from adding learned gating under this budget. The feature probes must be interpreted separately: lower action MSE indicates accessible action information, while high domain accuracy shows that domain identity remains linearly decodable. Neither mask sparsity nor a training-domain classifier's loss proves domain invariance. These observations do not establish that gating prevents alignment from destroying useful control information.

### What would support the preservation hypothesis

A stronger preservation claim would require compatible evidence from control performance and representation diagnostics. Gating would need to improve the aligned policy while retaining a useful action signal, and its benefit should differ from the gain obtained without alignment. Our paired interaction addresses the latter question, but its unfavorable estimate does not support the expected complementarity. Merely reducing the variance of return, or displaying a visually plausible mask, would not establish that the required information survived.

The probes also constrain the interpretation. All action probes perform substantially better than the constant-action reference MSE of approximately 0.6689, so the measured action signal remains accessible in every condition. However, DA+IG does not improve on DA+Inv's action-prediction error. Perfect domain decoding by every independently fitted domain probe shows that the raw features still reveal the domain to this predictor family. It does not identify which features cause policy errors, but it prevents interpreting the experiment as successful removal of decodable domain identity.

Several explanations remain plausible. The alignment strength or discriminator optimization may be insufficient, the gate may favor features that do not improve deployment, or the raw-input inference convention may differ from the feature distribution emphasized by masked training. Domain–state correlations can also limit marginal alignment. These are hypotheses for later controlled tests. The present results do not isolate any one as the demonstrated cause.

## Limitations and Conclusion

This is one-task, three-seed evidence under a fixed budget. All policies use BC; there is no reward-learning critic. As a result this is not a reproduction of the original domain-adversarial RL paper or a comparison with its published scores. Evaluation is static within episodes. Independent linear probes do not prove causal sufficiency. The exact-action-sequence audit excludes identical cross-split action trajectories but not approximate shared physical states. Intermediate checkpoints were not selected using test returns. The experiment provides a negative result for the proposed control benefit under this setup. Broader tasks, faithful published-method comparisons, alignment-strength sweeps and stronger representation diagnostics remain necessary before a broad conference claim.

The experiment code and small result records are available in our repository [10]. Large datasets, environments and checkpoints are excluded from ordinary Git history.

## Acknowledgment

AI assistance was used for implementation, experiment orchestration, analysis code, and manuscript drafting. Experimental outcomes are drawn from saved runs, and no synthetic performance values are reported.

## References



[1] C. Lu, P. J. Ball, T. G. J. Rudner, J. Parker-Holder, M. A. Osborne, and Y. W. Teh, "Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations," Transactions on Machine Learning Research, 2023. https://arxiv.org/abs/2206.04779

[2] D. Yarats, R. Fergus, A. Lazaric, and L. Pinto, "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning," International Conference on Learning Representations, 2022. https://arxiv.org/abs/2107.09645

[3] J.-Y. Chang, T. Westfechtel, T. Osa, and T. Harada, "Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training," Transactions on Machine Learning Research, 2024. https://openreview.net/forum?id=dce6ZGkJ1Z

[4] S. Wan, Z. Chen, L. Gan, S. Feng, and D.-C. Zhan, "SeMOPO: Learning High-quality Model and Policy from Low-quality Offline Visual Datasets," Proceedings of the 41st International Conference on Machine Learning, PMLR, vol. 235, pp. 49867-49889, 2024. https://proceedings.mlr.press/v235/wan24b.html

[5] A. H. Güzel, I. Bogunovic, and J. Parker-Holder, "Synthetic Data is Sufficient for Zero-Shot Visual Generalization from Offline Data," Transactions on Machine Learning Research, 2025. https://arxiv.org/abs/2508.12356 (author surname: Güzel with umlaut on u).

[6] R. Islam et al., "Principled Offline RL in the Presence of Rich Exogenous Information," Proceedings of the 40th International Conference on Machine Learning, PMLR, vol. 202, pp. 14390-14421, 2023. https://proceedings.mlr.press/v202/islam23a.html

[7] M. Tomar, R. Islam, M. E. Taylor, S. Levine, and P. Bachman, "Ignorance is Bliss: Robust Control via Information Gating," Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2303.06121

[8] Y. Zhang, K. Chen, and Y. Liu, "Causal representation learning in offline visual reinforcement learning," Knowledge-Based Systems, vol. 320, article 113565, 2025. https://doi.org/10.1016/j.knosys.2025.113565

[9] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," Journal of Machine Learning Research, vol. 17, no. 59, pp. 1–35, 2016. https://jmlr.org/papers/v17/15-239.html

[10] W. alsafa and S. Kadery, "Offline Visual RL with Distractions," source code and experimental records, 2026. https://github.com/wasi-coder/offline-visual-rl-distractions


[11] A. Stone, O. Ramirez, K. Konolige, and R. Jonschkowski, "The Distracting Control Suite: A Challenging Benchmark for Reinforcement Learning from Pixels," arXiv:2101.02722, 2021. https://arxiv.org/abs/2101.02722

[12] S. Fujimoto and S. S. Gu, "A Minimalist Approach to Offline Reinforcement Learning," Advances in Neural Information Processing Systems, 2021. https://arxiv.org/abs/2106.06860
