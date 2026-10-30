# Hybrid methods draft: matched action prediction and adversarial alignment

## Research question and status

Can learned information gating improve the trade-off between action-related information and visual-domain invariance in distracted offline learning? We compare the same inverse-prediction representation and BC policy with and without learned gating and gradient-reversal alignment. This is an adapted experiment, not a reproduction of the domain-adversarial paper's full RL algorithm or the published InfoGating protocol. Results must be inserted only after recorded evaluations and probes pass checks.

## Fixed data protocol

Use easy and medium distracted cheetah-run medium-expert data. Each domain contributes25completeepisodes from each of8shards. Within each source shard,20episodes train the representation/policy,3fit independent probes, and2evaluate probes. This yields160,000training,24,000probe-fit and16,000probe-test transitions. Original source observations are84x84 RGB with6D actions; policies use3-frame histories. No clean observations train the model. The previous easy-only study is contextual pilot evidence, not a matched baseline.

Episode partitions and SHA256 hashes are recorded. An audit validates48splitfiles, complete500transition episodes, valid shapes, finite bounded actions and disjoint within-source episode IDs. Matching500-step action-sequence hashes finds no exact cross-split duplicates; this does not prove independence of approximately related simulator states across archives.

## Architecture and objectives

All conditions use the supplied InfoGating encoder (feature dimension256), inverse contrastive model and BC policy, with the same initialization seed list, batch size16 and learning rate0.0001. Each batch has8samples from each domain. Future observations remain within the sampled episode with the author adapter's horizon cap15. The author inverse model creates256random negative actions; this count is independent of minibatch size.

The learned-gate condition adds the supplied UNet, mapping stacked images to retention values through relu(tanh(output)). Suppressed pixels are zero-filled. Half of current and future minibatches are replaced independently with unmasked images, following the author implementation. At inference the policy receives raw observations, following that implementation. No semantic segmentation labels are used.

The domain classifier is a256-to128-to2 MLP with ReLU. First train it by cross-entropy on detached encoder features. Freeze its parameters during the representation update, and reverse the feature gradient. In encoder/gate parameters the resulting effective alignment objective is to increase domain classification loss, while the classifier minimizes it. Alignment strength increases linearly to0.1 over10,000updates.

Representation learning minimizes inverse contrastive loss plus optional gating penalties and the reversed domain gradient. Before2,500updates, masked-image branches are detached for representation learning and a negative0.1mask-mean term encourages retention. Afterwards the gate receives inverse/alignment gradients with0.1mask-mean and0.001squared-mask penalties. This is sparsity pressure, not a guarantee that only action-relevant pixels survive.

The BC policy minimizes MSE between sampled policy actions and recorded actions. Encoder features are detached from this policy objective in every condition. There is no reward-learning critic in this study. Both ungated conditions instantiate the unused gate module to preserve the common initialization procedure; unused masks do not receive updates. Domain classifiers are fitted for diagnostics in all conditions; only adversarial conditions send reversed gradients to the encoder.

## Budget and evaluation

After a user-imposed deadline amendment before training began, target40,000updates for each of3seeds across4conditions. Run the6primary adversarial comparisons first, then the6component controls if the absolute budget allows. The hard training cutoff is recorded in deadline_budget.json. Partial and unstarted runs are disclosed, not presented as full-budget runs.

Evaluate at20,000and40,000updates, with final40,000preselected for the main comparison. Each condition uses5fresh physical seeds per familiar/held-out-source setting. Evaluation episodes do not enter replay. Distractions are static within each episode. Average test episodes within training seed before calculating across-seed means/SD. Report matched-seed gating and alignment effects and their interaction; see analysis_plan.md.

Independent probes fit fixed-penalty linear ridge models on frozen raw-input encoder features from probe-fit episodes and test on separate probe-test episodes. Measure action MSE, constant-action reference error, domain accuracy and feature variance. These test linear accessibility, not universal sufficiency or causal invariance. No test-based selection of encoder checkpoint or probe penalty is allowed.

## References and scientific limits

Primary methods: Chang et al., Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training (TMLR2024), https://openreview.net/forum?id=dce6ZGkJ1Z ; Tomar et al., Ignorance is Bliss: Robust Control via Information Gating (NeurIPS2023), https://arxiv.org/abs/2303.06121 . Related work includes ACRO, MMCS and the V-D4RL benchmark; retain full bibliography from docs/related_work_and_references.md.

The experiment tests one task, two training conditions and a fixed short update budget. It does not establish improved published benchmark performance, broad dynamic-background robustness or convergence. Three training seeds give limited statistical precision. A successful return/probe interaction would support the proposed mechanism under this setup; a null or negative result is reported equally. A conference submission still needs an appropriate claim, novelty review and venue-specific requirements.
