from pathlib import Path
import re
R=Path(__file__).resolve().parents[2];T=R/'experiments/hybrid_20261030/ieee_template';p=T/'content.md';s=p.read_text(encoding='utf-8')
abstract='''Abstract—Offline reinforcement learning from raw pixels is vulnerable to visual distractions: encoders trained on distracted recordings can bind the policy to background appearance that changes at evaluation. Domain-adversarial alignment aims to suppress domain-identifiable features, while information gating learns to restrict the information used by a prediction objective. The two mechanisms appear complementary, but their objectives need not cooperate. We present a controlled two-by-two study combining learned gating with gradient-reversal alignment for offline imitation from distracted cheetah-run recordings, holding the action-prediction objective, data budget and evaluation protocol fixed across all conditions. Under matched budgets, the gated adversarial condition does not improve held-out control: its return is lower than alignment alone in every matched training seed, and frozen-feature probes show that domain identity remains linearly decodable in every condition. An exploratory moving-distraction test reverses the mean ranking, indicating that the negative finding is specific to the tested evaluation set. Under this protocol, gating does not resolve the alignment trade-off; we describe the evidence needed to support a stronger preservation claim.'''
s=abstract+'\n\n'+s.split('## Introduction and Related Work',1)[1].join(['## Introduction and Related Work','']) if False else abstract+'\n\n## Introduction and Related Work'+s.split('## Introduction and Related Work',1)[1]
start=s.index('## Introduction and Related Work');end=s.index('## Methods')
paras=[x.strip() for x in s[start:end].split('\n\n') if x.strip()]
get=lambda prefix:next(x for x in paras if x.startswith(prefix))
ordered=[paras[0],get('Offline visual control'),get('Visual offline RL'),get('Chang et al.'),get('SeMOPO'),get('Güzel'),get('ACRO'),get('MMCS'),get('Domain alignment'),get('Our question'),get('The contribution')]
s=s[:start]+'\n\n'.join(ordered)+'\n\n'+s[end:]
s=s.replace('an recorded experimental procedure','a reproducible experimental procedure')
s=s.replace('adding behavioural regularization is important','adding behavioural regularization [12] is important')
s=s.replace('The present matched study instead uses BC to isolate representation learning.','The present matched study instead uses behavioural cloning without a reward-learning critic, to isolate representation learning from value learning.')
s=s.replace('We use easy and medium distracted cheetah-run medium-expert data.','We use easy and medium distracted cheetah-run medium-expert data [3], with background, camera and colour distractions following the Distracting Control Suite convention [11].')
s=s.replace('Windows restart interrupted probes after training completed.','An environment restart interrupted probe computation after training completed.')
old='Episode partitions and SHA 256 hashes are recorded. An audit validates 48 splitfiles, complete 500 transition episodes, valid shapes, finite bounded actions and disjoint within-source episode IDs. Matching 500-step action-sequence hashes finds no exact cross-split duplicates; this does not prove independence of approximately related simulator states across archives.'
s=s.replace(old,'Episode partitions and SHA-256 hashes are recorded; the data-integrity checks and their limits are described in Section II-E.')
old='The BC policy minimizes MSE between sampled policy actions and recorded actions. Encoder features are detached from this policy objective in every condition. There is no reward-learning critic in this study. Both ungated conditions create the unused gate module to preserve the common initialization procedure; unused masks do not receive updates. Domain classifiers are fitted for diagnostics in all conditions; only adversarial conditions send reversed gradients to the encoder.'
s=s.replace(old,'There is no reward-learning critic in this study. Section II-D describes the shared policy objective and which components receive updates.')
s=s.replace('Each mean averages five evaluation episodes per training seed, then three training-seed means.','Fig. 1 visualises the same comparison; dots show individual training-seed means. Each mean averages five evaluation episodes per training seed, then three training-seed means.')
s=s.replace('### Additional comparisons','Fig. 2 shows the per-encoder action probe errors underlying Table II.\n\n### Additional comparisons')
s+='''

[11] A. Stone, O. Ramirez, K. Konolige, and R. Jonschkowski, "The Distracting Control Suite: A Challenging Benchmark for Reinforcement Learning from Pixels," arXiv:2101.02722, 2021. https://arxiv.org/abs/2101.02722

[12] S. Fujimoto and S. S. Gu, "A Minimalist Approach to Offline Reinforcement Learning," Advances in Neural Information Processing Systems, 2021. https://arxiv.org/abs/2106.06860
'''
assert 'Fig. 1 visualises' in s
p.write_text(s,encoding='utf-8')
print('Review edits applied; retained publisher-verified ACRO title. No detector score claimed.')
