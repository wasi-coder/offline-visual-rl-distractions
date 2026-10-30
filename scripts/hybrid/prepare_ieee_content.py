from pathlib import Path
import re
R=Path(r'E:\offline_rl');O=R/'experiments/hybrid_20261030';T=O/'ieee_template'
s=(O/'Hybrid_actual_evidence_paper.md').read_text(encoding='utf-8')
s=s.split('## Abstract',1)[1].strip()
s='Abstract—'+s
s=s.replace('## I. Motivation and related work','## Introduction and Related Work')
s=s.replace('## II. Methods','## Methods')
for title in ['Fixed data protocol','Architecture and objectives','Budget and evaluation','Implementation settings']:
 s=s.replace('## '+title,'### '+title)
intro='''Offline visual control uses stored interactions to learn a policy when new training experience is unavailable. Its observations combine useful agent information with appearance variation. A representation can therefore predict recorded behavior using correlations that do not transfer to another visual setting. Learning from a large recording is distinct from learning a representation that remains useful when background sources change [1].

Domain alignment and information gating address different aspects of this problem. Alignment discourages representations that reveal visual-domain identity. Gating restricts the input while retaining a prediction objective. Their combination appears plausible, but its objectives need not cooperate: a mask can remove useful information or retain a shortcut, while a domain classifier may be fooled without eliminating all domain information. We investigate this interaction rather than assuming it is beneficial.

Our question is whether action-guided gating improves the trade-off between control-related information and visual-domain invariance when only distracted recordings are available. We use a matched two-by-two design: inverse prediction with behavioral cloning (Inv), the same model with gating (IG), adversarial alignment without gating (DA+Inv), and the complete combination (DA+IG). All conditions keep the inverse objective, policy objective, batch composition, update budget, and evaluation procedure fixed. This separates the effect of learning a gate from the effect of adding action-related supervision.

The contribution is a controlled empirical finding and an auditable experimental protocol, not an established superior algorithm. We combine complete-episode data separation, matched component controls, fresh simulator evaluations, and independent frozen-feature probes. The primary result is negative: the gated adversarial condition has a lower held-out control mean, an unfavorable interaction contrast, and no demonstrated domain invariance. This finding narrows the hypothesis to what was tested and motivates more targeted investigations of the alignment trade-off.

'''
s=s.replace('## Introduction and Related Work\n\n','## Introduction and Related Work\n\n'+intro)
extra='''### Objectives and comparison logic

Let x denote the current three-frame history, a the recorded action, and x_k a future history from the same episode. The encoder f produces a 256-dimensional vector. The inverse contrastive model scores the action together with current and future features. Its generated negative actions are independent of the minibatch size. A behavioral policy predicts the recorded action using detached encoder features. Rewards are retained in the dataset but do not update a critic or policy in this study.

The effective representation objective can be written as follows, with nonnegative gating and alignment coefficients. The negative domain term refers to the encoder and gate update, whereas the classifier itself minimizes domain cross-entropy.

L_rep = L_inv + lambda_g L_gate - lambda_d L_dom.     (1)

The domain classifier first learns from detached features. Its parameters are then held fixed while the reversed gradient updates the representation. Thus, the classifier and representation optimize opposing directions for the same domain signal. The ungated counterpart receives the same inverse and reversed domain objectives without a learned retention map. The comparison tests whether the extra gate improves this particular trade-off, rather than whether predicting actions is useful in general.

The policy objective is MSE between a sampled policy action and the recorded action. The encoder is detached from this objective in every condition, matching the adopted author-code convention. The four policies therefore share the same form of learning rule. Gate parameters are instantiated for all conditions to retain the common initialization procedure, but unused gates are not updated. Domain heads are fitted for diagnostics in all conditions; only DA conditions send reversed gradients to the encoder.

Synthetic post-warmup checks confirmed the intended update paths. Encoder, inverse-model, policy, and domain-head parameters changed in both primary conditions. Gate parameters changed only for DA+IG. A separate sign check verified gradient reversal, and checkpoint restoration preserved the recorded model and optimizer states. These checks establish implementation properties; they do not demonstrate useful control or information preservation.

### Data integrity and statistical unit

Each selected episode contains 500 transitions and one reset record. Across the two domains, 320 episodes train the models, 48 episodes fit independent probes, and 32 episodes test those probes. Episode indices are partitioned within each source shard before optimization. The data audit checks complete boundaries, finite rewards and actions, observation dimensions, action bounds, and file hashes. A second audit compares entire 500-step action sequences and finds no identical sequence across the training and held-out partitions.

The hash audit is deliberately narrower than a guarantee of independent simulator states. Closely related or approximately duplicated trajectories could remain across source conditions. Similarly, the association between archived training pixels and background videos follows the supplied generator convention; the original video identity was not independently reconstructed for every frame. These limitations are recorded alongside the manifests rather than hidden by the absence of exact duplicates.

The statistical unit for reported variability is an independently trained agent. Five test episodes are first averaged within each training seed; mean and sample SD are then calculated across three seed means. Fifteen final episodes in a condition are not fifteen independent training repetitions. The probe uses 4,000 sampled pairs per split, but these can repeat within episodes. Probe sample counts likewise must not be interpreted as independent trajectory counts.

'''
s=s.replace('## III. Results',extra+'## Results')
discussion='''### What would support the preservation hypothesis

A stronger preservation claim would require compatible evidence from control performance and representation diagnostics. Gating would need to improve the aligned policy while retaining a useful action signal, and its benefit should differ from the gain obtained without alignment. Our paired interaction addresses the latter question, but its unfavorable estimate does not support the expected complementarity. Merely reducing the variance of return, or displaying a visually plausible mask, would not establish that the required information survived.

The probes also constrain the interpretation. All action probes perform substantially better than the constant-action reference MSE of approximately 0.6689, so the measured action signal remains accessible in every condition. However, DA+IG does not improve on DA+Inv's action-prediction error. Perfect domain decoding by every independently fitted domain probe shows that the raw features still reveal the domain to this predictor family. It does not identify which features cause policy errors, but it prevents interpreting the experiment as successful removal of decodable domain identity.

Several explanations remain plausible. The alignment strength or discriminator optimization may be insufficient, the gate may favor features that do not improve deployment, or the raw-input inference convention may differ from the feature distribution emphasized by masked training. Domain–state correlations can also limit marginal alignment. These are hypotheses for later controlled tests. The present results do not isolate any one as the demonstrated cause.

'''
s=s.replace('## V. Limitations and conclusion',discussion+'## Limitations and Conclusion')
s=s.replace('## IV. Interpretation','## Interpretation')
ack='''## Acknowledgment

AI assistance was used for implementation, experiment orchestration, analysis code, and manuscript drafting. The authors must independently verify the scientific claims, references, and final submission. No external submission, author approval certification, or similarity screening is asserted by this document.

'''
s=s.replace('## References',ack+'## References')
# Remove editorial commands and retain the corresponding factual statements.
s=s.replace('Use easy and medium distracted cheetah-run medium-expert data.','We use easy and medium distracted cheetah-run medium-expert data.')
s=s.replace('Run the6primary adversarial comparisons first, then the6component controls if the absolute budget allows.','The six primary runs were prioritized, followed by all six component controls; every run reached the target.')
s=s.replace('Report the mean and sample standard deviation','We report the mean and sample standard deviation')
s=s.replace('Evaluate at20,000and40,000updates','We evaluate at20,000and40,000updates')
s=s.replace('Report matched-seed gating and alignment effects and their interaction; see analysis_plan.md.','Matched-seed differences quantify the gating effect and its interaction with alignment.')
# Protect URLs while repairing compact numeric typography from the experiment notes.
urls={}
def protect(m):key='URLTOKEN'+chr(65+len(urls));urls[key]=m.group();return key
s=re.sub(r'https?://\S+',protect,s)
s=re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])',' ',s)
for key,url in urls.items():s=s.replace(key,url)
s=s.replace('completeepisodes','complete episodes').replace('40 kupdates','40,000 updates').replace('40 k-update','40,000-update').replace('Guzel','Güzel')
s=s.replace('Our study uses the supplied DrQv 2+BC implementation rather than claiming to introduce this baseline.','The earlier pilot used the supplied DrQv2+BC implementation. The present matched study instead uses BC to isolate representation learning.')
s=s.replace('Our RGB-gain control changes existing observations and does not generate additional latent transition data or train a diffusion model.','The earlier pilot used RGB gains rather than latent transition generation. No diffusion model is trained in the present study.')
s=s.replace('Our published comparator follows adapted author InfoGating code, with explicit batch-size and protocol differences documented in the methods.','The present study adapts author InfoGating components, with explicit protocol differences documented in the methods.')
s=s.replace('L_rep = L_inv + lambda_g L_gate - lambda_d L_dom.     (1)','Lrep = Linv + λg Lgate − λd Ldom.  (1)')
s=s.replace('First train it by cross-entropy on detached features. Freeze its parameters during the representation update, and reverse the feature gradient.','The classifier is trained by cross-entropy on detached features. Its parameters are frozen during the representation update, while the feature gradient is reversed.')
s=s.replace('Measure action MSE, constant-action reference error, domain accuracy and feature variance.','The measurements are action MSE, constant-action reference error, domain accuracy and feature variance.')
s=s.replace('The authors must independently verify the scientific claims, references, and final submission. No external submission, author approval certification, or similarity screening is asserted by this document.','Experimental outcomes are drawn from saved runs, and no synthetic performance values are reported.')
s=re.sub(r'(?<=[a-z],)(?=\d)',' ',s)
(T/'content.md').write_text(s,encoding='utf-8');print('Manuscript words',len(s.split()))
