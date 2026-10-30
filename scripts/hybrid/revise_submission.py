from pathlib import Path
R=Path(__file__).resolve().parents[2];T=R/'experiments/hybrid_20261030/ieee_template'
p=T/'content.md';s=p.read_text(encoding='utf-8')
anchor='### Fixed data protocol'
addition='''Chang et al. [3] use a domain discriminator and reversed gradients to reduce visual differences in the learned features. Their method also uses DropBlock and is evaluated on more than one locomotion task. Section 9 identifies two important limits: two datasets must be available, and the effect of their quality and differences needs further study. Appendix D already considers two distracted datasets. Therefore, using only distracted observations is not, by itself, our contribution. We study a narrower question: whether a learned gate helps alignment when the action-prediction objective and training budget are held fixed.

Gradient reversal follows the domain-adversarial approach of Ganin et al. [9]. It encourages features that make domain classification harder while retaining information needed by the main task. In our study, that main representation task is inverse action prediction. This objective does not guarantee that all useful control information is preserved; the separate probes and simulator tests are needed to check that claim.

'''
s=s.replace('## Methods\n','## Methods\n\n'+addition,1)
for a,b in {'auditable experimental protocol':'recorded experimental procedure','exogenous':'external','informational':'information','parsimonious':'compact','preselected':'chosen in advance','Consequently':'As a result','instantiate':'create','indiscriminate':'unable to distinguish','41 st':'41st','40 th':'40th'}.items():s=s.replace(a,b)
s=s.replace('## Acknowledgment','The experiment code and small result records are available in our repository [10]. Large datasets, environments and checkpoints are excluded from ordinary Git history.\n\n## Acknowledgment')
s+='''
[9] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," Journal of Machine Learning Research, vol. 17, no. 59, pp. 1–35, 2016. https://jmlr.org/papers/v17/15-239.html

[10] W. alsafa and S. Kadery, "Offline Visual RL with Distractions," source code and experimental records, 2026. https://github.com/wasi-coder/offline-visual-rl-distractions
'''
p.write_text(s,encoding='utf-8')
(R/'docs/REFERENCE_CHECK.md').write_text('''# Reference checks, 30 October 2026

References 1, 2, 4, 5, 6, 7 and 9 resolved to matching primary-source titles and authors on arXiv, PMLR or JMLR. Reference 3 matches the supplied complete TMLR October 2024 PDF; OpenReview returned a browser-verification page. Reference 8 DOI could not be resolved by the browser tool in this session; do not describe that URL as verified working. It is retained as the previously recorded publisher reference, with no claim that its full paywalled paper was reviewed. Reference 10 is the user-supplied project repository. In-text references 1–10 are present; no claim of a new superior algorithm is made. Similarity screening has not been performed.
''',encoding='utf-8')
