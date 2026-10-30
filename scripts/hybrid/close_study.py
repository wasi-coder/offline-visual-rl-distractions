"""Update local project navigation after all recorded analyses and manuscript QA."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030'
p=R/'README.md';text=p.read_text(encoding='utf-8-sig')
# Remove a malformed UTF-16 shell-appended duplicate heading, retaining the document.
if '\x00' in text:text=text[:text.index('#\x00')].rstrip()+'\n'
text=text.replace('**Planned, not implemented:** InfoGating plus domain-adversarial training.','**Completed:** controlled information-gating plus domain-adversarial offline imitation study (12 runs, 40,000 updates each).')
text=text.replace('configs/infogating_adversarial/: planned hybrid only.','configs/infogating_adversarial/: completed hybrid protocol.')
text=text.replace('CITATION.cff describes this software; author metadata and repository URL should be added when supplied by the project owner.','CITATION.cff includes the supplied author names and first email; affiliations and the second email remain unspecified.')
if '## Completed controlled hybrid study' not in text:
 text+='''
## Completed controlled hybrid study

Twelve runs, three seeds per condition, reached 40,000 updates using 160,000 distracted training transitions. This is BC with inverse representation learning, optional gating and gradient-reversal alignment; it is not the original full domain-adversarial RL method. Primary held-out return: alignment without gating 157.73 +/- 38.63; with gating 109.61 +/- 9.51. All raw-feature domain probes decoded domain perfectly, so neither improved control nor domain invariance is established.

See [hybrid study and reproduction caveats](docs/HYBRID_STUDY.md), [small evidence archive](results/hybrid_20261030) and its separate exploratory static/dynamic supplement. Verify hashes and primary arithmetic with `python scripts/hybrid/verify_export.py`. Prior pilot results above use a different protocol and must not be pooled with these results. Experiments, weights and data remain excluded from ordinary Git history.
'''
p.write_text(text,encoding='utf-8')
contract=O/'ieee_template/tmp/artifact.md'
c=contract.read_text(encoding='utf-8').replace('anonymousauthor block','supplied-author block').replace('Intentional author-slot simplification while names unknown.','Intentional author-slot simplification with both supplied names and the first email; affiliations and second email unknown.').replace('anonymous author slot','supplied names and email')
c+='\nFinal metadata contains the supplied names; docProps/core.xml title/creator/lastModifiedBy intentionally updated. All four final pages were inspected. Source styles/numbering/theme remain byte-identical. Final verification and hashes: ../verification.json.\n'
contract.write_text(c,encoding='utf-8')
with (O/'CONTINUATION.md').open('a',encoding='utf-8') as f:
 f.write('\nSTUDY COMPLETE 30 October 2026: all12training runs40k,240primary/intermediateevals,12frozenprobes and120secondarypaired static/dynamic episodes completed. Secondaryvalidated+summarizedseparately; no modelselection. Export53small numericalevidencefiles toresults/hybrid_20261030, verifiedhashes+primaryarithmeticwithverify_export.py. FinalIEEEWordtemplatepaper4A4pagesandPDFinieee_template; all4pagesvisuallychecked,sourceunchangedstyles/numbering/themeidentical. Suppliedbothauthornames+firstemail; affiliations/secondemailunknown. Similarityunmeasured,noapproval/submissioncertification. Negativeprimaryfinding retained; fulloriginalpaperNOTreproduced. No retraining or external submission.\n')
print('Updated README and completion record; no Git commit, push or external submission.')
