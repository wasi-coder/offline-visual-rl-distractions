from pathlib import Path
import json,statistics,re
from xml.sax.saxutils import escape
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,PageBreak,NextPageTemplate,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parents[2];D=R/'experiments/hybrid_20261030';O=D/'output/pdf';O.mkdir(parents=True,exist_ok=True)
a=json.loads((D/'aggregate.json').read_text(encoding='utf-8-sig'));probes=[json.loads(x) for x in (D/'frozen_probes.jsonl').read_text(encoding='utf-8-sig').splitlines()]
assert a['completed_runs']==12 and len(probes)==12
helper=(R/'scripts/baseline/build_final_reports.py').read_text(encoding='utf-8-sig');exec(helper[helper.index('W,H=A4'):helper.index('related=')].replace('29 October 2026','30 October 2026'))
body.fontSize=9.3;body.leading=11.0
original_clean=clean
def clean(s):
 s=original_clean(s)
 links={}
 def protect(match):
  key='URLPLACEHOLDER'+chr(65+len(links));links[key]=match.group(0);return key
 s=re.sub(r'https?://\S+',protect,s)
 s=re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])',' ',s)
 s=s.replace('completeepisodes','complete episodes')
 for key,url in links.items():s=s.replace(key,url)
 return s
methods=['inv','gate','adv_inv','adv_gate'];labels={'inv':'Inverse + BC','gate':'Gating + inverse + BC','adv_inv':'Alignment + inverse + BC','adv_gate':'Alignment + gating + inverse + BC'}
def row(m,c):return next(x for x in a['summary'] if x['method']==m and x['condition']==c)
def score(x):return f"{x['mean']:.2f} +/- {x['seed_sd']:.2f}"
lines=['## III. Results','',f"Total measured update time: {a['training_update_hours']:.3f} hours. Hardware: GTX1650 (4GiB); batch16, feature256, learning rate0.0001. Preparation, replay sampling, checkpointing and evaluation are excluded from the update timer. Windows restart interrupted probes after training completed.",'', '| Method | Familiar return | Held-out return |','|---|---:|---:|']
for m in methods:lines.append(f"|{labels[m]}|{score(row(m,'familiar_background'))}|{score(row(m,'heldout_background'))}|")
lines+=['','Each mean averages five evaluation episodes per training seed, then three training-seed means. +/- is sample SD across those three means, not confidence intervals.','', '| Method | Action probe MSE | Domain accuracy |','|---|---:|---:|']
for m in methods:
 pp=[x for x in probes if x['run'].startswith(m+'_seed')];vals=[x['action_mse'] for x in pp];acc=[x['domain_accuracy'] for x in pp]
 lines.append(f"|{labels[m]}|{statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}|{statistics.mean(acc):.4f}|")
effects=json.loads((D/'paired_effects.json').read_text(encoding='utf-8-sig'));primary=next(x for x in effects['effects'] if x['contrast']=='gating_under_alignment' and x['measurement']=='heldout_return');interaction=next(x for x in effects['effects'] if x['contrast']=='interaction' and x['measurement']=='heldout_return')
interpret=f"The paired held-out gating difference under alignment is {primary['mean_difference']:.2f} (seed SD {primary['sample_sd_difference']:.2f}); the interaction is {interaction['mean_difference']:.2f} (seed SD {interaction['sample_sd_difference']:.2f}). The hybrid's lower held-out mean does not support a control-performance benefit from adding learned gating under this budget. The feature probes must be interpreted separately: lower action MSE indicates accessible action information, while high domain accuracy shows that domain identity remains linearly decodable. Neither mask sparsity nor a training-domain classifier's loss proves domain invariance. These observations do not establish that gating prevents alignment from destroying useful control information."
related=(R/'docs/related_work_and_references.md').read_text(encoding='utf-8-sig');review=related.split('## Related work draft')[1].split('## References')[0];refs=related.split('## References')[1].split('## Source verification notes')[0]
method=(D/'methods_draft.md').read_text(encoding='utf-8-sig').split('## Fixed data protocol')[1].split('## References and scientific limits')[0]
abstract=f"We test learned information gating and gradient-reversal alignment in a controlled visual offline imitation study. Twelve runs use160,000distracted training transitions and40,000updates per seed, with three seeds for each condition. Final policies are evaluated in120episodes; an additional120intermediate episodes record learning progress. Alignment plus inverse prediction yields held-out return {score(row('adv_inv','heldout_background'))}; adding gating yields {score(row('adv_gate','heldout_background'))}. We report frozen-feature probes and matched component controls. Results do not support an improved held-out control claim for the hybrid under this protocol."
text='# Information Gating Under Adversarial Alignment: A Controlled Visual Offline Imitation Study\n\nResearch draft for supervisor review; author metadata and target-venue requirements remain to be supplied.\n\n## Abstract\n\n'+abstract+'\n\n## I. Motivation and related work\n\n'+review+'\n\n## II. Methods\n\n## Fixed data protocol\n\n'+method+'\n\n'+'\n'.join(lines)+'\n\n## IV. Interpretation\n\n'+interpret+'\n\n## V. Limitations and conclusion\n\nThis is one-task, three-seed evidence under a fixed budget. All policies use BC; there is no reward-learning critic. Consequently this is not a reproduction of the original domain-adversarial RL paper or a comparison with its published scores. Evaluation is static within episodes. Independent linear probes do not prove causal sufficiency. The exact-action-sequence audit excludes identical cross-split action trajectories but not approximate shared physical states. Intermediate checkpoints were not selected using test returns. The experiment provides a negative result for the proposed control benefit under this setup. Broader tasks, faithful published-method comparisons, alignment-strength sweeps and stronger representation diagnostics remain necessary before a broad conference claim.\n\n## References\n\n'+refs
(D/'Hybrid_actual_evidence_paper.md').write_text(text,encoding='utf-8');story=markdown(text,CW)
story += [NextPageTemplate('wide'),PageBreak(),para('Final results and independent feature probes',head)]
from PIL import Image as PILImage
for name in ['returns','probes']:
 with PILImage.open(D/'figures'/f'{name}.png') as im:iw,ih=im.size
 story += [Image(str(D/'figures'/f'{name}.png'),width=(W-2*M)*.94,height=ih/iw*(W-2*M)*.94),Spacer(1,12)]
story +=[para('Return whiskers show sample SD across three training-seed means; dots show individual seed means. Probe dots represent separate trained encoders. Domain chance accuracy is0.5. All plots use the final40k-update checkpoints.',small)]
document(O/'Hybrid_actual_evidence_paper.pdf','Gating and alignment: controlled offline imitation',True).build(story)
summary='# Supervisor report: hybrid experiment\n\nAll12training runs completed, each with40,000updates. The study used160,000training transitions from easy and medium distracted cheetah data. Validation and probe-test episodes were excluded from policy training. There were240simulator evaluation episodes, including120at the final checkpoint.\n\nWe compared action-prediction representations with and without learned gating and adversarial alignment. All policies copied recorded actions through behavioural cloning; rewards were used only to assess control.\n\n'+'\n'.join(lines)+'\n\n## What this means\n\n'+interpret+'\n\nThe experiment ran successfully, but that is different from proving the proposed method works better. The current evidence should be presented as a controlled negative finding, with limitations. It does not justify claiming a superior conference-ready algorithm. The original6method experiment remains a separate pilot with another training protocol; its numbers are not directly merged with this comparison.\n\n## Next steps\n\nInvestigate why domain identity remains decodable, test alignment strength and stronger probes, and evaluate additional tasks. Preserve unsuccessful conditions and report computational costs. The final paper needs human author review, an appropriate contribution claim, venue formatting, author metadata and disclosure of substantial AI-assisted implementation/writing as applicable to the target venue. No external submission was performed.\n'
(D/'Hybrid_supervisor_report.md').write_text(summary,encoding='utf-8');document(O/'Hybrid_supervisor_report.pdf','Hybrid study supervisor report',False).build(markdown(summary.replace('## III. Results','## Results'),W-2*M,widebody))
print('Built hybrid PDFs; visual and scientific QA still required.')

