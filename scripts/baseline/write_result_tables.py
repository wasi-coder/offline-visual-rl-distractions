"""Build traceable result tables and accessible summary from verified aggregate only."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[2];report=ROOT/'reports'
a=json.loads((report/'aggregate.json').read_text())
if not a['complete'] or a['validation_errors']:
 raise SystemExit('Study incomplete or invalid: refusing to generate final result claims.')
names={'bc':'Behavioural cloning','drq':'DrQv2+BC','strong':'Stronger RGB augmentation','consistency':'Two-view policy consistency','mask':'Fixed top-band mask','infogating':'Adapted author InfoGating*'}
methods=list(names)
def row(suite,m,condition,n=100000):
 return next(r for r in a['summary'] if r['suite']==suite and r['method']==m and r['condition']==condition and r['transitions']==n and r['damage_band_pixels']==0)
def score(r):return f"{r['mean_return']:.2f} +/- {r['std_across_seed_means']:.2f}"
lines=['# Verified results tables','',f"Completed training: {a['completed_training_runs']} runs; {a['recorded_training_hours']:.4f} measured update hours. Main evaluation: {a['main_evaluation_rows']} episodes; size evaluation: {a['size_evaluation_rows']} episodes.",'','All +/- values are sample standard deviations across three training-seed means, not standard errors or confidence intervals. Each ordinary condition uses three evaluation episodes per training seed. Returns are raw reward sums.','', '## Main comparison: 100,000 transitions, 1,200 update seconds per run','', '| Method | Familiar sources | Held-out sources |','|---|---:|---:|']
for m in methods:lines.append(f"| {names[m]} | {score(row('main',m,'familiar_background'))} | {score(row('main',m,'heldout_background'))} |")
lines+=['','*InfoGating uses adapted author code with batch 16 rather than default 256, different training/evaluation protocol from the paper and different architecture/objectives from DrQv2+BC. This table does not establish improvement over published InfoGating.','', '## Dataset-size pilot: matched 300 update seconds per run','','| Method | Transitions | Familiar sources | Held-out sources |','|---|---:|---:|---:|']
for m in ['bc','drq']:
 for n in [10000,100000]:lines.append(f"| {names[m]} | {n:,} | {score(row('size',m,'familiar_background',n))} | {score(row('size',m,'heldout_background',n))} |")
lines+=['','## Compute','','| Suite | Method | Update range | Total update seconds | Peak process RSS (GiB) | Peak PyTorch CUDA allocation (MiB) |','|---|---|---:|---:|---:|---:|']
for r in a['resource_summary']:
 rss='unmeasured' if r['peak_process_rss_bytes'] is None else f"{r['peak_process_rss_bytes']/2**30:.2f}"
 gpu='unmeasured' if r['peak_torch_cuda_allocated_bytes'] is None else f"{r['peak_torch_cuda_allocated_bytes']/2**20:.1f}"
 lines.append(f"| {r['suite']} | {names[r['method']]} | {r['updates_min']:,}-{r['updates_max']:,} | {r['total_update_seconds']:.2f} | {rss} | {gpu} |")
lines+=['','Peak process RSS and allocated tensor memory are different measurements; the CUDA figure excludes driver/context and other allocations. Budgets match update time, not update count.','', '## Occlusion stress test: paired return change','','Only evaluation seed 701 is used. The gray central band is not a verified semantic mask.','', '| Method | 8-pixel band | 16-pixel band | 32-pixel band |','|---|---:|---:|---:|']
raw=[json.loads(x) for x in (report/'final_evaluation.jsonl').read_text().splitlines() if x.strip()]
paired=[]
for m in methods:
 cells=[]
 for d in [8,16,32]:
  diffs=[]
  for s in range(3):
   def value(b):return next(r['return'] for r in raw if r['mode']==m and r['training_seed']==s and r['condition']=='heldout_background' and r['eval_seed']==701 and r['damage_band_pixels']==b)
   diffs.append(value(d)-value(0))
  cells.append(f'{statistics.mean(diffs):.2f} +/- {statistics.stdev(diffs):.2f}')
  paired.append({'method':m,'band_pixels':d,'per_training_seed_change':diffs,'mean_change':statistics.mean(diffs),'sd_change':statistics.stdev(diffs)})
 lines.append('| '+names[m]+' | '+' | '.join(cells)+' |')
(report/'results_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(report/'occlusion_paired_results.json').write_text(json.dumps(paired,indent=2))
best=max(methods,key=lambda m:row('main',m,'heldout_background')['mean_return'])
summary=['# Supervisor summary: actual offline RL experiments','', 'My research goal is to learn visual control from offline data containing distractions, without requiring a separately clean training dataset. Before this study I learned RL fundamentals, implemented a Snake agent by following a tutorial, and began studying the UC Berkeley deep RL course. These learning activities are self-reported background; the experiment evidence below comes from saved machine logs.','',f"The study completed {a['completed_training_runs']} training runs and {a['recorded_training_hours']:.2f} hours of measured training updates. Loading data, evaluations and writing files took additional time. The main study used 100,000 transitions from 200 complete cheetah episodes, with three training seeds for each of six methods. A smaller pilot compared 10,000 and 100,000 transitions under the same short training budget.",'','The agent saw recorded images, actions and rewards during offline training. Afterwards it acted in the simulator for evaluation. Those new simulator transitions were not used to train it. This tests whether a policy learned from stored data can control the simulated cheetah. It does not test a physical robot.','',f"The highest observed held-out-source mean in this particular main comparison was {names[best]}: {score(row('main',best,'heldout_background'))}. This is a descriptive observation, not proof that this algorithm is generally best. See the table for all seed variation and comparator qualifications.",'']
summary+=lines[lines.index('## Main comparison: 100,000 transitions, 1,200 update seconds per run'):lines.index('## Dataset-size pilot: matched 300 update seconds per run')]
summary+=['','A higher return means more reward from running during an evaluation episode. The +/- number describes how much the average result changed between the three independently trained agents. A small mean does not prove that an algorithm cannot learn: these runs were short and hyperparameters were not tuned. A high mean with large variation can indicate unreliable learning under this setup.','', 'The fixed mask simply hides the top of the image. It does not identify the cheetah. The consistency method encourages similar actions for colour-changed images, but does not directly enforce an invariant encoder. The adapted InfoGating comparison uses learned masks and action-related representation learning, but its batch size and protocol differ from the published paper.','', 'These experiments support a pilot thesis study and a draft empirical paper. They do not yet establish a novel method or a submission-ready journal contribution. Next steps are to explain the observed failures, implement a justified improvement, compare against stronger faithful baselines, and evaluate more tasks, seeds and training budgets. Adaptive masking/consistency remains proposed future work.','', 'Evidence: results_tables.md, aggregate.json, final_evaluation.jsonl, size_evaluation.jsonl, data_audit.json, sampler_validation.json, manifests, training logs and saved checkpoints. Videos show simulator rendering, which may differ from the exact transformed policy input.']
(report/'Supervisor_actual_results_summary.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
print('Wrote verified result tables, paired occlusion data and supervisor summary.')
