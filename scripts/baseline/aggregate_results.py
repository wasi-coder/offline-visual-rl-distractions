from pathlib import Path
import json,csv,statistics,hashlib,math
root=Path(__file__).resolve().parents[2]
def readlines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
results=[]
for folder in ['runs','runs_10k','runs_100k_matched']:
 for path in sorted((root/folder).glob('*/result.json')):
  r=json.loads(path.read_text());r['run']=folder+'/'+path.parent.name;results.append(r)
evals=readlines(root/'reports/final_evaluation.jsonl');size=readlines(root/'reports/size_evaluation.jsonl')
validation_errors=[]
expected_runs={f'runs/{m}_seed{s}' for m in ['bc','drq','strong','consistency','mask','infogating'] for s in range(3)} | {f'{f}/{m}_seed{s}' for f in ['runs_10k','runs_100k_matched'] for m in ['bc','drq'] for s in range(3)}
actual_runs={r['run'] for r in results}
if actual_runs-expected_runs:validation_errors.append('Unexpected training runs: '+str(sorted(actual_runs-expected_runs)))
for r in results:
 budget=1200 if r['run'].startswith('runs/') else 300
 if not math.isfinite(r['training_seconds']) or r['training_seconds']<budget:validation_errors.append('Incomplete training budget: '+r['run'])
 if r['updates']<=0:validation_errors.append('No updates: '+r['run'])
 checkpoint='final.pt' if 'infogating' in r['run'] else 'latest.pt'
 if not (root/r['run']/checkpoint).exists():validation_errors.append('Missing checkpoint: '+r['run'])
expected_eval=set()
for run in expected_runs:
 suite='main' if run.startswith('runs/') else 'size'
 key=run.split('/',1)[1] if suite=='main' else run
 for condition in ['familiar_background','heldout_background']:
  for seed in [701,702,703]:expected_eval.add((suite,key,condition,seed,0))
 if suite=='main':
  for damage in [8,16,32]:expected_eval.add((suite,key,'heldout_background',701,damage))
actual_eval=set()
for suite,rows in [('main',evals),('size',size)]:
 for r in rows:
  key=(suite,r['run'],r['condition'],r['eval_seed'],r['damage_band_pixels'])
  if key in actual_eval:validation_errors.append('Duplicate evaluation: '+str(key))
  actual_eval.add(key)
  if not math.isfinite(r['return']) or r['steps']!=500:validation_errors.append('Non-finite return or incomplete episode: '+str(key))
if actual_eval-expected_eval:validation_errors.append('Unexpected evaluation keys: '+str(sorted(actual_eval-expected_eval)))
summary=[]
for label,rows in [('main',evals),('size',size)]:
 keys=sorted(set((r['mode'],r.get('dataset_transitions',100000),r['condition'],r['damage_band_pixels']) for r in rows))
 for mode,n,condition,damage in keys:
  selected=[r for r in rows if (r['mode'],r.get('dataset_transitions',100000),r['condition'],r['damage_band_pixels'])==(mode,n,condition,damage)]
  seeds=sorted(set(r['training_seed'] for r in selected));means=[statistics.mean(r['return'] for r in selected if r['training_seed']==s) for s in seeds]
  summary.append({'suite':label,'method':mode,'transitions':n,'condition':condition,'damage_band_pixels':damage,'training_seeds':len(seeds),'evaluation_episodes':len(selected),'mean_return':statistics.mean(means),'std_across_seed_means':statistics.stdev(means) if len(means)>1 else None,'seed_means':dict(zip(seeds,means))})
status={'completed_training_runs':len(results),'recorded_training_hours':sum(r['training_seconds'] for r in results)/3600,'main_evaluation_rows':len(evals),'size_evaluation_rows':len(size),'expected_main_rows':162,'expected_size_rows':72,'complete':len(results)==30 and len(evals)==162 and len(size)==72,'summary':summary,'run_results':results,'interpretation':'Descriptive mean and sample standard deviation across independent training-seed means. Equal time, not equal updates. Small N=3; no significance or causal claim.'}
status['validation_errors']=validation_errors
resource_summary=[]
for suite in ['runs','runs_10k','runs_100k_matched']:
 for method in sorted({r['run'].split('/')[1].split('_seed')[0] for r in results if r['run'].startswith(suite+'/')}):
  rr=[r for r in results if r['run'].startswith(suite+'/'+method+'_seed')]
  resource_summary.append({'suite':suite,'method':method,'runs':len(rr),'updates_min':min(r['updates'] for r in rr),'updates_max':max(r['updates'] for r in rr),'total_update_seconds':sum(r['training_seconds'] for r in rr),'peak_process_rss_bytes':max((r['peak_rss'] for r in rr if 'peak_rss' in r),default=None),'peak_torch_cuda_allocated_bytes':max((r['peak_cuda_allocated'] for r in rr if 'peak_cuda_allocated' in r),default=None)})
status['resource_summary']=resource_summary
status['missing_training_runs']=sorted(expected_runs-actual_runs)
status['missing_evaluation_count']=len(expected_eval-actual_eval)
status['complete']=actual_runs==expected_runs and actual_eval==expected_eval and not validation_errors
(root/'reports/aggregate.json').write_text(json.dumps(status,indent=2))
if summary:
 with (root/'reports/summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scripts/baseline','src','envs','configs'] for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.md']}
(root/'reports/code_hashes.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in status.items() if k not in ['summary','run_results','resource_summary']},indent=2))
