from pathlib import Path
import time,sys,subprocess,json,statistics,os
import psutil
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030'
def training_alive():
 for p in psutil.process_iter(['cmdline']):
  try:
   if any(str(x).replace('\\','/').endswith('/scripts/hybrid/train.py') for x in (p.info['cmdline'] or [])):return True
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 return False
if __name__=='__main__':
 while training_alive():time.sleep(30)
 os.environ['OPENBLAS_NUM_THREADS']='4'
 subprocess.run([sys.executable,str(R/'scripts/hybrid/probe_features.py')],check=True,cwd=R)
 results=[];summary=[]
 for p in sorted((O/'runs').glob('*/result.json')):
  x=json.loads(p.read_text());x['run']=p.parent.name;results.append(x)
 for mode in ['adv_inv','adv_gate','inv','gate']:
  valid=[r for r in results if r['mode']==mode and r.get('complete')]
  for cond in ['familiar_background','heldout_background']:
   means={}
   for run in valid:
    rows=[json.loads(x) for x in (O/'runs'/run['run']/'evaluation.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['updates']==40000 and r['condition']==cond]
    assert len(selected)==5 and len({r['eval_seed'] for r in selected})==5 and all(r['steps']==500 for r in selected)
    means[run['seed']]=statistics.mean(r['return'] for r in selected)
   if means:summary.append({'method':mode,'condition':cond,'training_seeds':len(means),'seed_means':means,'mean':statistics.mean(means.values()),'seed_sd':statistics.stdev(means.values()) if len(means)>1 else None})
 output={'completed_runs':sum(r.get('complete',False) for r in results),'expected_runs':12,'runs':results,'summary':summary,'training_update_hours':sum(r['training_seconds'] for r in results)/3600,'publication_status':'Preliminaryadaptedhybrid evidence; does not establishmechanism or conferencereadiness.'}
 (O/'aggregate.json').write_text(json.dumps(output,indent=2))
 lines=['# Hybrid study actual results','',f"Completed runs: {output['completed_runs']}/12; measured update hours: {output['training_update_hours']:.3f}.",'','| Method | Condition | Seeds | Mean return | Seed SD |','|---|---|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['method']}|{x['condition']}|{x['training_seeds']}|{x['mean']:.2f}|{x['seed_sd']}|")
 lines+=['','Figures and final paper still require scientific review. Only completed40000-update runs are compared. Partial/skippedruns must be disclosed. Data/probe audits and implementation differences remain limitations.']
 (O/'actual_results.md').write_text('\n'.join(lines)+'\n')
 print('PROBES AND BASIC AGGREGATION COMPLETE; FINAL SCIENTIFIC REVIEW/PAPER STILL REQUIRED',flush=True)
