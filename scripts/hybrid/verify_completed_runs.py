from pathlib import Path
import json,torch,math,hashlib
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030'
if __name__=='__main__':
 torch.set_num_threads(1);records=[]
 for p in sorted((O/'runs').glob('*/result.json')):
  result=json.loads(p.read_text());checkpoint=p.parent/'latest.pt';c=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert c['updates']==result['updates'] and abs(c['training_seconds']-result['training_seconds'])<1e-7
  assert c['mode']==result['mode']
  assert all(torch.isfinite(v).all() for state in c['models'].values() for v in state.values())
  assert all(torch.isfinite(v).all() for v in c['domain'].values())
  rows=[json.loads(x) for x in (p.parent/'evaluation.jsonl').read_text().splitlines()]
  keys=[(x['updates'],x['condition'],x['eval_seed']) for x in rows];assert len(keys)==len(set(keys))
  assert all(x['steps']==500 and math.isfinite(x['return']) for x in rows)
  if result['complete']:
   assert result['updates']==40000 and len(rows)==20
   for step in [20000,40000]:
    for cond in ['familiar_background','heldout_background']:
     assert {x['eval_seed'] for x in rows if x['updates']==step and x['condition']==cond}=={801,802,803,804,805}
  records.append({'run':p.parent.name,'updates':c['updates'],'finite_models':True,'metadata_matches':True,'unique_complete_eval_episodes':len(rows),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
  del c
 out={'completed_run_files_checked':len(records),'passed':True,'records':records,'scope':'Checks recorded completed runs only; not an assertion all planned runs are complete or method scientifically correct.'}
 (O/'completed_runs_audit.json').write_text(json.dumps(out,indent=2));print('VERIFIED',len(records),'completed run files')
