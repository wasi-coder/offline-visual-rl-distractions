"""Secondary paired static/dynamic held-out stress test; no model updates."""
from pathlib import Path
import sys,os,json
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';sys.path[:0]=[str(R/'scripts/hybrid'),str(R)]
import train as study
import torch,numpy as np
from src import dmc
if __name__=='__main__':
 torch.set_num_threads(4);out=O/'dynamic_evaluation.jsonl';done=set()
 if out.exists():done={(x['run'],x['seed'],x['dynamic']) for x in map(json.loads,out.read_text().splitlines())}
 os.environ['SMOKE_DAVIS_PATH']=str(R/'data/heldout_backgrounds')
 for p in sorted((O/'runs').glob('*/result.json')):
  r=json.loads(p.read_text())
  if not r.get('complete'):continue
  model=study.Model(r['mode'],int(p.parent.name.split('_seed')[1]));model.restore(torch.load(p.parent/'latest.pt',weights_only=False));model.train(False)
  for seed in [901,902,903,904,905]:
   for dynamic in [False,True]:
    if (p.parent.name,seed,dynamic) in done:continue
    kwargs={'easy':{'difficulty':'easy','fixed_distraction':False,'dynamic':dynamic,'background_seed':seed+1000,'camera_seed':seed+2000,'color_seed':seed+3000,'background_dataset_videos':['blackswan','bmx-trees','breakdance','camel']}}
    env=dmc.make('cheetah','run',3,2,seed,'easy',kwargs_dict=kwargs);ts=env.reset();ret=0;steps=0
    try:
     while not ts.last():
      with torch.no_grad():action=model.act(ts.observation,100000,True)
      assert np.isfinite(action).all() and abs(action).max()<=1.00001
      ts=env.step(action);ret+=float(ts.reward);steps+=1
    finally:
     if hasattr(env,'close'):env.close()
    assert steps==500
    record={'run':p.parent.name,'mode':r['mode'],'seed':seed,'training_seed':int(p.parent.name.split('_seed')[1]),'dynamic':dynamic,'return':ret,'steps':steps,'scope':'Exploratory secondary paired new-seed test; dynamic toggles backgrounds, camera and colour together. No model selection or training.'}
    with out.open('a') as f:f.write(json.dumps(record)+'\n')
    print(record,flush=True)
 print('SECONDARY DYNAMIC EVALUATION COMPLETE',flush=True)
