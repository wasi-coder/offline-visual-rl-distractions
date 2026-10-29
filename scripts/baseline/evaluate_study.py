from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/baseline'))
import core_suite as core
from core_suite import Agent
import torch,numpy as np,imageio.v2 as imageio
from src import dmc
BASE=core.BASE

def episode(agent,kind,condition,seed,damage=0,video=None):
 held=condition=='heldout_background'
 os.environ['SMOKE_DAVIS_PATH']=str(ROOT/'data/heldout_backgrounds') if held else os.environ.get('OFFLINE_RL_DAVIS_PATH',str(ROOT/'data/DAVIS/JPEGImages/480p'))
 kwargs={'easy':{'difficulty':'easy','fixed_distraction':False,'background_seed':seed+1000,'camera_seed':seed+2000,'color_seed':seed+3000}}
 if held:kwargs['easy']['background_dataset_videos']=['blackswan','bmx-trees','breakdance','camel']
 env=dmc.make('cheetah','run',3,2,seed,'easy',kwargs_dict=kwargs);ts=env.reset();ret=0;steps=0
 writer=imageio.get_writer(video,fps=25,codec='libx264',quality=6) if video else None
 try:
  while not ts.last():
   obs=ts.observation.copy()
   if kind=='mask':obs[:,:16,:]=128
   if damage:obs[:,42-damage//2:42+damage//2,:]=128
   with torch.no_grad():
    if kind=='infogating':a=agent.act(obs,100000,True)
    else:a,_=agent.act_with_logprob(obs/255.,100000,True)
   assert np.isfinite(a).all() and np.max(np.abs(a))<=1.00001
   if writer:writer.append_data(env.physics.render(height=256,width=256,camera_id=0))
   ts=env.step(a);ret+=float(ts.reward);steps+=1
 finally:
  if writer:writer.close()
  if hasattr(env,'close'):env.close()
 return {'condition':condition,'eval_seed':seed,'damage_band_pixels':damage,'return':ret,'steps':steps,'video':str(video) if video else None}
if __name__=='__main__':
 torch.set_num_threads(4)
 test='--validate' in sys.argv
 if not test:
  while not (ROOT/'runs/infogating_seed2/result.json').exists():time.sleep(30)
 records=[];out=ROOT/'reports/final_evaluation.jsonl'
 done=set()
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);done.add((r['run'],r['condition'],r['eval_seed'],r['damage_band_pixels']))
 runs=[ROOT/'runs/bc_seed0'] if test else sorted((ROOT/'runs').glob('*_seed*'))
 for run in runs:
  if not (run/'result.json').exists():continue
  kind=run.name.split('_seed')[0];seed=int(run.name.split('_seed')[1])
  if kind=='infogating':
   import published_comparator as pc
   agent=pc.mod.InfoGatingAgent((9,84,84),(6,),'cuda',1e-4,256,1024,.01,2000,2,'linear(1.0, 0.1, 250000)',.3,True,lam=.1)
   data=torch.load(run/'final.pt',weights_only=False)
  else:agent=core.build(kind,seed);data=torch.load(run/'latest.pt',weights_only=False)
  for n,v in data['models'].items():getattr(agent,n).load_state_dict(v)
  agent.train(False)
  cases=[('heldout_background',701,0)] if test else [(c,s,0) for c in ['familiar_background','heldout_background'] for s in [701,702,703]]+ [('heldout_background',701,d) for d in [8,16,32]]
  for condition,es,damage in cases:
   if (run.name,condition,es,damage) in done:continue
   videos=ROOT/'videos';videos.mkdir(exist_ok=True)
   path=videos/(run.name+'.mp4') if condition=='heldout_background' and es==701 and damage==0 else None
   result=episode(agent,kind,condition,es,damage,path);result['run']=run.name;result['training_seed']=seed;result['mode']=kind
   with (ROOT/'reports'/('evaluation_validation.jsonl' if test else 'final_evaluation.jsonl')).open('a') as f:f.write(json.dumps(result)+'\n')
   print(json.dumps(result),flush=True)
 print('EVALUATION DONE',flush=True)
