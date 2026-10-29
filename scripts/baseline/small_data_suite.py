from pathlib import Path
import os,sys,time,json,csv,random,traceback
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT
os.environ['MUJOCO_GL']='glfw';os.environ['SMOKE_DAVIS_PATH']=os.environ.get('OFFLINE_RL_DAVIS_PATH',str(ROOT/'data/DAVIS/JPEGImages/480p'));os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,psutil
import torch.nn.functional as F
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from src.utils import ReplayBuffer,load_offline_dataset_into_buffer,set_seed_everywhere,schedule
from src.agents.drqv2_default import DrQv2DefaultAgent
from src import dmc
class Agent(DrQv2DefaultAgent):
 def sample(self,replay,batch):
  obs,a,r,d,n=super().sample(replay,batch)
  if self.mode=='strong':
   gain=torch.empty(obs.shape[0],3,1,1,device=self.device).uniform_(.6,1.4).repeat(1,3,1,1)
   obs=(obs*gain).clamp(0,1);n=(n*gain).clamp(0,1)
  if self.mode=='mask':obs=obs.clone();n=n.clone();obs[:,:,:16,:]=.5;n[:,:,:16,:]=.5
  self.original=obs.detach()
  return obs,a,r,d,n
 def update_actor(self,obs,step,behavioural_action=None):
  if self.mode!='consistency':return super().update_actor(obs,step,behavioural_action)
  std=schedule(self.stddev_schedule,step);dist=self.actor(obs,std);action=dist.sample(clip=self.stddev_clip)
  q1,q2=self.critic(obs,action);q=torch.minimum(q1,q2);bc=F.mse_loss(action,behavioural_action)
  loss=-q.mean()*(self.bc_weight/q.detach().abs().mean().clamp_min(1e-6))+bc
  gain=torch.empty(self.original.shape[0],3,1,1,device=self.device).uniform_(.6,1.4).repeat(1,3,1,1)
  with torch.no_grad():
   features,_=self.encoder((self.original*gain).clamp(0,1));orig,_=self.encoder(self.original);target=self.actor(orig,std).mean
  cons=F.mse_loss(self.actor(features,std).mean,target);loss=loss+cons
  self.actor_opt.zero_grad();loss.backward();self.actor_opt.step()
  return {'actor_loss':loss.item(),'actor_bc_loss':bc.item(),'consistency_loss':cons.item()}
 def update(self,replay,step,batch):
  if self.mode!='bc':return super().update(replay,step,batch)
  obs,a,_,_,_=self.sample(replay,batch);obs,_=self.aug(obs.float(),shift=None);feat,_=self.encoder(obs)
  loss=F.mse_loss(self.actor(feat,schedule(self.stddev_schedule,step)).mean,a)
  self.encoder_opt.zero_grad();self.actor_opt.zero_grad();loss.backward();self.encoder_opt.step();self.actor_opt.step()
  return {'bc_loss':loss.item()}
def build(mode,seed):
 set_seed_everywhere(seed)
 with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/upstream')):
  cfg=compose(config_name='config',overrides=['algo=drqv2','task=cheetah_run_medium_expert_easy','batch_size=16','replay_buffer_size=102000','gpu=0','no_track=True'])
 cfg.algo.agent._target_='__main__.Agent';cfg.algo.agent.obs_dim=[9,84,84];cfg.algo.agent.action_dim=[6];cfg.algo.agent.device='cuda:0'
 agent=instantiate(cfg.algo.agent);agent.mode=mode;return agent

def evaluate(agent,seed):
 ans=[];agent.train(False)
 for mode in ['fixed_easy','easy']:
  env=dmc.make('cheetah','run',3,2,seed,mode,kwargs_dict={mode:{'difficulty':'easy','fixed_distraction':mode.startswith('fixed_'),'background_seed':seed+1000,'camera_seed':seed+2000,'color_seed':seed+3000}})
  ts=env.reset();ret=0;steps=0
  while not ts.last():
   obs=ts.observation/255.
   if agent.mode=='mask':obs=obs.copy();obs[:,:16,:]=.5
   with torch.no_grad():a,_=agent.act_with_logprob(obs,100000,True)
   assert np.isfinite(a).all();ts=env.step(a);ret+=float(ts.reward);steps+=1
  ans.append({'condition':mode,'seed':seed,'return':ret,'steps':steps,'background_note':'same four available background videos; unseen seed only, not held-out background source'})
  if hasattr(env,'close'):env.close()
 agent.train(True);return ans

def save(agent,path,step,elapsed):
 obj={'step':step,'training_seconds':elapsed,'mode':agent.mode,'models':{n:getattr(agent,n).state_dict() for n in ['encoder','actor','critic','critic_target']},'optimizers':{n:getattr(agent,n).state_dict() for n in ['encoder_opt','actor_opt','critic_opt']},'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
if __name__=='__main__':
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
 test='--validate' in sys.argv
 if not test:
  while not (ROOT/'reports/final_evaluation.jsonl').exists() or len((ROOT/'reports/final_evaluation.jsonl').read_text().splitlines())<162:time.sleep(30)
 while not (ROOT/'reports/data_manifest.json').exists():
  if test:break
  time.sleep(10)
 directory=ROOT/'data/cheetah_10k'
 if test:directory=BASE/'data/cheetah_medium_expert_easy'
 replay=ReplayBuffer(102000 if not test else 12000,3,.99,3)
 load_offline_dataset_into_buffer([directory],replay,3,102000 if not test else 12000,verbose=False)
 print('REPLAY',len(replay),flush=True)
 plan=[(m,s) for s in range(3) for m in ['bc','drq']]
 if test:plan=[(m,0) for m in ['bc','drq','strong','consistency','mask']]
 for mode,seed in plan:
  dest=ROOT/'runs_10k'/f'{mode}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
  if not test and (dest/'result.json').exists():continue
  agent=build(mode,seed);agent.train(True);step=0;elapsed=0.;torch.cuda.reset_peak_memory_stats();peak=0
  checkpoint=dest/'latest.pt'
  if not test and checkpoint.exists():
   c=torch.load(checkpoint,weights_only=False)
   for n,v in c['models'].items():getattr(agent,n).load_state_dict(v)
   for n,v in c['optimizers'].items():getattr(agent,n).load_state_dict(v)
   step=c['step'];elapsed=c['training_seconds'];np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
  duration=0.3 if test else 300.
  with (dest/('validation.jsonl' if test else 'metrics.jsonl')).open('a') as log:
   while elapsed<duration:
    t=time.perf_counter();step+=1;metrics=agent.update(replay,step*2,16);torch.cuda.synchronize();elapsed+=time.perf_counter()-t
    assert all(np.isfinite(v) for v in metrics.values()),metrics
    peak=max(peak,psutil.Process().memory_info().rss)
    if step%100==0 or test:
     log.write(json.dumps({'update':step,'training_seconds':elapsed,**metrics})+'\n');log.flush()
    if step%5000==0 and not test:save(agent,checkpoint,step,elapsed);print(mode,seed,step,round(elapsed),flush=True)
   if test:print('VALIDATED',mode,step,flush=True);continue
  save(agent,checkpoint,step,elapsed)
  result={'mode':mode,'seed':seed,'updates':step,'training_seconds':elapsed,'peak_rss':peak,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'evaluation':evaluate(agent,501)}
  (dest/'result.json').write_text(json.dumps(result,indent=2));print('COMPLETED',json.dumps(result),flush=True)
 print('VALIDATION DONE' if test else 'SMALL DATA PILOT DONE',flush=True)
