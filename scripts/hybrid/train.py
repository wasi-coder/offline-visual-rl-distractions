"""Matched BC/inverse representation study; adapted InfoGating, not paper reproduction."""
from pathlib import Path
import sys,os,json,time,random,argparse
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';sys.path[:0]=[str(R/'scripts/baseline'),str(R)]
os.environ['OFFLINE_RL_DAVIS_PATH']=r'E:\offline_rl_smoke_test\data\DAVIS\JPEGImages\480p'
import published_comparator as pc
import torch,numpy as np
import torch.nn.functional as F
from src.utils import ReplayBuffer,load_offline_dataset_into_buffer,set_seed_everywhere
from src.models.encoders import grad_reverse
from evaluate_study import episode
class Model:
 def __init__(self,mode,seed):
  set_seed_everywhere(seed);self.mode=mode
  self.a=pc.mod.InfoGatingAgent((9,84,84),(6,),'cuda',1e-4,256,1024,.01,2000,2,'linear(1.0, 0.1, 200000)',.3,False,lam=.1)
  self.domain=torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Linear(128,2)).cuda()
  self.domain_opt=torch.optim.Adam(self.domain.parameters(),lr=1e-4)
 def train(self,t=True):self.a.train(t);self.domain.train(t)
 def act(self,*args):return self.a.act(*args)
 def update(self,batch,labels,step):
  a=self.a;obs,action,_,_,_,_,future=pc.mod.utils.to_torch(batch,'cuda')
  obs=a.aug(obs.float());future=a.aug(future.float());gate='gate' in self.mode;adv=self.mode.startswith('adv')
  if gate:
   m=F.relu(torch.tanh(a.mask_net(obs,step*2)));mk=F.relu(torch.tanh(a.mask_net(future,step*2)))
   mask=m.repeat_interleave(3,dim=1);maskk=mk.repeat_interleave(3,dim=1)
   x=obs*mask;y=future*maskk
   # Match author convention: half of each minibatch uses unmasked input.
   ix=torch.randperm(16,device='cuda')[:8];x=x.clone();x[ix]=obs[ix]
   ix=torch.randperm(16,device='cuda')[:8];y=y.clone();y[ix]=future[ix]
   sparse=(m.mean()+mk.mean())/2;square=(m.square().mean()+mk.square().mean())/2
   if step<2500:x=x.detach();y=y.detach()
  else:x=obs;y=future;sparse=obs.new_tensor(0.);square=sparse
  z=a.encoder(x);zk=a.encoder(y);inverse=a.inv_model(z,zk,action)
  lab=torch.as_tensor(labels,device='cuda',dtype=torch.long)
  # First train discriminator on detached features.
  self.domain_opt.zero_grad(set_to_none=True)
  dl=F.cross_entropy(self.domain(z.detach()),lab);dl.backward();self.domain_opt.step()
  # Freeze classifier parameters while GRL reverses encoder/gate gradients.
  for p in self.domain.parameters():p.requires_grad_(False)
  aligned=F.cross_entropy(self.domain(grad_reverse(z,scale=-1.)),lab)
  strength=min(1.,step/10000)*.1 if adv else 0.
  penalty=(-.1*sparse if step<2500 else .1*sparse+.001*square) if gate else sparse
  loss=inverse+penalty+strength*aligned
  a.encoder_opt.zero_grad(set_to_none=True);a.mask_opt.zero_grad(set_to_none=True)
  loss.backward();a.encoder_opt.step()
  if gate:a.mask_opt.step()
  for p in self.domain.parameters():p.requires_grad_(True)
  with torch.no_grad():features=a.encoder(x)
  dist=a.policy(features,pc.mod.utils.schedule(a.stddev_schedule,step*2));bc=F.mse_loss(dist.sample(clip=.3),action)
  a.policy_opt.zero_grad(set_to_none=True);bc.backward();a.policy_opt.step()
  return {'inverse_loss':inverse.item(),'domain_ce':dl.item(),'domain_accuracy':(self.domain(z.detach()).argmax(-1)==lab).float().mean().item(),'bc_loss':bc.item(),'mask_mean':sparse.item(),'adversarial_weight':strength,'representation_loss':loss.item()}
 def snapshot(self,step,elapsed):
  return {'mode':self.mode,'updates':step,'training_seconds':elapsed,'models':{n:getattr(self.a,n).state_dict() for n in ['encoder','mask_net','inv_model','policy']},'domain':self.domain.state_dict(),'optimizers':{n:getattr(self.a,n).state_dict() for n in ['encoder_opt','mask_opt','policy_opt']},'domain_opt':self.domain_opt.state_dict(),'rng':(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())}
 def restore(self,c):
  for n,v in c['models'].items():getattr(self.a,n).load_state_dict(v)
  self.domain.load_state_dict(c['domain'])
  for n,v in c['optimizers'].items():getattr(self.a,n).load_state_dict(v)
  self.domain_opt.load_state_dict(c['domain_opt']);random.setstate(c['rng'][0]);np.random.set_state(c['rng'][1]);torch.set_rng_state(c['rng'][2]);torch.cuda.set_rng_state_all(c['rng'][3])
def sample(replays):
 batches=[]
 for r in replays:
  if not hasattr(r,'hybrid_iterator'):r.hybrid_iterator=pc.Iterator(r)
  batches.append(next(r.hybrid_iterator))
 # Author sampler returns16 per domain. Use8 each for a balanced16 batch.
 batch=tuple(np.concatenate([batches[0][k][:8],batches[1][k][:8]]) for k in range(7))
 return batch,np.array([0]*8+[1]*8)
def save(model,path,step,elapsed):
 temp=path.with_suffix('.tmp');torch.save(model.snapshot(step,elapsed),temp);os.replace(temp,path)
def evaluate(model,dest,step):
 model.train(False);out=dest/'evaluation.jsonl';done=set()
 if out.exists():done={(x['updates'],x['condition'],x['eval_seed']) for x in map(json.loads,out.read_text().splitlines())}
 for condition in ['familiar_background','heldout_background']:
  for seed in [801,802,803,804,805]:
   if (step,condition,seed) in done:continue
   v=episode(model,'infogating',condition,seed);v['updates']=step;v['mode']=model.mode
   with out.open('a') as f:f.write(json.dumps(v)+'\n')
 model.train()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
 if args.validate:
  rng=np.random.default_rng(7);batch=(rng.integers(0,256,(16,9,84,84),dtype=np.uint8),rng.uniform(-1,1,(16,6)).astype('float32'),np.zeros((16,1)),np.ones((16,1)),rng.integers(0,256,(16,9,84,84),dtype=np.uint8),np.ones(16),rng.integers(0,256,(16,9,84,84),dtype=np.uint8))
  for mode in ['inv','gate','adv_inv','adv_gate']:
   model=Model(mode,0)
   for step in [1,3000]:
    metrics=model.update(batch,[0]*8+[1]*8,step);assert all(np.isfinite(v) for v in metrics.values());print('VALIDATED',mode,step,metrics,flush=True)
   c=model.snapshot(3000,0);model.restore(c)
  z=torch.tensor([1.],requires_grad=True,device='cuda');grad_reverse(z,-.25).sum().backward();assert z.grad.item()==-.25
  print('VALIDATION PASSED',flush=True);raise SystemExit
 while not (O/'data_audit.json').exists():time.sleep(30)
 assert json.loads((O/'data_audit.json').read_text())['passed']
 replays=[]
 for domain in ['easy','medium']:
  replay=ReplayBuffer(82000,3,.99,3);load_offline_dataset_into_buffer([O/'data/train'/domain],replay,3,82000,verbose=False);replays.append(replay)
 budget=json.loads((O/'deadline_budget.json').read_text())
 deadline=budget['training_stop_unix']
 target=budget['target_updates']
 plan=[(mode,seed) for seed in range(3) for mode in ['adv_inv','adv_gate']]+[(mode,seed) for seed in range(3) for mode in ['inv','gate']]
 for mode,seed in plan:
  if time.time()>=deadline:
   print('GLOBAL TRAINING DEADLINE; remaining runs not started',flush=True);break
  dest=O/'runs'/f'{mode}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'result.json').exists():continue
  model=Model(mode,seed);step=0;elapsed=0;checkpoint=dest/'latest.pt'
  if checkpoint.exists():
   c=torch.load(checkpoint,weights_only=False);model.restore(c);step=c['updates'];elapsed=c['training_seconds']
  with (dest/'metrics.jsonl').open('a') as log:
   while step<target and time.time()<deadline:
    batch,labels=sample(replays);torch.cuda.synchronize();t=time.perf_counter();step+=1;metrics=model.update(batch,labels,step);torch.cuda.synchronize();elapsed+=time.perf_counter()-t
    assert all(np.isfinite(v) for v in metrics.values()),metrics
    if step%100==0:log.write(json.dumps({'updates':step,'training_seconds':elapsed,**metrics})+'\n');log.flush()
    if step%2500==0:
     assert all(torch.isfinite(p).all() for m in [model.a.encoder,model.a.mask_net,model.a.policy,model.a.inv_model,model.domain] for p in m.parameters())
     save(model,checkpoint,step,elapsed);print(mode,seed,step,elapsed,flush=True)
    if step%20000==0 and time.time()<deadline:
     save(model,dest/f'checkpoint_{step}.pt',step,elapsed);evaluate(model,dest,step)
  save(model,checkpoint,step,elapsed)
  if time.time()<deadline and step%20000!=0:evaluate(model,dest,step)
  (dest/'result.json').write_text(json.dumps({'mode':mode,'seed':seed,'updates':step,'target_updates':target,'complete':step==target,'training_seconds':elapsed,'stopped_by_deadline':step<target}))
 print('TRAINING QUEUE ENDED: inspect complete flags and missing runs',flush=True)
