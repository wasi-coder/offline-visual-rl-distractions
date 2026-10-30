"""Independent frozen-feature linear probes, run AFTER training has stopped."""
from pathlib import Path
import sys,json,random
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';sys.path[:0]=[str(R/'scripts/hybrid'),str(R)]
import train as study
import numpy as np,torch
from src.utils import ReplayBuffer,load_offline_dataset_into_buffer

def collect(model,split):
 replays=[]
 for domain in ['easy','medium']:
  r=ReplayBuffer(14000,3,.99,3);load_offline_dataset_into_buffer([O/'data'/split/domain],r,3,14000,verbose=False);replays.append(r)
 np.random.seed(900 if split=='validation' else 901);torch.manual_seed(900)
 z=[];pair=[];action=[];domains=[]
 with torch.no_grad():
  for _ in range(250):
   batch,lab=study.sample(replays)
   current=model.a.encoder(torch.as_tensor(batch[0],device='cuda').float()).cpu().numpy()
   future=model.a.encoder(torch.as_tensor(batch[6],device='cuda').float()).cpu().numpy()
   z.append(current);pair.append(np.concatenate([current,future],axis=1));action.append(batch[1]);domains.append(lab)
 return np.concatenate(z),np.concatenate(pair),np.concatenate(action),np.concatenate(domains)
def ridge(x,y,t):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1
 x=np.column_stack([(x-mu)/sd,np.ones(len(x))]).astype('float64');t=np.column_stack([(t-mu)/sd,np.ones(len(t))]).astype('float64')
 penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
 w=np.linalg.solve(x.T@x+penalty,x.T@y)
 return t@w
if __name__=='__main__':
 torch.set_num_threads(4);out=O/'frozen_probes.jsonl';done=set()
 if out.exists():done={json.loads(x)['run'] for x in out.read_text().splitlines()}
 for path in sorted((O/'runs').glob('*/result.json')):
  if path.parent.name in done:continue
  result=json.loads(path.read_text())
  if not result.get('complete'):continue
  model=study.Model(result['mode'],int(path.parent.name.split('_seed')[1]));c=torch.load(path.parent/'latest.pt',weights_only=False);model.restore(c);model.train(False)
  x,xp,a,d=collect(model,'validation');t,tp,ta,td=collect(model,'probe_test')
  pred=ridge(xp,a,tp);domain=ridge(x,np.eye(2)[d],t).argmax(1)
  record={'run':path.parent.name,'updates':result['updates'],'fit_samples':len(x),'test_samples':len(t),'action_mse':float(np.mean((pred-ta)**2)),'constant_action_mse':float(np.mean((a.mean(0)-ta)**2)),'domain_accuracy':float(np.mean(domain==td)),'raw_feature_std_mean':float(t.std(0).mean()),'protocol':'Frozen raw-input encoder; ridge penalty1 fixed in advance, standardization from fit only; validation episodes used for fit, probe_test episodes for evaluation; balanced sampled domains, no hyperparameter selection; samples can repeat within episodes.'}
  with out.open('a') as f:f.write(json.dumps(record)+'\n')
  print(record,flush=True)
 print('PROBES COMPLETE',flush=True)
