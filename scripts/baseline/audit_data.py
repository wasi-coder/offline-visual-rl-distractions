from pathlib import Path
import h5py,numpy as np,json,hashlib,ast
root=Path(__file__).resolve().parents[2];rows=[]
for folder in ['cheetah_medium_expert_easy','cheetah_10k']:
 for p in sorted((root/'data'/folder).glob('*.hdf5')):
  with h5py.File(p,'r') as f:
   marks=f['step_type'][:];starts=np.flatnonzero(marks==0);ends=np.flatnonzero(marks==2)
   assert len(starts)==len(ends) and all(b-a==500 for a,b in zip(starts,ends))
   rewards=f['reward'][:];actions=f['action'][:]
   assert np.isfinite(actions).all() and np.isfinite(rewards).all() and np.abs(actions).max()<=1.00001
   returns=[float(rewards[a:b+1].sum()) for a,b in zip(starts,ends)]
   digest=hashlib.sha256()
   with p.open('rb') as stream:
    for b in iter(lambda:stream.read(1024*1024),b''):digest.update(b)
   rows.append({'dataset':folder,'filename':p.name,'episodes':len(starts),'transitions':len(marks)-len(starts),'return_mean':float(np.mean(returns)),'return_min':min(returns),'return_max':max(returns),'episode_returns':returns,'sha256':digest.hexdigest()})
module=ast.parse((root/'envs/distracting_control/background.py').read_text());lists={}
for node in module.body:
 if isinstance(node,ast.Assign):
  for name in node.targets:
   if isinstance(name,ast.Name) and name.id.startswith('DAVIS17_'):lists[name.id]=ast.literal_eval(node.value)
held=['blackswan','bmx-trees','breakdance','camel'];train=lists['DAVIS17_TRAINING_VIDEOS'][:4]
assert not set(held)&set(lists['DAVIS17_TRAINING_VIDEOS'])
audit={'files':rows,'familiar_sources_from_code':train,'heldout_sources':held,'heldout_excluded_from_entire_DAVIS_train_list':True,'caveat':'Training source membership inferred from supplied generator convention, not independently reconstructed for every stored pixel. No direct simulator-state provenance is stored in selected HDF5.'}
(root/'reports/data_audit.json').write_text(json.dumps(audit,indent=2))
for folder in ['cheetah_medium_expert_easy','cheetah_10k']:
 vals=[v for r in rows if r['dataset']==folder for v in r['episode_returns']]
 print(folder,len(vals),np.mean(vals),min(vals),max(vals))
print('Background split verified against code',train,held)
