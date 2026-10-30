from pathlib import Path
import h5py,numpy as np,hashlib,json
O=Path(r'E:\offline_rl/experiments/hybrid_20261030');seen={};overlaps=[];episodes=0
for split in ['train','validation','probe_test']:
 for domain in ['easy','medium']:
  for p in sorted((O/'data'/split/domain).glob('*.hdf5')):
   with h5py.File(p,'r') as f:
    marks=f['step_type'][:];actions=f['action'][:];rewards=f['reward'][:]
    for i,(start,end) in enumerate(zip(np.flatnonzero(marks==0),np.flatnonzero(marks==2))):
     sig=hashlib.sha256(actions[start+1:end+1].astype('<f4').tobytes()).hexdigest()
     entry={'split':split,'domain':domain,'file':p.name,'episode_in_split':i,'reward_sum':float(rewards[start:end+1].sum())}
     if sig in seen and seen[sig]['split']!=split:overlaps.append({'first':seen[sig],'second':entry,'action_sequence_sha256':sig})
     else:seen[sig]=entry
     episodes+=1
out={'episodes_checked':episodes,'cross_split_identical_action_sequences':overlaps,'passed':not overlaps,'scope':'Exact500-step6D action-sequence matching acrossdomains/splits; absence does not prove no related physical trajectories or approximate duplicates.'}
(O/'trajectory_overlap_audit.json').write_text(json.dumps(out,indent=2));print('episodes',episodes,'crosssplit matches',len(overlaps))
