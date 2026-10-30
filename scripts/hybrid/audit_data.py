from pathlib import Path
import json,time,hashlib,h5py,numpy as np
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030'
while not (O/'data_manifest.json').exists():time.sleep(30)
manifest=json.loads((O/'data_manifest.json').read_text());checks=[];total={}
for domain in ['easy','medium']:
 names=sorted({x['source'] for x in manifest if x['domain']==domain});assert len(names)==8
 for name in names:
  sets={split:set(next(x['selected_subset_episode_indices'] for x in manifest if x['domain']==domain and x['source']==name and x['split']==split)) for split in ['train','validation','probe_test']}
  assert len(sets['train'])==20 and len(sets['validation'])==3 and len(sets['probe_test'])==2
  assert not sets['train']&sets['validation'] and not sets['train']&sets['probe_test'] and not sets['validation']&sets['probe_test']
  for split in sets:
   p=O/'data'/split/domain/name;entry=next(x for x in manifest if x['domain']==domain and x['source']==name and x['split']==split)
   assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
   with h5py.File(p,'r') as f:
    marks=f['step_type'][:];st=np.flatnonzero(marks==0);en=np.flatnonzero(marks==2)
    assert len(st)==len(en)==len(sets[split]) and np.all(en-st==500)
    assert f['observation'].shape[1:]==(3,84,84)
    assert f['action'].shape[1:]==(6,)
    assert np.isfinite(f['reward'][:]).all() and np.isfinite(f['action'][:]).all() and np.abs(f['action'][:]).max()<=1.00001
   total[split]=total.get(split,0)+entry['transitions'];checks.append(str(p.relative_to(O)))
assert total=={'train':160000,'validation':24000,'probe_test':16000},total
(O/'data_audit.json').write_text(json.dumps({'passed':True,'files_checked':len(checks),'transitions':total,'disjoint_within_source_episode_splits':True,'limitation':'No simulator-state trajectory identifier permits independent proof that easy and medium archives do not contain corresponding physical trajectories. Splits are episode-disjoint within each source, not guaranteed independent across source collections.'},indent=2))
print('DATA AUDIT PASSED',total,flush=True)
