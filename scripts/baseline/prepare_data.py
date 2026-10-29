import os
from pathlib import Path
import zipfile,json,shutil,numpy as np,h5py
root=Path(__file__).resolve().parents[2]
out=root/'data/cheetah_medium_expert_easy';out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261029); manifest=[]
for archive in sorted(Path(os.environ.get('OFFLINE_RL_ARCHIVE_DIR',str(root/'data/archives'))).glob('*.zip')):
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   if 'vd4rl_extended/distracting/cheetah_run/medium_expert/84px/easy/' not in name or not name.endswith('.hdf5'):continue
   temp=root/'data/source_temp.hdf5'
   with z.open(name) as a,temp.open('wb') as b:shutil.copyfileobj(a,b)
   with h5py.File(temp,'r') as src:
    starts=np.flatnonzero(src['step_type'][:]==0);ends=np.flatnonzero(src['step_type'][:]==2)
    ids=np.sort(rng.choice(len(starts),25,replace=False));idx=np.concatenate([np.arange(starts[i],ends[i]+1) for i in ids])
    target=out/(Path(name).stem+'.hdf5')
    with h5py.File(target,'w') as dst:
     for key in src:dst.create_dataset(key,data=src[key][idx],compression='lzf')
    manifest.append({'archive':str(archive),'member':name,'episode_indices':ids.tolist(),'records':len(idx),'transitions':len(idx)-25,'file':str(target)})
   temp.unlink();print('Prepared',name,flush=True)
(root/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE transitions',sum(x['transitions'] for x in manifest),flush=True)
