from pathlib import Path
import h5py,numpy as np,json
root=Path(__file__).resolve().parents[2];dest=root/'data/cheetah_10k';dest.mkdir(parents=True,exist_ok=True)
manifest=[]
for i,path in enumerate(sorted((root/'data/cheetah_medium_expert_easy').glob('*.hdf5'))):
 with h5py.File(path,'r') as src:
  marks=src['step_type'][:];starts=np.flatnonzero(marks==0);ends=np.flatnonzero(marks==2)
  count=3 if i%4<2 else 2
  ids=np.sort(np.random.default_rng(918+i).choice(len(starts),count,replace=False));idx=np.concatenate([np.arange(starts[j],ends[j]+1) for j in ids])
  with h5py.File(dest/path.name,'w') as out:
   for k in src:out.create_dataset(k,data=src[k][idx],compression='lzf')
  manifest.append({'source':str(path),'selected_episode_indices':ids.tolist(),'transitions':len(idx)-count})
(root/'reports/data_10k_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Nested subset',sum(r['transitions'] for r in manifest))
