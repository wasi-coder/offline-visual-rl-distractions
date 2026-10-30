from pathlib import Path
import zipfile,json,shutil,numpy as np,h5py,hashlib
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';O.mkdir(parents=True,exist_ok=True)
manifest=[]
for domain in ['easy','medium']:
 dest=O/'data'/domain;dest.mkdir(parents=True,exist_ok=True)
 if domain=='easy': sources=list((R/'data/cheetah_medium_expert_easy').glob('*.hdf5'))
 else:
  sources=[]
  for zpath in sorted(Path(r'E:\offline rl dataset part 2').glob('*.zip')):
   with zipfile.ZipFile(zpath) as z:
    for member in z.namelist():
     if '/distracting/cheetah_run/medium_expert/84px/medium/' not in member or not member.endswith('.hdf5'):continue
     target=dest/Path(member).name
     if not target.exists():
      temp=dest/'source.tmp'
      with z.open(member) as a,temp.open('wb') as b:shutil.copyfileobj(a,b)
      with h5py.File(temp,'r') as f:
       st=np.flatnonzero(f['step_type'][:]==0);en=np.flatnonzero(f['step_type'][:]==2)
       ids=np.sort(np.random.default_rng(20261030+len(sources)).choice(len(st),25,False));ix=np.concatenate([np.arange(st[i],en[i]+1) for i in ids])
       with h5py.File(target,'w') as out:
        for k in f:out.create_dataset(k,data=f[k][ix],compression='lzf')
       (dest/(target.stem+'.selection.json')).write_text(json.dumps({'archive':zpath.name,'member':member,'episode_indices':ids.tolist()}))
      temp.unlink()
     sources.append(target);print('SOURCE',domain,target.name,flush=True)
 for j,p in enumerate(sorted(sources)):
  with h5py.File(p,'r') as f:
   st=np.flatnonzero(f['step_type'][:]==0);en=np.flatnonzero(f['step_type'][:]==2);assert len(st)==25
   perm=np.random.default_rng(430+j).permutation(25)
   for split,ids in [('train',perm[:20]),('validation',perm[20:23]),('probe_test',perm[23:])]:
    outdir=O/'data'/split/domain;outdir.mkdir(parents=True,exist_ok=True);target=outdir/p.name
    if not target.exists():
     ix=np.concatenate([np.arange(st[i],en[i]+1) for i in sorted(ids)])
     with h5py.File(target,'w') as out:
      for k in f:out.create_dataset(k,data=f[k][ix],compression='lzf')
    manifest.append({'domain':domain,'source':p.name,'split':split,'selected_subset_episode_indices':sorted(ids.tolist()),'transitions':len(ids)*500,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 assert len(sources)==8,(domain,len(sources))
(O/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print('PREPARATION COMPLETE',flush=True)
