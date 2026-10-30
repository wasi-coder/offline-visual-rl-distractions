from pathlib import Path
import sys,json
R=Path(r'E:\offline_rl');sys.path[:0]=[str(R/'scripts/hybrid'),str(R)]
import train as t
import torch,numpy as np
rng=np.random.default_rng(13)
b=(rng.integers(0,256,(16,9,84,84),dtype=np.uint8),rng.uniform(-1,1,(16,6)).astype('float32'),np.zeros((16,1)),np.ones((16,1)),rng.integers(0,256,(16,9,84,84),dtype=np.uint8),np.ones(16),rng.integers(0,256,(16,9,84,84),dtype=np.uint8))
records=[];torch.set_num_threads(4)
for mode in ['adv_inv','adv_gate']:
 m=t.Model(mode,9)
 modules={'encoder':m.a.encoder,'inverse':m.a.inv_model,'policy':m.a.policy,'domain':m.domain,'mask':m.a.mask_net}
 before={n:[p.detach().clone() for p in module.parameters()] for n,module in modules.items()}
 metrics=m.update(b,[0]*8+[1]*8,3000)
 changed={n:any(not torch.equal(old,new) for old,new in zip(before[n],module.parameters())) for n,module in modules.items()}
 assert all(changed[n] for n in ['encoder','inverse','policy','domain'])
 assert changed['mask']==(mode=='adv_gate'),changed
 assert all(p.requires_grad for p in m.domain.parameters())
 records.append({'mode':mode,'changed_parameters':changed,'finite_metrics':all(np.isfinite(x) for x in metrics.values())})
# Linear probe test with an exactly recoverable mapping; no author/GPU feature training involved.
x=rng.normal(size=(500,8));w=rng.normal(size=(8,2));test=rng.normal(size=(100,8))
from probe_features import ridge
pred=ridge(x,x@w,test);mse=float(np.mean((pred-test@w)**2));assert mse<.001
out={'gradient_path_checks':records,'ridge_known_mapping_mse':mse,'passed':True,'scope':'Synthetic postwarm parameter changes and independent probe algebra; not a learning result or proof of action preservation.'}
(R/'experiments/hybrid_20261030/implementation_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
