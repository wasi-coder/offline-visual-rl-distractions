"""CPU-only equivalence check of adapter against downloaded author sampler."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/baseline'))
from published_comparator import Iterator
from src.utils import ReplayBuffer
spec=importlib.util.spec_from_file_location('author_replay',ROOT/'references/agent-centric-representations-main/numpy_replay_buffer.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=ReplayBuffer.__new__(ReplayBuffer)
r.buffer_size=1020;r.index=1006;r.full=False;r.frame_stack=3;r.nstep=3;r.obs_shape=(9,84,84);r.discount_vec=.99**np.arange(3);r.next_dis=.99**3
rng=np.random.default_rng(19)
r.obs=rng.integers(0,256,(1020,3,84,84),dtype=np.uint8)
r.act=rng.normal(size=(1020,6)).astype('float32');r.rew=rng.random(1020).astype('float32');r.dis=np.ones(1020,dtype='float32')
r.step_type=np.zeros(1020,dtype=int);r.valid=np.zeros(1020,dtype=bool)
a=m.EfficientReplayBuffer.__new__(m.EfficientReplayBuffer)
a.__dict__.update(r.__dict__);a.batch_size=16;a.sarsa=False;a.k_step=np.zeros(1020)
for start in (3,506):
 end=start+499;r.step_type[end]=2;r.valid[start:end-1]=True
 for now,idx in enumerate(range(start,end+1)):
  # Original loader: consecutive terminal rows differ by 501 (500 transitions + reset).
  a.k_step[idx]=min(501-1,now+15)-now
assert np.array_equal(a.k_step[r.valid],np.minimum(15,np.array([502,1005])[np.searchsorted([502,1005],np.flatnonzero(r.valid))]-np.flatnonzero(r.valid)+1))
checks=0
for seed in range(100):
 np.random.seed(seed);expected=next(a)
 np.random.seed(seed);actual=next(Iterator(r))
 for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y);checks+=1
out={'passed':True,'minibatches':100,'arrays_compared':checks,'batch_size':16,'episodes':2,'transitions_per_episode':500,'scope':'Synthetic replay with independent author horizon calculation; exact outputs under shared random seed. Does not establish whole algorithm or paper reproduction.'}
(ROOT/'reports/sampler_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
