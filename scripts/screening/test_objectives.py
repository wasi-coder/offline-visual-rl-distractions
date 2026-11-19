"""Independent small CPU checks of the new objective boundaries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
from agents import ScreeningAgent,expectile_loss

# Upper expectiles must penalize an underestimate more than an equal overestimate.
d=torch.tensor([-2.,2.],requires_grad=True)
expectile_loss(d,.7).backward()
torch.testing.assert_close(d.grad,torch.tensor([-.6,1.4]))

class Encoder(nn.Module):
    def __init__(self):super().__init__();self.layer=nn.Linear(3,3)
    def forward(self,x):
        z=self.layer(x.mean((-1,-2)));return z,z
class Actor(nn.Module):
    def __init__(self):super().__init__();self.layer=nn.Linear(3,2)
    def forward(self,z,std):
        from src.models.actors import TruncatedNormal
        return TruncatedNormal(self.layer(z).tanh(),torch.ones((len(z),2))*std)
class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(5,2);self.actions=[]
    def forward(self,z,a):
        self.actions.append(a.detach().clone());q=self.layer(torch.cat([z,a],-1));return q[:,:1],q[:,1:]

# IQL must evaluate target Q only at the logged action, even when actor means
# differ. This tests the implementation's central offline-data boundary.
torch.manual_seed(5)
a=ScreeningAgent.__new__(ScreeningAgent)
a.mode='iql';a.device='cpu';a.encoder=Encoder();a.actor=Actor();a.critic=Critic();a.critic_target=Critic();a.value=nn.Linear(3,1)
a.critic_target_tau=.01;a.stddev_schedule='0.5';a.stddev_clip=.3
for name in ['encoder','actor','critic','value']:
    setattr(a,name+'_opt',torch.optim.SGD(getattr(a,name).parameters(),lr=.01))
batch=(torch.rand(4,3,2,2),torch.tensor([[.8,-.7]]).repeat(4,1),torch.ones(4,1),torch.ones(4,1)*.99,torch.rand(4,3,2,2))
a.sample=lambda replay,n:batch;a.aug=lambda x,shift=None:(x,None)
metrics=a.update(None,2,4)
assert len(a.critic_target.actions)==2
for seen in a.critic_target.actions:torch.testing.assert_close(seen,batch[1])
assert all(torch.isfinite(torch.tensor(x)) for x in metrics.values())
assert all(p.grad is None for p in a.critic_target.parameters()),'Target critic received a gradient'
print('PASS: expectile direction, dataset-only target Q actions, detached target critic, finite toy IQL update')
