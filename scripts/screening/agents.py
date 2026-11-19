"""Small-budget visual adaptations, not reproductions of the source papers.

IQL: Kostrikov et al., https://arxiv.org/abs/2110.06169
AWAC: Nair et al., https://arxiv.org/abs/2006.09359
SVEA: Hansen et al., https://arxiv.org/abs/2107.00644
Uses the attributed upstream DrQ implementation already supplied in this repo.
"""
import torch
from torch import nn
import torch.nn.functional as F
from src.agents.drqv2_default import DrQv2DefaultAgent
from src.utils import schedule, soft_update_params


def expectile_loss(delta, tau=.7):
    return (torch.where(delta > 0, tau, 1-tau)*delta.square()).mean()


class ScreeningAgent(DrQv2DefaultAgent):
    def configure(self, mode):
        self.mode = mode
        if mode == 'iql':
            self.value = nn.Sequential(nn.Linear(self.encoder.repr_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim), nn.Tanh(), nn.Linear(self.feature_dim,256),
                nn.ReLU(), nn.Linear(256,1)).to(self.device)
            self.value_opt = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        return self

    def update_actor(self, obs, step, behavioural_action=None):
        if self.mode not in ('iql','awac'):
            return super().update_actor(obs,step,behavioural_action)
        dist=self.actor(obs,schedule(self.stddev_schedule,step))
        with torch.no_grad():
            critic=self.critic_target if self.mode=='iql' else self.critic
            q=torch.minimum(*critic(obs,behavioural_action))
            if self.mode=='iql':
                advantage=q-self.value(obs)
                weights=(3.*advantage).clamp(max=4.605170186).exp()
            else:
                v=torch.minimum(*critic(obs,dist.sample(clip=self.stddev_clip)))
                weights=((q-v)/.3).clamp(max=4.605170186).exp()
        loss=-(weights*dist.log_prob(behavioural_action).sum(-1,keepdim=True)).mean()
        self.actor_opt.zero_grad(set_to_none=True);loss.backward();self.actor_opt.step()
        return {'actor_loss':loss.item(),'weight_mean':weights.mean().item()}

    def update(self,replay,step,batch_size):
        if self.mode in ('drq','awac'):
            return super().update(replay,step,batch_size)
        obs,act,rew,discount,nxt=self.sample(replay,batch_size)
        obs,shift=self.aug(obs.float(),shift=None)
        nxt,_=self.aug(nxt.float(),shift=shift)
        z,_=self.encoder(obs)
        if self.mode=='bc':
            loss=F.mse_loss(self.actor(z,schedule(self.stddev_schedule,step)).mean,act)
            self.encoder_opt.zero_grad(set_to_none=True);self.actor_opt.zero_grad(set_to_none=True)
            loss.backward();self.encoder_opt.step();self.actor_opt.step()
            return {'bc_loss':loss.item()}
        metrics={}
        if self.mode=='iql':
            with torch.no_grad():q=torch.minimum(*self.critic_target(z.detach(),act))
            vl=expectile_loss(q-self.value(z))
            self.value_opt.zero_grad(set_to_none=True);self.encoder_opt.zero_grad(set_to_none=True)
            vl.backward();self.value_opt.step();self.encoder_opt.step()
            z,_=self.encoder(obs)
            with torch.no_grad():target=rew.float()+discount*self.value(self.encoder(nxt)[0])
            metrics['value_loss']=vl.item()
        elif self.mode=='svea':
            # Strong RGB gain only on current-state extra critic branch. The
            # target and policy branch use base random shifts, never RGB gain.
            with torch.no_grad():
                zn,_=self.encoder(nxt)
                a=self.actor(zn,schedule(self.stddev_schedule,step)).sample(clip=self.stddev_clip)
                target=rew.float()+discount*torch.minimum(*self.critic_target(zn,a))
        else:raise ValueError(self.mode)
        q1,q2=self.critic(z,act)
        loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        if self.mode=='svea':
            gain=torch.empty(obs.shape[0],3,1,1,device=self.device).uniform_(.6,1.4).repeat(1,3,1,1)
            za,_=self.encoder((obs*gain).clamp(0,1));qa,qb=self.critic(za,act)
            extra=F.mse_loss(qa,target)+F.mse_loss(qb,target)
            loss=.5*loss+.5*extra
            metrics['augmented_critic_loss']=extra.item()
        self.encoder_opt.zero_grad(set_to_none=True);self.critic_opt.zero_grad(set_to_none=True)
        loss.backward();self.encoder_opt.step();self.critic_opt.step()
        with torch.no_grad():z,_=self.encoder(obs)
        metrics.update(self.update_actor(z,step,act));metrics['critic_loss']=loss.item()
        soft_update_params(self.critic,self.critic_target,self.critic_target_tau)
        return metrics
