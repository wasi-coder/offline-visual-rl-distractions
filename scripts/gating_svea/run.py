"""Matched offline DrQv2+BC factorial experiment."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'screening'))
import run_screen as base
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.utils import schedule, soft_update_params

OUT = base.ROOT / 'experiments/gating_svea_20261112'
DATA = base.ROOT / 'data/cheetah_medium_expert_easy'
MODES = ['baseline', 'gate', 'svea', 'gate_svea']
RUNS = [(s, m) for s in (11, 12, 13) for m in MODES]
CHUNK = 250
BUDGET = 8 * 3600
EVAL_SEEDS = list(range(1201, 1206))
CONDITIONS = ['familiar_background', 'heldout_background']


def encode(agent, obs):
    if not hasattr(agent, 'mask_net'):
        return agent.encoder(obs)[0], None
    mask = F.relu(torch.tanh(agent.mask_net(obs * 255)))
    # Keep pixels visible until the inverse objective can train the mask.
    mask = .05 + .95 * mask
    return agent.encoder(obs * mask.repeat_interleave(3, dim=1))[0], mask


def build(mode, seed):
    a = base.build('drq', seed)
    a.mode = mode
    if 'gate' in mode:
        a.mask_net = base.pc.mod.UNet(num_blocks=5, img_size=84, filter_start=32,
                                    in_chnls=9, out_chnls=3, norm='gn').cuda()
        a.projection = nn.Linear(a.encoder.repr_dim, 64).cuda()
        a.inverse = base.pc.mod.InfoNCE(134, 6, 1).cuda()
        a.gate_opt = torch.optim.Adam(
            list(a.mask_net.parameters()) + list(a.projection.parameters()) +
            list(a.inverse.parameters()), lr=1e-4)
    return a


def update(a, iterator, step):
    obs, action, reward, discount, nxt, _, future = [
        torch.as_tensor(x, device=a.device) for x in next(iterator)]
    obs, shift = a.aug(obs.float() / 255, shift=None)
    nxt, _ = a.aug(nxt.float() / 255, shift=shift)
    z, mask = encode(a, obs)
    with torch.no_grad():
        zn, _ = encode(a, nxt)
        next_action = a.actor(zn, schedule(a.stddev_schedule, step * 2)).sample(clip=a.stddev_clip)
        target = reward.float() + discount * torch.minimum(*a.critic_target(zn, next_action))
    loss = sum(F.mse_loss(q, target) for q in a.critic(z, action))
    if 'svea' in a.mode:
        gain = torch.empty(len(obs), 3, 1, 1, device=a.device).uniform_(.6, 1.4).repeat(1, 3, 1, 1)
        za, _ = encode(a, (obs * gain).clamp(0, 1))
        loss = .5 * loss + .5 * sum(F.mse_loss(q, target) for q in a.critic(za, action))
    metrics = {'critic_loss': loss.detach()}
    opts = [a.encoder_opt, a.critic_opt]
    if mask is not None:
        future, _ = a.aug(future.float() / 255, shift=shift)
        zf, mf = encode(a, future)
        inv = a.inverse(a.projection(z), a.projection(zf), action)
        coverage = .5 * (mask.mean() + mf.mean())
        loss = loss + inv + .1 * coverage
        metrics.update(inverse_loss=inv.detach(), mask_mean=coverage.detach())
        opts.append(a.gate_opt)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in opts:
        opt.step()
    with torch.no_grad():
        z, _ = encode(a, obs)
    metrics.update(a.update_actor(z, step * 2, action))
    soft_update_params(a.critic, a.critic_target, a.critic_target_tau)
    return {k: float(v) for k, v in metrics.items()}


class EvaluationPolicy:
    def __init__(self, agent):
        self.agent = agent

    def act_with_logprob(self, obs, step, eval_mode):
        z, _ = encode(self.agent, torch.as_tensor(obs, device='cuda').float()[None])
        return self.agent.actor(z, .1).mean.cpu().numpy()[0], None


def validate(iterator):
    for mode in MODES:
        a = build(mode, 90)
        cp = OUT / 'validation.pt'
        first = next(a.encoder.parameters()).detach().clone()
        base.finite(a, update(a, iterator, 1))
        assert not torch.equal(first, next(a.encoder.parameters()))
        base.save(a, cp, 1, 0)
        expected = update(a, iterator, 2)
        base.load(a, cp)
        assert expected == update(a, iterator, 2)
        with torch.no_grad():
            action, _ = EvaluationPolicy(a).act_with_logprob(next(iterator)[0][0] / 255., 0, True)
        assert action.shape == (6,) and np.isfinite(action).all()
        print('VALIDATED', mode, flush=True)
    cp.unlink()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    protocol = dict(dataset=str(DATA), complete_episodes=200, transition_count=100000,
                    runs=RUNS, batch_size=16, chunk=CHUNK, update_seconds=BUDGET,
                    eval_seeds=EVAL_SEEDS, conditions=CONDITIONS,
                    gate='Author UNet and InfoNCE adapted to shared DrQ encoder. Mask applied at train and evaluation; floor .05, coverage weight .1. Critic and inverse gradients train gate.',
                    svea='Matched target, equal-weight original and RGB gain critic branches',
                    scope='One distracted-only dataset; no simulator training or domain adversarial alignment. Development background pools previously used.')
    pp = OUT / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == json.loads(json.dumps(protocol))
    else:
        base.write_json(pp, protocol)
    replay = base.ReplayBuffer(102000, 3, .99, 3)
    base.load_offline_dataset_into_buffer([DATA], replay, 3, 102000, verbose=False)
    assert not replay.full and int((replay.step_type[:len(replay)] == 2).sum()) == 200
    assert int(replay.valid.sum()) == 99600
    iterator = base.pc.Iterator(replay)
    if not (OUT / 'validated.json').exists():
        validate(iterator)
        base.write_json(OUT / 'validated.json', {'passed': True})
    state_path = OUT / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(round=0, done={}, training_seconds=0)
    if not state.get('training_complete'):
        while state['training_seconds'] < BUDGET or state['done']:
            target = (state['round'] + 1) * CHUNK
            for seed, mode in RUNS:
                key = f'{mode}_seed{seed}'
                if key in state['done']:
                    continue
                run = OUT / key
                run.mkdir(exist_ok=True)
                cp = run / 'latest.pt'
                a = build(mode, seed)
                c = base.load(a, cp) if cp.exists() else dict(step=0, training_seconds=0)
                elapsed = c['training_seconds']
                for step in range(c['step'] + 1, target + 1):
                    start = time.perf_counter()
                    metrics = update(a, iterator, step)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    if step % 50 == 0:
                        base.finite(a, metrics)
                        base.save(a, cp, step, elapsed)
                        with (run / 'metrics.jsonl').open('a') as log:
                            log.write(json.dumps(dict(step=step, training_seconds=elapsed, **metrics)) + '\n')
                state['done'][key] = elapsed
                base.write_json(state_path, state)
                print(key, target, round(elapsed, 2), flush=True)
                del a
                gc.collect()
            state = dict(round=state['round'] + 1, done={},
                         training_seconds=sum(state['done'].values()))
            base.write_json(state_path, state)
            print('MATCHED ROUND', state, flush=True)
        state['training_complete'] = True
        base.write_json(state_path, state)
    results = []
    for seed, mode in RUNS:
        run = OUT / f'{mode}_seed{seed}'
        a = build(mode, seed)
        checkpoint = base.load(a, run / 'latest.pt')
        a.train(False)
        if hasattr(a, 'mask_net'):
            a.mask_net.eval()
        for condition in CONDITIONS:
            for es in EVAL_SEEDS:
                p = run / f'{condition}_{es}.json'
                if not p.exists():
                    base.set_seed_everywhere(10000 + es)
                    row = base.episode(EvaluationPolicy(a), mode, condition, es)
                    assert row['steps'] == 500 and np.isfinite(row['return'])
                    base.write_json(p, row)
                results.append(dict(mode=mode, seed=seed, updates=checkpoint['step'], **json.loads(p.read_text())))
        del a
    base.write_json(OUT / 'evaluations.json', results)
    lines = ['# Distracted-only offline training with InfoGating and SVEA', '',
             f"All 12 policies completed {state['round'] * CHUNK:,} updates each. "
             f"Measured update time totalled {state['training_seconds']/3600:.2f} hours.",
             '', '| Method | Familiar mean ± seed SD | Held-out mean ± seed SD |',
             '|---|---:|---:|']
    def mean(mode, condition, seed):
        return statistics.mean(r['return'] for r in results if
                               (r['mode'], r['condition'], r['seed']) == (mode, condition, seed))
    for mode in MODES:
        values = [[mean(mode, condition, seed) for seed in (11, 12, 13)] for condition in CONDITIONS]
        lines.append('| ' + mode + ' | ' + ' | '.join(
            f'{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}' for v in values) + ' |')
    lines += ['', 'Held-out paired differences by training seed (11, 12, 13):', '']
    for control in MODES[:-1]:
        delta = [mean('gate_svea', CONDITIONS[1], s)-mean(control, CONDITIONS[1], s) for s in (11, 12, 13)]
        lines.append(f'- Combined minus {control}: ' + ', '.join(f'{d:+.2f}' for d in delta))
    lines += ['', 'Every policy learned from the same distracted dataset of 200 episodes. '
              'The simulator was used only for evaluation. Returns measure transfer under these tested backgrounds. '
              'The study uses one task and three training seeds, and the background pools were used in prior work. '
              'There is no clean-data control or predefined threshold for sufficient performance.',
              '', 'Sources: [InfoGating](https://arxiv.org/abs/2303.06121), [SVEA](https://arxiv.org/abs/2107.00644). '
              'The combined method adapts both to a common DrQv2+BC backbone.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
