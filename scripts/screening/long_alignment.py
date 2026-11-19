"""Follow the screening winner with a paired, distracted-only alignment study."""
import json
import time
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_screen import (
    ROOT, DEST, MODES, SEEDS, build, update, finite, save, load, write_json,
    rng_state, restore_rng, episode, pc, ReplayBuffer, load_offline_dataset_into_buffer,
)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.models.encoders import grad_reverse
from src.utils import set_seed_everywhere

DATA = ROOT / 'experiments/hybrid_20261030'
OUTPUT = ROOT / 'experiments/screening_alignment_20261107'
TARGET = 40_000
CONDITIONS = ['familiar_background', 'heldout_background']
EVAL_SEEDS = [901, 902, 903, 904, 905]


class BalancedReplay:
    def __init__(self, replays):
        self.replays = replays
        self.iterators = [pc.Iterator(replay) for replay in replays]

    def sample(self, batch_size):
        assert batch_size == 16
        return self._combine([replay.sample(8) for replay in self.replays])

    def __next__(self):
        return self._combine([next(iterator) for iterator in self.iterators])

    def _combine(self, batches):
        batch = tuple(np.concatenate([a[:8], b[:8]]) for a, b in zip(*batches))
        self.images = batch[0]
        return batch


def alignment_update(agent, images, step, enabled):
    device = 'cuda'
    obs = torch.as_tensor(images, device=device).float()
    if agent.mode == 'infogating':
        features = agent.encoder(obs)
    else:
        features = agent.encoder(obs / 255.0)[1]
    labels = torch.arange(16, device=device) // 8
    classifier = agent.domain
    logits = classifier(features.detach())
    loss = F.cross_entropy(logits, labels)
    agent.domain_opt.zero_grad(set_to_none=True)
    loss.backward()
    agent.domain_opt.step()
    weight = 0.1 * min(step / 10000, 1.0) if enabled else 0.0
    if enabled:
        classifier.requires_grad_(False)
        agent.alignment_opt.zero_grad(set_to_none=True)
        (weight * F.cross_entropy(classifier(grad_reverse(features, -1.0)), labels)).backward()
        agent.alignment_opt.step()
        classifier.requires_grad_(True)
    return {'domain_loss': loss.item(), 'domain_accuracy': (logits.argmax(1) == labels).float().mean().item(),
            'reversal_strength': weight}


def add_classifier(agent):
    dimension = 256 if agent.mode == 'infogating' else agent.encoder.repr_dim
    with torch.random.fork_rng(devices=[0]):
        agent.domain = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, 2)).cuda()
    agent.domain_opt = torch.optim.Adam(agent.domain.parameters(), lr=1e-4)
    # Separate optimizer keeps the base method's momentum and task objective intact.
    agent.alignment_opt = torch.optim.SGD(agent.encoder.parameters(), lr=1e-3)


def evaluate(agent, path):
    previous_rng = rng_state()
    agent.train(False)
    try:
        rows = json.loads(path.read_text()) if path.exists() else []
        done = {(row['condition'], row['eval_seed']) for row in rows}
        for condition in CONDITIONS:
            for seed in EVAL_SEEDS:
                if (condition, seed) in done:
                    continue
                set_seed_everywhere(10000 + seed)
                row = episode(agent, agent.mode, condition, seed)
                assert row['steps'] == 500 and np.isfinite(row['return'])
                rows.append(row)
                write_json(path, rows)
        return rows
    finally:
        restore_rng(previous_rng)
        agent.train(True)


def main(*, replication=False):
    screening = json.loads((ROOT / 'results/screening_20261106/aggregate.json').read_text())
    assert screening['runs'] == len(MODES) * len(SEEDS) and screening['evaluations'] == 108
    winner = screening['summary'][0]['method']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'selected_method': winner, 'selection': 'Highest mean held-out return in completed screening',
        'updates_per_run': TARGET, 'training_seeds': SEEDS, 'eval_seeds': EVAL_SEEDS,
        'conditions': CONDITIONS, 'training_data': str(DATA / 'data/train'),
        'transitions': 160000, 'minibatch': '8 easy and 8 medium distracted samples',
        'alignment': 'Detached classifier followed by reversed gradient on raw policy encoder; separate SGD lr 0.001, strength ramps to 0.1 in 10000 updates',
        'scope': 'Adaptation of Chang et al. gradient reversal; no DropBlock, clean training data or warm start. Existing background sources, new simulator seeds.',
        'source': 'https://openreview.net/forum?id=dce6ZGkJ1Z',
    }
    protocol_path = OUTPUT / 'protocol.json'
    checkpoints = (20000, 40000) if replication else (20000,)
    if replication:
        protocol.update({
            'evaluation_updates': [*checkpoints, TARGET],
            'primary_update': 40000,
            'primary_condition': 'heldout_background',
            'extension_update': TARGET,
            'prior_training_seeds': [0, 1, 2],
            'purpose': 'Replicate the prior 40000-update contrast with fresh training seeds and extend training',
        })
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write_json(protocol_path, protocol)
    print('SELECTED', winner, flush=True)
    assert json.loads((DATA / 'data_audit.json').read_text())['passed']
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    replays = []
    for domain in ['easy', 'medium']:
        replay = ReplayBuffer(82000, 3, .99, 3)
        load_offline_dataset_into_buffer([DATA / 'data/train' / domain], replay, 3, 82000, verbose=False)
        assert not replay.full and (replay.step_type[:len(replay)] == 2).sum() == 160
        replays.append(replay)
    replay = BalancedReplay(replays)
    # Check that only the enabled branch changes encoder weights on the domain step.
    agent = build(winner, 99)
    add_classifier(agent)
    images = replay.sample(16)[0]
    parameter = next(agent.encoder.parameters())
    before = parameter.detach().clone()
    alignment_update(agent, images, 10000, False)
    assert torch.equal(before, parameter)
    alignment_update(agent, images, 10000, True)
    assert not torch.equal(before, parameter)
    del agent
    write_json(OUTPUT / 'validation.json', {'encoder_unchanged_without_alignment': True,
                                           'encoder_changed_with_alignment': True})
    for seed in SEEDS:
        for enabled in [False, True]:
            arm = 'aligned' if enabled else 'control'
            folder = OUTPUT / f'{arm}_seed{seed}'
            folder.mkdir(exist_ok=True)
            if (folder / 'result.json').exists():
                continue
            agent = build(winner, seed)
            add_classifier(agent)
            checkpoint = folder / 'latest.pt'
            step, elapsed = 0, 0.0
            if checkpoint.exists():
                state = load(agent, checkpoint)
                step, elapsed = state['step'], state['training_seconds']
                del state
            print('START', arm, seed, step, flush=True)
            if step in checkpoints:
                evaluate(agent, folder / f'eval_{step}.json')
            with (folder / 'metrics.jsonl').open('a') as log:
                while step < TARGET:
                    start = time.perf_counter()
                    step += 1
                    metrics = update(agent, replay, replay, step)
                    metrics.update(alignment_update(agent, replay.images, step, enabled))
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    if step % 250 == 0:
                        finite(agent, metrics)
                        log.write(json.dumps(dict(step=step, **metrics)) + '\n')
                        log.flush()
                    if step % 1000 == 0:
                        save(agent, checkpoint, step, elapsed)
                        print(arm, seed, step, round(elapsed), flush=True)
                    if step in checkpoints:
                        evaluate(agent, folder / f'eval_{step}.json')
            rows = evaluate(agent, folder / f'eval_{TARGET}.json')
            write_json(folder / 'result.json', dict(arm=arm, seed=seed, updates=step, seconds=elapsed,
                                                    evaluation=rows))
            del agent
            torch.cuda.empty_cache()
    report = ['# Longer domain alignment follow-up', '', f'Screening selection: {winner}.',
              'Each entry averages five final simulator episodes per training seed.', '']
    for checkpoint in ([40000, TARGET] if replication else [TARGET]):
      report += [f'## {checkpoint:,} updates', '']
      for condition in CONDITIONS:
        scores = {}
        for arm in ['control', 'aligned']:
            evaluations = [json.loads((OUTPUT / f'{arm}_seed{s}/eval_{checkpoint}.json').read_text()) for s in SEEDS]
            scores[arm] = [statistics.mean(e['return'] for e in rows if e['condition'] == condition)
                           for rows in evaluations]
        differences = [a-b for a,b in zip(scores['aligned'], scores['control'])]
        report += [f'### {condition}', '', f'Control: {scores["control"]}',
                   f'Aligned: {scores["aligned"]}', f'Paired differences: {differences}',
                   f'Mean difference: {statistics.mean(differences):.2f} (seed SD {statistics.stdev(differences):.2f}).', '']
    report += ['Three seeds on one task and previously used background sources. The comparison measures return under this protocol. Classifier training accuracy alone cannot establish representation invariance.']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print('FOLLOW-UP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
