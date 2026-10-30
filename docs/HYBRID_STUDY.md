# Controlled gating and alignment study

All 12 runs reached the fixed 40,000-update endpoint, using 160,000 training transitions from two distracted cheetah domains. All four conditions use the same BC policy and inverse representation objective, with optional learned gating and gradient-reversal alignment. This is an adapted offline imitation experiment; there is no reward-learning critic. It is not a reproduction of the original domain-adversarial RL paper.

The primary held-out return was 157.73 +/- 38.63 for alignment without gating and 109.61 +/- 9.51 with gating, over three trained seeds. Each seed mean averages five simulator episodes. All raw-feature domain probes obtained accuracy 1.0. Thus the study supports neither an improved-control claim nor domain invariance. Details, intermediate evaluations, matched controls and independent probe records are retained in `results/hybrid_20261030`.

Primary evaluation uses static per-episode distractions, seeds 801–805. A separately labelled exploratory paired static/dynamic stress test used seeds 901–905, changing background, camera and colour jointly. Its 120 raw episode records and summary are exported separately and were not used for model selection. Do not combine its numbers with the primary means.

## Reproducing and checking

Archived arithmetic verification requires only Python's standard library:

```sh
python scripts/hybrid/verify_export.py
```

The original training/evaluation scripts are under `scripts/hybrid/`; the fixed protocol is in `configs/infogating_adversarial/protocol.json`. Preparation, data audit and exact action-sequence overlap audit precede training; training saves weights, optimizers, step/time and RNG every 2,500 updates using atomic replacement. Probes fit frozen features from separate probe-fit episodes and evaluate separate probe-test episodes. `analyze_interaction.py` computes paired seed contrasts. No checkpoint is selected using final test returns.

These scripts retain the actual local execution configuration. A new machine must configure dataset, DAVIS and externally obtained InfoGating source paths, install the dependencies described in DATA.md/REPRODUCE.md and set a new absolute training budget deliberately. Do not run `train.py` against the old expired `deadline_budget.json` and expect training to begin. Do not overwrite the completed experiment directory when reproducing; use a separate experiment output directory and preserve the recorded protocol. Unlicensed upstream InfoGating source is excluded; see THIRD_PARTY_NOTICES.md. Exact hashes do not prove independence of approximately related physical trajectories.

## Manuscript

The final four-page IEEE template manuscript is in the local ignored experiment directory `experiments/hybrid_20261030/ieee_template/`. It reports the primary negative finding. Both Word and PDF were checked against the supplied template. Affiliations and the second author's email were not supplied. Similarity clearance, human author approval, venue compliance and acceptance are not certified. No external submission has been made.
