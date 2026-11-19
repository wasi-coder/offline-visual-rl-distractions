# Additional-method screening, 6 November 2026

This experiment asks which additional methods merit a longer test on distracted
offline cheetah data. It is a development screen, not a claim of publication-level
performance or a reproduction of the original papers. Previous runs are preserved.

## New candidates and sources

1. **Visual IQL adaptation.** I. Kostrikov, A. Nair and S. Levine,
   *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022.
   https://arxiv.org/abs/2110.06169 ; author code:
   https://github.com/ikostrikov/implicit_q_learning .
   Fits an upper-expectile value function using dataset actions, backs values into
   Q, and fits the actor by advantage-weighted likelihood. This tests whether a
   different offline value-learning rule helps; it does not explicitly remove
   visual distractions.
2. **Visual AWAC adaptation (offline phase only).** A. Nair, A. Gupta, M. Dalal
   and S. Levine, *AWAC: Accelerating Online Reinforcement Learning with Offline
   Datasets*, 2020/2021. https://arxiv.org/abs/2006.09359 ;
   https://awacrl.github.io/ . Uses Q estimates to put more weight on better
   recorded actions. The paper studies offline-to-online learning; this screen
   performs no online fine-tuning.
3. **SVEA-style DrQv2+BC.** N. Hansen, H. Su and X. Wang,
   *Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data
   Augmentation*, NeurIPS 2021. https://arxiv.org/abs/2107.00644 ;
   https://www.nicklashansen.com/SVEA/ . Both base and RGB-augmented current images
   predict the same critic target. Strong RGB augmentation is excluded from the
   target and policy branches. This is more targeted than simply augmenting both
   sides of every update.

## Anchors

Fresh BC, DrQv2+BC and adapted InfoGating runs use the same update budget.
InfoGating: M. Tomar, R. Islam, M. E. Taylor, S. Levine and P. Bachman,
*Ignorance is Bliss: Robust Control via Information Gating*, NeurIPS 2023.
https://arxiv.org/abs/2303.06121 . It uses the previously downloaded author source
through the existing wrapper; it is not redistributed here. See
`../../THIRD_PARTY_NOTICES.md` for source licences and attribution.

The existing upstream IQL/AWAC files were inspected but are not silently presented
as exact reproductions. `agents.py` implements their objectives using a common
DrQ actor/encoder interface. No new external implementation is vendored.

## Predeclared protocol

- Same existing 100,000 transitions / 200 complete episodes from 8 easy distracted
  cheetah-run medium-expert shards; no extra downloads or clean training data.
- Six methods, training seeds 0, 1, 2, 4,000 updates each, batch 16, from scratch.
  This is 64,000 sampled transition positions per run, drawn with replacement
  from 99,600 valid three-step starts; it does not mean every transition is seen.
- 3-frame 84px observations, action repeat 2, 3-step targets, discount 0.99.
- Common DrQ CNN, 50-dimensional actor trunk and 1024 hidden units for the five
  non-InfoGating methods; learning rate 3e-4, tau 0.01, random shifts.
- IQL expectile 0.7, exponential advantage scale 3, weight cap 100; compact value
  MLP with 50-dimensional normalized trunk, one 256 hidden layer. No extra BC
  mixture and no actor learning-rate decay. Shared encoder updated by value and
  critic, not by weighted actor likelihood.
- AWAC temperature 0.3, weight cap 100, one sampled action for the state-value
  approximation; no extra BC mixture, no entropy bonus, no online updates.
- IQL/AWAC inherit the scheduled-normal DrQ policy rather than their papers'
  precise distribution/architecture choices. Three-step targets also differ from
  standard one-step formulations. They are visual adaptations.
- SVEA weights 0.5/0.5 on twin-Q losses, per-image RGB gains U(0.6,1.4), shared
  across three stacked frames, clipped to [0,1]. Base random shifts remain on
  both next/current images; only strong RGB gain is excluded from the target.
  BC-regularized DrQ policy remains. This is SVEA-style, not original SVEA.
- InfoGating retains its previous architecture, lr 1e-4, feature dimension 256,
  lambda 0.1, 256 generated negative actions, and author warm-up step<5000.
  Runner step=2*update, so updates 1--2499 are warm-up and 2500--4000 include
  the full gate objective. The gate therefore has only 1501 full updates here.
- This matches iterations, not runtime, architectures or reward usage. InfoGating
  and BC do not use reward targets. Do not attribute a whole-method ranking to
  gating alone.
- Six final episodes per run: familiar and held-out-background conditions, each
  with seeds 701--703. Frozen policies, no learning in simulator, 500 decisions.
  These reuse the earlier pilot evaluator: **static within an episode** (the
  wrapper defaults to dynamic=False). fixed_distraction=False does not mean
  dynamic=True. The video sources and evaluation seeds have already been examined
  in prior work, so this is not a fresh confirmatory generalization test.
- Rank final held-out means; first average episodes within each training seed,
  then calculate mean/sample SD across 3 seeds. Never treat nine episodes as nine
  independent training repetitions. Do not pool old equal-time scores into this
  equal-update ranking or tune hyperparameters on these evaluation results.

## Execution and validation

From the repository root, using the established Python environment:

```powershell
& E:\offline_rl_smoke_test\environment\python\python.exe scripts/screening/run_screen.py --validate
& E:\offline_rl_smoke_test\environment\python\python.exe scripts/screening/run_screen.py
& E:\offline_rl_smoke_test\environment\python\python.exe scripts/screening/analyze.py
```

The runner restores atomic checkpoints containing models, optimizers, update count,
timings and Python/NumPy/Torch/CUDA RNG states. Checkpoints every 250 updates;
evaluation records resume by condition and seed. Validation checks finite updates,
encoder changes, valid bounded actions, and exact next-update equivalence after
checkpoint restore, including the full gate phase. The initial embedded-Python
module-path failure was fixed before training; any failure is logged.

Artifacts live in ignored `experiments/screening_20261106/`. Small result exports
and figures go to `results/screening_20261106/`. Nothing is pushed automatically.
Training will be underpowered for convergence; a poor short-run result is not
evidence that the published algorithm cannot work.

## Longer alignment follow-up

Run `scripts/screening/continue_alignment.ps1` to resume the screen, aggregate
its results and launch the follow-up with the highest held-out mean. The launcher
waits for a live screening process and holds a lock against duplicate launches.

The follow-up starts the selected method from scratch in six runs (three paired
seeds with alignment off/on). Both arms see the same audited 160,000-transition
easy/medium training split, with eight samples per domain in every update, and
train for 40,000 updates. The classifier learns from detached encoder features.
In the aligned arm, a second optimizer step reverses the domain gradient on
raw-image encoder features. This adapts the gradient-reversal component of
[Chang et al. (TMLR 2024)](https://openreview.net/forum?id=dce6ZGkJ1Z).
The adaptation omits DropBlock and, if InfoGating wins, does not route the
alignment loss through the mask network. The base objective stays the same
across the paired arms.

Checkpoints are written every 1,000 updates. Evaluate at 20,000 and 40,000
updates using five new simulator seeds on the existing familiar and held-out
background sets. The experiment folder is
`experiments/screening_alignment_20261107/`. The final report uses the paired
seed differences in return. This measures alignment on the screening winner
under this protocol, not the paper's complete algorithm.

## Other relevant methods found (not trained in this screen)

- **SODA**, Hansen and Wang, *Generalization in Reinforcement Learning by Soft
  Data Augmentation*, ICRA 2021. https://www.nicklashansen.com/SODA/ . A separate
  representation objective relates augmented and original observations while
  keeping strong augmentation out of policy learning. A possible next comparison
  against the earlier actor-only two-view consistency control, but requires its
  own projector/predictor/target implementation and validation.
- **SGQN**, Bertoin et al., *Look where you look! Saliency-guided Q-networks for
  generalization in visual Reinforcement Learning*.
  https://arxiv.org/abs/2209.09203 ; author code https://github.com/SuReLI/SGQN .
  Uses critic saliency to guide learning toward useful pixels. Relevant to the
  masking question, but saliency quality and the full auxiliary objective need
  careful checks before calling an adaptation SGQN. Not implemented or measured
  here; no results are assigned to it.
