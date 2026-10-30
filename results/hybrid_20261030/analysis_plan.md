# Hybrid analysis plan and limits

The final checkpoint is fixed at40,000updates. Intermediate20,000-update evaluation is a learning-curve diagnostic, not a checkpoint-selection rule. Primary outcome: held-out-source raw return, averaging five fixed evaluation episodes within each training seed, then reporting the mean and sample SD across three training seeds. Familiar-source return is secondary. Incomplete runs must be labeled and not silently substituted at another update budget.

## Matched comparisons

Let R(mode,seed) be the final held-out return averaged across five physical seeds.

- Primary gating difference under alignment: R(adv_gate,s)-R(adv_inv,s).
- Gating difference without alignment: R(gate,s)-R(inv,s).
- Alignment effect without gating: R(adv_inv,s)-R(inv,s).
- Alignment effect with gating: R(adv_gate,s)-R(gate,s).
- Interaction: [R(adv_gate,s)-R(gate,s)]-[R(adv_inv,s)-R(inv,s)].

Report each seed's value and the mean/sample SD of differences. Only pair matching seed IDs. Three training seeds give weak precision and do not justify broad significance claims. Different gate architectures consume random numbers differently, so same seed IDs do not ensure identical optimization noise. Positive interaction supports complementarity under this protocol, not an automatic causal proof of information preservation.

## Mechanism diagnostics

Fit fixed-capacity linear ridge probes using frozen raw-input encoder features from validation episodes; evaluate on probe_test episodes. Report action MSE against a constant-action baseline, domain classification accuracy (balanced chance0.5), feature variance, and the mask retention statistics. Current/future pairs use the same sampler/horizon rules across methods. Normalization uses probe-fit data only. No test-based tuning of ridge penalty or encoder checkpoint is allowed. Probe samples can repeat within episodes; they are not independent trajectory repetitions.

Better action-prediction MSE indicates linear accessibility of this action signal, not proof of complete Markov-state sufficiency. Lower domain accuracy may reflect worse features or a probe too weak to decode domain information; inspect control returns and feature variance as well. Separate nonlinear probes and adversarial-strength sweeps would strengthen a later paper but are not guaranteed within this deadline.

## Evidence boundaries

Training uses two actual distracted archive conditions, easy and medium, with no clean training observations. Within-source episode splits are disjoint. An exact500-step action-sequence audit found no cross-split identical trajectories; absence of approximate/related physical states is unproven.

This is a BC+multi-step inverse representation experiment with optional learned gating and gradient-reversal alignment. It does not reproduce the original reward-based adversarial control algorithm or establish improvement over its published scores. Previously completed six-method runs use another dataset/budget/policy protocol and remain pilot context, not matched controls.

Static familiar/held-out evaluation uses different source images. Robustness to moving distractions, other tasks, long-run convergence and general method superiority remain untested unless separately measured. No full conference-readiness guarantee follows from completing this plan. All failed, skipped and time-limited conditions must be disclosed.
