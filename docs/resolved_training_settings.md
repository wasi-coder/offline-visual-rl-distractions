# Resolved training settings and schedule caveat

The custom runner overrides the source config batch size256 with16, replay capacity1,000,000 with102,000, and training termination with measured update seconds. Source config num_timesteps and eval_episodes are not the custom runner's budgets. The runner calls update with step=2*completed_update_count.

| Setting | Core methods |
|---|---|
| Optimizer learning rate | 0.0003 |
| Feature dimension | 50 |
| Hidden dimension | 1024 |
| Batch size | 16 |
| Discount / n-step | 0.99 / 3 |
| Target update coefficient | 0.01 |
| BC configuration weight | 2.5, source Q-normalized actor loss |
| Image history | Three84x84 RGB frames |
| Action repeat | 2 |
| Training action standard deviation | Linear1.0to0.1 over250,000 source steps |
| Action noise clip | 0.3 |
| Additional consistency coefficient | 1 |
| RGB gain range | 0.6to1.4 perchannel, temporally shared |
| Fixed mask | Top16rows replaced by gray |
| Training seeds | 0,1,2 |

Adapted InfoGating uses the author architecture, feature256, hidden1024, learning rate0.0001, lambda0.1 and the same form of action-standard-deviation schedule, with batch16. Its representation warmup and sampled-action BC differ from the core methods. See literature_audit.md for all protocol differences.

**Schedule confounding:** equal update time gives different update counts and therefore different endpoints of the source action-noise schedule. For example, about32,000 core DrQ updates reach source step64,000, giving a standard deviation about0.77; about16,600 InfoGating updates reach step33,200, giving about0.88. BC optimizes the policy mean and does not use sampled-action noise in its MSE. Differences in stochastic training targets and warmup progress are part of this practical equal-time comparison. They prevent treating this experiment as an isolated test of representation quality. A stronger follow-up should include matched-update or normalized-schedule comparisons and tune methods with a validation protocol before testing.

Saved checkpoint/metric update counts are authoritative; approximate examples here are explanatory. Evaluation takes deterministic mean actions, so this schedule caveat concerns optimization, not random action sampling during the reported test episodes.
