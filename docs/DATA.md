# Data and checkpoint acquisition

No dataset or checkpoint is included in Git. Obtain the distracted extended dataset using the links in the original project: https://github.com/Atine/offlineRL_domain_adversarial . V-D4RL: https://github.com/conglu1997/v-d4rl . Obtain DAVIS2017 480p image resources using the provider instructions linked by the original repository. Respect their terms.

Set OFFLINE_RL_ARCHIVE_DIR to the directory holding the downloaded ZIP shards. Run prepare_data.py to select25completeepisodes per easy cheetah medium-expert shard (8expected,200episodes100ktransitions), RNG20261029. Run prepare_small_data.py for the nested20episode10ksubset. Compare selected indices and data hashes with results/baseline_comparison/data_manifest.json and data_audit.json. These commands reproduce selection, not download copyrighted datasets automatically.

Set OFFLINE_RL_DAVIS_PATH to the familiar DAVIS JPEGImages/480p directory containing bear,bmx-bumps,boat,boxing-fisheye. Run prepare_eval_backgrounds.py to extract held-out `blackswan`, `bmx-trees`, `breakdance`, `camel` from the archived collection. Evaluation uses static distractors. No clean training samples enter replay.

Checkpoints remain local under runs/, runs_10k/, runs_100k_matched/. No public checkpoint URL exists yet. To publish them later, upload selected final checkpoints to release assets or a model host, add download URLs and SHA256 checksums, and document matching code versions. Until then fresh evaluation requires retraining or access to the local checkpoints. Never imply weights are downloadable before publishing them.
