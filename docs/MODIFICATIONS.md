# Packaging and experiment changes

Original local source and runs are retained in ignored directories. Publication source is copied into src/, envs/ and configs/upstream. Executable runners moved to scripts/baseline; hardcoded study paths replaced with repository-relative paths. Dataset archives and familiar DAVIS images are configured using OFFLINE_RL_ARCHIVE_DIR and OFFLINE_RL_DAVIS_PATH. Original experiment numbers are preserved; local paths in exported evidence were redacted.

Study-specific code implements equal-time baseline/augmentation/consistency/fixed-mask controls, an external author-code InfoGating adapter, deterministic episode selection, held-out backgrounds and occlusion tests, metadata audits and aggregation. This packaging does not add the proposed hybrid. Derived source preserves upstream licences. Historical source hashes refer to the original study, not this refactored copy; a separate publication hash manifest is provided.
