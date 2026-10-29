# Reproduction

## Verify the published numbers without a GPU or dataset
```sh
python scripts/verify_archived_results.py
```
This independently recomputes all archived means and sample standard deviations from234unique500-step evaluation episodes. It verifies arithmetic and recorded identities, not scientific validity or successful fresh training.

## Environment
Python3.11; original training used PyTorch2.5.1+cu121 on GTX1650. Install PyTorch for your hardware, then:
```sh
pip install -r requirements.txt
```
Full original package inventory: results/baseline_comparison/requirements_snapshot.txt. This is an observed environment snapshot, not a portable lockfile. CUDA training and simulator rendering require a suitable local environment; headless rendering was not tested.

## Data and upstream InfoGating
Follow DATA.md. InfoGating source is deliberately not redistributed because the inspected snapshot has no licence. If you obtain it under appropriate terms, place it at references/agent-centric-representations-main. Optional retrieval helper requires an explicit revision:
```sh
python scripts/setup_infogating.py --revision UPSTREAM_COMMIT
```
The original download was a ZIP without Git metadata; its commit is not known. Compare file hashes against docs/infogating_source_hashes.json to establish the exact source match rather than claiming a pinned historical commit.

## Prepare the same subsets
Set OFFLINE_RL_ARCHIVE_DIR and OFFLINE_RL_DAVIS_PATH as described in DATA.md. Then:
```sh
python scripts/baseline/prepare_data.py
python scripts/baseline/prepare_small_data.py
python scripts/baseline/prepare_eval_backgrounds.py
python scripts/baseline/audit_data.py
```
Create reports/ before preparation (`mkdir reports` if absent). Compare selection manifests and hashes with the archived results.

## Run the recorded protocol, in order
These are substantial CUDA training jobs; do not run duplicate instances. Runners skip runs containing result.json and can resume available checkpoints. Existing local experiment runs are preserved, so in this workspace completed runs will be skipped. For a genuinely fresh replication use a separate clone with empty runs/ directories.
```sh
python scripts/baseline/core_suite.py
python scripts/baseline/published_comparator.py
python scripts/baseline/evaluate_study.py
python scripts/baseline/small_data_suite.py
python scripts/baseline/matched_data_suite.py
python scripts/baseline/evaluate_size.py
python scripts/baseline/aggregate_results.py
python scripts/baseline/audit_checkpoints.py
python scripts/baseline/plot_results.py
python scripts/baseline/write_result_tables.py
```
Dependencies wait on earlier output files; follow the order. The main suite uses15core runs and3adaptedInfoGating runs at1200update seconds each; the size pilot12runs at300seconds. Total7update hours plus loading, checkpointing and evaluation. Time-based stopping and hardware variation change update counts. Standard-deviation schedules progress with updates, not elapsed time. Test distractions are static within episodes.

Original results remain under results/baseline_comparison. Newly generated outputs go to ignored runs/, reports/ and videos/. The --validate switch in training runners uses local data and performs updates; it is not a dataset-free unit test.

Portable packaging was syntax/import checked and archived arithmetic was verified. A fresh end-to-end seven-hour replication after the directory refactor has not been performed.
