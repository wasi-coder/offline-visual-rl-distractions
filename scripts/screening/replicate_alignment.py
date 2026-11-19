"""Launch a new paired cohort without writing to the original experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_alignment as study

if __name__ == '__main__':
    study.OUTPUT = study.ROOT / 'experiments/alignment_replication_20261109'
    study.SEEDS = [3, 4, 5]
    study.TARGET = 44000
    # Match the earlier simulator cases to isolate replication across training seeds.
    study.main(replication=True)
