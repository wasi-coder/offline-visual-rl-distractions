"""Independent published-evidence arithmetic check (no GPU or training)."""
import subprocess,sys,unittest
from pathlib import Path
class EvidenceTest(unittest.TestCase):
 def test_archived_evidence(self):
  subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/verify_archived_results.py')],check=True)
if __name__=='__main__':unittest.main()
