"""Export small hybrid evidence without datasets or model weights."""
from pathlib import Path
import json,shutil,argparse,hashlib
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030'
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
 a=json.loads((O/'aggregate.json').read_text())
 if not args.allow_partial:assert a['completed_runs']==12,'Explicit --allow-partial required for incomplete study.'
 dest=R/'results/hybrid_20261030';dest.mkdir(parents=True,exist_ok=True)
 names=['aggregate.json','paired_effects.json','frozen_probes.jsonl','protocol.json','deadline_budget.json','data_manifest.json','data_audit.json','trajectory_overlap_audit.json','implementation_validation.json','completed_runs_audit.json','resource_observations.jsonl','analysis_plan.md','methods_draft.md','actual_results.md']
 names+=['dynamic_evaluation.jsonl','dynamic_summary.json','dynamic_results.md']
 for name in names:
  p=O/name
  if p.exists():shutil.copy2(p,dest/name)
 for p in (O/'runs').glob('*/*'):
  if p.name in ['result.json','metrics.jsonl','evaluation.jsonl']:
   target=dest/'runs'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 files=[]
 for p in dest.rglob('*'):
  if p.is_file():
   assert p.suffix in ['.json','.jsonl','.md'] and p.stat().st_size<10*1024**2
   files.append({'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (dest/'export_manifest.json').write_text(json.dumps({'complete':a['completed_runs']==12,'files':files,'scope':'Numerical results and manifests only; no weights, data, environments or third-party source.'},indent=2))
 print('Exported',len(files),'small evidence files. Review before staging.')
