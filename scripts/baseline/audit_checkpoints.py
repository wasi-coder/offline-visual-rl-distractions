"""Audit saved final model weights on CPU without advancing training."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
records=[]
for folder in ['runs','runs_10k','runs_100k_matched']:
 for result_path in sorted((ROOT/folder).glob('*/result.json')):
  result=json.loads(result_path.read_text());run=result_path.parent
  comparator=run.name.startswith('infogating_')
  path=run/('final.pt' if comparator else 'latest.pt')
  checkpoint=torch.load(path,map_location='cpu',weights_only=False)
  expected={'encoder','mask_net','inv_model','policy'} if comparator else {'encoder','actor','critic','critic_target'}
  assert set(checkpoint['models'])==expected,(str(path),'model keys')
  step=checkpoint['updates'] if comparator else checkpoint['step']
  assert step==result['updates'],(str(path),'step mismatch')
  assert abs(checkpoint['training_seconds']-result['training_seconds'])<1e-6,(str(path),'timer mismatch')
  count=0
  for module,state in checkpoint['models'].items():
   for name,tensor in state.items():
    assert torch.isfinite(tensor).all(),(str(path),module,name)
    count+=tensor.numel()
  records.append({'run':folder+'/'+run.name,'checkpoint':str(path),'updates':step,'training_seconds':checkpoint['training_seconds'],'state_tensor_elements':count,'finite':True,'result_metadata_matches':True})
  del checkpoint
out={'checked_runs':len(records),'all_completed_checkpoints_passed':True,'records':records,'scope':'Final saved model tensors finite; model keys and update/timer metadata match result.json. Does not prove convergence or algorithm correctness.'}
(ROOT/'reports/checkpoint_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
