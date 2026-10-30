from pathlib import Path
import json,statistics
O=Path(__file__).resolve().parents[2]/'experiments/hybrid_20261030'
if __name__=='__main__':
 a=json.loads((O/'aggregate.json').read_text());p=[json.loads(x) for x in (O/'frozen_probes.jsonl').read_text().splitlines()]
 def ret(mode,seed):
  row=next(x for x in a['summary'] if x['method']==mode and x['condition']=='heldout_background')
  return row['seed_means'][str(seed)]
 def probe(mode,seed,key):return next(x[key] for x in p if x['run']==f'{mode}_seed{seed}')
 effects=[]
 contrasts={'gating_under_alignment':{'adv_gate':1,'adv_inv':-1},'gating_without_alignment':{'gate':1,'inv':-1},'alignment_without_gating':{'adv_inv':1,'inv':-1},'alignment_with_gating':{'adv_gate':1,'gate':-1},'interaction':{'adv_gate':1,'gate':-1,'adv_inv':-1,'inv':1}}
 for name,weights in contrasts.items():
  for measurement in ['heldout_return','action_mse','domain_accuracy']:
   values={}
   for seed in [0,1,2]:
    try:values[seed]=sum(w*(ret(mode,seed) if measurement=='heldout_return' else probe(mode,seed,measurement)) for mode,w in weights.items())
    except (KeyError,StopIteration):pass
   effects.append({'contrast':name,'measurement':measurement,'matched_seed_count':len(values),'seed_values':values,'mean_difference':statistics.mean(values.values()) if values else None,'sample_sd_difference':statistics.stdev(values.values()) if len(values)>1 else None})
 out={'effects':effects,'all_three_seed_contrasts_available':all(x['matched_seed_count']==3 for x in effects),'interpretation':'Return: positive better. Action MSE: negative better. Domain accuracy: lower only suggests less linear domain decodability, not guaranteed invariance. Small N; descriptive paired differences, no significance assertion.'}
 (O/'paired_effects.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
