"""Validate and summarize the predeclared exploratory secondary stress test."""
from pathlib import Path
import json, math, statistics as S
R=Path(__file__).resolve().parents[2]; O=R/'experiments/hybrid_20261030'
rows=[json.loads(s) for s in (O/'dynamic_evaluation.jsonl').read_text().splitlines() if s.strip()]
expected={(f'{mode}_seed{seed}',ev,d) for mode in ['inv','gate','adv_inv','adv_gate'] for seed in range(3) for ev in range(901,906) for d in [False,True]}
keys={(r['run'],r['seed'],r['dynamic']) for r in rows}
assert len(rows)==len(keys)==120 and keys==expected
assert all(r['steps']==500 and math.isfinite(r['return']) for r in rows)
out={'scope':'Exploratory secondary new-seed held-out test. Dynamic changes background, camera and colour jointly; not a background-only causal test. No model selection.', 'episodes':120,'modes':{}}
lines=['# Exploratory secondary static/dynamic stress test','','Five fresh evaluation seeds (901–905), separate from the primary test. Each table entry is the mean and sample SD over three training-seed means; each seed mean averages five episodes. Dynamic changes background, camera and colour together. These are secondary observations, not checkpoint-selection evidence.','','| Condition | Static return | Dynamic return | Paired dynamic minus static |','|---|---:|---:|---:|']
for mode in ['inv','gate','adv_inv','adv_gate']:
 vals={d:[S.mean(r['return'] for r in rows if r['mode']==mode and r['training_seed']==s and r['dynamic']==d) for s in range(3)] for d in [False,True]}
 delta=[b-a for a,b in zip(vals[False],vals[True])]
 stats=lambda v:dict(mean=S.mean(v),sample_sd=S.stdev(v),seed_means=v)
 out['modes'][mode]={'static':stats(vals[False]),'dynamic':stats(vals[True]),'paired_change':stats(delta)}
 fmt=lambda v:f'{S.mean(v):.2f} +/- {S.stdev(v):.2f}'
 lines.append(f'| {mode} | {fmt(vals[False])} | {fmt(vals[True])} | {fmt(delta)} |')
lines+=['','This test does not establish a causal explanation for the primary negative gating result. It uses one task and three trained seeds per condition; no significance or broad robustness claim is made. The manuscript reports the primary static protocol; this stress test is retained as a separate supplement.']
(O/'dynamic_summary.json').write_text(json.dumps(out,indent=2))
(O/'dynamic_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
