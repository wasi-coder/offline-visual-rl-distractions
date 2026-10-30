"""CPU-only verification of exported hybrid arithmetic and file hashes."""
from pathlib import Path
import json,hashlib,statistics as S,math
R=Path(__file__).resolve().parents[2]/'results/hybrid_20261030'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=read(R/'aggregate.json');assert a['completed_runs']==12
manifest=read(R/'export_manifest.json')
for f in manifest['files']:
 p=R/f['path'];assert p.stat().st_size==f['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for s in a['summary']:
 means=[]
 for p in sorted((R/'runs').glob(s['method']+'_seed*/evaluation.jsonl')):
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  assert len(rows)==20 and len({(x['updates'],x['condition'],x['eval_seed']) for x in rows})==20
  assert all(x['steps']==500 and math.isfinite(x['return']) for x in rows)
  final=[x['return'] for x in rows if x['updates']==40000 and x['condition']==s['condition']]
  assert len(final)==5;means.append(S.mean(final))
 assert len(means)==3 and abs(S.mean(means)-s['mean'])<1e-9 and abs(S.stdev(means)-s['seed_sd'])<1e-9
probes=[json.loads(x) for x in (R/'frozen_probes.jsonl').read_text().splitlines()];assert len(probes)==12
dynamic=[json.loads(x) for x in (R/'dynamic_evaluation.jsonl').read_text().splitlines()]
assert len(dynamic)==120 and len({(r['run'],r['seed'],r['dynamic']) for r in dynamic})==120
assert all(x['steps']==500 and math.isfinite(x['return']) for x in dynamic)
print('PASS: file hashes, 12 completed runs, primary return arithmetic, 12 probe records and 120 secondary episodes. This checks saved evidence, not scientific validity or simulator independence.')
