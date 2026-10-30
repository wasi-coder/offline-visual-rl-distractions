from pathlib import Path
import json,statistics as S,sys
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';T=O/'ieee_template'
sys.path.insert(0,str(R/'report_dependencies'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
a=json.loads((O/'aggregate.json').read_text());d=json.loads((O/'dynamic_summary.json').read_text());e=json.loads((O/'paired_effects.json').read_text())['effects']
methods=['inv','gate','adv_inv','adv_gate'];labels=['Inv','IG','DA+Inv','DA+IG']
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(figsize=(3.4,2.4),layout='constrained')
for i,key in enumerate(['gating_without_alignment','gating_under_alignment']):
 row=next(x for x in e if x['contrast']==key and x['measurement']=='heldout_return')
 vals=list(row['seed_values'].values());ax.scatter([i]*3,vals,s=22);ax.plot([i-.15,i+.15],[S.mean(vals)]*2,color='black')
ax.axhline(0,color='gray',ls='--');ax.set_xticks([0,1],['Without alignment','With alignment']);ax.set_ylabel('Return change after adding gating');f.savefig(T/'figures/paired.png',dpi=300);plt.close(f)
lines=['### Additional comparisons','','The paired comparison asks how much the return changes when gating is added to the same training seed. Figure 3 shows that all three differences under alignment are negative. The three seeds are too few for a broad claim, but this makes the observed direction clear.','','FIGURE_PAIRED','','The methods use the same number of updates, but gating takes more computation per update. Table III reports measured update time only; data loading, evaluation and checkpoint writing are excluded. These values describe this GPU and implementation.','','| Method | Updates per run | Update minutes |','|---|---:|---:|']
for m,l in zip(methods,labels):
 vals=[x['training_seconds']/60 for x in a['runs'] if x['mode']==m];lines.append(f'| {l} | 40,000 | {S.mean(vals):.2f} +/- {S.stdev(vals):.2f} |')
lines+=['','### Exploratory moving-distraction test','','We also ran a separate stress test using five fresh evaluation seeds (901–905). Static and moving settings share these seeds, but the moving setting changes background, camera and colour together. Table IV reports means and sample SD across three trained seeds, with five episodes per seed and condition. These results were not used to select checkpoints.','','| Method | Static return | Moving return |','|---|---:|---:|']
for m,l in zip(methods,labels):
 v=d['modes'][m];fmt=lambda x:f"{x['mean']:.2f} +/- {x['sample_sd']:.2f}";lines.append(f"| {l} | {fmt(v['static'])} | {fmt(v['dynamic'])} |")
lines+=['','The method ranking changes in this secondary test: the combined model has a higher mean than alignment alone here, unlike the primary test. The static returns also change with the fresh evaluation seeds. This shows that the primary negative finding is specific to the tested evaluation set; the results do not establish that gating is always harmful or reliably helpful. Because several distraction types change together, this test cannot identify which one caused the differences.','']
p=T/'content.md';s=p.read_text(encoding='utf-8');assert '### Additional comparisons' not in s;s=s.replace('## Interpretation','\n'.join(lines)+'\n## Interpretation');p.write_text(s,encoding='utf-8')
print('Added one paired plot and two tables from saved measurements.')

