from pathlib import Path
import sys,json,statistics
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'report_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
figdir=root/'reports/figures';figdir.mkdir(exist_ok=True)
data=json.loads((root/'reports/aggregate.json').read_text())
if not data['complete']:raise SystemExit('Final plots require complete study; no missing results filled in.')
methods=['bc','drq','strong','consistency','mask','infogating']
labels=['BC','DrQv2+BC','RGB aug.','Consistency','Top-band mask','InfoGating*']
colors=['#355c7d','#b35c44']
fig,ax=plt.subplots(figsize=(7.2,3.3),layout='constrained')
for j,condition in enumerate(['familiar_background','heldout_background']):
 rows=[next(r for r in data['summary'] if r['suite']=='main' and r['method']==m and r['condition']==condition and r['damage_band_pixels']==0) for m in methods]
 x=np.arange(len(methods))+(j-.5)*.32
 ax.bar(x,[r['mean_return'] for r in rows],.3,yerr=[r['std_across_seed_means'] for r in rows],capsize=3,label=['Familiar sources','Held-out sources'][j],color=colors[j],alpha=.85)
 for xx,r in zip(x,rows):ax.scatter([xx]*3,list(r['seed_means'].values()),s=12,color='black',zorder=5)
ax.set_xticks(range(len(methods)),labels,rotation=12);ax.set_ylabel('Episode return');ax.legend(frameon=False);ax.set_title('100k transitions; 20 minutes per training seed')
fig.savefig(figdir/'main_returns.png',dpi=220);fig.savefig(figdir/'main_returns.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
for ax,condition in zip(axes,['familiar_background','heldout_background']):
 for j,m in enumerate(['bc','drq']):
  rows=[next(r for r in data['summary'] if r['suite']=='size' and r['method']==m and r['transitions']==n and r['condition']==condition) for n in [10000,100000]]
  ax.errorbar([10,100],[r['mean_return'] for r in rows],yerr=[r['std_across_seed_means'] for r in rows],marker='o',capsize=3,label=labels[methods.index(m)],color=colors[j])
 ax.set_xticks([10,100],['10k','100k']);ax.set_xlabel('Training transitions');ax.set_title(condition.replace('_',' '));ax.set_ylabel('Episode return');ax.legend(frameon=False)
fig.savefig(figdir/'data_size.png',dpi=220);fig.savefig(figdir/'data_size.svg');plt.close(fig)
rows=[json.loads(x) for x in (root/'reports/final_evaluation.jsonl').read_text().splitlines()]
fig,ax=plt.subplots(figsize=(7.2,3.2),layout='constrained')
for m,label in zip(methods,labels):
 means=[];std=[]
 for band in [0,8,16,32]:
  values=[]
  for seed in range(3):
   base=next(r['return'] for r in rows if r['mode']==m and r['training_seed']==seed and r['condition']=='heldout_background' and r['eval_seed']==701 and r['damage_band_pixels']==0)
   val=next(r['return'] for r in rows if r['mode']==m and r['training_seed']==seed and r['condition']=='heldout_background' and r['eval_seed']==701 and r['damage_band_pixels']==band)
   values.append(val-base)
  means.append(statistics.mean(values));std.append(statistics.stdev(values))
 ax.errorbar([0,8,16,32],means,yerr=std,marker='o',capsize=2,label=label,linewidth=1)
ax.axhline(0,color='gray',linewidth=.7);ax.set_xticks([0,8,16,32]);ax.set_xlabel('Central occlusion band height (pixels)');ax.set_ylabel('Return change from unoccluded episode');ax.legend(frameon=False,ncol=3,fontsize=8);ax.set_title('Paired diagnostic; one evaluation seed per training seed')
fig.savefig(figdir/'mask_damage.png',dpi=220);fig.savefig(figdir/'mask_damage.svg');plt.close(fig)
print('Saved six final plot files')
