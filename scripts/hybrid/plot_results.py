from pathlib import Path
import sys,json,statistics
R=Path(__file__).resolve().parents[2];O=R/'experiments/hybrid_20261030';sys.path.insert(0,str(R/'report_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
if __name__=='__main__':
 a=json.loads((O/'aggregate.json').read_text());rows=a['summary'];figdir=O/'figures';figdir.mkdir(exist_ok=True)
 names={'inv':'Inverse + BC','gate':'Gating + inverse + BC','adv_inv':'Alignment + inverse + BC','adv_gate':'Alignment + gating + inverse + BC'}
 modes=[m for m in names if any(x['method']==m for x in rows)]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
 for j,cond in enumerate(['familiar_background','heldout_background']):
  selected=[next(x for x in rows if x['method']==m and x['condition']==cond) for m in modes]
  xx=np.arange(len(modes))+(j-.5)*.34
  ax.bar(xx,[x['mean'] for x in selected],.32,yerr=[x['seed_sd'] or 0 for x in selected],capsize=3,label=cond.replace('_',' '),alpha=.8)
  for x,item in zip(xx,selected):ax.scatter([x]*len(item['seed_means']),list(item['seed_means'].values()),s=15,c='black',zorder=5)
 ax.set_xticks(range(len(modes)),[names[m].replace(' + ','\n+ ') for m in modes]);ax.set_ylabel('Episode return');ax.set_title('Final40k-update checkpoints; static visual distractions');ax.legend(frameon=False)
 fig.savefig(figdir/'returns.png',dpi=220);fig.savefig(figdir/'returns.svg');plt.close(fig)
 probes=[json.loads(x) for x in (O/'frozen_probes.jsonl').read_text().splitlines()]
 fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
 for ax,key,title in zip(axes,['action_mse','domain_accuracy'],['Held-out action prediction MSE (lower is better)','Held-out linear domain probe accuracy']):
  for i,m in enumerate(modes):
   vals=[x[key] for x in probes if x['run'].startswith(m+'_seed')]
   ax.scatter([i]*len(vals),vals,s=25)
   if vals:ax.plot([i-.2,i+.2],[statistics.mean(vals)]*2,color='black')
  ax.set_xticks(range(len(modes)),[m for m in modes],rotation=15);ax.set_title(title)
  if key=='domain_accuracy':ax.axhline(.5,color='gray',linestyle='--',label='Balanced chance');ax.legend(frameon=False);ax.set_ylim(0,1)
 fig.savefig(figdir/'probes.png',dpi=220);fig.savefig(figdir/'probes.svg');plt.close(fig)
 print('Hybrid figures generated; visual review required.')
