from pathlib import Path
import sys,json,statistics
r=Path(r'E:\offline_rl');o=r/'experiments/hybrid_20261030';sys.path.insert(0,str(r/'report_dependencies'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=o/'ieee_template/figures';out.mkdir(parents=True,exist_ok=True);a=json.loads((o/'aggregate.json').read_text());methods=['inv','gate','adv_inv','adv_gate'];plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(figsize=(3.4,2.7),layout='constrained')
for j,c in enumerate(['familiar_background','heldout_background']):
 rows=[next(x for x in a['summary'] if x['method']==m and x['condition']==c) for m in methods];xx=np.arange(4)+(j-.5)*.35
 ax.bar(xx,[x['mean'] for x in rows],.32,yerr=[x['seed_sd'] for x in rows],capsize=2,label=['Familiar','Held out'][j],alpha=.85)
 for x,v in zip(xx,rows):ax.scatter([x]*3,list(v['seed_means'].values()),c='black',s=8,zorder=4)
ax.set_xticks(range(4),['Inv','IG','DA+Inv','DA+IG']);ax.set_ylabel('Episode return');ax.legend(frameon=False,ncol=2,fontsize=7);f.savefig(out/'returns.png',dpi=300);plt.close(f)
p=[json.loads(x) for x in (o/'frozen_probes.jsonl').read_text().splitlines()];f,ax=plt.subplots(figsize=(3.4,2.4),layout='constrained')
for i,m in enumerate(methods):
 vals=[x['action_mse'] for x in p if x['run'].startswith(m+'_seed')];ax.scatter([i]*3,vals,s=18);ax.plot([i-.18,i+.18],[statistics.mean(vals)]*2,c='black')
ax.set_xticks(range(4),['Inv','IG','DA+Inv','DA+IG']);ax.set_ylabel('Held-out action probe MSE');f.savefig(out/'probes.png',dpi=300);plt.close(f)
print('Column-sized scientific figures ready')
