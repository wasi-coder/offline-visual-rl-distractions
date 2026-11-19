"""Draw the implemented training architecture from the recorded hybrid protocol."""
from pathlib import Path
import sys,re
R=Path(__file__).resolve().parents[2];T=R/'experiments/hybrid_20261030/ieee_template'
sys.path.insert(0,str(R/'report_dependencies'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(6.5,7.6));ax.set_xlim(0,10);ax.set_ylim(0,12);ax.axis('off')
def box(x,y,w,h,text,shade='#f2f2f2'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=shade,edgecolor='black',linewidth=1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12.5)
def arrow(a,b,label=None):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color='black'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2,label,fontsize=11,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
box(.6,10.6,8.8,1,'Distracted offline data: easy + medium\nCurrent / future image stacks; actions; domain labels')
box(1.6,8.65,6.8,1.05,'Optional learned gate\nMasked + raw training samples', '#e5edf3');arrow((5,10.6),(5,9.7))
box(2.4,6.9,5.2,.85,'Shared image encoder');arrow((5,8.65),(5,7.75))
box(.35,4.1,2.8,1.35,'Inverse-action\npredictor\n(current + future)')
box(3.65,4.1,2.8,1.35,'Domain\nclassifier\n(easy / medium)')
box(6.95,4.1,2.7,1.35,'BC policy\nCopy recorded\nactions')
arrow((3.1,6.9),(1.75,5.45));arrow((5.05,6.9),(5.05,5.45));arrow((6.95,6.9),(8.3,5.45))
ax.text(8.3,6.1,'Stop gradient',fontsize=11,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
box(.35,1.6,2.8,1.5,'Inverse loss\nupdates encoder\nand active gate')
box(3.65,1.6,2.8,1.5,'Fit classifier on\ndetached features;\nreverse gradients*')
box(6.95,1.6,2.7,1.5,'Action MSE\nupdates policy\nonly')
for x in [1.75,5.05,8.3]:arrow((x,4.1),(x,3.1))
ax.text(5,.85,'* Alignment conditions only: reversed gradients update encoder / active gate.\nGate also has warm-up and sparsity penalties. Ungated modes bypass it.',fontsize=11,ha='center',va='center')
ax.text(5,.05,'Evaluation: raw images → encoder → policy → simulator (no updates).',fontsize=12,ha='center')
fig.tight_layout(pad=.5);fig.savefig(T/'figures/methodology.png',dpi=300,bbox_inches='tight');fig.savefig(T/'figures/methodology.svg',bbox_inches='tight');plt.close(fig)
p=T/'content.md';s=p.read_text(encoding='utf-8')
if 'FIGURE_METHOD' not in s:
 s=re.sub(r'Fig\. ([123])',lambda m:'Fig. '+str(int(m[1])+1),s)
 s=s.replace('Figure 3 shows','Figure 4 shows')
 s=s.replace('### Fixed data protocol','Fig. 1 summarises the implemented architecture. The inverse task learns image features, the domain head supplies the alignment signal, and behavioural cloning trains the policy. The gate and adversarial signal are enabled only in their respective conditions.\n\nFIGURE_METHOD\n\n### Fixed data protocol',1)
 p.write_text(s,encoding='utf-8')
print('Method diagram and manuscript cross-reference ready.')
