"""Build IEEE-style empirical draft and readable supervisor PDF from complete evidence."""
from pathlib import Path
import json,re
from xml.sax.saxutils import escape
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,PageBreak,NextPageTemplate,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parents[2];D=R/'reports';O=R/'output/pdf';O.mkdir(parents=True,exist_ok=True)
a=json.loads((D/'aggregate.json').read_text())
if not a['complete'] or a['validation_errors']:raise SystemExit('Verified complete results required.')
if json.loads((D/'checkpoint_audit.json').read_text())['checked_runs']!=30:raise SystemExit('Audit all 30 checkpoints first.')
W,H=A4;M=40;G=18;CW=(W-2*M-G)/2
body=ParagraphStyle('body',fontName='Times-Roman',fontSize=9.5,leading=11.4,alignment=TA_JUSTIFY,spaceAfter=5)
head=ParagraphStyle('head',parent=body,fontName='Times-Bold',fontSize=10,spaceBefore=9,keepWithNext=True,alignment=0)
small=ParagraphStyle('small',parent=body,fontSize=7.5,leading=9,alignment=0)
widebody=ParagraphStyle('wide',parent=body,fontName='Helvetica',fontSize=10,leading=14,alignment=0,spaceAfter=8)
def clean(s):
 return s.replace(' (author surname: Guzel with umlaut on u).','').replace('Guzel','Güzel').replace('**','').replace('`','')
def para(s,style=body):return Paragraph(escape(clean(s)),style)
def footer(c,doc):
 c.setFont('Helvetica',7);c.setFillColor(colors.HexColor('#555555'));c.drawString(M,22,'Actual-evidence research draft | 29 October 2026');c.drawRightString(W-M,22,str(doc.page))
def document(path,title,two):
 doc=BaseDocTemplate(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=42,bottomMargin=38,title=title)
 def frame(x,y,w,h,name):return Frame(x,y,w,h,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
 if two:
  frames=[frame(M,38,CW,H-80,'left'),frame(M+CW+G,38,CW,H-80,'right')]
 else:frames=[frame(M,38,W-2*M,H-80,'full')]
 doc.addPageTemplates([PageTemplate(id='normal',frames=frames,onPage=footer),PageTemplate(id='wide',frames=[frame(M,38,W-2*M,H-80,'fig')],onPage=footer)])
 return doc
def markdown(text,width,style=body):
 out=[];lines=text.splitlines();i=0
 while i<len(lines):
  line=lines[i].strip()
  if not line:i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
    if not all(re.fullmatch(r'[-: ]+',x) for x in cells):rows.append(cells)
    i+=1
   n=len(rows[0]);ts=small if width==CW else ParagraphStyle('t',parent=widebody,fontSize=8,leading=10)
   table=Table([[para(c,ts) for c in row] for row in rows],colWidths=[width/n]*n,repeatRows=1,hAlign='LEFT')
   table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf2')),('LINEBELOW',(0,0),(-1,0),.5,colors.gray),('LINEBELOW',(0,-1),(-1,-1),.4,colors.gray),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]));out+=[table,Spacer(1,7)];continue
  if line.startswith('#'):
   out.append(para(line.lstrip('# ').strip(),head));i+=1;continue
  text=line;i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):
   text+=' '+lines[i].strip();i+=1
  out.append(para(text,style))
 return out
related=(D/'related_work_and_references.md').read_text()
review=related.split('## Related work draft',1)[1].split('## References',1)[0].strip()
refs=related.split('## References',1)[1].split('## Source verification notes',1)[0].strip()
methods=(D/'manuscript_methods_draft.md').read_text()
methods=methods.split('## Research question',1)[1]
methods=re.sub(r'(?m)^([1-6]\. )', r'\n\1', methods)
settings=(D/'resolved_training_settings.md').read_text().split('| Setting |',1)[1].split('Adapted InfoGating',1)[0]
methods+='\n\n## Implementation settings\n\nThe study ran on an NVIDIA GTX 1650 with 4 GiB GPU memory, Python 3.11.9 and PyTorch 2.5.1+cu121. Installed software versions are archived in environment_snapshot.json.\n\n| Setting |'+settings
tables=(D/'results_tables.md').read_text()
# Keep the wide resource table in the reproducibility supplement, not a narrow column.
results=tables.split('## Main comparison',1)[1].split('## Compute',1)[0]
results='## Main comparison'+results
rows=[r for r in a['summary'] if r['suite']=='main' and r['condition']=='heldout_background' and r['damage_band_pixels']==0]
best=max(rows,key=lambda r:r['mean_return'])
abstract=f"We study visual offline reinforcement learning using distracted data alone under a limited local compute budget. Six methods are trained on 100,000 cheetah-run transitions with three seeds and 1,200 measured update seconds per run. A matched 300-second pilot compares 10,000 and 100,000 transitions for BC and DrQv2+BC. The study records {a['recorded_training_hours']:.2f} update hours and 234 evaluation episodes. The highest observed held-out-source mean is {best['method']} ({best['mean_return']:.2f}, seed standard deviation {best['std_across_seed_means']:.2f}). Evaluation distractions are static within episodes. Results are descriptive and specific to this task and budget; the adapted InfoGating comparator is not a reproduction of published performance."
text='# Visual Offline Reinforcement Learning from Distracted Data: A Compute-Limited Cheetah Study\n\nEmpirical draft for supervisor review. Author and institutional metadata must be supplied before submission.\n\n## Abstract\n\n'+abstract+'\n\n## I. Motivation and related work\n\n'+review+'\n\n## II. Experimental methods\n\n'+methods+'\n\n## III. Actual results\n\n'+results+'\n\n## IV. Interpretation and submission scope\n\nThese observations compare practical interventions under equal update time. They do not isolate every architectural difference or establish a generally superior method. Seed variation and short training budgets require caution. Occlusion differences are paired closed-loop diagnostics, not proof that an encoder discarded causal or noncausal features. The earlier adaptive weighting proposal was not implemented. A submission requires a justified contribution, broader task coverage, stronger comparator protocol matching and expanded uncertainty analysis.\n\n## References\n\n'+refs
interpretation=(D/'main_results_interpretation.md').read_text().split('Five methods have lower averages',1)[1]
interpretation='Five methods have lower averages'+interpretation
size_notes=[]
for method in ['bc','drq']:
 def size_result(n):return next(r for r in a['summary'] if r['suite']=='size' and r['method']==method and r['transitions']==n and r['condition']=='heldout_background' and r['damage_band_pixels']==0)
 low,high=size_result(10000),size_result(100000)
 size_notes.append(f"For {method.upper()}, the held-out mean changed from {low['mean_return']:.2f} (seed SD {low['std_across_seed_means']:.2f}) with 10,000 transitions to {high['mean_return']:.2f} (seed SD {high['std_across_seed_means']:.2f}) with 100,000 transitions under the matched 300-second budget. The difference of {high['mean_return']-low['mean_return']:.2f} is descriptive; these short runs do not establish a general data-scaling benefit.")
interpretation+='\n\n'+ '\n\n'.join(size_notes)
damage=json.loads((D/'occlusion_paired_results.json').read_text())
interpretation+='\n\n## Occlusion findings\n\n'
for method in ['bc','drq','strong','consistency','mask','infogating']:
 r=next(x for x in damage if x['method']==method and x['band_pixels']==32)
 interpretation+=f"For {method}, the 32-pixel central band changed return by {r['mean_change']:.2f} on average (paired-change seed SD {r['sd_change']:.2f}). "
interpretation+='Only one evaluation seed per checkpoint was used for this stress test. Low unoccluded returns can create a floor effect; small deterioration does not by itself indicate robust control.'
text=text.replace('## IV. Interpretation and submission scope','## IV. Interpretation and submission scope\n\n'+interpretation)
(D/'IEEE_actual_evidence_paper.md').write_text(text,encoding='utf-8')
story=markdown(text,CW)
story += [NextPageTemplate('wide'),PageBreak(),para('Visual results and diagnostic controls',head)]
for name,caption in [('main_returns','Main comparison: means and sample SD across three training-seed means. Each seed mean uses three evaluation episodes. InfoGating* is adapted author code.'),('data_size','Dataset-size pilot: matched 300-second budgets. Whiskers show sample SD across training-seed means.'),('mask_damage','Paired occlusion diagnostic: return change relative to the same checkpoint with no central occlusion, evaluation seed 701 only. Bands are not semantic masks.')]:
 from PIL import Image as PILImage
 with PILImage.open(D/'figures'/f'{name}.png') as im: iw,ih=im.size
 image=Image(str(D/'figures'/f'{name}.png'),width=(W-2*M)*.90,height=ih/iw*(W-2*M)*.90);story += [image,para(caption,small),Spacer(1,9)]
document(O/'IEEE_actual_evidence_paper.pdf','Visual offline RL: actual evidence',True).build(story)
summary=(D/'Supervisor_actual_results_summary.md').read_text().split('\n\n## Evaluation scope',1)[0]
summary+='\n\n## Dataset-size findings\n\n'+'\n\n'.join(size_notes)+'\n\n## Evaluation scope\n\nBackground images, colours and camera distractions remain static within each episode (dynamic=False). These tests measure transfer to unseen source images, not robustness to moving distraction videos.\n\n## Papers I reviewed\n\n'+review+'\n\n## References\n\n'+refs
(D/'Supervisor_actual_results_summary.md').write_text(summary,encoding='utf-8')
document(O/'Supervisor_actual_results_summary.pdf','Offline RL supervisor progress report',False).build(markdown(summary,W-2*M,widebody))
print('Created actual-evidence paper and supervisor PDF; visual QA still required.')
