from pathlib import Path
from io import BytesIO
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parents[2];src=Path(r'C:\Users\wasia\Desktop\RLHF papers\my papers\ilovepdf_merged.pdf')
out=src.with_name('Supervisor_Summary_With_Methodology_Diagram.pdf')
pdfmetrics.registerFont(TTFont('TimesReport',r'C:\Windows\Fonts\times.ttf'));pdfmetrics.registerFont(TTFont('TimesBoldReport',r'C:\Windows\Fonts\timesbd.ttf'))
body=ParagraphStyle('body',fontName='TimesReport',fontSize=11,leading=14,spaceAfter=8)
head=ParagraphStyle('head',fontName='TimesBoldReport',fontSize=16,leading=19,spaceAfter=13)
sub=ParagraphStyle('sub',fontName='TimesBoldReport',fontSize=12,leading=15,spaceAfter=7)
buf=BytesIO();story=[]
def para(s,st=body):story.append(Paragraph(s,st))
para('6.1 Methodology Block Diagram',head)
para('The diagram shows the implemented pipeline used in the controlled study. All conditions use inverse action prediction and behavioural cloning. Learned gating and adversarial alignment are enabled only in the conditions being tested.')
pic=R/'experiments/hybrid_20261030/ieee_template/figures/methodology.png'
from PIL import Image as PI
with PI.open(pic) as im:w,h=im.size
story.append(Image(str(pic),width=420,height=420*h/w))
para('Figure 1. Training and evaluation pipeline for the controlled gating-and-alignment study. Arrows show feature flow; the lower boxes state which components are updated. The adversarial signal reverses the representation gradient, while the BC policy receives detached features.',ParagraphStyle('caption',parent=body,fontSize=9,leading=11))
story.append(PageBreak());para('6.2 Explanation of the Methodology',head)
items=[('1 Recorded observations and training domains','We sample recorded cheetah observations from easy and medium distraction datasets. Each training batch contains eight samples from each domain. Current and later image stacks come from the same recorded episode. Actions provide the prediction target; domain labels tell the classifier which dataset a sample came from. Clean observations are not used for training.'),('2 Optional learned gate and shared encoder','In the gated conditions, a learned mask weakens selected pixels. Half of each current/future minibatch remains unmasked, following the adopted author-code convention. Warm-up and sparsity penalties control gate training. The shared encoder turns the image stacks into compact features. A gate is not a verified cheetah segmentation mask: it can retain distracting details or hide useful information.'),('3 Inverse action prediction','The inverse model uses features from current and later observations to score the recorded action against generated alternatives. This encourages features that help explain actions. Its loss updates the encoder and, after warm-up, the active gate. This encourages action-related information but does not guarantee that all control information is preserved.'),('4 Domain-adversarial alignment','The domain classifier first learns to distinguish easy from medium using detached features. Its weights are then held fixed while a reversed gradient encourages the encoder and active gate to make domain prediction harder. This reversed signal is enabled only for DA+Inv and DA+IG. Inv and IG retain the classifier for diagnostics without adversarial representation updates.'),('5 Behavioural cloning policy','The policy learns to copy recorded actions by minimizing action mean squared error. The stop-gradient branch prevents this policy loss from updating the encoder or gate. All four conditions share this policy rule. There is no reward-learning critic in this controlled study; rewards are used to evaluate behaviour.'),('6 Evaluation and the four comparisons','Evaluation sends raw, unmasked images through the encoder and policy to choose simulator actions. The gate and domain classifier are not needed on this inference path, and no model updates occur. Inv enables neither gating nor alignment; IG enables gating; DA+Inv enables alignment; DA+IG enables both. These controls test whether adding a gate changes performance under alignment. The primary negative result and separate moving-distraction results are reported in Chapter 7.')]
for title,text in items:para(title,sub);para(text)
SimpleDocTemplate(buf,pagesize=A4,leftMargin=57,rightMargin=57,topMargin=52,bottomMargin=52).build(story)
insert=PdfReader(buf);assert len(insert.pages)==2
original=PdfReader(src);assert len(original.pages)==11
# Replace only contents; preserve the user's custom cover and original chapter content.
toc=BytesIO();c=canvas.Canvas(toc,pagesize=A4);c.setFont('TimesBoldReport',18);c.drawString(57,783,'Contents');y=752
rows=[('1 Foundations of Reinforcement Learning',3),('2 Offline Learning and the Research Problem',4),('3 Learning Progress and Literature Review',5),('4 Initial Pipeline Validation',6),('5 Comparative Study of Six Training Methods',7),('6 Controlled Study of Gating and Alignment',8),('    6.1 Methodology Block Diagram',9),('    6.2 Explanation of the Methodology',10),('7 Experimental Results and Their Meaning',11),('8 Research Progress and Manuscript Preparation',12),('9 Conclusions and Future Research',13)]
c.setFont('TimesReport',11)
for text,num in rows:c.drawString(57,y,text);c.drawRightString(535,y,str(num));y-=25
c.save();contents=PdfReader(toc).pages[0]
pages=[original.pages[0],contents]+list(original.pages[2:8])+list(insert.pages)+list(original.pages[8:])
writer=PdfWriter()
for i,page in enumerate(pages,1):
 if i>1:
  overlay=BytesIO();cc=canvas.Canvas(overlay,pagesize=(float(page.mediabox.width),float(page.mediabox.height)))
  cc.setFillColorRGB(1,1,1);cc.rect(510,25,60,25,fill=1,stroke=0);cc.setFillColorRGB(0,0,0);cc.setFont('TimesReport',10);cc.drawRightString(539,39,str(i));cc.save();page.merge_page(PdfReader(overlay).pages[0])
 writer.add_page(page)
writer.write(out);print(out)
