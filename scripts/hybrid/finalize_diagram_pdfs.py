from pathlib import Path
from pypdf import PdfReader,PdfWriter
T=Path(__file__).resolve().parents[2]/'experiments/hybrid_20261030/ieee_template'
for name in ['ECCE2027_Anonymous_Submission','ECCE2027_Gating_Alignment_Manuscript']:
 r=PdfReader(T/'tmp'/('diagram_ready_'+name)/(name+'.pdf'));assert len(r.pages)<=6
 w=PdfWriter();w.append(r);w.add_metadata({'/Author':'','/Creator':''});w.write(T/(name+'.pdf'))
 if 'Anonymous' in name:
  text=' '.join(p.extract_text() for p in r.pages).lower()
  assert not any(s in text for s in ['wasi','sadad','gmail','rajshahi'])
 print(name,len(r.pages),'pages verified')
