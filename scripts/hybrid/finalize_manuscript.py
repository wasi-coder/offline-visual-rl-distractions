"""Verify template fidelity and the rendered, manually inspected final manuscript."""
from pathlib import Path
import zipfile,hashlib,json,shutil
from pypdf import PdfReader
from lxml import etree as E
R=Path(__file__).resolve().parents[2]; O=R/'experiments/hybrid_20261030'; T=O/'ieee_template'
source=Path(r'C:\Users\wasia\Desktop\IEEE-paper-format-template.docx')
docx=T/'ECCE2027_Gating_Alignment_Manuscript.docx'
pdf=T/'ECCE2027_Gating_Alignment_Manuscript.pdf'
shutil.copy2(T/'tmp/render2'/pdf.name,pdf)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='51bfa42ba03851f47af7a7309f962ef91d52c4b6f263866f0fed4901b56be375'
with zipfile.ZipFile(source) as a,zipfile.ZipFile(docx) as b:
 preserved=['word/styles.xml','word/numbering.xml','word/theme/theme1.xml']
 assert all(a.read(n)==b.read(n) for n in preserved)
 root=E.fromstring(b.read('word/document.xml'));ns={'w':root.nsmap['w']}
 text=' '.join(root.xpath('//w:t/text()',namespaces=ns))
 assert all(s in text for s in ['Wasi alsafa','Sadad Kadery','wasialsafa025@gmail.com','157.73','109.61'])
 assert not any(s in text for s in ['First A. Author','XXX-X-XXXX','Type Your Title'])
pdfdoc=PdfReader(pdf);assert len(pdfdoc.pages)==4
assert all(abs(float(p.mediabox.width)-595.3)<1 and abs(float(p.mediabox.height)-841.9)<1 for p in pdfdoc.pages)
pdftext='\n'.join(p.extract_text() for p in pdfdoc.pages)
assert all(s in pdftext for s in ['Wasi alsafa','Sadad Kadery','157.73','109.61','113565'])
assert '\ufffd' not in pdftext
audit={'template_sha256':sha(source),'template_unchanged':True,'preserved_parts_byte_identical':preserved,'pages':4,'page_size':'A4','body_columns':2,'visual_review':'All four render2 pages inspected; equation readable; no overlaps or clipping observed. Metadata-only DOCX update after render does not change document.xml.','authors_supplied':['Wasi alsafa','Sadad Kadery'],'missing_metadata':['Both affiliations','Sadad Kadery email'],'similarity_check':'Not performed; cannot certify <30%.','outputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [docx,pdf]]}
(T/'verification.json').write_text(json.dumps(audit,indent=2))
(T/'SUBMISSION_READINESS.md').write_text('''# Manuscript readiness

The editable manuscript and its rendered PDF use the supplied IEEE template and contain four A4 pages, including references, in double-column body format. All final pages were visually reviewed. The template source was preserved, including its styles, numbering and theme. Author details include only the names and first email actually supplied. Affiliations and Sadad Kadery's email remain to be added by the authors.

The title and conclusions describe the tested interaction without claiming a gating benefit. This is a controlled offline imitation study, not a reproduction of the full domain-adversarial RL paper. The primary static test is reported in the manuscript. Additional fresh-seed static/dynamic results are a separately labelled exploratory supplement; no test-based model selection occurred.

Before submission, all authors must review the scientific claims, approve authorship and the final text, confirm that the work is original and not simultaneously under review elsewhere, and add any required affiliations/contact details. Run the conference's approved similarity check: the requested overall score below 30% has not been measured or guaranteed. Proper citation does not guarantee a particular similarity score. Check venue anonymity, AI disclosure, copyright and final PDF-validation requirements against the actual submission portal. No paper was externally submitted.

This file records checks and remaining author responsibilities; it is not a certification of acceptance, novelty, plagiarism clearance or IEEE PDF compliance.
''',encoding='utf-8')
print(json.dumps(audit,indent=2))
