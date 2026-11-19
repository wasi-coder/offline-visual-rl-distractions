"""Make a separate anonymous review copy from the final author version."""
from pathlib import Path
from lxml import etree as E
import zipfile,re
T=Path(__file__).resolve().parents[2]/'experiments/hybrid_20261030/ieee_template'
with zipfile.ZipFile(T/'ECCE2027_Gating_Alignment_Manuscript.docx') as z:parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);ns={'w':root.nsmap['w']}
for table in root.findall('.//w:tbl',ns):
 text=' '.join(table.xpath('.//w:t/text()',namespaces=ns))
 if 'wasialsafa025' in text:table.getparent().remove(table)
for p in root.findall('.//w:p',ns):
 text=''.join(p.xpath('.//w:t/text()',namespaces=ns))
 if ('experiment code and small result records are available' in text or text.startswith('[10]')):
  p.getparent().remove(p);continue
 for t in p.findall('.//w:t',ns):
  if t.text:t.text=re.sub(r'\[(11|12)\]',lambda m:'['+str(int(m[1])-1)+']',t.text)
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
for name in ['docProps/core.xml','docProps/app.xml']:
 root=E.fromstring(parts[name])
 for node in root.iter():
  if E.QName(node).localname in ['creator','lastModifiedBy','Company','Manager','description','keywords']:node.text=''
 parts[name]=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
for name,data in parts.items():
 if name.endswith('.xml'):
  assert not any(v in data.lower() for v in [b'wasialsafa',b'sadadkadery',b'wasi alsafa',b'sadad kadery',b'wasi-coder',b'rajshahi']) ,name
out=T/'ECCE2027_Anonymous_Submission.docx'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,d in parts.items():z.writestr(n,d)
print(out)
