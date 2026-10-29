import os
from pathlib import Path
from zipfile import ZipFile
import shutil,json
root=Path(__file__).resolve().parents[2]; videos={'blackswan','bmx-trees','breakdance','camel'}; records=[]
for archive in sorted(Path(os.environ.get('OFFLINE_RL_ARCHIVE_DIR',str(root/'data/archives'))).glob('*.zip')):
 with ZipFile(archive) as z:
  for n in z.namelist():
   p=n.split('/')
   if len(p)==6 and p[:4]==['dataset','DAVIS','JPEGImages','480p'] and p[4] in videos and p[5].endswith('.jpg'):
    target=root/'data/heldout_backgrounds'/p[4]/p[5];target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(n) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
    records.append(n)
(root/'reports/heldout_background_manifest.json').write_text(json.dumps({'videos':sorted(videos),'files':records,'note':'DAVIS validation sources; verify source training video list before generalization claim'},indent=2))
print('Extracted',len(records),flush=True)
