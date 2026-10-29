from pathlib import Path
import importlib.metadata as md,json,platform,sys
root=Path(__file__).resolve().parents[2]
packages={d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']}
record={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':dict(sorted(packages.items(),key=lambda x:x[0].lower())),'scope':'Installed software snapshot; package versions alone do not guarantee bitwise reproducibility.'}
(root/'reports/environment_snapshot.json').write_text(json.dumps(record,indent=2))
(root/'reports/requirements_snapshot.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n')
print('Recorded',len(packages),'installed packages')
