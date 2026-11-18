from pathlib import Path
import zipfile
E=Path(__file__).resolve().parent
files=set()
for pat in ['*.csv','*.json','*.py','*.log','*.md','source/*/*.json','source/*/*.npz','demographics/*.csv','recovery/*','status/*.json']:
 files.update(p for p in E.glob(pat) if p.is_file())
with zipfile.ZipFile(E/'final_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,str(p.relative_to(E)))
print('package_files',len(files),'bytes',(E/'final_results.zip').stat().st_size)
