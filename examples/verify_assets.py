"""Verify the companion numeric asset archive against its SHA-256 manifest."""
from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'data/ASSET_MANIFEST.json').read_text(encoding='utf8'))
errors=[]
for item in manifest['files']:
    path=root/item['file']
    if not path.is_file():
        errors.append(item['file']+': missing')
        continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    if path.stat().st_size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
        errors.append(item['file']+': integrity mismatch')
if errors:
    raise SystemExit('\n'.join(errors))
print('Verified',len(manifest['files']),'asset files.')
