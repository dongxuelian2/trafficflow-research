import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'vendor'/'official'
manifest=json.loads((root/'UPSTREAM.json').read_text())
bad=[]
for item in manifest['files']:
    data=(root/item['path']).read_bytes()
    digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if digest!=item['blob_sha']:
        bad.append(item['path'])
if bad:
    raise SystemExit('Upstream byte mismatch: '+str(bad))
print('Verified',len(manifest['files']),'upstream blobs at',manifest['commit'])
