"""Extract one public panel early using ordinary HTTP ranges of the official ZIP.

Completed ranges of the ongoing download are read locally; uncovered bytes come
only from its authorized official download URL. ZipFile checks entry CRCs.
"""
from collections import OrderedDict
import argparse
import io
import json
from pathlib import Path
import zipfile
import requests
from download_data import authenticated_session,ENDPOINT,COMPETITION

class OfficialZipReader(io.RawIOBase):
    def __init__(self,archive):
        r=authenticated_session().get(ENDPOINT,stream=True,timeout=(30,60))
        if r.status_code!=200:
            raise RuntimeError(f'Official archive HTTP {r.status_code}')
        self.url=r.url;self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag','');r.close()
        self.archive=archive;self.pos=0;self.block_size=8*1024**2;self.cache=OrderedDict()
        ledger=json.loads(archive.with_suffix('.parts.json').read_text())
        if ledger['total_bytes']!=self.size or ledger['etag']!=self.etag:
            raise ValueError('Public archive differs from ongoing download')
        self.completed=set(ledger['completed']);self.chunk_size=ledger['chunk_bytes']
        self.remote_bytes=0;self.local_bytes=0

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=(offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset))
        if self.pos<0:raise ValueError('Negative ZIP seek')
        return self.pos

    def block(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        start=index*self.block_size;end=min(start+self.block_size,self.size)-1
        covered=all(i in self.completed for i in range(start//self.chunk_size,end//self.chunk_size+1))
        if covered:
            with self.archive.open('rb') as f:
                f.seek(start);body=f.read(end-start+1)
            self.local_bytes+=len(body)
        else:
            for attempt in range(4):
                try:
                    with requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,90)) as r:
                        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                            raise RuntimeError('Invalid official ZIP range response')
                        body=r.content
                    if len(body)!=end-start+1:raise RuntimeError('Short official ZIP range')
                    break
                except (requests.RequestException,RuntimeError):
                    if attempt==3:raise RuntimeError('Public ZIP range transfer failed') from None
            self.remote_bytes+=len(body)
        self.cache[index]=body
        if len(self.cache)>4:self.cache.popitem(last=False)
        return body

    def read(self,size=-1):
        stop=min(self.size,self.pos+size) if size>=0 else self.size
        if stop<=self.pos:return b''
        chunks=[]
        while self.pos<stop:
            index=self.pos//self.block_size;data=self.block(index)
            within=self.pos%self.block_size;count=min(len(data)-within,stop-self.pos)
            chunks.append(data[within:within+count]);self.pos+=count
        return b''.join(chunks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',default='D12_I5_N')
    ap.add_argument('--output',type=Path,default=Path('data/early'))
    ap.add_argument('--archive',type=Path,default=Path('data')/(COMPETITION+'.zip'))
    args=ap.parse_args();reader=OfficialZipReader(args.archive)
    with zipfile.ZipFile(reader) as z:
        prefixes=[f'kaggle_public/corridors/{args.panel}/',f'kaggle_public/task1/{args.panel}/',
                  f'kaggle_public/task2/{args.panel}/',f'kaggle_public/task4/{args.panel}/','kaggle_public/config/']
        members=[m for m in z.infolist() if not m.is_dir() and (any(m.filename.startswith(p) for p in prefixes) or m.filename=='kaggle_public/README.md')]
        # Public package path inventory supplies exact expected byte counts.
        expected={f['name']:f['bytes'] for f in json.loads(Path('data/official_file_list.json').read_text())['files']}
        count=0
        for m in members:
            dst=(args.output/m.filename).resolve()
            if not dst.is_relative_to(args.output.resolve()):raise ValueError('ZIP entry escapes workspace')
            if m.filename not in expected or expected[m.filename]!=m.file_size:raise ValueError('ZIP/public inventory mismatch')
            if dst.exists() and dst.stat().st_size==m.file_size:continue
            dst.parent.mkdir(parents=True,exist_ok=True)
            with z.open(m) as src,dst.open('wb') as out:
                while block:=src.read(4*1024**2):out.write(block)
            count+=1
            if count%50==0:print(f'{args.panel}: extracted {count} assets; reused {reader.local_bytes/1024**3:.2f} GiB, fetched {reader.remote_bytes/1024**3:.2f} GiB',flush=True)
    report={'source':'authorized ranges of current official ZIP','etag':reader.etag,'panel':args.panel,
            'entries':len(members),'local_bytes_reused':reader.local_bytes,'remote_bytes':reader.remote_bytes,'CRC':'ZipFile verified during extraction'}
    (args.output/f'{args.panel}_staging.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
