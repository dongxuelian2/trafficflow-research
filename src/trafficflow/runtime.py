"""Runtime evidence without additional monitoring dependencies."""
import hashlib
import importlib.metadata as md
import os
from pathlib import Path
import platform

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def capture_source_snapshot(path):
    """Freeze source/dependency files before a new long-running experiment."""
    import zipfile
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite a source snapshot')
    root=Path(__file__).resolve().parents[2]
    sources=set(root.glob('src/**/*.py'))|set(root.glob('scripts/*.py'))|set(root.glob('vendor/official/**/*.py'))
    sources.update(root.glob('config/*.json'))
    sources.update(p for p in [root/'pyproject.toml',root/'LICENSE',root/'requirements.lock.txt',root/'requirements.research.lock.txt'] if p.exists())
    path.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):
            name=source.relative_to(root).as_posix()
            content=source.read_bytes()
            archive.writestr(name,content)
            records.append({'path':name,'sha256':hashlib.sha256(content).hexdigest()})
    return {'archive':path.as_posix(),'archive_sha256':sha256_file(path),
            'captured_files':records,'n_files':len(records),
            'scope':'Own source, vendored official Python and dependency locks; no data, credentials or model weights'}

def evidence(gpu_used=False):
    import ctypes
    peak=None
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('page_faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
              ['peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            peak=counters.peak_working_set/1024**2
    source=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for path in sorted(source.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    return {'python':platform.python_version(),'platform':platform.platform(),'logical_cpus':os.cpu_count(),
            'process_peak_working_set_MiB':peak,'peak_scope':'current process lifetime, not isolated per model',
            'source_sha256':h.hexdigest(),'model_threads':4,'gpu_used':bool(gpu_used),
            'source_fingerprint_scope':'On-disk source at evidence() call; a captured start archive is primary when present',
            'versions':{p:md.version(p) for p in ['numpy','pandas','scipy','pyarrow','lightgbm','catboost','scikit-learn']}}
