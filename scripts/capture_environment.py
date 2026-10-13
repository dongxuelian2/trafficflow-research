"""Capture numerical dependency closure and reproducibility metadata."""
import importlib.metadata as md
import json
import platform
from pathlib import Path
import subprocess
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
DIRECT=['numpy','pandas','scipy','pyarrow','lightgbm','catboost','scikit-learn','requests','pytest','setuptools','kaggle']
seen={};pending=DIRECT.copy()
while pending:
    name=pending.pop().lower().replace('_','-')
    if name in seen:
        continue
    seen[name]=md.version(name)
    for text in md.requires(name) or []:
        requirement=Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate({'extra':''}):
            pending.append(requirement.name)
header='# Numerical/runtime dependency closure captured from the verified environment.\n# Python '+platform.python_version()+'; install in a clean venv.\n'
(ROOT/'requirements.lock.txt').write_text(header+'\n'.join(f'{k}=={v}'+('; sys_platform == "win32"' if k=='pywin32' else '') for k,v in sorted(seen.items()))+'\n')
info={'python':platform.python_version(),'platform':platform.platform(),'dependencies':seen,
      'development_environment':'Windows project venv with inherited preinstalled numerical packages; clean-lock reproduction pending',
      'threads_per_model':4,'gpu_used':False}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports'/'environment.json').write_text(json.dumps(info,indent=2))
print('Pinned',len(seen),'dependencies')
