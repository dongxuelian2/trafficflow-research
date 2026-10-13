"""Record local research model/config checksums without redistributing weights."""
import argparse
import json
from pathlib import Path
from trafficflow.runtime import sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new checkpoint manifest name')
    root=Path('artifacts/research')
    records=[]
    skipped=[]
    def completed(path):
        directory=path.parent
        while directory!=root:
            if (directory/'experiment.json').exists() or (directory/'prepared.json').exists() or (Path('reports/research')/(directory.name+'.json')).exists():
                return True
            directory=directory.parent
        return False
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.pt','.pkl','.json'):
            if not completed(path):
                skipped.append(path.as_posix())
                continue
            before=path.stat()
            digest=sha256_file(path)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                skipped.append(path.as_posix())
                continue
            records.append({'path':path.as_posix(),'bytes':path.stat().st_size,
                            'sha256':digest})
    result={'purpose':'Local artifact integrity and reproducibility checkpoint',
            'formal_best_ref':57138874,'formal_best_total':0.82940,
            'formal_best_submission_sha256':'43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1',
            'artifacts':records,'n_artifacts':len(records),
            'skipped_unfinished_or_changing':skipped,
            'dependency_locks':{p:sha256_file(p) for p in
                ['requirements.lock.txt','requirements.research.lock.txt']},
            'limitations':'Only folders with completion metadata are included; changing files are skipped. No cache arrays or predictions are redistributed. Older checkpoints did not exclude unfinished folders.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(len(records),'artifact checksums recorded')


if __name__=='__main__':main()
