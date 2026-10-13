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
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.pt','.pkl','.json'):
            records.append({'path':path.as_posix(),'bytes':path.stat().st_size,
                            'sha256':sha256_file(path)})
    result={'purpose':'Local artifact integrity and reproducibility checkpoint',
            'formal_best_ref':57138874,'formal_best_total':0.82940,
            'formal_best_submission_sha256':'43f25e68ce91c975c2a926479a2b592ef29663dc9d8dfa34c5492948fd45bae1',
            'artifacts':records,'n_artifacts':len(records),
            'dependency_locks':{p:sha256_file(p) for p in
                ['requirements.lock.txt','requirements.research.lock.txt']},
            'limitations':'Models absent from this checkpoint may still be running. No cache arrays or predictions are redistributed.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(len(records),'artifact checksums recorded')


if __name__=='__main__':main()
