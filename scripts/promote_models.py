"""Freeze production artifacts, choosing queue versions on calibration only."""
import argparse,json,shutil
from pathlib import Path
from trafficflow.data import Release
from trafficflow.runtime import sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--campaign',type=Path,default=Path('artifacts/campaign_v1'))
    ap.add_argument('--queue-refinement',type=Path,default=Path('artifacts/queue_refinement'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/best_models'));args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh model directory')
    manifest={}
    for panel in Release(args.release_root).panels():
        original=args.campaign/panel;dest=args.output/panel;dest.mkdir(parents=True)
        d=json.loads((original/'experiment.json').read_text());sel=d['selection'];state=sel['state_artifact']
        for name in [state,'state_final.pkl']:shutil.copyfile(original/name,dest/name)
        if not panel.startswith('D12_I405'):
            root=args.queue_refinement/panel;ref=json.loads((root/'research.json').read_text());name=ref['selected']
            cal=ref['results'][name]['calibration']['proxy_IoU'];old=d['queue'][sel['queue']]['calibration']['proxy_IoU']
            use=cal>=old-1e-12
            if use and not ref['production_refit']:raise ValueError('Refined production model missing')
            source=root if use else original
            shutil.copyfile(source/('selected.pkl' if use else sel['queue_artifact']),dest/'queue_selected.pkl')
            shutil.copyfile(source/'queue_final.pkl',dest/'queue_final.pkl')
            d['queue_refinement']=ref;sel['queue_artifact']='queue_selected.pkl'
            if use:sel['queue']='lightgbm_refined'
        (dest/'experiment.json').write_text(json.dumps(d,indent=2))
        manifest[panel]={p.name:sha256_file(p) for p in dest.glob('*.pkl')}
    (args.output/'model_manifest.json').write_text(json.dumps(manifest,indent=2));print('Frozen',len(manifest),'panels')

if __name__=='__main__':main()
