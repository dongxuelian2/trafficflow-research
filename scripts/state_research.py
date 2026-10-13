"""Calibration-only selection of flux features and latent-density coupling."""
import argparse,json,time
from pathlib import Path
from trafficflow.data import Release
from trafficflow.experiments import groups,raw_for,training_rows,evaluate_state
from trafficflow.state import Profile,StateModel
from trafficflow.runtime import evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',default='D12_I5_N')
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/state_physics'))
    ap.add_argument('--iterations',type=int,default=400)
    ap.add_argument('--max-rows',type=int,default=300000)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    r=Release(args.release_root);parts,dates=groups(r,args.panel)
    profile=Profile.fit(raw_for(r,args.panel,parts['profile']));results={};models={}
    def checkpoint():
        (args.output/'research.json').write_text(json.dumps({'panel':args.panel,'dates':dates,'results':results,
          'formal_scores':None,'iterations':args.iterations,'max_rows':args.max_rows},indent=2))
    for version,method in [(1,'lightgbm'),(2,'joint_lightgbm')]:
        started=time.perf_counter();x,y,base=training_rows(r,args.panel,parts['fit'],profile,args.max_rows,2026,version)
        model=StateModel(profile,method,iterations=args.iterations,feature_version=version,num_leaves=63).fit(x,y,base)
        del x,y,base
        blends=[0.] if version==1 else [0.,.5,1.]
        for blend in blends:
            model.density_blend=blend
            name=f'v{version}_{method}_density_{blend}'
            cal=evaluate_state(r,args.panel,parts['calibration'],model)
            models[name]=name+'.pkl';model.save(args.output/models[name])
            results[name]={'calibration':cal,'seconds_since_version_start':time.perf_counter()-started,
                          'density_blend':blend,'feature_version':version,'method':method}
            checkpoint();print(name,'calibration State',cal['S_state'],flush=True)
    selected=max(results,key=lambda name:results[name]['calibration']['S_state'])
    model=StateModel.load(args.output/models[selected])
    results[selected]['test']=evaluate_state(r,args.panel,parts['test'],model)
    # Freeze the selected model separately for production refitting/review.
    model.save(args.output/'selected.pkl')
    out={'panel':args.panel,'dates':dates,'selected':selected,'results':results,'runtime':evidence(),
         'iterations':args.iterations,'max_rows':args.max_rows,'formal_scores':None,
         'selection_protocol':'calibration chose version/blend; test evaluated only after selection'}
    (args.output/'research.json').write_text(json.dumps(out,indent=2))
    print('selected',selected,'test State',results[selected]['test']['S_state'],flush=True)

if __name__=='__main__':main()
