"""Strict streaming merge; absent predictions never silently become zeros."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .data import STATE_KEY,QUEUE_KEY,OD_KEY,check_keys

KEYS={'state':STATE_KEY,'queue':QUEUE_KEY,'odme':OD_KEY}
VALUES={'state':['speed_kmh','flow_vph'],'queue':['queue_pred'],'odme':['path_flow']}
OUTPUT=['submission_id','task','speed_kmh','flow_vph','queue_pred','path_flow']

def validate_task(frame,task):
    check_keys(frame,KEYS[task],task)
    for c in VALUES[task]:
        values=pd.to_numeric(frame[c],errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError(f'{task}: non-finite or negative prediction')
        if task=='queue' and not np.isin(values,[0,1]).all():
            raise ValueError('Queue predictions must be binary')

def merge_files(files:dict[str,Path],key_path:Path,output:Path,chunk_size=100000):
    output.parent.mkdir(parents=True,exist_ok=True)
    db_path=output.with_suffix('.merge.sqlite')
    if db_path.exists():
        raise FileExistsError(f'Use a fresh output directory; intermediate database exists: {db_path}')
    conn=sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF')
    counts={}
    try:
        for task in KEYS:
            keys,values=KEYS[task],VALUES[task]
            fields=','.join(f'"{c}" '+('TEXT' if c in keys else 'REAL')+' NOT NULL' for c in keys+values)
            unique=','.join(f'"{c}"' for c in keys)
            conn.execute(f'CREATE TABLE {task} ({fields}, PRIMARY KEY ({unique})) WITHOUT ROWID')
            key_fields=','.join(f'"{c}" TEXT NOT NULL' for c in keys)
            conn.execute(f'CREATE TABLE key_seen_{task} ({key_fields}, PRIMARY KEY ({unique})) WITHOUT ROWID')
            counts[task]=0
            for frame in pd.read_csv(files[task],chunksize=chunk_size,dtype=str,keep_default_na=False):
                validate_task(frame,task)
                for c in values:
                    frame[c]=pd.to_numeric(frame[c])
                rows=list(frame[keys+values].itertuples(index=False,name=None))
                try:
                    conn.executemany(f'INSERT INTO {task} VALUES ({",".join("?" for _ in keys+values)})',rows)
                except sqlite3.IntegrityError as e:
                    raise ValueError(f'{task}: duplicate keys across input chunks') from e
                counts[task]+=len(rows)
            conn.commit()
        conn.execute('CREATE TABLE seen (submission_id TEXT PRIMARY KEY) WITHOUT ROWID')
        matched={k:0 for k in KEYS};written=0
        temp_output=output.with_suffix('.csv.partial')
        for chunk in pd.read_csv(key_path,chunksize=chunk_size,dtype=str,keep_default_na=False):
            if not chunk.task.isin(KEYS).all():
                raise ValueError('Unknown task in submission key')
            if not chunk.submission_id.str.match(r'^\d+$').all():
                raise ValueError('Invalid submission_id')
            try:
                conn.executemany('INSERT INTO seen VALUES (?)',[(x,) for x in chunk.submission_id])
            except sqlite3.IntegrityError as e:
                raise ValueError('Duplicate submission_id') from e
            out=chunk[['submission_id','task']].copy()
            for c in OUTPUT[2:]:
                out[c]=0.
            for task in KEYS:
                selected=chunk.task==task
                if not selected.any():
                    continue
                keys,values=KEYS[task],VALUES[task]
                want=chunk.loc[selected,keys].copy()
                check_keys(want,keys,'submission key '+task)
                try:
                    conn.executemany(f'INSERT INTO key_seen_{task} VALUES ({",".join("?" for _ in keys)})',
                                     list(want.itertuples(index=False,name=None)))
                except sqlite3.IntegrityError as e:
                    raise ValueError('Duplicate natural keys across master-key chunks') from e
                # Temporary wanted table enables one indexed join per chunk.
                conn.execute('DROP TABLE IF EXISTS wanted')
                want.insert(0,'ord',np.arange(len(want)))
                want.to_sql('wanted',conn,index=False)
                join=' AND '.join(f'p."{c}"=w."{c}"' for c in keys)
                cols=','.join('p."'+c+'"' for c in values)
                found=conn.execute(f'SELECT {cols} FROM wanted w LEFT JOIN {task} p ON {join} ORDER BY w.ord').fetchall()
                val=np.asarray(found,dtype=float)
                if val.shape!=(len(want),len(values)) or not np.isfinite(val).all():
                    raise ValueError(f'{task}: missing predictions for required master-key rows')
                out.loc[selected,values]=val
                matched[task]+=len(want)
            out[OUTPUT].to_csv(temp_output,index=False,mode='w' if written==0 else 'a',header=written==0)
            written+=len(out)
        if matched!=counts:
            raise ValueError(f'Extra task predictions or repeated natural keys: supplied={counts}, matched={matched}')
        if written==0:
            raise ValueError('Empty master submission key')
        temp_output.replace(output)
        return {'rows':written,'task_rows':matched,'complete':True}
    finally:
        conn.close()
        # This is a known temporary file in the output workspace, never a recursive delete.
        db_path.unlink(missing_ok=True)
