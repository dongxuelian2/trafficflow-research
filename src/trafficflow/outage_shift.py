"""New public-data outage geometries;never reads measured traffic values."""
import hashlib
import zlib
import numpy as np
import pandas as pd


SHIFT_CASES=[
    {'name':'block_2h_15pct','steps':24,'fraction':.15,'placement':'uniform'},
    {'name':'block_4h_50pct','steps':48,'fraction':.5,'placement':'uniform'},
    {'name':'block_8h_90pct','steps':96,'fraction':.9,'placement':'uniform'},
    {'name':'day_start_90min_50pct','steps':18,'fraction':.5,'placement':'start'},
    {'name':'day_end_4h_50pct','steps':48,'fraction':.5,'placement':'end'},
    {'name':'whole_corridor_4h','steps':48,'fraction':1.,'placement':'uniform'},
    {'name':'scattered_30pct','steps':288,'fraction':.3,'placement':'scattered'},
    {'name':'cross_midnight_4h_70pct','steps':48,'fraction':.7,'placement':'cross_midnight'},
]


def shifted_outage_masks(frames,panel,date,seed,case):
    if case not in SHIFT_CASES:raise ValueError('Registered shift case required')
    cross=case['placement']=='cross_midnight'
    if len(frames)!=(2 if cross else 1):raise ValueError('One day per case,two consecutive days for midnight case')
    grids=[];links=None
    for frame in frames:
        times=sorted(frame.timestamp.unique());dt=pd.to_datetime(times,utc=True)
        if len(times)!=288 or not (np.diff(dt.as_unit('s').asi8)==300).all() or dt[0].hour!=0 or dt[0].minute!=0:
            raise ValueError('Complete five-minute UTC days required')
        pos=frame.groupby('link_id').milepost.median()
        if not np.isfinite(pos.to_numpy(float)).all():raise ValueError('Public mileposts required')
        order=sorted(pos.index,key=lambda link:(pos[link],str(link)))
        if links is not None and order!=links:raise ValueError('Consecutive days require the same public sensor grid')
        links=order;grids.append(times)
    if str(grids[0][0])[:10]!=date or cross and pd.Timestamp(grids[1][0])-pd.Timestamp(grids[0][0])!=pd.Timedelta(days=1):
        raise ValueError('Date or consecutive midnight day differs')
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(date.replace('-','')),zlib.crc32(panel.encode()),zlib.crc32(case['name'].encode())]))
    width=int(np.ceil(case['fraction']*len(links)));center=int(rng.integers(len(links)))
    first=min(max(center-width//2,0),len(links)-width);chosen=links[first:first+width]
    starts=[];lengths=[]
    if cross:starts=[276,0];lengths=[12,case['steps']-12]
    elif case['placement']=='start':starts=[0];lengths=[case['steps']]
    elif case['placement']=='end':starts=[288-case['steps']];lengths=[case['steps']]
    elif case['placement']=='uniform':starts=[int(rng.integers(289-case['steps']))];lengths=[case['steps']]
    else:starts=[0];lengths=[288]
    masks=[];segments=[]
    for frame,times,start,length in zip(frames,grids,starts,lengths):
        if case['placement']=='scattered':
            grid=rng.random((288,len(links)))<case['fraction'];time_map={t:i for i,t in enumerate(times)};link_map={link:i for i,link in enumerate(links)}
            mask=grid[frame.timestamp.map(time_map).to_numpy(int),frame.link_id.map(link_map).to_numpy(int)]
        else:mask=(frame.timestamp.isin(times[start:start+length])&frame.link_id.isin(chosen)).to_numpy()
        masks.append(mask);segments.append({'date':str(times[0])[:10],'start_index':start,'steps':length,
            'first_timestamp':times[start],'last_timestamp':times[start+length-1],
            'selected_links':list(map(str,chosen)) if case['placement']!='scattered' else list(map(str,links)),
            'selected_frame_rows':int(mask.sum()),'row_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
    return masks,{'case':case,'segments':segments,'source':'Public timestamp/link/milepost metadata and seeded RNG only;never speed/flow/quality/teacher'}
