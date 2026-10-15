"""Metadata-only outage geometry and complete measurement removal."""
import numpy as np
import pandas as pd


MEASUREMENTS=['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm','pct_observed']


def outage_geometry(frame,panel,date,seed,duration_steps,spatial_fraction):
    if duration_steps not in [36,72] or spatial_fraction not in [.3,.7]:raise ValueError('Registered outage sizes required')
    times=sorted(frame.timestamp.unique())
    if len(times)!=288:raise ValueError('Complete released five-minute day required')
    dt=pd.to_datetime(times,utc=True)
    if not (np.diff(dt.as_unit('s').asi8)==300).all():raise ValueError('Nonconsecutive day')
    positions=frame.groupby('link_id').milepost.median()
    if not np.isfinite(positions.to_numpy(float)).all():raise ValueError('Published sensor mileposts required')
    links=sorted(positions.index,key=lambda link:(positions[link],str(link)))
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(date.replace('-','')),sum(map(ord,panel))]))
    center_time=int(rng.integers(108,181));center_link=int(rng.integers(len(links)))
    width=int(np.ceil(spatial_fraction*len(links)));first_link=min(max(center_link-width//2,0),len(links)-width)
    start=center_time-duration_steps//2;end=start+duration_steps
    chosen=links[first_link:first_link+width]
    selected=frame.timestamp.isin(times[start:end]).to_numpy()&frame.link_id.isin(chosen).to_numpy()
    return selected,{'start_timestamp':times[start],'last_removed_timestamp':times[end-1],
        'center_time_index':center_time,'center_link_index_in_milepost_order':center_link,
        'selected_links':list(map(str,chosen)),'duration_steps':duration_steps,'spatial_fraction':spatial_fraction,
        'n_removed_frame_rows':int(selected.sum()),'geometry_source':'Published timestamps/link IDs/mileposts and RNG only'}


def remove_measurements(frame,selected):
    selected=np.asarray(selected,bool)
    if selected.shape!=(len(frame),):raise ValueError('Outage row mask shape differs')
    result=frame.copy(deep=True)
    for column in MEASUREMENTS:
        if column in result:result.loc[selected,column]=np.nan
    for column,value in [('is_observed',False),('is_imputed',False),('is_score_eligible',False),('is_missing',True)]:
        if column in result:
            replacement=int(value) if pd.api.types.is_numeric_dtype(result[column].dtype) and not pd.api.types.is_bool_dtype(result[column].dtype) else value
            result.loc[selected,column]=replacement
    return result
