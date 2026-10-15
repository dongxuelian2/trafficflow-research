"""Causal joint reconstruction/forecast teachers and strong-tree features."""
import numpy as np
from .dense_queue_data import prepare_dense_day,dense_window


def prepare_joint_day(raw,masked,links,net):
    pack=prepare_dense_day(raw,masked,links,net)
    regimes=masked.pivot(index='timestamp',columns='link_id',values='mask_regime').reindex(index=pack['timestamps'],columns=pack['links'])
    if regimes.isna().any().any() or not regimes.isin(['R1','R2','R3']).all().all():raise ValueError('Complete public mask regime metadata required')
    pack['mask_regime']=regimes.to_numpy(str);pack['v_cut_scaled']=net.v_cut.reindex(links).to_numpy(np.float32)/25.;return pack


def joint_window(pack,at,ongoing=None):
    a=dense_window(pack,at,ongoing);a.update(state_window(pack,at))
    a['truth_flow']=pack['raw_channels'][at+1:at+7,:,1].copy();return a


def state_window(pack,at):
    if not isinstance(at,(int,np.integer)) or at<12 or at>=len(pack['available']):raise ValueError('State context needs12maskedpast slots and within-dayT')
    observed=pack['masked_channels'][at,:,2:4].T>.5
    masked=pack['masked_channels'][at-12:at+1];state_condition=bool((((masked[:12,:,0]<=pack['v_cut_scaled'])&(masked[:12,:,2]>.5)).sum(axis=0)>=2).any())
    time=np.broadcast_to(pack['time_channels'][at-12:at+1,None,:],(13,len(pack['links']),2));state_x=np.concatenate([masked,time,np.full((13,len(pack['links']),1),state_condition,np.float32)],axis=-1)
    return {'truth_current':pack['raw_channels'][at,:,:2].T.copy(),'current_domain':pack['available'][at][None,:]&~observed,'current_regime':pack['mask_regime'][at].copy(),'current_available':pack['available'][at].copy(),'state_x':state_x}


def latest_state(x,geometry):
    """No teacher: carry latest eligible value; public-geometry fallback."""
    x=np.asarray(x,np.float32);batch,times,nodes,_=x.shape
    if times!=13 or x.shape[-1]!=9:raise ValueError('Expected causal13slot9channel array')
    steps=np.arange(13)[None,:,None];values=[];ages=[]
    for channel,mask,default in [(0,2,geometry[0]/25.),(1,3,geometry[1]/1200.)]:
        at=np.where(x[:,:,:,mask]>.5,steps,-1).max(axis=1);v=np.take_along_axis(x[:,:,:,channel],at.clip(min=0)[:,None,:],axis=1)[:,0]
        values.append(np.where(at>=0,v,default[None]));ages.append((12-at).astype(np.float32)/13.)
    return np.stack(values,axis=1).astype(np.float32),np.stack(ages,axis=1)


def reconstruction_features(x,graph):
    """Node rows, past/current only; same input contract as the joint network."""
    base,age=latest_state(x,graph['geometry']);rows=[x.transpose(0,2,1,3).reshape(len(x)*x.shape[2],-1)]
    rows.extend(z.transpose(0,2,1).reshape(len(x)*x.shape[2],-1) for z in [base,age])
    static=graph['geometry'].T/np.array([25.,600.,25.,25.],np.float32)
    rows.append(np.tile(static,(len(x),1)))
    for kernel in [graph['incoming'],graph['outgoing']]:
        z=np.einsum('bcn,mn->bcm',base,kernel);rows.append(z.transpose(0,2,1).reshape(len(x)*x.shape[2],-1))
    result=np.concatenate(rows,axis=1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Finite reconstruction features required')
    return result
