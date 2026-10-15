"""Public-fit label diagnostics;never an inference-time feature builder."""
import numpy as np


def transition_diagnostics(truth,valid,incoming,outgoing):
    truth=np.asarray(truth,float);valid=np.asarray(valid,bool)
    if truth.ndim!=3 or truth.shape[-1]!=2 or truth.shape[:2]!=valid.shape or len(incoming)!=truth.shape[1] or len(outgoing)!=truth.shape[1]:
        raise ValueError('Expected time/link/(kmh,lane-flow) truth and eligibility')
    valid=valid&np.isfinite(truth).all(axis=-1)
    delta=np.full_like(truth,np.nan);pair=np.zeros_like(valid);pair[1:]=valid[1:]&valid[:-1]
    delta[1:]=np.where(pair[1:,:,None],truth[1:]-truth[:-1],np.nan)
    drop=delta[...,0]<=-20;recover=delta[...,0]>=20;flow=np.abs(delta[...,1])>=150
    future1=np.zeros_like(valid);future2=np.zeros_like(valid)
    future1[1:-1]=valid[:-2]&valid[1:-1]&valid[2:]
    future2[1:-2]=valid[:-3]&valid[1:-2]&valid[2:-1]&valid[3:]
    reversal_v=np.zeros_like(valid);reversal_q=np.zeros_like(valid);persist_v=np.zeros_like(valid);persist_q=np.zeros_like(valid)
    for c,threshold,return_width,persist_width,destination_reversal,destination_persist in [
        (0,20.,5.,10.,reversal_v,persist_v),(1,150.,50.,75.,reversal_q,persist_q)]:
        d=delta[...,c];large=np.abs(d)>=threshold
        destination_reversal[1:-1]=large[1:-1]&future1[1:-1]&(np.abs(truth[2:,:,c]-truth[:-2,:,c])<=return_width)
        sign=np.sign(d[1:-2]);first=sign*(truth[2:-1,:,c]-truth[:-3,:,c]);second=sign*(truth[3:,:,c]-truth[:-3,:,c])
        destination_persist[1:-2]=large[1:-2]&future2[1:-2]&(first>=persist_width)&(second>=persist_width)
    available=np.zeros_like(valid);supported=np.zeros_like(valid)
    for link in range(truth.shape[1]):
        ns=sorted(set(incoming[link]+outgoing[link]))
        if any(j<0 or j>=truth.shape[1] or j==link for j in ns):raise ValueError('Invalid measured topology neighbor')
        if not ns:continue
        for lag in range(-2,3):
            first=max(0,-lag);last=min(len(valid),len(valid)-lag)
            adjacent=delta[first+lag:last+lag,ns,0];current=delta[first:last,link,0]
            available[first:last,link]|=np.isfinite(adjacent).any(axis=1)
            supported[first:last,link]|=((current[:,None]*adjacent>0)&(np.abs(adjacent)>=10)).any(axis=1)
    return {'pair_valid':pair,'delta':delta,'drop20':drop,'recover20':recover,'flow150':flow,
        'speed_reversal5':reversal_v,'speed_persistent10':persist_v,'flow_reversal50':reversal_q,'flow_persistent75':persist_q,
        'future1_eligible':future1,'future2_eligible':future2,'neighbor_available':available,'neighbor_same_sign10':supported}
