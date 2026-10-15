import numpy as np
from trafficflow.joint_queue_data import latest_state,reconstruction_features,joint_window
from test_dense_queue_data import example


def test_joint_teachers_units_masks_and_feature_isolation():
    p=example();p['mask_regime']=np.full(p['available'].shape,'R2');p['v_cut_scaled']=np.array([2.4,2.4],np.float32);a=joint_window(p,30)
    assert a['truth_current'].shape==(2,2);np.testing.assert_array_equal(a['truth_flow'],[[1.,.5]]*6)
    np.testing.assert_array_equal(a['current_domain'],[[True,False],[True,False]])
    np.testing.assert_array_equal(a['current_regime'],['R2','R2'])
    assert not a['state_x'][:,0,2].any() and a['x'][:12,0,2].all()
    g={'geometry':np.array([[100.,100.],[2000.,2000.],[150.,150.],[60.,60.]],np.float32),'incoming':np.array([[0,1],[1,0]],np.float32),'outgoing':np.zeros((2,2),np.float32)}
    x=a['x'][None];base,age=latest_state(x,g['geometry']);np.testing.assert_array_equal(base[0],[[3.,3.],[1.,.5]])
    assert age[0,0,0]==np.float32(1/13)
    before=reconstruction_features(x,g);p['truth_queue'][31:]=1;p['raw_channels'][30:]=999;p['available'][31:]=False
    np.testing.assert_array_equal(reconstruction_features(joint_window(p,30)['x'][None],g),before)
    original=a['state_x'].copy();p['raw_channels'][:30]=999;p['truth_queue'][:30]=1
    np.testing.assert_array_equal(joint_window(p,30)['state_x'],original)


def test_causal_fallback_and_node_permutation():
    x=np.zeros((1,13,3,9),np.float32);geometry=np.array([[100.,80.,90.],[2400.,1800.,1200.],[150.]*3,[60.]*3],np.float32);k=np.array([[0,1,0],[0,0,1],[1,0,0]],np.float32);g={'geometry':geometry,'incoming':k,'outgoing':k.T}
    base,age=latest_state(x,geometry);np.testing.assert_allclose(base[0],np.stack([geometry[0]/25,geometry[1]/1200]));np.testing.assert_array_equal(age,1.)
    order=[2,0,1];other={key:(v[:,order] if key=='geometry' else v[np.ix_(order,order)]) for key,v in g.items()}
    np.testing.assert_array_equal(reconstruction_features(x[:,:,order],other),reconstruction_features(x,g)[order])
