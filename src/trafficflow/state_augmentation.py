"""Rebuild every State feature after removing public measurements.

Teacher arrays are deliberately absent from this API. Cached targets retain
their original supervision masks in the training experiment.
"""
import numpy as np
from .state import StateFeatures
from .outage_stress import remove_measurements


def rebuild_after_outage(frame,selected,network,topology,profile,version=3):
    """Discard old derived estimates; reconstruct from the modified frame."""
    return StateFeatures(remove_measurements(frame,selected),network,topology,profile,version=version)


def draw_training_plan(seed,regimes,steps,batch_size=4,probability=.5,n_views=4):
    """Independent RNG streams preserve the original uniform patch sampler.

    Both control and augmentation consume the same plan. Whether a view is
    actually used never changes any subsequent day/center/label exposure.
    Returns regime index,day index,supervised center,augmentation view or -1.
    Regimes contain ordered pairs (date,number of time steps),not labels.
    """
    if not 0<=probability<=1 or steps<1 or batch_size<1 or n_views<1 or len(regimes)!=3 or any(not group for group in regimes):
        raise ValueError('Invalid State training plan')
    patch_rng=np.random.default_rng(seed)
    augmentation_rng=np.random.default_rng(np.random.SeedSequence([seed,20261109]))
    result=np.empty((steps,batch_size,4),np.int32)
    for step in range(steps):
        for sample in range(batch_size):
            group=int(patch_rng.integers(3));index=int(patch_rng.integers(len(regimes[group])))
            center=int(patch_rng.integers(regimes[group][index][1]))
            use=augmentation_rng.random()<probability
            view=int(augmentation_rng.integers(n_views)) if use else -1
            result[step,sample]=(group,index,center,view)
    return result
