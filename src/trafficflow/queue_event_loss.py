"""Fixed event-balanced losses for the official window/condition macro objective."""
import numpy as np
import pandas as pd
from .queue_calibration import RefinedQueueModel


def event_weights(data,class_balance=False):
    frame=pd.DataFrame({'window':data.window,'condition':data.condition,'y':data.y})
    if frame.groupby('window').condition.nunique().max()!=1:
        raise ValueError('An event cannot have more than one condition')
    event_count=frame[['window','condition']].drop_duplicates().groupby('condition').window.count()
    condition_windows=frame.condition.map(event_count).to_numpy(float)
    if class_balance:
        counts=frame.groupby(['window','y']).y.transform('size').to_numpy(float)
        classes=frame.groupby('window').y.transform('nunique').to_numpy(float)
        w=1/(condition_windows*classes*counts)
    else:
        counts=frame.groupby('window').y.transform('size').to_numpy(float)
        w=1/(condition_windows*counts)
    return w/w.mean()


class EventWeightedQueueModel(RefinedQueueModel):
    def __init__(self,class_balance=False,**kwargs):
        super().__init__(**kwargs)
        self.class_balance=class_balance

    def fit(self,data):
        from lightgbm import LGBMClassifier
        self.model=LGBMClassifier(n_estimators=self.iterations,learning_rate=.035,num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,reg_lambda=8.,random_state=self.seed,n_jobs=self.threads,
            verbosity=-1,deterministic=True,force_col_wise=True)
        self.model.fit(data.x,data.y,sample_weight=event_weights(data,self.class_balance))
        return self
