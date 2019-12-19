import scipy.io as sio
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN,SMOTETomek
import utils.tools as utils

def elasticNet(data,label,alpha =np.array([0.001, 0.002, 0.003,0.004, 0.005, 0.006, 0.007, 0.008,0.009, 0.01])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.1).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.1)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask
data_train = sio.loadmat(r'D:\ctd\T_CTDPSE(PC.mat')
data=data_train.get('PC')
row=data.shape[0]
column=data.shape[1]
shu=data[:,np.array(range(1,column))]
X=shu  
y=data[:,0] 
sme = SMOTEENN(random_state=42)
X_res, y_res = sme.fit_sample(X, y)
data_2,mask2=elasticNet(X_res,y_res)
data=data_2
data_test= sio.loadmat(r'D:\ctd\T1_PSSMCTDPSE(T1.mat')
test_data=data_test.get('T1')
row1=test_data.shape[0]
column1=test_data.shape[1]
test_shu=test_data[:,np.array(range(1,column1))]

test_label=test_data[:,0]
test_X=test_shu 
test_y=test_data[:,0]
sme = SMOTEENN(random_state=42)
X_ris, y_ris = sme.fit_sample(test_X, test_y)