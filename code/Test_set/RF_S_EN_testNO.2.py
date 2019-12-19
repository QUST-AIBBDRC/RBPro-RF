import scipy.io as sio
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

data_train = sio.loadmat(r'D:\ctd\T_EN_2(T.mat')
data=data_train.get('T')
row=data.shape[0]
column=data.shape[1]
shu=data[:,np.array(range(1,column))]
X=shu  
y=data[:,0]
data_test= sio.loadmat(r'D:\ctd\T3_EN_2(T3.mat')
test_data=data_test.get('T3')
row1=test_data.shape[0]
column1=test_data.shape[1]
test_shu=test_data[:,np.array(range(1,column1))]
test_label=test_data[:,0]
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
cv_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10)                                 
y_train=utils.to_categorical(y)
hist=cv_clf.fit(shu,y)
y_score=cv_clf.predict_proba(test_shu)
y_test=utils.to_categorical(test_label)    
y_class= utils.categorical_probas_to_classes(y_score)
y_test_tmp=test_label
acc, precision,npv, sensitivity, specificity, mcc,f1= utils.calculate_performace(len(y_class), y_class, y_test_tmp)