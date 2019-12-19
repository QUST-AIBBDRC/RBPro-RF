import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
def elasticNet(data,label,alpha =np.array([0.001, 0.002, 0.003,0.004, 0.005, 0.006, 0.007, 0.008,0.009, 0.01])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.1).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.1)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask
data_train = sio.loadmat(r'D:\ctd\T_PSSMCTDPSE(T.mat')
data=data_train.get('T')
row=data.shape[0]
column=data.shape[1]
shu=data[:,np.array(range(1,column))]
X=shu  
y=data[:,0] 
sme = SMOTEENN(random_state=42)
X_res, y_res = sme.fit_sample(X, y)
data_2,mask2=elasticNet(X_res,y_res)
data=data_2
X=data
y=y_res
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
cv_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10)                                 
skf= StratifiedKFold(n_splits=10)
for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])
    hist=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('RF:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('RFyscore_sum2.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('RFytest_sum2.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0]) 
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='RF ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('RFresult2.csv')
colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp','MCC','F1','AUC']
ind=['1', '2', '3','4','5','6','7','8','9','10','11']
data_csv = pd.DataFrame(columns=colum, data=result,index=ind)