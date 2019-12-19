import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
import itertools
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
lw=1
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.subplot(122)
ytest= pd.read_csv('PSSMytest_sum.csv', index_col=0) 
ytest_PSSM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('PSSMyscore_sum.csv', index_col=0)
yscore_PSSM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('PSSMresult.csv',index_col=0)
auc_PSSM=np.array(auc_,dtype=np.float)
auc_score_PSSM=auc_PSSM[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_PSSM[:,0], yscore_PSSM[:,0]) 
aupr1=average_precision_score(ytest_PSSM[:,0], yscore_PSSM[:,0])
plt.plot(tpr, fpr, color='darkorange',
lw=1.2, label='PSSM-400(AUPR=0.982)')
ytest= pd.read_csv('CTDytest_sum.csv', index_col=0) 
ytest_CTD=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('CTDyscore_sum.csv', index_col=0)
yscore_CTD=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('CTDresult.csv',index_col=0)
auc_CTD=np.array(auc_,dtype=np.float)
auc_score_CTD=auc_CTD[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_CTD[:,0], yscore_CTD[:,0])  
aupr2=average_precision_score(ytest_CTD[:,0], yscore_CTD[:,0])
plt.plot(tpr, fpr, color='blue',
lw=1.2, label='PP(AUPR=0.969)')
ytest= pd.read_csv('PSEytest_sum.csv', index_col=0) 
ytest_PSE=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('PSEyscore_sum.csv', index_col=0)
yscore_PSE=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('PSEresult.csv',index_col=0)
auc_PSE=np.array(auc_,dtype=np.float)
auc_score_PSE=auc_PSE[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_PSE[:,0], yscore_PSE[:,0])
aupr3=average_precision_score(ytest_PSE[:,0], yscore_PSE[:,0])
plt.plot(tpr, fpr, color='limegreen',
lw=1.2, label='PseAAC(AUPR=0.989)')
ytest= pd.read_csv('Tytest_sum.csv', index_col=0) 
ytest_T=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Tyscore_sum.csv', index_col=0)
yscore_T=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Tresult.csv',index_col=0)
auc_T=np.array(auc_,dtype=np.float)
auc_score_T=auc_T[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_T[:,0], yscore_T[:,0]) 
aupr4=average_precision_score(ytest_T[:,0], yscore_T[:,0])
plt.plot(tpr, fpr, color='red',
lw=1.2, label='PSSM-PP-PseAAC(AUPR=0.990)')
ax = plt.gca()
ax.spines['left'].set_linewidth(0.3)
ax.spines['right'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['top'].set_linewidth(0.3)
plt.xlim([-0.03, 1.03])
plt.ylim([0.22, 1.03])
plt.xlabel('Recall')
plt.ylabel('Precision')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 6}
legend = plt.legend(prop=font,loc="lower left")
plt.subplots_adjust(left=0.125, right=1.5, bottom=0.1, top=0.9,hspace=0.2, wspace=0.3)
fig_size=matplotlib.pyplot.gcf()
fig_size.set_size_inches(7,3.15)
plt.show
plt.subplot(121)
ytest= pd.read_csv('PSSMytest_sum.csv', index_col=0) 
ytest_PSSM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('PSSMyscore_sum.csv', index_col=0)
yscore_PSSM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('PSSMresult.csv',index_col=0)
auc_PSSM=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_PSSM[:,0], yscore_PSSM[:,0])
plt.plot(fpr, tpr, color='darkorange',
lw=1.2, label='PSSM-400(AUC=0.989)')
ytest= pd.read_csv('CTDytest_sum.csv', index_col=0) 
ytest_CTD=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('CTDyscore_sum.csv', index_col=0)
yscore_CTD=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('CTDresult.csv',index_col=0)
auc_CTD=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_CTD[:,0], yscore_CTD[:,0])
plt.plot(fpr, tpr, color='blue',
lw=1.2, label='PP(AUC=0.985)')
ytest= pd.read_csv('PSEytest_sum.csv', index_col=0) 
ytest_PSE=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('PSEyscore_sum.csv', index_col=0)
yscore_PSE=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('PSEresult.csv',index_col=0)
scores= pd.read_csv('PSEscores.csv', index_col=0) 
scores_PSE=np.array(ytest,dtype=np.float)
auc_PSE=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_PSE[:,0], yscore_PSE[:,0])
plt.plot(fpr, tpr, color='limegreen',
lw=1.2, label='PseAAC(AUC=0.991)')
ytest= pd.read_csv('Tytest_sum.csv', index_col=0) 
ytest_T=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Tyscore_sum.csv', index_col=0)
yscore_T=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Tresult.csv',index_col=0)
auc_T=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_T[:,0], yscore_T[:,0])
plt.plot(fpr, tpr, color='red',
lw=1.2, label='PSSM-PP-PseAAC(AUC=0.995)')
ax = plt.gca()
ax.spines['left'].set_linewidth(0.3)
ax.spines['right'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['top'].set_linewidth(0.3)
plt.xlim([-0.03, 1.03])
plt.ylim([-0.03, 1.03])
plt.xlabel('False positive rate')
plt.ylabel('Ture positive rate')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 6}
legend = plt.legend(prop=font,loc="lower right")
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,hspace=0.2, wspace=0.3)
fig_size=matplotlib.pyplot.gcf()
fig_size.set_size_inches(7,3.15)
plt.show
plt.savefig(r'D:\SUN\tezheng_pr_AUC.svg',format='svg',dpi=2000)
