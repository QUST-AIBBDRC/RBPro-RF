import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
lw=1.2
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

plt.subplot(121)
ytest= pd.read_csv('KNNytest_sum.csv', index_col=0) 
ytest_PSSM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('KNNyscore_sum.csv', index_col=0)
yscore_PSSM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('KNNresult.csv',index_col=0)
auc_PSSM=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_PSSM[:,0], yscore_PSSM[:,0])
plt.plot(fpr, tpr, color='deepskyblue',
lw=lw, label='KNN(AUC=0.984)')
ytest= pd.read_csv('Lytest_sum.csv', index_col=0) 
ytest_PSSM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Lyscore_sum.csv', index_col=0)
yscore_PSSM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Lresult.csv',index_col=0)
auc_PSSM=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_PSSM[:,0], yscore_PSSM[:,0])
plt.plot(fpr, tpr, color='limegreen',
lw=lw, label='Logistic regression(AUC=0.985)')

ytest= pd.read_csv('NBytest_sum2.csv', index_col=0) 
ytest_PSSM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('NByscore_sum2.csv', index_col=0)
yscore_PSSM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('NBresult2.csv',index_col=0)
auc_PSSM=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_PSSM[:,0], yscore_PSSM[:,0])
plt.plot(fpr, tpr, color='blue',
lw=lw, label='Naive bayes(AUC=0.840)')

A = open('Decisiontree_Label.csv')
B = open('Decisiontree_Probability.csv')
label = pd.read_csv(A)
score_= pd.read_csv(B)
label = np.array(label)
score_= np.array(score_)
y_class = np.delete(label,[0,2],axis=1)
y_test_tmp = np.delete(label,[0,1],axis=1)
y_class = np.array(y_class)
y_test_tmp = np.array(y_test_tmp)
score= np.delete(score_,0,axis=1)
ytest = np.append(y_test_tmp,score,axis=1)
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
#plt.figure(figsize=(6,5)) 
#plt.title('Enzyme',fontsize=18)
plt.plot(fpr, tpr, color='darkorchid',
lw=lw, label='Decision tree(AUC=0.965)')

ytest= pd.read_csv('MLytest_sum.csv', index_col=0) 
ytest_T=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('MLyscore_sum.csv', index_col=0)
yscore_T=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('MLresult.csv',index_col=0)
auc_T=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_T[:,0], yscore_T[:,0])
plt.plot(fpr, tpr, color='lightpink',
lw=lw, label='MLP(AUC=0.994)')

ytest= pd.read_csv('SVMytest_sum14.csv', index_col=0) 
ytest_T=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('SVMyscore_sum14.csv', index_col=0)
yscore_T=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('SVMresult14.csv',index_col=0)
auc_T=np.array(auc_,dtype=np.float)
fpr4, tpr4, _ = roc_curve(ytest_T[:,0], yscore_T[:,0])
plt.plot(fpr4, tpr4, color='darkorange',
lw=lw, label='SVM(AUC=0.961)')

ytest= pd.read_csv('RFytest_sum.csv', index_col=0) 
ytest_T=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('RFyscore_sum.csv', index_col=0)
yscore_T=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('RFresult.csv',index_col=0)
auc_T=np.array(auc_,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_T[:,0], yscore_T[:,0])
plt.plot(fpr, tpr, color='red',
lw=lw, label='Random forest(AUC=0.995)')


plt.subplot(122)
ytest= pd.read_csv('KNNytest_sum.csv', index_col=0) 
ytest_KNN=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('KNNyscore_sum.csv', index_col=0)
yscore_KNN=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('KNNresult.csv',index_col=0)
auc_KNN=np.array(auc_,dtype=np.float)
auc_scoreKNN=auc_KNN[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_KNN[:,0], yscore_KNN[:,0]) 
aupr1=average_precision_score(ytest_KNN[:,0], yscore_KNN[:,0])
plt.plot(tpr, fpr, color='deepskyblue',
lw=1.2, label='KNN(AUPR=0.974)')

ytest= pd.read_csv('Lytest_sum.csv', index_col=0) 
ytest_KNN=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Lyscore_sum.csv', index_col=0)
yscore_KNN=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Lresult.csv',index_col=0)
auc_KNN=np.array(auc_,dtype=np.float)
auc_scoreKNN=auc_KNN[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_KNN[:,0], yscore_KNN[:,0]) 
aupr2=average_precision_score(ytest_KNN[:,0], yscore_KNN[:,0])
plt.plot(tpr, fpr, color='limegreen',
lw=1.2, label='Logistic regression(AUPR=0.966)')
#pdb.set_path

ytest= pd.read_csv('NBytest_sum2.csv', index_col=0) 
ytest_QD=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('NByscore_sum2.csv', index_col=0)
yscore_QD=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('NBresult2.csv',index_col=0)
auc_NB=np.array(auc_,dtype=np.float)
auc_score_NB=auc_NB[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_QD[:,0], yscore_QD[:,0])  
aupr3=average_precision_score(ytest_QD[:,0], yscore_QD[:,0])
plt.plot(tpr, fpr, color='blue',
lw=1.2, label='Naive bayes(AUPR=0.643)')

A = open('Decisiontree_Label.csv')
B = open('Decisiontree_Probability.csv')
label = pd.read_csv(A)
score_= pd.read_csv(B)
label = np.array(label)
score_= np.array(score_)
y_class = np.delete(label,[0,2],axis=1)
y_test_tmp = np.delete(label,[0,1],axis=1)
y_class = np.array(y_class)
y_test_tmp = np.array(y_test_tmp)

score= np.delete(score_,0,axis=1)
ytest = np.append(y_test_tmp,score,axis=1)
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = precision_recall_curve(ytest_GTB[:,0], ytest_GTB[:,2])
aupr2=average_precision_score(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
plt.plot(tpr, fpr, color='darkorchid',
lw=lw, label='Decision tree(AUPR=0.957)')

ytest= pd.read_csv('MLytest_sum.csv', index_col=0) 
ytest_RF=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('MLyscore_sum.csv', index_col=0)
yscore_RF=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('MLresult.csv',index_col=0)
auc_RF=np.array(auc_,dtype=np.float)
auc_score_RF=auc_RF[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_RF[:,0], yscore_RF[:,0])
aupr5=average_precision_score(ytest_RF[:,0], yscore_RF[:,0])
plt.plot(tpr, fpr, color='lightpink',
lw=1.2, label='MLP(AUPR=0.970)')

ytest= pd.read_csv('SVMytest_sum14.csv', index_col=0) 
ytest_SVM=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('SVMyscore_sum14.csv', index_col=0)
yscore_SVM=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('SVMresult14.csv',index_col=0)
auc_SVM=np.array(auc_,dtype=np.float)
auc_score_SVM=auc_SVM[5,7]
fpr2, tpr2, _ = precision_recall_curve(ytest_SVM[:,0], yscore_SVM[:,0])
aupr7=average_precision_score(ytest_SVM[:,0], yscore_SVM[:,0])
plt.plot(tpr2, fpr2, color='darkorange',
lw=lw, label='SVM(AUPR=0.912)')


ytest= pd.read_csv('RFytest_sum.csv', index_col=0) 
ytest_RF=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('RFyscore_sum.csv', index_col=0)
yscore_RF=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('RFresult.csv',index_col=0)
auc_RF=np.array(auc_,dtype=np.float)
auc_score_RF=auc_RF[5,7]
fpr, tpr, _ = precision_recall_curve(ytest_RF[:,0], yscore_RF[:,0])
aupr6=average_precision_score(ytest_RF[:,0], yscore_RF[:,0])
plt.plot(tpr, fpr, color='red',
lw=1.2, label='Random forest(AUPR=0.990)')



#


ax = plt.gca()
ax.spines['left'].set_linewidth(0.3)
ax.spines['right'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['top'].set_linewidth(0.3)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.03])
plt.ylim([0.24, 1.03])
plt.xlabel('Recall')
plt.ylabel('Precision')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 5.2}
legend = plt.legend(prop=font,loc="lower left")
#plt.legend(loc="lower right")
#plt.title('B')
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,hspace=0.2, wspace=0.3)
fig_size=matplotlib.pyplot.gcf()
fig_size.set_size_inches(7,3.15)
plt.show





ax = plt.gca()
ax.spines['left'].set_linewidth(0.3)
ax.spines['right'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['top'].set_linewidth(0.3)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.03, 1.03])
plt.ylim([-0.03, 1.03])
plt.xlabel('False positive rate')
plt.ylabel('Ture positive rate')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 5.2}
legend = plt.legend(prop=font,loc="lower right=0.75")
#plt.legend(loc="lower right")
#plt.title('B')
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,hspace=0.2, wspace=0.3)
fig_size=matplotlib.pyplot.gcf()
fig_size.set_size_inches(7,3.15)
plt.show
plt.savefig(r'Curve_pr_auc.svg',format='svg',dpi=2000)


