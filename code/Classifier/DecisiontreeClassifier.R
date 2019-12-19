##Import Data
setwd("D:/SUN/Decision tree curve modification")
xx=read.csv("jiangwei_x.csv",head=F) 
yy=read.csv("jiangwei_y.csv",head=F)
newdata<-data.frame(yy,xx)
newData$yy<-factor(newData$yy)
##Cross-validation
library(caret)
folds <- createFolds(newData$yy,k=10)
#####Decision tree classifier
library(C50)
sum<-0
sum1<-0
sum2<-0
sum3<-0
A<-c(99,99)
B<-c(99,99)
for(i in 1:10)
{fold_test <- newData[folds[[i]],]   #Take folds [[i]] as the test set  
fold_train <- newData[-folds[[i]],]   # The remaining data is used as the training set
print(i)#i represents the group number
m<-C5.0(fold_train[,-1],fold_train$yy,trials=1,costs=NULL)
p<-predict(m,fold_test[,-1],type="response")
p1<-predict(m,fold_test[,-1],type="prob")
A<-rbind(A,p1)
duibi<-data.frame(prob=p,obs=fold_test$yy)
B<-rbind(B,duibi)
library(caret)
jieguo<-confusionMatrix(duibi$prob,duibi$obs,positive = "1")
print(jieguo)
sum<-jieguo$overall[1]+sum
average<-sum/10
sum1<-jieguo$byClass[1]+sum1
average1<-sum1/10
sum2<-jieguo$byClass[2]+sum2
average2<-sum2/10
##Draw roc curve
p11=data.frame(p1)
roc_results=data.frame(fold_test$yy,p11)
library(ROCR)
pred<-prediction(predictions = roc_results$X1,labels = roc_results$fold_test.yy)
perf<-performance(pred,measure = "tpr",x.measure = "fpr")
plot(perf,main="Decision tree",col="blue",lwd=3)
abline(a=0,b=1,lty=2)
perf.auc<-performance(pred,measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
sum3<-as.numeric(perf.auc@y.values)+sum3
average3<-sum3/10
}
print(average)
print(average1)
print(average2)
print(average3)
write.csv(A,file="D:/SUN/Decision tree curve modification/Probability.csv")
write.csv(B,file="D:/SUN/Decision tree curve modification/Label.csv")
