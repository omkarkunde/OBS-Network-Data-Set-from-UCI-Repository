# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:45:15 2018

@author: User
"""

# Decision Tree Random Forest Extra Tree Classifier
import numpy as np
import pandas as pd
# identify final state of packet, multi-class data set
#%%
pd.set_option("display.max_columns",None)
network_data=pd.read_csv(r"F:\ANACONDA\spyder\OBS_Network_data.csv",header=None,delimiter=" *, *",engine='python')
network_data.head()
#%%
network_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]
#%%
print(network_data.isnull().sum())
#%%
network_data_rev=pd.DataFrame.copy(network_data)
network_data_rev.head()
#%%
#after dropping start a new code block
network_data_rev=network_data_rev.drop('Packet Size_Byte',axis=1)
#%%
network_data_rev.shape
#%%
colname=['Node','Full_Bandwidth','Node Status','Class']
colname
#%%
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()

for x in colname:
    network_data_rev[x]=le[x].fit_transform(network_data_rev.__getattr__(x))
#%%
network_data_rev.head()
#%%
Y=network_data_rev.values[:,-1]
X=network_data_rev.values[:,:-1]
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)
Y=Y.astype(int)
#%%
from sklearn.model_selection import train_test_split
#split data in2 test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#%%
#predicting using Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)
#%%
#fit the model on data and predict the values
Y_pred=model_DecisionTree.predict(X_test)
#print(Y_pred)
print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
classifier=DecisionTreeClassifier()
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#bcoz the value is 99.8% it was accurate and not overfitting , if it was overfitting result would b 80%
#%%
#svm <d.t
from sklearn import svm
classifier=svm.SVC(kernel='rbf',C=1,gamma=0.1)
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
#cannot predict 4 more than 2 classes accurately
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%

#%%
from sklearn import tree
with open(r'F:\ANACONDA\spyder\model_DecisionTree.txt',"w") as f:
    f=tree.export_graphviz(model_DecisionTree,out_file=f)
#%%
#pen txt file and copy paste and go to webgraphviz.com and generate graph
#%%
# predicting using the extratreesclassifier
from sklearn.ensemble import ExtraTreesClassifier
model=(ExtraTreesClassifier(21))
#fit the model on data and predict values
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
from sklearn.ensemble import RandomForestClassifier
model_RandomForest=RandomForestClassifier(501)
model_RandomForest.fit(X_train,Y_train)
Y_pred=model_RandomForest.predict(X_test)
#%%

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
#boosting
#ADABOOST
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100))
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)
#%%

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model_GradientBoosting=GradientBoostingClassifier()
model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
#ensemble modelling
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
#%%
estimators=[]
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
print(estimators)
#%%
ensemble=VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
print(Y_pred)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


