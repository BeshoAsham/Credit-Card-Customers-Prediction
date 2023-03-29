#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
import keras
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.model_selection import cross_validate
import seaborn as sns


# In[3]:


dataset = pd.read_csv ("E:\\FCAI - HU\\LEVEL 3\\1ST TERM\\Selected 1\\project\\numrical DATASET\\BankChurners.csv")

#dataset.columns.values[-1] = 'classification'
dataset.head()

#cleaning
#if any coloumn has null value
dataset.isna().sum()

#if we have duplicated rows (boolean)
dataset.duplicated().any()

#one hot/label encoding
#categorial(object) to numerical
dataset.info()


# In[4]:


#separate them in two dataframes
numerical = dataset.select_dtypes(exclude=['object','bool'])
categorical = dataset.select_dtypes (include=['object','bool'])

numerical.head()

categorical.head()

le= preprocessing.LabelEncoder()
label_encoded_categorical = categorical.apply(le.fit_transform)
label_encoded_categorical.head()

#combine them again by concatenation func
df=pd.concat([numerical, label_encoded_categorical], axis=1)
df.head()


# In[5]:


#feature selection
x= df.drop(['classification'], axis=1)
y= df.classification

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(x, y,
    test_size=0.20, shuffle = True, random_state =33)
# Use the same function above for the validation set, random 10,5
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
#test_size=0.25, random_state= 33,shuffle =True)
# 60 train, 20 validate, 20 test

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train
X_test


# In[6]:


LogisticRegressionModel = LogisticRegression( penalty='l2',solver='newton-cg',C=1.0,dual=False,tol=0.0001,
                                             #,random_state=33,fit_intercept=True,intercept_scaling=1
                                             class_weight='balanced',max_iter=100, l1_ratio=None,
                               multi_class='auto', verbose=0,warm_start=False, n_jobs=None)
LogisticRegressionModel.fit(X_train, y_train)

y_pred = LogisticRegressionModel.predict(X_test)
LogisticRegressionModel.n_iter_


# In[7]:


cm = confusion_matrix(y_test, y_pred)
cm

sns.heatmap(cm, center =True)
plt.show()


# In[9]:


F1=f1_score(y_test, y_pred)
print('F1 Score is : ', F1)

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  
#recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)

RecallScore = recall_score(y_test, y_pred) #it can be : binary,macro,weighted,samples
print('Recall Score is : ', RecallScore)

#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  
#precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)

PrecisionScore = precision_score(y_test, y_pred) #it can be : binary,macro,weighted,samples
print('Precision Score is : ', PrecisionScore)

AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score is : ', AccScore)


# In[10]:


LogisticRegressionModel = MLPClassifier(#hidden_layer_sizes=(32, 32),
              activation='relu',
              solver='adam',
              learning_rate='adaptive',
              early_stopping=True)
LogisticRegressionModel.fit(X_train,y_train)
test_acc = accuracy_score(y_test, y_pred) * 100.
loss_values = LogisticRegressionModel.loss_curve_
print (loss_values)

plt.plot(loss_values,label='loss curve')
plt.legend()
plt.show()


# In[12]:


fprValue, tprValue, thresholdsValue = roc_curve(y_test,y_pred)
fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test,y_pred)
AUCValue = auc(fprValue2, tprValue2)
print('AUC Value  : ', AUCValue)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(fprValue,tprValue, marker='.', label='Logistic (auc = %0.2f)' % AUCValue)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# In[ ]:




