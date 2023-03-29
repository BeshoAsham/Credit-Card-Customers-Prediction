#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
#Import Label encoder Library
from sklearn import preprocessing
#Import feature selection Library
import statsmodels.api as sm
#Import the train test split Library from sklearn
from sklearn.model_selection import train_test_split
#Import the Linear regression model Library from sklearn
from sklearn.linear_model import LogisticRegression
#Import the metrixs for evaluating the model performance
from sklearn.metrics import classification_report
#scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# In[2]:


dataset = pd.read_csv ("E:\\FCAI - HU\\LEVEL 3\\1ST TERM\\Selected 1\\project\\numrical DATASET\\BankChurners.csv")

#dataset.columns.values[-1] = 'classification'
dataset

#cleaning
#if any coloumn has null value
dataset.isna().sum()

#if we have duplicated rows (boolean)
dataset.duplicated().any()

#one hot/label encoding
#categorial(object) to numerical
dataset.info()


# In[3]:


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

#split_dataset_into_test_and_train
X_train, X_test, y_train, y_test = train_test_split(x, y,
    test_size=0.20, shuffle = True, random_state = 10)
# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
test_size=0.25, random_state= 5)
#X_train=60%_X_val=20%_X_test=20%
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train
X_test


# In[80]:


SVCModel = SVC(C=0.1, kernel='linear', degree=3, gamma='auto', shrinking=False,
                probability=True, tol=0.001, cache_size=200, class_weight=None,verbose=False,
                max_iter=-1, random_state =40
)
SVCModel.fit(X_train, y_train)

y_pred = SVCModel.predict(X_test)

AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score is : ', AccScore)


# In[82]:


from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# drawing confusion matrix
import seaborn as sns
sns.heatmap(cm, center = True)
plt.show()


# In[83]:


f1_score(y_test, y_pred)

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


# In[84]:


#Calculating ROC:  
#roc_curve(y_test, y_pred, pos_label=None, sample_weight=None,drop_intermediate=True)
#Calculating ROC AUC Score:  
#roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

#ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
#print('ROCAUC Score : ', ROCAUCScore)
fprValue, tprValue, thresholdsValue = roc_curve(y_test,y_pred)
#print('fpr Value  : ', fprValue)
#print('tpr Value  : ', tprValue)
#print('thresholds Value  : ', thresholdsValue)

 

fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test,y_pred)
AUCValue = auc(fprValue2, tprValue2)
#Calculating Area Under the Curve AUC : 
print('AUC Value  : ', AUCValue)

plt.show()

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(fprValue,tprValue, marker='.', label='SVC (auc = %0.2f)' % AUCValue)
plt.legend()
plt.show()


# In[25]:


from sklearn.neural_network import MLPClassifier
SVCModel = MLPClassifier(#hidden_layer_sizes=(32, 32),
              activation='relu',
              solver='adam',
              learning_rate='adaptive',
              early_stopping=True)
SVCModel.fit(X_train,y_train)
test_acc = accuracy_score(y_test, y_pred) * 100.
loss_values = SVCModel.loss_curve_
print (loss_values)
plt.plot(loss_values,label='loss curve')
plt.legend()
plt.show()


# In[16]:


from matplotlib import pyplot
#correlation between features
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none' )
fig.colorbar(cax)
pyplot.show()


# In[71]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[ ]:




