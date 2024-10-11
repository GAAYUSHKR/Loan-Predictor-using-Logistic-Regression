
# coding: utf-8

# In[72]:


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import math
import matplotlib
import numpy as np
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as sns
from sklearn.metrics import accuracy_score
import scipy.stats as ss
import numpy.random as nr
from sklearn.linear_model import LogisticRegression
from sklearn import feature_selection as fs
from sklearn import linear_model
import sklearn.decomposition as skde
from sklearn import svm


# In[2]:


matplotlib.style.use('ggplot')


# In[6]:


train=pd.read_csv("file:///C:/Users/Avinash/Downloads/csv file/ln train.csv")
test=pd.read_csv("file:///C:/Users/Avinash/Downloads/csv file/ln test.csv")


# In[81]:


test=test=pd.read_csv("file:///C:/Users/Avinash/Downloads/ln test.csv")


# In[7]:


train.shape


# In[5]:


test.shape


# In[7]:


train.head()


# In[8]:


train.columns


# In[8]:


train.dtypes


# In[60]:


df=np.array(train['Loan_Status'])
len1=len(df)
for i in range(len1):
    if df[i]=='Y':
        train['Loan_Status_One']=1
if df[i]=='N':
    train['Loan_Status_One']=0


# In[9]:


def plot1_hist(train,cols):
    for col in cols:
        train[col].plot.hist(bins=10)
        plt.show()


# In[10]:


num_cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
plot1_hist(train,num_cols)


# In[12]:


def plot_bar(train,cols):
    for col in cols:
        counts=train[col].value_counts()
        counts.plot.bar()
        plt.show()


# In[13]:


cols=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
plot_bar(train,cols)


# In[14]:


def plot_scatter(train,col_x,col_y='LoanAmount'):
    for col in col_x:
        train.plot.scatter(x=col_x,y=col_y,alpha=0.4)
        plt.show()


# In[15]:


num_cols1=['Loan_Amount_Term']
plot_scatter(train,num_cols1)


# In[20]:


def box_plot(train,cols,col_y='ApplicantIncome'):
    for col in cols:
        sn.boxplot(x=col,y=col_y,data=train)
        plt.show()


# In[21]:


cols=['Gender','Married','Education','Self_Employed','Property_Area']
box_plot(train,cols)


# In[22]:


train.dtypes


# In[23]:


def plot_crosstab(train,cols,col_y='Loan_Status'):
    for col in cols:
        df=pd.crosstab(train[col],train[col_y])
        df.div(df.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
        plt.show()


# In[24]:


cols=['Gender','Married','Education','Self_Employed','Property_Area']
plot_crosstab(train,cols)


# In[11]:


train.isnull().sum()


# In[12]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)


# In[13]:


train['Self_Employed'].value_counts()


# In[14]:


train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)


# In[15]:


train['Loan_Amount_Term'].value_counts()


# In[16]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


# In[17]:


train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)


# In[18]:


train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)


# In[19]:


train.isnull().sum()


# In[20]:


train.columns


# In[21]:


test.isnull().sum()


# In[22]:


test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0],inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].mean(),inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace=True)


# In[23]:


test.isnull().sum()


# In[24]:


train['Loan_Amount_Log']=np.log(train['LoanAmount'])
test['Loan_Amount_Log']=np.log(test['LoanAmount'])


# In[24]:


pd.DataFrame(train,columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status','Loan_Amount_Log']).to_csv('ln_train_prep.csv')


# In[28]:


pd.DataFrame(test,columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area','Loan_Amount_Log']).to_csv('ln_test_prep.csv')


# In[39]:


train['Loan_Amount_Log'].plot.hist(bins=10)


# In[40]:


print(train.Loan_ID.unique().shape)


# In[41]:


train.shape


# In[25]:


features2=train['Property_Area']
features1=test['Property_Area']
enc=preprocessing.LabelEncoder()
enc.fit(features2)
enc.fit(features1)
features2=enc.transform(features2)
features1=enc.transform(features1)


# In[26]:


ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(features2.reshape(-1,1))
encoded1=ohe.fit(features1.reshape(-1,1))
features2 = encoded.transform(features2.reshape(-1,1)).toarray()
features1 = encoded.transform(features1.reshape(-1,1)).toarray()


# In[27]:


features2[:10,:]


# In[28]:


def encoded_string(cat_fet):
    enc=preprocessing.LabelEncoder()
    enc.fit(cat_fet)
    enc_cat_fet=enc.transform(cat_fet)
    ohe=preprocessing.OneHotEncoder()
    encoded=ohe.fit(enc_cat_fet.reshape(-1,1))
    return encoded.transform(enc_cat_fet.reshape(-1,1)).toarray()


# In[29]:


train.dtypes


# In[30]:



test.ApplicantIncome=test.ApplicantIncome.astype(float)
test.CoapplicantIncome=test.CoapplicantIncome.astype(float)
test.dtypes


# In[31]:


cat_fet=['Gender','Married','Dependents','Education','Self_Employed']
for col in cat_fet:
    temp=encoded_string(train[col])
    temp1=encoded_string(test[col])
    features2=np.concatenate([features2,temp],axis=1)
    features1=np.concatenate([features1,temp1],axis=1)


# In[35]:





# In[32]:


features2.shape


# In[33]:


features2[:2,:]


# In[34]:


train.ApplicantIncome=train.ApplicantIncome.astype(float)


# In[35]:


features2=np.concatenate([features2,np.array(train[['ApplicantIncome','CoapplicantIncome','Loan_Amount_Log','Loan_Amount_Term','Credit_History']])],axis=1)


# In[36]:


features1=np.concatenate([features1,np.array(test[['ApplicantIncome','CoapplicantIncome','Loan_Amount_Log','Loan_Amount_Term','Credit_History']])],axis=1)


# In[37]:


features2[:2,:]


# In[38]:


features1[:2,:]


# In[39]:


features2.shape


# In[40]:


features1.shape


# In[59]:


nr.seed(9988)
labels=np.array(train['Loan_Status'])
index=range(features2.shape[0])
index=ms.train_test_split(index,test_size=0.4)
x_train=features2[index[0],:]
y_train=np.ravel(labels1[index[0]])
x_test=features2[index[1],:]
y_test=np.ravel(labels1[index[1]])

Scaler=preprocessing.StandardScaler().fit(x_train[:,15:]
x_train[:,15:]=Scaler.transform(x_train[:,15:]
x_test[:,15:]=Scaler.transform(x_test[:,15:]
features1[:,15:]=Scaler.transform(features1[:,15:])
# In[56]:


train.loc[train['Loan_Status'] == 'Y', 'Loan_Status_1'] = 1
train.loc[train['Loan_Status'] == 'N', 'Loan_Status_1'] = 0


# In[57]:


train.head()


# In[58]:


labels1=np.array(train['Loan_Status_1'])


# In[62]:


x_train[:2,:]


# In[63]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[64]:


predicted=model.predict(x_test)


# In[65]:


accuracy_score(y_test,predicted)


# In[66]:


test.head()


# In[155]:


pred_test=model.predict(features1)


# In[157]:


submission=pd.read_csv("file:///C:/Users/Avinash/Downloads/ln_submission.csv")


# In[158]:


submission.head()


# In[159]:


submission['Loan_Status']=pred_test


# In[160]:


submission['Loan_ID']=test['Loan_ID']


# In[161]:


submission.head()


# In[162]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic 1.csv')


# In[67]:


print(features2.shape)


# In[100]:


sel=fs.VarianceThreshold(threshold=(0.8*(1-0.8)))
features_reduced=sel.fit_transform(features2)
features_reduced_2=sel.fit_transform(features1)


# In[69]:


features_reduced.shape


# In[102]:


features_reduced_2.shape


# In[105]:


labels1=labels1.reshape(labels1.shape[0],)
nr.seed(998)
features_folds=ms.KFold(n_splits=10,shuffle=True)
logistic_mod=linear_model.LogisticRegression(C=10,class_weight={0:0.45,1:0.55})
nr.seed(9955)
selector=fs.RFECV(estimator=logistic_mod,cv=features_folds,scoring='roc_auc')
selector=selector.fit(features_reduced,labels1)
features_reduced=selector.transform(features_reduced)
features_reduced_2=selector.transform(features_reduced_2)
features_reduced.shape


# In[71]:


nr.seed(9955)
inside=ms.KFold(n_splits=10,shuffle=True)
nr.seed(9955)
outside=ms.KFold(n_splits=10,shuffle=True)


# In[72]:


nr.seed(9945)
param_grid={"C":[0.1,1,10,100,1000]}
logistic_mod=linear_model.LogisticRegression(class_weight={0:0.45,1:0.55})
clf=ms.GridSearchCV(estimator=logistic_mod,param_grid=param_grid,cv=inside,scoring='roc_auc',return_train_score=True)
clf.fit(features_reduced,labels1)
clf.best_estimator_.C


# In[95]:


nr.seed(9955)
indx=range(features_reduced.shape[0])
indx=ms.train_test_split(indx,test_size=0.4)
x_train=features_reduced[indx[0],:]
y_train=np.ravel(labels1[indx[0]])
x_test=features_reduced[indx[1],:]
y_test=np.ravel(labels1[indx[1]])


# In[96]:


logistic_model=linear_model.LogisticRegression(C=1,class_weight={0:0.45,1:0.55})
logistic_model.fit(x_train,y_train)


# In[91]:


predicted=logistic_model.predict(x_test)


# In[92]:


accuracy_score(y_test,predicted)


# In[106]:


pred_test=logistic_model.predict(features_reduced_2)


# In[107]:


submission=pd.read_csv("file:///C:/Users/Avinash/Downloads/csv file/ln_submission.csv")


# In[108]:


submission['Loan_Status']=pred_test


# In[109]:


submission['Loan_ID']=test['Loan_ID']


# In[112]:


submission.head()


# In[111]:


submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[113]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic_imp.csv')


# In[44]:


pca_mod=skde.PCA()
pca_comp=pca_mod.fit(x_train)
pca_comp


# In[46]:


print(pca_comp.explained_variance_ratio_)


# In[48]:


print(np.sum(pca_comp.explained_variance_ratio_))


# In[49]:


def plot_explained(mod):
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x]          
    plt.plot(x,comps)


# In[50]:


plot_explained(pca_comp)


# In[89]:


pca_mod_5=skde.PCA(n_components=5)
pca_mod_5.fit(x_train)
comps=pca_mod_5.transform(x_train)
comps.shape


# In[90]:


log_mod_5=linear_model.LogisticRegression(C=1.0,class_weight={0:0.45,1:0.55})
log_mod_5.fit(comps,y_train)


# In[91]:


predicted=log_mod_5.predict(pca_mod_5.transform(x_test))


# In[92]:


accuracy_score(y_test,predicted)


# In[85]:


model=svm.LinearSVC()


# In[86]:


model.fit(x_train,y_train)


# In[87]:


predict=model.predict(x_test)


# In[88]:


accuracy_score(predict,y_test)

