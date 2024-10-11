
# coding: utf-8

# In[129]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, svm
from sklearn import linear_model
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import seaborn as sns
import matplotlib
import math
from sklearn.metrics import accuracy_score


# In[130]:


matplotlib.style.use('ggplot')


# In[131]:


train=pd.read_csv("file:///C:/Users/Avinash/Downloads/csv file/all/train.csv")
test=pd.read_csv("file:///C:/Users/Avinash/Downloads/csv file/all/test.csv")


# In[132]:


train.shape


# In[133]:


test.shape


# In[134]:


train.head()


# In[136]:


train.dtypes


# In[137]:


train.isnull().sum()


# In[138]:


train=train.drop(labels=['Cabin'],axis=1)


# In[139]:


train['Age'].fillna(train['Age'].mean(),inplace=True)


# In[140]:


train.isnull().sum()


# In[144]:


train=train.dropna(axis=0)


# In[145]:


train.count()


# In[146]:


test.isna().sum()


# In[147]:


test['Age'].fillna(test['Age'].mean(),inplace=True)
test=test.drop(labels=['Cabin'],axis=1)


# In[148]:


test=test.dropna(axis=0)


# In[149]:


test.shape


# In[234]:


features=train['Sex']
enc=preprocessing.LabelEncoder()
enc.fit(features)
features=enc.transform(features)
ohe=preprocessing.OneHotEncoder()
encoded=ohe.fit(features.reshape(-1,1))
features=encoded.transform(features.reshape(-1,1)).toarray()


# In[151]:


features1=train['Embarked']
enc=preprocessing.LabelEncoder()
enc.fit(features1)
features1=enc.transform(features1)
ohe=preprocessing.OneHotEncoder()
encoded=ohe.fit(features1.reshape(-1,1))
features1=encoded.transform(features1.reshape(-1,1)).toarray()


# In[235]:


features.dtype


# In[163]:



train.Pclass=train.Pclass.astype(float)
train.SibSp=train.SibSp.astype(float)
train.Parch=train.Parch.astype(float)


# In[236]:


features=np.concatenate([features,np.array(train[['Pclass','Age','Fare','SibSp']])],axis=1)


# In[222]:


features=np.concatenate([features,features1],axis=1)


# In[237]:


labels=np.array(train['Survived'])


# In[238]:


print(labels.shape)
features.shape


# In[246]:


scaler=preprocessing.StandardScaler().fit(features[:,3:])
features[:,3:]=scaler.transform(features[:,3:])


# In[247]:


index=range(features.shape[0])


# In[248]:


index=ms.train_test_split(index,test_size=0.2)


# In[249]:


x_train=features[index[0],:]
y_train=np.ravel(labels[index[0]])
x_test=features[index[1],:]
y_test=np.ravel(labels[index[1]])


# In[250]:


features=preprocessing.scale(features)


# In[251]:


x_test.shape


# In[252]:


x_test.shape


# In[139]:


x_train, y_train, x_test, y_test=ms.train_test_split(features,labels,test_size=0.2)


# In[257]:


lin_model=linear_model.LogisticRegression()


# In[258]:


lin_model.fit(x_train,y_train)


# In[259]:


pred_y=lin_model.predict(x_test)


# In[263]:


accuracy_score(y_test,pred_y)


# In[ ]:




