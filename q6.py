#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error,accuracy_score,precision_score,f1_score,recall_score,r2_score,classification_report


# In[3]:


np.random.seed(42)
no_samples=1000
class1=[1,2]
cov1=[[1,0.5],[0.5,1]]
class2=[4,5]
cov2=[[1,-0.5],[-0.5,1]]


# In[4]:


class1data=np.random.multivariate_normal(class1,cov1,no_samples//2)
class2data=np.random.multivariate_normal(class2,cov2,no_samples//2)


# In[5]:


X=np.concatenate((class1data,class2data),axis=0)
y=np.concatenate(([0]*np.zeros(no_samples//2),[1]*np.ones(no_samples//2)),axis=0)


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[7]:


model=GaussianNB()
model.fit(X_train,y_train)


# In[8]:


y_pred=model.predict(X_test)


# In[14]:


acc=accuracy_score(y_test,y_pred)
pre=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print(f"accuracy : {acc}")
print(f"precision : {pre}")
print(f"recall : {rec}")
print(f"f1 : {f1}")

print(classification_report(y_test,y_pred))


# In[ ]:




