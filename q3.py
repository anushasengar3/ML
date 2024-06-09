#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


# In[13]:


data=pd.read_csv('diabetes.csv')


# In[14]:


input=data.drop(['Outcome'],axis=1)
target=data['Outcome']


# In[15]:


target


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(input,target,test_size=0.2)


# In[19]:


model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)


# In[20]:


model.score(X_test,y_test)


# In[22]:


tree=model.estimators_[0]


# In[23]:


plt.figure(figsize=(20,10))
plot_tree(tree,filled=True,feature_names=input.columns,class_names=['yes','no'],precision=2,max_depth=2)
plt.show()


# In[ ]:




