#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import plot_tree


# In[24]:


data=pd.read_csv('salaries.csv')


# In[25]:


input=data.drop('salary_more_then_100k',axis='columns')
target=data['salary_more_then_100k']


# In[26]:


le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()


# In[27]:


input['companyEnc']=le_company.fit_transform(input['company'])
input['jobEnc']=le_job.fit_transform(input['job'])
input['degreeEnc']=le_degree.fit_transform(input['degree'])


# In[28]:


inputs=input.drop(['company','job','degree'],axis='columns')


# In[29]:


model=tree.DecisionTreeClassifier(criterion='gini')
model.fit(inputs,target)


# In[30]:


plt.figure(figsize=(20,10))
plot_tree(model,feature_names=input.columns,class_names=['No','Yes'],filled=True)
plt.title('Decision Tree')
plt.show()


# In[ ]:




