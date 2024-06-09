#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[18]:


df=pd.read_csv('diabetes.csv')


# In[19]:


missing_values=df.isnull().sum()
print("Missing values in each column:\n", missing_values)


# In[20]:


X=df.drop(columns=['Outcome'])
y=df['Outcome']


# In[21]:


scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[23]:


def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))


# In[24]:


def knn_predict(X_train,y_train,X_test,k=3):
    y_pred=[]
    for test_point in X_test:
        distances=[]
        for i in range(len(X_train)):
            distance=euclidean_distance(test_point,X_train[i])
            distances.append((distance,y_train.iloc[i]))
        distances.sort(key=lambda x:x[0])
        k_nearest_neighbours=distances[:k]
        k_nearest_labels=[label for (_,label) in k_nearest_neighbours]
        most_common_label=max(set(k_nearest_labels),key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)


# In[25]:


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


# In[26]:


def knn_predict_manhattan(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            distance = manhattan_distance(test_point, X_train[i])
            distances.append((distance, y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (_, label) in k_nearest_neighbors]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)


# In[27]:


y_pred = knn_predict(X_train, y_train, X_test, k=3)


# In[28]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Euclidean distance and k=3:", accuracy)


# In[29]:


k_values = range(1, 21)
accuracies = []
for k in k_values:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracies.append(accuracy_score(y_test, y_pred))


# In[30]:


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('K vs. Accuracy (Euclidean Distance)')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# In[31]:


y_pred_manhattan = knn_predict_manhattan(X_train, y_train, X_test, k=3)


# In[32]:


accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
print("Accuracy with Manhattan distance and k=3:", accuracy_manhattan)


# In[ ]:




