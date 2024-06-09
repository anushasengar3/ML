#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_curve,auc
from sklearn.model_selection import train_test_split


# In[28]:


data=pd.read_csv('diabetes.csv')


# In[29]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


y_pred=np.random.randint(2,size=len(y_test))


# In[32]:


def calculate_metrics(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred)
    tp,tn,fp,fn=cm[1,1],cm[0,0],cm[0,1],cm[1,0]
    accuracy=(tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)!=0 else 0
    precision=tp/(tp+fp) if (tp+fp)!=0 else 0
    recall=tp/(tp+fn) if (tp+fn)!=0 else 0
    f1=2*precision*recall/(precision+recall) if (precision+recall)!=0 else 0
    specificity=tn/(tn+fp) if (tn+fp)!=0 else 0
    npv=tn/(tn+fn) if (tn+fn)!=0 else 0
    mcc_denominator=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=((tp*tn)-(fp*fn)) if mcc_denominator!=0 else 0
    return tp,tn,fp,fn,accuracy,precision,recall,f1,specificity,npv,mcc
tp,tn,fp,fn,accuracy,precision,recall,f1,specificity,npv,mcc=calculate_metrics(y_test,y_pred)
print(f"TP:{tp},TN:{tn},FP:{fp},FN:{fn}")
print(f"Accuracy:{accuracy:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")
print(f"MCC: {mcc:.4f}")


# In[33]:


metrics_sklearn={
    "Accuracy(scikit-learn)":accuracy_score(y_test,y_pred),
     "Precision(scikit-learn)":precision_score(y_test,y_pred),
    "Recall(Sensivity)(scikit-learn)":recall_score(y_test,y_pred),
    "F1_score(scikit-learn)":f1_score(y_test,y_pred),
    "MCC(scikit-learn)":matthews_corrcoef(y_test,y_pred),
}


# In[34]:


for metric,value in metrics_sklearn.items():
    print(f"{metric}:{value:.4f}")


# In[35]:


print("Confusion matrix(scikit-learn):\n",confusion_matrix(y_test,y_pred))


# In[36]:


fpr,tpr,_=roc_curve(y_test,y_pred)
auc_score=auc(fpr,tpr)
print(f"AUC:{auc_score:.4f}")


# In[39]:


plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='blue',lw=2,label=f'AUC={auc_score:.4f}')
plt.plot([0,1],[0,1],color='gray',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




