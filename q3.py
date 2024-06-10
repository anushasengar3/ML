import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

data=pd.read_csv('diabetes.csv')

input=data.drop(['Outcome'],axis=1)
target=data['Outcome']

target

X_train,X_test,y_train,y_test=train_test_split(input,target,test_size=0.2)

model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)

model.score(X_test,y_test)

tree=model.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(tree,filled=True,feature_names=input.columns,class_names=['yes','no'],precision=2,max_depth=2)
plt.show()
