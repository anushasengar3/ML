import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import plot_tree

data=pd.read_csv('salaries.csv')

input=data.drop('salary_more_then_100k',axis='columns')
target=data['salary_more_then_100k']

le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

input['companyEnc']=le_company.fit_transform(input['company'])
input['jobEnc']=le_job.fit_transform(input['job'])
input['degreeEnc']=le_degree.fit_transform(input['degree'])

inputs=input.drop(['company','job','degree'],axis='columns')

model=tree.DecisionTreeClassifier(criterion='gini')
model.fit(inputs,target)

plt.figure(figsize=(20,10))
plot_tree(model,feature_names=input.columns,class_names=['No','Yes'],filled=True)
plt.title('Decision Tree')
plt.show()
