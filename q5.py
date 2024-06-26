import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error

iris = load_iris()
X = iris.data[:, 2:3]  
y = (iris.target == 2).astype(int)  # Binary classification: Iris-virginica vs. not Iris-virginica

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_prob)
rmse = np.sqrt(mse)

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_prob, color='red', linewidth=2, label='Logistic Regression Curve')
plt.xlabel('Petal Length')
plt.ylabel('Probability')
plt.title('Logistic Regression: Probability of Iris-virginica')
plt.legend()
plt.show()

new_data = np.array([[1.5], [3.5], [5.0]])
new_pred = model.predict(new_data)
new_prob = model.predict_proba(new_data)[:, 1]

print(f'New Data Predictions: {new_pred}')
print(f'New Data Probabilities: {new_prob}')




