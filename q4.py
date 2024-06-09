#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Create the dataset
data = {
"Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7],
"Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25]
}
# Create a DataFrame
df = pd.DataFrame(data)
# Save to CSV
df.to_csv('student_scores.csv', index=False)
# Load the dataset
data = pd.read_csv('student_scores.csv')
# Separate features (X) and target variable (y)
X = data[['Hours']] # Feature
y = data['Scores'] # Target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Linear Regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
# Plotting the regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.legend()
plt.show()


# In[ ]:




