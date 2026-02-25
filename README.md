# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset: Import required libraries and read the Salary.csv dataset.
2. Prepare the Data: Separate the independent variable (Position Level) as X and the dependent variable (Salary) as y.
3. Train the Model: Create a Decision Tree Regressor model and fit it with X and y.
4. Predict the Salary: Use the trained model to predict the salary for a given position level.

## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Sujitha S 
RegisterNumber: 25015880 
*/

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")
le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])
X = data.iloc[:, 1:2].values  
y = data.iloc[:, 2].values   
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
print(f"Predicted Salary for Level 6.5: ${y_pred[0]}")
import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Regression')
plt.title('Salary vs Level (Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```
## Output:
![alt text](<exp 09.png>)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
