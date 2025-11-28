# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:LOGESHWARAN S
 
RegisterNumber:25007255
  
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression (X1, y):
    X = np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    num_iters = 1000   # number of iterations
    learning_rate = 0.01
    for _ in range(num_iters):
       predictions = X.dot(theta).reshape(-1, 1)
       errors = (predictions - y).reshape(-1, 1)
       theta = theta - learning_rate * (1 / len(X)) * X.T.dot(errors)
       return theta
data=pd.read_csv("/content/drive/MyDrive/50_Startups.csv")
data.head(11)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print("X =",X)
print("X1_Scaled =",X1_Scaled)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")
```

## Output:
![linear regression using gradient descent](sam.png)<img width="368" height="734" alt="image" src="https://github.com/user-attachments/assets/cfb33122-c3be-4053-a70f-1cb37a8878e4" />

<img width="631" height="719" alt="image" src="https://github.com/user-attachments/assets/29a813af-b036-41a3-9faf-30479a348a4d" />

<img width="385" height="65" alt="image" src="https://github.com/user-attachments/assets/d3b1858b-0dab-40ed-b0ea-a31ecabf1994" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
