# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Initialize the model parameters (slope m and intercept b) with initial values (usually 0) and choose a learning rate α.
2.Compute the predicted output using the linear equation
3.Calculate the error using the cost function (Mean Squared Error) and find the gradients of the cost with respect to m and b.
4.Update the parameters m and b using gradient descent and repeat the steps until the error becomes minimum..

 
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: sharmila 
RegisterNumber: 212225230261
*/

import numpy as np
import matplotlib.pyplot as plt
x = np.array([10000, 20000, 30000, 40000, 50000, 60000])
y = np.array([15000, 30000, 45000, 55000, 70000, 80000])
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std
w = 0.0
b = 0.0

alpha = 0.01
epochs = 1000
n = len(x)

losses = []

for _ in range(epochs):
    y_hat = w * x + b
    
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x, y)

x_sorted = np.argsort(x)
plt.plot(x[x_sorted], (w*x + b)[x_sorted], color='red')

plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit (scaled)")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()
new_value = float(input("Enter R&D Spend: "))
new_scaled = (new_value - x_mean) / x_std

pred = w * new_scaled + b
pred = pred * y_std + y_mean

print("Predicted Profit:", pred)

print("Final weight (w):", w)
print("Final bias (b):", b)


```

## Output:
<img width="993" height="702" alt="image" src="https://github.com/user-attachments/assets/fece45f6-9568-43e8-aec9-7277335ce186" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
