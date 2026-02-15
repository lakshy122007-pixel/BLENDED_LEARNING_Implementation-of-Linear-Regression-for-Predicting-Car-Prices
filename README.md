# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import libraries (pandas, numpy, sklearn matplotlib, seaborn, statsmodels)
3. Load dataset
4. Read CarPrice_Assignment.csv into a dataframe df.
5. Select features and target
6. Set X = [enginesize, horsepower, citympg, highwaympg]
 Set Y = price
8. Split dataset
9. Divide into training and testing sets using train_test_split (80% train, 20% test).
10. Scale the features
11. Create StandardScaler
12. Fit scaler on X_train and transform it → X_train_scaled
13. Transform X_test using same scaler → X_test_scaled
14. Train the model
15. Create LinearRegression model and fit training data
16. Predict prices for X_test_scaled → Y_pred
17. Evaluate performance
Compute and print: MSE, MAE, RMSE, R²
Check linearity
18. Plot Actual (Y_test) vs Predicted (Y_pred) with reference line
19. Residual analysis
Compute residuals = Y_test - Y_pred
Calculate Durbin-Watson statistic for autocorrelation
Check homoscedasticity
20. Plot residuals vs predicted using sns.residplot


## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv("CarPrice_Assignment.csv")

df.head()

X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,Y_train)
Y_pred=model.predict(X_test_scaled)
Y_pred

print('Name:Lakshiya Rajkmar ')
print('Reg. No:212225240076 ')
print("MODEL COEFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(Y_test,Y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(Y_test,Y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):>10.2f}")

plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred,alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show

residuals=Y_test - Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}","\n(Values close to 2 indicates no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residual vs Predicted")
plt.xlabel("Predicted Price($)")
plt.ylabel("Residuals($)")
plt.grid(True)
plt.show()

Developed by: Lakshiya Rajkumar
RegisterNumber:  25000390
*/
```

## Output:
![alt text](<Screenshot 2026-02-15 115724.png>)
![alt text](<Screenshot 2026-02-15 115752.png>)
![alt text](<Screenshot 2026-02-15 115811.png>)
![alt text](<Screenshot 2026-02-15 115822.png>)
![alt text](<Screenshot 2026-02-15 115838.png>)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
