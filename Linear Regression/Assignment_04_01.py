import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"]=housing.target

print(df.head())

X=df[["MedInc"]]
Y=df["Price"]

X_train ,X_test, Y_train , Y_test=train_test_split(
    X,Y,test_size=0.25,random_state=42
)

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

# Evalution

print("Mean Absolute Error: ", mean_absolute_error(Y_test,Y_pred))
print("Mean Squared Error: ", mean_squared_error(Y_test,Y_pred))
print("R2 Score: ", r2_score(Y_test,Y_pred))

# Visualize

plt.scatter(X_test,Y_test,colorizer='blue',label='Actual')
plt.scatter(X_test,Y_pred,colorizer='red',label='Predicted')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.show()