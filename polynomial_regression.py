#Polynomial Regression Model ********************************************************************

#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#import file and differentiate x and y varialbes 
df = pd.read_csv("Position_Salaries.csv")
print(df)
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

#Make a Linear Regression Model on which Polynomial Regression will be applied
#For a polynomial Regression to work it needs a Linear Regression model to be applied on
lr = LinearRegression()
lr_model = lr.fit(x, y)

#Show graph for Linear Regression Model
plt.scatter(x, y, color="red")
plt.plot(x, lr_model.predict(x))
plt.title("Linear Regression Model")
plt.xlabel("Level")
plt.ylabel("salary")
plt.show()

#Make polynomial Regression Model by making another Linear Regression Model
#degree states the number of co-ef to be included in the equation
pf = PolynomialFeatures(degree=4)
pf_model = pf.fit_transform(x)
lr_model_2 = LinearRegression()
lr_model_2.fit(pf_model, y)

#Show the Polynomial Regression Model graph 
plt.scatter(x, y, color="red")
plt.plot(x, lr_model_2.predict(pf.fit_transform(x)))
plt.xlabel("Level")
plt.ylabel("salary")
plt.title("Polynomial Regression Model")
plt.show()
