#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

#Import the csv file and make a dataset

dataset = pd.read_csv("Position_Salaries.csv")

#Divide dependent and non-dependent varialbes 

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print("x")
print(x)
print("y")
print(y)

#reshape the non dependent variable to match the dependent variable 

y = y.reshape(len(y), 1)
print("y reshape:")
print(y, y.shape)

#Perform feature scaling using Standard Scaler on all the variables
#Feature scaling os done because the range of x variables and y variables
#are widely apart x -> 1-10 |-| y-> 45000 - 1000000

sc=StandardScaler()
sc_2=StandardScaler()
x = sc.fit_transform(x)
y = sc_2.fit_transform(y)

#make a support vector regressor object using the class SVR

regressor = SVR(kernel="rbf")

#Train the object with fit method and original x - y variables

regressor.fit(x, y)

#Test the prediction with new data or test data 

sc_2.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(-1, 1))

#Show the prediction on a graph but before showing make sure to reverse the feature scaling
#done on the variables before training and testing 
#the method inverse_transform reverses/inverses the feature scaling done on the variables and 
#converts them to their original value to be shown on the graph

plt.scatter(sc.inverse_transform(x), sc_2.inverse_transform(y), color="red")
plt.plot(sc.inverse_transform(x), sc_2.inverse_transform(regressor.predict(x).reshape(-1, 1)) )
plt.show()#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

#Import the csv file and make a dataset

dataset = pd.read_csv("Position_Salaries.csv")

#Divide dependent and non-dependent varialbes 

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print("x")
print(x)
print("y")
print(y)

#reshape the non dependent variable to match the dependent variable 

y = y.reshape(len(y), 1)
print("y reshape:")
print(y, y.shape)

#Perform feature scaling using Standard Scaler on all the variables
#Feature scaling os done because the range of x variables and y variables
#are widely apart x -> 1-10 |-| y-> 45000 - 1000000

sc=StandardScaler()
sc_2=StandardScaler()
x = sc.fit_transform(x)
y = sc_2.fit_transform(y)

#make a support vector regressor object using the class SVR

regressor = SVR(kernel="rbf")

#Train the object with fit method and original x - y variables

regressor.fit(x, y)

#Test the prediction with new data or test data 

sc_2.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(-1, 1))

#Show the prediction on a graph but before showing make sure to reverse the feature scaling
#done on the variables before training and testing 
#the method inverse_transform reverses/inverses the feature scaling done on the variables and 
#converts them to their original value to be shown on the graph

plt.scatter(sc.inverse_transform(x), sc_2.inverse_transform(y), color="red")
plt.plot(sc.inverse_transform(x), sc_2.inverse_transform(regressor.predict(x).reshape(-1, 1)) )
plt.show()
