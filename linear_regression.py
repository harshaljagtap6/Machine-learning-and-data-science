import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("student-mat.csv", sep=";")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["famsup", "schoolsup", "traveltime"])], remainder='passthrough')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Co: ", linear.coef_)
print("Int: ", linear.intercept_)

pred = linear.predict(x_test)

for i in range(len(pred)):
    print("Predicted Answer: ", pred[i],"Actual Data:",  x_test[i],"Actual Answer: ", y_test[i])