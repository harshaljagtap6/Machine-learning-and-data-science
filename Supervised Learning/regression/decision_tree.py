import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("50_Startups.csv")
print(df.head())

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), )
print(len(x))
print(len(y))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)
plt.plot(x_test, y_pred, color='b')
plt.grid(color='black', linestyle=':', linewidth='1')
plt.show()


dtree=DecisionTreeRegressor(max_depth=10)
plt.figure(figsize=(15, 15))
dtree = dtree.fit(x_train, y_train)
tree.plot_tree(dtree, filled=True, fontsize=10)

print("Score:", dtree.score(x_test, y_test))

# plt.savefig(sys.stdout.buffer)
# sys.stdout.flush()
