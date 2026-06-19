import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest import RandomForest

iris = load_iris()
print(iris.DESCR)
x = iris.data.astype(np.float32)
y = iris.target.astype(np.int32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('Started training...')
rf = RandomForest(x_train, y_train)

print('Finished training')
preds = rf.predict(x_test)

print(np.mean(preds == y_test))