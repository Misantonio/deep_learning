import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

df = pd.read_csv('breast_cancer.data')
df.replace('?', -99999, inplace=True)

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)

example_measures = np.array([[1,2,3,4,5,6,7,8,9], [6,5,3,7,23,2,1,5,8]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)