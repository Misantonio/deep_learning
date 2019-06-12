import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

style.use('ggplot')

def plot_2D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    for target, color in zip(targets,colors):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], 
                    df.loc[indicesToKeep, 'PC2'], 
                    c = color,
                    s = 50)
    ax.legend(targets)
    plt.show()

def plot_3D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_zlabel('PC3', fontsize = 15)
    for target, color in zip(targets, colors):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], 
                   df.loc[indicesToKeep, 'PC2'], 
                   df.loc[indicesToKeep, 'PC3'],
                   c = color,
                   s = 50)
    ax.legend(targets)
    plt.show()


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal length','sepal width','petal length','petal width','target']

df = pd.read_csv(url, names=columns)
X = df.loc[:, columns[:-1]].values
Y = df.loc[:, columns[-1]].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, X_test.shape)

ncol = X.shape[1]
n_components = 3

model = Sequential()
model.add(Dense(6, activation='tanh', input_shape=(ncol,)))
model.add(Dense(n_components, activation='tanh', name="bottleneck"))
model.add(Dense(6, activation='tanh'))
model.add(Dense(ncol))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, X_train, epochs=50000, verbose=0)

mse = model.evaluate(X_train, X_train)

code_layer_model = Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
code_output = code_layer_model.predict(X_train)

labels = ['PC'+str(x) for x in range(1, n_components+1)]
df = pd.DataFrame(data=code_output, columns=labels)
df['target'] = Y_train
targets = set(Y_test)
colors = ['b', 'g', 'r']

if n_components == 2:
    plot_2D()
elif n_components == 3:
    plot_3D()