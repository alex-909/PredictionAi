import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import keras
import graphviz

df = pd.read_csv('data\\train_data.csv')

x = df.iloc[:, 0:32]
y = df.iloc[:, 32:34]

x = preprocessing.MinMaxScaler().fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

model = Sequential()
model.add(Input(shape=(32, )))
model.add(Dense(10, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(2, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=10, validation_data=(x_val, y_val))
#keras.utils.plot_model(model, "multi_input_and_output_model.png")
