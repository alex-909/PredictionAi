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
model.add(Dense(16, activation='linear'))
model.add(Dense(12, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(2, activation='relu'))

test = [[10,1,-2.2,3,1,-5.100000000000001,-10,1,1.4999999999999991,-8,-2,-1.5999999999999996,3,-1,-3.3000000000000007,-4,13,1,1.9000000000000004,-1,-2,-4.300000000000001,-15,1,-1.6,-7,-2,-2.1000000000000005,-7,-2,-3.5,-3]]
test = preprocessing.MinMaxScaler().fit_transform(x)


model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=(x_val, y_val))
keras.utils.plot_model(model, "multi_input_and_output_model.png")


tensors = model(test, training=False)

hg = tensors[0].numpy()[0]
ag = tensors[0].numpy()[1]

print(f"{hg} : {ag}")
#print(model.predict(test))
model.save("cnp_3.h5")
