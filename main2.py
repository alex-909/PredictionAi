import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Concatenate, Dense, Input
import keras
import graphviz

df = pd.read_csv('data\\train_data.csv')

x = df.iloc[:, 0:32]
y = df.iloc[:, 32:33]

x = preprocessing.MinMaxScaler().fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

# region arrays

input_th = Input(shape=(1,))
input_h1 = Input(shape=(3,))
input_h2 = Input(shape=(3,))
input_h3 = Input(shape=(3,))
input_h4 = Input(shape=(3,))
input_h5 = Input(shape=(3,))

input_ta = Input(shape=(1,))
input_a1 = Input(shape=(3,))
input_a2 = Input(shape=(3,))
input_a3 = Input(shape=(3,))
input_a4 = Input(shape=(3,))
input_a5 = Input(shape=(3,))

input_arr = [
    input_th,
    input_h1, input_h2, input_h3, input_h4, input_h5,
    
    input_ta, 
    input_a1, input_a2, input_a3, input_a4, input_a5,
]

x_train_th = x_train[:, 0]
x_train_h1 = x_train[:, 1:4]
x_train_h1 = x_train[:, 4:7]
x_train_h1 = x_train[:, 7:10]
x_train_h1 = x_train[:, 10:13]
x_train_h1 = x_train[:, 13:16]

x_train_ta = x_train[:, 16]
x_train_a1 = x_train[:, 17:20]
x_train_a1 = x_train[:, 20:23]
x_train_a1 = x_train[:, 23:26]
x_train_a1 = x_train[:, 26:29]
x_train_a1 = x_train[:, 29:32]
#add train/val/input arrays
x_train_arr = [
    x_train_th,    
    x_train_h1, x_train_h1, x_train_h1, x_train_h1, x_train_h1,

    x_train_ta,
    x_train_a1, x_train_a1, x_train_a1, x_train_a1, x_train_a1,
]

x_val_th = x_val[:, 0]
x_val_h1 = x_val[:, 1:4]
x_val_h1 = x_val[:, 4:7]
x_val_h1 = x_val[:, 7:10]
x_val_h1 = x_val[:, 10:13]
x_val_h1 = x_val[:, 13:16]

x_val_ta = x_val[:, 16]
x_val_a1 = x_val[:, 17:20]
x_val_a1 = x_val[:, 20:23]
x_val_a1 = x_val[:, 23:26]
x_val_a1 = x_val[:, 26:29]
x_val_a1 = x_val[:, 29:32]

x_val_arr = [
    x_val_th,    
    x_val_h1, x_val_h1, x_val_h1, x_val_h1, x_val_h1,

    x_val_ta,
    x_val_a1, x_val_a1, x_val_a1, x_val_a1, x_val_a1,
]

# endregion

# region model

npg = 1     #   how many neurons per game? (game has 3 inputs)
# 5 last games of home team
h1 = Dense(npg, activation='linear')(input_h1)
h2 = Dense(npg, activation='linear')(input_h2)
h3 = Dense(npg, activation='linear')(input_h3)
h4 = Dense(npg, activation='linear')(input_h4)
h5 = Dense(npg, activation='linear')(input_h5)

# 5 last games of away team
a1 = Dense(npg, activation='linear')(input_a1)
a2 = Dense(npg, activation='linear')(input_a2)
a3 = Dense(npg, activation='linear')(input_a3)
a4 = Dense(npg, activation='linear')(input_a4)
a5 = Dense(npg, activation='linear')(input_a5)


# Merge the processed game data in one layer?
games_h = Concatenate()([h1, h2, h3, h4, h5])
games_a = Concatenate()([a1, a2, a3, a4, a5])

games_h = Dense(8, activation='linear')(games_h)
games_a = Dense(8, activation='linear')(games_a)

# merge all
merged = Concatenate()([input_th, games_h, input_ta, games_a])
merged = Dense(8, activation='linear')(merged)
merged = Dense(4, activation='linear')(merged)
output = Dense(1, activation='linear')(merged)

# Erstellen des Modells
model = Model(inputs=input_arr, outputs=output)

# endregion 

#test = [[5,-1,5.899999999999999,11,3,6.500000000000001,6,-1,2.7,-2,1,-0.5,1,3,3.8000000000000003,2,9,0,-0.5,-2,0,-0.5999999999999996,7,1,3.200000000000001,8,4,3.0000000000000004,-1,1,-0.7000000000000002,-4]]
#test = preprocessing.MinMaxScaler().fit_transform(x)


model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

model.fit(x_train_arr, y_train, batch_size=64, epochs=15, validation_data=(x_val_arr, y_val))
keras.utils.plot_model(model, "multi_input_and_output_model.png")

"""
tensors = model(test, training=False)

hg = tensors[0].numpy()[0]
ag = tensors[0].numpy()[1]

print(f"{int(hg)} : {int(ag)}")
#print(model.predict(test))
model.save("cnp_3.h5")
"""