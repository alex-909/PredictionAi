import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import keras
import graphviz
from keras import layers

df = pd.read_csv('data\\train_data.csv')

#x = df.iloc[:, 0:32]
#y = df.iloc[:, 32:34]

#x = preprocessing.MinMaxScaler().fit_transform(x)

#x_train, x_val, y_train, y_val  = train_test_split(x,y, test_size=0.3)

th = keras.Input(shape=(1, ), name="th")
h1 = keras.Input(shape=(3, ), name="h1")
h2 = keras.Input(shape=(3, ), name="h2")
h3 = keras.Input(shape=(3, ), name="h3")
h4 = keras.Input(shape=(3, ), name="h4")
h5 = keras.Input(shape=(3, ), name="h5")

ta = keras.Input(shape=(1, ), name="ta")
a1 = keras.Input(shape=(3, ), name="a1")
a2 = keras.Input(shape=(3, ), name="a2")
a3 = keras.Input(shape=(3, ), name="a3")
a4 = keras.Input(shape=(3, ), name="a4")
a5 = keras.Input(shape=(3, ), name="a5") 

ch1 = Dense(units=32, name="ch1")(h1)
ch2 = Dense(units=32, name="ch2")(h2)
ch3 = Dense(units=32, name="ch3")(h3)
ch4 = Dense(units=32, name="ch4")(h4)
ch5 = Dense(units=32, name="ch5")(h5)
ca1 = Dense(units=32, name="ca1")(a1)
ca2 = Dense(units=32, name="ca2")(a2)
ca3 = Dense(units=32, name="ca3")(a3)
ca4 = Dense(units=32, name="ca4")(a4)
ca5 = Dense(units=32, name="ca5")(a5)

con1 = layers.concatenate([th, ch1, ch2, ch3, ch4, ch5])
con2 = layers.concatenate([ta, ca1, ca2, ca3, ca4, ca5])

allcon = layers.concatenate([con1, con2])
output = Dense(units=2, name="out")(allcon)

model = keras.Model(
    inputs=[th, h1, h2, h3, h4, h5, ta, a1, a2, a3, a4, a5],
    outputs=[output]
)
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
print("compiling")
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
    metrics = ['accuracy'],
)

print("fit")
model.fit(       
    {
     "th": df.iloc[:, 0],                                          
     "h1": df.iloc[:, 1:4],                                          
     "h2": df.iloc[:, 4:7],                                         
     "h3": df.iloc[:, 7:10],                                         
     "h4": df.iloc[:, 10:13],                                         
     "h5": df.iloc[:, 13:16],                                            

     "ta": df.iloc[:, 16],                                          
     "a1": df.iloc[:, 17:20],                                          
     "a2": df.iloc[:, 20:23],                                         
     "a3": df.iloc[:, 23:26],                                         
     "a4": df.iloc[:, 26:29],                                         
     "a5": df.iloc[:, 29:32],                                            
    
    },
    {"out": df.iloc[:, 32:34]},
    epochs=10,
    batch_size=32,
)
