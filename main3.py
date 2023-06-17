import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Concatenate, Dense, Input
import keras
import graphviz
import random

path = 'data\\train_data_edit.csv'

def test_model(model):
    # Read the CSV file
    data = pd.read_csv("data\\test.csv")

    # Randomly select 10 rows
    #random_indices = random.sample(range(1, len(data)), k=10)
    selected_data = data #.iloc[random_indices]

    # Select the input columns for scaling
    input_columns = ['th', 'lh1', 'lh2', 'lh3', 'lh4', 'lh5', 'ta', 'la1', 'la2', 'la3', 'la4', 'la5']
    output_column = 'diff'  # Replace with the actual output column name from your CSV file


    # Extract the input data for selected rows
    input_data = selected_data[input_columns].values
    output_data = selected_data[output_column].values

    # Apply min-max scaling
    scaler = MinMaxScaler()
    scaled_input_data = scaler.fit_transform(input_data)

    # Make predictions for the scaled input data
    predictions = model.predict([scaled_input_data[:, 0:1], scaled_input_data[:, 1:6], scaled_input_data[:, 6:7], scaled_input_data[:, 7:12]])

    # Print the predictions
    for i, prediction in enumerate(predictions):
        print(f"Prediction for row {i+1}: {prediction} | real : {output_data[i]}")

df = pd.read_csv(path)

x = df.iloc[:, 0:12]
y = df.iloc[:, 12]

x = preprocessing.MinMaxScaler().fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

# region arrays

input_th = Input(shape=(1,))
input_lh = Input(shape=(5,))

input_ta = Input(shape=(1,))
input_la = Input(shape=(5,))

input_arr = [
    input_th,
    input_lh,
    input_ta, 
    input_la
]

x_train_th = x_train[:, 0]
x_train_lh = x_train[:, 1:6]

x_train_ta = x_train[:, 6]
x_train_la = x_train[:, 7:12]
#add train/val/input arrays
x_train_arr = [
    x_train_th,    
    x_train_lh,
    x_train_ta,
    x_train_la
]

x_val_th = x_val[:, 0]
x_val_lh = x_val[:, 1:6]

x_val_ta = x_val[:, 6]
x_val_la = x_val[:, 7:12]

x_val_arr = [
    x_val_th,    
    x_val_lh,
    x_val_ta,
    x_val_la
]

# endregion

# region model

npg = 4     #   how many neurons per team performance?

lh = Dense(npg, activation='linear')(input_lh)
la = Dense(npg, activation='linear')(input_la)
lh = Dense(1, activation='linear')(lh)
la = Dense(1, activation='linear')(la)

# Merge the processed game data in one layer?
merged1 = Concatenate()([input_th, input_ta])
merged2 = Concatenate()([lh, la])

merged1 = Dense(2, activation='linear')(merged1)
merged2 = Dense(2, activation='linear')(merged2)

merge = Concatenate()([merged1, merged2])
merge = Dense(4, activation='linear')(merge)
merge = Dense(2, activation='linear')(merge)
output = Dense(1, activation='linear')(merge)

# Erstellen des Modells
model = Model(inputs=input_arr, outputs=output)

# endregion 

#test = [[5,-1,5.899999999999999,11,3,6.500000000000001,6,-1,2.7,-2,1,-0.5,1,3,3.8000000000000003,2,9,0,-0.5,-2,0,-0.5999999999999996,7,1,3.200000000000001,8,4,3.0000000000000004,-1,1,-0.7000000000000002,-4]]
#test = preprocessing.MinMaxScaler().fit_transform(x)


model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

model.fit(x_train_arr, y_train, batch_size=16, epochs=280, validation_data=(x_val_arr, y_val))
test_model(model)
keras.utils.plot_model(model, "out/structure.png")