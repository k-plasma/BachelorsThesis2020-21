import tensorflow as tf
import gradio as gr
import datetime
import pandas as pd
import numpy as np
import keras
import math
from keras.layers import LSTM
from keras.preprocessing import sequence
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

url_wheatArea = 'https://raw.githubusercontent.com/k-plasma/Machine-Learning-Models-for-Agricultural-Data-Applications/master/wheatArea.csv'
url_wheatGrossProduction = 'https://raw.githubusercontent.com/k-plasma/Machine-Learning-Models-for-Agricultural-Data-Applications/master/wheatGrossProductionValue.csv'
url_wheatProductionQuantity = 'https://raw.githubusercontent.com/k-plasma/Machine-Learning-Models-for-Agricultural-Data-Applications/master/wheatProductionQuantity.csv'
url_wheatYield = 'https://raw.githubusercontent.com/k-plasma/Machine-Learning-Models-for-Agricultural-Data-Applications/master/wheatYield.csv'


df_wheatArea = pd.read_csv(url_wheatArea)
df_wheatGrossProduction = pd.read_csv(url_wheatGrossProduction)
df_wheatProductionQuantity = pd.read_csv(url_wheatProductionQuantity)
df_wheatYield = pd.read_csv(url_wheatYield)


df_wheatArea.drop(columns=["Domain Code","Domain",
                         "Area Code","Area","Element Code","Element","Item Code","Item",
                         "Year Code","Unit","Flag","Flag Description","Note"], axis=1, inplace=True)

df_wheatGrossProduction.drop(columns=["Domain Code","Domain",
                         "Area Code","Area","Element Code","Element","Item Code","Item","Year",
                         "Year Code","Unit","Flag","Flag Description","Note"], axis=1, inplace=True)


df_wheatProductionQuantity.drop(columns=["Domain Code","Domain",
                         "Area Code","Area","Element Code","Element","Item Code","Item","Year",
                         "Year Code","Unit","Flag","Flag Description","Note"], axis=1, inplace=True)

df_wheatYield.drop(columns=["Domain Code","Domain",
                         "Area Code","Area","Element Code","Element","Item Code","Item","Year",
                         "Year Code","Unit","Flag","Flag Description","Note"], axis=1, inplace=True)

df_finalTable = pd.concat([df_wheatArea,df_wheatProductionQuantity,df_wheatYield,df_wheatGrossProduction], axis=1)

df_finalTable.set_index('Year', inplace=True)

url_finalTable = 'https://raw.githubusercontent.com/k-plasma/Machine-Learning-Models-for-Agricultural-Data-Applications/master/finalTable.csv'
df_finalTable = pd.read_csv(url_finalTable)
df_finalTable.set_index('Year', inplace=True)

df_finalTable.head()

scaler = MinMaxScaler()
df_scaledFinalTable = pd.DataFrame(scaler.fit_transform(df_finalTable), columns=df_finalTable.columns, index=df_finalTable.index)

X = df_scaledFinalTable.iloc[:, :3]

y = df_scaledFinalTable.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False)

X_train_np = X_train.values

y_train_np = y_train.values

X_test_np = X_test.values

y_test_np = y_test.values

X_train_np = np.reshape(X_train_np, newshape = (47, 1, 3))


X_test_np = np.reshape(X_test_np, newshape= (9, 1, 3))

y_train_np = np.reshape(y_train_np, newshape= (47, 1, 1))

y_test_np = np.reshape(y_test_np, newshape= (9, 1, 1))

model = Sequential()
model.add(GRU(100,return_sequences=True, dropout=0.2, recurrent_dropout=0.1, input_shape=(1,3)))
model.add(GRU(200, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train_np,y_train_np,
          validation_data=(X_test_np,y_test_np),
          epochs=150,
          batch_size=64,verbose=1,
          shuffle=False,
          callbacks=[tensorboard_callback])


def predict(area, quantity, wheat_yield):
  area = float(area)
  quantity = float(quantity)
  wheat_yield = float(wheat_yield)

  predict_list = np.array([[area, quantity, wheat_yield, 0]])
  predict_list_scaled = scaler.transform(predict_list)
  predict_list_scaled = np.array([np.array([predict_list_scaled[0][:-1]])])
  
  #holdup = tf.expand_dims(predict_list_scaled, axis=2)
  #print(dir(holdup))
  #print(holdup._numpy())
  #print(holdup)
  
  output = model(predict_list_scaled)
  test=np.array([np.array([0,0,0,output])])
  prediction = scaler.inverse_transform(test)
  return prediction[0,-1]


gr.Interface(predict,
    [
        gr.inputs.Textbox(label="Area"),
        gr.inputs.Textbox(label="Quantity"),
        gr.inputs.Textbox(label="Yield")
    ],
    gr.outputs.Textbox(label="GPV")).launch()








