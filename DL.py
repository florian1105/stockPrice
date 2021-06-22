# %%
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# %%


def load_data(company, start, end):
    data = web.DataReader(company, 'yahoo', start, end)
    return data


# %%
# Company to be focused on: facebook
company = 'FB'

data = load_data(company=company,
                 start=dt.datetime(2012, 1, 1),
                 end=dt.datetime(2019, 1, 1))

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# how many days do i want to base my predictions on ?
prediction_days = 60
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#%%
print(data)
# %%
# Models construct

def LSTM_model():

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model
#%%

def GRU_model():

    model = Sequential()

    model.add(GRU(units=100, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(GRU(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model


# %%
# Training
model1 = LSTM_model()

model1.summary()
model1.compile(optimizer='adam',
              loss='mean_squared_error')

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath='weights_best.hdf5',
                               verbose=2,
                               save_best_only=True)

model1.fit(x_train,
          y_train,
          epochs=25,
          batch_size=32,
          callbacks=[checkpointer])
#%%
# Training GRU
model2 = GRU_model()

model2.summary()
model2.compile(optimizer='adam',
              loss='mean_squared_error')

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath='weights_best.hdf5',
                               verbose=2,
                               save_best_only=True)
model2.fit(x_train,
          y_train,
          epochs=25,
          batch_size=32,
          callbacks=[checkpointer])          

#%%
# test model accuracy on existing data
test_data = load_data(company = 'FB',
                      start = dt.datetime(2019,1,1),
                      end = dt.datetime.now())

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Construction du jeu de test
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

#prediction 
predicted_prices = model2.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#affichage 
plt.plot(actual_prices, color='black', label=f"Actual {company} price")
plt.plot(predicted_prices, color= 'green', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel("time")
plt.ylabel(f"{company} share price")
plt.legend()
plt.show()

# predicting next day
real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model2.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"prediction: {prediction}")
# %%
