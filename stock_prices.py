import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt

TIME_STEPS = 183
TRAIN_SET_PERCENT = 80
NEURON_UNITS = 50
EPOCHS = 20
BATCH_SIZE = 30
FILE_NAME = 'ing.csv'


def get_train_set_size(dataframe, train_set_percent):
    return round(len(dataframe) * train_set_percent / 100)


def create_time_steps_data_group(time_steps, data_set):
    X_tab = []
    Y_tab = []
    for i in range(time_steps, len(data_set)):
        X_tab.append(data_set[i - time_steps:i, 0])
        Y_tab.append(data_set[i, 0])

    X_tab, Y_tab = np.array(X_tab), np.array(Y_tab)
    X_tab = np.reshape(X_tab, (X_tab.shape[0], X_tab.shape[1], 1))
    return X_tab, Y_tab


def create_model(X_train, neuron_units, output_neuron_units):
    # building model
    model = Sequential()
    model.add(LSTM(units=neuron_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neuron_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neuron_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neuron_units))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_neuron_units)) # when output_neuron_units is 1 then there will be predicted close price for only one day, 2 for 2 days and so on...

    return model


def train_model(model, X_train, Y_train, epochs, batch_size):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)


def load_data(file_path, separator):
    return pd.read_csv(file_path, separator).iloc[::-1]


if __name__ == "__main__":

    # loading data
    df = load_data(FILE_NAME, ',')
    print(f'dataframe: {df}')

    # split into training and test set
    train_set_size = get_train_set_size(df, TRAIN_SET_PERCENT)
    training_dataset = df.iloc[:train_set_size, 2:3]
    test_dataset = df.iloc[train_set_size:, 2:3]
    print(f'test_dataset: {test_dataset}')

    # scalling data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scalled = sc.fit_transform(training_dataset.values)
    test_set_scalled = sc.fit_transform(test_dataset.values)

    # creating and training model
    X_train, Y_train = create_time_steps_data_group(TIME_STEPS, training_set_scalled)
    model = create_model(X_train, NEURON_UNITS, output_neuron_units=1)
    train_model(model, X_train, Y_train, EPOCHS, BATCH_SIZE)

    # test model
    X_test, Y_test = create_time_steps_data_group(TIME_STEPS, test_set_scalled)
    loss_value = model.evaluate(X_test, Y_test)
    print(f'loss_value: {loss_value}')

    # prediction
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(f'predicted_stock_price: {predicted_stock_price}')
    # plot
