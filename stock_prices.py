import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from utils import transform_data_from_csv, convert_to_one_dim_array, file_exists
from time import time

FILE_NAME = 'ing.csv'
MODEL_FILEPATH = 'lstm_stock_prediction_ing.model'
CLOSE = 'Zamknięcie'
DATE = 'Data'
X_LABEL = 'Time in days'
Y_LABEL = 'Close price of stock'
# Params which can be changed
TIME_STEPS = 183
TRAIN_SET_PERCENT = 80
NEURON_UNITS = 30
EPOCHS = 5
BATCH_SIZE = 30
OUTPUTS = 1
PREDICTION_DAYS = 50
ELAPSED_TIME = 0


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


def get_model(X_train, neuron_units, outputs, filepath):
    if not file_exists(filepath):
        model = create_model(X_train, neuron_units, output_neuron_units=outputs)
        model.save(filepath)
        return model
    else:
        return load_model(filepath)


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
    # when output_neuron_units is 1 then there will be predicted close price for only one day, 2 for 2 days and so on...
    # but it returns 10 outputs so propably we need to take average result
    model.add(Dense(units=output_neuron_units))
    return model


def train_model(model, X_train, Y_train, epochs, batch_size):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)


def get_plot_title(epochs, prediction_days, neuron_units, elapsed_time):
    return f'''
        ING stock prediction for {prediction_days} days later with {epochs} epochs
        with {neuron_units} neurons with elapsed time {round(elapsed_time, 2)} seconds
    '''

def get_test_plot_title(epochs, time_steps, neuron_units, elapsed_time):
    return f'''
           ING stock test prediction within {time_steps} time steps with {epochs} epochs
           with {neuron_units} neurons with elapsed time {round(elapsed_time, 2)} seconds
       '''

def prediction_test(model, test_set_scalled, scaller, test_dataset, *train_tuple, start_time):
    # test model and prediction
    X_train = train_tuple[0][0]
    Y_train = train_tuple[0][1]
    X_test, Y_test = create_time_steps_data_group(TIME_STEPS, test_set_scalled)
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaller.inverse_transform(predicted_stock_price)
    ELAPSED_TIME = time() - start_time
    print(f'predicted values: {predicted_stock_price}')
    print(f'train loss: {model.evaluate(X_train, Y_train)}')
    print(f'test loss: {model.evaluate(X_test, Y_test)}')
    print(f'model summary: {model.summary()}')
    print(f'model processing time: {ELAPSED_TIME}')

    # transform prediction output into one dim array, the same with test data
    predicted_data = convert_to_one_dim_array(predicted_stock_price)
    test_data = convert_to_one_dim_array(test_dataset.values)
    print(f'test_dataset count: {test_data.shape[0]}')
    print(f'predicted values: {predicted_data.shape[0]}')

    # plot for 1 output
    predict_range = range(TIME_STEPS, TIME_STEPS + len(predicted_data))
    print(f'predict range: {predict_range}')
    plt.plot(test_data, color='blue', label='Real values')
    plt.plot(predict_range, predicted_stock_price, color='red', label='Prediction')
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.legend()
    plt.title(get_test_plot_title(EPOCHS, TIME_STEPS, NEURON_UNITS, ELAPSED_TIME))
    plt.show()


def append_predicted_value(X_total, predicted):
    temp_input = list(X_total)
    temp_input = temp_input[-1].tolist()
    temp_input = temp_input[1:]
    temp_input.append([predicted])
    X_total = X_total[1:]
    return np.insert(X_total, len(X_total), temp_input, axis=0)


if __name__ == "__main__":

    # preparing data for usage - remove null etc.
    df = transform_data_from_csv(FILE_NAME, ',', [DATE, CLOSE])
    print(f'dataframe: {df}')

    # split into training and test set
    train_set_size = get_train_set_size(df, TRAIN_SET_PERCENT)
    training_dataset = df.iloc[:train_set_size, 1:2]
    test_dataset = df.iloc[train_set_size:, 1:2]
    print(f'training_dataset: {training_dataset}')
    print(f'test_dataset: {test_dataset}')

    # scalling data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scalled = sc.fit_transform(training_dataset.values)
    test_set_scalled = sc.fit_transform(test_dataset.values)

    # creating and training model
    X_train, Y_train = create_time_steps_data_group(TIME_STEPS, training_set_scalled)
    # model can be kept in a file
    # model = get_model(X_train, NEURON_UNITS, OUTPUTS, MODEL_FILEPATH)
    model = create_model(X_train, NEURON_UNITS, output_neuron_units=OUTPUTS)

    start_time = time()
    train_model(model, X_train, Y_train, EPOCHS, BATCH_SIZE)

    # uncomment if show prediction test
    # prediction_test(model, test_set_scalled, sc, test_dataset, (X_train, Y_train), start_time=start_time)


    print(f'**************      {PREDICTION_DAYS} DAYS PREDICTION       ****************')

    total_dataset = pd.concat((training_dataset, test_dataset), axis=0)
    total_dataset_scalled = sc.fit_transform(total_dataset)
    print(f'total dataset: {total_dataset}')

    X_total, Y_total = create_time_steps_data_group(TIME_STEPS, total_dataset_scalled)

    predicted_values = []
    for i in range(0, PREDICTION_DAYS):
        predictions = model.predict(X_total)
        last_predicted_value = convert_to_one_dim_array(predictions)[-1]
        predicted_values.append(last_predicted_value)
        # every day we predict with one more value, so we need to add it to X_total
        X_total = append_predicted_value(X_total, last_predicted_value)

    ELAPSED_TIME = time() - start_time
    predicted_values = sc.inverse_transform([predicted_values])
    predicted_values = convert_to_one_dim_array(predicted_values)
    predicted_values = np.array(predicted_values).transpose()
    print('************ AFTER PREDICTION LOOP **************')
    print(f'predicted_values: {predicted_values}')
    print(f'train data loss: {model.evaluate(X_train, Y_train)}')
    print(f'real data loss: {model.evaluate(X_total, Y_total)}')
    print(f'model summary: {model.summary()}')

    prediction_range = range(len(total_dataset), len(total_dataset) + PREDICTION_DAYS)
    plt.plot(prediction_range, predicted_values, color='red', label='Prediction values')
    plt.plot(total_dataset.values, color='blue', label='Real values')
    plt.xlim(left=1500)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(get_plot_title(EPOCHS, PREDICTION_DAYS, NEURON_UNITS, ELAPSED_TIME))
    plt.legend()
    plt.show()