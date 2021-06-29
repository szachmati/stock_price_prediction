import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, save_model
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from utils import transform_data_from_csv, convert_to_one_dim_array, file_exists

TIME_STEPS = 183
TRAIN_SET_PERCENT = 80
NEURON_UNITS = 50
EPOCHS = 1
BATCH_SIZE = 30
FILE_NAME = 'ing.csv'
OUTPUTS = 1
MODEL_FILEPATH = 'lstm_stock_prediction_ing.model'


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
    if not file_exists(MODEL_FILEPATH):
        model = create_model(X_train, neuron_units, output_neuron_units=outputs)
        model.save(MODEL_FILEPATH)
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


if __name__ == "__main__":

    # preparing data for usage - remove null etc.
    df = transform_data_from_csv(FILE_NAME, ',', ['Data', 'Zamknięcie'])
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
    # model = get_model(X_train, NEURON_UNITS, OUTPUTS, MODEL_FILEPATH)
    model = create_model(X_train, NEURON_UNITS, output_neuron_units=OUTPUTS)
    train_model(model, X_train, Y_train, EPOCHS, BATCH_SIZE)

    # test model and prediction
    X_test, Y_test = create_time_steps_data_group(TIME_STEPS, test_set_scalled)
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(f'predicted values: {predicted_stock_price}')
    print(f'train loss: {model.evaluate(X_train, Y_train)}')
    print(f'test loss: {model.evaluate(X_test, Y_test)}')
    print(f'model summary: {model.summary()}')

    # transform prediction output into one dim array, the same with test data
    predicted_data = convert_to_one_dim_array(predicted_stock_price)
    test_data = convert_to_one_dim_array(test_dataset.values)

    # plot for 1 outputs
    print(f'test_dataset count: {test_dataset.shape[0]}')
    predict_range = range(TIME_STEPS, TIME_STEPS + len(predicted_data))
    print(f'predict range: {predict_range}')
    plt.plot(test_dataset.values, color='blue', label='Real values')
    plt.plot(predict_range, predicted_stock_price, color='red', label='Prediction')
    plt.xlabel('Time in days')
    plt.ylabel('Predicted close price')
    plt.legend()
    plt.show()

    #
