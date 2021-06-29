
import pandas as pd
import os.path

def transform_data_from_csv(file_path, separator, columns):
    return pd.read_csv(file_path, separator, usecols=columns).dropna().iloc[::-1]


def convert_to_one_dim_array(two_dim_array):
    return two_dim_array.ravel()


def file_exists(file_path):
    return os.path.isfile(file_path)

######## tests #########

# print(transform_data_from_csv('ing.csv', ',', ['Otwarcie', 'Zamknięcie']))
# print(convert_to_one_dim_array(array([[1, 2, 3], [4, 5, 6]])))
# print(file_exists('utils.py'))