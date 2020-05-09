import os
import sys
import warnings

# off warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# data preprocessing
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer

# loading model from .h5 file
from tensorflow.keras.models import load_model


# parse input path argument
try:
    input_path = sys.argv[1]
except Exception:
    raise FileNotFoundError('No directory specified.')

# parse output-file argument
try:
    output_file = sys.argv[2]
except Exception:
    output_file = 'output.txt'


def preprocess_arguments(input_path:str) -> bool:
    '''
    This function checks command-line arguments.
    Return True, if input_path exists and input_path is folder.
    Returns False, if input_path is .csv file.
    '''

    # input path is folder
    if len(input_path.split('\\')) != 1:
        # if folder is not exists
        if not os.path.exists(input_path):
            raise NotADirectoryError('Input directory does not exist.')

        is_folder = True
    # if path is file:
    else:
        print('we are here')
        # if file is not coma-sepparated-file
        if input_path.split('.')[-1] != 'csv':
            raise TypeError('Input file is not .csv file.')

        # file is not exist
        try:
            open(input_path, 'r')
        except Exception:
            raise FileExistsError('Input file does not exist.')

        is_folder = False

    return is_folder


def read_data(input_path:str, is_folder:bool) -> np.ndarray:
    '''This function reads data from .csv file or folder with .csv files.'''

    # list for data
    data = list()

    if is_folder:
        
        # read each file in folder
        for file in os.listdir(input_path):

            if file.split('.')[-1] != 'csv':
                raise TypeError('File is not .csv.')

            # full name of file
            file = os.path.join(input_path, file)

            # read data with pandas and add this to list
            data.append(pd.read_csv(file)['activity'].values)
    else:
        # read data with pandas and add this to list
        data.append(pd.read_csv(input_path)['activity'].values)

    # returns np.ndarray
    return np.array(data)


def load_model_normalizer() -> tuple:
    '''
    This function loads Keras model and Sklearn normalizer from 'model.h5' and 'normalizer.saved'
    '''

    model = load_model('model.h5')

    normalizer = joblib.load('normalizer.saved')

    return model, normalizer


def predict(input_path:str) -> np.ndarray:
    '''This function reads data, preprocesses it and predicts probability of schizophrenia'''

    # reading data from input path
    data = read_data(input_path, preprocess_arguments(input_path))

    # loading model and normalizer
    model, normalizer = load_model_normalizer()

    # normalize data
    data = normalizer.transform(data)

    # reshaping data for Keras model prediction
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

    # model's predictions
    predictions = model.predict(data)

    return np.reshape(predictions, (predictions.shape[0], ))


def output(input_path:str, output_file:str) -> None:
    '''This function writes predictions to output_file'''

    # model's predictions
    predictions = predict(input_path)

    # output
    with open(output_file, 'w') as f:
        f.write(' '.join([str(value) for value in predictions]))

if __name__ == '__main__':
    output(input_path, output_file)











