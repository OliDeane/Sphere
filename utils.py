""" General functions for loading in the data """
import os
import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer


def define_paths():
    """ Gets paths to public data and meta data """
    nb_dir = os.path.split(os.getcwd())
    nb_dir2 = nb_dir[0]+'/'+nb_dir[1]
    if nb_dir not in sys.path:
        sys.path.append(nb_dir)
    if nb_dir2 not in sys.path:
        sys.path.append(nb_dir2)

    public_data_path = nb_dir2+'/data' 
    metadata_path = nb_dir2+'/data/metadata'

    return public_data_path, metadata_path

def load_XY_dfs(public_data_path):
    """Get feature and targets data combined acorss all sequences from
    the training dataset. Returns as separate dfs """ 
    x_df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, '00001'))
    for file_id in [2,3,4,5,6,7,8,9,10]:
        filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

        new_df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, filename))
        x_df = x_df.append(new_df)

    y_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, '00001'))
    for file_id in [2,3,4,5,6,7,8,9,10]:
        filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

        new_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename))
        y_df = y_df.append(new_df)

    return x_df, y_df

def load_sequence(file_id, public_data_path):
    filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

    df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, filename))
    data = df.values
    target = np.asarray(pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename)))[:, 2:]

    return data, target

def load_sequences(file_ids, public_data_path):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id, public_data_path)

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)

def load_train_test_arrays(public_data_path):
    """ Returns train and test data - 
    combined across all training/testing sets """
    train_x, train_y = load_sequences([1, 2, 3, 4, 5, 6, 7, 8], public_data_path)
    test_x, test_y = load_sequences([9, 10], public_data_path)

    return train_x, test_x, train_y, test_y

def simple_impute(train_x, test_x, train_y, test_y):
    """ 
    Impute missing training data with simple impute.
    Also, not all data is annotated, so we select only the annotated rows
    """
    imputer = SimpleImputer()
    imputer.fit(train_x)

    train_x = imputer.transform(train_x)
    test_x = imputer.transform(test_x)

    # Select only annotated rows
    train_y_has_annotation = np.isfinite(train_y.sum(1))
    train_x = train_x[train_y_has_annotation]
    train_y = train_y[train_y_has_annotation]

    test_y_has_annotation = np.isfinite(test_y.sum(1))
    test_x = test_x[test_y_has_annotation]
    test_y = test_y[test_y_has_annotation]

    return train_x, test_x, train_y, test_y

def impute_data(X,Y):
    imputer = SimpleImputer()
    imputer.fit(X)
    X = imputer.transform(X)

    # Select only annotated rows
    y_has_annotation = np.isfinite(Y.sum(1))
    X = X[y_has_annotation]
    Y = Y[y_has_annotation]

    return X,Y

def impute_none_labels(x_data, y_data):
    y_has_annotation = np.isfinite(y_data.sum(1))
    x_data = x_data[y_has_annotation]
    y_data = y_data[y_has_annotation]
    return x_data, y_data

def get_acceleration_df(df):
    """ Given a full dataframe, this removes any non-acceleration
    columns (As acceleration seems the most predictive feature) """

    cols = [col for col in df.columns if 'acceleration_' in col]
    acc_df = df[cols]

    return acc_df

def get_subset_df(df, label = None, multiple_labels = None):
    if multiple_labels:
        cols = [col for col in df.columns if label[0] in col or label[1] in col]
        new_df = df[cols]
    else:
        cols = [col for col in df.columns if label in col]
        new_df = df[cols]

    return new_df

def complex_impute(x_df):

    """Imputes rooms and acc and removes pir features """

    # create three different dataframes - acceleration, videos/rssi, pir
    acc_df = get_subset_df(x_df, label = 'acceleration')
    rooms_df = get_subset_df(x_df, label = ['video','rssi'], multiple_labels = True)
    pir_df = get_subset_df(x_df, label = 'pir')

    # total_df = list(acc_df.columns) + list(rooms_df.columns) + list(pir_df.columns)

    # fill acc_df with forwardfill
    acc_df = acc_df.ffill(axis = 0)

    # fill rooms_df with zerosa
    rooms_df = rooms_df.fillna(0)

    # Ignore pir for now. 

    frames = [acc_df, rooms_df]
    return pd.concat(frames, axis=1)

def forward_fill_impute(df):
    """ Imputes with forward fill, but only for when lkess than 3 NaN values in a row. Fill remainder with zeros """
    df = df.fillna(value=None, method='ffill', axis=None, limit=2, downcast=None).fillna(0)
    return df

def interpolate_impute(df):

    """ Replace with mean up to a limit of 3 NaN values in a row. The issue here is that
    if the person leaves a room (for e.g.), then this will still impute the following two data points. """
    df = df.interpolate(limit=3).fillna(0)
    return df




def complex_impute(x_df):
    """Imputes rooms and acc and removes pir features - has been improved iupon by other functions"""

    # create three different dataframes - acceleration, videos/rssi, pir
    acc_df = get_subset_df(x_df, label = 'acceleration')
    rooms_df = get_subset_df(x_df, label = ['video','rssi'], extra = True)
    pir_df = get_subset_df(x_df, label = 'pir')

    # total_df = list(acc_df.columns) + list(rooms_df.columns) + list(pir_df.columns)

    # fill acc_df with forwardfill
    acc_df = acc_df.ffill(axis = 0)

    # fill rooms_df with zerosa
    rooms_df = rooms_df.fillna(0)

    # Ignore pir for now. 

    frames = [acc_df, rooms_df]
    return pd.concat(frames, axis=1)
