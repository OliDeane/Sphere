from utils import *
# Tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import pca_script
import matplotlib.pyplot as plt

public_data_path, metadata_path = define_paths()
x_df, y_df = load_XY_dfs(public_data_path)
y_df = y_df.drop(['start', 'end'], axis =1)

X,Y = impute_data(x_df.values, y_df.values)
