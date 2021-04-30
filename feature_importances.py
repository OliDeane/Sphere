from utils import *
# Tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import pca_script
import operator
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def process_importances(x_df, importance, min_rank = 100):
    # importance = model.feature_importances_
    dic = {}
    # summarize feature importance
    for i,v in zip(x_df.columns,importance):
        dic[i] = v

    sorted_importances_dict = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    importance_features = list(sorted_importances_dict.keys())[0:min_rank]
    importance_values = list(sorted_importances_dict.values())[0:min_rank]

    plt.barh(importance_features, importance_values)
    plt.show()

    return importance_features, importance_values

def xgboost_importances(X,y):
    model = XGBClassifier()
    model.fit(X, y)
    importance = model.feature_importances_
    return importance

def permutation_importances(X,y,x_df):
    model = KNeighborsClassifier()
    # fit the model
    model.fit(X, y)

    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    importance_features, importance_values = process_importances(x_df, importance)

def randomforest_importances(X,y):

    model = RandomForestClassifier()
    model.fit(X, y)
    importance = model.feature_importances_
    return importance

def get_and_plot_imp_features(X,y,x_df, min_rank = 100):
    importance = randomforest_importances(X, y)
    importance_features, importance_values = process_importances(x_df, importance, min_rank = min_rank)
    return importance_features

if __name__ == '__main__':

    # Get and impute initial data
    public_data_path, metadata_path = define_paths()
    x_df, y_df = load_XY_dfs(public_data_path)
    y_df = y_df.drop(['start', 'end'], axis =1)
    x_df = forward_fill_impute(x_df)
    X, y = impute_none_labels(x_df.values, y_df.values)

    # Get argmax of y lables
    y_df = pd.DataFrame(data = y, columns = y_df.columns)
    y_df = y_df.fillna(0)
    y = np.argmax(y_df.values,axis=1)

    #Get feature importance
    x_df = get_and_plot_imp_features(X,y,x_df, min_rank = 100)
"""
Get into a csv where all y labels are there plus the label encoder and the confidence
of the argmax

"""
