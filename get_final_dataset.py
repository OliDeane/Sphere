from utils import *
import pandas as pd
import pca_script
from feature_importances import * 
from sklearn.preprocessing import MinMaxScaler

# Load in initial data
public_data_path, metadata_path = define_paths()
x_df, y_df = load_XY_dfs(public_data_path)
y_df = y_df.drop(['start', 'end'], axis =1)

# Impute data
x_df = forward_fill_impute(x_df)
X, y = impute_none_labels(x_df.values, y_df.values)

# Get Argmax of y
y_df = pd.DataFrame(data = y, columns = y_df.columns)
y_df = y_df.fillna(0)
y = np.argmax(y_df.values,axis=1)

# Also scale X and create new dataframe x_df with new imputed X array
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
x_df = pd.DataFrame(data=X, columns=x_df.columns)

# Get list of important features
imp_features = get_and_plot_imp_features(X,y,x_df, min_rank = 200)
x_df = x_df[imp_features]

# Then get the principal components of the new X data
pca_X = pca_script.do_pca(x_df.values, variance = 0.99, show_plot=True)
x_df = pd.DataFrame(data=pca_X,columns=list(range(pca_X.shape[1])))

processed_df = pd.concat([x_df, y_df.reindex(x_df.index)], axis=1)
processed_df['encoded label'] = list(y) # Create column for the encoded label i the y_df

print(processed_df.shape)
print(processed_df.columns)

processed_df.to_csv('processed_dataset.csv', index=False)
