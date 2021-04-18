from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_cum_cov(pca, data, threshold):
    """ Plot the cumulative covariance """
    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = np.arange(1, data.shape[1] + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, data.shape[1] + 1, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.text(0.5, 0.85, f'{threshold} cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

def do_pca(data_rescaled, variance, show_plot):
    if show_plot:
        plot_cum_cov(PCA().fit(data_rescaled), data_rescaled, threshold=variance)
        
    pca = PCA(n_components = variance)
    pca.fit(data_rescaled)
    return pca.transform(data_rescaled)

if __name__ == '__main__':
    
    # Load in the dataframe datasets
    public_data_path, meta_data_path = define_paths()
    x_df, y_df = load_XY_dfs(public_data_path)

    # Only use the acceleration df
    x_df = get_acceleration_df(x_df)
    y_df = y_df.drop(['start', 'end'], axis =1)

    # train test split and simple impute the datasets
    train_x, test_x, train_y, test_y = train_test_split(x_df.values,y_df.values)
    train_x, test_x, train_y, test_y = simple_impute(train_x, test_x, train_y, test_y)

    # Run PCA on the trainX data
    # scale the data to the range between 0 and 1 before using PCA
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(train_x)
    pca_data = do_pca(data_rescaled, variance = 0.99)



