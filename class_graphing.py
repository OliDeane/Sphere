from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_broken_bar_list(gaze_object_list):

  output_list = []
  for i in gaze_object_list:
    output_list.append((i,1))

  return output_list

def prepare_for_graphing(data_set, unique_cats): # Prepare data for graphing


    unique_cats_2 = unique_cats
    data_set_2 = data_set
    graph_list = []
    for category in unique_cats:
        graph_list.append([i for i, e in enumerate(data_set) if e == category])

    graphing_dict = {}
    for i in range(0,len(unique_cats)):
        graphing_dict.update( {unique_cats[i]: get_broken_bar_list(graph_list[i])} )

    return graphing_dict

def generate_graph(graphing_dict, unique_cats, system_data, coder_identity,ax):

  y_tick_labels = [] # For the Ytick Labels
  iterations = []

  color_list = ['purple', 'olive', 'brown', 'cyan', 'pink']*50
  color_dict = {}
  count = 0
  # Get colour for the graphing
  for i in unique_cats:
    if i == 'p_stand':
      color_dict.update( {i: 'green'} )
    elif i == 'p_sit':
      color_dict.update( {i:'red'})
    elif i == 'p_lie':
      color_dict.update( {i:'orange'})
    elif i == 't_stand_sit':
      color_dict.update( {i:'blue'})
    else:
      color_dict.update( {i: color_list[count]} )
    count += 1


  count = 0
  for i in graphing_dict:
    count += 1
    ax.broken_barh(graphing_dict[i], ((count*5), 5), facecolors='tab:{}'.format(color_dict[i])) 

    y_tick_labels.append(count*5+2.5)
    iterations.append(count*5)


  ax.set_ylim(5, len(unique_cats)*5+5)
  ax.set_xlim(0, len(system_data))
  ax.set_xlabel('Second')
  ax.set_ylabel(coder_identity)

  ax.set_yticks(y_tick_labels)
  ax.set_yticklabels(unique_cats)
  ax.grid(False)

  #iterations = [10,15,20,25,30] # Add in lines around bars
  for i in iterations:
    ax.axhline(y=i,linewidth=1, color='gray', alpha = 0.3)

  return ax

def get_activity_list(second_limit):

    public_data_path, metadata_path = define_paths()

    x_df, y_df = load_XY_dfs(public_data_path)
    y_df = y_df.drop(['start', 'end'], axis =1)

    # Create a list of y variables for each annotator
    X = x_df.values
    Y = y_df.values

    # train_x, test_x, train_y, test_y = train_test_split(X,Y)
    # train_x, test_x, train_y, test_y = simple_impute(train_x, test_x, train_y, test_y)
    X,Y = impute_none_labels(X, Y)

    label_index = np.argmax(Y, axis = 1)
    labels = [y_df.columns[i] for i in label_index][0:second_limit]

    return labels, list(set(labels)), label_index


if __name__ == '__main__':
    second_limit = 500
    labels, unique_labels, label_index = get_activity_list(second_limit)
    graphing_dict = prepare_for_graphing(labels, unique_labels)

    #Initiate Graph
    fig, plotter = plt.subplots(1, figsize=(8,3), sharex = True, gridspec_kw = {'hspace' : 0.05})
    plt.title(f" Ground Truth activity for {second_limit} second time sequence ")
    plt.rc('font', family='sans serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.tight_layout()

    # Fill graph with subplots - one for each coder
    generate_graph(graphing_dict, unique_labels, labels, 'Activity', plotter)

    plt.show()

