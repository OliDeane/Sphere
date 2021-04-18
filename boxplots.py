from class_graphing import *
from os import listdir
from pandas import read_csv
import matplotlib.pyplot as plt
import os

# load sequence for each subject, returns a list of numpy arrays
def load_dataset(prefix=''):
	subjects = list()
	directory = prefix + '/HAR/'
	for name in listdir(directory):
		filename = directory + '/' + name
		if not filename.endswith('.csv'):
			continue
		df = read_csv(filename, header=None)
		# drop row number
		values = df.values[:, 1:]
		subjects.append(values)
	return subjects

# returns a list of dict, where each dict has one sequence per activity
def group_by_activity(subjects, activities):
	grouped = [{a:s[s[:,-1]==a] for a in activities} for s in subjects]
	return grouped
 
# calculate total duration in sec for each activity per subject and plot
def plot_durations(grouped, activities, unique_labels):
	# calculate the lengths for each activity for each subject

    freq = 1
    durations = [[len(s[a])/freq for s in grouped] for a in activities]
    plt.boxplot(durations, labels=unique_labels)
    ax = plt.gca()
    ax.set_ylabel('Seconds')
    ax.tick_params(axis='x', labelrotation=45)
    plt.show()

def get_all_targetDF_sequences(public_data_path):

    starter_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, '00001'))
    starter_df = starter_df.drop(['start', 'end'], axis =1)
    sequences = [starter_df]
    for file_id in [2,3,4,5,6,7,8,9,10]:
        filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

        new_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename))
        new_df = new_df.drop(['start', 'end'], axis =1)

        # Impute y data
        y_has_annotation = np.isfinite(new_df.sum(1))
        new_df = new_df[y_has_annotation]
        sequences.append(new_df)
    
    return sequences

def create_single_label_col(sequences):
    subjects = []
    for seq in sequences:
        label_index = np.argmax(seq.values, axis = 1)
        seq['max_label'] = label_index
        subjects.append(seq.values)
    
    return subjects

if __name__ == '__main__':
    # load in each sequence
    public_data_path, metadata_path = define_paths()


    sequences = get_all_targetDF_sequences(public_data_path)
    unique_labels = sequences[0].columns.tolist()

    # Create single column of 
    subjects = create_single_label_col(sequences)

    # Put into a list of arrays
    activities = [i for i in range(0,20)]
    grouped = group_by_activity(subjects, activities)
    plot_durations(grouped, activities, unique_labels)
