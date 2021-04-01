# For number crunching
import numpy as np
import pandas as pd

# For visualisation
import matplotlib.pyplot as pl 

# For prediction 
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import isnan
from utils import *

public_data_path, metadata_path = define_paths()
x_df, y_df = load_XY_dfs(public_data_path)



# # First load in as pandas dataframe (to see all columsn etc.)
# x_df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, '00001'))
# for file_id in [2,3,4,5,6,7,8,9,10]:
#     filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

#     new_df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, filename))
#     x_df = x_df.append(new_df)
 
# y_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, '00001'))
# for file_id in [2,3,4,5,6,7,8,9,10]:
#     filename = str(file_id).zfill(5) # zfill fills with 5 zeros at the beginning of the string

#     new_df = pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename))
#     y_df = y_df.append(new_df)


cols = [col for col in x_df.columns if '_bb_2d' in col and 'mean' in col]
bb_df = x_df[cols]
bb_df = bb_df.reset_index()


bb_values = []
for i in range(0,len(bb_df)):
    if not isnan(bb_df['video_living_room_bb_2d_br_x_mean'][i]):
        room = 'living_room'
    elif not isnan(bb_df['video_kitchen_bb_2d_br_x_mean'][i]):
        room = 'kitchen'
    elif not isnan(bb_df['video_hallway_bb_2d_br_x_mean'][i]):
        room = 'hallway'
    else:
        room = 'No Data'
    
    if room != 'No Data':
        bb_values.append([bb_df[f'video_{room}_bb_2d_br_x_mean'][i], bb_df[f'video_{room}_bb_2d_br_y_mean'][i], bb_df[f'video_{room}_bb_2d_tl_x_mean'][i],	bb_df[f'video_{room}_bb_2d_tl_y_mean'][i], room])
    else:
        bb_values.append([0,0,0,0,room])


bb_df['Room Label'] = bb_values


fig, ax = plt.subplots()

ax.set_xlim([0,1280])
ax.set_ylim([0,720])
plt.gca().invert_yaxis()
# plt.show()
plt.ion()


for box in bb_values:
    plt.cla()

    ax.set_xlim([0,852])
    ax.set_ylim([0,480])
    plt.gca().invert_yaxis()

    # box_X = df2.filter(regex='x_mean').to_numpy()[i].tolist()
    # box_Y = df2.filter(regex='y_mean').to_numpy()[i].tolist()
    ax.add_patch(Rectangle((box[2], box[3]), box[0]-box[2] , box[1]-box[3]))
    ax.text(650, 420, box[4], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()
    plt.pause(0.1)
    

plt.ioff()    
plt.show()

