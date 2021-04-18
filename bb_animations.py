import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import isnan
from utils import *

def get_bb_df(x_df):
    """ Creates DF of only bounding box features """ 
    cols = [col for col in x_df.columns if '_bb_2d' in col and 'mean' in col]
    bb_df = x_df[cols]
    bb_df = bb_df.reset_index()

    return bb_df

def get_bb_coords(bb_df):
    """ Grabs the x,y coords (bottom right and top left) for generating the bounding boxes """
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

    return bb_df, bb_values

def plot_bb_animation(bb_values):
    """ Plots bounding boxes in loop to create animation  """
    fig, ax = plt.subplots()

    ax.set_xlim([0,1280])
    ax.set_ylim([0,720])
    plt.gca().invert_yaxis()
    plt.ion()

    for box in bb_values:
        plt.cla()

        ax.set_xlim([0,852])
        ax.set_ylim([0,480])
        plt.gca().invert_yaxis()

        ax.add_patch(Rectangle((box[2], box[3]), box[0]-box[2] , box[1]-box[3]))
        ax.text(650, 420, box[4], style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        plt.show()
        plt.pause(0.1)
        

    plt.ioff()    
    plt.show()

if __name__ == "__main__":

    public_data_path, metadata_path = define_paths()

    x_df, y_df = load_XY_dfs(public_data_path)

    bb_df = get_bb_df(x_df)

    bb_df, bb_values = get_bb_coords(bb_df)
    
    plot_bb_animation(bb_values)

