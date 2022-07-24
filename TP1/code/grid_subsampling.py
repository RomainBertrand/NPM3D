#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, dl):

    # YOUR CODE
    # print(np.amax(points, 1))
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)
    print("got limits")
    
    x_axis = np.arange(min_x, max_x, dl)
    y_axis = np.arange(min_y, max_y, dl)
    z_axis = np.arange(min_z, max_z, dl)

    decimated_points = []
    decimated_colors = []
    decimated_labels = []

    # grid_axis = np.meshgrid(x_axis, y_axis, z_axis)
    grid = np.zeros((int((max_x - min_x)//dl) + 1, int((max_y-min_y)//dl) + 1, int((max_z-min_z)//dl + 1)))
    print(np.shape(grid))
    for i, point in enumerate(tqdm(points)):
        if grid[int((point[0]-min_x)//dl), int((point[1]-min_y)//dl), int((point[2]-min_z)//dl)] == 0:
            decimated_points.append(point)
            decimated_colors.append(colors[i])
            decimated_labels.append(labels[i])
            grid[int((point[0]-min_x)//dl), int((point[1]-min_y)//dl), int((point[2]-min_z)//dl)] = 1





    return np.array(decimated_points), np.array(decimated_colors), np.array(decimated_labels)





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    dl = 0.2

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, dl)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    print('Done')
