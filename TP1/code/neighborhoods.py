#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = [] 
    for center in queries:
        neighborhoods.append([])
        for point in tqdm(supports):
            if np.linalg.norm(point-center) < radius:
                neighborhoods[-1].append(point)
    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for center in queries:
        distances = []
        for point in tqdm(supports):
            distances.append([np.linalg.norm(point-center)])
        indexes = np.argsort(distances)[:k+1] # pour ne pas prendre en compte center point
        neighborhoods.append(supports[indexes])
    return neighborhoods





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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000
        radius = 0.2

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        t = [time.time()]
        leaf_sizes = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 20000, 30000, 50000, 100000]
        time_list = [[] for i in leaf_sizes]
        for i in range(10):
            for j, leaf_size in enumerate(tqdm(leaf_sizes)):
                tree = KDTree(points, leaf_size=leaf_size)
                neighbors = tree.query_radius(queries, radius)
                t.append(time.time())
                time_list[j].append(t[-1]-t[-2])
                # print(f"Leaf_size = {leaf_size}, time = {time_list[j]}")
        for i, timing in enumerate(time_list):
            time_list[i] = np.mean(timing)
        plt.plot(leaf_sizes, time_list)
        plt.title("Time spent for each leaf_size")
        plt.show()

    if True:
        leaf_size = 20000
        #print(points.shape[0]*0.2/3600)

        # Define the search parameters
        num_queries = 1000
        radiuses = np.linspace(0.2, 2, 20)
        
        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        
        tree = KDTree(points, leaf_size=leaf_size)
        t = [time.time()]
        time_list = [[] for i in radiuses]
        for _ in range(10):
            for j, radius in enumerate(tqdm(radiuses)):
                neighbors = tree.query_radius(queries, radius.astype(int))
                t.append(time.time())
                time_list[j].append(t[-1]-t[-2])
        for i, timing in enumerate(time_list):
            time_list[i] = np.mean(timing)
        print(time_list[0])
        plt.plot(radiuses, time_list)
        plt.title("Time spent for each radius")
        plt.show()

