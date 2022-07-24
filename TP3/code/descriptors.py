#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

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
#   Here you can define usefull functions to be used in the main
#


def PCA(points):
    centroid = np.mean(points, axis=0, keepdims=True)

    cov_matrix = 1/len(points) * (centroid - points).T@(centroid - points)

    

    return np.linalg.eigh(cov_matrix)


def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

    tree = KDTree(cloud_points)
    nearest_neighbors = tree.query_radius(query_points, radius)
    # alternatively
    # _, nearest_neighbors = tree.query(query_points, k=30)
    for i, neighbors in enumerate(tqdm(nearest_neighbors)):
        eigenvalues, eigenvectors = PCA(cloud_points[neighbors])
        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    epsilon = 1e-6

    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    lambda3, lambda2, lambda1 = all_eigenvalues[:, 0], all_eigenvalues[:, 1], all_eigenvalues[:, 2]
    lambda1 += epsilon  # only denominator
    normals = all_eigenvectors[:, :, 0]

    verticality = 2 * np.arcsin(np.abs(normals@np.array([[0], [0], [1]]))) / np.pi
    verticality = verticality.reshape(len(verticality))
    linearity = 1 - lambda2/lambda1
    planarity = (lambda2 - lambda3)/lambda1
    sphericity = lambda3/lambda1

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(
            cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply',
                  (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        query = cloud
        vert, lin, plan, spher = compute_features(query, cloud, 0.5)

        # Save cloud with normals
        # write_ply('../Lille_street_small.ply',
        # (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
        write_ply('../Lille_small_feat.ply', [query, vert, lin, plan, spher], [
                  'x', 'y', 'z', 'vert', 'lin', 'plan', 'spher'])
