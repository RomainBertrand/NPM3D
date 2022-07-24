#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 02/02/2018
#


# Import time package
import time

# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh

# Import functions to read and write ply files
from ply import read_ply


# Hoppe surface reconstruction
def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    tree = KDTree(points, leaf_size=20)

    dimensions = []
    for i in range(3):
        dimensions.append(min_grid[i] + size_voxel[i]
                          * np.arange(grid_resolution + 1))

    mesh = np.meshgrid(dimensions[0], dimensions[1], dimensions[2])

    nodes = np.stack(mesh, 3).reshape(-1, 3)

    _, closest_points_indexes = tree.query(nodes, 1)
    closest_points_indexes = closest_points_indexes.reshape(len(nodes))
    hoppe = np.sum(normals[closest_points_indexes] *
                   (nodes - points[closest_points_indexes]), axis=1)
    scalar_field[:, :] = hoppe.reshape(np.shape(scalar_field))
    return


# IMLS surface reconstruction
def compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn):
    tree = KDTree(points)

    dimensions = []
    for i in range(3):
        dimensions.append(min_grid[i] + size_voxel[i] * np.arange(grid_resolution + 1))

    mesh = np.meshgrid(dimensions[0], dimensions[1], dimensions[2])

    nodes = np.stack(mesh, 3).reshape(-1, 3)

    knn = 30
    _, closest_points_indexes = tree.query(nodes, knn)
    closest_points_indexes = closest_points_indexes.reshape((len(nodes), knn))

    xp_i = nodes[:, np.newaxis, :] - points[closest_points_indexes]
    h = 0.001
    theta_i = np.exp(-np.linalg.norm(xp_i, axis=2)**2/h**2)
    theta_i = np.clip(theta_i, 1e-8, np.inf)
    n_ixp_i = np.sum(normals[closest_points_indexes] * xp_i, axis=2)
    imls = np.sum(n_ixp_i * theta_i, axis=1) / np.sum(theta_i, axis=1)
    scalar_field[:, :] = imls.reshape(np.shape(scalar_field))

    return


if __name__ == '__main__':

    t0 = time.time()

    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    # Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

    # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

    # grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 100  # 100
    size_voxel = np.array([(max_grid[0]-min_grid[0])/grid_resolution, (max_grid[1] -
                          min_grid[1])/grid_resolution, (max_grid[2]-min_grid[2])/grid_resolution])

    # Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros(
        (grid_resolution+1, grid_resolution+1, grid_resolution+1), dtype=np.float32)

    # Compute the scalar field in the grid
    # compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel)
    compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, 30)

    # Compute the mesh from the scalar field based on marching cubes algorithm
    # not lewiner in Python 3.9
    verts, faces, normals_tri, values_tri = measure.marching_cubes(
        scalar_field, level=0.0, spacing=(size_voxel[0], size_voxel[1], size_voxel[2]))

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(file_obj='../bunny_mesh_imls_100.ply', file_type='ply')

    print("Total time for surface reconstruction : ", time.time()-t0)
