########################################################################################################################
#                                  DEEP NEURAL NETWORKS FOR AERODYNAMIC LOW PREDICTION                                 #
#                                             Student: Albert García Plaza                                             #
#                                              Supervisor: Eduardo Alonso                                              #
#                                               INM363 Individual project                                              #
#                              MSc in Artificial Intelligence - City, University of London                             #
########################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import stats
import re


def sorted_alphanumeric(data):
    """
    Given a list of numeric values passed as strings, it returns the same list but alphanumerically sorted (e.g. '8',
    '9', '10', '11', ...)
    :param data: List of strings to be sorted
    :return: List of strings corresponding to the input list alphanumerically sorted
    """
    # Convert string of numbers to int types
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # Read all numerical characters in each string value
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]

    # Return the list alphanumerically sorted following the natural numbers order
    return sorted(data, key=alphanum_key)


def nan2mean(a, index):
    """
    Convert non-assigned values to the average flow field value
    """
    pointer = len(a)
    for i in range(1, pointer):
        if a[index - i] is not None:
            left_bound_id = i
            left_bound = a[index - i]
            break
    for i in range(1, pointer):
        if a[index + i] is not None:
            right_bound_id = i
            right_bound = a[index + i]
            break
    n_gap = right_bound_id + left_bound_id
    val_gap = right_bound - left_bound
    new_val = int((left_bound + val_gap / n_gap * left_bound_id) / grid_resolution) * grid_resolution

    return new_val


def openFOAM2Npy():
    """
    Take all simulation output and geometry/mesh files from OpenFOAM located at the /output/ folder and convert them
    to .npy files with shape (2, 600, 600) -two channels: first for velocity along X, and second for velocity along Y
    :return:
    """

    # Read the files located at /output/ folder
    output_files = os.listdir('outputs')
    airfoil_data = [file for file in output_files if file.endswith('Centres')]      # pick mesh centres values
    airfoil_pts = [file for file in output_files if file.endswith('.dat')]          # pick the airfoil geometry files
    output_files = [file for file in output_files if not file.endswith('.dat')]
    output_files = [file for file in output_files if not file.endswith('Centres')]  # pick simulation results files

    # Number of datapoints read
    n_files = len(output_files)

    # Sort all lists alphanumerically
    airfoil_data = sorted_alphanumeric(airfoil_data)
    airfoil_pts = sorted_alphanumeric(airfoil_pts)
    output_files = sorted_alphanumeric(output_files)

    # Create a linearly spaced mesh with the given dimensions
    x_range = 29
    y_range = 18
    grid_resolution = 0.005
    x_points = int(x_range / grid_resolution)
    y_points = int(y_range / grid_resolution)
    X = np.linspace(-9, 20, x_points + 1)
    Y = np.linspace(-9, 9, y_points + 1)
    X_grid, Y_grid = np.meshgrid(X, Y)
    del X, Y

    # For each experimental simulated case
    for id_main, file_main in enumerate(output_files):
        # List where data will be sorted before casting it to a Numpy array
        master_ar = []

        # Each airfoil has been simulated in 24 different experimental points (velocity and angle), so each 24 cases
        # the simulation include a new airfoil geometry
        airfoil_index = id_main // 24

        # Verbose
        print("File {} of {}: Converting {} - {} - {}".format(id_main, n_files, file_main, airfoil_data[airfoil_index],
                                                              airfoil_pts[airfoil_index]))

        # Read mesh centres file
        with open(os.path.join('outputs', airfoil_data[airfoil_index]), 'r') as f:
            file = f.readlines()
        file = [line.replace('\n', '') for line in file]
        n_points = int(file[20])
        file = np.array([line[1:-1].split(' ') for line in file if line.startswith('(')][1:n_points+1]).astype('float')

        # Store mesh centres data into x and y Numpy arrays
        x = []
        y = []
        for i in range(n_points):
            x.append(file[i][0])
            y.append(file[i][1])
        x = np.array(x)
        y = np.array(y)

        # Read simulation final results file
        with open(os.path.join('outputs', file_main), 'r') as f:
            file = f.readlines()
        os.remove(os.path.join('outputs', file_main))
        file = [line.replace('\n', '') for line in file]
        n_points = int(file[20])
        file = np.array([line[1:-1].split(' ') for line in file if line.startswith('(')][1:n_points+1]).astype('float')

        # Store simulation final results file into u and v Numpy arrays
        u = []
        v = []
        try:
            for i in range(len(x)):
                u.append(file[i][0])
                v.append(file[i][1])
            u = np.array(u)
            v = np.array(v)

            # Correlate mesh centres points with linearly spaced created grid and interpolate velocity values
            data_array = np.array((x, y, u, v, np.zeros_like(x)))
            U_grid = griddata((x, y), u, (X_grid, Y_grid), method='linear')
            V_grid = griddata((x, y), v, (X_grid, Y_grid), method='linear')
            del x, y, u, v

            # Read airfoil geometry file. This is required to set velocity to 0 in all interna airfoil points
            coords = np.loadtxt(os.path.join("outputs", airfoil_pts[airfoil_index]), skiprows=1)
            x_airfoil = (coords[:,0] / grid_resolution).astype('int') * grid_resolution
            y_airfoil = (coords[:,1] / grid_resolution).astype('int') * grid_resolution
            x_airfoil_min = min(x_airfoil)
            x_airfoil_max = max(x_airfoil)

            # Search points of the linearly spaced grid which lay inside the airfoil body
            coords_grid = {}
            for x_layer in range(int(x_airfoil_min / grid_resolution), int(x_airfoil_max / grid_resolution + 1)):
                x_layer = x_layer * grid_resolution
                if x_layer not in coords_grid.keys():
                    coords_grid[x_layer] = []
                y_points = []
                for j in range(len(x_airfoil)):
                    if x_airfoil[j] == x_layer:
                        coords_grid[x_layer].append(y_airfoil[j])
            elements = []
            for key in coords_grid.keys():
                elements.append(len(coords_grid[key]))
            y_sup_line = []
            y_inf_line = []
            for id, element in enumerate(elements):
                if element > 1 or element == elements[0] or element == elements[-1]:
                    y_sup_line.append(max(coords_grid[list(coords_grid.keys())[id]]))
                    y_inf_line.append(min(coords_grid[list(coords_grid.keys())[id]]))
                else:
                    y_sup_line.append(None)
                    y_inf_line.append(None)

            # Convert all points in the flow field with NaN velocity to the flow velocity averaged value
            y_sup_line_full = []
            for id, y_sup_elem in enumerate(y_sup_line):
                if y_sup_elem is None:
                    y_sup_line_full.append(nan2mean(y_sup_line, id))
                else:
                    y_sup_line_full.append(y_sup_elem)
            y_inf_line_full = []
            for id, y_inf_elem in enumerate(y_inf_line):
                if y_inf_elem is None:
                    y_inf_line_full.append(nan2mean(y_inf_line, id))
                else:
                    y_inf_line_full.append(y_inf_elem)

            # Set velocity of points laying inside the airfoil body to 0
            for id, key in enumerate(coords_grid.keys()):
                coords_grid[key] = [y_sup_line_full[id], y_inf_line_full[id]]
            for id, key in enumerate(coords_grid.keys()):
                if id == 0:
                    x_index = int(np.argwhere(X_grid[0] == x_airfoil_min + id * grid_resolution))
                else:
                    x_index += 1
                for y_index in range(int(min(coords_grid[key]) / grid_resolution), int(max(coords_grid[key]) /
                                                                                       grid_resolution + 1)):
                    y_index = int(np.argwhere(abs(Y_grid.T[0] - y_index * grid_resolution) == min(abs(Y_grid.T[0] -
                                                                                                      y_index *
                                                                                                      grid_resolution)))
                                  )
                    U_grid[y_index][x_index] = 0
                    V_grid[y_index][x_index] = 0

            # Slice and select the central region of interest of the mesh
            U_grid = U_grid[1600:2200][:, 1600:2200]
            V_grid = V_grid[1600:2200][:, 1600:2200]
            master_ar.append([U_grid, V_grid])
            del(U_grid, V_grid)

            # Convert result array to Numpy file and store it in the folder /finalOutput/
            master_ar = np.array(master_ar)
            np.save('finalOutput/master_ar_{}'.format(file_main), master_ar)

        # If simulation results file contains error, show message but continue encoding process with other datapoints
        except:
            print("ERROR ON File {} of {}: Converting {} - {} - {}".format(id_main, n_files, file_main,
                                                                           airfoil_data[airfoil_index],
                                                                           airfoil_pts[airfoil_index]))
