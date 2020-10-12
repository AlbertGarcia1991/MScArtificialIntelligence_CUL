########################################################################################################################
#                                  DEEP NEURAL NETWORKS FOR AERODYNAMIC LOW PREDICTION                                 #
#                                             Student: Albert García Plaza                                             #
#                                              Supervisor: Eduardo Alonso                                              #
#                                               INM363 Individual project                                              #
#                              MSc in Artificial Intelligence - City, University of London                             #
#                                                                                                                      #
# Code implementation based on:                                                                                        #
#   - ResNet:   Based on https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278              #
#               Based on https://github.com/usuyama/pytorch-unet                                                       #
#   - LeNet:    Based on https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320       #
#   - AlexNet:  Based on https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-    #
#                           practical-cnns-in-pytorch-1-61daa679c74a                                                   #
########################################################################################################################

import numpy as np
import os
import random
import pdb

import torch

# Airfoils geometry files should be located into the directory /data/airfoils/ and simulations .npy files
# into /data/simulations
cwd = os.getcwd()
airfoils_path = os.path.join(cwd, 'data', 'airfoils')
simulations_path = os.path.join(cwd, 'data', 'simulations')


def np_cat(element):
    """
    Given the filenames of simulations files as a list, return the concatenated Numpy array ready to be caster as a
    PyTorch Tensor.
    """
    try:
        out = np.load(os.path.join(simulations_path, element))
        # If the simulation file contains non-numeric values, return False and no array (no consider this datapoint)
        if np.inf in out or -np.inf in out or np.nan in out:
            return False, None
        # If all the content inside the .npy file is numeric, return True and the normalized values of the simulation
        else:
            out /= np.linalg.norm(out)
            return True, out

    # If any error processing the airfoil, return False and not consider this data point
    except:
        return False, None


def create_x1(filename):
    """
    Create array with input/output size (600x600) full of 0's inside the airfoil body, and 1's in the air field (where
    the air can flow).
    """
    placeholder, out = np_cat(filename)
    # Taking advantage of the simulations results, the airflow has no velocity only in the interior and surface of the
    # airfoil.
    try:
        out = out[0][0]
        out[out != 0] = 1

    # If any error processing the airfoil, return False and not consider this data point
    except:
        placeholder, out = False, None

    return placeholder, out


def create_x2x3(filename):
    """
    Create the second and third input channels, which only are representing the air velocity and angle of attack.
    Given the filename, the velocity and angle are encoded. So, this encoded values have to be converted to standard
    units and return its normalized values.
    """
    # Read the velocity and angle encoding from the filename
    velocity_coded = int(filename[:-4].split('_')[3])
    aoa_coded = int(filename[:-4].split('_')[4])

    # Decode this values into real values (in m/s and radians)
    velocity = (10 ** (4 + velocity_coded)) * (0.0000181 / 1.225)
    aoa_rad = np.deg2rad(9 * aoa_coded)

    # Apply trigonometry to compute the resultant velocity in both pane directions (X and Y)
    u_vel = velocity * np.cos(aoa_rad)
    v_vel = velocity * np.sin(aoa_rad)

    # Normalize these velocities
    v_vel = v_vel / u_vel
    u_vel = 1

    # Return two 600x600 arrays where all values are the resultant velocity in each Cartesian direction
    return np.full((600, 600), u_vel), np.full((600, 600), v_vel)


def train_loader(train_files, batch, batch_size=64):
    """
    Given a list of filenames of elements to use as train dataset, split in batchs of given size and return the input
    and truth value for the required batch number.
    """
    # Assign to each element in the train dataset a batch index
    batch_indices = np.array([index // batch_size for index in range(len(train_files))])

    # Only for the elements belonging to the required batch, generate the input and truth value for the whole batch
    train_batch = train_files[np.where(batch_indices == batch)]
    X = []
    Y = []
    for train in train_batch:
        placeholder, x1 = create_x1(train)
        if placeholder == True:
            x2, x3 = create_x2x3(train)
            X.append([x1, x2, x3])
            _, y = np_cat(train)
            Y.append(y[0])

    # Convert to PyTorch tensors
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y)
    train_x[train_x == np.inf] = 0  # ensure no infite values in the array
    train_y[train_y == np.inf] = 0  # ensure no infite values in the array

    return train_x, train_y


def test_loader(test_files, batch, batch_size=64):
    """
    Given a list of filenames of elements to use as test dataset, split in batchs of given size and return the input
    and truth value for the required batch number.
    """
    # Assign to each element in the test dataset a batch index
    batch_indices = np.array([index // batch_size for index in range(len(test_files))])

    # Only for the elements belonging to the required batch, generate the input and truth value for the whole batch
    test_batch = test_files[np.where(batch_indices == batch)]
    X = []
    Y = []
    for test in test_batch:
        placeholder, x1 = create_x1(test)
        if placeholder == True:
            x2, x3 = create_x2x3(test)
            X.append([x1, x2, x3])
            _, y = np_cat(test)
            Y.append(y[0])

    # Convert to PyTorch tensors
    test_x = torch.FloatTensor(X)
    test_y = torch.FloatTensor(Y)
    test_x[test_x == np.inf] = 0  # ensure no infite values in the array
    test_y[test_y == np.inf] = 0  # ensure no infite values in the array

    return test_x, test_y
