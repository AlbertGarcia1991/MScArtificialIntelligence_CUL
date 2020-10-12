########################################################################################################################
#                                  DEEP NEURAL NETWORKS FOR AERODYNAMIC LOW PREDICTION                                 #
#                                             Student: Albert García Plaza                                             #
#                                              Supervisor: Eduardo Alonso                                              #
#                                               INM363 Individual project                                              #
#                              MSc in Artificial Intelligence - City, University of London                             #
########################################################################################################################

import createBlockMeshDict
import createNPyFiles

import os
import shutil
import numpy as np
import subprocess
from math import cos, sin
from time import time


def wind_change(wind_vel, wind_ang):
    """
    Given the desired airfoil velocity and angle of attack, changes the OpenFOAM files where is specified these
    intial conditions.
    :param wind_vel: Wind velocity (module) in m/s
    :param wind_ang: Wind angle of attack in radians.
    """
    # Store the path to the OpenFOAM initial wind conditions file
    src = os.path.join(cwd, '0/U')

    # Open the file and retrieve its content
    with open(src, 'r') as f:
        file = f.readlines()

    # Compute the velocity components in X (u), and in Y (v)
    u = wind_vel * cos(wind_ang)
    v = wind_vel * sin(wind_ang)

    # Change the required line to update the wind conditions
    file[18] = str(file[18][:25]) + "{} {} 0);\n".format(u, v)
    with open(src, 'w') as f:
        f.writelines(file)

# Indicate the folder where the airfoil geometry files are located and filter only .dat files to avoid errors
airfoils_db = os.listdir('airfoilsDataset/')
airfoils_db = [airfoil for airfoil in airfoils_db if airfoil.endswith('.dat')]
cwd = os.getcwd()  # working path

# Set 4 different Reynolds numbers to run each case.
wind_velocities = np.array([1e4, 1e5, 1e6, 1e7]) * (1.81e-5 / 1.225)
# Set 5 different angle of attach to run each case.
wind_angles = np.radians(np.arange(6.) * 9)

# Encode each simulation (each combination od airfoil, velocity and angle) numerically and store all experiments
cases = []
for id_airfoil, airfoil in enumerate(airfoils_db):
    for id_wind_vel, wind_vel in enumerate(wind_velocities):
        for id_wind_ang, wind_ang in enumerate(wind_angles):
            cases.append([id_airfoil, airfoil, id_wind_vel, wind_vel, id_wind_ang, wind_ang])

# This section allows to re-start the simulation from the last experimental point simulated, rather than start again
# every time this Python file is run
run_cases = airfoils_db = os.listdir('outputs/')
if not run_cases:
    pass
else:
    run_cases = [case for case in run_cases if not case.startswith('B')]
    run_cases = [case for case in run_cases if not case.endswith('.dat')]
    run_cases = [case.split('_') for case in run_cases if not case.endswith('Centres')]
    run_cases = np.array(run_cases).astype('int')
    max_airfoil = max(run_cases.T[0])
    max_wind_vel = max(run_cases.T[1])
    max_wind_ang = max(run_cases.T[2])
    for id, cas in enumerate(cases):
        if int(cas[0]) == max_airfoil and int(cas[2]) == max_wind_vel and int(cas[4]) == max_wind_ang:
            cases = cases[id+1:]
            break

# Clean all previous legacy of past simulations
subprocess.call("./AllClean")
new_airfoil = True

# Loop over each experimental case setup
for id_case, case in enumerate(cases):
    print(case)  # verbose

    # if the current geometry has not its mesh created yet (different velocities and angles of attack are run along the
    # same mesh. This only changes for each different airfoil.
    if new_airfoil:
        # Pick the current case geometry and copy it to the OpenFOAM case folder
        src = os.path.join(cwd, 'airfoilsDataset/{}'.format(case[1]))
        shutil.copy(src, cwd)
        dst = 'outputs/{}.dat'.format(case[0])
        shutil.copy(src, dst)

        # Create the mesh
        createBlockMeshDict.create_blockMeshDict(case[1])
        subprocess.call("./BlockMeshRun")

        # Since the mesh for current geometry has been created, set to false this variable avoiding re-meshing again
        new_airfoil = False

    # Simulation starting time
    t0 = time()

    # Set the correct wind velocity and angle for before running the simulation
    wind_change(case[3], case[5])

    # Ensure no prior simulations outputs exist on the case folder
    subprocess.call("./CleanSim")

    # Run the simulation
    subprocess.call("./SolverRun")

    # Since the case is solved following an iterative process, OpenFOAM creates a folder for each step named with an
    # increase serie of numbers (e.g. first iteration is 0, second is 10, thirs is 20, ad so on). The number of
    # iterations between created folder can be controlled in the case setup files.
    results = os.listdir()  # read all created solution folder
    # Ensure that only created solution folders are selected
    results = [result for result in results if result.endswith('0')]
    results = [int(result) for result in results if not result.startswith('p')]

    # Pick the folder with the biggest number in its name since this will be the last iteration (convergence reached)
    last_folder = str(sorted(results)[-1])

    # If no solution folder has been created, the directory will only contain the initial case folder named '0'. Omit it
    if last_folder == '0':
        pass
    else:
        # Copy the simulated velocity-field from the case folder to the created output folder
        src = os.path.join(cwd, last_folder, 'U')
        dst = os.path.join(cwd, 'outputs/{}_{}_{}'.format(case[0], case[2], case[4]))
        shutil.copy(src, dst)

    # Apart from the simulated velocity-field, the mesh geometry is also required to correlate velocity value with
    # Cartesian coordinate in the simulated region.
    if case[3] == wind_velocities[-1] and case[5] == wind_angles[-1]:
        # As the cell centres are not computed by default, a script has to be run to compute these centres coordinates
        subprocess.call("./MakeCellCentres")

        # Copy this geometry file to the same output folder as well
        src = os.path.join(cwd, last_folder, 'C')
        dst = os.path.join(cwd, 'outputs/{}_Centres'.format(case[0]))
        try:
            shutil.copy(src, dst)
        except:
            pass

        # If the current case is the last velocity and last angle of attack, next case will contain a new geometry. So,
        # the mesh has to be created and to do this, this variable is set to True again
        new_airfoil = True

        # Clean all old mesh and geometry files since next case will contain a different airfoil
        subprocess.call("./AllClean")

    # Clean all simulation files at the end of each case
    subprocess.call("./CleanSim")

    # Print time spent in last case run
    print(time() - t0)

# Encode all results into .npy files
createNPyFiles.openFOAM2Npy()
