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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from datetime import date
import pickle
import os
import random
import numpy as np
import pdb

from data import train_loader, test_loader
from networks import ResNet, LeNet, AlexNet


# Set the global constants and parameters to do a grid search (learning rate, weight decay, and network)
RANDOM_SEED = 0
N_EPOCHS = 50
LOSS_MULTIPLIER = 10000000
LEARNING_RATE = [0.01]  # If grid search, populate this list with the learning rates to experiment
WEIGHT_DECAY = [0.1]  # If grid search, populate this list with the weight decays to experiment
NETWORK = ["LeNet"]  # If grid search, populate this list with the networks to experiment

# Set the random seed to obtain reproducible and comparable results
torch.manual_seed(RANDOM_SEED)

# Set the loss for all experiments as the Mean-Squared Error
#MSE_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()

# Initialize the variables specifying the filepath to required files
cwd = os.getcwd()
airfoils_path = os.path.join(cwd, 'data', 'airfoils')
simulations_path = os.path.join(cwd, 'data', 'simulations')

# Split the available files into train (70%), validation (15%), and test (15%)
files = os.listdir(simulations_path)
random.shuffle(files)
cut_index = len(files) - int(len(files) * 0.30)
cut_index_val = len(files) - int(len(files) * 0.15)
train_files = np.array(files[:cut_index])
test_files = np.array(files[cut_index:cut_index_val])

# Start the gris search over all defined parameters
for network in NETWORK:
    # Set the correct network for each experiment
    if network == "ResNet":
        model = ResNet()
        BATCH_SIZE = [4]  # Due to memory issues, the biggest batch size that the hardware can handle is 4 for ResNet
    elif network == "LeNet":
        model = LeNet()
        BATCH_SIZE = [32]  # leNet can be run with maximum batch size 32

    # If CUDA, move the model to the GPU and set other parameters as deterministics to allow reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model.cuda()

    # Create dicts to store the results at each epoch
    lr_dict = {}
    for lr in LEARNING_RATE:

        wd_dict = {}
        for wd in WEIGHT_DECAY:
            # All experiments will use the ADAM optimizer
            params = model.parameters()
            optimizer = optim.Adam(params=params, lr=lr, weight_decay=wd)

            for bs in BATCH_SIZE:

                batch_train_indices = np.array([index // bs for index in range(len(train_files))])
                n_train_batches = int(max(batch_train_indices))

                batch_test_indices = np.array([index // bs for index in range(len(test_files))])
                n_test_batches = int(max(batch_test_indices))

                # Reset performance variables of the current experiment
                train_loss = []
                test_loss = []
                
                epoch_dict = {}
                for epoch in range(N_EPOCHS):
                    model.train()


                    print("Computing epoch {}".format(epoch))
                    prov_train_loss = []

                    for batch in range(n_train_batches):
                        print(n_train_batches, batch)

                        xs, ys = train_loader(train_files, batch, bs)

                        # Store training inputs and outputs as Torch Variables
                        xs, ys = Variable(xs), Variable(ys)

                        # If CUDA available, mode these Variables to the GPU
                        if torch.cuda.is_available():
                            xs, ys = xs.cuda(), ys.cuda()

                        # Reset gradient array
                        optimizer.zero_grad()

                        # Do the forward pass with the current inputs to have the network outputs (predictions)
                        preds = model(xs)

                        # Compare predictions with truth values to compute the loss, and do the backward pass
                        trn_loss = MSE_loss(preds, ys)
                        trn_loss.backward()

                        # Store the current training loss value
                        prov_train_loss.append(trn_loss.data.item())

                        # Update network weights
                        optimizer.step()

                        del xs, ys, preds, trn_loss

                    train_loss.append(np.nanmean(prov_train_loss) * LOSS_MULTIPLIER)

                    
                    with torch.no_grad():
                        # Store the network weights at the very first epoch, and compute the test loss
                        if epoch % 10 == 0:
                            prov_test_loss = []
                            for batch in range(n_test_batches):
                                xt, yt = test_loader(test_files, batch, bs)

                                # Store testing inputs and outputs as Torch Variables
                                xt, yt = Variable(xt), Variable(yt)

                                # If CUDA available, mode these Variables to the GPU
                                if torch.cuda.is_available():
                                    xt, yt = xt.cuda(), yt.cuda()

                                # Show results every 10 epochs, and compute the test loss
                                model.eval()
                                preds = model(xt)
                                tst_loss = MSE_loss(preds, yt)
                                prov_test_loss.append(tst_loss.data.item())

                                del xt, yt, preds, tst_loss


                    test_loss.append(np.nanmean(prov_test_loss) * LOSS_MULTIPLIER)
                    print("Network: {}. Learning rate: {}. Weight Decay: {}. Batch size: {}\n".format(network, lr, wd,
                                                                                                      bs))
                    print("Epoch: {}. Train loss: {:.4f}. Test loss: {:.4f}\n\n".format(epoch, train_loss[-1],
                                                                                        test_loss[-1]))

                    with open('log.txt', 'a+') as f:
                        f.write("Network: {}. Learning rate: {}. Weight Decay: {}. Batch size: {}\n Epoch: {}. "
                                "Train loss: {:.4f}. Test loss: {:.4f}\n\n".format(network, lr, wd, bs, epoch,
                                                                                   train_loss[-1], test_loss[-1]))
                    # If the test loss has been improved, store the current weights. If the current test loss is
                    #  worst than the mean of the previous 10 epochs, stop the training (early stop)
                    if len(test_loss) > 20:
                        if test_loss[-1] < np.mean(test_loss[-12:-1]):
                            pass
                            torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}'.format(network, lr, wd, bs))
                        else:
                            pass

                    epoch_dict[epoch] = [train_loss, test_loss]
            wd_dict[wd] = epoch_dict
        lr_dict[lr] = wd_dict

    # Store the results dictionary locally as a pickled file
    with open('final_results_dict_{}.pkl'.format(network), 'wb') as f:
        pickle.dump(lr_dict, f, pickle.HIGHEST_PROTOCOL)
