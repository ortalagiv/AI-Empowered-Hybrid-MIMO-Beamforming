# -* coding: utf-8 -*-
'''
@File；Params.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-15 15:45
Configure global variables
'''
import os
import torch
import math
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Parameters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# System parameters
Nt = 12
Nr = 4
Nrf = 4
Ns = 4
N = 2 * Nt * Nrf
Ncl = 1  ## number of clusters
Nray = 4 ## number of rays in each cluster
GHz = 1e+9
K = 16
Bandwidth = 30 * GHz  # system bandwidth
fc = 300 * GHz   # carrier frequency
Array_Type = 'UPA'
Num_layers = math.ceil(math.log2(Nt))
Sub_Connected = True
# Sub_Structure_Type = 'fixed'
Sub_Structure_Type = 'dyn'




# Training parameters
Seed_train = 1
Seed_test = 101

train_batch_size = 10
Ntrain_Batch = 10
test_batch_size = 100

training_method = 'unsupervised'
Iterative_Training = True
Iterations_train = 3
Continue_Train = False
Rdm_Ini = True

Keep_Bias = False
Residule_NN = False
Loss_coef = 1

SUM_LOSS = 1
Weight_decay = 0  # add L2 regularizer to weight, the penalty is larger with high Weight_decay
start_learning_rate = 1e-4
Log_interval = 10  # interval for print loss
set_Lr_decay = False
Lr_min = 1e-5
Lr_keep_steps = 5
Lr_decay_factor = 0.95


Iterations_test = 10
Noisy_Channel = False
epision = 0.2


# To save trained model


if Sub_Connected:
    if Sub_Structure_Type == 'fixed':
        subsubfile = 'PC_HB_Fixed/'
    else:
        subsubfile = 'PC_HB_Dyn/'
else:
    subsubfile = 'FC-HB/'

if Iterative_Training:
    sssbfile = 'Iterative_train/'
else:
    sssbfile = 'Non-iterative_train/'

dataset_file = "./trained_model/" + str(Nt) + "x" + str(Nr) + "x" + str(Nrf) + "x" + str(K) + "/"
directory_model = dataset_file + subsubfile + sssbfile
dat_file_name = directory_model + 'data''.mat'

# stamp = int(Bandwidth/GHz)
# if Noisy_Channel:
#     stamp = epision
# dat_file_name = directory_model + 'data_B'+str(stamp)+'G.mat'

model_file_name = directory_model + "trained_model"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)



Cuda_set = 0  # whether to use GPU
# 检测GPU
MainDevice = torch.device("cuda:0" if torch.cuda.is_available() and Cuda_set else "cpu")
NumGPU = torch.cuda.device_count()
print(MainDevice, flush=True)
use_gpu = torch.cuda.is_available() and Cuda_set
if use_gpu:
    print('using GPU for training:\n',flush=True)
    print('cuda.is_available:', torch.cuda.is_available(),flush=True)
    print('cuda.device_count:', torch.cuda.device_count(),flush=True)
    print('cuda.device_name:', torch.cuda.get_device_name(0),flush=True)
else:
    print('Using CPU for training:\n',flush=True)


pass

