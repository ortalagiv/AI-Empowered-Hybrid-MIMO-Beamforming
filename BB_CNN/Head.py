# -*- oding:utf-8 -*-
'''
# @File: Head.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2023-01-30 12:40 PM
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
K = 16
SNR = 1

OutputFbb=True



Conv_Type = '2D'
THz_channl = False
if THz_channl:
    train_data_name = 'data_train_H0.npy'
    test_data_name = 'data_test_H0.npy'
else:
    train_data_name = 'H_train.npy'
    test_data_name = 'H_test.npy'
test_batch_size = 100

# Training parameters
Seed_train = 1
Seed_test = 101

training_set_size_truncated = 1000
train_batch_size = 32
Ntrain_batch_total = 1500 # total number of training batches without considering the iteratively generated batches
Ntrain_Batch_perEpoch = training_set_size_truncated // train_batch_size
Ntrain_Epoch = math.ceil(Ntrain_batch_total/Ntrain_Batch_perEpoch)
# Ntrain_Epoch = 10
# Ntrain_batch_total = int(Ntrain_Epoch*Ntrain_Batch_perEpoch)

training_method = 'unsupervised'
Continue_Train = False


Residule_NN = False



Weight_decay = 0  # add L2 regularizer to weight, the penalty is larger with high Weight_decay
start_learning_rate = 1e-4
Log_interval = 10  # interval for print loss
set_Lr_decay = False
Lr_min = 1e-5
Lr_keep_steps = 5
Lr_decay_factor = 0.95

# To save trained model


dataset_file = "./trained_model/" + str(Nt) + "x" + str(Nr) + "x" + str(Nrf) + "x" + str(K) + "/"
directory_model = dataset_file
dat_file_name = directory_model + 'data'+'.mat'

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


