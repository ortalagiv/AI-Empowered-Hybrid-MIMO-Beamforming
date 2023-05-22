# -*- oding:utf-8 -*-
'''
# @File: BB_FuncLr.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2023-01-30 12:49 PM
'''
import numpy as np
import torch

from Head import *
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class My_dataset(Dataset):
    def __init__(self, file_dir, file_name, conv_type='3D', datatype=THz_channl):
        super(My_dataset, self).__init__()
        self.file_dir = file_dir
        self.file_name = file_name
        self.conv_type = conv_type
        data_file = os.path.join(self.file_dir, self.file_name)
        self.data = np.load(data_file)
        if datatype:
            self.data = np.transpose(self.data,(0, 3, 1, 2))
        self.len, self.Nc, self.Nr, self.Nt = self.data.shape
        ccc = 1
    def __getitem__(self, index):
        ch_smp_3D = self.data[index, :, :, :]
        if self.conv_type == '3D':
            ch_smp_out = np.zeros([3, self.Nc, self.Nr, self.Nt], dtype=np.float32)
            ch_smp_out[0, :, :, :] = np.abs(ch_smp_3D)
            ch_smp_out[1, :, :, :] = np.real(ch_smp_3D)
            ch_smp_out[2, :, :, :] = np.imag(ch_smp_3D)
        else:
            ch_smp_cat = np.zeros([self.Nc*self.Nr, self.Nt], dtype=np.complex64)
            for k in range(self.Nc):
                ch_smp_cat[k*self.Nr:(k+1)*self.Nr, :] = ch_smp_3D[k, :, :]
            ch_smp_out = np.zeros([3, self.Nc*self.Nr, self.Nt], dtype=np.float32)
            ch_smp_out[0, :, :] = np.abs(ch_smp_cat)
            ch_smp_out[1, :, :] = np.real(ch_smp_cat)
            ch_smp_out[2, :, :] = np.imag(ch_smp_cat)
        # tmp = ch_smp_3D.astype(np.complex64)
        ccc = 1
        return ch_smp_3D.astype(np.complex64), ch_smp_out


    def __len__(self):
        return self.len


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def matmul_complex(t1,t2):
    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag,t1.real @ t2.imag + t1.imag @ t2.real),dim=2))


def loss_caclu(fvec_est, batch_H):
    LOSS_cache = 0
    for ii in range(fvec_est.shape[0]):
        Hii = batch_H[ii, :, :, :]  # prepare testing data
        ff = fvec_est[ii, :]
        FF = reshape_fortran(ff, [-1, 2])
        ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
        Frf_DNN = reshape_fortran(ff_complex, [Nt, -1])  # convert to RF precoding matrix
        FRF = Frf_DNN / torch.abs(Frf_DNN)  # normalize to get 1-modulus
        FRF_pinv = torch.linalg.pinv(FRF)
        Rate = 0
        for k in range(K):
            U, S, VH = torch.linalg.svd(Hii[k, :, :], full_matrices=False)
            Woptk = U[:, 0:Ns]
            V = VH.H
            Foptk = V[:, 0:Ns]
            FBBk = torch.matmul(FRF_pinv, Foptk)  # compute BB precoder
            # tmp = matmul_complex(FRF, FBBk)
            tmp_matrix = matmul_complex(torch.squeeze(Hii[k, :, :]), matmul_complex(FRF, FBBk))
            tmp2 = torch.eye(Nr) + SNR / Ns * matmul_complex(tmp_matrix, tmp_matrix.H)
            Rk = torch.real(torch.log(torch.linalg.det(tmp2)))
            Rate = Rate + Rk
        loss_ii = Rate / K
        LOSS_cache = LOSS_cache + loss_ii
    LOSS = LOSS_cache / fvec_est.shape[0]
    return -LOSS


def loss_caclu_bb(fvec_est, Fbb_est, batch_H):
    LOSS_cache = 0
    dim = 2 * Nrf * Nr
    batch_size = fvec_est.shape[0]
    Frf_batch = torch.zeros([batch_size, Nt, Nrf], dtype=torch.complex64)
    Fbb_batch = torch.zeros([batch_size, Nrf, Nr, K], dtype=torch.complex64)
    for ii in range(batch_size):
        Hii = batch_H[ii, :, :, :]  # prepare testing data
        ff = fvec_est[ii, :]
        FF = reshape_fortran(ff, [-1, 2])
        ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
        Frf_DNN = reshape_fortran(ff_complex, [Nt, -1])  # convert to RF precoding matrix

        FRF = Frf_DNN / torch.abs(Frf_DNN)  # normalize to get 1-modulus

        Frf_batch[ii, :, :] = FRF
        Rate = 0
        fbb_all = Fbb_est[ii, :]
        for k in range(K):
            fbbk = fbb_all[k*dim:(k+1)*dim]
            fbb_column = reshape_fortran(fbbk, [-1, 2])
            fbb_complex = fbb_column[:, 0] + 1j * fbb_column[:, 1]
            Fbb_DNN = reshape_fortran(fbb_complex, [Nrf, -1])
            Decoef = torch.linalg.norm(matmul_complex(FRF, Fbb_DNN))
            Fbbk = np.sqrt(Ns)/Decoef * Fbb_DNN  # do normalization

            Fbb_batch[ii, :, :, k] = Fbbk

            tmp_matrix = matmul_complex(torch.squeeze(Hii[k, :, :]), matmul_complex(FRF, Fbbk))
            tmp2 = torch.eye(Nr) + SNR / Ns * matmul_complex(tmp_matrix, tmp_matrix.H)
            Rk = torch.real(torch.log(torch.linalg.det(tmp2)))
            Rate = Rate + Rk

        loss_ii = Rate / K
        LOSS_cache = LOSS_cache + loss_ii
    LOSS = LOSS_cache / batch_size
    return -LOSS, Frf_batch, Fbb_batch


class CNN_v1(nn.Module):
    def __init__(self, Nrf, Nt):
        super(CNN_v1, self).__init__()
        self.ap_dims = 2 * Nrf * Nt
        self.conv1 = nn.Sequential(         # input shape (2, Nc, Nt)
            nn.Conv3d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, Nc, Nt)
            nn.ReLU(),                      # activation
            nn.MaxPool3d(kernel_size=2, stride=2),    # choose max value in 2x2 area, output shape (16, Nc/2, Nt/2)
            nn.BatchNorm3d(16)
        )
        self.conv2 = nn.Sequential(         # input shape (2, Nc, Nt)
            nn.Conv3d(
                in_channels=16,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, Nc, Nt)
            nn.ReLU(),                      # activation
            nn.MaxPool3d(kernel_size=2, stride=2),    # choose max value in 2x2 area, output shape (16, Nc/2, Nt/2)
            nn.BatchNorm3d(32)
        )

        fc1_hidden1, fc1_hidden2 = 384, 128
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_hidden1, fc1_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden2),
            nn.Linear(fc1_hidden2, self.ap_dims)
        )


    def forward(self, x_csi):

        x = self.conv1(x_csi)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7=1568)
        fc_out = self.fc1(x)  # output extracted feature    DoD_az, DoD_el, tau
        x_out = -1 + torch.nn.functional.relu(fc_out + 0.5) / 0.5 - torch.nn.functional.relu(
            fc_out - 0.5) / 0.5
        out = 1 * x_out
        return x_out # return feature_other and phase

class CNN_v2(nn.Module):
    def __init__(self, Nrf, Nt):
        super(CNN_v2, self).__init__()
        self.ap_dims = 2 * Nrf * Nt
        self.conv1 = nn.Sequential(         # input shape ( Nc, Nt)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, Nc, Nt)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=3, stride=(2,1)),    # choose max value in 2x2 area, output shape (16, Nc/2, Nt/2)
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(         # input shape (16, Nc/2, Nt/2)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, Nc/2, Nt/2)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=3, stride=(2,1)),                # output shape (32, Nc/4, Nt/4)
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(         # input shape (32, Nc/4, Nt/4)
            nn.Conv2d(32, 64, 3, 1, 1),     # output shape (64, Nc/4, Nt/4)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=3, stride=(2,1)),                # output shape (64, Nc/8, Nt/8)
            nn.BatchNorm2d(64)
        )

        fc1_hidden1, fc1_hidden2, fc1_hidden3 = 2688, 512, 128
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_hidden1, fc1_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden2),
            nn.Linear(fc1_hidden2, fc1_hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden3),
            nn.Linear(fc1_hidden3, self.ap_dims)
        )



    def forward(self, x_csi):

        x = self.conv1(x_csi)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7=1568)
        fc_out = self.fc1(x)  # output extracted feature    DoD_az, DoD_el, tau
        x_out = -1 + torch.nn.functional.relu(fc_out + 0.5) / 0.5 - torch.nn.functional.relu(
            fc_out - 0.5) / 0.5

        out = 1 * x
        return x_out # return feature_other and phase


class CNN_v3(nn.Module):
    def __init__(self, Nrf, Nt, Nr, K):
        super(CNN_v3, self).__init__()
        self.ap_dims = 2 * Nrf * Nt
        self.dp_dims = 2 * Nrf * Nr * K
        self.conv1 = nn.Sequential(  # input shape ( Nc, Nt)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, Nc, Nt)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # choose max value in 2x2 area, output shape (16, Nc/2, Nt/2)
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(  # input shape (16, Nc/2, Nt/2)
            nn.Conv2d(16, 32, 3, 1, 1),  # output shape (32, Nc/2, Nt/2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # output shape (32, Nc/4, Nt/4)
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(  # input shape (32, Nc/4, Nt/4)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, Nc/4, Nt/4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # output shape (64, Nc/8, Nt/8)
            nn.BatchNorm2d(64)
        )

        fc1_hidden1, fc1_hidden2, fc1_hidden3 = 2688, 512, 256
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_hidden1, fc1_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden2),
            nn.Linear(fc1_hidden2, fc1_hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden3)
        )

        self.Fbb = nn.Sequential(
            nn.Linear(fc1_hidden3, self.dp_dims),
            nn.PReLU()
        )

        self.Frf = nn.Linear(fc1_hidden3, self.ap_dims)

    def forward(self, x_csi):
        x = self.conv1(x_csi)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7=1568)
        fc_out = self.fc1(x) # output extracted feature    DoD_az, DoD_el, tau
        Fbb = self.Fbb(fc_out)  # output digital precoder
        Frf_out = self.Frf(fc_out)  # output analog precoder
        Frf = -1 + torch.nn.functional.relu(Frf_out + 0.5) / 0.5 - torch.nn.functional.relu(
            Frf_out - 0.5) / 0.5

        out = 1 * x
        return Frf, Fbb # return feature_other and phase

def FLOPs_Cov(C_in,C_out,kernel_size, FL, FW):
    FLOPs_Num = ( (C_in * kernel_size ** 2) + (C_in * kernel_size ** 2-1) +1 ) * C_out * FL * FW
    return FLOPs_Num

def FLOPs_Fc(NN_in, NN_out):
    FLOPs_Num = 2 * NN_in * NN_out
    return FLOPs_Num
if __name__ == '__main__':
    # test = My_dataset(file_dir=dataset_file, file_name='H_train.npy')
    # test.__getitem__(1)
    FLOPs_model = FLOPs_Cov(C_in=3, C_out=16,kernel_size=3, FL=64, FW=12)
    + FLOPs_Cov(C_in=16, C_out=32, kernel_size=3, FL=31, FW=10)
    + FLOPs_Cov(C_in=32, C_out=64, kernel_size=3, FL=15, FW=8)
    + FLOPs_Fc(2688, 512) + FLOPs_Fc(512, 128) + FLOPs_Fc(128, 96)
    print(f'Num of FLOPS={FLOPs_model}')