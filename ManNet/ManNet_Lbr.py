# -* coding: utf-8 -*-
'''
@File；ManNet.py
@Author: Mengyuan Ma, Nhan Nguyen
@Contact: {mengyuan.ma, nhan.nguyen}@oulu.fi
@Time: 2022-08-16 15:15
 This is the self-defined function/module library
'''
import numpy as np
import torch

from Params import *
import math
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader


def array_dimension(Nt):
    '''
    :param Nt: number of total antennas
    :return: the configuration of UPA that minimizes beam squint effect
    '''
    n = math.ceil(Nt ** 0.5)
    for i in range(n+1,1,-1):
        if Nt%i==0:
            Nth = i
            Ntv = int(Nt/i)
            break
    return Nth, Ntv

def pulase_filter(t, Ts, beta):
    '''
    Raised cosine filter
    :param t: time slot
    :param Ts: sampling frequency
    :param beta: roll-off factor
    :return: filtered value
    '''
    if abs(t-Ts/2/beta)/abs(t) <1e-4 or abs(t+Ts/2/beta)/abs(t)<1e-4:
        p = np.pi/4 * np.sinc(1/2/beta)
    else:
        p = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts)/(1-(2*beta*t/Ts)**2)
    return p


def array_response(Nh,Nv, Angle_H, Angle_V, f,fc, array_type = 'UPA', AtDs=0.5):
    '''
    This function defines a steering vector for a Nh*Nv uniform planar array (UPA).
    See paper 'Dynamic Hybrid Beamforming With Low-Resolution PSs for Wideband mmWave MIMO-OFDM Systems'
    :param Nh: number of antennas in horizontal direction
    :param Nv: number of antennas in vertical direction
    :param fc: carrier frequency
    :param f: actual frequency
    :param AtDs: normalized antenna spacing distance, set to 0.5 by default
    :return: steering a vector at frequency f with azimuth and elevation angles
    '''
    N = int(Nh*Nv)
    Np = Angle_H.shape[0]
    AtDs_h = AtDs
    AtDs_v = AtDs
    array_matrix = np.zeros([N,Np], dtype=np.complex_)
    if array_type == 'ULA':
        spatial_h = np.sin(Angle_H)
        factor_h = np.array(range(N))
        for n in range(Np):
            array_matrix[:, n] = 1/np.sqrt(N)*np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])

    else:
        # Nh, Nv = array_dimension(N)
        spatial_h = np.sin(Angle_H) * np.sin(Angle_V)
        spatial_v = np.cos(Angle_V)
        factor_h = np.array(range(Nh))
        factor_v = np.array(range(Nv))
        for n in range(Np):
            steering_vector_h = 1/np.sqrt(Nh) * np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])
            steering_vector_v = 1/np.sqrt(Nv) * np.exp(1j*2*np.pi* AtDs_v * factor_v*f/fc*spatial_v[n])
            array_matrix[:,n] = np.kron(steering_vector_h, steering_vector_v)
    ccc = 1
    return array_matrix


def channel_model(Nt, Nr, Pulse_Filter = True, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    Np = Ncl * Nray
    gamma = np.sqrt(Nt * Nr / Np)  # normalization factor
    sigma = 1  # according to the normalization condition of the H
    Ntv = 4
    Nth = Nt // Ntv

    Nrh = 2
    Nrv = Nr // Nrh

    beta = 1
    Ts = 1/Bandwidth
    Delay_taps = int(K/4)
    angle_sigma = 10 / 180 * np.pi  # standard deviation of the angles in azimuth and elevation both of Rx and Tx

    AoH_all = np.zeros([2, Np])  # azimuth angle at Tx and Rx
    AoV_all = np.zeros([2, Np])  # elevation angle at Tx and Rx

    for cc in range(Ncl):
        AoH = np.random.uniform(0, 2, 2) * np.pi
        AoV = np.random.uniform(-0.5, 0.5, 2) * np.pi

        AoH_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoH_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoV_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi
        AoV_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi

    # alpha = np.sqrt(sigma / 2) * (
    #         np.random.normal(0, 1, size=[Np, K]) + 1j * np.random.normal(0, 1, size=[Np, K]))
    alpha = np.sqrt(sigma / 2) * (
            np.random.normal(0, 1, size=[Np, ]) + 1j * np.random.normal(0, 1, size=[Np, ]))
    Delay = np.random.uniform(0, Delay_taps, size=Np) * Ts
    # AoH_all = np.random.uniform(-1, 1, size=[2, Np]) * np.pi
    # AoV_all = np.random.uniform(-0.5, 0.5, size=[2, Np]) * np.pi
    Coef_matrix = np.zeros([Np, K], dtype='complex_')
    H_all = np.zeros([Nr, Nt, K], dtype='complex_')
    At_all = np.zeros([Nt, Np, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    for k in range(K):
        # fk = 2
        fk = fc + bandwidth * (2 * k - K + 1) / (2 * K)
        At = array_response(Nth, Ntv, AoH_all[0, :], AoV_all[0, :], fk, fc, array_type=Array_Type)
        Ar = array_response(Nrh, Nrv, AoH_all[1, :], AoV_all[1, :], fk, fc, array_type=Array_Type)

        # AhA_t=np.matmul(At.conj().T, At)
        # AhA_r = np.matmul(Ar.conj().T, Ar)

        At_all[:, :, k] = At
        for n in range(Np):
            if Pulse_Filter:
                med = 0
                for d in range(Delay_taps):
                    med += pulase_filter(d * Ts - Delay[n], Ts, beta) * np.exp(-1j * 2 * np.pi * k * d / K)
                Coef_matrix[n, k] = med
            else:
                Coef_matrix[n, k] = np.exp(-1j * 2 * np.pi * Delay[n] * fk)
        gain = gamma * Coef_matrix[:, k] * alpha#[:, k]
        H_all[:, :, k] = np.matmul(np.matmul(Ar, np.diag(gain)), At.conj().T)
        # power_H = np.linalg.norm(H_all[:, :, k],'fro') ** 2 / (Nr * Nt)
        # print(f'channel power is {power_H}')

    if Noisy_Channel:
        noise = np.sqrt(1 / 2) * (np.random.normal(0, 1, size=[Nr, Nt, K]) + 1j * np.random.normal(0, 1, size=[Nr, Nt, K]))
        H_all = np.sqrt(1-epision) * H_all + np.sqrt(epision*Nr*Nt) * noise
        ccc = 1
    return H_all, At_all


def masking_dyn(H, sub_connected=False, sub_structure_type="fixed"):
    batch_size, Nr, Nt, K = H.shape
    N = 2 * Nt * Nrf
    bin_mask_mat = np.ones([batch_size, Nt, Nrf], dtype='int_') + 1j * np.ones([batch_size, Nt, Nrf], dtype='int_')
    bin_mask_vec_real = np.zeros([batch_size, N])

    for ii in range(batch_size):
        if sub_connected:
            if sub_structure_type == "fixed":
                bin_mask_mat[ii, Nt // 2:Nt, 0] = 0
                bin_mask_mat[ii, 0:Nt // 2, 1] = 0
            else:  # dynamic
                # choose best channel
                power_H = np.zeros([K], dtype='float')
                for k in range(K):
                    power_H[k] = np.linalg.norm(H[ii,:, :, k])

                k_max = np.argmax(power_H)
                Hmax = H[ii, :, :, k]
                # print(Hmax)
                D = np.abs(Hmax.T)
                # print(np.shape(D))
                bin_mask_mat_k = np.ones([Nt, Nrf], dtype='int_') + 1j * np.ones([Nt, Nrf], dtype='int_')
                for m in range(Nt // Nrf):
                    for n in range(Nrf):
                        m_min = np.argmin(D[:, n], axis=0)
                        bin_mask_mat_k[m_min, n] = 0
                        D[m_min, :] = 1000
                # print(bin_mask_mat_k)

                bin_mask_mat[ii, :, :] = bin_mask_mat_k

            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
        # print(bin_mask_mat[ii, :, :])

        else:
            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
    return bin_mask_vec_real


def normalize(FRF,sub_connected=False, sub_structure_type="fixed"):
    Nt, Nrf = FRF.shape
    if sub_connected:
        if sub_structure_type == "fixed":
            FRF[0:Nt // 2, 0] = FRF[0:Nt // 2, 0] / np.abs(FRF[0:Nt // 2, 0])
            FRF[Nt // 2:, 1] = FRF[Nt // 2:, 1] / np.abs(FRF[Nt // 2:, 1])

        else:
            for tt in range(Nt):
                for nn in range(Nrf):
                    if np.abs(FRF[tt, nn]) > 0.0001:
                        FRF[tt, nn] = FRF[tt, nn] / np.abs(FRF[tt, nn])
    else:
        FRF = FRF / np.abs(FRF)
    ccc=1
    return FRF

def water_filling(singular_values, N0, power_max):

    # Bisection search for mu
    mu_low = 0  # Initial low
    mu_high = (power_max + np.sum(N0 / singular_values))  # Initial high

    stop_threshold = 1e-7  # Stop threshold

    # Iterate while low/high bounds are further than stop_threshold
    while np.abs(mu_low - mu_high) > stop_threshold:
        mu = (mu_low + mu_high) / 2  # Test value in the middle of low/high

        # Solve the power allocation
        power = 1 / mu - N0 / singular_values
        power[power < 0] = 0  # Consider only positive power allocation

        # Test sum-power constraints
        if (np.sum(power) > power_max):  # Exceeds power limit => lower the upper bound
            mu_low = mu
        else:  # Less than power limit => increase the lower bound
            mu_high = mu

    return power


def gen_data_wideband(Nt, Nr, Nrf, Ns, batch_size=1,
                      Sub_Connected=False,
                      Sub_Structure_Type='fixed',
                      Pulse_Filter=False,
                      fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning

    batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab

    for ii in range(batch_size):

        FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
        FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
        FRF_vec = FRF.flatten('F')
        batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)


        H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
        batch_H[ii, :, :, :] = H_ii
        batch_At[ii, :, :, :] = At_ii

        for k in range(K):
            At = At_ii[:, :, k]
            U, S, VH = np.linalg.svd(H_ii[:, :, k])
            V = VH.T.conj()
            power = water_filling(S[0:Ns], 1, Ns)
            Fopt = np.matmul(V[:, 0:Ns], np.diag(np.sqrt(power)) )  # np.sqrt(Ns) *
            # Fopt = V[:, 0:Ns]   # np.sqrt(Ns) *
            Wopt = U[:, 0:Ns]

            ## construct training data
            ztilde = Fopt.flatten('F')
            z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
            # z_vector = np.matrix(z)

            FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
            FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')


            # FBB = np.matmul(np.linalg.pinv(FRF), Fopt)

            Btilde = np.kron(FBB.T, np.identity(Nt))
            B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
            B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
            B = np.concatenate((B1, B2), axis=0)
            # print(np.shape(B))

            # new for array response
            AtH = At.conj().T
            Atilde = np.kron(np.identity(Nrf), AtH)
            A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
            A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
            A = np.concatenate((A1, A2), axis=0)
            # print(np.shape(A))

            batch_Bz[ii, :, k] = np.matmul(B.T, z)
            batch_BB[ii, :, :, k] = np.matmul(B.T, B)
            batch_z[ii, :, k] = z
            batch_B[ii, :, :, k] = B.T
            batch_Fopt[ii, :, :, k] = Fopt
            batch_Wopt[ii, :, :, k] = Wopt
            batch_Fbb[ii, :, :, k] = FBB
            batch_AA[ii, :, :, k] = np.matmul(A.T, A)

    return batch_Bz, batch_BB, batch_X, batch_z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At

def gen_data_QuaDriGA(Nrf, Ns, channel_train='H_train.npy', generated_data_name='train_data_name'):

    # Channel setup
    channel_type = 'geometry'
    # data to get
    # channel_train = 'H_train.npy'
    train_channel_path = dataset_file + channel_train
    Hall = np.load(train_channel_path)
    sample_num, K, Nr, Nt = Hall.shape
    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([sample_num, N], dtype='float32')  # only use when employing supervised learning
    batch_z = np.zeros([sample_num, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([sample_num, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([sample_num, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([sample_num, N, N, K], dtype='float32')  # input to DNN for training
    # input to DNN for training

    batch_H = np.zeros([sample_num, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([sample_num, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([sample_num, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([sample_num, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
   # use to save testing data, used latter in Matlab


    for ii in range(sample_num):



        FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
        FRF_vec = FRF.flatten('F')
        batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)

        # generate channel matrix
        if channel_type == 'Rician':
            Hii = 1 / np.sqrt(2) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))
            batch_H[ii, :, :, :] = Hii
        else:
            H_ii = Hall[ii, :, :, :].transpose(1,2,0)
            batch_H[ii, :, :, :] = H_ii

            for k in range(K):

                U, S, VH = np.linalg.svd(H_ii[:, :, k])
                V = VH.T.conj()
                # power = water_filling(S[0:Ns], 1, Ns)
                # Fopt = np.matmul(V[:, 0:Ns], np.diag(np.sqrt(power)))  # np.sqrt(Ns) *
                Fopt = V[:, 0:Ns]  # np.sqrt(Ns) *
                Wopt = U[:, 0:Ns]

                ## construct training data
                ztilde = Fopt.flatten('F')
                z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
                # z_vector = np.matrix(z)


                FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')


                Btilde = np.kron(FBB.T, np.identity(Nt))
                B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                B = np.concatenate((B1, B2), axis=0)
                # print(np.shape(B))

                # new for array response

                # print(np.shape(A))

                # Assign data to the ii-th batch
                # err = z_vector.dot(B) -np.matmul(B.T, z)
                # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

                batch_Bz[ii, :, k] = np.matmul(B.T, z)
                batch_BB[ii, :, :, k] = np.matmul(B.T, B)
                batch_z[ii, :, k] = z
                batch_B[ii, :, :, k] = B.T
                batch_Fopt[ii, :, :, k] = Fopt
                batch_Wopt[ii, :, :, k] = Wopt
                batch_Fbb[ii, :, :, k] = FBB


            # Hgap = np.linalg.norm(H,ord='fro')/np.sqrt(Nt*Nr)
            # print(f'HQ is: {Hgap:.4f}')
            # Compute optimal digital precoder


    data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_z,
                'batch_B': batch_B,
                'batch_H_real': batch_H.real,
                'batch_H_imag': batch_H.imag,
                'batch_Fopt_real': batch_Fopt.real,
                'batch_Fopt_imag': batch_Fopt.imag,
                }

    train_data_path = dataset_file + generated_data_name
    file_handle = h5py.File(train_data_path, 'w')
    # for name in data_all.keys():
    #     file_handle.attrs[name]=data_all[name]
    # file_handle.close()
    for name in data_all:
        dshp = data_all[name].shape
        dims = list(dshp[1:])
        dims.insert(0, None)
        # print(f'dshp shape:{dshp}, dims shape:{dims}')
        file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                   compression_opts=9)

    print(f'Finished Data Generating ', flush=True)

    ccc = 1


def exponentially_decay_lr(lr_ini, lr_lb, decay_factor, learning_steps, decay_steps, staircase=1):
    '''
    The latex formular is given as
        $\alpha = \max(\alpha_0 \beta^{\left \lfloor \frac{t}{{\Delta t}^I}\right \rfloor},\alpha_e)$

    :param lr_ini(\alpha_0): initial learning rate
    :param lr_lb(\alpha_e): learning rate lower bound
    :param decay_factor(\beta): decay factor of learning rate
    :param learning_steps(t): number of learning steps
    :param decay_steps(\Delta t): the number of steps that the learning rate keeps the same
    :param staircase(I): whether the staircase decrease of learning rate is adopted. 1 indicates True by default. If it is
    False, then the decay_steps doesn't function anymore.
    :return: decayed learning rate (\alpha)
    '''
    import math
    if staircase:
        med_steps = decay_steps
    else:
        med_steps = 1
    lr_decayed = lr_ini*decay_factor**(math.floor(learning_steps/med_steps))
    lr = max(lr_decayed,lr_lb)
    return lr




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        The architecture of ScNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class VectorLinear(nn.Module):
    def __init__(self, N, keep_bias=True):
        super(VectorLinear, self).__init__()
        self.keep_bias = keep_bias
        # print(f'mask is {self.mask}')
        self.weight = nn.Parameter(torch.randn([1, N]))  # initialize weight
        # print(f'0 weight is {self.weight}')
        if self.keep_bias:
            self.bias = nn.Parameter(torch.randn([1, N]))  # initialize bias
        # print(f'0 bias is {self.bias}')
        self.reset_parameters()  # self-defined initialization

    def forward(self, input):
        if self.keep_bias:
            return input*self.weight + self.bias
        else:
            return input * self.weight

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)
            else:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)


class ScNet(nn.Module):
    def __init__(self, in_dim, num_subcarriers, num_layer, Loss_scalar=1, Keep_Bias=False, BN = True, Sub_Connected=False, training_method='unsupervised'):
        super(ScNet, self).__init__()
        self.in_dim = in_dim
        self.training_method = training_method
        self.dobn = BN
        self.Sub_Connected = Sub_Connected
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers1K = nn.ModuleList()
        self.layers2K = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bnsK = nn.ModuleList()
        self.num_layer = num_layer
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers
        # self.t = torch.tensor(data=[0.5])

        for i in range(num_layer):  # define all layers

            # self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers2.append(VectorLinear(N, keep_bias=Keep_Bias))
            for k in range(self.num_subcarriers):
                self.layers1K.append(VectorLinear(N, keep_bias=Keep_Bias))
                self.layers2K.append(VectorLinear(N, keep_bias=Keep_Bias))
                if self.dobn:
                    # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                    # setattr(self, 'bn_layers%i'%i, bn_layer)
                    self.bnsK.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))

            if self.dobn:
                # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                # setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bns.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))


    def forward(self, x_ini,batch_BB_sum, batch_Bz_sum, x, z, B, Mask):
        # batch_size = zB.size()[0]
        LOSS = []
        Out_list = []
        x_est = x_ini
        # x_est = torch.zeros_like(x, requires_grad=True)

        for l in range(self.num_layer):
            # batch_Bz_sum = 0
            # batch_BB_sum = 0
            # for k in range(self.num_subcarriers):
            #     index = l * self.num_subcarriers + k
            #     batch_Bz_sum = batch_Bz_sum + zB[:, :, k]
            #     batch_BB_sum = batch_BB_sum + BB[:, :, :, k]
            #     if self.IL:
            #         aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
            #         out = self.layers1K[index](aux_term) + self.layers2K[index](x_est)
            #         # out = self.layers1K[index](aux_term + x_est)
            #         if self.dobn:
            #             x_est = self.bnsK[index](out)
            #         x_est = x_est * Mask
            #         x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
            #             x_est - 0.5) / 0.5

            aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
            if self.Sub_Connected:
                out = self.layers1[l](aux_term * Mask) + self.layers2[l](x_est)
            else:
                out = self.layers1[l](aux_term) + self.layers2[l](x_est)
            # out = self.layers1[l](aux_term + x_est)
            if self.dobn:
                x_est = self.bns[l](out)
            x_est = x_est * Mask
            # if l<self.num_layer-1:
            #     x_est = torch.nn.functional.relu(x_est)
            #     # x_est = torch.nn.functional.leaky_relu(x_est)
            # else:
            #     # x_est = -1 + tf.nn.relu(x_tmp + t) / tf.abs(t) - tf.nn.relu(x_tmp - t) / tf.abs(t)
            #     # x_est = -1 + torch.nn.functional.relu(x_est + t) / torch.abs(t) - torch.nn.functional.relu(x_est - t) / torch.abs(t)
            #     x_est = torch.tanh(x_est)
            x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                x_est - 0.5) / 0.5
            # Out_list.append(x_est.detach().numpy())

            if self.training_method == 'supervised':

                dis = torch.mean(torch.square(x - x_est))

            else:
                dis_sum = 0
                for k in range(self.num_subcarriers):
                    diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                    dis_sum += torch.mean(torch.square(diff))

            LOSS.append(self.scalar*np.log(l+1) * dis_sum)

        return x_est, LOSS


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gen_data_QuaDriGA(Nrf, Ns,  channel_train='H_train.npy', generated_data_name=train_data_name)
    gen_data_QuaDriGA(Nrf, Ns, channel_train='H_test.npy', generated_data_name=test_data_name)
    #
    train_data_path = dataset_file + train_data_name
    hf = h5py.File(train_data_path, 'r')
    print('----------------------training data-------------------------')
    for key in hf.keys():
        print(key, hf[key])

    test_data_path = dataset_file + test_data_name
    hf = h5py.File(test_data_path, 'r')
    print('----------------------testing data-------------------------')
    for key in hf.keys():
        print(key, hf[key])

    ccc=1
    pass
    def draw_lrfunc():
        lr_ini = 0.001
        lr_lb = 1e-4
        decay_factor = 0.8
        decay_steps = 10
        staircase = 1
        num_learning_steps = 400
        Lr_all = []
        for step in range(num_learning_steps):
            lr = exponentially_decay_lr(lr_ini, lr_lb, decay_factor=decay_factor, learning_steps=step,
                                        decay_steps=decay_steps, staircase=staircase)
            Lr_all.append(lr)

        plt.figure(dpi=100)
        plt.plot(Lr_all, label=r'$\psi(x)$')

        plt.legend(loc='center right')
        # plt.xticks(x)
        plt.xlabel('steps')
        plt.ylabel('lr value')
        plt.grid(True)
        plt.show()
        # print(batch_Z)
