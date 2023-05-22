# -* coding: utf-8 -*-
'''
@File:main.py
@Author: Mengyuan Ma, Nhan Nguyen
@Contact: {mengyuan.ma, nhan.nguyen}@oulu.fi
@Time: 2022-08-15 9:18

These codes are correspoinding to the paper <Deep Unfolding Hybrid Beamforming Designs for THz Massive MIMO systems> with
the link https://arxiv.org/pdf/2302.12041.pdf
You must cite our paper if you use (parts of) our codes.
'''

import logging
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io as spio
import torch

from ManNet_Lbr import *
from Params import *

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable INFO\WARNING prompt
logging.disable(logging.WARNING)  # forbidden all log info

time_now_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings starts at the time:', time_now_start)  # 当前时间

t_start = time.time()


# train the model
def train_model():
    if Continue_Train:
        modal_path = model_file_name + '.pt'
        checkpoint = torch.load(modal_path, map_location=MainDevice)
        myModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch_idx = checkpoint['batch_idx']
        Loss_cache = checkpoint['loss']
    else:
        Loss_cache = []
        batch_idx = 0

    torch.manual_seed(Seed_train)
    np.random.seed(Seed_train)
    myModel.train()  # training mode

    Lr_list = []
    Loss_cache = []
    Loss_end = []

    for batch_idx in range(Ntrain_Batch):
        # Gen training data
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(Nt, Nr, Nrf, Ns,
                                                                                                                                          batch_size=train_batch_size,
                                                                                                                                          Sub_Connected=Sub_Connected,
                      Sub_Structure_Type=Sub_Structure_Type)
        batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
        batch_Bz_sum = 0
        batch_BB_sum = 0
        for k in range(K):
            batch_Bz_sum += batch_Bz[:, :, k]
            batch_BB_sum += batch_BB[:, :, :, k]

        batch_X = torch.from_numpy(batch_X).float()
        batch_Z = torch.from_numpy(batch_Z).float()
        batch_B = torch.from_numpy(batch_B).float()
        batch_Mask = torch.from_numpy(batch_Mask).float()
        batch_Bz_sum = torch.from_numpy(batch_Bz_sum).float()
        batch_BB_sum = torch.from_numpy(batch_BB_sum).float()

        x_ini = torch.zeros_like(batch_X, requires_grad=True)
        s_hat, loss_list = myModel(x_ini.to(MainDevice), batch_BB_sum.to(MainDevice), batch_Bz_sum.to(MainDevice),
                                   batch_X.to(MainDevice), batch_Z.to(MainDevice),
                                   batch_B.to(MainDevice), batch_Mask.to(MainDevice))
        if SUM_LOSS==1:
            loss = sum(loss_list)
        else:
            loss = loss_list[-1]
        Loss_end.append(loss_list[-1].item())

        Loss_cache.append(loss.item())

        if set_Lr_decay:
            for g in optimizer.param_groups:
                g['lr'] = exponentially_decay_lr(lr_ini=start_learning_rate, lr_lb=Lr_min, decay_factor=Lr_decay_factor,
                                                 learning_steps=batch_idx, decay_steps=Lr_keep_steps, staircase=1)
        optimizer.zero_grad()  # zero gradient
        loss.backward()  # backpropagation
        optimizer.step()  # update training prapameters
        Lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # for name, params in myModule.named_parameters():
        #     if 'layer_axz_0.weight' in name:
        #         print(f'epoch {epoch} after update: name {name}, params {params}')
        if (batch_idx) % Log_interval == 0:
            len_loss = len(Loss_cache)
            if len_loss > 2 * Log_interval:
                avr_loss = np.mean(Loss_cache[len_loss-Log_interval:])  # 取倒数Log_interval个loss做平均
                print(f'batch_id:{batch_idx}, learning rate: {Lr_list[-1]:.5f}, average loss:{avr_loss:.6f}', flush=True)
        if  Iterative_Training:  # start iterative training

            # batch_Bz = batch_Bz.numpy()
            # batch_BB = batch_BB.numpy()
            batch_X = batch_X.numpy()
            batch_Z = batch_Z.numpy()
            batch_B = batch_B.numpy()

            for jj in range(Iterations_train):
                s_hat = s_hat.detach().cpu().numpy()
                # s_dim = s_hat.shape[0]
                # 1. Update input to the network: only update data related to Frf, not change the channels
                for ii in range(s_hat.shape[0]):
                    ff = s_hat[ii, :]  # prepare testing data
                    FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
                    ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                    FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
                    FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)

                    for k in range(K):
                        Fopt_ii = batch_Fopt[ii, :, :, k]  # recall optimal fully digital precoder
                        Hii = batch_H[ii, :, :, k]
                        Uo, So, VoH = np.linalg.svd(Hii)

                        # solution to Fbb

                        FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder

                        FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')

                        Btilde = np.kron(FBB.T, np.identity(Nt))

                        # convert to real values
                        z_ii = batch_Z[ii, :, k]
                        B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                        B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                        B_ii = np.concatenate((B1, B2), axis=0)

                        # B1 = B_ii.T
                        batch_Bz[ii, :, k] = np.matmul(B_ii.T, z_ii)  # update values
                        batch_BB[ii, :, :, k] = np.matmul(B_ii.T, B_ii)
                        batch_B[ii, :, :, k] = B_ii.T

                # Update training data

                batch_Bz = torch.from_numpy(batch_Bz)
                batch_BB = torch.from_numpy(batch_BB)
                batch_X = torch.from_numpy(batch_X)
                batch_Z = torch.from_numpy(batch_Z)
                batch_B = torch.from_numpy(batch_B)
                if Rdm_Ini:
                    s_ini = torch.zeros_like(batch_X, requires_grad=True)
                else:
                    s_ini = torch.from_numpy(s_hat)

                batch_Bz_sum = 0
                batch_BB_sum = 0
                for k in range(K):
                    batch_Bz_sum += batch_Bz[:, :, k]
                    batch_BB_sum += batch_BB[:, :, :, k]

                s_hat, loss_list = myModel(s_ini.to(MainDevice), batch_BB_sum.to(MainDevice),
                                           batch_Bz_sum.to(MainDevice), batch_X.to(MainDevice),
                                           batch_Z.to(MainDevice),
                                           batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                if SUM_LOSS == 1:
                    loss = sum(loss_list)
                else:
                    loss = loss_list[-1]

                torch.cuda.empty_cache()
                optimizer.zero_grad()  # zero gradient
                loss.backward()  # backpropagation
                optimizer.step()  # update training prapameters
                torch.cuda.empty_cache()

                batch_X = batch_X.numpy()
                batch_Z = batch_Z.numpy()
                batch_B = batch_B.numpy()
                batch_Bz = batch_Bz.numpy()
                batch_BB = batch_BB.numpy()

        if batch_idx >= 250:
            avr_loss_old = np.mean(Loss_cache[batch_idx - 200:batch_idx - 100])
            avr_loss_new = np.mean(Loss_cache[batch_idx - 100:])
            if abs(avr_loss_new - avr_loss_old) <= 2:
                if batch_idx % 100 == 0:
                    checkpoint = {
                        'batch_idx': batch_idx,
                        'loss': Loss_cache,
                        'model_state_dict': myModel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
                    modal_path = model_file_name + '_{:d}.pt'.format(batch_idx)
                    torch.save(checkpoint, modal_path)  # save model



        checkpoint = {
            'batch_idx': batch_idx,
            'loss': Loss_cache,
            'model_state_dict': myModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        modal_path = model_file_name + '.pt'
        torch.save(checkpoint, modal_path)  # save model

    return Loss_cache, Loss_end, Lr_list


def test_model(BB_beamformer = 'LS'):
    torch.manual_seed(Seed_test)
    np.random.seed(Seed_test)

    myModel.eval()  # training mode
    myModel.to('cpu')  # test on CPU
    f_all = []
    batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(
        Nt, Nr, Nrf, Ns, test_batch_size)

    batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
    batch_Mask = torch.from_numpy(batch_Mask).float()
    batch_Bz = torch.from_numpy(batch_Bz).float()
    batch_BB = torch.from_numpy(batch_BB).float()
    batch_X = torch.from_numpy(batch_X).float()
    batch_Z = torch.from_numpy(batch_Z).float()
    batch_B = torch.from_numpy(batch_B).float()

    batch_Bz_sum = 0
    batch_BB_sum = 0
    for k in range(K):
        batch_Bz_sum += batch_Bz[:, :, k]
        batch_BB_sum += batch_BB[:, :, :, k]

    x_ini = torch.zeros_like(batch_X, requires_grad=True)
    s_hat, loss = myModel(x_ini, batch_BB_sum, batch_Bz_sum, batch_X, batch_Z, batch_B, batch_Mask)

    s_hat = s_hat.detach().numpy()
    batch_Bz = batch_Bz.numpy()
    batch_BB = batch_BB.numpy()
    batch_X = batch_X.numpy()
    batch_Z = batch_Z.numpy()
    batch_B = batch_B.numpy()

    f_all.append(s_hat)


    for jj in range(Iterations_test):
        # 1. Update input to the network: only update data related to Frf, not change the channels
        for ii in range(test_batch_size):
            ff = s_hat[ii, :]# prepare testing data
            FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
            ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
            FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
            FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
            FRF_vec = FRF.flatten('F')
            batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)
            for k in range(K):
                Fopt_ii = batch_Fopt[ii, :, :, k]  # recall optimal fully digital precoder
                Hii = batch_H[ii, :, :, k]
                # Uo, So, VoH = np.linalg.svd(Hii)
                # Wopt = Uo[:, 0:Ns]
                # solution to Fbb

                FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder


                FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')
                Btilde = np.kron(FBB.T, np.identity(Nt))

                # convert to real values
                z_ii = batch_Z[ii, :, k]
                B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                B_ii = np.concatenate((B1, B2), axis=0)

                # B1 = B_ii.T
                batch_Bz[ii, :, k] = np.matmul(B_ii.T, z_ii)  # update values
                batch_BB[ii, :, :, k] = np.matmul(B_ii.T, B_ii)
                batch_B[ii, :, :, k] = B_ii.T
                batch_Fbb[ii, :, :, k] = FBB

        # Update training data


        batch_Bz = torch.from_numpy(batch_Bz).float()
        batch_BB = torch.from_numpy(batch_BB).float()
        batch_X = torch.from_numpy(batch_X).float()
        batch_Z = torch.from_numpy(batch_Z).float()
        batch_B = torch.from_numpy(batch_B).float()
        if Rdm_Ini:
            s_ini = torch.zeros_like(batch_X, requires_grad=True)
        else:
            s_ini = torch.from_numpy(s_hat)

        batch_Bz_sum = 0
        batch_BB_sum = 0
        for k in range(K):
            batch_Bz_sum += batch_Bz[:, :, k]
            batch_BB_sum += batch_BB[:, :, :, k]

        s_hat, loss_list = myModel(s_ini, batch_BB_sum, batch_Bz_sum, batch_X, batch_Z, batch_B, batch_Mask)

        if SUM_LOSS==1:
            loss = sum(loss_list)
        else:
            loss = loss_list[-1]
        s_hat = s_hat.detach().numpy()
        f_all.append(s_hat)
        batch_X = batch_X.numpy()
        batch_Z = batch_Z.numpy()
        batch_B = batch_B.numpy()
        batch_Bz = batch_Bz.numpy()
        batch_BB = batch_BB.numpy()

        print(f'Iteration:{jj}, loss:{loss:.4f}',flush=True)

    return s_hat, f_all, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_At

# define the network

myModel = ScNet(N, K, Num_layers, Loss_coef, Keep_Bias=Keep_Bias, BN=True, Sub_Connected=Sub_Connected)

myModel.to(MainDevice)
# for name, params in myModel.named_parameters():
#     print(f'e name {name}, params device {params.device}')
optimizer = torch.optim.Adam(myModel.parameters(), lr=start_learning_rate, weight_decay=Weight_decay)
Loss_cache, Loss_end, Lr_list = train_model()

time_now_end = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings ends at the time:', time_now_end, flush=True)  # 当前时间
t_end = time.time()
time_cost = (t_end - t_start)/3600
print(f'---------End training------time cost: {time_cost:.4f}h', flush=True)

# --------------------draw figure----------------------------
fig, axs = plt.subplots(ncols=2, nrows=1)

ax = axs.flatten()

ax[0].plot(np.arange(len(Loss_cache)), Loss_cache)

ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('loss value')
ax[0].grid(True)

ax[1].plot(np.arange(len(Lr_list)), Lr_list)
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('learning rate')
ax[1].grid(True)

fig.tight_layout()
fig_name = 'loss_lr-Epoch.png'
fig_path = directory_model + '/' + fig_name
plt.savefig(fig_path)  # save figure
plt.show()
# plt.plot(Loss_cache, label='loss')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.grid()
# plt.show()
var_dict = {'loss': Loss_cache, 'loss_end': Loss_end,  'Lr': Lr_list}
fullpath = directory_model + '/' + 'training_record.mat'
spio.savemat(fullpath, var_dict)



print('-----------------------------Start Test---------------------------------', flush=True)
s_hat, f_all, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, At = test_model()
spio.savemat(dat_file_name,
             {"H": batch_H, "Fopt": batch_Fopt, "Wopt": batch_Wopt, "Fbb": batch_Fbb, "f": s_hat,  'At': At,'f_all': f_all})

print('-----------------------------Test Finished---------------------------------', flush=True)

