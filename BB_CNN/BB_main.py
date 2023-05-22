# -*- oding:utf-8 -*-
'''
# @File: BB_main.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2023-01-30 1:32 PM
'''
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import torch

from Head import *
import BB_FuncLr as BF
from torch.utils.data import Dataset, DataLoader

time_now_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings starts at the time:', time_now_start, flush=True)  # 当前时间
t_start = time.time()

# 加载data
train_set = BF.My_dataset(file_dir=dataset_file, file_name=train_data_name, conv_type=Conv_Type, datatype=THz_channl)
test_set = BF.My_dataset(file_dir=dataset_file, file_name=test_data_name, conv_type=Conv_Type, datatype=THz_channl)

n_smp_tr = train_set.len
n_smp_te = test_set.len


# 设置batchsize
myloader_tr = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
myloader_te = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False)

# myModel = BF.CNN_v1(Nrf, Nt)
if OutputFbb:
    myModel = BF.CNN_v3(Nrf, Nt, Nr, K)
else:
    myModel = BF.CNN_v2(Nrf, Nt)
optimizer = torch.optim.Adam(myModel.parameters(), lr=start_learning_rate, weight_decay=Weight_decay)


def train():
    torch.manual_seed(Seed_train)
    np.random.seed(Seed_train)
    myModel.train()
    batch_count = 0
    Loss_cache=[]
    Lr_list = []
    for epoch in range(Ntrain_Epoch):
        for batch_idx, (batch_H, batch_data) in enumerate(myloader_tr):
            batch_count = batch_count + 1

            if OutputFbb:
                fvec_est, Fbb_est = myModel(batch_data)
                loss, _, _ = BF.loss_caclu_bb(fvec_est, Fbb_est, batch_H)
            else:
                fvec_est = myModel(batch_data)
                loss = BF.loss_caclu(fvec_est, batch_H)


            Loss_cache.append(loss.item())


            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            Lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

            if batch_count % Log_interval == 0:
                len_loss = len(Loss_cache)
                if len_loss > 2 * Log_interval:
                    avr_loss = np.mean(Loss_cache[len_loss-Log_interval:])  # 取倒数Log_interval个loss做平均
                    print(f'Epoch:{epoch}, batch_id:{batch_count}, learning rate: {Lr_list[-1]:.5f}, average loss:{avr_loss:.6f}',flush=True)
            ccc =1
    checkpoint = {
        'Epoch': epoch,
        'Batch_count': batch_count,
        'loss': Loss_cache,
        'model_state_dict': myModel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}
    modal_path = model_file_name +'.pt'
    torch.save(checkpoint, modal_path)  # save model
    return Loss_cache, Lr_list

def test():
    torch.manual_seed(Seed_test)
    np.random.seed(Seed_test)
    myModel.eval()  # testing mode
    batch_count = 0
    Loss_cache=[]
    Lr_list = []

    for batch_idx, (batch_H, batch_data) in enumerate(myloader_te):
        batch_count = batch_count + 1


        if OutputFbb:
            Frf_est, Fbb_est = myModel(batch_data)
            loss, Frf_batch, Fbb_batch = BF.loss_caclu_bb(Frf_est, Fbb_est, batch_H)
            print(f'batch_idx:{batch_idx}, loss:{loss.item():.6f}')
            return Frf_batch, Fbb_batch, batch_H
        else:
            Frf_est = myModel(batch_data)
            loss = BF.loss_caclu(Frf_est, batch_H)
            print(f'batch_idx:{batch_idx}, loss:{loss.item():.6f}')
            return Frf_est, batch_H




loss_all,lr_all=train()
time_now_end = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings ends at the time:', time_now_end, flush=True)  # 当前时间
t_end = time.time()
time_cost = (t_end - t_start) / 3600


# --------------------draw figure----------------------------
fig, axs = plt.subplots(ncols=2, nrows=1)

ax = axs.flatten()

ax[0].plot(np.arange(len(loss_all)), loss_all)

ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('loss value')
ax[0].grid(True)

ax[1].plot(np.arange(len(lr_all)), lr_all)
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('learning rate')
ax[1].grid(True)

fig.tight_layout()
fig_name = 'loss_lr-Epoch.png'
fig_path = directory_model + '/' + fig_name
plt.savefig(fig_path)  # save figure
# plt.show()
# plt.plot(Loss_cache, label='loss')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.grid()
# plt.show()
var_dict = {'loss': loss_all, 'Lr': lr_all}
fullpath = directory_model + '/' + 'training_record.mat'
spio.savemat(fullpath, var_dict)

print('-----------------------------Start Test---------------------------------', flush=True)
if OutputFbb:
    Frf_batch, Fbb_batch, batch_H = test()
    spio.savemat(dat_file_name,
                 {"H": batch_H.detach().numpy(), 'Frf_est': Frf_batch.detach().numpy(), 'Fbb_batch': Fbb_batch.detach().numpy()})

else:
    Frf_batch, batch_H = test()
    spio.savemat(dat_file_name,
                 {"H": batch_H.detach().numpy(), 'Frf_est': Frf_batch.detach().numpy()})

print('-----------------------------Test Finished---------------------------------')

