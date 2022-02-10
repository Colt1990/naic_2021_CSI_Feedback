#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder,NMSE_cuda, NMSELoss,NMSE
import os
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import random
# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
use_single_gpu =False  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
random.seed(42)
batch_size =16
epochs = 100
learning_rate = 1e-3
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 512

# Model construction
model = AutoEncoder(feedback_bits)
#model=model.half()

# model.encoder.load_state_dict(torch.load('Modelsave/encoder.pth.tar')['state_dict'])
# model.decoder.load_state_dict(torch.load('Modelsave/decoder.pth.tar')['state_dict'])

if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    
    model = torch.nn.DataParallel(model).cuda()
import scipy.io as scio
criterion = nn.MSELoss().cuda()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
data_load_address = '../train'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
print(np.shape(x_train))
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))

# x_train = np.concatenate((x_train,x_test[:1500]),0) #
# x_test = x_test[1500:]
# print(np.shape(x_train))
# Data loading


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def sort_input(input):
    input = input.permute(0,3,2,1)
    B = len(input)
    #print(input.shape)
    x_var = torch.mean((input.reshape(B, 128, -1).detach() - 0.5) ** 2,
                       dim=-1)
    x_sort = torch.sort(-x_var, dim=-1)[1] + torch.arange(B).unsqueeze(-1).to(
        x_var.device) * 128
    x_sort = x_sort.view(-1)
    # x_sort = torch.sort(x_sort, dim=-1)[1]

    input = input.reshape(B * 128, 126, 2)
    input = torch.index_select(input, 0, x_sort).view(B, 128, 126, 2).permute(0,3,2,1)
    return input



# dataLoader for training
train_dataset = DatasetFolder(x_train,'train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
best_loss = 1.01


for epoch in range(epochs):
    # model training
    model.train()
    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.5
    for i, input in enumerate(train_loader):
        # adjust learning rate
    
        
        if  np.random.random() < 0.5:
            input = 1 - input  # 数据增强
        
        input = input.cuda()

        output = model(input)
        loss = criterion(output[:,:,:60,:],input[:,:,:60,:])

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.6f}  '.format(
                epoch, i, len(train_loader), loss=loss.item()))
    # model evaluating
    model.eval()
    total_loss = 0
    y_test = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):

            input = input.cuda()
            output= model(input)
        
            
            total_loss += criterion(output, input).item() * input.size(0)
            
            if i == 0:
                y_test = output.cpu()
            else:
                y_test = np.concatenate((y_test, output.cpu()), axis=0)
        average_loss = total_loss / len(test_dataset)

        if  epoch % 1 == 0:
            print('The NMSE is ' + np.str(NMSE(np.transpose(x_test, (0, 2, 3, 1)), np.transpose(y_test, (0, 2, 3, 1)))))
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)

            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
            print("Model saved")
            print('average_loss')
            best_loss = average_loss
