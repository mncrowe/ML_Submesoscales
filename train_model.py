# trains a model on the oceananigans data

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# import my custom scripts from here or Python_Modules directory
import tools.load_nc_data as ld
from tools.load_tensor import *

import mc_ML.models as md

# define parameters...
#input_type = 1    # 1 - b&p to w, 2 - b to w, 3 - p to w
N_epoch = 200      # 100 seems to be fine, may need longer with dropout
batch_size = 64   # 16-64 generally good
dropout = 0.5     # dropout probability
saveall = 1        # saves model at each epoch

for input_type in range(1,2):
  
  # get some data...
  print('Loading data ...')
  if input_type == 1:
    XY_train = load_tensor('data/Pytorch_data.nc',['b_train','p_train'],'w_avg_train')
    X_test, Y_test =  load_tensor('data/Pytorch_data.nc',['b_test','p_test'],'w_avg_test',out=2)
    model_name = 'bp_to_w'
  if input_type == 2:
    XY_train = load_tensor('data/Pytorch_data.nc','b_train','w_avg_train')
    X_test, Y_test =  load_tensor('data/Pytorch_data.nc','b_test','w_avg_test',out=2)
    model_name = 'b_to_w'
  if input_type == 3:
    XY_train = load_tensor('data/Pytorch_data.nc','p_train','w_avg_train')
    X_test, Y_test =  load_tensor('data/Pytorch_data.nc','p_test','w_avg_test',out=2)
    model_name = 'p_to_w'
  
  Nt,Nl,Nx,Ny = X_test.shape
  
  # define model, optimiser and loss function...
  autoencoder = md.UNet(N_in=Nl,p_skip=dropout)
  autoencoder.cuda()        # .cpu() or .cuda(), send the model to the GPU (*waves goodbye*)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(autoencoder.parameters())
  train_loader = DataLoader(XY_train, batch_size=batch_size, shuffle=True)
  
  # now train it... (validation done using epoch saves in evaluate_models.py)
  autoencoder.train()     # set model to train mode
  for epoch in range(N_epoch):
    with tqdm(train_loader,unit="batch") as tepoch:
      for data, target in tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{N_epoch}")
        data, target = data.cuda(), target.cuda()    # .cpu() or .cuda()
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        tepoch.set_postfix(loss=loss.item())
    if saveall == 1:
      torch.save(autoencoder.state_dict(),'models/epochs/'+model_name+'_'+str(epoch+1)+'.pt')

  # save model... if not already saved...
  if saveall == 0:
    torch.save(autoencoder.state_dict(),'models/'+model_name+'.pt')