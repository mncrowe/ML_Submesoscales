# evaluation script to determine average MSE as a function of number of epochs

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics

# other scripts in this directory
import tools.load_nc_data as ld
import pytorch_scripts.pytorch_models as pm
from tools.load_tensor import *
from pytorch_scripts.activation_functions import *
from pytorch_scripts.loss_functions import *

import mc_ML.models as md

# define parameters...
batch_size = 64
input_type = 1
num_epochs = 200
avgsize = 10

# load test data...
print('loading test data...')
if input_type == 1:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc',['b_test','p_test'],'w_avg_test',out=2)
  model_name = 'bp_to_w'
if input_type == 2:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc','b_test','w_avg_test',out=2)
  model_name = 'b_to_w'
if input_type == 3:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc','p_test','w_avg_test',out=2)
  model_name = 'p_to_w'
N,Nl,Nx,Ny = X_test.shape
Nb = -(N // -batch_size)   # number of batches

# define model, set same as train_model.py
autoencoder = md.UNet(N_in=Nl,p_skip=0.5)
MSE = np.zeros(num_epochs)
Y_pred = torch.tensor(np.zeros((N,1,Nx,Ny)), dtype=torch.float32)

# loop through all models
for i_epoch in range(num_epochs):
  
  print('loading model ' + str(i_epoch+1) + '/' + str(num_epochs) + '...')
  autoencoder.load_state_dict(torch.load('models/epochs/'+model_name+'_'+str(i_epoch+1)+'.pt'))
  autoencoder.eval(); autoencoder.cuda()

  for i in range(Nb):
    #print(' - predicting batch ' + str(i+1) + ' of ' + str(Nb) + '...')
    index = range(batch_size*i,min(batch_size*(i+1),N))
    with torch.no_grad():
      Y_pred[index] = autoencoder(X_test[index].cuda()).cpu()

  MSE[i_epoch] = np.sqrt(statistics.mean(np.squeeze(np.sum((Y_pred.numpy()-Y_test.numpy())**2,axis=(2,3))/(Nx*Ny)) \
    /np.squeeze(np.sum(Y_test.numpy()**2,axis=(2,3))/(Nx*Ny))))
  
  print(' - <MSE> = ' + str(MSE[i_epoch]))

MSE_movavg = np.convolve(MSE, np.ones(avgsize)/avgsize, mode='valid')

print('Plotting MSE per epoch ...')
plt.plot(range(1,num_epochs+1),MSE,'b',range(avgsize,num_epochs+1),MSE_movavg,'r')
plt.xlabel('Epoch')
plt.xlim(1,num_epochs)
plt.ylabel('Mean (MSE /$<Y^2>)^{1/2}$')
plt.ylim(0,0.03)#0.00016)
plt.grid()
plt.tight_layout()
plt.show()

