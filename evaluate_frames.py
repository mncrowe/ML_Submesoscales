import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import os

# other scripts in this directory
import tools.load_nc_data as ld
import pytorch_scripts.pytorch_models as pm
from tools.load_tensor import *
from pytorch_scripts.activation_functions import *
from pytorch_scripts.loss_functions import *
import mc_figs.save_frames as sf
import mc_ML.models as md

# define parameters...
model_name = 'bp_to_w'
batch_size = 64

# load test data...
print('loading all data...')
X_test, Y_test =  load_tensor('data/Pytorch_data.nc',['b_test','p_test'],'w_avg_test',out=2)
X_train, Y_train =  load_tensor('data/Pytorch_data.nc',['b_train','p_train'],'w_avg_train',out=2)
X = torch.cat((X_train,X_test),dim=0)
Y = torch.cat((Y_train,Y_test),dim=0)

N,Nl,Nx,Ny = X.shape

print(N)

Nb = -(N // -batch_size)   # number of batches

print('loading model...')
autoencoder = md.UNet(N_in=Nl,p_skip=0.5)
autoencoder.load_state_dict(torch.load('models/'+model_name+'.pt'))
autoencoder.eval()        # set model to evaluate mode
autoencoder.cuda()

print('predicting output...')
Y_pred = torch.tensor(np.zeros((N,1,Nx,Ny)), dtype=torch.float32)
for i in range(Nb):
  print(' - predicting batch ' + str(i+1) + ' of ' + str(Nb) + '...')
  index = range(batch_size*i,min(batch_size*(i+1),N))
  with torch.no_grad():
    Y_pred[index] = autoencoder(X[index].cuda()).cpu()

print('merging split frames ...')
X = ld.unsplit(X.numpy(),2,2)
Y = ld.unsplit(Y.numpy(),2,2)
Yp = ld.unsplit(Y_pred.numpy(),2,2)

#B = 1e-8*np.reshape(np.linspace(-5e3,5e3,512),(512,1,1)) # background gradient

Y_max = np.maximum(np.max(Y),np.max(Yp))
Y_min = np.minimum(np.min(Y),np.min(Yp))
lims = (Y_min,Y_max)

print('Saving frames ...')
if Nl == 2:
  sf.save_frames(np.transpose(X[:,0,:,:],(1,2,0)),'frames/frame_b',disp=1)
  sf.save_frames(np.transpose(X[:,1,:,:],(1,2,0)),'frames/frame_p',disp=1)
if Nl == 1:
  sf.save_frames(np.transpose(X[:,0,:,:],(1,2,0)),'frames/frame_b',disp=1)
sf.save_frames(np.transpose(Y[:,0,:,:],(1,2,0)),'frames/frame_w',lims=lims,disp=1)
sf.save_frames(np.transpose(Yp[:,0,:,:],(1,2,0)),'frames/frame_wp',lims=lims,disp=1)

print('Merging frames ...')
os.system('bash conc_frames.sh')

print('Creating movie ...')
os.system('ffmpeg -framerate 30 -i conc_all/frame_%04d.png -pix_fmt yuv420p Mov.mp4')