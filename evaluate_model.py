# evaluation script to determine if a model output is any good, loads models from /models

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

# load test data...
print('loading test data...')
if input_type == 1:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc',['b_test','p_test'],'w_avg_test',out=2)
  model_name = 'bp_to_w_100_BEST'
if input_type == 2:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc','b_test','w_avg_test',out=2)
  model_name = 'b_to_w_100'
if input_type == 3:
  X_test, Y_test =  load_tensor('data/Pytorch_data.nc','p_test','w_avg_test',out=2)
  model_name = 'p_to_w_100'
N,Nl,Nx,Ny = X_test.shape
Nb = -(N // -batch_size)   # number of batches

print('loading model...')
autoencoder = md.UNet(N_in=Nl,p_skip=0.5) # md.UNet(N_in=Nl) # 
autoencoder.load_state_dict(torch.load('models/'+model_name+'.pt'))
autoencoder.eval()        # set model to evaluate mode
autoencoder.cuda()

print('predicting output...')
Y_pred = torch.tensor(np.zeros((N,1,Nx,Ny)), dtype=torch.float32)
for i in range(Nb):
  print(' - predicting batch ' + str(i+1) + ' of ' + str(Nb) + '...')
  index = range(batch_size*i,min(batch_size*(i+1),N))
  with torch.no_grad():
    Y_pred[index] = autoencoder(X_test[index].cuda()).cpu()

MSE = np.squeeze(np.sum((Y_pred.numpy()-Y_test.numpy())**2,axis=(2,3))/(Nx*Ny))
MSE_norm = np.squeeze(np.sum(Y_test.numpy()**2,axis=(2,3))/(Nx*Ny))
print('<Y_norm> = ' + str(MSE_norm.mean()))
print('<MSE> = ' + str(statistics.mean(MSE)))

print('Plotting MSE per frame ...')
plt.plot(np.sqrt(MSE/MSE_norm))
plt.xlabel('Test datapoint')
plt.xlim(0,3000)
plt.ylabel('$(MSE /<Y^2>)^{1/2}$')
plt.ylim(0,0.03)#8e-4)
plt.grid()
plt.tight_layout()
plt.show()

np.random.seed(1)
i0 = 2999  #np.random.randint(N)

interp = 'none'
cmap = 'seismic'

#Nl = 0 # set to 0 to prevent plotting of full fields

if Nl == 2:
  # plot surface plot comparison
  X1 = np.squeeze(X_test[i0].numpy()[0,:,:])
  X2 = np.squeeze(X_test[i0].numpy()[1,:,:])
  Y1t = np.squeeze(Y_test[i0].numpy())
  Y1p = np.squeeze(Y_pred[i0].numpy())
  
  x1max,x1min = X1.max(),X1.min()
  x2max,x2min = X2.max(),X2.min()
  ymax,ymin = 0.5+np.array((1,-1))*max(max(Y1t.max(),Y1p.max())-0.5,0.5-min(Y1t.min(),Y1p.min()))
  
  #plt.figure(figsize=(6,6))
  #plt.subplot(221)
  #plt.imshow(X1,interpolation=interp,cmap=cmap)
  #plt.title('Buoyancy input')
  #plt.clim(x1min,x1max)
  #plt.axis('off')
  #plt.ylabel('y')
  #plt.subplot(222)
  #plt.imshow(X2,interpolation=interp,cmap=cmap)
  #plt.title('Pressure input')
  #plt.clim(x2min,x2max)
  #plt.axis('off')
  #plt.subplot(223)
  #plt.imshow(Y1t,interpolation=interp,cmap=cmap)
  #plt.title('Simulation output')
  #plt.clim(ymin,ymax)
  #plt.axis('off')
  #plt.xlabel('x')
  #plt.ylabel('y')
  #plt.subplot(224)
  #plt.imshow(Y1p,interpolation=interp,cmap=cmap)
  #plt.title('CNN predicted output')
  #plt.clim(ymin,ymax)
  #plt.axis('off')
  #plt.xlabel('x')
  
  #plt.tight_layout()
  #plt.show()
  
  print('Plotting fields for data point ' + str(i0))
  
  x = np.linspace(0,10,256)
  y = np.linspace(0,10,256)
  
  plt.figure(figsize=(4.5,4))
  plt.pcolor(x,y,X1,cmap=cmap,rasterized=True); plt.clim(x1min,x1max)
  plt.xlabel('x (km)'); plt.ylabel('y (km)')
  plt.title('Normalised Input (Buoyancy)')
  plt.colorbar(shrink=0.9)
  plt.text(0.3,9.2,'a)',color = "white",fontsize = 18)
  ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
  plt.show()
  
  plt.figure(figsize=(4.5,4))
  plt.pcolor(x,y,X2,cmap=cmap,rasterized=True); plt.clim(x2min,x2max)
  plt.xlabel('x (km)'); plt.ylabel('y (km)')
  plt.title('Normalised Input (Pressure)')
  plt.colorbar(shrink=0.9)
  plt.text(0.3,9.2,'b)',color = "white",fontsize = 18)
  ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
  plt.show()
  
  plt.figure(figsize=(4.5,4))
  plt.pcolor(x,y,Y1t,cmap=cmap,rasterized=True); plt.clim(ymin,ymax)
  plt.xlabel('x (km)'); plt.ylabel('y (km)')
  plt.title('Simulation Output (W)')
  plt.colorbar(shrink=0.9)
  plt.text(0.3,9.2,'c)',color = "black",fontsize = 18)
  ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
  plt.show()
  
  plt.figure(figsize=(4.5,4))
  plt.pcolor(x,y,Y1p,cmap=cmap,rasterized=True); plt.clim(ymin,ymax)
  plt.xlabel('x (km)'); plt.ylabel('y (km)')
  plt.title('CNN Predicted Output (W)')
  plt.colorbar(shrink=0.9)
  plt.text(0.3,9.2,'d)',color = "black",fontsize = 18)
  ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
  plt.show()
  
if Nl == 1:
  # plot surface plot comparison
  X1 = np.squeeze(X_test[i0].numpy())
  Y1t = np.squeeze(Y_test[i0].numpy())
  Y1p = np.squeeze(Y_pred[i0].numpy())
  
  x1max,x1min = X1.max(),X1.min()
  ymax,ymin = max(Y1t.max(),Y1p.max()),min(Y1t.min(),Y1p.min())
  
  plt.figure(figsize=(12,5))
  plt.subplot(131)
  plt.imshow(X1,interpolation=interp,cmap=cmap)
  plt.title('Buoyancy input')
  plt.clim(x1min,x1max)
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(Y1t,interpolation=interp,cmap=cmap)
  plt.title('Simulation output')
  plt.clim(ymin,ymax)
  plt.axis('off')
  plt.subplot(133)
  plt.imshow(Y1p,interpolation=interp,cmap=cmap)
  plt.title('CNN predicted output')
  plt.clim(ymin,ymax)
  plt.axis('off')
  
  plt.tight_layout()
  plt.show()
  

#print('Plotting comparison of random frames ...')
#np.random.seed(1)
#i0 = np.random.randint(N)
#if Nl == 1:
#  X1 = np.squeeze(X_test[i0].numpy())
#  Y1t = np.squeeze(Y_test[i0].numpy())
#  Y1p = np.squeeze(Y_pred[i0].numpy())
#  ld.plot_comp(X1,Y1t,Y1p)
#if Nl == 2:
#  X1 = np.squeeze(X_test[i0].numpy()[0,:,:])
#  X2 = np.squeeze(X_test[i0].numpy()[1,:,:])
#  Y1t = np.squeeze(Y_test[i0].numpy())
#  Y1p = np.squeeze(Y_pred[i0].numpy())
#  ld.plot_comp_4(X1,X2,Y1t,Y1p)
  