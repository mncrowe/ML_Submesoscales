# functions to load nc data using xarray

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)

def load_data(file_name,field):
  # loads 3D spatial data fields (+ 1 time dimension) and reshapes array
  # output dimensions are (x,y,z,t)
  F = xr.open_dataset(file_name)[field]
  if np.size(np.shape(F)) == 2: s = (1,0)
  if np.size(np.shape(F)) == 3: s = (2,1,0)
  if np.size(np.shape(F)) == 4: s = (3,2,1,0)
  return np.transpose(F.values,s)

def pytorch_input(field,Sx,Sy):
  # splits data into Sx x Sy subfields and reshapes into pytorch imput shape
  # also normalises to between 0 and 1
  field = np.squeeze(field)
  F2,F1 = field.max(),field.min()
  field = (field-F1)/(F2-F1)
  Nx,Ny,Nt = np.shape(field)
  Mx,My = np.int32((Nx/Sx,Ny/Sy))
  G = np.empty((Mx,My,Sx,Sy,Nt))
  for i in range(Sx):
    for j in range(Sy):
      G[:,:,i,j,:] = field[i*Mx:(i+1)*Mx,j*My:(j+1)*My,:]
  return np.transpose(np.reshape(G,(1,Mx,My,Nt*Sx*Sy)),(3,0,1,2))

def plot_comp(F,G,H):
  interp = 'none'  # 'bilinear'
  plt.figure(figsize=(15,5))
  plt.subplot(131)
  plt.imshow(F,interpolation=interp,cmap = 'seismic')
  plt.subplot(132)
  plt.imshow(G,interpolation=interp,cmap = 'seismic')
  plt.subplot(133)
  plt.imshow(H,interpolation=interp,cmap = 'seismic')
  plt.show()
  
def plot_comp_4(F,G,H,I):
  interp = 'none'  # 'bilinear'
  plt.figure(figsize=(8,8))
  plt.subplot(221)
  plt.imshow(F,interpolation=interp,cmap = 'seismic')
  plt.subplot(222)
  plt.imshow(G,interpolation=interp,cmap = 'seismic')
  plt.subplot(223)
  plt.imshow(H,interpolation=interp,cmap = 'seismic')
  plt.subplot(224)
  plt.imshow(I,interpolation=interp,cmap = 'seismic')
  plt.show()
  
def fancy_split(F,Sx,Sy,r=5,index=None,seed=0):
  # takes data from 'load_data' and splits into Sx x Sy subframes, normalises and randomises training/test
  # r is ratio of total data to test data size, i.e. r = 5 corresponds to a 80%/20% training/test split
  # index is the range of time indices to keep, if None, index = (0,Nt-1)
  
  random.seed(seed)
  N = (Sx*Sy) // r  # number of test subframes
  I = range(Sx*Sy)
  J = random.sample(I,k = N)  # test indices
  K = list(set(I) - set(J))   # train indices
  
  F = np.squeeze(F)
  F2,F1 = F.max(),F.min()     # normalise
  F = (F-F1)/(F2-F1)
  Nx,Ny,Nt = np.shape(F)
  Mx,My = np.int32((Nx/Sx,Ny/Sy))
  G = np.empty((Mx,My,Sx*Sy,Nt))
  
  for i in range(Sx):
    for j in range(Sy):
      G[:,:,i*Sy+j,:] = F[i*Mx:(i+1)*Mx,j*My:(j+1)*My,:]  # split into subframes
  
  if index is None:
    index = (0,Nt-1)
  F_test, F_train = G[:,:,J,index[0]:index[1]], G[:,:,K,index[0]:index[1]]  # select subframes to be train/test
  
  F_train = np.transpose(np.reshape(F_train,(1,Mx,My,-1)),(3,0,1,2))
  F_test = np.transpose(np.reshape(F_test,(1,Mx,My,-1)),(3,0,1,2))
  
  return F_train, F_test
  
def unsplit(F,Sx,Sy):
  # converts a concatenated train + test dataset back to a field
  
  N,Nl,Nx,Ny = np.shape(F)
  Nt = N//(Sx*Sy)
  
  G = np.zeros((Nt,Nl,Sx*Nx,Sy*Ny))
  
  for i in range(Sx):
    for j in range(Sy):
      G[:,:,i*Nx:(i+1)*Nx,j*Ny:(j+1)*Ny] = F[(i*Sy+j)*Nt:(i*Sy+j+1)*Nt,:,:,:]
      
  return G    