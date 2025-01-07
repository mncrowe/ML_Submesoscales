# loads h5 data into tensor format

from . import load_nc_data as ld
import torch
import numpy as np
from torch.utils.data import TensorDataset
import xarray as xr

def load_tensor(filename,X_name,Y_name,out=1):
  # loads X and Y from NetCDF file, arguments are strings or lists of strings
  if isinstance(X_name, str): X_name = [X_name]
  if isinstance(Y_name, str): Y_name = [Y_name]
  X = np.transpose(np.array([load_data(filename,X_name[i]) for i in range(len(X_name))]),(2,1,0,3,4))[0]
  Y = np.transpose(np.array([load_data(filename,Y_name[j]) for j in range(len(Y_name))]),(2,1,0,3,4))[0]
  if out == 1:
    return TensorDataset(torch.tensor(X, dtype=torch.float32),torch.tensor(Y, dtype=torch.float32))
  if out == 2:
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def load_tensor_fast(filename,X_name,Y_name,out=1):
  # loads X and Y from NetCDF file, arguments are strings
  X = torch.tensor(load_data(filename,X_name), dtype=torch.float32)
  Y = torch.tensor(load_data(filename,Y_name), dtype=torch.float32)
  if out == 1: return TensorDataset(X,Y)
  if out == 2: return X, Y
  
def load_data(file_name,field):
  # loads 3D spatial data fields (+ 1 time dimension) and reshapes array
  # output dimensions are (x,y,z,t)
  F = xr.open_dataset(file_name)[field]
  return F.values
  